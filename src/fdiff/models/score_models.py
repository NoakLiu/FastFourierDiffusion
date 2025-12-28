from typing import Callable, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers.optimization import get_cosine_schedule_with_warmup
from einops import rearrange
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchvision.ops import MLP

from fdiff.models.transformer import (
    GaussianFourierProjection,
    PositionalEncoding,
    TimeEncoding,
)
from fdiff.models.cached_transformer import CachedTransformerEncoderLayer
from fdiff.schedulers.sde import SDE
from fdiff.utils.caching import E2CRFCache
from fdiff.utils.dataclasses import DiffusableBatch
from fdiff.utils.losses import get_sde_loss_fn


class ScoreModule(pl.LightningModule):
    def __init__(
        self,
        n_channels: int,
        max_len: int,
        noise_scheduler: SDE,
        fourier_noise_scaling: bool = True,
        d_model: int = 60,
        num_layers: int = 3,
        n_head: int = 12,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
        likelihood_weighting: bool = False,
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.max_len = max_len
        self.n_channels = n_channels

        self.noise_scheduler = noise_scheduler
        self.num_warmup_steps = num_training_steps // 10
        self.num_training_steps = num_training_steps
        self.lr_max = lr_max
        self.d_model = d_model
        self.scale_noise = fourier_noise_scaling

        # Loss function
        self.likelihood_weighting = likelihood_weighting
        self.training_loss_fn, self.validation_loss_fn = self.set_loss_fn()

        # Model components
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=self.max_len)
        self.time_encoder = self.set_time_encoder()
        self.embedder = nn.Linear(in_features=n_channels, out_features=d_model)
        self.unembedder = nn.Linear(in_features=d_model, out_features=n_channels)
        
        # Use standard transformer layer (can be replaced with cached version when caching is enabled)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers
        )
        
        # Cached layers (created when caching is enabled)
        self.cached_backbone: Optional[nn.ModuleList] = None

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Caching support
        self.cache: Optional[E2CRFCache] = None
        self.use_cache: bool = False
        self.cached_backbone: Optional[nn.ModuleList] = None

    def forward(
        self, 
        batch: DiffusableBatch,
        recompute_tokens: Optional[set[int]] = None,
        step: int = 0,
        return_crf: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Channel embedding
        X = self.embedder(X)

        # Add positional encoding
        X = self.pos_encoder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps)

        # Backbone with caching support
        if self.use_cache and recompute_tokens is not None and self.cached_backbone is not None:
            X, crf = self._forward_with_cache(X, recompute_tokens, step)
        else:
            # Standard forward pass
            X = self.backbone(X)
            crf = None

        # Channel unembedding
        X = self.unembedder(X)

        assert isinstance(X, torch.Tensor)
        
        if return_crf:
            return X, crf
        return X
    
    def _forward_with_cache(
        self,
        X: torch.Tensor,
        recompute_tokens: set[int],
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with caching support using cached transformer layers.
        
        Args:
            X: Input tensor (batch_size, max_len, d_model)
            recompute_tokens: Set of token indices to recompute
            step: Current diffusion step
            
        Returns:
            Output tensor and CRF (cumulative residual features)
        """
        # Store CRF at each layer
        crf_list = []
        h = X  # Initial embedding
        
        # OPTIMIZATION: If we need to recompute all tokens, use standard forward pass (faster)
        # But still use cached layers so they can populate the cache efficiently
        if len(recompute_tokens) == self.max_len:
            if self.cached_backbone is not None:
                # Use cached layers (they'll use standard attention internally when recomputing all)
                crf_list = []
                h_temp = h
                for layer_idx, layer in enumerate(self.cached_backbone):
                    h_temp = layer(h_temp, recompute_tokens=recompute_tokens)
                    # Store CRF (cumulative residual feature) - take first batch element
                    crf_list.append(h_temp[0].detach())
                h_new = h_temp
                crf = torch.stack(crf_list, dim=0)
                return h_new, crf
            else:
                # Fallback to standard backbone
                h_new = self.backbone(h)
                crf_list = []
                temp_h = h
                for layer in self.backbone.layers:
                    temp_h = layer(temp_h)
                    crf_list.append(temp_h[0].detach())
                crf = torch.stack(crf_list, dim=0)
                return h_new, crf
        
        # Use cached transformer layers for selective computation
        if self.cached_backbone is None:
            # Fallback to standard computation if cached backbone not initialized
            h_new = self.backbone(h)
            crf_list = []
            temp_h = h
            for layer in self.backbone.layers:
                temp_h = layer(temp_h)
                crf_list.append(temp_h[0].detach())
            crf = torch.stack(crf_list, dim=0)
            return h_new, crf
        
        # Process through cached transformer layers
        h_temp = h
        for layer_idx, layer in enumerate(self.cached_backbone):
            h_temp = layer(h_temp, recompute_tokens=recompute_tokens)
            
            # Store CRF (cumulative residual feature) - take first batch element
            # h_temp shape: (batch_size, max_len, d_model)
            # CRF shape per layer: (max_len, d_model)
            # Use detach() instead of clone() for better performance
            if h_temp.dim() == 3:
                crf_list.append(h_temp[0].detach())  # Take first batch element, detach to break gradient
            else:
                crf_list.append(h_temp.detach())
        
        h_new = h_temp
        crf = torch.stack(crf_list, dim=0)
        return h_new, crf
        
        # Stack CRF from all layers: (num_layers, max_len, d_model)
        # Each element in crf_list should be (max_len, d_model)
        crf = torch.stack(crf_list, dim=0)  # (num_layers, max_len, d_model)
        
        return h, crf
    
    def enable_caching(
        self,
        cache: Optional[E2CRFCache] = None,
        **cache_kwargs
    ) -> None:
        """Enable caching for inference.
        
        Args:
            cache: Optional E2CRFCache instance. If None, creates a new one.
            **cache_kwargs: Arguments to pass to E2CRFCache if creating new instance.
        """
        if cache is None:
            # Get number of heads from the first transformer layer
            first_layer = self.backbone.layers[0]
            n_head = getattr(first_layer.self_attn, 'num_heads', 12)  # Default to 12 if not found
            # Get device, with fallback to CPU
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            cache = E2CRFCache(
                max_len=self.max_len,
                num_layers=len(self.backbone.layers),
                device=device,
                **cache_kwargs
            )
        self.cache = cache
        self.use_cache = True
        
        # Create cached transformer layers
        if self.cached_backbone is None:
            # Get device from model
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            
            self.cached_backbone = nn.ModuleList()
            for layer_idx, orig_layer in enumerate(self.backbone.layers):
                # Get dim_feedforward and dropout from original layer
                dim_feedforward = 2048
                dropout = 0.1
                if hasattr(orig_layer, 'linear1') and orig_layer.linear1 is not None:
                    linear1 = orig_layer.linear1
                    if hasattr(linear1, 'out_features') and isinstance(linear1.out_features, (int, torch.Tensor)):
                        dim_feedforward = int(linear1.out_features) if isinstance(linear1.out_features, int) else int(linear1.out_features.item())
                if hasattr(orig_layer, 'dropout1') and orig_layer.dropout1 is not None:
                    dropout_val = getattr(orig_layer.dropout1, 'p', 0.1)
                    if isinstance(dropout_val, (int, float)):
                        dropout = float(dropout_val)
                
                # Get nhead from original layer
                nhead = getattr(orig_layer.self_attn, 'num_heads', n_head)
                
                # Create cached layer with same parameters on correct device
                cached_layer = CachedTransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                ).to(device)
                
                # Copy weights from original layer using state_dict
                orig_state = orig_layer.state_dict()
                cached_state = cached_layer.state_dict()
                
                # Map original layer state to cached layer state
                # Ensure tensors are on the same device
                for key in cached_state.keys():
                    if key in orig_state:
                        orig_tensor = orig_state[key]
                        # Move to device if needed
                        if isinstance(orig_tensor, torch.Tensor) and orig_tensor.device != device:
                            orig_tensor = orig_tensor.to(device)
                        cached_state[key] = orig_tensor
                
                cached_layer.load_state_dict(cached_state, strict=False)
                
                # Set cache reference
                cached_layer.set_cache(cache, layer_idx)
                self.cached_backbone.append(cached_layer)
    
    def disable_caching(self) -> None:
        """Disable caching."""
        self.use_cache = False
        self.cache = None

    def training_step(
        self, batch: DiffusableBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        loss = self.training_loss_fn(self, batch)

        self.log_dict(
            {"train/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=True,
        )
        return loss

    def validation_step(
        self, batch: DiffusableBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        loss = self.validation_loss_fn(self, batch)
        self.log_dict(
            {"val/loss": loss},
            prog_bar=True,
            batch_size=len(batch),
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr_max)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def set_loss_fn(
        self,
    ) -> tuple[
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
    ]:
        # Depending on the scheduler, get the right loss function

        if isinstance(self.noise_scheduler, SDE):
            training_loss_fn = get_sde_loss_fn(
                scheduler=self.noise_scheduler,
                train=True,
                likelihood_weighting=self.likelihood_weighting,
            )
            validation_loss_fn = get_sde_loss_fn(
                scheduler=self.noise_scheduler,
                train=False,
                likelihood_weighting=self.likelihood_weighting,
            )

            return training_loss_fn, validation_loss_fn

        else:
            raise NotImplementedError(
                f"Scheduler {self.noise_scheduler} not implemented yet, cannot set loss function."
            )

    def set_time_encoder(self) -> Union[TimeEncoding, GaussianFourierProjection]:
        if isinstance(self.noise_scheduler, SDE):
            return GaussianFourierProjection(d_model=self.d_model)

        else:
            raise NotImplementedError(
                f"Scheduler {self.noise_scheduler} not implemented yet, cannot set time encoder."
            )


class MLPScoreModule(ScoreModule):
    def __init__(
        self,
        n_channels: int,
        max_len: int,
        noise_scheduler: SDE,
        fourier_noise_scaling: bool = True,
        d_model: int = 72,
        d_mlp: int = 512,
        num_layers: int = 3,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
        likelihood_weighting: bool = False,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            max_len=max_len,
            noise_scheduler=noise_scheduler,
            fourier_noise_scaling=fourier_noise_scaling,
            d_model=d_model,
            num_layers=num_layers,
            n_head=1,
            num_training_steps=num_training_steps,
            lr_max=lr_max,
            likelihood_weighting=likelihood_weighting,
        )

        # Change the components that should be different in our score model
        self.embedder = nn.Linear(
            in_features=max_len * n_channels, out_features=d_model
        )
        self.unembedder = nn.Linear(
            in_features=d_model, out_features=max_len * n_channels
        )

        self.backbone = nn.ModuleList(  # type: ignore
            [
                MLP(in_channels=d_model, hidden_channels=[d_mlp, d_model], dropout=0.1)
                for _ in range(num_layers)
            ]
        )
        self.pos_encoder = None

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Flatten the tensor
        X = rearrange(X, "b t c -> b (t c)")

        # Channel embedding
        X = self.embedder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps, use_time_axis=False)

        # Backbone
        for layer in self.backbone:  # type: ignore
            X = X + layer(X)

        # Channel unembedding
        X = self.unembedder(X)

        # Unflatten the tensor
        X = rearrange(X, "b (t c) -> b t c", t=self.max_len, c=self.n_channels)

        assert isinstance(X, torch.Tensor)

        return X


class LSTMScoreModule(ScoreModule):
    def __init__(
        self,
        n_channels: int,
        max_len: int,
        noise_scheduler: SDE,
        fourier_noise_scaling: bool = True,
        d_model: int = 72,
        num_layers: int = 3,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
        likelihood_weighting: bool = False,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            max_len=max_len,
            noise_scheduler=noise_scheduler,
            fourier_noise_scaling=fourier_noise_scaling,
            d_model=d_model,
            num_layers=num_layers,
            n_head=1,
            num_training_steps=num_training_steps,
            lr_max=lr_max,
            likelihood_weighting=likelihood_weighting,
        )

        # Change the components that should be different in our score model
        self.backbone = nn.ModuleList(  # type: ignore
            [
                nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model,
                    batch_first=True,
                    bidirectional=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.pos_encoder = None

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        assert X.size()[1:] == (
            self.max_len,
            self.n_channels,
        ), f"X has wrong shape, should be {(X.size(0), self.max_len, self.n_channels)}, but is {X.size()}"

        timesteps = batch.timesteps
        assert timesteps is not None and timesteps.size(0) == len(batch)

        # Channel embedding
        X = self.embedder(X)

        # Add time encoding
        X = self.time_encoder(X, timesteps)

        # Backbone
        for layer in self.backbone:  # type: ignore
            X = X + layer(X)[0]

        # Channel unembedding
        X = self.unembedder(X)

        assert isinstance(X, torch.Tensor)

        return X
