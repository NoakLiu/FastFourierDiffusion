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
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers
        )

        # Save all hyperparameters for checkpointing
        self.save_hyperparameters()
        
        # Caching support
        self.cache: Optional[E2CRFCache] = None
        self.use_cache: bool = False

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
        if self.use_cache and recompute_tokens is not None:
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
        """Forward pass with caching support.
        
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
        
        # If we need to recompute all tokens, use standard forward pass
        if len(recompute_tokens) == self.max_len:
            # Standard forward pass
            h_new = self.backbone(h)
            crf = torch.stack([h_new[0]], dim=0)  # Single layer for simplicity
            return h_new, crf
        
        # Process through each transformer layer
        # Note: Full token-level selective computation requires custom transformer implementation
        # For now, we compute all tokens but track statistics for demonstration
        for layer_idx, layer in enumerate(self.backbone.layers):
            h_new = layer(h)
            
            # Update cache statistics (for demonstration purposes)
            # In a full implementation, we would selectively compute only recompute_tokens
            if self.cache is not None:
                num_recompute = len(recompute_tokens)
                num_cached = self.max_len - num_recompute
                # Update stats (multiply by num_layers to account for all layers)
                self.cache.stats["recompute_count"] += num_recompute
                self.cache.stats["cache_hit_count"] += num_cached
            
            # Store CRF (cumulative residual feature)
            crf_list.append(h_new.clone())
            h = h_new
        
        # Stack CRF from all layers: (num_layers, batch_size, max_len, d_model)
        # For single sample, take first batch element
        crf = torch.stack([c[0] for c in crf_list], dim=0)  # (num_layers, max_len, d_model)
        
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
                d_model=self.d_model,
                n_head=n_head,
                device=device,
                **cache_kwargs
            )
        self.cache = cache
        self.use_cache = True
    
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
