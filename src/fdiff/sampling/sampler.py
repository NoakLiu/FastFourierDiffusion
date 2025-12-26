from typing import Optional, Literal

import torch
from tqdm import tqdm

from fdiff.models.score_models import ScoreModule
from fdiff.schedulers.sde import SDE
from fdiff.utils.caching import E2CRFCache
from fdiff.utils.dataclasses import DiffusableBatch
from fdiff.utils.fourier import dft, idft
from fdiff.utils.fresca import apply_fresca_to_score, analyze_frequency_content


class DiffusionSampler:
    def __init__(
        self,
        score_model: ScoreModule,
        sample_batch_size: int,
        use_cache: bool = False,
        cache_kwargs: Optional[dict] = None,
        # FreSca parameters
        use_fresca: bool = False,
        fresca_low_scale: float = 1.0,
        fresca_high_scale: float = 1.5,
        fresca_cutoff_ratio: float = 0.5,
        fresca_cutoff_strategy: Literal["spatial", "energy"] = "energy",
    ) -> None:
        self.score_model = score_model
        self.noise_scheduler = score_model.noise_scheduler

        self.sample_batch_size = sample_batch_size
        self.n_channels = score_model.n_channels
        self.max_len = score_model.max_len
        
        # Caching support
        self.use_cache = use_cache
        if use_cache:
            cache_kwargs = cache_kwargs or {}
            self.score_model.enable_caching(**cache_kwargs)
        
        # FreSca support
        self.use_fresca = use_fresca
        self.fresca_low_scale = fresca_low_scale
        self.fresca_high_scale = fresca_high_scale
        self.fresca_cutoff_ratio = fresca_cutoff_ratio
        self.fresca_cutoff_strategy = fresca_cutoff_strategy

    def reverse_diffusion_step(
        self, 
        batch: DiffusableBatch,
        step: int = 0,
        recompute_tokens: Optional[set[int]] = None,
    ) -> torch.Tensor:
        # Get X and timesteps
        X = batch.X
        timesteps = batch.timesteps

        # Check the validity of the timestep (current implementation assumes same time for all samples)
        assert timesteps is not None and timesteps.size(0) == len(batch)
        assert torch.min(timesteps) == torch.max(timesteps)

        # Predict score for the current batch with caching support
        if self.use_cache and recompute_tokens is not None:
            score, crf = self.score_model(
                batch, 
                recompute_tokens=recompute_tokens,
                step=step,
                return_crf=True
            )
            # Update cache with CRF (with timestep for FreqCa)
            if self.score_model.cache is not None:
                timestep_val = timesteps[0].item() if hasattr(timesteps[0], 'item') else float(timesteps[0])
                self.score_model.cache.update_crf(crf, timestep=timestep_val)
                self.score_model.cache.current_step = step
        else:
            score = self.score_model(batch)
        
        # Apply FreSca frequency scaling if enabled
        if self.use_fresca:
            timestep_val = timesteps[0].item() if hasattr(timesteps[0], 'item') else float(timesteps[0])
            # Ensure cutoff_strategy is correct type
            cutoff_strategy: Literal["spatial", "energy"] = (
                "energy" if self.fresca_cutoff_strategy == "energy" else "spatial"
            )
            score = apply_fresca_to_score(
                score,
                low_scale=self.fresca_low_scale,
                high_scale=self.fresca_high_scale,
                cutoff_ratio=self.fresca_cutoff_ratio,
                cutoff_strategy=cutoff_strategy,
                timestep=timestep_val,
                num_steps=getattr(self, '_num_diffusion_steps', None),
            )
        
        # Apply a step of reverse diffusion
        output = self.noise_scheduler.step(
            model_output=score, timestep=timesteps[0].item(), sample=X
        )

        X_prev = output.prev_sample
        assert isinstance(X_prev, torch.Tensor)

        return X_prev

    def sample(
        self, num_samples: int, num_diffusion_steps: Optional[int] = None
    ) -> torch.Tensor:
        # Store num_diffusion_steps for FreSca dynamic scheduling
        if num_diffusion_steps is not None:
            self._num_diffusion_steps = num_diffusion_steps
        
        # Set the score model in eval mode and move it to GPU
        self.score_model.eval()

        # If the number of diffusion steps is not provided, use the number of training steps
        num_diffusion_steps = (
            self.score_model.num_training_steps
            if num_diffusion_steps is None
            else num_diffusion_steps
        )
        self.noise_scheduler.set_timesteps(num_diffusion_steps)

        # Create the list that will store the samples
        all_samples = []

        # Compute the required amount of batches
        num_batches = max(1, num_samples // self.sample_batch_size)

        # No need to track gradients when sampling
        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc="Sampling",
                unit="batch",
                leave=False,
                colour="blue",
            ):
                # Compute the batch size
                batch_size = min(
                    num_samples - batch_idx * self.sample_batch_size,
                    self.sample_batch_size,
                )
                # Sample from noise distribution
                X = self.sample_prior(batch_size)
                
                # Reset cache for new batch
                if self.use_cache and self.score_model.cache is not None:
                    self.score_model.cache.reset()

                # Perform the diffusion step by step
                timestep_list = list(self.noise_scheduler.timesteps)
                for step_idx, t in enumerate(tqdm(
                    timestep_list,
                    desc="Diffusion",
                    unit="step",
                    leave=False,
                    colour="green",
                )):
                    # Define timesteps for the batch
                    t_val = t.item() if hasattr(t, 'item') else float(t)
                    timesteps = torch.full(
                        (batch_size,),
                        t_val,
                        dtype=torch.long if isinstance(t_val, int) else torch.float32,
                        device=self.score_model.device,
                        requires_grad=False,
                    )
                    # Create diffusable batch
                    batch = DiffusableBatch(X=X, y=None, timesteps=timesteps)
                    
                    # Determine which tokens to recompute (for caching)
                    recompute_tokens = None
                    if self.use_cache and self.score_model.cache is not None:
                        cache = self.score_model.cache
                        cache.current_step = step_idx
                        
                        # Convert to frequency domain for event intensity computation
                        X_tilde = dft(X)
                        
                        # Compute event intensity using CRF if available
                        # Try FreqCa prediction first, then fallback to cached CRF
                        event_intensity = 1.0
                        crf_for_intensity = None
                        
                        if cache.use_freqca:
                            # Try to predict CRF using FreqCa
                            predicted_crf = cache.predict_crf_freqca(target_timestep=t_val)
                            if predicted_crf is not None:
                                crf_for_intensity = predicted_crf.unsqueeze(0) if predicted_crf.dim() == 2 else predicted_crf
                        
                        # Fallback to cached CRF if prediction not available
                        if crf_for_intensity is None and cache.crf_cache is not None:
                            # Use final layer CRF for intensity computation
                            crf_final = cache.crf_cache[-1] if cache.crf_cache.dim() == 3 else cache.crf_cache
                            crf_for_intensity = crf_final.unsqueeze(0) if crf_final.dim() == 2 else crf_final
                        
                        if crf_for_intensity is not None:
                            # Stack to match expected shape (num_layers, max_len, d_model)
                            if crf_for_intensity.dim() == 2:
                                crf_for_intensity = crf_for_intensity.unsqueeze(0)
                            # Expand to match num_layers if needed
                            if crf_for_intensity.shape[0] == 1 and cache.num_layers > 1:
                                crf_for_intensity = crf_for_intensity.repeat(cache.num_layers, 1, 1)
                            event_intensity = cache.compute_event_intensity(
                                crf_for_intensity, step_idx
                            )
                        else:
                            event_intensity = 1.0  # High intensity on first step
                        
                        # Determine recompute set using the cache's method
                        recompute_tokens = cache.determine_recompute_set(
                            X_tilde, event_intensity, step_idx
                        )
                    
                    # Return denoised X
                    X = self.reverse_diffusion_step(
                        batch, 
                        step=step_idx,
                        recompute_tokens=recompute_tokens
                    )

                # Add the samples to the list
                all_samples.append(X.cpu())

        return torch.cat(all_samples, dim=0)

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        # Sample from the prior distribution
        if isinstance(self.noise_scheduler, SDE):
            X = self.noise_scheduler.prior_sampling(
                (batch_size, self.max_len, self.n_channels)
            ).to(device=self.score_model.device)

        else:
            raise NotImplementedError("Scheduler not recognized.")

        assert isinstance(X, torch.Tensor)
        return X
