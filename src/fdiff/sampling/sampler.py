from typing import Optional

import torch
from tqdm import tqdm

from fdiff.models.score_models import ScoreModule
from fdiff.schedulers.sde import SDE
from fdiff.utils.caching import E2CRFCache
from fdiff.utils.dataclasses import DiffusableBatch
from fdiff.utils.fourier import dft, idft


class DiffusionSampler:
    def __init__(
        self,
        score_model: ScoreModule,
        sample_batch_size: int,
        use_cache: bool = False,
        cache_kwargs: Optional[dict] = None,
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
            # Update cache with CRF
            if self.score_model.cache is not None:
                self.score_model.cache.update_crf(crf)
                self.score_model.cache.current_step = step
        else:
            score = self.score_model(batch)
        
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
                        
                        # Compute event intensity if we have previous CRF
                        # Use a lightweight approximation: compute energy in time domain
                        if cache.crf_cache is not None:
                            # Use time domain energy as a proxy for frequency domain changes
                            # This avoids expensive DFT computation every step
                            X_energy = torch.norm(X, dim=-1).pow(2)  # (batch_size, max_len)
                            
                            # Simple heuristic: if energy changed significantly, recompute more
                            if hasattr(cache, '_prev_energy') and cache._prev_energy is not None:
                                energy_change = torch.norm(X_energy - cache._prev_energy).item()
                                energy_norm = torch.norm(cache._prev_energy).item() + cache.eta
                                event_intensity = (energy_change / energy_norm) if energy_norm > 0 else 1.0
                            else:
                                event_intensity = 1.0
                            cache._prev_energy = X_energy.clone()
                        else:
                            event_intensity = 1.0  # High intensity on first step
                            cache._prev_energy = torch.norm(X, dim=-1).pow(2).clone()
                        
                        # Determine recompute set based on event intensity
                        # For simplicity, use a fixed strategy based on event intensity
                        if event_intensity > cache.tau_warn:
                            # High intensity: recompute all
                            recompute_tokens = set(range(self.max_len))
                        else:
                            # Low intensity: only recompute low-frequency tokens + some high-freq
                            recompute_tokens = set(range(cache.K))
                            # Add some high-frequency tokens randomly
                            high_freq = list(range(cache.K, self.max_len))
                            num_probe = max(1, int(len(high_freq) * cache.random_probe_ratio))
                            if num_probe > 0 and len(high_freq) > 0:
                                probe_indices = torch.randperm(len(high_freq), device=X.device)[:num_probe]
                                recompute_tokens.update([high_freq[i] for i in probe_indices.cpu().tolist()])
                    
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
