import logging
import time
from typing import Optional, List, Dict, Any

import pytorch_lightning as pl
import torch

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.metrics import Metric, MetricCollection
from fdiff.sampling.sampler import DiffusionSampler

from .fourier import idft


class SamplingCallback(pl.Callback):
    def __init__(
        self,
        every_n_epochs: int,
        sample_batch_size: int,
        num_samples: int,
        num_diffusion_steps: int,
        metrics: list[Metric],
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.sample_batch_size = sample_batch_size
        self.num_samples = num_samples
        self.num_diffusion_steps = num_diffusion_steps
        self.metrics = metrics
        self.datamodule_initialized = False

    def setup_datamodule(self, datamodule: Datamodule) -> None:
        # Exract the necessary information from the datamodule
        self.standardize = datamodule.standardize
        self.fourier_transform = datamodule.fourier_transform
        self.feature_mean, self.feature_std = datamodule.feature_mean_and_std
        self.metric_collection = MetricCollection(
            metrics=self.metrics,
            original_samples=datamodule.X_train,
            include_baselines=False,
        )
        self.datamodule_initialized = True

    def on_train_start(self, trainer: pl.Trainer, pl_module: ScoreModule) -> None:
        # Initialize the sampler with the score model
        self.sampler = DiffusionSampler(
            score_model=pl_module,
            sample_batch_size=self.sample_batch_size,
        )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            or trainer.current_epoch + 1 == trainer.max_epochs
        ):
            # Sample from score model
            X = self.sample()

            # Compute metrics
            results = self.metric_collection(X)

            # Add a metrics/ suffix to the keys in results
            results = {f"metrics/{key}": value for key, value in results.items()}

            # Log metrics
            pl_module.log_dict(results, on_step=False, on_epoch=True)

    def sample(self) -> torch.Tensor:
        # Check that the dqtqmodule is initialized
        assert self.datamodule_initialized, (
            "The datamodule has not been initialized. "
            "Please call `setup_datamodule` before sampling."
        )

        # Sample from score model

        X = self.sampler.sample(
            num_samples=self.num_samples,
            num_diffusion_steps=self.num_diffusion_steps,
        )

        # Map to the original scale if the input was standardized
        if self.standardize:
            X = X * self.feature_std + self.feature_mean

        # If sampling in frequency domain, bring back the sample to time domain
        if self.fourier_transform:
            X = idft(X)
        assert isinstance(X, torch.Tensor)
        return X


class DiffusionMethodComparisonCallback(pl.Callback):
    """Callback to compare different diffusion methods during training.
    
    Compares multiple diffusion configurations (different steps, cache settings, etc.)
    at the end of each epoch.
    """
    
    def __init__(
        self,
        every_n_epochs: int = 1,
        num_samples: int = 3,  # Small number for fast comparison
        methods: Optional[List[Dict[str, Any]]] = None,
        warmup_steps: int = 2,  # Minimal warmup
    ) -> None:
        """Initialize diffusion method comparison callback.
        
        Args:
            every_n_epochs: Run comparison every N epochs
            num_samples: Number of samples for each method
            methods: List of method configurations to compare.
                     Each dict should contain:
                     - name: str (method name)
                     - num_diffusion_steps: int
                     - use_cache: bool (optional, default False)
                     - cache_kwargs: dict (optional)
                     - use_fresca: bool (optional, default False)
                     - fresca_kwargs: dict (optional)
            warmup_steps: Number of warmup steps
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        
        # Default methods if not provided
        if methods is None:
            methods = [
                {
                    "name": "baseline",
                    "num_diffusion_steps": 5,
                    "use_cache": False,
                },
                {
                    "name": "cache_default",
                    "num_diffusion_steps": 5,
                    "use_cache": True,
                    "cache_kwargs": {},
                },
            ]
        self.methods = methods
        
        self.results_history = []
    
    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: ScoreModule
    ) -> None:
        """Compare different diffusion methods at end of epoch."""
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            or trainer.current_epoch + 1 == trainer.max_epochs
        ):
            logging.info(
                f"Comparing diffusion methods at epoch {trainer.current_epoch}..."
            )
            
            epoch_results = {
                "epoch": trainer.current_epoch,
                "methods": {},
            }
            
            # Compare each method
            for method_config in self.methods:
                method_name = method_config["name"]
                num_steps = method_config.get("num_diffusion_steps", 5)
                use_cache = method_config.get("use_cache", False)
                cache_kwargs = method_config.get("cache_kwargs", {})
                use_fresca = method_config.get("use_fresca", False)
                fresca_kwargs = method_config.get("fresca_kwargs", {})
                
                # Create sampler
                sampler = DiffusionSampler(
                    score_model=pl_module,
                    sample_batch_size=1,
                    use_cache=use_cache,
                    cache_kwargs=cache_kwargs,
                    use_fresca=use_fresca,
                    **fresca_kwargs,
                )
                
                # Reset cache if enabled
                if use_cache and pl_module.cache is not None:
                    pl_module.cache.reset()
                
                # Warmup
                _ = sampler.sample(num_samples=1, num_diffusion_steps=self.warmup_steps)
                
                # Reset cache again after warmup
                if use_cache and pl_module.cache is not None:
                    pl_module.cache.reset()
                
                # Benchmark
                start_time = time.time()
                _ = sampler.sample(
                    num_samples=self.num_samples,
                    num_diffusion_steps=num_steps,
                )
                elapsed_time = time.time() - start_time
                
                # Get cache statistics if available
                cache_stats = {}
                if use_cache and pl_module.cache is not None:
                    cache_stats = pl_module.cache.get_cache_stats()
                
                # Store results
                method_result = {
                    "time": elapsed_time,
                    "time_per_sample": elapsed_time / self.num_samples,
                    "time_per_step": elapsed_time / (self.num_samples * num_steps),
                    "num_steps": num_steps,
                    "cache_stats": cache_stats,
                }
                epoch_results["methods"][method_name] = method_result
                
                # Log to trainer
                pl_module.log_dict(
                    {
                        f"diffusion_comparison/{method_name}/time": elapsed_time,
                        f"diffusion_comparison/{method_name}/time_per_sample": elapsed_time / self.num_samples,
                        f"diffusion_comparison/{method_name}/time_per_step": elapsed_time / (self.num_samples * num_steps),
                    },
                    on_step=False,
                    on_epoch=True,
                )
                
                if cache_stats:
                    pl_module.log_dict(
                        {
                            f"diffusion_comparison/{method_name}/cache_hit_ratio": cache_stats.get("cache_hit_ratio", 0.0),
                            f"diffusion_comparison/{method_name}/cache_ratio": cache_stats.get("cache_ratio", 0.0),
                        },
                        on_step=False,
                        on_epoch=True,
                    )
                
                logging.info(
                    f"  {method_name}: {elapsed_time:.2f}s "
                    f"({elapsed_time / self.num_samples:.3f}s/sample, "
                    f"{elapsed_time / (self.num_samples * num_steps):.4f}s/step)"
                    + (f", cache_hit: {cache_stats.get('cache_hit_ratio', 0):.1%}" if cache_stats else "")
                )
            
            # Calculate speedups relative to baseline
            baseline_time = None
            for method_name, method_result in epoch_results["methods"].items():
                if "baseline" in method_name.lower() or baseline_time is None:
                    baseline_time = method_result["time"]
                    break
            
            if baseline_time is not None:
                for method_name, method_result in epoch_results["methods"].items():
                    if method_result["time"] > 0:
                        speedup = baseline_time / method_result["time"]
                        epoch_results["methods"][method_name]["speedup"] = speedup
                        pl_module.log_dict(
                            {
                                f"diffusion_comparison/{method_name}/speedup": speedup,
                            },
                            on_step=False,
                            on_epoch=True,
                        )
            
            self.results_history.append(epoch_results)
            
            # Print summary
            if baseline_time is not None:
                logging.info("  Speedups relative to baseline:")
                for method_name, method_result in epoch_results["methods"].items():
                    if "speedup" in method_result:
                        logging.info(
                            f"    {method_name}: {method_result['speedup']:.2f}x"
                        )
