"""Training script with automatic cache performance benchmarking.

This script trains a model and then automatically benchmarks cache performance
after training completes.
"""

import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.callbacks import SamplingCallback
from fdiff.utils.extraction import dict_to_str, get_training_params
from fdiff.utils.wandb import maybe_initialize_wandb


def benchmark_cache_performance(
    score_model: ScoreModule,
    num_samples: int = 5,
    num_diffusion_steps: int = 5,
    cache_kwargs: Optional[dict] = None,
    use_fresca: bool = False,
    fresca_kwargs: Optional[dict] = None,
) -> dict:
    """Benchmark cache performance after training.
    
    Args:
        score_model: The trained score model
        num_samples: Number of samples for benchmark
        num_diffusion_steps: Number of diffusion steps
        cache_kwargs: Optional cache configuration
        use_fresca: Whether to test with FreSca
        fresca_kwargs: Optional FreSca configuration
        
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 80)
    print("Cache Performance Benchmark")
    print("=" * 80)
    
    # Prepare FreSca kwargs
    fresca_kwargs = fresca_kwargs or {}
    if use_fresca:
        fresca_kwargs.setdefault("fresca_low_scale", 1.0)
        fresca_kwargs.setdefault("fresca_high_scale", 1.5)
        fresca_kwargs.setdefault("fresca_cutoff_ratio", 0.5)
        fresca_kwargs.setdefault("fresca_cutoff_strategy", "energy")
    
    # Benchmark without cache
    print("\n1. Benchmarking WITHOUT caching...")
    sampler_no_cache = DiffusionSampler(
        score_model=score_model,
        sample_batch_size=1,
        use_cache=False,
    )
    
    # Warmup (reduced steps for faster benchmark)
    _ = sampler_no_cache.sample(num_samples=1, num_diffusion_steps=5)
    
    # Benchmark
    start_time = time.time()
    _ = sampler_no_cache.sample(
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
    )
    time_no_cache = time.time() - start_time
    print(f"   Time: {time_no_cache:.2f}s")
    
    # Benchmark with cache
    print("\n2. Benchmarking WITH E2-CRF caching...")
    sampler_with_cache = DiffusionSampler(
        score_model=score_model,
        sample_batch_size=1,
        use_cache=True,
        cache_kwargs=cache_kwargs or {},
        use_fresca=use_fresca,
        **fresca_kwargs,
    )
    
    # Reset cache
    if score_model.cache is not None:
        score_model.cache.reset()
    
    # Warmup (reduced steps for faster benchmark)
    _ = sampler_with_cache.sample(num_samples=1, num_diffusion_steps=5)
    
    # Reset cache again after warmup
    if score_model.cache is not None:
        score_model.cache.reset()
    
    # Benchmark
    start_time = time.time()
    _ = sampler_with_cache.sample(
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
    )
    time_with_cache = time.time() - start_time
    print(f"   Time: {time_with_cache:.2f}s")
    
    # Get cache statistics
    cache_stats = {}
    if score_model.cache is not None:
        cache_stats = score_model.cache.get_cache_stats()
    
    # Calculate speedup
    speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0.0
    print(f"   Speedup: {speedup:.2f}x")
    
    # Print cache statistics
    if cache_stats:
        print(f"\n   Cache Statistics:")
        print(f"   - Cache hit ratio: {cache_stats.get('cache_hit_ratio', 0):.2%}")
        print(f"   - Cache ratio: {cache_stats.get('cache_ratio', 0):.2%}")
        print(f"   - Recompute count: {cache_stats.get('recompute_count', 0)}")
        print(f"   - Cache hit count: {cache_stats.get('cache_hit_count', 0)}")
        if 'freq_decomp_count' in cache_stats:
            print(f"\n   FreqCa Statistics:")
            print(f"   - Frequency decompositions: {cache_stats.get('freq_decomp_count', 0)}")
            print(f"   - Decompositions skipped: {cache_stats.get('freq_decomp_skipped', 0)}")
            if cache_stats.get('freq_decomp_count', 0) > 0:
                decomp_ratio = cache_stats.get('freq_decomp_ratio', 0)
                print(f"   - Decomposition ratio: {decomp_ratio:.2%}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Baseline (no cache):     {time_no_cache:.2f}s")
    print(f"                         {time_no_cache / num_samples:.2f}s per sample")
    print(f"                         {time_no_cache / (num_samples * num_diffusion_steps):.4f}s per step")
    print(f"E2-CRF (with cache):     {time_with_cache:.2f}s")
    print(f"                         {time_with_cache / num_samples:.2f}s per sample")
    print(f"                         {time_with_cache / (num_samples * num_diffusion_steps):.4f}s per step")
    print(f"Overall speedup:         {speedup:.2f}x")
    if cache_stats:
        print(f"\nCache Performance:")
        print(f"  Cache hit ratio:      {cache_stats.get('cache_hit_ratio', 0):.2%}")
        print(f"  Cache coverage:       {cache_stats.get('cache_ratio', 0):.2%}")
        if 'freq_decomp_count' in cache_stats:
            print(f"  Freq decompositions:  {cache_stats.get('freq_decomp_count', 0)}")
            print(f"  Decomp efficiency:   {cache_stats.get('freq_decomp_ratio', 0):.2%}")
    print("=" * 80)
    
    return {
        "time_no_cache": time_no_cache,
        "time_with_cache": time_with_cache,
        "speedup": speedup,
        "cache_stats": cache_stats,
    }


class TrainingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        torch.manual_seed(cfg.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Read out the config
        logging.info(
            f"Welcome in the training script! You are using the following config:\n{dict_to_str(cfg)}"
        )

        # Maybe initialize wandb
        run_id = maybe_initialize_wandb(cfg)

        # Instantiate all the components
        self.score_model: ScoreModule = instantiate(cfg.score_model)
        self.trainer: pl.Trainer = instantiate(cfg.trainer)
        self.datamodule: Datamodule = instantiate(cfg.datamodule)

        # Save the config to the log directory
        save_dir = Path.cwd() / "lightning_logs" / run_id
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Saving the config into {save_dir}.")
        OmegaConf.save(config=cfg, f=save_dir / "train_config.yaml")
        
        self.save_dir = save_dir
        self.run_id = run_id

        # Set-up dataset
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

        # Finish instantiation of the model if necessary
        if isinstance(self.score_model, partial):
            training_params = get_training_params(self.datamodule, self.trainer)
            self.score_model = self.score_model(**training_params)

        # Possibly setup the datamodule in the sampling callback
        for callback in self.trainer.callbacks:  # type: ignore
            if isinstance(callback, SamplingCallback):
                callback.setup_datamodule(datamodule=self.datamodule)

    def train(self) -> None:
        assert not (
            self.score_model.scale_noise and not self.datamodule.fourier_transform
        ), "You cannot use noise scaling without the Fourier transform."
        self.trainer.fit(model=self.score_model, datamodule=self.datamodule)


@hydra.main(version_base=None, config_path="conf", config_name="train_with_cache_benchmark")
def main(cfg: DictConfig) -> None:
    """Main function: train model and benchmark cache performance.
    
    Args:
        cfg: Hydra configuration
    """
    # Train the model
    runner = TrainingRunner(cfg)
    runner.train()
    
    # After training, benchmark cache performance
    print("\n" + "=" * 80)
    print("Training completed! Starting cache performance benchmark...")
    print("=" * 80)
    print("Note: Using reduced parameters for faster benchmark.")
    print("      Override with: cache_benchmark.num_samples=X cache_benchmark.num_diffusion_steps=Y")
    print("=" * 80)
    
    # Get benchmark parameters from config or use defaults (reduced for faster benchmark)
    num_samples = cfg.get("cache_benchmark", {}).get("num_samples", 5)
    num_diffusion_steps = cfg.get("cache_benchmark", {}).get("num_diffusion_steps", 50)
    cache_kwargs = cfg.get("cache_benchmark", {}).get("cache_kwargs", {})
    use_fresca = cfg.get("cache_benchmark", {}).get("use_fresca", False)
    fresca_kwargs = cfg.get("cache_benchmark", {}).get("fresca_kwargs", {})
    
    # Run benchmark
    results = benchmark_cache_performance(
        score_model=runner.score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        cache_kwargs=cache_kwargs,
        use_fresca=use_fresca,
        fresca_kwargs=fresca_kwargs,
    )
    
    # Save results to file
    import json
    results_file = runner.save_dir / "cache_benchmark_results.json"
    # Convert cache_stats to serializable format
    results_to_save = {
        "time_no_cache": results["time_no_cache"],
        "time_with_cache": results["time_with_cache"],
        "speedup": results["speedup"],
        "cache_stats": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in results["cache_stats"].items()
        },
        "num_samples": num_samples,
        "num_diffusion_steps": num_diffusion_steps,
    }
    with open(results_file, "w") as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nBenchmark results saved to: {results_file}")


if __name__ == "__main__":
    main()

