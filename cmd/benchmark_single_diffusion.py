"""Simple benchmark script for comparing single diffusion speed with and without cache.

This script runs a single diffusion process (multiple steps) and compares
the speed with and without caching.
"""

import time
from pathlib import Path
from typing import Optional
import warnings

import hydra
import torch
from omegaconf import DictConfig

from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.caching import E2CRFCache

# Fix for PyTorch 2.6+ weights_only loading
from fdiff.schedulers.sde import VPScheduler, VEScheduler
torch.serialization.add_safe_globals([VPScheduler, VEScheduler])

warnings.filterwarnings("ignore")


def benchmark_single_diffusion(
    score_model: ScoreModule,
    num_diffusion_steps: int = 100,
    use_cache: bool = False,
    cache_kwargs: Optional[dict] = None,
) -> dict:
    """Benchmark a single diffusion process with or without caching.
    
    Args:
        score_model: The score model to use
        num_diffusion_steps: Number of diffusion steps
        use_cache: Whether to use caching
        cache_kwargs: Optional cache configuration
        
    Returns:
        Dictionary with timing and statistics
    """
    sampler = DiffusionSampler(
        score_model=score_model,
        sample_batch_size=1,
        use_cache=use_cache,
        cache_kwargs=cache_kwargs or {},
    )
    
    # Reset cache before benchmarking to get accurate statistics
    if use_cache and score_model.cache is not None:
        score_model.cache.reset()
    
    # Warmup: run a few steps to initialize (need at least 2 steps for scheduler)
    warmup_steps = min(10, num_diffusion_steps)
    if warmup_steps >= 2:
        _ = sampler.sample(num_samples=1, num_diffusion_steps=warmup_steps)
        
        # Reset cache again after warmup to get clean statistics
        if use_cache and score_model.cache is not None:
            score_model.cache.reset()
    
    # Benchmark: single diffusion process
    print(f"Running {num_diffusion_steps} diffusion steps...")
    start_time = time.time()
    sample = sampler.sample(
        num_samples=1,
        num_diffusion_steps=num_diffusion_steps
    )
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    # Get cache statistics if available
    cache_stats = {}
    if use_cache and score_model.cache is not None:
        cache_stats = score_model.cache.get_cache_stats()
    
    return {
        "elapsed_time": elapsed_time,
        "sample": sample,
        "cache_stats": cache_stats,
        "num_diffusion_steps": num_diffusion_steps,
    }


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    """Main benchmarking function.
    
    Args:
        cfg: Hydra configuration
    """
    # Load model
    model_id = cfg.get("model_id", "latest")
    log_dir = Path("lightning_logs")
    
    if model_id == "latest" or model_id == "your_model_id":
        # Find latest checkpoint
        checkpoints = list(log_dir.glob("*/checkpoints/*.ckpt"))
        if not checkpoints:
            raise ValueError("No checkpoints found in lightning_logs. Please train a model first or specify a valid model_id.")
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        # Extract actual model_id from path
        model_id = checkpoint_path.parent.parent.name
        print(f"Using latest model: {model_id}")
    else:
        checkpoint_path = log_dir / model_id / "checkpoints" / "*.ckpt"
        checkpoint_files = list(checkpoint_path.parent.glob(checkpoint_path.name))
        if not checkpoint_files:
            # Try to find latest if specified model not found
            print(f"Warning: Model {model_id} not found. Trying to use latest model...")
            checkpoints = list(log_dir.glob("*/checkpoints/*.ckpt"))
            if not checkpoints:
                raise ValueError(f"No checkpoint found for model_id: {model_id}, and no checkpoints found in lightning_logs.")
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            model_id = checkpoint_path.parent.parent.name
            print(f"Using latest model instead: {model_id}")
        else:
            checkpoint_path = checkpoint_files[0]
    
    # Load model from checkpoint
    score_model = ScoreModule.load_from_checkpoint(
        str(checkpoint_path),
        weights_only=False
    )
    score_model.eval()
    # Move to appropriate device
    if torch.cuda.is_available():
        score_model = score_model.cuda()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        score_model = score_model.to('mps')
    
    num_diffusion_steps = cfg.get("num_diffusion_steps", 100)
    
    print("=" * 80)
    print("Single Diffusion Speed Comparison")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Diffusion steps: {num_diffusion_steps}")
    print(f"Device: {next(score_model.parameters()).device}")
    print()
    
    # Benchmark without caching
    print("1. Benchmarking WITHOUT caching...")
    results_no_cache = benchmark_single_diffusion(
        score_model=score_model,
        num_diffusion_steps=num_diffusion_steps,
        use_cache=False,
    )
    time_no_cache = results_no_cache["elapsed_time"]
    print(f"   Time: {time_no_cache:.4f}s")
    print(f"   Time per step: {time_no_cache / num_diffusion_steps:.6f}s")
    print()
    
    # Benchmark with caching (default settings)
    print("2. Benchmarking WITH E2-CRF caching (default settings)...")
    results_cache = benchmark_single_diffusion(
        score_model=score_model,
        num_diffusion_steps=num_diffusion_steps,
        use_cache=True,
        cache_kwargs={},
    )
    time_cache = results_cache["elapsed_time"]
    cache_stats = results_cache["cache_stats"]
    speedup = time_no_cache / time_cache
    print(f"   Time: {time_cache:.4f}s")
    print(f"   Time per step: {time_cache / num_diffusion_steps:.6f}s")
    print(f"   Speedup: {speedup:.2f}x")
    if cache_stats:
        print(f"\n   Cache Statistics:")
        print(f"   - Cache hit ratio: {cache_stats.get('cache_hit_ratio', 0):.2%}")
        print(f"   - Cache ratio: {cache_stats.get('cache_ratio', 0):.2%}")
        print(f"   - Recompute count: {cache_stats.get('recompute_count', 0)}")
        print(f"   - Cache hit count: {cache_stats.get('cache_hit_count', 0)}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Without cache:     {time_no_cache:.4f}s ({time_no_cache / num_diffusion_steps:.6f}s/step)")
    print(f"With cache:        {time_cache:.4f}s ({time_cache / num_diffusion_steps:.6f}s/step)")
    print(f"Speedup:           {speedup:.2f}x")
    if speedup < 1.0:
        slowdown = 1.0 / speedup
        print(f"Slowdown:          {slowdown:.2f}x (cache is slower)")
    print("=" * 80)


if __name__ == "__main__":
    main()

