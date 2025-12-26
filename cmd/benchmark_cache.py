"""Benchmarking script for E2-CRF caching acceleration.

This script compares the speedup achieved by E2-CRF caching versus
standard inference, and performs ablation studies.
"""

import time
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig

from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.caching import E2CRFCache

# Fix for PyTorch 2.6+ weights_only loading
from fdiff.schedulers.sde import VPScheduler, VEScheduler
torch.serialization.add_safe_globals([VPScheduler, VEScheduler])


def benchmark_sampling(
    score_model: ScoreModule,
    num_samples: int = 10,
    num_diffusion_steps: int = 100,
    use_cache: bool = False,
    cache_kwargs: Optional[dict] = None,
) -> dict:
    """Benchmark sampling with or without caching.
    
    Args:
        score_model: The score model to use
        num_samples: Number of samples to generate
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
        cache_kwargs=cache_kwargs,
    )
    
    # Warmup
    _ = sampler.sample(num_samples=1, num_diffusion_steps=10)
    
    # Benchmark
    start_time = time.time()
    samples = sampler.sample(
        num_samples=num_samples,
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
        "samples": samples,
        "cache_stats": cache_stats,
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
    
    if model_id == "latest":
        # Find latest checkpoint
        checkpoints = list(log_dir.glob("*/checkpoints/*.ckpt"))
        if not checkpoints:
            raise ValueError("No checkpoints found")
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    else:
        checkpoint_path = log_dir / model_id / "checkpoints" / "*.ckpt"
        checkpoint_files = list(checkpoint_path.parent.glob(checkpoint_path.name))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint found for model_id: {model_id}")
        checkpoint_path = checkpoint_files[0]
    
    # Load model from checkpoint
    # Use weights_only=False for PyTorch 2.6+ compatibility
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
    
    num_samples = cfg.get("num_samples", 10)
    num_diffusion_steps = cfg.get("num_diffusion_steps", 100)
    
    print("=" * 80)
    print("E2-CRF Caching Benchmark")
    print("=" * 80)
    
    # Benchmark without caching
    print("\n1. Benchmarking WITHOUT caching...")
    results_no_cache = benchmark_sampling(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        use_cache=False,
    )
    time_no_cache = results_no_cache["elapsed_time"]
    print(f"   Time: {time_no_cache:.2f}s")
    
    # Benchmark with caching (default settings)
    print("\n2. Benchmarking WITH E2-CRF caching (default settings)...")
    results_cache = benchmark_sampling(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        use_cache=True,
        cache_kwargs={},
    )
    time_cache = results_cache["elapsed_time"]
    cache_stats = results_cache["cache_stats"]
    print(f"   Time: {time_cache:.2f}s")
    print(f"   Speedup: {time_no_cache / time_cache:.2f}x")
    if cache_stats:
        print(f"   Cache hit ratio: {cache_stats.get('cache_hit_ratio', 0):.2%}")
        print(f"   Cache ratio: {cache_stats.get('cache_ratio', 0):.2%}")
    
    # Ablation: Different K values (low-frequency tokens always recomputed)
    print("\n3. Ablation: Varying K (low-frequency tokens)...")
    for K in [0, 3, 5, 10]:
        print(f"   K={K}: ", end="", flush=True)
        results = benchmark_sampling(
            score_model=score_model,
            num_samples=num_samples,
            num_diffusion_steps=num_diffusion_steps,
            use_cache=True,
            cache_kwargs={"K": K},
        )
        time_k = results["elapsed_time"]
        speedup = time_no_cache / time_k
        print(f"Time: {time_k:.2f}s, Speedup: {speedup:.2f}x")
    
    # Ablation: Different R values (error-feedback interval)
    print("\n4. Ablation: Varying R (error-feedback interval)...")
    for R in [5, 10, 20, 50]:
        print(f"   R={R}: ", end="", flush=True)
        results = benchmark_sampling(
            score_model=score_model,
            num_samples=num_samples,
            num_diffusion_steps=num_diffusion_steps,
            use_cache=True,
            cache_kwargs={"R": R},
        )
        time_r = results["elapsed_time"]
        speedup = time_no_cache / time_r
        print(f"Time: {time_r:.2f}s, Speedup: {speedup:.2f}x")
    
    # Ablation: Different tau_0 values (base threshold)
    print("\n5. Ablation: Varying tau_0 (base threshold)...")
    for tau_0 in [0.05, 0.1, 0.2, 0.5]:
        print(f"   tau_0={tau_0}: ", end="", flush=True)
        results = benchmark_sampling(
            score_model=score_model,
            num_samples=num_samples,
            num_diffusion_steps=num_diffusion_steps,
            use_cache=True,
            cache_kwargs={"tau_0": tau_0},
        )
        time_tau = results["elapsed_time"]
        speedup = time_no_cache / time_tau
        print(f"Time: {time_tau:.2f}s, Speedup: {speedup:.2f}x")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Baseline (no cache):     {time_no_cache:.2f}s")
    print(f"E2-CRF (default):       {time_cache:.2f}s")
    print(f"Overall speedup:         {time_no_cache / time_cache:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()

