"""Simple cache benchmark with detailed section timing."""
import time
from pathlib import Path
from typing import Optional
import warnings

import hydra
import torch
from omegaconf import DictConfig

from fdiff.models.score_models import ScoreModule
from fdiff.models.cached_transformer import get_timing_stats, reset_timing_stats
from fdiff.sampling.sampler import DiffusionSampler

# Fix for PyTorch 2.6+ weights_only loading
from fdiff.schedulers.sde import VPScheduler, VEScheduler
torch.serialization.add_safe_globals([VPScheduler, VEScheduler])

warnings.filterwarnings("ignore")


class TimingProfiler:
    """Simple profiler to track timing for different sections."""
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, name: str):
        """Start timing a section."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.start_times[name] = time.time()
    
    def end(self, name: str):
        """End timing a section."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
            del self.start_times[name]
            return elapsed
        return 0.0
    
    def get_summary(self) -> dict:
        """Get summary of all timings."""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                "total": sum(times),
                "mean": sum(times) / len(times) if times else 0,
                "count": len(times),
            }
        return summary


def benchmark_step(
    score_model: ScoreModule,
    num_steps: int = 100,
    use_cache: bool = False,
    cache_kwargs: Optional[dict] = None,
) -> dict:
    """Benchmark a single diffusion step with detailed timing.
    
    Args:
        score_model: The score model
        num_steps: Number of diffusion steps
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
    
    # Reset cache before benchmarking
    if use_cache and score_model.cache is not None:
        score_model.cache.reset()
    
    # Warmup: run 2 steps
    if num_steps >= 2:
        _ = sampler.sample(num_samples=1, num_diffusion_steps=2)
        if use_cache and score_model.cache is not None:
            score_model.cache.reset()
    
    # Benchmark: run num_steps
    print(f"Running {num_steps} diffusion steps...")
    
    # Reset timing stats
    if use_cache:
        reset_timing_stats()
    
    # Time the sampling process
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    sample = sampler.sample(num_samples=1, num_diffusion_steps=num_steps)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elapsed_time = time.time() - start_time
    
    # Get cache statistics if available
    cache_stats = {}
    if use_cache and score_model.cache is not None:
        cache_stats = score_model.cache.get_cache_stats()
    
    # Get detailed timing stats
    timing_stats = {}
    if use_cache:
        timing_stats = get_timing_stats()
    
    return {
        "elapsed_time": elapsed_time,
        "sample": sample,
        "cache_stats": cache_stats,
        "timing_stats": timing_stats,
        "num_steps": num_steps,
    }


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    """Main benchmarking function."""
    # Load model
    model_id = cfg.get("model_id", "latest")
    log_dir = Path("lightning_logs")
    
    checkpoint_path = None
    if model_id == "latest":
        checkpoints = list(log_dir.glob("*/checkpoints/*.ckpt"))
        if not checkpoints:
            print("No checkpoints found. Please train a model first.")
            return
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        model_id = checkpoint_path.parent.parent.name
        print(f"Using latest model: {model_id}")
    else:
        checkpoint_files = list((log_dir / model_id / "checkpoints").glob("*.ckpt"))
        if not checkpoint_files:
            print(f"Warning: Model {model_id} not found. Trying latest...")
            checkpoints = list(log_dir.glob("*/checkpoints/*.ckpt"))
            if not checkpoints:
                print("No checkpoints found. Please train a model first.")
                return
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            model_id = checkpoint_path.parent.parent.name
        else:
            checkpoint_path = checkpoint_files[0]
    
    if checkpoint_path is None:
        print("No model checkpoint available.")
        return

    # Load model
    score_model = ScoreModule.load_from_checkpoint(
        str(checkpoint_path),
        weights_only=False
    )
    score_model.eval()
    
    # Move to device
    if torch.cuda.is_available():
        score_model = score_model.cuda()
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        score_model = score_model.to('mps')
        device = "mps"
    else:
        device = "cpu"
    
    num_steps = cfg.get("num_diffusion_steps", 100)
    
    print("=" * 80)
    print("Cache Performance Comparison")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Diffusion steps: {num_steps}")
    print("=" * 80)
    
    # Benchmark WITHOUT cache
    print("\n[1] WITHOUT Cache")
    print("-" * 80)
    results_no_cache = benchmark_step(
        score_model=score_model,
        num_steps=num_steps,
        use_cache=False,
    )
    time_no_cache = results_no_cache["elapsed_time"]
    print(f"Total time: {time_no_cache:.4f}s")
    print(f"Time per step: {time_no_cache / num_steps:.6f}s")
    
    # Benchmark WITH cache
    print("\n[2] WITH Cache")
    print("-" * 80)
    results_cache = benchmark_step(
        score_model=score_model,
        num_steps=num_steps,
        use_cache=True,
        cache_kwargs={},
    )
    time_cache = results_cache["elapsed_time"]
    cache_stats = results_cache["cache_stats"]
    timing_stats = results_cache.get("timing_stats", {})
    print(f"Total time: {time_cache:.4f}s")
    print(f"Time per step: {time_cache / num_steps:.6f}s")
    
    if cache_stats:
        print(f"\nCache Statistics:")
        print(f"  - Cache hit ratio: {cache_stats.get('cache_hit_ratio', 0):.2%}")
        print(f"  - Cache ratio: {cache_stats.get('cache_ratio', 0):.2%}")
        print(f"  - Recompute count: {cache_stats.get('recompute_count', 0)}")
        print(f"  - Cache hit count: {cache_stats.get('cache_hit_count', 0)}")
    
    # Print detailed section timing
    if timing_stats:
        print(f"\nDetailed Section Timing:")
        total_section_time = sum(stats.get("total", 0) for stats in timing_stats.values())
        for section_name in ["cache_lookup", "cache_store", "recompute_linear", "tensor_assembly", "attention"]:
            if section_name in timing_stats:
                stats = timing_stats[section_name]
                total = stats.get("total", 0)
                mean = stats.get("mean", 0)
                count = stats.get("count", 0)
                percentage = (total / time_cache * 100) if time_cache > 0 else 0
                print(f"  - {section_name:20s}: {total:8.4f}s (mean: {mean:.6f}s, count: {count:6d}, {percentage:5.1f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    speedup = time_no_cache / time_cache if time_cache > 0 else 0
    print(f"Without cache:  {time_no_cache:.4f}s ({time_no_cache / num_steps:.6f}s/step)")
    print(f"With cache:     {time_cache:.4f}s ({time_cache / num_steps:.6f}s/step)")
    print(f"Speedup:        {speedup:.2f}x")
    if speedup < 1.0:
        print(f"⚠️  Slowdown:     {1/speedup:.2f}x (cache is slower!)")
    else:
        print(f"✓ Speedup achieved!")
    print("=" * 80)
    
    # Detailed section comparison
    print("\n" + "=" * 80)
    print("Detailed Section Analysis")
    print("=" * 80)
    
    # Print section timing breakdown
    if timing_stats:
        print(f"\nSection Timing Breakdown (WITH cache):")
        print(f"{'Section':<25s} {'Total Time':>12s} {'% of Total':>12s} {'Mean Time':>12s} {'Count':>8s}")
        print("-" * 80)
        
        for section_name in ["cache_lookup", "cache_store", "recompute_linear", "tensor_assembly", "attention"]:
            if section_name in timing_stats:
                stats = timing_stats[section_name]
                total = stats.get("total", 0)
                mean = stats.get("mean", 0)
                count = stats.get("count", 0)
                percentage = (total / time_cache * 100) if time_cache > 0 else 0
                print(f"{section_name:<25s} {total:12.4f}s {percentage:11.1f}% {mean:12.6f}s {count:8d}")
        
        # Find bottlenecks
        print(f"\nBottleneck Analysis:")
        section_times = {k: v.get("total", 0) for k, v in timing_stats.items() if k in ["cache_lookup", "cache_store", "recompute_linear", "tensor_assembly", "attention"]}
        if section_times:
            sorted_sections = sorted(section_times.items(), key=lambda x: x[1], reverse=True)
            for i, (section, time_val) in enumerate(sorted_sections[:3]):
                percentage = (time_val / time_cache * 100) if time_cache > 0 else 0
                print(f"  {i+1}. {section}: {time_val:.4f}s ({percentage:.1f}%)")
    
    # Analyze cache statistics
    if cache_stats:
        recompute_count = cache_stats.get('recompute_count', 0)
        cache_hit_count = cache_stats.get('cache_hit_count', 0)
        total_ops = recompute_count + cache_hit_count
        
        if total_ops > 0:
            recompute_ratio = recompute_count / total_ops
            cache_hit_ratio = cache_hit_count / total_ops
            
            print(f"\nOperation Breakdown:")
            print(f"  Total operations: {total_ops:,}")
            print(f"  - Recompute (F.linear): {recompute_count:,} ({recompute_ratio:.2%})")
            print(f"  - Cache hit (offset):   {cache_hit_count:,} ({cache_hit_ratio:.2%})")
            
            # Estimate time breakdown
            if recompute_count > 0 and cache_hit_count > 0:
                # Rough estimate: recompute takes ~100x longer than cache lookup
                # This is a heuristic based on F.linear vs tensor indexing
                estimated_recompute_time = time_cache * recompute_ratio * 50  # Heuristic
                estimated_cache_time = time_cache * cache_hit_ratio * 0.1  # Heuristic
                overhead_time = time_cache - estimated_recompute_time - estimated_cache_time
                
                print(f"\nEstimated Time Breakdown:")
                print(f"  - Recompute time:     {estimated_recompute_time:.4f}s ({estimated_recompute_time/time_cache:.2%})")
                print(f"  - Cache lookup time:  {estimated_cache_time:.4f}s ({estimated_cache_time/time_cache:.2%})")
                print(f"  - Overhead time:      {overhead_time:.4f}s ({overhead_time/time_cache:.2%})")
                print(f"    (Includes: cache management, tensor ops, attention computation)")
            
            # Performance analysis
            print(f"\nPerformance Analysis:")
            if cache_hit_ratio > 0.9:
                print(f"  ✓ High cache hit ratio ({cache_hit_ratio:.2%}) - cache is working well")
            elif cache_hit_ratio > 0.5:
                print(f"  ⚠️  Moderate cache hit ratio ({cache_hit_ratio:.2%}) - could be improved")
            else:
                print(f"  ✗ Low cache hit ratio ({cache_hit_ratio:.2%}) - cache not effective")
            
            if speedup < 1.0:
                print(f"  ✗ Cache is SLOWER than no cache - overhead too high!")
                print(f"  Recommendations:")
                if recompute_ratio > 0.3:
                    print(f"    - Reduce recompute count (currently {recompute_ratio:.2%})")
                if overhead_time / time_cache > 0.3:
                    print(f"    - Reduce cache overhead (currently {overhead_time/time_cache:.2%})")
            else:
                print(f"  ✓ Speedup achieved: {speedup:.2f}x")
    else:
        print("No cache statistics available for detailed analysis.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
