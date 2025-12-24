"""Ablation study script for E2-CRF caching components.

This script performs ablation studies to demonstrate the contribution
of different components of E2-CRF caching.
"""

import time
from pathlib import Path
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig

from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.sampling.metrics import compute_metrics
from fdiff.utils.caching import E2CRFCache


def run_ablation(
    score_model: ScoreModule,
    num_samples: int = 20,
    num_diffusion_steps: int = 100,
    config_name: str = "baseline",
    cache_kwargs: Optional[dict] = None,
    use_cache: bool = False,
) -> dict:
    """Run a single ablation configuration.
    
    Args:
        score_model: The score model to use
        num_samples: Number of samples to generate
        num_diffusion_steps: Number of diffusion steps
        config_name: Name of the configuration
        cache_kwargs: Optional cache configuration
        use_cache: Whether to use caching
        
    Returns:
        Dictionary with results
    """
    sampler = DiffusionSampler(
        score_model=score_model,
        sample_batch_size=1,
        use_cache=use_cache,
        cache_kwargs=cache_kwargs,
    )
    
    # Generate samples
    start_time = time.time()
    samples = sampler.sample(
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps
    )
    elapsed_time = time.time() - start_time
    
    # Compute metrics (if ground truth available)
    metrics = {}
    # Note: In practice, you would load ground truth data here
    # metrics = compute_metrics(samples, ground_truth)
    
    # Get cache statistics if available
    cache_stats = {}
    if use_cache and score_model.cache is not None:
        cache_stats = score_model.cache.get_cache_stats()
    
    return {
        "config_name": config_name,
        "elapsed_time": elapsed_time,
        "metrics": metrics,
        "cache_stats": cache_stats,
    }


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    """Main ablation study function.
    
    Args:
        cfg: Hydra configuration
    """
    # Load model
    model_id = cfg.get("model_id", "latest")
    log_dir = Path("lightning_logs")
    
    if model_id == "latest":
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
    
    score_model = ScoreModule.load_from_checkpoint(str(checkpoint_path))
    score_model.eval()
    if torch.cuda.is_available():
        score_model = score_model.cuda()
    
    num_samples = cfg.get("num_samples", 20)
    num_diffusion_steps = cfg.get("num_diffusion_steps", 100)
    
    print("=" * 80)
    print("E2-CRF Caching Ablation Study")
    print("=" * 80)
    
    results = []
    
    # 1. Baseline: No caching
    print("\n1. Baseline (no caching)...")
    result = run_ablation(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        config_name="Baseline",
        use_cache=False,
    )
    results.append(result)
    print(f"   Time: {result['elapsed_time']:.2f}s")
    
    # 2. E2-CRF: Full method
    print("\n2. E2-CRF (full method)...")
    result = run_ablation(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        config_name="E2-CRF (full)",
        use_cache=True,
        cache_kwargs={},
    )
    results.append(result)
    print(f"   Time: {result['elapsed_time']:.2f}s")
    print(f"   Speedup: {results[0]['elapsed_time'] / result['elapsed_time']:.2f}x")
    
    # 3. Ablation: Without event-driven trigger (always cache high-freq)
    print("\n3. Ablation: Without event-driven trigger...")
    result = run_ablation(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        config_name="No event trigger",
        use_cache=True,
        cache_kwargs={
            "tau_warn": 0.0,  # Disable event-driven triggering
        },
    )
    results.append(result)
    print(f"   Time: {result['elapsed_time']:.2f}s")
    print(f"   Speedup: {results[0]['elapsed_time'] / result['elapsed_time']:.2f}x")
    
    # 4. Ablation: Without error-feedback correction
    print("\n4. Ablation: Without error-feedback correction...")
    result = run_ablation(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        config_name="No error feedback",
        use_cache=True,
        cache_kwargs={
            "R": 999999,  # Effectively disable error-feedback
            "tau_warn": 999999,
        },
    )
    results.append(result)
    print(f"   Time: {result['elapsed_time']:.2f}s")
    print(f"   Speedup: {results[0]['elapsed_time'] / result['elapsed_time']:.2f}x")
    
    # 5. Ablation: Without energy-weighted threshold
    print("\n5. Ablation: Without energy-weighted threshold...")
    result = run_ablation(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        config_name="No energy weighting",
        use_cache=True,
        cache_kwargs={
            "tau_0": 0.0,  # Disable energy-weighted threshold
        },
    )
    results.append(result)
    print(f"   Time: {result['elapsed_time']:.2f}s")
    print(f"   Speedup: {results[0]['elapsed_time'] / result['elapsed_time']:.2f}x")
    
    # 6. Ablation: Naive caching (always cache high-freq, no adaptive logic)
    print("\n6. Ablation: Naive caching (always cache high-freq)...")
    result = run_ablation(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        config_name="Naive caching",
        use_cache=True,
        cache_kwargs={
            "K": 5,  # Only recompute first 5 tokens
            "tau_0": 0.0,
            "tau_warn": 0.0,
            "R": 999999,
        },
    )
    results.append(result)
    print(f"   Time: {result['elapsed_time']:.2f}s")
    print(f"   Speedup: {results[0]['elapsed_time'] / result['elapsed_time']:.2f}x")
    
    # Summary table
    print("\n" + "=" * 80)
    print("Ablation Study Results")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Time (s)':<12} {'Speedup':<10} {'Cache Hit Ratio':<15}")
    print("-" * 80)
    
    baseline_time = results[0]["elapsed_time"]
    for r in results:
        speedup = baseline_time / r["elapsed_time"]
        cache_hit = r["cache_stats"].get("cache_hit_ratio", 0.0) if r["cache_stats"] else 0.0
        print(f"{r['config_name']:<30} {r['elapsed_time']:<12.2f} {speedup:<10.2f} {cache_hit:<15.2%}")
    
    print("=" * 80)
    
    # Save results
    output_dir = Path("ablation_results")
    output_dir.mkdir(exist_ok=True)
    
    import json
    results_dict = {
        r["config_name"]: {
            "elapsed_time": r["elapsed_time"],
            "speedup": baseline_time / r["elapsed_time"],
            "cache_stats": r["cache_stats"],
        }
        for r in results
    }
    
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()

