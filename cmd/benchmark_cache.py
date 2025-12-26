"""Benchmarking script for E2-CRF caching acceleration.

This script compares the speedup achieved by E2-CRF caching versus
standard inference, and performs ablation studies with visualization.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

import hydra
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from omegaconf import DictConfig

from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.caching import E2CRFCache

# Fix for PyTorch 2.6+ weights_only loading
from fdiff.schedulers.sde import VPScheduler, VEScheduler
torch.serialization.add_safe_globals([VPScheduler, VEScheduler])

# Disable LaTeX rendering to avoid errors when LaTeX is not installed
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

try:
    plt.style.use("science")
except Exception:
    # Fallback if scienceplots is not available
    plt.style.use("seaborn-v0_8")
    
warnings.filterwarnings("ignore")


def benchmark_sampling(
    score_model: ScoreModule,
    num_samples: int = 10,
    num_diffusion_steps: int = 100,
    use_cache: bool = False,
    cache_kwargs: Optional[dict] = None,
    use_fresca: bool = False,
    fresca_kwargs: Optional[dict] = None,
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
    # Prepare FreSca kwargs
    fresca_kwargs = fresca_kwargs or {}
    if use_fresca:
        fresca_kwargs.setdefault("fresca_low_scale", 1.0)
        fresca_kwargs.setdefault("fresca_high_scale", 1.5)
        fresca_kwargs.setdefault("fresca_cutoff_ratio", 0.5)
        fresca_kwargs.setdefault("fresca_cutoff_strategy", "energy")
    
    sampler = DiffusionSampler(
        score_model=score_model,
        sample_batch_size=1,
        use_cache=use_cache,
        cache_kwargs=cache_kwargs,
        use_fresca=use_fresca,
        **fresca_kwargs,
    )
    
    # Reset cache before benchmarking to get accurate statistics
    if use_cache and score_model.cache is not None:
        score_model.cache.reset()
    
    # Warmup
    _ = sampler.sample(num_samples=1, num_diffusion_steps=10)
    
    # Reset cache again after warmup to get clean statistics
    if use_cache and score_model.cache is not None:
        score_model.cache.reset()
    
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
        "num_samples": num_samples,
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
        use_fresca=False,
    )
    time_cache = results_cache["elapsed_time"]
    cache_stats = results_cache["cache_stats"]
    speedup = time_no_cache / time_cache
    print(f"   Time: {time_cache:.2f}s")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   Time per sample: {time_cache / num_samples:.2f}s")
    print(f"   Time per diffusion step: {time_cache / (num_samples * num_diffusion_steps):.4f}s")
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
            print(f"   - Decomposition ratio: {cache_stats.get('freq_decomp_ratio', 0):.2%}")
            if cache_stats.get('freq_decomp_count', 0) > 0:
                avg_steps_per_decomp = cache_stats.get('current_step', num_diffusion_steps) / cache_stats.get('freq_decomp_count', 1)
                print(f"   - Avg steps per decomposition: {avg_steps_per_decomp:.1f}")
    
    # Benchmark with caching + FreSca
    print("\n2b. Benchmarking WITH E2-CRF caching + FreSca...")
    results_cache_fresca = benchmark_sampling(
        score_model=score_model,
        num_samples=num_samples,
        num_diffusion_steps=num_diffusion_steps,
        use_cache=True,
        cache_kwargs={"use_fresca_in_cache": True},
        use_fresca=True,
        fresca_kwargs={"fresca_high_scale": 1.5, "fresca_cutoff_ratio": 0.5},
    )
    time_cache_fresca = results_cache_fresca["elapsed_time"]
    speedup_fresca = time_no_cache / time_cache_fresca
    cache_fresca_stats = results_cache_fresca.get("cache_stats", {})
    print(f"   Time: {time_cache_fresca:.2f}s")
    print(f"   Speedup: {speedup_fresca:.2f}x")
    print(f"   Improvement over cache-only: {time_cache / time_cache_fresca:.2f}x")
    if cache_fresca_stats:
        print(f"   Cache hit ratio: {cache_fresca_stats.get('cache_hit_ratio', 0):.2%}")
    
    
    # Collect all results for visualization
    all_results = []
    
    # Add baseline
    all_results.append({
        "Config": "No Cache",
        "Parameter": "baseline",
        "Value": None,
        "Time (s)": time_no_cache,
        "Speedup": 1.0,
        "Time per Sample (s)": time_no_cache / num_samples,
        "Time per Step (s)": time_no_cache / (num_samples * num_diffusion_steps),
        "Cache Hit Ratio": 0.0,
        "Cache Ratio": 0.0,
        "Freq Decomp Count": 0,
    })
    
    # Add default cache
    all_results.append({
        "Config": "E2-CRF (default)",
        "Parameter": "default",
        "Value": None,
        "Time (s)": time_cache,
        "Speedup": speedup,
        "Time per Sample (s)": time_cache / num_samples,
        "Time per Step (s)": time_cache / (num_samples * num_diffusion_steps),
        "Cache Hit Ratio": cache_stats.get('cache_hit_ratio', 0) if cache_stats else 0.0,
        "Cache Ratio": cache_stats.get('cache_ratio', 0) if cache_stats else 0.0,
        "Freq Decomp Count": cache_stats.get('freq_decomp_count', 0) if cache_stats else 0,
    })
    
    # Add cache + FreSca
    cache_fresca_stats = results_cache_fresca.get("cache_stats", {})
    all_results.append({
        "Config": "E2-CRF + FreSca",
        "Parameter": "fresca",
        "Value": None,
        "Time (s)": time_cache_fresca,
        "Speedup": speedup_fresca,
        "Time per Sample (s)": time_cache_fresca / num_samples,
        "Time per Step (s)": time_cache_fresca / (num_samples * num_diffusion_steps),
        "Cache Hit Ratio": cache_fresca_stats.get('cache_hit_ratio', 0) if cache_fresca_stats else 0.0,
        "Cache Ratio": cache_fresca_stats.get('cache_ratio', 0) if cache_fresca_stats else 0.0,
        "Freq Decomp Count": cache_fresca_stats.get('freq_decomp_count', 0) if cache_fresca_stats else 0,
    })
    
    # Add ablation results
    ablation_results = []
    
    # K values
    print("\n3. Ablation: Varying K (low-frequency tokens)...")
    for K in [0, 3, 5, 10]:
        print(f"   K={K}: ", end="", flush=True)
        results = benchmark_sampling(
            score_model=score_model,
            num_samples=num_samples,
            num_diffusion_steps=num_diffusion_steps,
            use_cache=True,
            cache_kwargs={"K": K},
            use_fresca=False,
        )
        time_k = results["elapsed_time"]
        speedup_k = time_no_cache / time_k
        stats_k = results.get("cache_stats", {})
        cache_hit = stats_k.get('cache_hit_ratio', 0)
        print(f"Time: {time_k:.2f}s, Speedup: {speedup_k:.2f}x, Hit: {cache_hit:.1%}")
        ablation_results.append({
            "Config": f"K={K}",
            "Parameter": "K",
            "Value": K,
            "Time (s)": time_k,
            "Speedup": speedup_k,
            "Time per Sample (s)": time_k / num_samples,
            "Time per Step (s)": time_k / (num_samples * num_diffusion_steps),
            "Cache Hit Ratio": cache_hit,
            "Cache Ratio": stats_k.get('cache_ratio', 0),
            "Freq Decomp Count": stats_k.get('freq_decomp_count', 0),
        })
    
    # R values
    print("\n4. Ablation: Varying R (error-feedback interval)...")
    for R in [5, 10, 20, 50]:
        print(f"   R={R}: ", end="", flush=True)
        results = benchmark_sampling(
            score_model=score_model,
            num_samples=num_samples,
            num_diffusion_steps=num_diffusion_steps,
            use_cache=True,
            cache_kwargs={"R": R},
            use_fresca=False,
        )
        time_r = results["elapsed_time"]
        speedup_r = time_no_cache / time_r
        stats_r = results.get("cache_stats", {})
        cache_hit = stats_r.get('cache_hit_ratio', 0)
        print(f"Time: {time_r:.2f}s, Speedup: {speedup_r:.2f}x, Hit: {cache_hit:.1%}")
        ablation_results.append({
            "Config": f"R={R}",
            "Parameter": "R",
            "Value": R,
            "Time (s)": time_r,
            "Speedup": speedup_r,
            "Time per Sample (s)": time_r / num_samples,
            "Time per Step (s)": time_r / (num_samples * num_diffusion_steps),
            "Cache Hit Ratio": cache_hit,
            "Cache Ratio": stats_r.get('cache_ratio', 0),
            "Freq Decomp Count": stats_r.get('freq_decomp_count', 0),
        })
    
    # tau_0 values
    print("\n5. Ablation: Varying tau_0 (base threshold)...")
    for tau_0 in [0.05, 0.1, 0.2, 0.5]:
        print(f"   tau_0={tau_0}: ", end="", flush=True)
        results = benchmark_sampling(
            score_model=score_model,
            num_samples=num_samples,
            num_diffusion_steps=num_diffusion_steps,
            use_cache=True,
            cache_kwargs={"tau_0": tau_0},
            use_fresca=False,
        )
        time_tau = results["elapsed_time"]
        speedup_tau = time_no_cache / time_tau
        stats_tau = results.get("cache_stats", {})
        cache_hit = stats_tau.get('cache_hit_ratio', 0)
        print(f"Time: {time_tau:.2f}s, Speedup: {speedup_tau:.2f}x, Hit: {cache_hit:.1%}")
        ablation_results.append({
            "Config": f"tau_0={tau_0}",
            "Parameter": "tau_0",
            "Value": tau_0,
            "Time (s)": time_tau,
            "Speedup": speedup_tau,
            "Time per Sample (s)": time_tau / num_samples,
            "Time per Step (s)": time_tau / (num_samples * num_diffusion_steps),
            "Cache Hit Ratio": cache_hit,
            "Cache Ratio": stats_tau.get('cache_ratio', 0),
            "Freq Decomp Count": stats_tau.get('freq_decomp_count', 0),
        })
    
    # freq_decomp_interval values
    print("\n6. Ablation: Varying freq_decomp_interval (FreqCa frequency)...")
    for interval in [5, 10, 20, 50]:
        print(f"   interval={interval}: ", end="", flush=True)
        results = benchmark_sampling(
            score_model=score_model,
            num_samples=num_samples,
            num_diffusion_steps=num_diffusion_steps,
            use_cache=True,
            cache_kwargs={"freq_decomp_interval": interval},
            use_fresca=False,
        )
        time_interval = results["elapsed_time"]
        speedup_interval = time_no_cache / time_interval
        stats_interval = results.get("cache_stats", {})
        cache_hit = stats_interval.get('cache_hit_ratio', 0)
        freq_decomp = stats_interval.get('freq_decomp_count', 0)
        print(f"Time: {time_interval:.2f}s, Speedup: {speedup_interval:.2f}x, Hit: {cache_hit:.1%}, Decomp: {freq_decomp}")
        ablation_results.append({
            "Config": f"interval={interval}",
            "Parameter": "freq_decomp_interval",
            "Value": interval,
            "Time (s)": time_interval,
            "Speedup": speedup_interval,
            "Time per Sample (s)": time_interval / num_samples,
            "Time per Step (s)": time_interval / (num_samples * num_diffusion_steps),
            "Cache Hit Ratio": cache_hit,
            "Cache Ratio": stats_interval.get('cache_ratio', 0),
            "Freq Decomp Count": freq_decomp,
        })
    
    # Ablation: FreSca high-frequency scaling
    print("\n7. Ablation: Varying FreSca high_scale...")
    for h_scale in [1.0, 1.2, 1.5, 2.0]:
        print(f"   h_scale={h_scale}: ", end="", flush=True)
        results = benchmark_sampling(
            score_model=score_model,
            num_samples=num_samples,
            num_diffusion_steps=num_diffusion_steps,
            use_cache=True,
            cache_kwargs={"use_fresca_in_cache": True},
            use_fresca=True,
            fresca_kwargs={"fresca_high_scale": h_scale, "fresca_cutoff_ratio": 0.5},
        )
        time_fresca = results["elapsed_time"]
        speedup_fresca = time_no_cache / time_fresca
        stats_fresca = results.get("cache_stats", {})
        cache_hit = stats_fresca.get('cache_hit_ratio', 0)
        print(f"Time: {time_fresca:.2f}s, Speedup: {speedup_fresca:.2f}x, Hit: {cache_hit:.1%}")
        ablation_results.append({
            "Config": f"FreSca h={h_scale}",
            "Parameter": "fresca_high_scale",
            "Value": h_scale,
            "Time (s)": time_fresca,
            "Speedup": speedup_fresca,
            "Time per Sample (s)": time_fresca / num_samples,
            "Time per Step (s)": time_fresca / (num_samples * num_diffusion_steps),
            "Cache Hit Ratio": cache_hit,
            "Cache Ratio": stats_fresca.get('cache_ratio', 0),
            "Freq Decomp Count": stats_fresca.get('freq_decomp_count', 0),
        })
    
    all_results.extend(ablation_results)
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save to CSV
    output_dir = Path("outputs") / "cache_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"cache_benchmark_{model_id}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df_results, output_dir, model_id)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Baseline (no cache):     {time_no_cache:.2f}s")
    print(f"                         {time_no_cache / num_samples:.2f}s per sample")
    print(f"                         {time_no_cache / (num_samples * num_diffusion_steps):.4f}s per step")
    print(f"E2-CRF (default):       {time_cache:.2f}s")
    print(f"                         {time_cache / num_samples:.2f}s per sample")
    print(f"                         {time_cache / (num_samples * num_diffusion_steps):.4f}s per step")
    print(f"Overall speedup:         {time_no_cache / time_cache:.2f}x")
    if cache_stats:
        print(f"\nCache Performance:")
        print(f"  Cache hit ratio:      {cache_stats.get('cache_hit_ratio', 0):.2%}")
        print(f"  Cache coverage:       {cache_stats.get('cache_ratio', 0):.2%}")
        if 'freq_decomp_count' in cache_stats:
            print(f"  Freq decompositions:  {cache_stats.get('freq_decomp_count', 0)}")
            print(f"  Decomp efficiency:   {cache_stats.get('freq_decomp_ratio', 0):.2%}")
    print("=" * 80)
    print(f"\nAll results and visualizations saved to {output_dir}")


def create_visualizations(
    df_results: pd.DataFrame,
    output_dir: Path,
    model_id: str,
) -> None:
    """Create visualizations for cache benchmark results.
    
    Args:
        df_results: DataFrame with benchmark results
        output_dir: Output directory for saving figures
        model_id: Model ID for filename
    """
    # Disable LaTeX to avoid errors
    matplotlib.rcParams['text.usetex'] = False
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Speedup comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df_results[df_results["Config"] != "No Cache"].copy()
    df_plot = df_plot.sort_values("Speedup", ascending=False)
    
    colors = ['#2ecc71' if x > 1.0 else '#e74c3c' for x in df_plot["Speedup"]]
    bars = ax.barh(df_plot["Config"], df_plot["Speedup"], color=colors)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='Baseline (1.0x)')
    ax.set_xlabel("Speedup (x)", fontsize=12)
    ax.set_title(f"Cache Performance Comparison - {model_id}", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"speedup_comparison_{model_id}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved speedup comparison to {figures_dir / f'speedup_comparison_{model_id}.pdf'}")
    
    # 2. Time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df_results.sort_values("Time (s)", ascending=True)
    colors = ['#3498db' if x == "No Cache" else '#e74c3c' if x == "E2-CRF (default)" else '#95a5a6' 
              for x in df_plot["Config"]]
    bars = ax.barh(df_plot["Config"], df_plot["Time (s)"], color=colors)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_title(f"Sampling Time Comparison - {model_id}", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f"time_comparison_{model_id}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved time comparison to {figures_dir / f'time_comparison_{model_id}.pdf'}")
    
    # 3. Cache hit ratio vs speedup
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot = df_results[df_results["Config"] != "No Cache"].copy()
    scatter = ax.scatter(
        df_plot["Cache Hit Ratio"],
        df_plot["Speedup"],
        s=100,
        alpha=0.6,
        c=df_plot["Time (s)"],
        cmap='viridis_r',
    )
    ax.set_xlabel("Cache Hit Ratio", fontsize=12)
    ax.set_ylabel("Speedup (x)", fontsize=12)
    ax.set_title(f"Cache Hit Ratio vs Speedup - {model_id}", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Time (s)')
    plt.tight_layout()
    plt.savefig(figures_dir / f"cache_hit_vs_speedup_{model_id}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved cache hit vs speedup to {figures_dir / f'cache_hit_vs_speedup_{model_id}.pdf'}")
    
    # 4. Ablation studies by parameter
    for param in ["K", "R", "tau_0", "freq_decomp_interval"]:
        df_param = df_results[df_results["Parameter"] == param].copy()
        if df_param.empty:
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Speedup by parameter value
        df_param = df_param.sort_values("Value")
        ax1.plot(df_param["Value"], df_param["Speedup"], marker='o', linewidth=2, markersize=8)
        ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel(f"{param} Value", fontsize=12)
        ax1.set_ylabel("Speedup (x)", fontsize=12)
        ax1.set_title(f"Speedup vs {param}", fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Cache hit ratio by parameter value
        ax2.plot(df_param["Value"], df_param["Cache Hit Ratio"], marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel(f"{param} Value", fontsize=12)
        ax2.set_ylabel("Cache Hit Ratio", fontsize=12)
        ax2.set_title(f"Cache Hit Ratio vs {param}", fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"ablation_{param.lower()}_{model_id}.pdf", bbox_inches="tight")
        plt.close()
        print(f"  Saved ablation study for {param} to {figures_dir / f'ablation_{param.lower()}_{model_id}.pdf'}")
    
    # 5. Summary comparison table visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Select key metrics for table
    df_table = df_results[["Config", "Time (s)", "Speedup", "Cache Hit Ratio", "Freq Decomp Count"]].copy()
    df_table["Time (s)"] = df_table["Time (s)"].round(2)
    df_table["Speedup"] = df_table["Speedup"].round(2)
    df_table["Cache Hit Ratio"] = (df_table["Cache Hit Ratio"] * 100).round(1).astype(str) + "%"
    df_table["Freq Decomp Count"] = df_table["Freq Decomp Count"].astype(int)
    
    table = ax.table(
        cellText=df_table.values.tolist(),
        colLabels=df_table.columns.tolist(),
        cellLoc='center',
        loc='center',
        bbox=(0, 0, 1, 1)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code rows
    for i in range(len(df_table)):
        if df_table.iloc[i]["Config"] == "No Cache":
            for j in range(len(df_table.columns)):
                table[(i+1, j)].set_facecolor('#ecf0f1')
        elif df_table.iloc[i]["Config"] == "E2-CRF (default)":
            for j in range(len(df_table.columns)):
                table[(i+1, j)].set_facecolor('#d5f4e6')
        elif df_table.iloc[i]["Speedup"] > 1.0:
            for j in range(len(df_table.columns)):
                table[(i+1, j)].set_facecolor('#e8f8f5')
    
    ax.set_title(f"Cache Benchmark Summary - {model_id}", fontsize=16, fontweight='bold', pad=20)
    plt.savefig(figures_dir / f"summary_table_{model_id}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved summary table to {figures_dir / f'summary_table_{model_id}.pdf'}")


if __name__ == "__main__":
    main()

