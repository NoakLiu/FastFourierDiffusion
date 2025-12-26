"""Results analysis and visualization for LSTM-based diffusion models.

This module provides functions to analyze and visualize sample quality metrics
for LSTM-based diffusion models in both Time and Frequency domains.
"""

from pathlib import Path
from typing import Any, Optional
from itertools import product
import warnings

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import scienceplots
from yaml import safe_load
from einops import rearrange

plt.style.use("science")
warnings.filterwarnings("ignore")


def infer_dataset(config: dict[str, Any]) -> str:
    """Infer dataset name from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dataset name string
    """
    datamodule = config["datamodule"]["_target_"]
    if "ECG" in datamodule:
        return "ECG"
    if "MIMICIII" in datamodule:
        return "MIMIC-III"
    if "NASDAQ" in datamodule:
        return "NASDAQ-2019"
    if "Droughts" in datamodule:
        return "US-Droughts"
    if "NASA" in datamodule:
        if config["datamodule"].get("subdataset") == "charge":
            return "NASA-Charge"
        else:
            return "NASA-Discharge"
    return "Unknown"


def infer_diffusion_domain(config: dict[str, Any]) -> str:
    """Infer diffusion domain (Time or Frequency) from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        "Time" or "Frequency"
    """
    fourier_transform = config.get("fourier_transform", False)
    if fourier_transform:
        return "Frequency"
    else:
        return "Time"


def calculate_metrics(results: dict) -> list[dict]:
    """Calculate metrics from results dictionary.
    
    Args:
        results: Results dictionary containing Wasserstein distances
        
    Returns:
        List of metric dictionaries
    """
    data = []
    for domain, method in product({"time", "freq"}, {"sliced", "marginal"}):
        key = f"{domain}_{method}_wasserstein_all"
        if key in results:
            all_distances = results[key]
            data.extend(
                [
                    {
                        "Value": distance,
                        "Metric Domain": "Frequency" if domain == "freq" else "Time",
                        "Metric": (
                            "Sliced Wasserstein"
                            if method == "sliced"
                            else "Marginal Wasserstein"
                        ),
                    }
                    for distance in all_distances
                ]
            )
    return data


def calculate_baselines(results: dict) -> list[dict]:
    """Calculate baseline metrics from results dictionary.
    
    Args:
        results: Results dictionary containing baseline Wasserstein distances
        
    Returns:
        List of baseline metric dictionaries
    """
    data = []
    for baseline, domain, method in product(
        {"dummy", "self"}, {"time", "freq"}, {"sliced", "marginal"}
    ):
        key = f"{domain}_{method}_wasserstein_mean_{baseline}"
        if key in results:
            distance = results[key]
            data.append(
                {
                    "Value": distance,
                    "Baseline": "Mean" if baseline == "dummy" else "Half Train",
                    "Metric Domain": "Frequency" if domain == "freq" else "Time",
                    "Metric": (
                        "Sliced Wasserstein"
                        if method == "sliced"
                        else "Marginal Wasserstein"
                    ),
                }
            )
    return data


def infer_tensor_shapes(sample_path: Path) -> tuple[int, int]:
    """Infer tensor shapes from sample file.
    
    Args:
        sample_path: Path to samples.pt file
        
    Returns:
        Tuple of (sequence_length, n_channels)
    """
    samples = torch.load(sample_path)
    return samples.shape[-2:]


def calculate_spectral_density(
    marginal_spectral: list[float], sample_path: Path
) -> torch.Tensor:
    """Calculate spectral density from marginal spectral Wasserstein distances.
    
    Args:
        marginal_spectral: List of marginal spectral Wasserstein distances
        sample_path: Path to samples.pt file
        
    Returns:
        Tensor of spectral density values
    """
    _, n_channels = infer_tensor_shapes(sample_path)
    marginal_spectral = torch.tensor(marginal_spectral)
    marginal_spectral = rearrange(
        marginal_spectral, "(freq channels) -> freq channels", channels=n_channels
    )
    return marginal_spectral.mean(dim=1)


def process_results(
    run_ids: list[str],
    runs_dir: Path,
    output_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process results from multiple LSTM runs.
    
    Args:
        run_ids: List of run IDs to process
        runs_dir: Directory containing run directories
        output_dir: Optional output directory for saving CSV files
        
    Returns:
        Tuple of (metrics_df, baselines_df)
    """
    df_list = []
    baselines_list = []
    
    for run_id in run_ids:
        run_path = runs_dir / run_id
        config_path = run_path / "train_config.yaml"
        results_path = run_path / "results.yaml"
        
        if not config_path.exists() or not results_path.exists():
            print(f"Warning: Skipping {run_id} - config or results file not found")
            continue
            
        with open(config_path, "r") as f:
            config = safe_load(f)
            dataset = infer_dataset(config)
            domain = infer_diffusion_domain(config)
            
        with open(results_path, "r") as f:
            results = safe_load(f)
            df = pd.DataFrame(calculate_metrics(results))
            df_baselines = pd.DataFrame(calculate_baselines(results))
            df["Dataset"] = dataset
            df["Diffusion Domain"] = domain
            df["Model"] = "LSTM"  # Mark as LSTM model
            df_baselines["Dataset"] = dataset
            df_baselines["Diffusion Domain"] = domain
            df_baselines["Model"] = "LSTM"
            df_list.append(df)
            baselines_list.append(df_baselines)
    
    if not df_list:
        raise ValueError("No valid runs found")
        
    df = pd.concat(df_list, ignore_index=True)
    df_baselines = pd.concat(baselines_list, ignore_index=True)
    
    # Save to CSV if output directory is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "metrics_lstm.csv", index=False)
        df_baselines.to_csv(output_dir / "baselines_lstm.csv", index=False)
        print(f"Saved metrics to {output_dir / 'metrics_lstm.csv'}")
        print(f"Saved baselines to {output_dir / 'baselines_lstm.csv'}")
    
    return df, df_baselines


def plot_sample_quality(
    df: pd.DataFrame,
    df_baselines: pd.DataFrame,
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
) -> None:
    """Plot sample quality metrics for LSTM models.
    
    Args:
        df: Metrics dataframe
        df_baselines: Baselines dataframe
        output_dir: Optional output directory for saving figures
        show_plots: Whether to display plots
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    with plt.style.context("science"):
        for metric in df.Metric.unique():
            for dataset in df.Dataset.unique():
                df_sub = df[(df.Metric == metric) & (df.Dataset == dataset)]
                if df_sub.empty:
                    continue
                    
                df_sub["Baseline"] = df_sub["Diffusion Domain"].apply(
                    lambda x: "Time Diff." if x == "Time" else "Frequ. Diff."
                )
                ax = sns.boxplot(
                    data=df_sub,
                    x="Metric Domain",
                    y="Value",
                    hue="Baseline",
                    hue_order=["Time Diff.", "Frequ. Diff."],
                    showfliers=False,
                )
                sns.pointplot(
                    data=df_baselines[
                        (df_baselines.Metric == metric) & (df_baselines.Dataset == dataset)
                    ],
                    y="Value",
                    x="Metric Domain",
                    hue="Baseline",
                    hue_order=["Mean", "Half Train"],
                    palette=["#FF6A74", "#70ff70"],
                    ax=ax,
                )
                plt.ylabel(f"{metric} ($\downarrow$)")
                plt.title(f"LSTM Model - {dataset}")
                plt.legend(fontsize=6, title="Baseline", frameon=True, title_fontsize=7)
                
                if output_dir is not None:
                    filename = f"lstm_{metric.lower().replace(' ', '_')}_{dataset.lower().replace('-', '_')}.pdf"
                    plt.savefig(output_dir / filename, bbox_inches="tight")
                    print(f"Saved figure to {output_dir / filename}")
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()


def process_spectral_analysis(
    run_ids: list[str],
    runs_dir: Path,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Process spectral density analysis for LSTM models.
    
    Args:
        run_ids: List of run IDs to process
        runs_dir: Directory containing run directories
        output_dir: Optional output directory for saving CSV files
        
    Returns:
        DataFrame with spectral density data
    """
    spectral_data = []
    
    for run_id in run_ids:
        run_path = runs_dir / run_id
        config_path = run_path / "train_config.yaml"
        results_path = run_path / "results.yaml"
        samples_path = run_path / "samples.pt"
        
        if not all(p.exists() for p in [config_path, results_path, samples_path]):
            print(f"Warning: Skipping {run_id} - required files not found")
            continue
            
        with open(config_path, "r") as f:
            config = safe_load(f)
            dataset_name = infer_dataset(config)
            diffusion_domain = infer_diffusion_domain(config)
            
        with open(results_path, "r") as f:
            results = safe_load(f)
            
        if "spectral_marginal_wasserstein_all" not in results:
            print(f"Warning: Skipping {run_id} - spectral data not found")
            continue
            
        spectral_density_tensor = calculate_spectral_density(
            results["spectral_marginal_wasserstein_all"],
            sample_path=samples_path,
        )
        freqs = torch.arange(0, 1, 1 / spectral_density_tensor.shape[0])
        spectral_density = spectral_density_tensor
        spectral_data.extend(
            [
                {
                    "Dataset": dataset_name,
                    "Diffusion Domain": diffusion_domain,
                    "Model": "LSTM",
                    "Frequency": freqs[k].item(),
                    "Spectral Density": spectral_density[k].item(),
                }
                for k in range(spectral_density.shape[0])
            ]
        )
    
    if not spectral_data:
        raise ValueError("No spectral data found")
        
    spectral_df = pd.DataFrame(spectral_data)
    
    # Save to CSV if output directory is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        spectral_df.to_csv(output_dir / "spectral_density_lstm.csv", index=False)
        print(f"Saved spectral density to {output_dir / 'spectral_density_lstm.csv'}")
    
    return spectral_df


def plot_spectral_density(
    spectral_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
) -> None:
    """Plot spectral density analysis for LSTM models.
    
    Args:
        spectral_df: DataFrame with spectral density data
        output_dir: Optional output directory for saving figures
        show_plots: Whether to display plots
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in spectral_df.Dataset.unique():
        ax = sns.lineplot(
            data=spectral_df[spectral_df.Dataset == dataset],
            x="Frequency",
            y="Spectral Density",
            hue="Diffusion Domain",
            hue_order=["Time", "Frequency"],
        )
        ax.set_yscale("log")
        plt.ylabel(
            r"Wasserstein Distance on $\|\tilde{\mathbf{x}}_\kappa \|^2 \ (\downarrow)$"
        )
        plt.xlabel(r"Normalized Frequency $\omega_\kappa  /  \omega_{\mathrm{Nyq}}$")
        plt.title(f"LSTM Model - {dataset}")
        
        if output_dir is not None:
            filename = f"lstm_spectral_density_{dataset.lower().replace('-', '_')}.pdf"
            plt.savefig(output_dir / filename, bbox_inches="tight")
            print(f"Saved figure to {output_dir / filename}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def create_summary_table(
    df: pd.DataFrame,
    metric_name: str = "Sliced Wasserstein",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Create summary table for a specific metric.
    
    Args:
        df: Metrics dataframe
        metric_name: Name of metric to summarize
        output_dir: Optional output directory for saving CSV files
        
    Returns:
        Pivoted dataframe with summary statistics
    """
    df_sub = df[df["Metric"] == metric_name]
    if df_sub.empty:
        raise ValueError(f"No data found for metric: {metric_name}")
    
    df_pivot = pd.pivot_table(
        df_sub,
        index=["Dataset", "Metric Domain"],
        columns="Diffusion Domain",
        values="Value",
        aggfunc=["mean", "std"],  # Use std instead of sem for pivot_table
    )
    # Calculate sem manually
    df_pivot_sem = pd.pivot_table(
        df_sub,
        index=["Dataset", "Metric Domain"],
        columns="Diffusion Domain",
        values="Value",
        aggfunc=lambda x: x.std() / (len(x) ** 0.5),  # sem = std / sqrt(n)
    )
    # Combine mean and sem
    df_pivot = pd.concat([df_pivot["mean"], df_pivot_sem], keys=["mean", "sem"], axis=1)
    df_pivot = round(df_pivot, 3)
    
    # Create formatted string for LaTeX
    df_pivot_formatted = (
        "$"
        + df_pivot["mean"].astype(str)
        + r" \ \pm \ "
        + (2 * df_pivot["sem"]).astype(str)
        + "$"
    )
    
    # Save to CSV (unformatted version)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = f"lstm_{metric_name.lower().replace(' ', '_')}_summary.csv"
        df_pivot.to_csv(output_dir / csv_filename)
        print(f"Saved summary table to {output_dir / csv_filename}")
        
        # Also save LaTeX version
        tex_filename = f"lstm_{metric_name.lower().replace(' ', '_')}.tex"
        df_pivot_formatted.to_latex(output_dir / tex_filename)
        print(f"Saved LaTeX table to {output_dir / tex_filename}")
    
    return df_pivot


def main(
    run_ids: list[str],
    runs_dir: Path,
    output_dir: Path,
    show_plots: bool = False,
) -> None:
    """Main function to process and visualize LSTM results.
    
    Args:
        run_ids: List of run IDs to process
        runs_dir: Directory containing run directories
        output_dir: Output directory for saving figures and CSV files
        show_plots: Whether to display plots
    """
    # Process metrics
    print("Processing LSTM metrics...")
    df, df_baselines = process_results(run_ids, runs_dir, output_dir)
    
    # Plot sample quality
    print("Plotting LSTM sample quality...")
    plot_sample_quality(df, df_baselines, output_dir / "figures", show_plots)
    
    # Create summary tables
    print("Creating LSTM summary tables...")
    for metric in df.Metric.unique():
        create_summary_table(df, metric, output_dir / "tables")
    
    # Process spectral analysis
    print("Processing LSTM spectral analysis...")
    try:
        spectral_df = process_spectral_analysis(run_ids, runs_dir, output_dir)
        plot_spectral_density(spectral_df, output_dir / "figures", show_plots)
    except ValueError as e:
        print(f"Warning: Spectral analysis failed: {e}")
    
    print(f"\nAll LSTM results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage for LSTM models
    run_list = [
        "zdziznic",
        "va4krqr6",
        "bf3lrfx9",
        "emk7nyz3",
        "2adxm708",
        "zj9secov",
        "np2b68ru",
        "eho1sajh",
        "r9nol7jo",
        "qr1ukbne",
    ]
    
    runs_dir = Path("lightning_logs")
    output_dir = Path("outputs")
    
    main(run_list, runs_dir, output_dir, show_plots=False)

