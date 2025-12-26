"""Spectral interpretation and localization analysis for time series datasets.

This module provides functions to analyze spectral properties and localization
metrics for different datasets, supporting both Time and Frequency domain analysis.
"""

from pathlib import Path
from typing import Optional
import math
import warnings

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

from fdiff.dataloaders.datamodules import (
    ECGDatamodule,
    NASDAQDatamodule,
    NASADatamodule,
    MIMICIIIDatamodule,
    USDroughtsDatamodule,
    Datamodule,
)
from fdiff.utils.fourier import spectral_density, localization_metrics

plt.style.use("science")
warnings.filterwarnings("ignore")

EPS = 1e-15


def process_dataset(
    dataset_name: str,
    datamodule: Datamodule,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process a single dataset to extract spectral and localization metrics.
    
    Args:
        dataset_name: Name of the dataset
        datamodule: Datamodule instance for the dataset
        
    Returns:
        Tuple of (spectral_df, temporal_df, localization_df, localization_joint_df)
    """
    # Prepare data
    datamodule.prepare_data()
    datamodule.setup()
    
    # Extract training features
    X_train = datamodule.X_train
    
    # Compute spectral representation
    X_spec = spectral_density(X_train)
    
    # Compute mean and standard error of the normalized spectral density
    X_spec_norm_mean = torch.mean(
        (
            X_spec.sum(dim=2, keepdim=True)
            / (EPS + X_spec.sum(dim=(1, 2), keepdim=True))
        ),
        dim=(0, 2),
    )
    X_spec_norm_se = torch.std(
        (X_spec.sum(dim=2, keepdim=True) / X_spec.sum(dim=(1, 2), keepdim=True)),
        dim=(0, 2),
    ) / math.sqrt(len(X_spec))
    
    # Compute normalized frequency
    freq_norm = [k / (X_spec.shape[1] - 1) for k in range(X_spec.shape[1])]
    
    # Record the spectral data
    spectral_data = [
        {
            "Dataset": dataset_name,
            "Normalized Frequency": freq_norm[k],
            "Normalized Spectral Density": X_spec_norm_mean[k].item(),
            "SE": X_spec_norm_se[k].item(),
        }
        for k in range(len(freq_norm))
    ]
    
    # Compute the mean and standard error of the normalized energy
    X_energy_mean = torch.mean(
        (X_train**2).sum(dim=2, keepdim=True)
        / (EPS + (X_train**2).sum(dim=(1, 2), keepdim=True)),
        dim=(0, 2),
    )
    X_energy_std = torch.std(
        (X_train**2).sum(dim=2, keepdim=True)
        / (X_train**2).sum(dim=(1, 2), keepdim=True),
        dim=(0, 2),
    )
    
    # Compute the normalized time
    time_norm = [k / (X_train.shape[1] - 1) for k in range(X_train.shape[1])]
    
    # Record the temporal data
    temporal_data = [
        {
            "Dataset": dataset_name,
            "Normalized Time": time_norm[k],
            "Normalized Energy": X_energy_mean[k].item(),
            "SE": X_energy_std[k].item(),
        }
        for k in range(len(time_norm))
    ]
    
    # Compute localization metrics
    X_loc, X_spec_loc = localization_metrics(X_train)
    
    # Record the localization data
    localization_data = [
        {
            "Dataset": dataset_name,
            "Delocalization": X_loc[b].item(),
            "Domain": "Time",
        }
        for b in range(len(X_loc))
    ]
    localization_data.extend(
        [
            {
                "Dataset": dataset_name,
                "Delocalization": X_spec_loc[b].item(),
                "Domain": "Frequency",
            }
            for b in range(len(X_spec_loc))
        ]
    )
    
    # Record the joint localization data
    localization_data_joint = [
        {
            "Dataset": dataset_name,
            "Delocalization Time": X_loc[b].item(),
            "Delocalization Frequency": X_spec_loc[b].item(),
        }
        for b in range(len(X_loc))
    ]
    
    spectral_df = pd.DataFrame(spectral_data)
    temporal_df = pd.DataFrame(temporal_data)
    localization_df = pd.DataFrame(localization_data)
    localization_joint_df = pd.DataFrame(localization_data_joint)
    
    return spectral_df, temporal_df, localization_df, localization_joint_df


def process_all_datasets(
    data_path: Path,
    output_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process all datasets to extract spectral and localization metrics.
    
    Args:
        data_path: Path to data directory
        output_dir: Optional output directory for saving CSV files
        
    Returns:
        Tuple of (spectral_df, temporal_df, localization_df, localization_joint_df)
    """
    datasets: dict[str, Datamodule] = {
        "ECG": ECGDatamodule(data_dir=data_path),
        "MIMIC-III": MIMICIIIDatamodule(data_dir=data_path, n_feats=40),
        "NASDAQ-2019": NASDAQDatamodule(data_dir=data_path),
        "NASA-Charge": NASADatamodule(data_dir=data_path),
        "NASA-Discharge": NASADatamodule(data_dir=data_path, subdataset="discharge"),
        "US-Droughts": USDroughtsDatamodule(data_dir=data_path),
    }
    
    spectral_data: list[dict] = []
    temporal_data: list[dict] = []
    localization_data: list[dict] = []
    localization_data_joint: list[dict] = []
    
    for dataset_name, datamodule in datasets.items():
        print(f"Processing {dataset_name}...")
        try:
            spec_df, temp_df, loc_df, loc_joint_df = process_dataset(
                dataset_name, datamodule
            )
            spectral_data.extend(spec_df.to_dict("records"))
            temporal_data.extend(temp_df.to_dict("records"))
            localization_data.extend(loc_df.to_dict("records"))
            localization_data_joint.extend(loc_joint_df.to_dict("records"))
        except Exception as e:
            print(f"Warning: Failed to process {dataset_name}: {e}")
            continue
    
    spectral_df = pd.DataFrame(spectral_data)
    temporal_df = pd.DataFrame(temporal_data)
    localization_df = pd.DataFrame(localization_data)
    localization_joint_df = pd.DataFrame(localization_data_joint)
    
    # Save to CSV if output directory is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        spectral_df.to_csv(output_dir / "spectral_density_datasets.csv", index=False)
        temporal_df.to_csv(output_dir / "temporal_energy_datasets.csv", index=False)
        localization_df.to_csv(output_dir / "localization_datasets.csv", index=False)
        localization_joint_df.to_csv(
            output_dir / "localization_joint_datasets.csv", index=False
        )
        print(f"Saved all datasets to {output_dir}")
    
    return spectral_df, temporal_df, localization_df, localization_joint_df


def plot_spectral_density(
    spectral_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
) -> None:
    """Plot spectral density across datasets.
    
    Args:
        spectral_df: DataFrame with spectral density data
        output_dir: Optional output directory for saving figures
        show_plots: Whether to display plots
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    ax = sns.lineplot(
        data=spectral_df,
        x="Normalized Frequency",
        y="Normalized Spectral Density",
        hue="Dataset",
    )
    plt.ylabel(
        r"Spectral Density $\|\tilde{\mathbf{x}}_\kappa\|^2_2  /  \| \tilde{\mathbf{x}} \|^2$"
    )
    plt.hlines(0, 0, 1, colors="black", linestyles="dashed")
    plt.xlabel(r" Frequency $\omega_\kappa  /  \omega_{\mathrm{Nyq}}$")
    plt.yscale("log")
    plt.ylim(top=1, bottom=1e-6)
    plt.legend(fontsize=7, title="Dataset", title_fontsize=8)
    
    if output_dir is not None:
        filename = "spectral_density_datasets.pdf"
        plt.savefig(output_dir / filename, bbox_inches="tight")
        print(f"Saved figure to {output_dir / filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_temporal_energy(
    temporal_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
) -> None:
    """Plot temporal energy across datasets.
    
    Args:
        temporal_df: DataFrame with temporal energy data
        output_dir: Optional output directory for saving figures
        show_plots: Whether to display plots
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.lineplot(
        data=temporal_df, x="Normalized Time", y="Normalized Energy", hue="Dataset"
    )
    plt.ylabel(r"Energy Density $\|\mathbf{x}_\tau \|^2_2 / \| \mathbf{x} \|^2$")
    plt.xlabel(r"Time $\tau / N$")
    plt.yscale("log")
    plt.ylim(top=1, bottom=1e-6)
    plt.legend(fontsize=7, title="Dataset", title_fontsize=8, loc="lower center")
    
    if output_dir is not None:
        filename = "temporal_energy_datasets.pdf"
        plt.savefig(output_dir / filename, bbox_inches="tight")
        print(f"Saved figure to {output_dir / filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_localization(
    localization_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
) -> None:
    """Plot localization metrics across datasets.
    
    Args:
        localization_df: DataFrame with localization data
        output_dir: Optional output directory for saving figures
        show_plots: Whether to display plots
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    ax = sns.barplot(
        data=localization_df, x="Dataset", y="Delocalization", hue="Domain"
    )
    ax.set_yscale("log")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.ylabel("Delocalization Metric")
    plt.legend(loc="upper left", fontsize=7, title="Domain")
    
    if output_dir is not None:
        filename = "localization_datasets.pdf"
        plt.savefig(output_dir / filename, bbox_inches="tight")
        print(f"Saved figure to {output_dir / filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_localization_joint(
    localization_joint_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
) -> None:
    """Plot joint localization metrics across datasets.
    
    Args:
        localization_joint_df: DataFrame with joint localization data
        output_dir: Optional output directory for saving figures
        show_plots: Whether to display plots
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    ax = sns.scatterplot(
        data=localization_joint_df,
        x="Delocalization Time",
        y="Delocalization Frequency",
        hue="Dataset",
        alpha=0.3,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.axline((1, 1), (10, 10), color="black", linestyle=":")
    plt.legend(loc="lower left", fontsize=7, title="Dataset")
    
    if output_dir is not None:
        filename = "localization_joint_datasets.png"
        plt.savefig(output_dir / filename, bbox_inches="tight")
        print(f"Saved figure to {output_dir / filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def main(
    data_path: Path,
    output_dir: Path,
    show_plots: bool = False,
) -> None:
    """Main function to process and visualize spectral interpretation.
    
    Args:
        data_path: Path to data directory
        output_dir: Output directory for saving figures and CSV files
        show_plots: Whether to display plots
    """
    # Process all datasets
    print("Processing all datasets...")
    (
        spectral_df,
        temporal_df,
        localization_df,
        localization_joint_df,
    ) = process_all_datasets(data_path, output_dir)
    
    # Plot spectral density
    print("Plotting spectral density...")
    plot_spectral_density(spectral_df, output_dir / "figures", show_plots)
    
    # Plot temporal energy
    print("Plotting temporal energy...")
    plot_temporal_energy(temporal_df, output_dir / "figures", show_plots)
    
    # Plot localization
    print("Plotting localization metrics...")
    plot_localization(localization_df, output_dir / "figures", show_plots)
    
    # Plot joint localization
    print("Plotting joint localization...")
    plot_localization_joint(
        localization_joint_df, output_dir / "figures", show_plots
    )
    
    print(f"\nAll spectral interpretation results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_path = Path("data")
    output_dir = Path("outputs")
    
    main(data_path, output_dir, show_plots=False)

