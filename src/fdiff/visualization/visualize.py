"""Visualization of generated samples from diffusion models.

This module provides functions to visualize generated samples from both
Time and Frequency domain diffusion models, comparing them with training data.
"""

from pathlib import Path
from typing import Optional, Dict
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

plt.style.use("science")
warnings.filterwarnings("ignore")


def ordering(name: str) -> int:
    """Order function for sorting domains.
    
    Args:
        name: Domain name
        
    Returns:
        Order value
    """
    if name == "train":
        return 0
    elif name == "freq":
        return 1
    elif name == "time":
        return 2
    else:
        return 3


LEGEND_MAPPING = {
    "train": "Training samples",
    "freq": "Generated samples (Frequency domain model)",
    "time": "Generated samples (Time domain model)",
}


def get_train_samples(
    model_id: str,
    runs_dir: Path,
    data_dir: Path,
) -> torch.Tensor:
    """Get training samples for a given model.
    
    Args:
        model_id: Model ID (run directory name)
        runs_dir: Directory containing run directories
        data_dir: Directory containing data
        
    Returns:
        Training samples tensor
    """
    config_path = runs_dir / model_id / "train_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with initialize(version_base=None, config_path=str(runs_dir / model_id)):
        train_config = compose(config_name="train_config")
    
    train_config.datamodule.data_dir = data_dir
    datamodule = instantiate(train_config.datamodule)
    datamodule.prepare_data()
    datamodule.setup("plot")
    train_samples = datamodule.X_train
    return train_samples


def plot_samples(
    samples_dict: Dict[str, torch.Tensor],
    n_samples: int = 5,
    output_path: Optional[Path] = None,
    show_plots: bool = False,
) -> tuple:
    """Plot samples from different domains.
    
    Args:
        samples_dict: Dictionary mapping domain names to sample tensors
        n_samples: Number of samples to plot
        output_path: Optional path to save figure
        show_plots: Whether to display plots
        
    Returns:
        Tuple of (fig, ax) matplotlib objects
    """
    n_columns = len(samples_dict.keys())
    # Plot samples
    fig, ax = plt.subplots(
        n_samples, n_columns, figsize=(5 * n_columns, 10 * n_samples)
    )
    
    # Handle case where n_samples == 1
    if n_samples == 1:
        ax = ax.reshape(1, -1)
    
    for k in range(n_samples):
        for i, (domain, samples) in enumerate(
            sorted(samples_dict.items(), key=lambda x: ordering(x[0]))
        ):
            # Ensure we don't exceed available samples
            sample_idx = min(k, samples.shape[0] - 1)
            sample = samples[sample_idx]
            
            # Handle different tensor shapes
            if sample.dim() == 3:
                # (batch, time, channels) -> take first batch
                sample = sample[0]
            
            for j in range(sample.shape[-1]):
                ax[k, i].plot(sample[:, j], label=f"Feature {j}")
            
            ax[k, i].legend(fontsize=15)
            if k == 0:
                ax[k, i].set_title(LEGEND_MAPPING.get(domain, domain), fontsize=25)
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def heatmap_samples(
    samples_dict: Dict[str, torch.Tensor],
    n_samples: int = 5,
    output_path: Optional[Path] = None,
    show_plots: bool = False,
) -> tuple:
    """Plot samples as heatmaps.
    
    Args:
        samples_dict: Dictionary mapping domain names to sample tensors
        n_samples: Number of samples to plot
        output_path: Optional path to save figure
        show_plots: Whether to display plots
        
    Returns:
        Tuple of (fig, ax) matplotlib objects
    """
    n_columns = len(samples_dict.keys())
    # Plot samples
    fig, ax = plt.subplots(
        n_samples, n_columns, figsize=(5 * n_columns, 10 * n_samples)
    )
    
    # Handle case where n_samples == 1
    if n_samples == 1:
        ax = ax.reshape(1, -1)
    
    for k in range(n_samples):
        for i, (domain, samples) in enumerate(
            sorted(samples_dict.items(), key=lambda x: ordering(x[0]))
        ):
            # Ensure we don't exceed available samples
            sample_idx = min(k, samples.shape[0] - 1)
            sample = samples[sample_idx]
            
            # Handle different tensor shapes
            if sample.dim() == 3:
                # (batch, time, channels) -> take first batch
                sample = sample[0]
            
            max_val = torch.abs(sample).max(dim=0)[0].max(dim=0)[0]
            min_val = torch.abs(sample).min(dim=0)[0].min(dim=0)[0]
            
            # Transpose for heatmap: (time, channels)
            sns.heatmap(
                sample.transpose(1, 0).cpu().numpy(),
                cmap="RdBu_r",
                vmin=min_val.item(),
                vmax=max_val.item(),
                ax=ax[k, i],
            )
            
            if k == 0:
                ax[k, i].set_title(LEGEND_MAPPING.get(domain, domain), fontsize=25)
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


def load_samples(
    model_ids: Dict[str, str],
    runs_dir: Path,
    include_train: bool = True,
    data_dir: Optional[Path] = None,
    random_seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Load samples from different model runs.
    
    Args:
        model_ids: Dictionary mapping domain names to model IDs
        runs_dir: Directory containing run directories
        include_train: Whether to include training samples
        data_dir: Optional data directory (required if include_train=True)
        random_seed: Random seed for shuffling
        
    Returns:
        Dictionary mapping domain names to sample tensors
    """
    samples_dict = {}
    rand_gen = torch.Generator()
    rand_gen.manual_seed(random_seed)
    
    # Load generated samples
    for domain, model_id in model_ids.items():
        samples_path = runs_dir / model_id / "samples.pt"
        if not samples_path.exists():
            print(f"Warning: Samples not found for {domain} at {samples_path}")
            continue
        
        samples = torch.load(samples_path)
        # Shuffle samples
        perm = torch.randperm(samples.shape[0], generator=rand_gen)
        samples = samples[perm]
        samples_dict[domain] = samples
    
    # Load training samples if requested
    if include_train and data_dir is not None:
        # Use first model_id to get training samples
        first_model_id = list(model_ids.values())[0]
        try:
            train_samples = get_train_samples(first_model_id, runs_dir, data_dir)
            # Shuffle training samples
            perm = torch.randperm(train_samples.shape[0], generator=rand_gen)
            train_samples = train_samples[perm]
            samples_dict["train"] = train_samples
        except Exception as e:
            print(f"Warning: Failed to load training samples: {e}")
    
    return samples_dict


def visualize_samples(
    model_ids: Dict[str, str],
    runs_dir: Path,
    output_dir: Path,
    dataset_name: Optional[str] = None,
    n_samples: int = 5,
    include_train: bool = True,
    data_dir: Optional[Path] = None,
    plot_type: str = "line",  # "line" or "heatmap"
    show_plots: bool = False,
    random_seed: int = 0,
) -> None:
    """Main function to visualize samples from different models.
    
    Args:
        model_ids: Dictionary mapping domain names ("freq", "time") to model IDs
        runs_dir: Directory containing run directories
        output_dir: Output directory for saving figures
        dataset_name: Optional dataset name for filename
        n_samples: Number of samples to plot
        include_train: Whether to include training samples
        data_dir: Optional data directory (required if include_train=True)
        plot_type: Type of plot ("line" or "heatmap")
        show_plots: Whether to display plots
        random_seed: Random seed for shuffling
    """
    # Load samples
    print("Loading samples...")
    samples_dict = load_samples(
        model_ids, runs_dir, include_train, data_dir, random_seed
    )
    
    if not samples_dict:
        raise ValueError("No samples loaded")
    
    # Determine filename
    if dataset_name is None:
        # Try to infer from first model_id
        first_model_id = list(model_ids.values())[0]
        dataset_name = first_model_id
    
    filename = f"{dataset_name}_samples_{plot_type}.pdf"
    output_path = output_dir / "figures" / filename
    
    # Plot samples
    print(f"Plotting {plot_type} plots...")
    if plot_type == "line":
        plot_samples(samples_dict, n_samples, output_path, show_plots)
    elif plot_type == "heatmap":
        heatmap_samples(samples_dict, n_samples, output_path, show_plots)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'line' or 'heatmap'")
    
    print(f"Visualization complete. Results saved to {output_dir}")


def main(
    model_ids: Dict[str, str],
    runs_dir: Path,
    output_dir: Path,
    data_dir: Optional[Path] = None,
    n_samples: int = 5,
    show_plots: bool = False,
) -> None:
    """Main function for command-line usage.
    
    Args:
        model_ids: Dictionary mapping domain names to model IDs
        runs_dir: Directory containing run directories
        output_dir: Output directory for saving figures
        data_dir: Optional data directory for training samples
        n_samples: Number of samples to plot
        show_plots: Whether to display plots
    """
    # Create both line and heatmap plots
    for plot_type in ["line", "heatmap"]:
        visualize_samples(
            model_ids=model_ids,
            runs_dir=runs_dir,
            output_dir=output_dir,
            n_samples=n_samples,
            include_train=(data_dir is not None),
            data_dir=data_dir,
            plot_type=plot_type,
            show_plots=show_plots,
        )


if __name__ == "__main__":
    # Example usage
    model_ids = {"freq": "20d9c1kc", "time": "tip2g8eh"}
    runs_dir = Path("lightning_logs")
    output_dir = Path("outputs")
    data_dir = Path("data")
    
    main(model_ids, runs_dir, output_dir, data_dir, n_samples=5, show_plots=False)

