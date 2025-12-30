"""Training script with diffusion method comparison.

This script trains a model for 1 epoch and compares different diffusion methods
(baseline, with cache, different cache configs) during training.
"""

import logging
import os
from functools import partial
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.utils.callbacks import SamplingCallback, DiffusionMethodComparisonCallback
from fdiff.utils.extraction import dict_to_str, get_training_params
from fdiff.utils.wandb import maybe_initialize_wandb


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

        # Setup callbacks and track comparison callback
        self.comparison_callback = None
        for callback in self.trainer.callbacks:  # type: ignore
            if isinstance(callback, DiffusionMethodComparisonCallback):
                self.comparison_callback = callback
            elif isinstance(callback, SamplingCallback):
                callback.setup_datamodule(datamodule=self.datamodule)

    def train(self) -> None:
        assert not (
            self.score_model.scale_noise and not self.datamodule.fourier_transform
        ), "You cannot use noise scaling without the Fourier transform."
        self.trainer.fit(model=self.score_model, datamodule=self.datamodule)
        
        # Save comparison results if available
        if self.comparison_callback is not None and self.comparison_callback.results_history:
            # Convert results to DataFrame
            all_results = []
            for epoch_result in self.comparison_callback.results_history:
                epoch = epoch_result["epoch"]
                for method_name, method_result in epoch_result["methods"].items():
                    all_results.append({
                        "epoch": epoch,
                        "method": method_name,
                        "time": method_result["time"],
                        "time_per_sample": method_result["time_per_sample"],
                        "time_per_step": method_result["time_per_step"],
                        "num_steps": method_result.get("num_steps", 0),
                        "speedup": method_result.get("speedup", 1.0),
                        "cache_hit_ratio": method_result.get("cache_stats", {}).get("cache_hit_ratio", 0.0),
                        "cache_ratio": method_result.get("cache_stats", {}).get("cache_ratio", 0.0),
                    })
            
            df_results = pd.DataFrame(all_results)
            
            # Save to CSV
            csv_path = self.save_dir / "diffusion_comparison_results.csv"
            df_results.to_csv(csv_path, index=False)
            logging.info(f"Comparison results saved to: {csv_path}")
            
            # Print summary
            print("\n" + "=" * 80)
            print("Diffusion Method Comparison Summary")
            print("=" * 80)
            for epoch_result in self.comparison_callback.results_history:
                print(f"\nEpoch {epoch_result['epoch']}:")
                print("-" * 80)
                baseline_time = None
                for method_name, method_result in epoch_result["methods"].items():
                    if "baseline" in method_name.lower() or baseline_time is None:
                        baseline_time = method_result["time"]
                        break
                
                if baseline_time:
                    print(f"{'Method':<30} {'Time (s)':<12} {'Speedup':<10} {'Cache Hit':<12}")
                    print("-" * 80)
                    for method_name, method_result in epoch_result["methods"].items():
                        speedup = method_result.get("speedup", baseline_time / method_result["time"] if method_result["time"] > 0 else 0.0)
                        cache_hit = method_result.get("cache_stats", {}).get("cache_hit_ratio", 0.0)
                        print(f"{method_name:<30} {method_result['time']:<12.2f} {speedup:<10.2f} {cache_hit:<12.1%}")
            print("=" * 80)


@hydra.main(version_base=None, config_path="conf", config_name="train_diffusion_comparison")
def main(cfg: DictConfig) -> None:  # noqa: ARG001
    """Main function: train model for 1 epoch and compare diffusion methods.
    
    Args:
        cfg: Hydra configuration
    """
    # Force 1 epoch
    cfg.trainer.max_epochs = 1
    print("=" * 80)
    print("Training with 1 epoch for diffusion method comparison")
    print("=" * 80)
    
    # Train the model
    runner = TrainingRunner(cfg)
    runner.train()
    
    print("\n" + "=" * 80)
    print("Training and comparison completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

