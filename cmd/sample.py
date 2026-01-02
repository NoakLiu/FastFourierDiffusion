import logging
from pathlib import Path

import hydra
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fdiff.dataloaders.datamodules import Datamodule
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.metrics import MetricCollection
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.extraction import dict_to_str, get_best_checkpoint, get_model_type
from fdiff.utils.fourier import idft


class SamplingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        # Initialize torch
        self.random_seed: int = cfg.random_seed
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Read out the config
        logging.info(
            f"Welcome in the sampling script! You are using the following config:\n{dict_to_str(cfg)}"
        )

        # Get model path and id
        self.model_path = Path(cfg.model_path)
        self.model_id = cfg.model_id

        # Save sampling config to model directory
        self.save_dir = self.model_path / self.model_id
        
        # Check if model directory exists
        if not self.save_dir.exists():
            available_models = [d.name for d in self.model_path.iterdir() if d.is_dir()] if self.model_path.exists() else []
            raise FileNotFoundError(
                f"Model directory not found: {self.save_dir}\n"
                f"Available model IDs: {', '.join(available_models[:10])}" + 
                (f" (and {len(available_models) - 10} more)" if len(available_models) > 10 else "")
            )
        
        # Create directory if it doesn't exist (should already exist from training)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=cfg, f=self.save_dir / "sample_config.yaml")

        # Read training config from model directory and instantiate the right datamodule
        train_cfg_path = self.save_dir / "train_config.yaml"
        if not train_cfg_path.exists():
            raise FileNotFoundError(
                f"Training config not found: {train_cfg_path}\n"
                f"This model may not have been trained yet."
            )
        train_cfg = OmegaConf.load(train_cfg_path)
        self.datamodule: Datamodule = instantiate(train_cfg.datamodule)
        self.fourier_transform: bool = self.datamodule.fourier_transform
        self.datamodule.prepare_data()
        self.datamodule.setup()
        # Get number of steps and samples
        self.num_samples: int = cfg.num_samples
        self.num_diffusion_steps: int = cfg.num_diffusion_steps

        # Load score model from checkpoint
        best_checkpoint_path = get_best_checkpoint(self.save_dir / "checkpoints")
        model_type = get_model_type(train_cfg)
        # Use weights_only=False for PyTorch 2.6+ compatibility
        # (checkpoints contain custom classes like VPScheduler)
        self.score_model = model_type.load_from_checkpoint(
            checkpoint_path=best_checkpoint_path,
            weights_only=False
        )
        if torch.cuda.is_available():
            self.score_model.to(device=torch.device("cuda"))

        # Instantiate sampler (with optional cache support)
        sampler_partial = instantiate(cfg.sampler)
        # Add cache support if specified in config
        use_cache = cfg.get("use_cache", False)
        cache_kwargs = cfg.get("cache_kwargs", {})
        if use_cache:
            self.sampler: DiffusionSampler = sampler_partial(
                score_model=self.score_model,
                use_cache=True,
                cache_kwargs=cache_kwargs
            )
        else:
            self.sampler: DiffusionSampler = sampler_partial(score_model=self.score_model)

        # Instantiate metrics
        metrics_partial = instantiate(cfg.metrics)
        self.metrics: MetricCollection = metrics_partial(
            original_samples=self.datamodule.X_train
        )

    def sample(self) -> None:
        # Sample from score model

        X = self.sampler.sample(
            num_samples=self.num_samples, num_diffusion_steps=self.num_diffusion_steps
        )

        # Map to the original scale if the input was standardized
        if self.datamodule.standardize:
            feature_mean, feature_std = self.datamodule.feature_mean_and_std
            X = X * feature_std + feature_mean

        # If sampling in frequency domain, bring back the sample to time domain
        if self.fourier_transform:
            X = idft(X)

        # Compute metrics
        results = self.metrics(X)
        logging.info(f"Metrics:\n{dict_to_str(results)}")

        # Save everything
        logging.info(f"Saving samples ands metrics to {self.save_dir}.")
        yaml.dump(
            data=results,
            stream=open(self.save_dir / "results.yaml", "w"),
        )
        torch.save(X, self.save_dir / "samples.pt")
        
        # Save cache samples separately if cache was used
        if self.score_model.cache is not None:
            cache_stats = self.score_model.cache.get_cache_stats()
            if cache_stats:
                logging.info(f"Cache statistics: {dict_to_str(cache_stats)}")
                cache_output_dir = self.save_dir / "samples_cache"
                cache_output_dir.mkdir(exist_ok=True)
                torch.save(X, cache_output_dir / "samples.pt")
                logging.info(f"Cache samples saved to {cache_output_dir / 'samples.pt'}")


@hydra.main(version_base=None, config_path="conf", config_name="sample")
def main(cfg: DictConfig) -> None:
    runner = SamplingRunner(cfg)
    runner.sample()


if __name__ == "__main__":
    main()
