import os
from omegaconf import DictConfig

import wandb
from fdiff.utils.extraction import flatten_config


def maybe_initialize_wandb(cfg: DictConfig) -> str | None:
    """Initialize wandb if necessary."""
    cfg_flat = flatten_config(cfg)
    if "pytorch_lightning.loggers.WandbLogger" in cfg_flat.values():
        # Entity defaults to user's own account (can be overridden via WANDB_ENTITY env var)
        # Mode defaults to online (can be overridden via WANDB_MODE env var, e.g., "offline" or "disabled")
        init_kwargs = {
            "project": "FourierDiffusion",
            "config": cfg_flat,
        }
        # Only set entity if explicitly specified via environment variable
        # Otherwise, wandb will use the user's own account
        if "WANDB_ENTITY" in os.environ:
            init_kwargs["entity"] = os.environ["WANDB_ENTITY"]
        # Only set mode if explicitly specified via environment variable
        # Otherwise, wandb will use the default online mode
        if "WANDB_MODE" in os.environ:
            init_kwargs["mode"] = os.environ["WANDB_MODE"]
        wandb.init(**init_kwargs)
        assert wandb.run is not None
        run_id = wandb.run.id
        assert isinstance(run_id, str)
        return run_id
    else:
        return None
