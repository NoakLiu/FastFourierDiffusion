#!/bin/bash
# Generate samples with E2-CRF cache for multiple models
# Usage: ./generate_cache_samples.sh [model_id]

# Model IDs from visualize.ipynb (frequency domain models)
MODELS=(
    "dvd5vv3s"  # NASA charge
    "l7y2cljb"  # NASA discharge
    "emk7nyz3"  # ECG
    "xxqse6xu"  # NASDAQ
    "h8x1c1no"  # USDroughts
)

NUM_SAMPLES=100
NUM_DIFFUSION_STEPS=100

# Use specific model if provided, otherwise use all
[ $# -eq 1 ] && MODELS=("$1")

echo "=========================================="
echo "Generating samples with E2-CRF cache"
echo "=========================================="
echo "Samples: $NUM_SAMPLES, Steps: $NUM_DIFFUSION_STEPS"
echo "Models: ${MODELS[*]}"
echo ""

for model_id in "${MODELS[@]}"; do
    echo "Processing: $model_id"
    
    python <<PYTHON_SCRIPT
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import torch
from omegaconf import OmegaConf
import yaml
from hydra.utils import instantiate
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler
from fdiff.utils.extraction import get_best_checkpoint, get_model_type
from fdiff.utils.fourier import idft
from fdiff.dataloaders.datamodules import Datamodule

model_id = '$model_id'
model_dir = Path.cwd() / 'lightning_logs' / model_id

# Load config and convert to dict to manually handle interpolations
import yaml
with open(model_dir / 'train_config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Replace hydra interpolations in data_dir
if 'datamodule' in config_dict and 'data_dir' in config_dict['datamodule']:
    data_dir = config_dict['datamodule']['data_dir']
    if isinstance(data_dir, str) and ('hydra' in data_dir or data_dir.startswith('\${') or data_dir.startswith('${')):
        config_dict['datamodule']['data_dir'] = str(Path.cwd() / 'data')

# Create OmegaConf from dict (no interpolations to resolve)
train_cfg = OmegaConf.create(config_dict)
datamodule = instantiate(train_cfg.datamodule)
datamodule.prepare_data()
datamodule.setup()

score_model = get_model_type(train_cfg).load_from_checkpoint(
    get_best_checkpoint(model_dir / 'checkpoints'),
    weights_only=False
)
score_model.eval()
if torch.cuda.is_available():
    score_model = score_model.cuda()
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    score_model = score_model.to('mps')

sampler = DiffusionSampler(score_model=score_model, sample_batch_size=1, use_cache=True, cache_kwargs={})
samples = sampler.sample(num_samples=${NUM_SAMPLES}, num_diffusion_steps=${NUM_DIFFUSION_STEPS})

if datamodule.standardize:
    mean, std = datamodule.feature_mean_and_std
    samples = samples * std + mean
if datamodule.fourier_transform:
    samples = idft(samples)

output_dir = model_dir / 'samples_cache'
output_dir.mkdir(exist_ok=True)
torch.save(samples, output_dir / 'samples.pt')

stats = score_model.cache.get_cache_stats() if score_model.cache else {}
print(f'Saved to: {output_dir / "samples.pt"}')
if stats:
    print(f'Cache hit ratio: {stats.get("cache_hit_ratio", 0):.2%}')
PYTHON_SCRIPT
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed for $model_id"
        exit 1
    fi
    
    echo ""
done

echo "Done!"