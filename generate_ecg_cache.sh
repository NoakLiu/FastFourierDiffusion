#!/bin/bash
# Train and sample ECG with cache

# Train
python cmd/train.py datamodule=ecg fourier_transform=true

# Get latest model
MODEL_ID=$(ls -t lightning_logs/*/checkpoints/*.ckpt 2>/dev/null | head -1 | sed 's|.*/\([^/]*\)/checkpoints.*|\1|')

# Sample with cache
python cmd/sample.py model_id=$MODEL_ID use_cache=true cache_kwargs={} #num_samples=100 num_diffusion_steps=100

