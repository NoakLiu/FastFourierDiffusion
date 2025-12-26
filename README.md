# Accelerating Frequency Domain Diffusion Models with Error-Feedback Event-Driven Caching

This repository implements time series diffusion in the frequency domain with E2-CRF (Error-Feedback Event-Driven Caching) acceleration.
 
## 1. Install

From repository:
1. Clone the repository.
2. Create and activate a new environment with conda (with `Python 3.10` or newer).

```shell
conda create -n fdiff python=3.10
conda activate fdiff
```
3. Install the requirement.
```shell
pip install freqdiff
```

4. If you intend to train models, make sure that wandb is correctly configured on your machine by following [this guide](https://docs.wandb.ai/quickstart). 
5. Download datasets (required for training):
   
   First, get your Kaggle API token from [Kaggle Settings](https://www.kaggle.com/settings) and configure it:
   
   ```shell
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
   
   Then download the ECG dataset (default dataset for training):
   
   ```shell
   kaggle datasets download -d shayanfazeli/heartbeat
   unzip heartbeat.zip
   mkdir -p data/ecg
   mv mitbih_train.csv data/ecg/
   mv mitbih_test.csv data/ecg/
   rm heartbeat.zip
   ```
   
   Note: Other datasets (NASDAQ, NASA, etc.) will be automatically downloaded when you run training if Kaggle API is configured.

When the packages are installed and datasets are downloaded, you are ready to train diffusion models!

## 2. Use

### 2.1 Train
In order to train models, you can simply run the following command:

```shell
python cmd/train.py 
```

By default, this command will train a score model in the time domain with the `ecg` dataset for 200 epochs. To reduce the number of epochs, you can use:

```shell
# example for training with 10 epochs (quick test)
python cmd/train.py trainer.max_epochs=10
```

In order to modify other hyperparameters, you can use [hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). The following hyperparameters can be modified to retrain all the models appearing in the paper:

| Hyperparameter | Description | Values |
|----------------|-------------|---------------|
|fourier_transform | Whether or not to train a diffusion model in the frequency domain. | true, false |
| datamodule | Name of the dataset to use. | ecg, mimiciii, nasa, nasdaq, usdroughts|
| datamodule.subdataset | For the NASA dataset only. Selects between the charge and discharge subsets. | charge, discharge |
| datamodule.smoother_width | For the ECG dataset only. Width of the Gaussian kernel smoother applied in the frequency domain. | $\mathbb{R}^+$
| score_model | The backbone to use for the score model. | default, lstm |

At the end of training, your model is stored in the `lightning_logs` directory, in a folder named after the current `run_id`. You can find the `run_id` in the logs of the training and in the [wandb dashboard](https://wandb.ai/) if you have correctly configured wandb.

**Example:** After training, you might see a folder like `lightning_logs/03wb0ssr/`. In this case, `03wb0ssr` is your `model_id`.

### 2.2 Sample

In order to sample from a trained model, you can simply run the following command:

```shell
python cmd/sample.py model_id=XYZ
```
    
where `XYZ` is the `run_id` of the model you want to sample from. At the end of sampling, the samples are stored in the `lightning_logs` directory, in a folder named after the current `run_id`. 

One can then reproduce the plots in the paper by including the  `run_id` to the `run_list` list appearing in [this notebook](notebooks/results.ipynb) and running all cells.

### 2.3 E2-CRF Caching Acceleration

This repository includes E2-CRF (Error-Feedback Event-Driven Cumulative Residual Feature) caching for accelerating frequency domain diffusion models. E2-CRF achieves 2-4× speedup while maintaining sample quality through:

1. **KV Caching**: Caching transformer key-value pairs across diffusion steps
2. **Event-Driven Triggers**: Adaptively recomputing tokens based on CRF residual intensity
3. **Error-Feedback Correction**: Preventing quality degradation through closed-loop error correction
4. **Energy-Weighted Thresholds**: Using spectral energy to determine caching strategy

#### Basic Usage with Caching

To enable caching during sampling, you can modify the sampling code:

```python
from fdiff.models.score_models import ScoreModule
from fdiff.sampling.sampler import DiffusionSampler

# Load your trained model
score_model = ScoreModule.load_from_checkpoint("path/to/checkpoint.ckpt")
score_model.eval()

# Create sampler with caching enabled
sampler = DiffusionSampler(
    score_model=score_model,
    sample_batch_size=1,
    use_cache=True,  # Enable caching
    cache_kwargs={}  # Use default cache parameters
)

# Generate samples (caching is automatically used)
samples = sampler.sample(num_samples=10, num_diffusion_steps=100)
```

#### Custom Cache Configuration

You can customize cache parameters for different speed/quality trade-offs:

```python
sampler = DiffusionSampler(
    score_model=score_model,
    sample_batch_size=1,
    use_cache=True,
    cache_kwargs={
        "K": 5,              # Number of low-frequency tokens always recomputed
        "tau_0": 0.1,        # Base threshold for adaptive caching
        "R": 10,             # Error-feedback correction interval
        "tau_warn": 0.5,     # Warning threshold for event intensity
        "random_probe_ratio": 0.05,  # Ratio of high-freq tokens to probe
    }
)
```

#### Cache Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| K | 5 | Number of low-frequency tokens always recomputed. Lower values = more aggressive caching but potential quality loss. |
| tau_0 | 0.1 | Base threshold for adaptive caching. Lower values = more aggressive caching. |
| R | 10 | Error-feedback correction interval. Lower values = more frequent correction but slower. |
| tau_warn | 0.5 | Warning threshold for event intensity. When exceeded, all tokens are recomputed. |
| random_probe_ratio | 0.05 | Ratio of high-frequency tokens to randomly probe for recalibration. |

**Tuning Guidelines:**
- **For maximum speedup**: Increase K, decrease tau_0, increase R
- **For quality preservation**: Decrease K, increase tau_0, decrease R
- **For balanced performance**: Use default values

#### How to Get Model ID

The `model_id` is the `run_id` generated during training. You can find it in several ways:

1. **List all available models:**
   ```shell
   ls lightning_logs/
   ```
   Each folder name (e.g., `03wb0ssr`, `c3kntdck`) is a model_id.

2. **Check training logs:** The `run_id` is printed during training and saved in the `lightning_logs` directory.

3. **Use `latest`:** Automatically use the most recent checkpoint:
   ```shell
   python cmd/benchmark_cache.py model_id=latest
   ```

4. **Check wandb dashboard:** If wandb is configured, the run_id is also available in the [wandb dashboard](https://wandb.ai/).

### 2.4 Benchmarking

#### Speedup Benchmark

Run the speedup benchmark to compare cached vs. non-cached inference:

```shell
# Use latest model
python cmd/benchmark_cache.py model_id=latest num_samples=10 num_diffusion_steps=100

# Or specify a specific model_id
python cmd/benchmark_cache.py model_id=q5x6ifzc num_samples=10 num_diffusion_steps=100
```

This will output:
- Baseline time (no cache)
- Cached time
- Speedup factor
- Cache statistics (hit ratio, etc.)
- Ablation studies for different parameters

#### Ablation Study

Run the ablation study to understand the contribution of each component:

```shell
# Use latest model
python cmd/ablation_cache.py model_id=latest num_samples=20 num_diffusion_steps=100

# Or specify a specific model_id
python cmd/ablation_cache.py model_id=03wb0ssr num_samples=20 num_diffusion_steps=100
```

This compares:
1. Baseline (no caching)
2. E2-CRF (full method)
3. Without event-driven trigger
4. Without error-feedback correction
5. Without energy-weighted threshold
6. Naive caching

### 2.5 E2-CRF Expected Performance

- **Speedup**: 2-4× on real-world datasets
- **Quality**: Maintained (measured by sliced Wasserstein distance)
- **Memory**: O(1) overhead per diffusion step

<!-- # 3. Contribute

If you wish to contribute, please make sure that your code is compliant with our tests and coding conventions. To do so, you should install the required testing packages with:

```shell
pip install freqdiff[test]
```

Then, you can run the tests with:

```shell
pytest
```

Before any commit, please make sure that your staged code is compliant with our coding conventions by running:

```shell
pre-commit
``` -->

# 3. Citation

If you use the E2-CRF caching implementation, please cite:

```
@project{e2crf2025,
  title={Accelerating Frequency Domain Diffusion with Error-Feedback Caching},
  author={Dong Liu},
  year={2025}
}
```

Our code implementation based on paper https://arxiv.org/abs/2402.05933 and its repo.
