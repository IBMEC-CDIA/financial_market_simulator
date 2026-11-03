# GRU autoencoder

Recurrent autoencoder that flags anomalous windows of daily log-returns.
It follows the same contract as `base_autoencoder` (specs in
`specs/project_structure_check/`), so it runs in the shared training and
evaluation pipeline without any change to the workflows.

## How the model works

Each window of `window_size` normalized log-returns is read as a sequence
with one value per time step:

1. **Encoder:** a GRU reads the window. Its last hidden state
   (`hidden_layer_size` units) is projected by a linear layer to the
   latent vector (`latent_layer_size` units).
2. **Decoder:** the latent vector is projected back to `hidden_layer_size`
   units and repeated over every time step. A second GRU produces one hidden
   state per step, and a linear layer maps each one to the reconstructed
   value. There is no final activation, so the output may be negative.

The mean squared reconstruction error of a window is its anomaly score. A
window is an outlier when its score is above the percentile
`100 * (1 - contamination)` of the scores of the complete series.

Compared with the dense `SimpleAutoencoder`, the GRU reads the window in
time order, so it can capture the order of the returns inside the window
and not only their values.

## Modules

| Module | Content |
|---|---|
| `autoencoder.py` | `GRUAutoencoder` architecture |
| `config.py` | `GRUAutoencoderTrainingConfig` and its sub-configurations |
| `trainer.py` | `GRUAutoencoderTrainer`: data preparation, training, evaluation and wandb tracking |
| `trainer_factory.py` | `build_trainer`, the contract used by `run_model_training` (spec 06) |
| `overfitting.py` | overfitting report and its evaluation command (spec 04) |
| `run_train_script.py` | training command for one ticker (spec 02) |

The wandb runs go to the project `gru-autoencoder-outlier-trading-agent`.
The files and artifacts are named `gru_autoencoder-<ticker_slug>`, for
example `gru_autoencoder-aapl.pt` and `gru_autoencoder-aapl-overfitting.json`.

## Training

Several tickers, with the shared command:

```bash
uv run python -m generative_models.models.run_model_training \
    --model gru_autoencoder --tickers AAPL,MSFT --epochs 300
```

One ticker, from a CSV in the format of spec 03:

```bash
uv run python -m generative_models.models.gru_autoencoder.run_train_script \
    --ticker AAPL --price-data-path data/AAPL.csv --epochs 2
```

Check the overfitting report:

```bash
uv run python -m generative_models.models.gru_autoencoder.overfitting \
    --report-path artifacts/gru_autoencoder-aapl-overfitting.json
```

Set `WANDB_MODE=disabled` to train without a wandb key.

## Known limits

- The serving layer (`serving/`) rebuilds only the `SimpleAutoencoder` of
  `base_autoencoder` (spec 08). Do not put the GRU `.pt` and
  `-overfitting.json` files in the folder of `AUTOENCODER_MODEL_DIR`: the
  local registry would try to load them as a `SimpleAutoencoder` and fail.
  Serving this model needs its own reader, in a separate PR for `serving/`.
- The encoder and the decoder have a single GRU layer each.
- The GRU is slower to train than the dense model, mainly on CPU.
