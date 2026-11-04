# Convolutional autoencoder

A 1D convolutional autoencoder (CNN) that flags unusual windows of daily
log-returns. It is an alternative to the dense model of
`base_autoencoder`, and it follows the same contracts, so the shared
training command, the ticker manager and the overfitting report of the
`evaluate` job work with it unchanged.

## What the model does

The encoder slides two strided convolutions over the window of returns
instead of connecting every time step to its own weight. A filter that
learns, for example, a sharp drop followed by a recovery recognizes that
shape anywhere in the window, so the number of parameters does not grow
with the window size. The feature maps are then projected to a latent
vector, and the decoder mirrors the path with two transposed
convolutions that rebuild the window.

As in the dense model, the reconstruction error of a window is its
anomaly score, and the anomaly threshold is the percentile
`100 * (1 - contamination)` of the scores of the whole series.

### Shapes

The model takes and returns tensors of shape `(batch_size,
window_size)`, exactly like `SimpleAutoencoder`. The channel dimension
that the convolutions need is added and removed inside `forward`. The
decoder restores the input length for **any** window size, including odd
ones, because the output padding of each transposed convolution is
computed from the encoder lengths in `__init__`.

### Architecture

| Stage | Layers |
|---|---|
| `encoder_convolutions` | `Conv1d(1, C)` → ReLU → `Conv1d(C, C)` → ReLU |
| `encoder_projection` | `Flatten` → `Linear(C * L, latent)` → ReLU |
| `decoder_projection` | `Linear(latent, C * L)` → ReLU |
| `decoder_convolutions` | `ConvTranspose1d(C, C)` → ReLU → `ConvTranspose1d(C, 1)` |

`C` is `hidden_channel_count` (16 by default), `latent` is
`latent_layer_size` (4 by default) and `L` is the window length after
the two convolutions. Every convolution uses kernel 3, stride 2 and
padding 1. The latent vector is non-negative, and the last layer has no
activation, so reconstructed returns may be negative.

### Regularization

A convolutional encoder fits a training window much more tightly than a
dense one, so without a brake it reaches a far lower training error than
validation error. Two settings of `OptimizationConfig` prevent that:

| Setting | Default | What it does |
|---|---|---|
| `weight_decay` | `1e-3` | L2 penalty of Adam, which keeps the filters small. |
| `early_stopping_patience` | `20` | Stops after 20 epochs without a better validation error, and restores the weights of the best epoch. |

`number_of_epochs` (and the `--epochs` argument) is therefore an upper
bound, not the number of epochs that run. On the tickers measured below
training stops around epoch 60 to 80 out of 300. The epoch whose weights
were restored is logged to wandb as `best_epoch`, next to
`stopped_at_epoch`. Set `early_stopping_patience` to `None` to train
every epoch and keep the last weights.

Measured effect on real data (`2020-01-01` to `2026-09-15`, window 10,
300 epochs, learning rate 0.001):

| Ticker | Ratio without regularization | Ratio with it | Test error vs naive baseline |
|---|---|---|---|
| AMD | 1.156 | 0.997 | 1.116 < 1.811 |
| INTC | 3.365 | 1.844 | 2.125 < 3.098 |

## How to train it

With the shared command, which trains several tickers and writes one
overfitting report per ticker:

```bash
uv run python -m generative_models.models.run_model_training \
    --model convolutional_autoencoder \
    --tickers AAPL,MSFT \
    --epochs 300 \
    --learning-rate 0.001 \
    --window-size 10
```

To check the report of one ticker, as the `evaluate` job does:

```bash
uv run python -m generative_models.models.convolutional_autoencoder.overfitting \
    --report-path artifacts/convolutional-autoencoder-aapl-overfitting.json
```

The weights are written to
`artifacts/convolutional-autoencoder-<ticker>.pt`, and the report to
`artifacts/convolutional-autoencoder-<ticker>-overfitting.json`.

This package has no `run_train_script.py`. That module is optional in
spec 01, and the shared command above covers both one and several
tickers.

## Experiment tracking

Every run is tracked with `WandbExperimentTracker`, in the project
`autoencoder-convolutional-outlier-trading-agent`, which is separate
from the project of the dense model. The run config holds every
hyperparameter, the training loss is logged once per epoch, and the
trained weights are registered as an artifact named
`convolutional-autoencoder-<ticker>`. The API key is read from
`WANDB_API_KEY` in the local `.env` file, never from the code.

## Known limits

- **A ticker whose volatility jumps cannot pass the `evaluate` job, with
  any model.** The windows are normalized with the mean and the standard
  deviation of the training split, so when the validation period is more
  volatile than the training period, the validation error rises by the
  ratio of the variances even for a model that generalizes perfectly.
  The lowest `overfitting_ratio` a ticker can reach is therefore the
  ratio between the naive-baseline error of its validation split and
  that of its training split. For `INTC` over
  `2020-01-01` to `2026-09-15` that floor is **1.681**, above the 1.5
  limit of spec 04, because its validation window covers the 2024
  collapse: the validation returns are 1.28 times, and the test returns
  1.76 times, the training standard deviation. `AMD` over the same range
  has a floor of 0.933 and passes. Driving the ratio below the floor is
  only possible by making the model learn nothing, which then fails the
  baseline criterion. The useful check for this model is the one it
  passes on both tickers: the test error is far below the naive
  baseline.
- **The serving API does not load this model.** The registries in
  `serving/` rebuild a `SimpleAutoencoder` and only look at artifacts
  whose name starts with `simple-autoencoder-`, so a model of this
  package is ignored instead of being loaded with the wrong
  architecture. Serving the CNN model needs a change in the shared
  `serving/` package, which spec 07 reserves for a separate pull
  request.
- **`hidden_layer_size` in the report means channels.** The report
  format of spec 04 is fixed, so `hidden_channel_count` is written to
  the `hidden_layer_size` field. It is a number of filters, not a number
  of units of a dense layer.
- **Two fixed downsampling steps.** The encoder always applies two
  stride-2 convolutions. For a very short window the encoded length
  collapses to a single step, which leaves the filters little to read.
  A window of 10 or more returns is the useful range.
- **Only the close price is used.** As in the dense model, the input is
  the series of log-returns of the closing price. Volume and intraday
  prices are ignored.
