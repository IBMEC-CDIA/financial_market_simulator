# Financial Market Simulator

Financial Market Simulator is a research and teaching project that
explores generative deep learning models applied to financial time
series (for example, stock log-returns). The goal is to model,
reconstruct and simulate realistic market data using approaches that
range from simple regression baselines to autoencoders, variational
autoencoders (VAEs) and conditional VAEs (CVAEs).

## Project structure

```
financial_market_simulator/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── CLAUDE.md
├── .github/
│   ├── copilot-instructions.md
│   └── workflows/
│       └── tests.yml
│
├── notebooks/                  # exploratory and teaching notebooks
├── specs/                      # module specs for students
│   ├── training/               # training pipeline specs
│   └── serving/                # inference and REST API specs
│
├── src/
│   └── generative_models/
│       ├── data/                # datasets and data loading utilities
│       ├── common/               # shared helpers used across models
│       ├── tracking/             # experiment tracking (e.g. Weights & Biases)
│       ├── models/
│       │   ├── regression/       # regression baselines
│       │   ├── autoencoder/      # autoencoder models
│       │   ├── vae/              # variational autoencoder models
│       │   └── cvae/             # conditional variational autoencoder models
│       ├── serving/              # model inference / serving utilities
│       └── mcp_server/           # exposes trained models through an MCP server
│
├── scripts/                     # standalone scripts (training, data prep, etc.)
├── tests/                       # unit tests, mirroring src/generative_models
└── docker/                      # container definitions
```

## Tech stack

- **Python** 3.12
- **PyTorch** for model definition and training
- **NumPy** / **Pandas** for data manipulation
- **yfinance** for market data retrieval
- **Weights & Biases (wandb)** for experiment tracking
- **FastAPI** / **uvicorn** for model serving
- **Jupyter** for notebooks
- **pytest** for unit testing
- **uv** as the package and environment manager

## Getting started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Python 3.12.10 (uv can install it automatically)

### Environment setup

Create the virtual environment with the pinned Python version:

```bash
uv venv --python 3.12.10
```

Install the full development environment (includes PyTorch with CUDA
support, TensorFlow, Jupyter, Weights & Biases, and other tooling used
locally):

```bash
uv pip install -r requirements.txt
```

Copy the environment variable template and fill in the required
values:

```bash
cp .env.example .env
```

### Running the tests

The lightweight dependencies needed to run the test suite (NumPy,
PyTorch and pytest) are declared in `pyproject.toml`:

```bash
uv sync --extra dev
uv run pytest
```

### Specs for students

The autoencoder training pipeline and its serving layer are **not**
part of this branch: students build them from the notebook
`draft/autoencoder_outlier_gab_draft.ipynb`, following the specs in
the [`specs/`](specs/README.md) folder (written in Portuguese). Each
spec describes the public interface, the expected behavior and
errors, and numbered acceptance criteria that must be turned into
unit tests. The specs are split into two tracks:

- [`specs/training/`](specs/training/README.md): `SimpleAutoencoder`,
  training configuration, `AutoencoderTrainer` and a training CLI;
- [`specs/serving/`](specs/serving/README.md): predictor, wandb
  model registry, prediction CLI, model service singleton and a
  FastAPI application.

The base modules the notebook already imports
(`data/market_data.py`, `data/datasets.py`, `common/normalization.py`
and `tracking/wandb_tracker.py`), as well as the `ModelRegistry`
interface (`serving/model_registry.py`), are provided and documented
by their own specs as reference. Start from
[`specs/README.md`](specs/README.md), which explains the workflow and
the order of the tracks.

Once both tracks are implemented, the project can be used with:

```bash
uv run python -m generative_models.serving.predict_autoencoder
uv run uvicorn generative_models.serving.api:app --reload
```

### Example: training an autoencoder from the command line

The training script trains the autoencoder of one ticker, logs the run
to wandb and saves the weights in the output directory. Only `--ticker`
is required; every other option defaults to the values of the
configuration dataclasses in
`src/generative_models/models/autoencoder/config.py`.

Quick run with a short history and two epochs (PowerShell):

```powershell
uv run python -m generative_models.models.autoencoder.train_autoencoder `
    --ticker AAPL `
    --start-date 2024-01-01 `
    --end-date 2024-12-31 `
    --window-size 10 `
    --batch-size 16 `
    --epochs 2 `
    --device cpu
```

The same run in bash:

```bash
uv run python -m generative_models.models.autoencoder.train_autoencoder \
    --ticker AAPL \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --window-size 10 \
    --batch-size 16 \
    --epochs 2 \
    --device cpu
```

A complete run with the default settings (300 epochs, window of 10
log-returns) and the wandb project and output directory set explicitly:

```bash
uv run python -m generative_models.models.autoencoder.train_autoencoder \
    --ticker PETR4.SA \
    --start-date 2020-01-01 \
    --end-date 2026-09-15 \
    --epochs 300 \
    --learning-rate 0.001 \
    --random-seed 42 \
    --contamination 0.05 \
    --project-name autoencoder-simple-outlier-trading-agent \
    --output-directory artifacts \
    --env-file-path .env
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--ticker` | required | Ticker symbol, for example `AAPL` or `PETR4.SA`. |
| `--start-date`, `--end-date` | `2020-01-01`, `2026-09-15` | Historical range (`YYYY-MM-DD`). |
| `--train-split-ratio` | `0.8` | Fraction of the log-returns used for training, in time order. |
| `--window-size` | `10` | Number of log-returns in each window. |
| `--batch-size` | `32` | Windows per training batch. |
| `--hidden-layer-size`, `--latent-layer-size` | `16`, `4` | Sizes of the autoencoder layers. |
| `--epochs` | `300` | Number of training epochs. |
| `--learning-rate` | `0.001` | Learning rate of the Adam optimizer. |
| `--random-seed` | `42` | Seed for weight initialization and batch shuffling. |
| `--contamination` | `0.05` | Expected fraction of anomalous windows; it sets the anomaly threshold. |
| `--project-name` | `autoencoder-simple-outlier-trading-agent` | wandb project name. |
| `--output-directory` | `artifacts` | Directory where the `.pt` weights are saved. |
| `--env-file-path` | `.env` | `.env` file with `WANDB_API_KEY`. |
| `--device` | automatic | `cpu` or `cuda`. |

The script prints a summary line when it finishes, for example:

```text
ticker=AAPL final_train_loss=1.039433 validation_loss=0.469093 anomaly_threshold=2.228271 model_file_path=artifacts\simple-autoencoder-aapl.pt
```

The weights are saved as `<output-directory>/simple-autoencoder-<ticker>.pt`
(the ticker is lowercased, and characters other than letters, digits,
`.`, `_` and `-` become `-`). The wandb login uses `WANDB_API_KEY` from
the `.env` file. Without a key, the run logs in anonymously. If the
login fails (for example, without internet access), the run falls back
to disabled tracking, and training still completes.

Run `--help` to see the same options from the terminal:

```bash
uv run python -m generative_models.models.autoencoder.train_autoencoder --help
```

## Continuous integration

The workflow defined in
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs the
test suite automatically:

- when a pull request is opened,
- on every new commit pushed to an open pull request,
- and when a pull request is merged into `main`.

The same workflow has a second job, `train`, which only runs after the
`test` job passes. It runs on every push to any branch and can also be
started by hand from the Actions tab, where the ticker is chosen
(default `AAPL`). It also runs on a cron schedule every 5 minutes
(`*/5 * * * *`, in UTC). Each push and each scheduled run therefore logs
a new run and model artifact to wandb. It trains the autoencoder for that ticker and uploads
the `.pt` weights as a workflow artifact.

Training logs the run and the model artifact to wandb, so the workflow
needs a `WANDB_API_KEY` repository secret (Settings > Secrets and
variables > Actions). Without the secret, the run falls back to disabled
tracking and the training still completes.

## Contributing guidelines

Coding conventions for this repository (docstring style, type hints,
notebook formatting, testing and documentation requirements) are
documented in [`CLAUDE.md`](CLAUDE.md) and
[`.github/copilot-instructions.md`](.github/copilot-instructions.md).
Please read them before contributing.
