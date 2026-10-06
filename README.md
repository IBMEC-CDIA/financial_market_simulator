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

### Training the autoencoder for a ticker

`AutoencoderTrainer`
(`src/generative_models/models/autoencoder/trainer.py`) trains the
`SimpleAutoencoder` on the daily log-returns of one ticker, driven by
an `AutoencoderTrainingConfig` object
(`src/generative_models/models/autoencoder/config.py`) and tracked
with Weights & Biases. To run it from the command line:

```bash
uv run python -m generative_models.models.autoencoder.train_autoencoder \
    --ticker AAPL
```

Every configuration value can be overridden with a flag (for example,
`--start-date`, `--epochs`, `--window-size`, `--learning-rate`); run
the script with `--help` for the full list. The trained weights are
saved to `artifacts/simple-autoencoder-<ticker>.pt` by default.

### Loading a trained autoencoder and predicting

`WandbAutoencoderRegistry`
(`src/generative_models/serving/wandb_autoencoder_registry.py`)
downloads autoencoder artifacts logged by `AutoencoderTrainer` from
Weights & Biases and wraps them in an `AutoencoderPredictor`
(`src/generative_models/serving/autoencoder_predictor.py`), which
normalizes new log-returns with the training statistics and flags
windows whose reconstruction error exceeds the stored anomaly
threshold. To load the most recently trained model and score the
last year of prices of its ticker:

```bash
uv run python -m generative_models.serving.predict_autoencoder
```

Use `--ticker` to load the latest model of a specific ticker and
`--start-date` / `--end-date` to change the scored period. Artifacts
are downloaded to `artifacts/wandb/`.

### Serving the autoencoder with the REST API

`src/generative_models/serving/api.py` exposes the trained autoencoders
through a FastAPI application. Models are downloaded from wandb once
and kept in memory by a process-wide singleton
(`AutoencoderModelService`), which reuses the loading logic of the
`predict_autoencoder` script. Start the API with:

```bash
uv run uvicorn generative_models.serving.api:app --reload
```

The interactive Swagger documentation, with every route, parameter
and schema described, is available at `http://127.0.0.1:8000/docs`
(ReDoc at `/redoc`). Main routes:

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Service status and models in memory. |
| `GET` | `/models/loaded` | Metadata of the models in memory. |
| `GET` | `/models/available` | Servable model artifacts stored in wandb. |
| `GET` | `/models/default` | Default model (most recently trained). |
| `POST` | `/models/reload` | Reload a model from wandb without restarting. |
| `POST` | `/predictions/log-returns` | Detect outliers in raw log-returns. |
| `POST` | `/predictions/prices` | Detect outliers from closing prices. |
| `GET` | `/predictions/market` | Detect outliers in recent market data. |

The API reads `WANDB_API_KEY` from `.env` and accepts the optional
`AUTOENCODER_WANDB_PROJECT`, `AUTOENCODER_WANDB_ENTITY`,
`AUTOENCODER_DEVICE` and `AUTOENCODER_PRELOAD_MODEL` environment
variables.

## Continuous integration

The workflow defined in
[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs the
test suite automatically:

- when a pull request is opened,
- on every new commit pushed to an open pull request,
- and when a pull request is merged into `main`.

## Contributing guidelines

Coding conventions for this repository (docstring style, type hints,
notebook formatting, testing and documentation requirements) are
documented in [`CLAUDE.md`](CLAUDE.md) and
[`.github/copilot-instructions.md`](.github/copilot-instructions.md).
Please read them before contributing.
