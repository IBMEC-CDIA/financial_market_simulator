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
uv run python -m generative_models.models.autoencoder.train_autoencoder     --ticker AAPL
uv run python -m generative_models.serving.predict_autoencoder
uv run uvicorn generative_models.serving.api:app --reload
```

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
