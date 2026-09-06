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
