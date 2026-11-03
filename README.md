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
│       └── autoencoder_pipeline.yml
│
├── Dockerfile                  # image that serves the FastAPI autoencoder API
├── .dockerignore
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
(`data/market_data.py`, `data/datasets.py`, `data/normalization.py`
and `tracking/wandb_tracker.py`), as well as the `ModelRegistry`
interface (`registry/model_registry.py`), are provided and documented
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
`src/generative_models/models/base_autoencoder/config.py`.

Quick run with a short history and two epochs (PowerShell):

```powershell
uv run python -m generative_models.models.base_autoencoder.run_train_script `
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
uv run python -m generative_models.models.base_autoencoder.run_train_script \
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
uv run python -m generative_models.models.base_autoencoder.run_train_script \
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
| `--train-split-ratio` | `0.7` | Fraction of the log-returns used for training, in time order. |
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
uv run python -m generative_models.models.base_autoencoder.run_train_script --help
```

## Continuous integration

The workflow defined in
[`.github/workflows/autoencoder_pipeline.yml`](.github/workflows/autoencoder_pipeline.yml)
(named "Autoencoder Pipeline") runs the test suite automatically:

- when a pull request is opened,
- on every new commit pushed to an open pull request,
- and when a pull request is merged into `main`.

The pipeline starts with `structure`, which checks the project layout:
the required files (`pyproject.toml`, `uv.lock`, `README.md`,
`requirements.txt`, `CLAUDE.md`, `Dockerfile`, `.gitignore`,
`.dockerignore`, `.trivyignore`), that the `tests` folder exists, that
`.env` is ignored by git and Docker and never tracked, that the Dockerfile
installs from `uv.lock`, and that `uv.lock` matches `pyproject.toml`.
For each model package that has a `trainer_factory.py`, the same job checks
that `trainer_factory.py`, `overfitting.py` and `__init__.py` exist, that the
`__init__.py` has a docstring, and that
`tests/generative_models/models/<name>/__init__.py` exists.
Placeholder packages are skipped until they have a trainer. Then `lint` runs pylint. Only after `lint` passes do `unit-tests`,
`end-to-end-tests` and `security` run, in parallel. `test-report` writes a
summary of the results to the job summary page. The `collect-data` job
starts only when every check before it passes.

Quality and security checks in the workflow:

- `unit-tests` runs the unit tests. It fails when the coverage
  of `src/generative_models` is below 90%, and it uploads the JUnit and
  coverage XML reports as the `unit-test-reports` artifact.
- `security` audits the runtime dependencies locked in `uv.lock` with
  `pip-audit` (Windows-only pins are left out, because CI runs on Linux),
  scans the source code with Bandit at medium severity and above, and scans
  the git history for committed secrets with Gitleaks.
- `test-report` ("Test report") runs after the tests and writes the job
  results, the number of unit tests and the line coverage to the summary.
- `pipeline-success` ("Pipeline Run Successfully") runs last. It fails when
  any job did not succeed. On pull requests, the jobs that only run on push
  are allowed to be skipped.
- `build-image` scans the image with Trivy and fails on `CRITICAL` or `HIGH`
  vulnerabilities that have a fix available. The list is written to the
  job summary, before the image is saved or pushed.
  Known exceptions are listed in `.trivyignore`, with a justification and
  an expiry date.

- `evaluate` writes the overfitting table to the job summary page.
- Changes that only touch Markdown files, `specs/` or `docs/` do not start
  the pipeline.
- The shared setup (Python, uv, cache and dependencies) lives in the
  composite action `.github/actions/setup-python-uv`.

The same workflow has three more jobs, which only run after `collect-data`
and its dependencies pass
(`collect-data` and `train` are also started by hand from the Actions tab,
where the ticker is chosen, default `AAPL`):

1. `collect-data` downloads the daily prices of the ticker with
   `generative_models.data.collect_market_data`. Before anything is saved,
   the prices are checked with Great Expectations
   (`generative_models.data.validation`): enough rows, no missing or
   non-positive prices, unique and increasing dates, and no daily log-return
   above 0.5 in absolute value. Invalid data stops the job. Then it saves a
   CSV file with the `close_price`, `log_return` and `split` columns. The
   `split` column marks the first 70% of the log-returns as `train`, the next
   15% as `validation` and the last 15% as `test`, in chronological order.
   The file is uploaded as the `market-data-<ticker>` artifact.
2. `train` downloads that artifact and trains the autoencoder with
   `--price-data-path`, using the `split` column. The validation split is
   used during training to monitor it. After training, the model is
   evaluated once on the test split, together with a naive baseline that
   always predicts zero. The overfitting report (`*-overfitting.json`)
   contains the training, validation and test errors and the baseline error.
   It uploads the `.pt` weights and the report as the
   `autoencoder-model-<ticker>` artifact.
3. `evaluate` runs after `train`. It reads the report and fails unless both
   conditions hold: the validation to training error ratio is at most 1.5
   (no strong overfitting), and the test error is smaller than the baseline
   error (the model learned something the naive prediction does not).

To run the same steps locally, the end-to-end test simulates the prices,
so it does not need network access or a wandb key:

```bash
uv run pytest tests/end_to_end
```

Training logs the run and the model artifact to wandb, so the workflow
needs a `WANDB_API_KEY` repository secret (Settings > Secrets and
variables > Actions). Without the secret, the run falls back to disabled
tracking and the training still completes.

The image is built by its own job, `build-image`, which starts after
`train` succeeds. It saves the image as a workflow artifact, and the
`push-image` job loads it, logs in to Docker Hub and pushes it with the
`latest` tag and the commit SHA tag. Both jobs run on every branch, and
training failures stop them. The workflow needs these repository secrets:

- `DOCKERHUB_USERNAME`: the Docker Hub account name. The image is
  published as `<DOCKERHUB_USERNAME>/financial-market-simulator`.
- `DOCKERHUB_TOKEN`: a Docker Hub access token with write permission.

The image serves the trained autoencoders through the FastAPI application
in `generative_models.serving.api`, on port `8000`. The models are part of
the image: the build downloads the `trained-models` artifact of a successful
Train run into `models/`, and the API loads every model of that folder when
it starts. A prediction is routed to the model of the ticker in the request,
and an unknown ticker gets a `404`. The container does not need the wandb key.

The build workflow (`build.yml`) takes the models from the Train run given in
its `train_run_id` input, or from the latest successful Train run of the
branch. To run the image locally, put the files of a trained model in
`models/` (the `*-overfitting.json` report and the `.pt` weights), then:

```bash
docker build -t financial-market-simulator .
docker run --rm -p 8000:8000 financial-market-simulator
```

Then open `http://127.0.0.1:8000/docs` for the interactive API docs, or
`http://127.0.0.1:8000/health` to check the service. The image sets
`AUTOENCODER_MODEL_DIR=/app/models`. The container accepts these optional
variables:

- `AUTOENCODER_MODEL_DIR`: folder with the model files. When it is set, the
  models are read from this folder and wandb is not used.
- `AUTOENCODER_WANDB_PROJECT` and `AUTOENCODER_WANDB_ENTITY`: used only when
  `AUTOENCODER_MODEL_DIR` is not set.
- `AUTOENCODER_DEVICE`: `cpu` or `cuda`. Defaults to `cuda` when available.
- `AUTOENCODER_PRELOAD_MODEL`: `false` to skip loading the models at startup.

### Training workflow (`train.yml`)

The training workflow is started by hand from the Actions tab (Run workflow),
or from the command line with
`gh workflow run train.yml -f tickers=AAPL,MSFT -f wandb_api_key=...`. Its
inputs are:

- `model`: the model package to train, for example `base_autoencoder`
  (default). The `plan` job checks that the package has `trainer_factory.py`
  and `overfitting.py`, so a placeholder package is refused.
- `tickers`: comma-separated ticker symbols, for example `AAPL,MSFT,PETR4.SA`.
  The jobs run once per ticker.
- `wandb_api_key`: the W&B API key. It is required. When it is left empty,
  the `WANDB_API_KEY` repository secret is used. A typed value is visible in
  the form and in the run inputs, so prefer the secret.
- `epochs`, `learning_rate` and `window_size`: training hyperparameters.

It runs these jobs in order:

1. `ci-check` stops the run unless the CI workflow passed for the same commit.
2. `plan` checks the ticker symbols and the W&B key, and turns the list into
   the matrix of the next jobs.
3. `test-end-to-end` runs the end-to-end tests.
4. `collect-data` downloads and validates the prices of each ticker, then
   uploads them.
5. `train-autoencoder` runs `models/run_model_training` for the chosen model. It
   trains the models of all the tickers with the `TickerModelManager`, in
   sequence, and uploads the weights and the overfitting reports as the
   `trained-models` artifact. The key is masked in the logs.
6. `evaluate` checks the overfitting ratio and the baseline comparison of every
   report in that artifact.
7. `pipeline-success` fails unless every job above succeeded.

### Build workflow (`build.yml`)

The build workflow is also started by hand, with Run workflow. It checks that
the CI passed for the same commit, builds the Docker image locally and scans
it with Trivy. It fails on fixable `CRITICAL` or `HIGH` vulnerabilities. It
does not log in to Docker Hub and does not push the image.

Both `train.yml` and `build.yml` end with a `pipeline-success` job. It fails
unless every job before it succeeded.

### Ticker model manager

`TickerModelManager` (`models/ticker_model_manager.py`) trains the model of
several tickers in sequence. For each ticker it downloads and validates the
prices, creates the trainer with the `trainer_factory` function and runs it.
Each model package provides its `build_trainer` function in
`trainer_factory.py`, which is the default factory when none is given. The
contract is described in
[`specs/project_structure_check/06-model-trainer-contract.md`](specs/project_structure_check/06-model-trainer-contract.md).

## Contributing guidelines

Coding conventions for this repository (docstring style, type hints,
notebook formatting, testing and documentation requirements) are
documented in [`CLAUDE.md`](CLAUDE.md) and
[`.github/copilot-instructions.md`](.github/copilot-instructions.md).
Please read them before contributing.
