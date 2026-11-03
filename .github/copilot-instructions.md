# GitHub Copilot Instructions

## About Python code

Always follow these guidelines when writing or editing Python code in
this project:

- Always use `uv` as the package manager whenever a package needs to be
  installed (including in Colab).
- Write all docstrings in English.
- Docstrings must be complete: describe the input parameters
  (`Parameters`) and the return value (`Returns`).
- When a function or method is complex, include a usage example in the
  docstring (`Example` section).
- Respect the per-line character limit in docstrings according to
  PEP 8 / PEP 257 (79 characters).
- Variables, functions, classes and methods must have descriptive,
  meaningful names written in English.
- Use type hints on function and method parameters and return values.
- Do not use type hints on simple variable assignments inside the body
  of the code.
- Do not add comments in the code — keep only the docstrings.
- Whenever a function or method has 2 or more parameters, break the
  signature across multiple lines, one parameter per line, for example:

```python
def func1(
    param_1: type_1,
    param_2: type_2,
    ...
    param_n: type_n,
) -> type_of_return:
```

- Whenever production code is created or changed, add (or update) the
  corresponding unit tests under `tests/`, mirroring the structure of
  `src/generative_models/`.
- Always check whether the existing documentation (README, docstrings,
  files under `docs/`, if any) is still consistent with the change; if
  it needs an update, update it right away.

## About experiment tracking

- Whenever a PyTorch model is trained (a training loop with an
  optimizer and a loss function), always integrate
  `WandbExperimentTracker` from
  `generative_models.tracking.wandb_tracker` to track the experiment,
  following the pattern already used in
  `notebooks/regression_log_return_training.ipynb` and
  `notebooks/outlier_detection_comparison.ipynb`:
  - Instantiate the tracker with a `project_name` and a `config`
    dictionary covering the experiment's hyperparameters (dataset
    settings, random seed, epochs, learning rate, architecture sizes,
    optimizer, loss function, etc.), then call `start_run()` before
    the training loop begins.
  - Inside the training loop, call `log_metrics` at least once per
    epoch with the training loss (and any other relevant metric).
  - After training finishes, call `log_model` to register the trained
    model as a wandb artifact, passing a `description` and any extra
    `metadata` not already covered by `config` (for example, the
    final training loss).
  - Call `finish_run()` once the experiment (including evaluation) is
    complete.
- Do not hardcode a wandb API key in code or notebooks. The key must
  be read from a `WANDB_API_KEY` variable in a local `.env` file, as
  already handled by `WandbExperimentTracker`.

## About notebooks

- Before each code cell, add a text (Markdown) cell briefly explaining
  what the following code cell does or should do.
- Every notebook must start with a header (Markdown cell) containing:
  - Name: Renan Santos Mendes
  - Email: renansantosmendes@gmail.com
  - Course subject: `<COURSE_SUBJECT_PLACEHOLDER>`
  - Program: `<PROGRAM_PLACEHOLDER>`
- Do not use emojis or emoticons anywhere in the notebook (code or text
  cells), to keep the course material looking professional.

## Cache and generated files

Cache and generated files must not stay in the repository. Whenever you
run tests, linters, coverage or scripts, delete the cache files they leave
behind when you are done. Delete them without asking:

- `__pycache__` folders and `*.pyc` files
- `.pytest_cache`, `.ruff_cache`, `.mypy_cache`
- `.coverage`, `coverage.xml`, `junit-unit.xml`

Never delete the `.venv` folder, the `.env` file, the `wandb/` folder, the
`artifacts/` folder or any file tracked by git. Those are the user's work or
its environment.

## Documentation

Documentation must stay up to date with the code. On every commit, review
`README.md` and every other documentation file that the change touches or
makes outdated: the `specs/` folder, the `docs/` folder when it exists, the
docstrings of the changed code, and the workflow comments and names. Update
them in the same commit as the code change. A commit that makes the README
or a spec describe the wrong behavior, paths or commands is not complete.

## Model packages

Every model lives in its own package in `src/generative_models/models/`.
Before you add or change a model, read the specs in
`specs/project_structure_check/`:

- spec 01 for the package layout and the required modules;
- spec 06 for the trainer contract (`build_trainer`) and the shared
  `run_model_training` command, which a model must not copy;
- spec 07 for the files a model branch may change.

Do not change shared files (workflows, `pyproject.toml`, `uv.lock`, the
root `README.md`, `specs/`, `data/`, `serving/`, `base_autoencoder/`) from a
model branch. The model's own documentation goes in
`src/generative_models/models/<model>/README.md`.
