# Load test of the autoencoder API

Locust test that sends requests to the models loaded by the API. Each
simulated user reads the loaded models with `GET /models/loaded` and then
calls, for a random model:

| Request | Weight | What it measures |
|---|---|---|
| `GET /signals?ticker=` | 6 | Suggestion of the day |
| `GET /signals/history?ticker=` | 3 | Suggestions of a 90 day window |
| `GET /predictions/market?ticker=` | 1 | Outlier prediction on market data (downloads prices) |
| `GET /health` | 1 | Health check |

## Load profiles

The profile is chosen with `LOAD_PROFILE`:

- `spike` (peak): baseline users, a burst to `LOAD_MAX_USERS`, then the
  baseline again. Each phase lasts `LOAD_SPIKE_SECONDS`.
- `ramp` (linear increase): from one user to `LOAD_MAX_USERS` over
  `LOAD_RAMP_SECONDS`.
- `step` (progressive): adds `LOAD_STEP_USERS` users every
  `LOAD_STEP_SECONDS`, until `LOAD_MAX_USERS` is reached and held for one
  stage.

The test fails (exit code 1) when the share of failed requests is above
`LOAD_MAX_FAILURE_RATIO` or when no request was sent.

| Variable | Default | Meaning |
|---|---|---|
| `LOAD_PROFILE` | `step` | `spike`, `ramp` or `step` |
| `LOAD_MAX_USERS` | `50` | Highest number of simulated users |
| `LOAD_RAMP_SECONDS` | `600` | Duration of the ramp profile |
| `LOAD_STEP_SECONDS` | `60` | Length of each step stage |
| `LOAD_STEP_USERS` | `5` | Users added at each step stage |
| `LOAD_SPIKE_SECONDS` | `60` | Length of each spike phase |
| `LOAD_MAX_FAILURE_RATIO` | `0.01` | Highest share of failed requests allowed |

## Running

Start the API with the model files in `models/` (it serves the models it
finds there). From the repository root:

```bash
uv run uvicorn generative_models.serving.api:app --port 8001
```

Open the Locust web interface, to watch the users grow:

```bash
uv run --project load_test locust -f load_test/locustfile.py \
    --host http://127.0.0.1:8001
```

Then open http://localhost:8089, and choose the users and the spawn rate
in the interface. The load shape of the file overrides both.

Run one profile without the interface and keep the CSV files:

```bash
LOAD_PROFILE=spike LOAD_MAX_USERS=100 uv run --project load_test locust \
    -f load_test/locustfile.py --host http://127.0.0.1:8001 \
    --headless --only-summary --csv load_test/results/spike
```

Run the three profiles one after another (CSV files go to
`load_test/results/`, which is ignored by git):

```bash
bash load_test/run_profiles.sh
```

The script exits with code 1 when any profile fails its threshold.

## Reading the results

- `--only-summary` prints one table at the end, with the requests per
  second, the failures and the response time percentiles per request.
- The CSV files hold the same tables: `*_stats.csv` for the requests and
  `*_stats_history.csv` for the values over time, which show where the
  response time starts to grow as the users increase.

## Notes

- The load test depends on the API and, for `/predictions/market`, on the
  price source. A slow download shows as a slow request in that line of
  the table.
- The test uses the models that the API loads at start. If the API reports
  no model, the users stop and the test fails.
- The Locust version is pinned in `load_test/uv.lock`.
