#!/usr/bin/env bash
# Runs the spike, ramp and step load profiles one after another against the
# autoencoder API and keeps the CSV results in load_test/results/.
#
# Usage, from the repository root:
#   bash load_test/run_profiles.sh
#
# Environment variables:
#   LOAD_HOST           API base URL (default: http://127.0.0.1:8001)
#   LOAD_RESULTS_DIR    folder for the CSV files (default: load_test/results)
#   LOAD_* settings     any setting read by load_test/load_profiles.py

set -uo pipefail

host="${LOAD_HOST:-http://127.0.0.1:8001}"
results_dir="${LOAD_RESULTS_DIR:-load_test/results}"
status=0

mkdir -p "$results_dir"

for profile in spike ramp step; do
    echo "=== profile: $profile ==="
    LOAD_PROFILE="$profile" uv run --project load_test locust \
        -f load_test/locustfile.py \
        --host "$host" \
        --headless \
        --only-summary \
        --csv "$results_dir/$profile" || status=1
done

exit "$status"
