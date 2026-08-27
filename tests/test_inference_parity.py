"""
Parity between the batch and single-row feature pipelines.

build_features_single exists because a prediction service has one delivery, not a
match to group over, so it recomputes the target-dependent columns instead of
calling add_target_features. That recomputation is exactly the kind of thing that
quietly drifts from the batch path over time - this test is what catches it. It
takes real match states, runs each one through both pipelines, and asserts every
feature column agrees to floating-point precision. A regression here means the
notebook's model and the serving path have started predicting different things
from the same state.
"""

import json
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from extraction import extract_match_state  # noqa: E402
from features import (  # noqa: E402
    FEATURES,
    PREDICTION_FLOOR,
    build_features,
    build_features_single,
)

JSON_FOLDER = os.path.join(REPO_ROOT, "data", "cricsheet_jsons")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

SAMPLE_SIZE = 200
SEED = 42
TOLERANCE = 1e-12


def _load_real_states():
    """Every delivery from every playable match on disk - the same universe of
    states the notebook trains and tests on."""
    json_files = sorted(f for f in os.listdir(JSON_FOLDER) if f.endswith(".json"))

    all_states = []
    for filename in json_files:
        with open(os.path.join(JSON_FOLDER, filename)) as f:
            match = json.load(f)

        if "winner" not in match["info"].get("outcome", {}):
            continue

        match_id = filename.replace(".json", "")
        for innings_num in range(len(match["innings"])):
            all_states.extend(extract_match_state(match, innings_num, match_id))

    return pd.DataFrame(all_states)


@pytest.fixture(scope="module")
def par_table():
    return joblib.load(os.path.join(MODELS_DIR, "par_table.joblib"))


@pytest.fixture(scope="module")
def raw_states():
    return _load_real_states()


def _state_from_row(row, targets):
    state = {
        "innings_number": int(row["innings_number"]),
        "balls_faced": int(row["balls_faced"]),
        "cumulative_runs": int(row["cumulative_runs"]),
        "cumulative_wickets": int(row["cumulative_wickets"]),
        "overs_completed": int(row["overs_completed"]),
        "batting_team": row["batting_team"],
        "venue": row["venue"],
    }
    if state["innings_number"] == 2:
        state["target_score"] = float(targets.loc[row["match_id"]])
    return state


def test_single_row_matches_batch_pipeline(raw_states, par_table):
    expected = build_features(raw_states.copy(), par_table)

    # Only states the prediction floor actually allows - anything below it is
    # covered separately, as a ValueError, not a feature comparison.
    eligible = raw_states[
        ((raw_states["innings_number"] == 1)
         & (raw_states["balls_faced"] >= PREDICTION_FLOOR[1]))
        | ((raw_states["innings_number"] == 2)
           & (raw_states["balls_faced"] >= PREDICTION_FLOOR[2]))
    ]

    targets = (
        raw_states[raw_states["innings_number"] == 1]
        .groupby("match_id")["cumulative_runs"]
        .max()
    )

    rng = random.Random(SEED)
    sample_idx = rng.sample(list(eligible.index), SAMPLE_SIZE)

    max_diff = {col: 0.0 for col in FEATURES}
    for idx in sample_idx:
        row = raw_states.loc[idx]
        state = _state_from_row(row, targets)

        actual = build_features_single(state, par_table)
        exp_row = expected.loc[idx, FEATURES]

        for col in FEATURES:
            a, e = actual[col].iloc[0], exp_row[col]
            diff = 0.0 if a == e else abs(float(a) - float(e))
            max_diff[col] = max(max_diff[col], diff)

    print(f"\nChecked {len(sample_idx)} rows across "
          f"{raw_states.loc[sample_idx, 'match_id'].nunique()} matches")
    for col in FEATURES:
        print(f"  {col:24s} max diff {max_diff[col]:.2e}")

    for col in FEATURES:
        assert max_diff[col] < TOLERANCE, (
            f"{col} diverged between build_features and build_features_single: "
            f"max diff {max_diff[col]:.2e}"
        )


def test_missing_target_on_chase_raises(par_table):
    state = {
        "innings_number": 2,
        "balls_faced": 40,
        "cumulative_runs": 50,
        "cumulative_wickets": 2,
        "overs_completed": 6,
        "batting_team": "Sydney Sixers",
        "venue": "Some Ground",
    }
    with pytest.raises(ValueError, match="target_score"):
        build_features_single(state, par_table)


@pytest.mark.parametrize("innings_number,balls_faced", [(1, 29), (2, 0)])
def test_below_prediction_floor_raises(par_table, innings_number, balls_faced):
    state = {
        "innings_number": innings_number,
        "balls_faced": balls_faced,
        "cumulative_runs": 10,
        "cumulative_wickets": 0,
        "overs_completed": 4,
        "batting_team": "Sydney Sixers",
        "venue": "Some Ground",
        "target_score": 150.0,
    }
    with pytest.raises(ValueError, match="prediction floor"):
        build_features_single(state, par_table)
