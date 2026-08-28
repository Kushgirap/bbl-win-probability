"""
Tests for the FastAPI prediction service in api/main.py.

The parity check at the bottom exists for the same reason
test_inference_parity.py does: the service and the notebook must agree on what
a given state predicts, and importing the model straight out of api.main
(rather than reloading it) is what makes "the same instance answered both
calls" actually true.
"""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from api.main import app, encoders, model, overs_completed_from_balls, par_table  # noqa: E402
from features import build_features_single, encode_single  # noqa: E402

client = TestClient(app)

FIRST_INNINGS_STATE = {
    "innings_number": 1,
    "balls_faced": 60,
    "cumulative_runs": 75,
    "cumulative_wickets": 2,
    "batting_team": "Sydney Sixers",
    "venue": "Sydney Cricket Ground",
}

CHASE_STATE = {
    "innings_number": 2,
    "balls_faced": 60,
    "cumulative_runs": 80,
    "cumulative_wickets": 3,
    "batting_team": "Melbourne Stars",
    "venue": "Melbourne Cricket Ground",
    "target_score": 165.0,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["best_iteration"] == model.best_iteration
    assert body["features"] == 17


def test_predict_first_innings():
    resp = client.post("/predict", json=FIRST_INNINGS_STATE)
    assert resp.status_code == 200

    body = resp.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert body["batting_team"] == "Sydney Sixers"
    assert body["is_chasing"] is False
    assert body["match_state"] == "live"
    assert body["confidence_note"] is None


def test_predict_chase():
    resp = client.post("/predict", json=CHASE_STATE)
    assert resp.status_code == 200

    body = resp.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert body["batting_team"] == "Melbourne Stars"
    assert body["is_chasing"] is True
    assert body["match_state"] == "live"
    assert body["confidence_note"] is None  # ball 60 is past the early-chase window


def test_chase_without_target_returns_422():
    state = {k: v for k, v in CHASE_STATE.items() if k != "target_score"}
    resp = client.post("/predict", json=state)

    assert resp.status_code == 422
    assert "target_score" in resp.json()["detail"]


def test_below_prediction_floor_returns_422():
    state = {**FIRST_INNINGS_STATE, "balls_faced": 10}  # floor for innings 1 is 30
    resp = client.post("/predict", json=state)

    assert resp.status_code == 422
    assert "prediction floor" in resp.json()["detail"]


def test_target_score_nan_returns_422():
    # httpx's own json= encoder refuses to serialize NaN, so a bare literal
    # never leaves the client - which would hide the very case this is
    # testing. Sending raw bytes reproduces what a looser client (or
    # Python's own json.dumps, which allows NaN by default) actually puts on
    # the wire, and confirms the server rejects it rather than relying on the
    # client to have already done so.
    state = {**CHASE_STATE, "target_score": float("nan")}
    body = json.dumps(state)  # allow_nan=True by default - emits a literal NaN
    resp = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )

    assert resp.status_code == 422
    assert "finite" in resp.text


def test_early_chase_returns_confidence_note():
    state = {**CHASE_STATE, "balls_faced": 12}
    resp = client.post("/predict", json=state)

    assert resp.status_code == 200
    body = resp.json()
    assert body["match_state"] == "live"  # advisory note, not a resolved match
    note = body["confidence_note"]
    assert note is not None
    assert "0.756" in note
    assert "0.818" in note


def test_chase_complete_resolves_as_win():
    state = {
        "innings_number": 2,
        "balls_faced": 90,
        "cumulative_runs": 200,
        "cumulative_wickets": 4,
        "batting_team": "Adelaide Strikers",
        "venue": "Adelaide Oval",
        "target_score": 150.0,
    }
    resp = client.post("/predict", json=state)

    assert resp.status_code == 200
    body = resp.json()
    assert body["win_probability"] == 1.0
    assert body["match_state"] == "resolved"
    assert "not a model prediction" in body["confidence_note"]


def test_chase_all_out_resolves_as_loss():
    state = {
        "innings_number": 2,
        "balls_faced": 95,
        "cumulative_runs": 120,
        "cumulative_wickets": 10,
        "batting_team": "Adelaide Strikers",
        "venue": "Adelaide Oval",
        "target_score": 150.0,
    }
    resp = client.post("/predict", json=state)

    assert resp.status_code == 200
    body = resp.json()
    assert body["win_probability"] == 0.0
    assert body["match_state"] == "resolved"
    assert "all out" in body["confidence_note"]


def test_chase_overs_exhausted_resolves_as_loss():
    state = {
        "innings_number": 2,
        "balls_faced": 120,
        "cumulative_runs": 140,
        "cumulative_wickets": 6,
        "batting_team": "Adelaide Strikers",
        "venue": "Adelaide Oval",
        "target_score": 150.0,
    }
    resp = client.post("/predict", json=state)

    assert resp.status_code == 200
    body = resp.json()
    assert body["win_probability"] == 0.0
    assert body["match_state"] == "resolved"
    assert "not a model prediction" in body["confidence_note"]


def test_innings_one_all_out_keeps_predicting():
    state = {
        "innings_number": 1,
        "balls_faced": 85,
        "cumulative_runs": 130,
        "cumulative_wickets": 10,
        "batting_team": "Adelaide Strikers",
        "venue": "Adelaide Oval",
    }
    resp = client.post("/predict", json=state)

    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert body["match_state"] == "innings_complete"


def test_innings_one_full_overs_keeps_predicting():
    state = {
        "innings_number": 1,
        "balls_faced": 120,
        "cumulative_runs": 180,
        "cumulative_wickets": 6,
        "batting_team": "Adelaide Strikers",
        "venue": "Adelaide Oval",
    }
    resp = client.post("/predict", json=state)

    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["win_probability"] <= 1.0
    assert body["match_state"] == "innings_complete"


@pytest.mark.parametrize("state", [FIRST_INNINGS_STATE, CHASE_STATE])
def test_api_matches_direct_call(state):
    resp = client.post("/predict", json=state)
    assert resp.status_code == 200

    payload = {k: v for k, v in state.items() if v is not None}
    payload["overs_completed"] = overs_completed_from_balls(state["balls_faced"])
    X = build_features_single(payload, par_table)
    X = encode_single(X, encoders)
    expected = float(model.predict_proba(X)[:, 1][0])

    assert resp.json()["win_probability"] == pytest.approx(expected, abs=1e-12)
