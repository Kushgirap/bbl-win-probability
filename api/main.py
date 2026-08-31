"""
Prediction service for the BBL win probability model.

Loads the four artifacts from models/ once, at import time, so a request never
pays for a disk read - only for calling the same feature pipeline the notebook
verified against, via build_features_single / encode_single (src/features.py).
"""

import json
import math
import os
import sys
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from features import BALLS_PER_INNINGS, build_features_single, encode_single  # noqa: E402

MODELS_DIR = os.path.join(REPO_ROOT, "models")
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")

model = joblib.load(os.path.join(MODELS_DIR, "win_probability_model.joblib"))
encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
par_table = joblib.load(os.path.join(MODELS_DIR, "par_table.joblib"))
with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    metadata = json.load(f)

# Below this ball, a chase is still predicted (section 5a), but only at the
# lower AUC recorded in metadata - callers get a note, not a different number.
EARLY_CHASE_BALL = metadata["training_filter"]["innings_2_min_ball"]
EARLY_CHASE_AUC = metadata["test_metrics"]["early_chase_auc"]
OVERALL_AUC = metadata["test_metrics"]["roc_auc"]

app = FastAPI(title="BBL Win Probability API")


def _sanitize_for_json(value):
    """Replace non-finite floats with their string form.

    A rejected NaN/Infinity target_score is exactly the value pydantic echoes
    back in its validation error as "input" - and FastAPI's default 422
    response serializes with allow_nan=False, so returning that error
    verbatim would crash the response with a 500 instead. Only the error
    payload goes through this; a normal prediction never contains one."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = _sanitize_for_json(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": errors})


def overs_completed_from_balls(balls_faced: int) -> int:
    """The over a delivery belongs to, 0-indexed - same convention as Cricsheet's
    `over` field. Derived rather than taken from the caller: balls_faced=72 with
    a claimed overs_completed=3 is not a state that can exist, and letting it
    through would feed wicket_loss_rate the wrong denominator with no error."""
    return (balls_faced - 1) // 6


def _prediction_context(is_chasing, balls_faced, features_row):
    """The numbers behind the headline percentage, straight off the row
    build_features_single produced - not recomputed, so this can't drift from
    what the model actually saw. balls_remaining isn't a model feature (see
    FEATURES in src/features.py), so it's the one value derived here, the same
    way overs_completed is above."""
    if is_chasing:
        return {
            "runs_needed": int(round(features_row["runs_needed"])),
            "balls_remaining": max(0, BALLS_PER_INNINGS - balls_faced),
            "required_run_rate": round(float(features_row["run_rate_needed"]), 2),
            "current_run_rate": round(float(features_row["run_rate"]), 2),
        }
    return {
        "projected_score": int(round(features_row["projected_final_score"])),
        "runs_vs_par": int(round(features_row["runs_vs_par"])),
    }


class MatchState(BaseModel):
    innings_number: int = Field(..., ge=1, le=2)
    balls_faced: int = Field(..., ge=0, le=120)
    cumulative_runs: int = Field(..., ge=0)
    cumulative_wickets: int = Field(..., ge=0, le=10)
    batting_team: str
    venue: str
    target_score: Optional[float] = None

    @field_validator("target_score")
    @classmethod
    def target_score_must_be_finite_and_positive(cls, v):
        # JSON's NaN/Infinity literals parse straight through Python's json
        # module. NaN in particular would otherwise survive exclude_none and
        # the `is None` check in build_features_single, landing as
        # runs_needed=0 - a missing target read as a near-certain win.
        if v is not None and (not math.isfinite(v) or v <= 0):
            raise ValueError("target_score must be a finite number greater than 0")
        return v


@app.get("/", include_in_schema=False)
def index():
    """Serves the frontend from the same origin as the API - no CORS
    middleware needed, because there's no cross-origin request to allow."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "best_iteration": metadata["best_iteration"],
        "features": len(metadata["features"]),
    }


@app.get("/options")
def options():
    """Team and venue choices, straight from the encoders that will actually
    handle them. A hardcoded list can drift from what training saw; anything
    not among these keys maps to -1 and predicts against the wrong team or
    venue with no error, so the dropdowns can only offer what's really here."""
    return {
        "teams": sorted(encoders["batting_team"].keys()),
        "venues": sorted(encoders["venue"].keys()),
    }


def _terminal_chase_result(state: MatchState):
    """None if the chase is still live; otherwise the arithmetic result of one
    that already isn't - the model was trained on live states only and will
    still return a plausible-looking number for a match that's already over.

    Only decidable once a target exists, so a missing target falls through to
    the usual "target_score is required" error instead of being guessed at
    here."""
    if state.target_score is None:
        return None

    if state.cumulative_runs > state.target_score:
        return 1.0, f"{state.batting_team} passed the target of {state.target_score:g}"

    if state.cumulative_wickets == 10:
        return 0.0, f"{state.batting_team} was all out short of the target of {state.target_score:g}"

    if state.balls_faced == 120:
        return (
            0.0,
            f"{state.batting_team} did not reach the target of {state.target_score:g} "
            "within 120 balls",
        )

    return None


@app.post("/predict")
def predict(state: MatchState):
    is_chasing = state.innings_number == 2

    if is_chasing:
        terminal = _terminal_chase_result(state)
        if terminal is not None:
            win_probability, reason = terminal
            return {
                "win_probability": win_probability,
                "batting_team": state.batting_team,
                "is_chasing": True,
                "match_state": "resolved",
                "confidence_note": f"{reason} - result is arithmetic, not a model prediction.",
                "context": None,  # no model call behind an arithmetic result
            }

    payload = state.model_dump(exclude_none=True)
    payload["overs_completed"] = overs_completed_from_balls(state.balls_faced)

    try:
        X = build_features_single(payload, par_table)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    context = _prediction_context(is_chasing, state.balls_faced, X.iloc[0])

    X = encode_single(X, encoders)
    win_probability = float(model.predict_proba(X)[:, 1][0])

    confidence_note = None
    if is_chasing and state.balls_faced < EARLY_CHASE_BALL:
        confidence_note = (
            f"Early-chase prediction (innings 2, before ball {EARLY_CHASE_BALL}): "
            f"measured at {EARLY_CHASE_AUC:.3f} AUC, against {OVERALL_AUC:.3f} overall."
        )

    # The first innings ending doesn't resolve the match - the second innings
    # still has to be played - so this keeps predicting rather than stopping,
    # just flagged as no longer a live delivery.
    match_state = "live"
    if not is_chasing and (state.cumulative_wickets == 10 or state.balls_faced == 120):
        match_state = "innings_complete"

    return {
        "win_probability": win_probability,
        "batting_team": state.batting_team,
        "is_chasing": is_chasing,
        "match_state": match_state,
        "confidence_note": confidence_note,
        "context": context,
    }


if __name__ == "__main__":
    import uvicorn

    # Render assigns the port at runtime via $PORT rather than a fixed one,
    # and 127.0.0.1 only accepts connections from inside the container -
    # 0.0.0.0 is what makes the service reachable from outside it.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
