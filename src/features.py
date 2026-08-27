"""
Feature engineering for the BBL win probability model.

Single source of truth: training and inference both call `build_features`, so the
two paths cannot drift apart. Recomputing features separately at serving time is
the most common way these systems fail - they don't crash, they just quietly get
worse as the two implementations diverge.

Features split into three groups by what they need:

  1. State      - row-wise only. Safe anywhere, no leakage possible.
  2. Target     - needs the first-innings total, so it groups within a match.
                  Attached to the chasing innings only.
  3. Par score  - LEARNED from data, so it must be fitted on training matches
                  only and then applied. Hence the fit/apply split below.

The par table is the reason this module has a fit step at all. Computing medians
across the full dataset would leak information about the test period into the
training features.
"""

import numpy as np
import pandas as pd

BALLS_PER_INNINGS = 120  # T20

POWERPLAY_BALLS = 36
MIDDLE_BALLS = 84

# Column order used by the model. Kept here so training and inference agree.
# Columns fed to the model.
#
# balls_remaining, overs_remaining and phase are deliberately NOT here. All three
# are exact transforms of balls_faced, so they carry no information the model
# doesn't already have - XGBoost splits on whichever it meets first and the rest
# score zero importance. They are still computed below, because other features
# depend on them and the by-phase analysis needs them.
FEATURES = [
    # state
    "cumulative_runs", "cumulative_wickets", "balls_faced", "wickets_remaining",
    # rate and efficiency
    "run_rate", "runs_per_wicket", "projected_final_score",
    # pressure
    "wicket_loss_rate", "wicket_pressure", "runs_per_ball_ratio",
    "efficiency_in_death",
    # context
    "is_chasing", "runs_needed", "run_rate_needed", "runs_vs_par",
    # categorical
    "batting_team", "venue",
]

CATEGORICAL = ["batting_team", "venue"]


def assign_phase(balls):
    """Powerplay / middle / death by ball number."""
    if balls <= POWERPLAY_BALLS:
        return "powerplay"
    if balls <= MIDDLE_BALLS:
        return "middle"
    return "death"


def add_state_features(df):
    """
    Row-wise match state. No cross-row information, so this is leakage-safe
    and can be applied to any split.
    """
    df = df.copy()

    # Rate
        # clip guards the opening-wide case: a wide on the first delivery creates a
    # row with balls_faced = 0, which sends run_rate (and everything derived
    # from it) to inf
    df["run_rate"] = (df["cumulative_runs"] / df["balls_faced"].clip(lower=1) * 6).round(2)
    df["balls_remaining"] = (BALLS_PER_INNINGS - df["balls_faced"]).clip(lower=0)
    df["overs_remaining"] = df["balls_remaining"] / 6
    df["wickets_remaining"] = 10 - df["cumulative_wickets"]

    # Efficiency. +1 in the denominator keeps a wicketless innings finite.
    df["runs_per_wicket"] = df["cumulative_runs"] / (df["cumulative_wickets"] + 1)
    df["projected_final_score"] = (
        df["cumulative_runs"] + df["run_rate"] * df["balls_remaining"] / 6
    )

    # Pressure. runs_per_ball_ratio is scoring rate against wicket-loss rate -
    # it separates a side scoring freely from one scoring fast but losing wickets.
    df["wicket_loss_rate"] = df["cumulative_wickets"] / (df["overs_completed"] + 1)
    df["wicket_pressure"] = df["wicket_loss_rate"] * df["balls_remaining"]
    df["runs_per_ball_ratio"] = df["run_rate"] / (df["wicket_loss_rate"] + 0.01)

    # Phase
    df["phase"] = df["balls_faced"].apply(assign_phase)
    df["efficiency_in_death"] = df["runs_per_wicket"] * (df["phase"] == "death")

    return df


def add_target_features(df):
    """
    Chasing context. The target is the first innings total, attached to the
    second innings only.

    Giving the first-innings side its own final total would hand the model the
    outcome mid-innings, so target_score is left NaN there and the derived
    columns fall back to 0.
    """
    df = df.copy()

    first_innings_totals = (
        df[df["innings_number"] == 1]
        .groupby("match_id")["cumulative_runs"]
        .max()
        .rename("target_score")
        .reset_index()
    )

    df = df.merge(first_innings_totals, on="match_id", how="left")
    df.loc[df["innings_number"] == 1, "target_score"] = np.nan

    df["is_chasing"] = (df["innings_number"] == 2).astype(int)
    df["runs_needed"] = (
        (df["target_score"] - df["cumulative_runs"]).fillna(0).clip(lower=0)
    )
    # clip(lower=1) guards the denominator - a rain-shortened innings can leave
    # balls_remaining at 0, which would produce inf and crash XGBoost.
    df["run_rate_needed"] = (
        df["runs_needed"] / df["balls_remaining"].clip(lower=1) * 6
    ).fillna(0)

    return df


def fit_par_table(train_df):
    """
    Learn par scores from TRAINING matches only.

    Par is the median first-innings score at a given venue and ball number -
    the reference point a setting side has in place of a target. Keyed on ball
    number rather than phase, since ball 37 and ball 84 are both "middle" but
    are not comparable positions.

    Returns (by_venue, global) tables to hand to `add_par_features`.
    """
    first_innings = train_df[train_df["innings_number"] == 1]

    par_by_venue = (
        first_innings.groupby(["venue", "balls_faced"])["cumulative_runs"]
        .median()
        .rename("par_venue")
        .reset_index()
    )

    par_global = (
        first_innings.groupby("balls_faced")["cumulative_runs"]
        .median()
        .rename("par_global")
        .reset_index()
    )

    return {"by_venue": par_by_venue, "global": par_global}


def add_par_features(df, par_table):
    """
    Apply a fitted par table. Venues absent from training fall back to the
    global median for that ball number.
    """
    df = df.copy()

    df = df.merge(par_table["by_venue"], on=["venue", "balls_faced"], how="left")
    df = df.merge(par_table["global"], on="balls_faced", how="left")

    df["par_score"] = df["par_venue"].fillna(df["par_global"])
    df["runs_vs_par"] = df["cumulative_runs"] - df["par_score"]

    return df.drop(columns=["par_venue", "par_global"])


def build_features(df, par_table):
    """
    Full pipeline. Call with a par table fitted on training data.

        par_table = fit_par_table(train_raw)
        train = build_features(train_raw, par_table)
        test  = build_features(test_raw,  par_table)
    """
    df = add_state_features(df)
    df = add_target_features(df)
    df = add_par_features(df, par_table)
    return df


def check_finite(X, name="X"):
    """
    Fail loudly on inf/NaN, at the point it appears rather than deep inside
    XGBoost's C++ layer where the error message is useless.
    """
    numeric = X.select_dtypes(include=[np.number])
    bad = ~np.isfinite(numeric)
    if bad.any().any():
        cols = bad.any()
        raise ValueError(
            f"{name} contains inf or NaN in: {cols[cols].index.tolist()}"
        )
    return True
