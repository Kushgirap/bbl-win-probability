"""
Leverage analysis for the BBL win probability model.

Leverage is the change in win probability caused by a single delivery. It turns
"that spell cost us the match" from a judgement into a measurement, and lets you
rank every passage of play by how much it actually moved the result.

Sign convention throughout: POSITIVE means the batting side gained, NEGATIVE
means the bowling side gained. Fixed to one reference team so the sign stays
meaningful across the innings break.
"""

import numpy as np
import pandas as pd


def compute_leverage(match_states, reference_team=None):
    """
    Win probability change per delivery, from one team's perspective.

    match_states : rows for ONE match_id, with `proba` attached. Needs columns
                   innings_number, balls_faced, batting_team, cumulative_runs,
                   cumulative_wickets, and ideally batter / bowler.
    reference_team : whose perspective. Defaults to the side batting first.

    Returns the frame with `wp` (reference team win probability) and `leverage`
    (change caused by that ball) added.

    The model outputs the BATTING side's probability, and the batting side
    changes at the innings break - so a raw diff across that boundary would be
    nonsense. Everything is flipped to the reference team first.
    """
    m = match_states.sort_values(["innings_number", "balls_faced"]).copy()

    inn1 = m[m["innings_number"] == 1]
    inn2 = m[m["innings_number"] == 2]
    if len(inn1) == 0 or len(inn2) == 0:
        raise ValueError("Need both innings to compute leverage")

    team1 = inn1["batting_team"].iloc[0]
    ref = reference_team or team1

    # Flip to reference team's perspective
    m["wp"] = np.where(m["batting_team"] == ref, m["proba"], 1 - m["proba"])

    # Continuous ball index across the match
    m["ball_index"] = m["balls_faced"] + (m["innings_number"] - 1) * 120
    m = m.sort_values("ball_index")

    # Leverage = change from the previous delivery.
    m["leverage"] = m["wp"].diff().fillna(0)

    # The first ball of each innings gets 0. Ball one of the match has no prior
    # state. Ball one of the chase is more subtle: at the break the model
    # switches framing - the target becomes known, is_chasing flips, runs_needed
    # appears - so the probability re-anchors for reasons that have nothing to
    # do with that delivery. Attributing that jump to the bowler would be wrong,
    # so it is stored separately as a diagnostic instead.
    first_balls = m.groupby("innings_number")["ball_index"].transform("min")
    break_jump = m.loc[
        (m["ball_index"] == first_balls) & (m["innings_number"] == 2), "leverage"
    ]
    m.attrs["innings_break_jump"] = (
        float(break_jump.iloc[0]) if len(break_jump) else 0.0
    )
    m.loc[m["ball_index"] == first_balls, "leverage"] = 0.0

    # Flag wickets for readability
    m["is_wicket"] = m["cumulative_wickets"].diff().fillna(0) > 0

    m.attrs["reference_team"] = ref
    return m


def top_deliveries(lev, n=10):
    """The n deliveries that moved the match most, by absolute leverage."""
    cols = [c for c in ["innings_number", "balls_faced", "batting_team", "bowler",
                        "batter", "cumulative_runs", "cumulative_wickets",
                        "is_wicket", "wp", "leverage"] if c in lev.columns]

    out = lev.reindex(lev["leverage"].abs().sort_values(ascending=False).index)
    return out[cols].head(n)


def leverage_by_over(lev):
    """
    Aggregate to over level. `net` is the total swing; `swing` is total absolute
    movement, which catches volatile overs that ended up roughly neutral.
    """
    g = lev.copy()
    g["over"] = g["balls_faced"].apply(lambda b: (b - 1) // 6 + 1)

    agg = {
        "leverage": [("net", "sum"), ("swing", lambda s: s.abs().sum())],
        "cumulative_runs": [("runs_end", "max")],
        "is_wicket": [("wickets", "sum")],
    }

    out = (g.groupby(["innings_number", "over"])
            .agg(net=("leverage", "sum"),
                 swing=("leverage", lambda s: s.abs().sum()),
                 runs_end=("cumulative_runs", "max"),
                 wickets=("is_wicket", "sum"))
            .round(4)
            .reset_index())

    if "bowler" in g.columns:
        bowlers = (g.groupby(["innings_number", "over"])["bowler"]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
                    .reset_index())
        out = out.merge(bowlers, on=["innings_number", "over"], how="left")

    return out.sort_values("net")


def leverage_by_bowler(lev):
    """
    Total win probability conceded or saved per bowler.

    Sign flips deliberately: leverage is positive when the BATTING side gains,
    so a bowler's contribution is the negative of the leverage during their
    deliveries. Positive `wp_saved` means the bowler helped their own side.
    """
    if "bowler" not in lev.columns:
        raise ValueError("No bowler column - add it to extract_match_state")

    out = (lev.groupby(["innings_number", "bowler"])
             .agg(balls=("leverage", "size"),
                  wp_saved=("leverage", lambda s: -s.sum()),
                  runs_conceded_swing=("leverage", lambda s: s.abs().sum()),
                  wickets=("is_wicket", "sum"))
             .round(4)
             .reset_index())

    out["wp_per_ball"] = (out["wp_saved"] / out["balls"]).round(5)
    return out.sort_values("wp_saved")


def match_summary(lev, top_n=5):
    """Printable summary of what decided the match."""
    ref = lev.attrs.get("reference_team", "reference team")
    lines = []

    final = lev["wp"].iloc[-1]
    jump = lev.attrs.get("innings_break_jump", 0.0)
    lines.append(f"Reference: {ref}   final win probability {final:.1%}")
    lines.append(
        f"Innings break re-anchor: {jump:+.1%} "
        f"(model reframing, not attributed to any delivery)"
    )
    lines.append("")
    lines.append(f"Biggest {top_n} swings:")

    for _, r in top_deliveries(lev, top_n).iterrows():
        over = (r["balls_faced"] - 1) // 6 + 1
        ball = (r["balls_faced"] - 1) % 6 + 1
        who = f" {r['bowler']} to {r['batter']}" if "bowler" in r else ""
        mark = " WICKET" if r.get("is_wicket") else ""
        lines.append(
            f"  Inn {int(r['innings_number'])}  {over:>2}.{ball}"
            f"  {int(r['cumulative_runs']):>3}/{int(r['cumulative_wickets'])}"
            f"{who}{mark}   {r['leverage']:+.1%}"
        )

    return "\n".join(lines)
