"""
Elo-style power rating model, built off EPA/play and success rate from the
prior completed season's play-by-play (see nfl_data.fetch_pbp).

Methodology (documented in full since "Elo-style" can mean a lot of
different things - this is the specific version used here):

1. Per team, per side of the ball: mean EPA/play and mean success rate on
   offense (posteam) and on defense (defteam, i.e. EPA/success rate
   *allowed*), over regular-season rush/pass plays only (the standard
   "meaningful play" filter used across the public EPA-analytics community -
   excludes kneels, spikes, kicks, and no-plays, which don't reflect team
   quality the same way a called run/pass does).
2. Net EPA = offense EPA/play - defense EPA/play allowed (higher is
   better). Same for net success rate. Both z-scored across the 32 teams so
   they're on a comparable scale, then blended 70/30 (EPA/play, success
   rate) - EPA is the more predictive of the two on its own in most public
   research, success rate adds a consistency signal EPA's magnitude-focus
   can miss (one explosive play vs. a string of efficient ones). The 70/30
   split is a documented starting assumption, not a fitted value - revisit
   once Phase 3's historical calibration check is built.
3. Fit a single logistic model on that same season's actual game results:
   P(home team wins) = sigmoid(b0 + b1 * (home_composite - away_composite)).
   b0 and b1 come from maximum-likelihood fit on real outcomes (scipy), not
   assumed - this is what makes it "Elo-style" in the way that matters: a
   logistic function of a rating gap, exactly Elo's win-probability formula,
   except the coefficients are fit to real data instead of borrowed from
   chess's arbitrary 400-point convention. b0 also absorbs home-field
   advantage automatically, since it's fit on real home/away results.
4. Regress composite ratings toward the league mean (0, since they're
   z-scored) before using them for the *next* season's Week 1 - a team's
   final-season rating overstates how good they'll be next year (roster
   turnover, injury regression, etc.). Shrinkage factor 0.72 is a documented
   starting assumption (public research on year-over-year EPA persistence
   typically lands in the 0.6-0.75 range) - same caveat as the 70/30 blend
   above: revisit with real calibration data once there's enough of it.

This module intentionally does NOT re-fit b0/b1 after regression - the
logistic relationship between rating gap and win probability is assumed
stable, only the ratings feeding into it shrink. Re-fitting weekly as 2026
results accumulate is Phase 2's job (fresh in-season EPA + game outcomes).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

MEANINGFUL_PLAYS = {"pass", "run"}
EPA_WEIGHT = 0.7
SUCCESS_WEIGHT = 0.3
PRESEASON_SHRINKAGE = 0.72  # fraction of last season's rating gap from the mean that carries over


def team_epa_success(pbp_df):
    """Per-team offense/defense EPA-per-play and success rate, REG season,
    rush/pass plays only. Returns a DataFrame indexed by team."""
    df = pbp_df[
        (pbp_df["season_type"] == "REG")
        & (pbp_df["play_type"].isin(MEANINGFUL_PLAYS))
        & pbp_df["epa"].notna()
    ]

    off = df.groupby("posteam").agg(off_epa=("epa", "mean"), off_success=("success", "mean"))
    dfn = df.groupby("defteam").agg(def_epa=("epa", "mean"), def_success=("success", "mean"))

    teams = sorted(set(off.index) | set(dfn.index))
    out = pd.DataFrame(index=teams)
    out = out.join(off).join(dfn)
    return out


def _zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def composite_ratings(pbp_df):
    """Raw (unshrunk) composite team ratings for the season pbp_df covers.
    Positive = above average, negative = below average, in z-score units."""
    stats = team_epa_success(pbp_df)
    net_epa = stats["off_epa"] - stats["def_epa"]
    net_success = stats["off_success"] - stats["def_success"]
    composite = EPA_WEIGHT * _zscore(net_epa) + SUCCESS_WEIGHT * _zscore(net_success)
    composite.name = "composite"
    return composite


def fit_logistic(composite, season_df):
    """
    Fits P(home wins) = sigmoid(b0 + b1 * (home_composite - away_composite))
    by maximum likelihood on season_df's actual REG-season results (games
    with a final score only). Returns (b0, b1).
    """
    reg = season_df[
        (season_df["game_type"] == "REG") & season_df["home_score"].notna()
    ].copy()
    reg = reg[reg["home_score"] != reg["away_score"]]  # NFL ties are rare and uninformative for a binary fit
    reg["home_win"] = (reg["home_score"] > reg["away_score"]).astype(float)
    reg = reg[reg["home_team"].isin(composite.index) & reg["away_team"].isin(composite.index)]

    diff = (composite.loc[reg["home_team"]].values - composite.loc[reg["away_team"]].values)
    y = reg["home_win"].values

    def neg_log_likelihood(params):
        b0, b1 = params
        z = b0 + b1 * diff
        # log-sum-exp-stable log-sigmoid
        log_p = -np.logaddexp(0, -z)
        log_1mp = -np.logaddexp(0, z)
        return -np.sum(y * log_p + (1 - y) * log_1mp)

    result = minimize(neg_log_likelihood, x0=[0.1, 1.0], method="Nelder-Mead")
    if not result.success:
        raise RuntimeError(f"Logistic fit did not converge: {result.message}")
    b0, b1 = result.x
    return float(b0), float(b1)


def shrink_toward_mean(composite, factor=PRESEASON_SHRINKAGE):
    """Preseason regression to the mean - see module docstring point 4."""
    return composite * factor


def win_probability(home_composite, away_composite, b0, b1):
    z = b0 + b1 * (home_composite - away_composite)
    return 1.0 / (1.0 + np.exp(-z))


def build_preseason_ratings(pbp_df, season_df):
    """
    End-to-end: raw composite ratings + logistic fit from last season's real
    data, then shrunk toward the mean for use as this season's Week 1
    ratings. Returns (shrunk_composite: Series indexed by team, b0, b1).
    """
    raw = composite_ratings(pbp_df)
    b0, b1 = fit_logistic(raw, season_df)
    shrunk = shrink_toward_mean(raw)
    return shrunk, b0, b1


if __name__ == "__main__":
    import nfl_data

    sched = nfl_data.fetch_schedule()
    rating_season = nfl_data.latest_completed_season(sched)
    pbp = nfl_data.fetch_pbp(rating_season)
    season_df = nfl_data.season_schedule(sched, rating_season)

    raw = composite_ratings(pbp)
    b0, b1 = fit_logistic(raw, season_df)
    print(f"Logistic fit on {rating_season}: b0={b0:.4f} (home-field log-odds), b1={b1:.4f} (sensitivity to rating gap)")
    print(f"Implied home-field win prob for two evenly-matched teams: {win_probability(0, 0, b0, b1):.1%}")

    shrunk = shrink_toward_mean(raw)
    ranked = shrunk.sort_values(ascending=False)
    print(f"\nTop 10 teams (shrunk composite rating, for next season's Week 1):")
    for team, val in ranked.head(10).items():
        print(f"  {team:>4}  {val:+.3f}")
    print(f"\nBottom 5:")
    for team, val in ranked.tail(5).items():
        print(f"  {team:>4}  {val:+.3f}")
