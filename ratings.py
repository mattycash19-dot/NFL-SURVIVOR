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

# Phase 3 finding (calibration.py, 2026-09-09): fit_logistic() below fits
# b0/b1 against the SAME season the ratings are computed from, which
# systematically overstates the rating-to-win-probability relationship - a
# team's own EPA numbers in a season are partly a consequence of that
# season's own wins (winning teams see more favorable game states,
# garbage-time effects on the losing side, etc.), so the in-sample fit
# doesn't transfer to predicting a DIFFERENT season. Checked against real
# held-out seasons (2019-2025), the in-sample fit was badly overconfident -
# predicted 80%+ buckets actually won only ~65-67% of the time. These two
# constants instead come from calibration.fit_out_of_sample_logistic(),
# pooling (shrunk prior-season rating gap, actual NEXT-season outcome)
# pairs across season transitions - genuinely out-of-sample by construction.
# Held-out validation (fit on an earlier window, tested on 2019-2025):
# Brier 0.2408 -> 0.2388, and the worst overconfidence band (80-90%
# predicted, ~65% actual) closed up substantially.
#
# 2026-09-10: dropped the pre-2016 transitions from the fit. The 2010-2015
# NFL had a genuinely larger home-field advantage (raw home win rate 56.7%
# for 2010-2019 vs 53.4% since 2020), and including it pulled b0 up to an
# even-matchup home win probability of ~55.5% - toward the high end of the
# modern game. Refit on 2016->2017 through 2024->2025 (n=2327): b0 0.2203
# -> 0.1722 (even-matchup home win prob ~54.3%), b1 0.4589 -> 0.4712.
# Held-out check on 2020-2025 (1,610 games): Brier 0.2382 -> 0.2380, and
# the 50-60% buckets (where most games and most survivor picks live)
# tightened - production predicted 57.3% / actual 55.4%; the recent fit
# predicts 57.4% / actual 56.9%. No bucket got worse.
#
# Still an honest, real limitation worth stating plainly: even properly
# calibrated, a pure preseason EPA-only rating has limited power to predict
# a DIFFERENT season on its own (Brier ~0.238 vs. the market's ~0.211 over
# the same kind of test - see calibration.py) - rosters change materially
# year over year in ways last season's play-by-play alone can't see. Trust
# these preseason/Week-1 numbers with real skepticism; build_inseason_
# ratings() should get better as current-season data starts dominating the
# blend. Re-run calibration.fit_out_of_sample_logistic() periodically as
# more seasons accumulate; keep the ~10-season trailing window so the fit
# tracks the current game rather than averaging in a different era.
FITTED_B0 = 0.1722
FITTED_B1 = 0.4712


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
    return fit_logistic_on_diffs(diff, y)


def fit_logistic_on_diffs(diff, y):
    """
    The actual MLE fit, split out from fit_logistic() so calibration.py can
    fit b0/b1 on a genuinely out-of-sample pool of (rating gap, outcome)
    pairs - see calibration.py's module docstring for why the in-sample
    version (fit_logistic() above, rating and outcome both from the same
    season) turned out to systematically overstate b1 and produce
    overconfident predictions once checked against real held-out seasons.
    """
    diff = np.asarray(diff)
    y = np.asarray(y)

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


def build_preseason_ratings(pbp_df, season_df=None):
    """
    End-to-end: raw composite ratings from last season's real data, shrunk
    toward the mean for use as this season's Week 1 ratings, paired with
    the historically-validated FITTED_B0/FITTED_B1 (not an in-sample fit
    against `season_df` - see the module docstring above FITTED_B0 for why
    that was wrong). `season_df` is accepted but unused, kept only so
    existing callers don't need updating; pass None for new code.
    Returns (shrunk_composite: Series indexed by team, b0, b1).
    """
    raw = composite_ratings(pbp_df)
    shrunk = shrink_toward_mean(raw)
    return shrunk, FITTED_B0, FITTED_B1


def build_inseason_ratings(schedule_df, current_season, nfl_data_mod):
    """
    Phase 2: ratings that actually update from the current season's own
    results, not just last season's preseason prior. Blends the shrunk
    preseason prior with a composite computed from this season's own
    play-by-play so far, weighted by how much of the season has been
    played - 0% current-season weight before Week 1 kicks off (identical
    to build_preseason_ratings), ramping to fully trusting current-season
    data once teams have played ~6 games each (a documented starting
    assumption - the point at which a single season's sample starts
    meaningfully outweighing last year's, not a fitted value). Falls back
    cleanly to the pure preseason prior if no games are final yet, or if
    nflverse hasn't published a current-season pbp file yet (both true
    before Week 1).

    `nfl_data_mod` is passed in (rather than imported here) to avoid a
    circular import - ratings.py is also usable standalone.
    """
    prior_season = current_season - 1
    pbp_prior = nfl_data_mod.fetch_pbp(prior_season)
    prior_raw = composite_ratings(pbp_prior)
    b0, b1 = FITTED_B0, FITTED_B1  # see module docstring above FITTED_B0 - not an in-sample fit
    prior_shrunk = shrink_toward_mean(prior_raw)

    current_reg = nfl_data_mod.season_schedule(schedule_df, current_season)
    played = current_reg[current_reg["home_score"].notna()]
    if played.empty:
        return prior_shrunk, b0, b1, 0.0  # pure preseason - no current-season games final yet

    try:
        pbp_current = nfl_data_mod.fetch_pbp(current_season, force_refresh=True)
    except Exception:
        return prior_shrunk, b0, b1, 0.0  # nflverse hasn't published this season's pbp file yet

    avg_games_per_team = 2 * len(played) / 32.0
    weight_current = min(1.0, avg_games_per_team / 6.0)

    current_raw = composite_ratings(pbp_current)
    blended = weight_current * current_raw.reindex(prior_shrunk.index) + (1 - weight_current) * prior_shrunk
    blended = blended.fillna(prior_shrunk)  # teams with zero current-season plays so far (bye, etc.) keep the prior
    blended.name = "composite"
    return blended, b0, b1, weight_current


if __name__ == "__main__":
    import nfl_data

    sched = nfl_data.fetch_schedule()
    rating_season = nfl_data.latest_completed_season(sched)
    pbp = nfl_data.fetch_pbp(rating_season)
    season_df = nfl_data.season_schedule(sched, rating_season)

    raw = composite_ratings(pbp)
    b0, b1 = fit_logistic(raw, season_df)
    print(f"In-sample fit on {rating_season} (diagnostic only, NOT used for real predictions - see FITTED_B0/B1 "
          f"above): b0={b0:.4f}, b1={b1:.4f}")
    print(f"Actual production fit (calibration.fit_out_of_sample_logistic, validated out-of-sample): "
          f"b0={FITTED_B0:.4f}, b1={FITTED_B1:.4f}")
    print(f"Implied home-field win prob for two evenly-matched teams (production fit): "
          f"{win_probability(0, 0, FITTED_B0, FITTED_B1):.1%}")

    shrunk = shrink_toward_mean(raw)
    ranked = shrunk.sort_values(ascending=False)
    print(f"\nTop 10 teams (shrunk composite rating, for next season's Week 1):")
    for team, val in ranked.head(10).items():
        print(f"  {team:>4}  {val:+.3f}")
    print(f"\nBottom 5:")
    for team, val in ranked.tail(5).items():
        print(f"  {team:>4}  {val:+.3f}")
