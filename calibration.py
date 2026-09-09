"""
Phase 3, part 1: does "win probability" actually mean what it claims to,
for both the market and this project's own model? Two separate checks:

1. historical_market_calibration() - devigs closing moneylines for every
   historical game (nflverse's schedule file carries these back to 1999,
   reliably populated from 2010 onward - no extra fetch needed) and checks
   whether the favorite's implied win probability matches the favorite's
   actual straight-up win rate. This is the "should the market cross-check
   in risk.py actually be trusted" question, tested rather than assumed.

2. historical_model_calibration() - the harder, more important check: for
   each season in `test_seasons`, builds preseason ratings from ONLY the
   prior season's play-by-play (ratings.build_preseason_ratings - the
   exact function Phase 1 uses for real), then scores those probabilities
   against that season's real outcomes. This is genuinely out-of-sample
   (build_preseason_ratings' own internal logistic fit is in-sample by
   construction - it fits b0/b1 on the same season it computes ratings
   from - so it can't answer "does this actually predict the NEXT season,"
   only this one). Pooling several seasons gives an honest read on whether
   the whole Phase 1 methodology - including the 70/30 EPA/success blend
   and 0.72 shrinkage factor, both flagged in ratings.py as unfitted
   starting assumptions - holds up against real results.

Both report Brier score (lower is better; 0.25 is the "always guess 50%"
baseline) and calibration buckets (does the ~70% bucket actually win ~70%
of the time?), always bucketed on the FAVORITE's probability (>=50%) so
"favorite" and "underdog" rows of the same game aren't double-counted as
if independent.
"""
import numpy as np
import pandas as pd

import odds_data
import ratings as ratings_mod

MARKET_CALIBRATION_MIN_SEASON = 2010  # home_moneyline is ~100% populated from here on (checked: <90% in several earlier seasons)


def _bucket_calibration(probs, outcomes, bin_width=0.05, lo=0.5, hi=1.0):
    probs = np.asarray(probs)
    outcomes = np.asarray(outcomes)
    edges = np.arange(lo, hi + 1e-9, bin_width)
    buckets = []
    for i in range(len(edges) - 1):
        lo_e, hi_e = edges[i], edges[i + 1]
        mask = (probs >= lo_e) & (probs < hi_e + 1e-9 if hi_e >= hi - 1e-9 else probs < hi_e)
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "range": f"{lo_e * 100:.0f}-{hi_e * 100:.0f}%",
            "n": n,
            "avg_predicted": round(float(probs[mask].mean()), 3),
            "actual_win_rate": round(float(outcomes[mask].mean()), 3),
        })
    return buckets


def _favorite_view(home_probs, home_wins):
    """Folds (home_prob, home_won) pairs into (favorite_prob,
    favorite_won) so each game contributes exactly one row, always framed
    as "did the side given >=50% actually win" - avoids double-counting a
    game as two independent home/away observations."""
    home_probs = np.asarray(home_probs)
    home_wins = np.asarray(home_wins)
    fav_is_home = home_probs >= 0.5
    fav_prob = np.where(fav_is_home, home_probs, 1 - home_probs)
    fav_won = np.where(fav_is_home, home_wins, 1 - home_wins)
    return fav_prob, fav_won


def historical_market_calibration(schedule_df, min_season=MARKET_CALIBRATION_MIN_SEASON, max_season=None):
    reg = schedule_df[(schedule_df["game_type"] == "REG") & (schedule_df["season"] >= min_season)]
    if max_season:
        reg = reg[reg["season"] <= max_season]
    reg = reg[
        reg["home_score"].notna() & reg["home_moneyline"].notna() & reg["away_moneyline"].notna()
    ]
    reg = reg[reg["home_score"] != reg["away_score"]]  # ties are rare and uninformative here

    home_probs, home_wins = [], []
    for _, g in reg.iterrows():
        raw_h = odds_data.moneyline_to_implied_prob(g["home_moneyline"])
        raw_a = odds_data.moneyline_to_implied_prob(g["away_moneyline"])
        h, _ = odds_data.devig_two_way(raw_h, raw_a)
        home_probs.append(h)
        home_wins.append(1.0 if g["home_score"] > g["away_score"] else 0.0)

    fav_prob, fav_won = _favorite_view(home_probs, home_wins)
    return {
        "n": len(fav_prob),
        "seasons": f"{min_season}-{int(reg['season'].max())}",
        "brier_score": round(float(np.mean((fav_prob - fav_won) ** 2)), 4),
        "buckets": _bucket_calibration(fav_prob, fav_won),
    }


def fit_out_of_sample_logistic(schedule_df, nfl_data_mod, train_seasons):
    """
    Fits b0/b1 by pooling (shrunk prior-season rating gap, actual
    NEXT-season outcome) pairs across many season transitions - e.g.
    2010's shrunk ratings vs. 2011's real results, 2011's vs. 2012's, and
    so on through `train_seasons`. This is what fit_logistic() should have
    been doing all along: the original version fit b0/b1 against the SAME
    season the ratings were computed from, which mildly overstates the
    rating-to-win-probability relationship (a team's own EPA numbers this
    season are partly a consequence of that season's own wins - winning
    teams see more favorable game states, garbage-time effects on the
    losing side, etc.) - real, not a rounding error, confirmed by
    historical_model_calibration() showing the in-sample-fit version
    badly overconfident (predicted 80%+ buckets winning only ~65-67% of
    the time) once checked against real held-out seasons. `train_seasons`
    must not overlap whatever seasons you evaluate calibration against
    afterward, or the "out-of-sample" claim is false.
    """
    all_diff, all_y = [], []
    for season in train_seasons:
        pbp = nfl_data_mod.fetch_pbp(season)
        raw = ratings_mod.composite_ratings(pbp)
        shrunk = ratings_mod.shrink_toward_mean(raw)

        next_season_df = nfl_data_mod.season_schedule(schedule_df, season + 1)
        next_season_df = next_season_df[next_season_df["home_score"].notna()]
        next_season_df = next_season_df[next_season_df["home_score"] != next_season_df["away_score"]]
        next_season_df = next_season_df[
            next_season_df["home_team"].isin(shrunk.index) & next_season_df["away_team"].isin(shrunk.index)
        ]
        if next_season_df.empty:
            continue
        diff = (shrunk.loc[next_season_df["home_team"]].values - shrunk.loc[next_season_df["away_team"]].values)
        y = (next_season_df["home_score"] > next_season_df["away_score"]).astype(float).values
        all_diff.extend(diff)
        all_y.extend(y)

    b0, b1 = ratings_mod.fit_logistic_on_diffs(np.array(all_diff), np.array(all_y))
    return b0, b1, len(all_diff)


def historical_model_calibration(schedule_df, nfl_data_mod, test_seasons, fixed_b0b1=None):
    """
    fixed_b0b1: optional (b0, b1) to use for every test season instead of
    each season's own in-sample fit_logistic() - pass the output of
    fit_out_of_sample_logistic() here to check genuinely out-of-sample
    calibration instead of the flawed in-sample version. See that
    function's docstring.
    """
    all_fav_prob, all_fav_won = [], []
    per_season = []

    for season in test_seasons:
        prior_season = season - 1
        try:
            pbp_prior = nfl_data_mod.fetch_pbp(prior_season)
        except Exception as e:
            per_season.append({"season": season, "n": 0, "skipped": str(e)})
            continue
        schedule_prior = nfl_data_mod.season_schedule(schedule_df, prior_season)
        if fixed_b0b1 is not None:
            raw = ratings_mod.composite_ratings(pbp_prior)
            team_ratings = ratings_mod.shrink_toward_mean(raw)
            b0, b1 = fixed_b0b1
        else:
            team_ratings, b0, b1 = ratings_mod.build_preseason_ratings(pbp_prior, schedule_prior)

        this_season = nfl_data_mod.season_schedule(schedule_df, season)
        this_season = this_season[this_season["home_score"].notna()]
        this_season = this_season[this_season["home_score"] != this_season["away_score"]]

        home_probs, home_wins = [], []
        for _, g in this_season.iterrows():
            home, away = g["home_team"], g["away_team"]
            if home not in team_ratings.index or away not in team_ratings.index:
                continue  # relocated/renamed franchise between the two seasons - skip rather than guess
            p_home = ratings_mod.win_probability(team_ratings[home], team_ratings[away], b0, b1)
            home_probs.append(p_home)
            home_wins.append(1.0 if g["home_score"] > g["away_score"] else 0.0)

        fav_prob, fav_won = _favorite_view(home_probs, home_wins)
        per_season.append({
            "season": season, "n": len(fav_prob),
            "brier_score": round(float(np.mean((fav_prob - fav_won) ** 2)), 4) if len(fav_prob) else None,
        })
        all_fav_prob.extend(fav_prob)
        all_fav_won.extend(fav_won)

    all_fav_prob, all_fav_won = np.array(all_fav_prob), np.array(all_fav_won)
    return {
        "n": len(all_fav_prob),
        "test_seasons": test_seasons,
        "brier_score": round(float(np.mean((all_fav_prob - all_fav_won) ** 2)), 4),
        "buckets": _bucket_calibration(all_fav_prob, all_fav_won),
        "per_season": per_season,
    }


if __name__ == "__main__":
    import nfl_data

    sched = nfl_data.fetch_schedule()

    print("=== Market (devigged closing moneyline) calibration ===")
    market = historical_market_calibration(sched)
    print(f"n={market['n']} games, seasons {market['seasons']}, Brier score {market['brier_score']} "
          f"(0.25 = always guessing 50/50, lower is better)")
    for b in market["buckets"]:
        print(f"  {b['range']:>8}  n={b['n']:>4}  predicted {b['avg_predicted']:.1%}  actual {b['actual_win_rate']:.1%}")

    print("\n=== Model calibration, in-sample fit (the original, flawed approach - see ratings.py's "
          "FITTED_B0 docstring) ===")
    test_seasons = list(range(2019, 2026))  # needs pbp for 2018-2025
    old = historical_model_calibration(sched, nfl_data, test_seasons)
    print(f"n={old['n']} games pooled across test seasons {test_seasons}, Brier score {old['brier_score']}")
    for b in old["buckets"]:
        print(f"  {b['range']:>8}  n={b['n']:>4}  predicted {b['avg_predicted']:.1%}  actual {b['actual_win_rate']:.1%}")

    print("\n=== Model calibration, production fit (ratings.FITTED_B0/FITTED_B1 - what run_baseline.py / "
          "run_weekly.py actually use) ===")
    new = historical_model_calibration(sched, nfl_data, test_seasons, fixed_b0b1=(ratings_mod.FITTED_B0, ratings_mod.FITTED_B1))
    print(f"n={new['n']} games, Brier score {new['brier_score']} (market's, above, was {market['brier_score']} - "
          f"the model's own preseason-only numbers still don't match the market's accuracy, an honest limitation, "
          f"see FITTED_B0's docstring)")
    for b in new["buckets"]:
        print(f"  {b['range']:>8}  n={b['n']:>4}  predicted {b['avg_predicted']:.1%}  actual {b['actual_win_rate']:.1%}")
