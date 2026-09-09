"""
Phase 3, part 2: situational patterns - letdown spots, lookahead spots,
bye-week rest effects, divisional volatility. Tested against history
(2010-2025, whenever moneylines are available) rather than just asserted,
same honesty standard as calibration.py and MLB Edge's detect_patterns().

Method: for every historical game, take the devigged market-implied
probability for the eventual favorite (same "favorite view" as
calibration.py) as the baseline expectation, then compare the favorite's
actual straight-up win rate against that baseline within each situational
bucket vs. everyone else. A bucket where the favorite reliably
underperforms its market price is real signal a survivor picker should
weight; a bucket that's statistically indistinguishable from the baseline
isn't, no matter how good the story sounds.

Every finding reports n and a rough standard error on the residual mean so
a 20-game fluke doesn't read the same as a 500-game pattern - deliberately
mirrors MLB Edge's compute_track_record()/detect_patterns() convention.
"""
import numpy as np
import pandas as pd

import odds_data

MIN_SEASON = 2010          # moneyline coverage starts being reliable here (see calibration.py)
BIG_WIN_MOV = 14           # margin-of-victory threshold for "big win last week"
BIG_FAVORITE_PROB = 0.75   # market-implied prob threshold for "big favorite" (letdown/lookahead framing)


def _team_game_log(reg):
    """One row per (team, season, week) they played - home and away
    perspectives stacked - with their own margin of victory, rest days,
    and whether it was a divisional game."""
    home = reg[["season", "week", "home_team", "away_team", "home_score", "away_score", "home_rest", "div_game"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_score": "pf", "away_score": "pa", "home_rest": "rest"})
    away = reg[["season", "week", "away_team", "home_team", "away_score", "home_score", "away_rest", "div_game"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_score": "pf", "home_score": "pa", "away_rest": "rest"})
    both = pd.concat([home, away], ignore_index=True)
    both["won"] = both["pf"] > both["pa"]
    both["mov"] = both["pf"] - both["pa"]
    return both.sort_values(["team", "season", "week"]).reset_index(drop=True)


def _favorite_prob_by_team(reg):
    """(season, week, team) -> that team's devigged market-implied win
    probability for that game, whichever side they're on."""
    out = {}
    for _, g in reg.iterrows():
        if pd.isna(g["home_moneyline"]) or pd.isna(g["away_moneyline"]):
            continue
        raw_h = odds_data.moneyline_to_implied_prob(g["home_moneyline"])
        raw_a = odds_data.moneyline_to_implied_prob(g["away_moneyline"])
        h, a = odds_data.devig_two_way(raw_h, raw_a)
        out[(g["season"], g["week"], g["home_team"])] = h
        out[(g["season"], g["week"], g["away_team"])] = a
    return out


def build_situational_dataset(schedule_df, min_season=MIN_SEASON):
    """
    One row per (team, game) with: market_prob (that team's devigged win
    probability), won (did they actually win), and the situational flags -
    prev_big_win, off_bye, opponent_off_bye, div_game, next_week_big_favorite
    (a proxy for "lookahead" - next week they're a big favorite, i.e. an
    easy game looms that might pull focus... note the spec's literal
    "big game next week" more often means a big RIVAL/marquee game, which
    isn't cleanly derivable from spread size alone - this proxies it as
    "next week is a mismatch in their favor," the closest testable signal
    available without hand-labeling marquee games).
    """
    reg = schedule_df[(schedule_df["game_type"] == "REG") & (schedule_df["season"] >= min_season)].copy()
    reg = reg[reg["home_score"].notna()]
    reg = reg[reg["home_score"] != reg["away_score"]]

    log = _team_game_log(reg)
    log["prev_mov"] = log.groupby(["team", "season"])["mov"].shift(1)
    log["prev_won"] = log.groupby(["team", "season"])["won"].shift(1)
    log["weeks_since_prev"] = log.groupby(["team", "season"])["week"].diff()
    log["off_bye"] = log["weeks_since_prev"] > 1

    opp_off_bye = log[["season", "week", "team", "off_bye"]].rename(
        columns={"team": "opponent", "off_bye": "opponent_off_bye"})
    log = log.merge(opp_off_bye, on=["season", "week", "opponent"], how="left")

    fav_prob = _favorite_prob_by_team(reg)
    log["market_prob"] = log.apply(lambda r: fav_prob.get((r["season"], r["week"], r["team"])), axis=1)
    log = log[log["market_prob"].notna()].copy()

    log["prev_big_win"] = log["prev_won"].fillna(False) & (log["prev_mov"].fillna(0) >= BIG_WIN_MOV)

    log["next_mov_placeholder"] = log.groupby(["team", "season"])["market_prob"].shift(-1)
    log["next_week_big_favorite"] = log["next_mov_placeholder"].fillna(0) >= BIG_FAVORITE_PROB
    log = log.drop(columns=["next_mov_placeholder"])

    return log


def _bucket_report(log, mask, label):
    flagged = log[mask]
    control = log[~mask]
    if len(flagged) < 8:
        return {"label": label, "n": len(flagged), "note": "too few games to say anything - not reported as a finding"}

    def resid_stats(df):
        resid = df["won"].astype(float) - df["market_prob"]
        n = len(resid)
        return float(resid.mean()), float(resid.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"), n

    f_mean, f_se, f_n = resid_stats(flagged)
    c_mean, c_se, c_n = resid_stats(control)
    return {
        "label": label,
        "n": f_n,
        "flagged_mean_residual": round(f_mean, 4),
        "flagged_se": round(f_se, 4),
        "control_mean_residual": round(c_mean, 4),
        "control_n": c_n,
        "gap": round(f_mean - c_mean, 4),
        "roughly_significant": abs(f_mean - c_mean) > 2 * f_se,  # crude ~95% gut-check, not a real hypothesis test
    }


def check_patterns(schedule_df, min_season=MIN_SEASON):
    log = build_situational_dataset(schedule_df, min_season)
    return {
        "n_games": len(log),
        "seasons": f"{min_season}-{int(log['season'].max())}",
        "letdown_spot (favorite, big win last week)": _bucket_report(
            log, (log["market_prob"] >= 0.5) & log["prev_big_win"], "letdown"),
        "lookahead_spot (big favorite this week, bigger mismatch looms next week)": _bucket_report(
            log, (log["market_prob"] >= BIG_FAVORITE_PROB) & log["next_week_big_favorite"], "lookahead"),
        "coming_off_bye": _bucket_report(log, log["off_bye"].fillna(False), "off_bye"),
        "opponent_coming_off_bye": _bucket_report(log, log["opponent_off_bye"].fillna(False), "opp_off_bye"),
        "divisional_game": _bucket_report(log, log["div_game"] == 1, "divisional"),
    }


if __name__ == "__main__":
    import nfl_data

    sched = nfl_data.fetch_schedule()
    results = check_patterns(sched)
    print(f"n={results['n_games']} team-game rows, seasons {results['seasons']}\n")
    for key, r in results.items():
        if key in ("n_games", "seasons"):
            continue
        print(f"--- {key} ---")
        if "note" in r:
            print(f"  {r['note']} (n={r['n']})")
        else:
            sig = " *** roughly significant ***" if r["roughly_significant"] else ""
            print(f"  n={r['n']}, mean residual {r['flagged_mean_residual']:+.3f} (SE {r['flagged_se']:.3f}) "
                  f"vs. control {r['control_mean_residual']:+.3f} (n={r['control_n']}) - gap {r['gap']:+.3f}{sig}")
        print()
