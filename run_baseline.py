#!/usr/bin/env python3
"""
Phase 1 entry point: build the full-season baseline survivor plan.

Run this once to get a season-long plan (before Week 1, or any time you
want to regenerate the from-scratch baseline). For in-season use once picks
have been made, Phase 2's run_weekly.py (not built yet) re-optimizes only
the remaining weeks with locked-in picks held fixed.

Usage:
    python run_baseline.py [--alternates N]

Output:
    plan.json       - top plan + alternates, structured (for dashboard.py /
                       later Phase 2 code to consume)
    dashboard.html  - open in a browser to view the plan
"""
import argparse
import json
import os
from datetime import datetime, timezone

import nfl_data
import ratings as ratings_mod
import optimizer
import dashboard


def build_plan(n_alternates=5):
    sched = nfl_data.fetch_schedule(force_refresh=True)
    rating_season = nfl_data.latest_completed_season(sched)
    plan_season, plan_week = nfl_data.current_season_and_week(sched)

    pbp = nfl_data.fetch_pbp(rating_season)
    rating_input_schedule = nfl_data.season_schedule(sched, rating_season)
    plan_schedule = nfl_data.season_schedule(sched, plan_season)

    team_ratings, b0, b1 = ratings_mod.build_preseason_ratings(pbp, rating_input_schedule)
    matrix = optimizer.build_win_prob_matrix(plan_schedule, team_ratings, b0, b1)
    byes = nfl_data.compute_byes(plan_schedule)

    plans = optimizer.top_plans(matrix, k=n_alternates)

    # Per-pick detail (opponent, home/away, div game) for the top plan and
    # each alternate, so the dashboard/CLI can show more than just the team
    # name - this is the "reasoning" half of the output spec (the risk-flag
    # half - injuries, weather, market cross-check - is Phase 2, once those
    # data sources are wired in).
    game_lookup = {}
    for _, g in plan_schedule.iterrows():
        wk, home, away = int(g["week"]), g["home_team"], g["away_team"]
        game_lookup[(wk, home)] = {"opponent": away, "is_home": True, "div_game": bool(g["div_game"])}
        game_lookup[(wk, away)] = {"opponent": home, "is_home": False, "div_game": bool(g["div_game"])}

    for plan in plans:
        detail = {}
        for wk, team in plan["picks"].items():
            info = game_lookup.get((wk, team), {})
            detail[wk] = {
                "team": team,
                "win_prob": matrix.loc[wk, team],
                "opponent": info.get("opponent"),
                "is_home": info.get("is_home"),
                "div_game": info.get("div_game"),
                "team_rating": float(team_ratings.get(team, float("nan"))),
            }
        plan["detail"] = detail

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rating_season": rating_season,
        "plan_season": plan_season,
        "plan_week": plan_week,
        "home_field_logodds": b0,
        "rating_sensitivity": b1,
        "byes": byes,
        "team_ratings": team_ratings.sort_values(ascending=False).round(4).to_dict(),
        "plans": plans,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternates", type=int, default=5, help="number of plans to output (top + near-optimal)")
    args = parser.parse_args()

    print("Fetching schedule + play-by-play, building ratings, solving season assignment...")
    result = build_plan(n_alternates=args.alternates)

    out_path = os.path.join(os.path.dirname(__file__), "plan.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    html_path = dashboard.render(result)
    print(f"Wrote {html_path}")

    top = result["plans"][0]
    print(f"\nTop plan - survival probability: {top['survival_prob']:.2%}")
    for wk, d in top["detail"].items():
        loc = "vs" if d["is_home"] else "@"
        div = " (div)" if d["div_game"] else ""
        print(f"  Week {wk:>2}: {d['team']:<4} {loc} {d['opponent']:<4}{div}  {d['win_prob']:.1%}")

    print(f"\n{len(result['plans']) - 1} alternate plan(s) also in plan.json / dashboard.html.")
    print("Open dashboard.html in a browser to see the full picture.")


if __name__ == "__main__":
    main()
