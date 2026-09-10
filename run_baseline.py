#!/usr/bin/env python3
"""
Phase 1 entry point: build the full-season baseline survivor plan.

Run this once to get a season-long plan (before Week 1, or any time you
want to regenerate the from-scratch baseline). For in-season use once picks
have been made, run_weekly.py re-optimizes only the remaining weeks with
locked-in picks held fixed. Both produce a Normal and a Circa plan (see
optimizer.CIRCA_SLOTS).

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


def _game_lookup(schedule_df, circa_slots):
    """(row_label, team) -> {opponent, is_home, div_game}. row_label is an
    int week for normal picks, or a Circa slot name for holiday picks."""
    lut = {}
    for _, g in schedule_df.iterrows():
        wk, home, away, div = int(g["week"]), g["home_team"], g["away_team"], bool(g["div_game"])
        lut[(wk, home)] = {"opponent": away, "is_home": True, "div_game": div}
        lut[(wk, away)] = {"opponent": home, "is_home": False, "div_game": div}
    for s in circa_slots:
        for _, g in s["games"].iterrows():
            home, away, div = g["home_team"], g["away_team"], bool(g["div_game"])
            lut[(s["slot"], home)] = {"opponent": away, "is_home": True, "div_game": div}
            lut[(s["slot"], away)] = {"opponent": home, "is_home": False, "div_game": div}
    return lut


def _attach_detail(plans, matrix, game_lookup, team_ratings):
    for plan in plans:
        detail = {}
        for slot, team in plan["picks"].items():
            info = game_lookup.get((slot, team), {})
            detail[slot] = {
                "team": team,
                "win_prob": float(matrix.loc[slot, team]),
                "opponent": info.get("opponent"),
                "is_home": info.get("is_home"),
                "div_game": info.get("div_game"),
                "team_rating": float(team_ratings.get(team, float("nan"))),
            }
        plan["detail"] = detail
    return plans


def build_plan(n_alternates=5):
    sched = nfl_data.fetch_schedule(force_refresh=True)
    rating_season = nfl_data.latest_completed_season(sched)
    plan_season, plan_week = nfl_data.current_season_and_week(sched)

    pbp = nfl_data.fetch_pbp(rating_season)
    plan_schedule = nfl_data.season_schedule(sched, plan_season)

    team_ratings, b0, b1 = ratings_mod.build_preseason_ratings(pbp)
    matrix = optimizer.build_win_prob_matrix(plan_schedule, team_ratings, b0, b1)
    circa_slots = optimizer.circa_holiday_slots(plan_schedule)
    circa_matrix, slot_order = optimizer.append_circa_slots(matrix, circa_slots, team_ratings, b0, b1)
    game_lookup = _game_lookup(plan_schedule, circa_slots)

    normal_plans = _attach_detail(optimizer.top_plans(matrix, k=n_alternates), matrix, game_lookup, team_ratings)
    circa_plans = _attach_detail(
        optimizer.top_plans(circa_matrix, k=n_alternates, slot_order=slot_order),
        circa_matrix, game_lookup, team_ratings)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rating_season": rating_season,
        "plan_season": plan_season,
        "plan_week": plan_week,
        "home_field_logodds": b0,
        "rating_sensitivity": b1,
        "byes": nfl_data.compute_byes(plan_schedule),
        "team_ratings": team_ratings.sort_values(ascending=False).round(4).to_dict(),
        "team_names": nfl_data.TEAM_FULL_NAMES,
        "plans": normal_plans,
        "circa_plans": circa_plans,
        "circa_slots": [{"slot": s["slot"], "date": s["date"], "week": s["week"]} for s in circa_slots],
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

    def _print_plan(plan, title):
        print(f"\n{title} - survival probability: {plan['survival_prob']:.2%}")
        for slot, d in plan["detail"].items():
            label = f"Week {slot}" if isinstance(slot, int) else slot
            loc = "vs" if d["is_home"] else "@"
            div = " (div)" if d["div_game"] else ""
            print(f"  {label:>16}: {d['team']:<4} {loc} {d['opponent']:<4}{div}  {d['win_prob']:.1%}")

    _print_plan(result["plans"][0], "Top NORMAL plan")
    if result["circa_plans"]:
        _print_plan(result["circa_plans"][0], "Top CIRCA plan (adds Thanksgiving Eve/Day, Black Friday, Christmas picks)")

    print(f"\n{len(result['plans']) - 1} alternate plan(s) per mode also in plan.json / dashboard.html.")
    print("Open dashboard.html in a browser to see the full picture.")


if __name__ == "__main__":
    main()
