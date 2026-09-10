#!/usr/bin/env python3
"""
Phase 1 entry point: build the full-season baseline survivor plan.

Run this once to get a season-long plan (before Week 1, or any time you
want to regenerate the from-scratch baseline). For in-season use once picks
have been made, run_weekly.py re-optimizes only the remaining weeks with
locked-in picks held fixed. Both produce a Normal and a Circa plan (see
optimizer.CIRCA_LEGS).

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
import planhelpers
import popularity as pop_mod
import field_model
import dashboard


def _attach_detail(plans, true_matrix, game_lookup, team_ratings,
                   popularity=None, ev_share=None, deltas=None):
    popularity = popularity or {}
    ev_share = ev_share or {}
    deltas = deltas or {}
    for plan in plans:
        detail = {}
        for slot, team in plan["picks"].items():
            info = game_lookup.get((slot, team), {})
            detail[slot] = {
                "team": team,
                "win_prob": float(true_matrix.loc[slot, team]),
                "opponent": info.get("opponent"),
                "is_home": info.get("is_home"),
                "div_game": info.get("div_game"),
                "team_rating": float(team_ratings.get(team, float("nan"))),
                "popularity": popularity.get(team),
                "ev_share": ev_share.get((slot, team)),
                "payout_delta": deltas.get((slot, team)),
            }
        plan["detail"] = detail
        plan["survival_prob"] = planhelpers.true_survival_prob(plan["picks"], true_matrix)
    return plans


def build_plan(n_alternates=5):
    sched = nfl_data.fetch_schedule(force_refresh=True)
    rating_season = nfl_data.latest_completed_season(sched)
    plan_season, plan_week = nfl_data.current_season_and_week(sched)

    pbp = nfl_data.fetch_pbp(rating_season)
    plan_schedule = nfl_data.season_schedule(sched, plan_season)

    team_ratings, b0, b1 = ratings_mod.build_preseason_ratings(pbp)
    matrix = optimizer.build_win_prob_matrix(plan_schedule, team_ratings, b0, b1)
    legs = optimizer.circa_legs(plan_schedule)
    circa_matrix, slot_order = optimizer.build_circa_matrix(plan_schedule, legs, team_ratings, b0, b1)
    game_lookup = planhelpers.game_lookup(plan_schedule, legs)

    # --- payout-share layer (Circa only; the private-pool Normal plan stays pure win-probability) ---
    real_pop, pop_meta = pop_mod.fetch_current_popularity()
    specs = planhelpers.build_leg_specs(circa_matrix, plan_schedule, legs, plan_week, real_pop)
    ev_share, _p_win, field_after = field_model.simulate(specs)
    deltas = field_model.blend_multipliers(ev_share, specs)
    circa_blended = planhelpers.apply_payout_delta(circa_matrix, deltas)

    circa_pure_top = optimizer.top_plans(circa_matrix, k=1, slot_order=slot_order)[0]["picks"]
    circa_blended_plans = optimizer.top_plans(circa_blended, k=n_alternates, slot_order=slot_order)
    blend_meta = planhelpers.payout_blend_meta(circa_pure_top, circa_blended_plans[0]["picks"], circa_matrix,
                                               field_model.POPULARITY_BLEND_WEIGHT, field_model.POPULARITY_BLEND_CAP,
                                               field_after, field_model.FIELD_SIZE)

    normal_plans = _attach_detail(
        optimizer.top_plans(matrix, k=n_alternates), matrix, game_lookup, team_ratings,
        popularity=real_pop)
    circa_plans = _attach_detail(
        circa_blended_plans, circa_matrix, game_lookup, team_ratings,
        popularity=real_pop, ev_share=ev_share, deltas=deltas)

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
        "circa_legs": [{"leg": s["leg"], "dates": s["dates"], "week": s["week"]} for s in legs],
        "schedule": planhelpers.schedule_slate(circa_matrix, plan_schedule, legs),
        "popularity_meta": pop_meta,
        "payout_blend": blend_meta,
        "future_value": planhelpers.future_value_teams(circa_matrix, circa_plans[0]["picks"]),
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
        _print_plan(result["circa_plans"][0], "Top CIRCA plan (20 legs: 18 weeks + Thanksgiving/Black Friday + Christmas)")

    print(f"\n{len(result['plans']) - 1} alternate plan(s) per mode also in plan.json / dashboard.html.")
    print("Open dashboard.html in a browser to see the full picture.")


if __name__ == "__main__":
    main()
