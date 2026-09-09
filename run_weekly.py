#!/usr/bin/env python3
"""
Phase 2 entry point: weekly re-optimization. Re-run this before each week's
pick deadline for a fresh recommendation - only picks already logged via
--lock are held fixed; everything else (including this week's pick, until
you lock it) is recomputed from scratch each run.

Usage:
    python run_weekly.py                  # show this week's recommendation + provisional rest-of-season plan
    python run_weekly.py --lock KC        # commit KC as this week's actual pick (writes picks_log.jsonl), then show the result
    python run_weekly.py --alternates N   # how many plans to compute (default 5)

Output: plan.json, dashboard.html (same as run_baseline.py), plus a
prominent console summary of the current week's pick with its risk flags
and reasoning.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import nfl_data
import ratings as ratings_mod
import optimizer
import risk
import picks_log
import dashboard


def _flags_for_pick(wk, team, injury_notes, rest_flags, weather_this_week, market_flags):
    flags = []
    inj = injury_notes.get((wk, team))
    if inj:
        flags.append(f"QB {inj['flag'].upper()}: {inj.get('comment') or inj.get('detail') or ''}".strip())
    flags.extend(rest_flags.get((wk, team), []))
    if team in weather_this_week:
        flags.extend(f"weather: {w}" for w in weather_this_week[team])
    mkt = market_flags.get((wk, team))
    if mkt:
        flags.append(f"model/market disagreement: model {mkt['model']:.1%} vs market {mkt['market']:.1%}")
    return flags


def _reasoning(wk, team, opponent, is_home, div_game, win_prob, flags):
    loc = "at home vs" if is_home else "on the road at"
    base = f"{team} {loc} {opponent}, model win probability {win_prob:.1%}"
    if div_game:
        # NOT flagged as extra risk - Phase 3 (trap_games.py) tested this
        # against 16 seasons of real results and found zero evidence
        # divisional games are more volatile or harder to predict than the
        # market's own price implies (see risk.py's rest_travel_flags
        # docstring for the numbers). Noted as context only.
        base += " (divisional matchup)"
    if flags:
        base += ". Flags: " + "; ".join(flags)
    else:
        base += ". No injury/rest/weather/market flags this run."
    return base


def build_weekly_plan(n_alternates=5):
    sched = nfl_data.fetch_schedule(force_refresh=True)
    plan_season, plan_week = nfl_data.current_season_and_week(sched)
    plan_schedule = nfl_data.season_schedule(sched, plan_season)

    locked = picks_log.load_locked_picks(plan_season)
    used_before = {team for wk, team in locked.items() if wk < plan_week}
    locked_for_solve = {wk: team for wk, team in locked.items() if wk >= plan_week}

    team_ratings, b0, b1, weight_current = ratings_mod.build_inseason_ratings(sched, plan_season, nfl_data)

    remaining_schedule = plan_schedule[plan_schedule["week"] >= plan_week]
    matrix = optimizer.build_win_prob_matrix(remaining_schedule, team_ratings, b0, b1)
    matrix = matrix.drop(columns=[t for t in used_before if t in matrix.columns])

    adjusted_matrix, injury_notes, injury_err = risk.apply_qb_injury_adjustment(matrix, remaining_schedule)
    rest_flags = risk.rest_travel_flags(remaining_schedule)
    weather_this_week = risk.weather_flags(remaining_schedule, plan_week)
    market_flags = risk.market_cross_check(adjusted_matrix, remaining_schedule, nfl_data.TEAM_FULL_NAMES)

    plans = optimizer.top_plans(adjusted_matrix, k=n_alternates, locked=locked_for_solve)

    game_lookup = {}
    for _, g in remaining_schedule.iterrows():
        wk, home, away = int(g["week"]), g["home_team"], g["away_team"]
        game_lookup[(wk, home)] = {"opponent": away, "is_home": True, "div_game": bool(g["div_game"])}
        game_lookup[(wk, away)] = {"opponent": home, "is_home": False, "div_game": bool(g["div_game"])}

    for plan in plans:
        detail = {}
        for wk, team in plan["picks"].items():
            info = game_lookup.get((wk, team), {})
            flags = _flags_for_pick(wk, team, injury_notes, rest_flags, weather_this_week if wk == plan_week else {}, market_flags)
            reasoning = _reasoning(wk, team, info.get("opponent"), info.get("is_home"), info.get("div_game"),
                                    matrix.loc[wk, team] if (wk in matrix.index and team in matrix.columns) else adjusted_matrix.loc[wk, team],
                                    flags)
            detail[wk] = {
                "team": team,
                "win_prob": adjusted_matrix.loc[wk, team],
                "opponent": info.get("opponent"),
                "is_home": info.get("is_home"),
                "div_game": info.get("div_game"),
                "team_rating": float(team_ratings.get(team, float("nan"))),
                "flags": flags,
                "reasoning": reasoning,
                "locked": wk in locked_for_solve,
            }
        plan["detail"] = detail

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_season": plan_season,
        "plan_week": plan_week,
        "home_field_logodds": b0,
        "rating_sensitivity": b1,
        "inseason_weight": weight_current,
        "byes": nfl_data.compute_byes(plan_schedule),
        "team_ratings": team_ratings.sort_values(ascending=False).round(4).to_dict(),
        "plans": plans,
        "locked_picks": locked,
        "injury_check_error": injury_err,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alternates", type=int, default=5)
    parser.add_argument("--lock", type=str, default=None, metavar="TEAM",
                         help="commit TEAM as this week's actual pick (writes picks_log.jsonl)")
    args = parser.parse_args()

    print("Re-optimizing remaining season with fresh ratings, injuries, rest/travel, and weather...")
    result = build_weekly_plan(n_alternates=args.alternates)
    plan_week = result["plan_week"]
    top = result["plans"][0]

    if args.lock:
        team = args.lock.upper()
        pick_row = top["detail"].get(plan_week)
        if pick_row is None:
            raise SystemExit(f"No plan detail for week {plan_week} - can't lock.")
        if pick_row["team"] != team:
            raise SystemExit(
                f"'{team}' isn't this run's top recommendation for week {plan_week} ({pick_row['team']} is). "
                f"Re-run without --lock to see all {args.alternates} plans and pick a team from one of them, "
                f"then lock that exact team."
            )
        win_prob = pick_row["win_prob"]
        picks_log.log_pick(
            season=result["plan_season"], week=plan_week, team=team,
            win_prob=win_prob, flags=pick_row["flags"], reasoning=pick_row["reasoning"],
            locked_at=result["generated_at"],
        )
        print(f"Locked: Week {plan_week} -> {team}. Logged to picks_log.jsonl.")
        # Re-run once more so the dashboard/plan.json reflect the lock
        result = build_weekly_plan(n_alternates=args.alternates)
        top = result["plans"][0]

    out_path = os.path.join(os.path.dirname(__file__), "plan.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    html_path = dashboard.render(result)

    print(f"\nWrote {out_path}\nWrote {html_path}\n")
    print(f"=== Week {plan_week} recommendation ===")
    d = top["detail"][plan_week]
    print(f"  {d['team']} ({d['win_prob']:.1%}){'  [LOCKED]' if d['locked'] else ''}")
    print(f"  {d['reasoning']}")
    print(f"\nRest-of-season provisional plan (top of {len(result['plans'])}):")
    for wk, dd in top["detail"].items():
        if wk == plan_week:
            continue
        flag_str = f"  [{', '.join(dd['flags'])}]" if dd["flags"] else ""
        print(f"  Week {wk:>2}: {dd['team']:<4} ({dd['win_prob']:.1%}){flag_str}")
    print(f"\nSurvival probability, remaining season: {top['survival_prob']:.2%}")


if __name__ == "__main__":
    main()
