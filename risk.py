"""
Phase 2 risk signals: injury, rest/travel, weather, and market cross-check,
combined into per-(week, team) flags for the reasoning/output layer, plus
the one signal quantified enough to actually move the win-probability
matrix before re-solving (starter-QB status).

Design choice, stated plainly: only the starter-QB check adjusts the
model's actual probabilities. Rest/travel/international/weather/divisional
volatility are surfaced as qualitative flags on the reasoning, not silently
folded into the number - turning "short week" or "divisional game" into a
precise probability multiplier without real backtested evidence behind it
would be exactly the kind of fabricated precision this vault's projects
have consistently avoided (see MLB Edge's writeup.py). Phase 3's historical
calibration check is the intended place to test whether any of these
actually deserve a quantified adjustment - until then, they're shown, not
baked in.
"""
import numpy as np

import injuries as injuries_mod
import weather as weather_mod
import odds_data

# Starting-assumption haircuts for a team's win probability when its
# scheduled starting QB is banged up - not fitted, flagged for Phase 3.
# "questionable" gets a light touch since a Q-tagged starter plays more
# often than not; "doubtful"/"out"/IR get a real haircut since backup QB
# play is a well-established, large drop in offensive performance.
QB_STATUS_HAIRCUT = {"questionable": 0.97, "doubtful": 0.88, "out": 0.80}

SHORT_REST_DAYS = 6         # < this many days since the team's last game
BIG_MARKET_DISAGREEMENT = 0.08  # model vs. devigged market prob gap worth flagging


def apply_qb_injury_adjustment(matrix, plan_schedule, home_field_col_lookup=None):
    """
    Checks EVERY remaining (week, team) cell in `matrix` against today's
    injury report and that game's scheduled starting QB - not just the
    current week - per the spec ("an injury to a team I'm saving for a
    later week reshuffles the plan now"). Each weekly re-run refreshes both
    the injury snapshot and the scheduled-starter snapshot together, so a
    stale read self-corrects on the next run rather than compounding.

    Returns (adjusted_matrix, notes) where notes is {(week, team): {...}}
    for every cell that got a haircut - the reasoning layer reads this.
    Degrades to a no-op (matrix unchanged, notes empty, feasible without
    injury data) if the injury fetch fails - an unofficial endpoint being
    flaky shouldn't take down the whole pipeline.
    """
    try:
        injury_data = injuries_mod.fetch_all_injuries()
    except Exception as e:
        return matrix.copy(), {}, f"injury check skipped: {e}"

    adjusted = matrix.copy()
    notes = {}
    qb_lookup = {}  # (week, team) -> scheduled qb name
    for _, g in plan_schedule.iterrows():
        wk = int(g["week"])
        qb_lookup[(wk, g["home_team"])] = g.get("home_qb_name")
        qb_lookup[(wk, g["away_team"])] = g.get("away_qb_name")

    for wk in adjusted.index:
        for team in adjusted.columns:
            p = adjusted.loc[wk, team]
            if not np.isfinite(p):
                continue
            qb_name = qb_lookup.get((wk, team))
            risk = injuries_mod.starter_qb_risk(injury_data, team, qb_name)
            if risk.get("flag"):
                haircut = QB_STATUS_HAIRCUT.get(risk["flag"], 1.0)
                adjusted.loc[wk, team] = p * haircut
                notes[(wk, team)] = {
                    "type": "qb_injury",
                    "flag": risk["flag"],
                    "comment": risk.get("comment"),
                    "haircut": haircut,
                }
    return adjusted, notes, None


def rest_travel_flags(plan_schedule):
    """
    Per (week, team) qualitative flags from data already in the schedule -
    no extra API calls. Returns {(week, team): [flag, ...]}.
    """
    flags = {}
    reg = plan_schedule.sort_values(["week"])
    last_played_week = {}  # team -> last week they played (to detect "coming off bye")

    for wk in sorted(reg["week"].unique()):
        week_games = reg[reg["week"] == wk]
        playing_this_week = set(week_games["home_team"]) | set(week_games["away_team"])
        for _, g in week_games.iterrows():
            for team, rest_col, opp, opp_rest_col in [
                (g["home_team"], "home_rest", g["away_team"], "away_rest"),
                (g["away_team"], "away_rest", g["home_team"], "home_rest"),
            ]:
                f = []
                rest_days = g.get(rest_col)
                if pd_notna(rest_days) and rest_days < SHORT_REST_DAYS:
                    f.append(f"short rest ({int(rest_days)} days)")
                if g.get("location") != "Home":
                    f.append("international/neutral site")
                if g.get("div_game"):
                    f.append("divisional matchup")
                if last_played_week.get(team) is not None and wk - last_played_week[team] > 1:
                    f.append("coming off bye")
                opp_last = last_played_week.get(opp)
                if opp_last is not None and wk - opp_last > 1:
                    f.append(f"opponent ({opp}) coming off bye")
                if f:
                    flags[(int(wk), team)] = f
        for team in playing_this_week:
            last_played_week[team] = wk
    return flags


def pd_notna(x):
    import pandas as pd
    return pd.notna(x)


def weather_flags(plan_schedule, week):
    """Only checked for the given week (NWS's forecast window is ~7 days -
    no point calling it for weeks further out). Returns {team: [flag]}."""
    flags = {}
    week_games = plan_schedule[plan_schedule["week"] == week]
    for _, g in week_games.iterrows():
        home = g["home_team"]
        outdoor = weather_mod.is_outdoor_this_game(home, g.get("roof"))
        if not outdoor:
            continue
        try:
            fc = weather_mod.forecast_for_game(home, g["gameday"], g.get("location", "Home"))
        except Exception:
            continue
        if not fc.get("available"):
            continue
        f = []
        precip = fc.get("precipitation_chance")
        if precip is not None and precip >= 50:
            f.append(f"{precip}% precip chance ({fc['short_forecast']})")
        wind = fc.get("wind", "")
        try:
            wind_mph = int("".join(c for c in wind.split()[0] if c.isdigit()))
            if wind_mph >= 20:
                f.append(f"high wind ({wind})")
        except (ValueError, IndexError):
            pass
        if f:
            flags[home] = f
            flags[g["away_team"]] = f  # affects both teams' game equally
    return flags


def market_cross_check(matrix, plan_schedule, team_full_names):
    """
    Secondary cross-check only - compares the model's win probabilities to
    devigged market consensus for games with posted odds, flags large
    disagreements. Returns {(week, team): {"model": p, "market": p}} for
    flagged cells only. Silently returns {} if no API key is configured or
    the request fails - this is explicitly optional per the spec.
    """
    try:
        events = odds_data.get_nfl_odds()
    except odds_data.NoApiKeyError:
        return {}
    except Exception:
        return {}

    market = odds_data.match_event_to_teams(events, team_full_names)
    flagged = {}
    for wk in matrix.index:
        for team in matrix.columns:
            p_model = matrix.loc[wk, team]
            if not np.isfinite(p_model) or team not in market:
                continue
            p_market = market[team]["market_prob"]
            if abs(p_model - p_market) >= BIG_MARKET_DISAGREEMENT:
                flagged[(int(wk), team)] = {"model": p_model, "market": p_market}
    return flagged
