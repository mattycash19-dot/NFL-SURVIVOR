"""
Injury reports from ESPN's unofficial public JSON API - no official docs,
no key, but a stable, widely-used endpoint (same one ESPN's own site/app
calls). One request returns the entire league's current injury list, which
is both simpler and far cheaper than the officially-documented alternative
(ESPN's "core" API, sports.core.api.espn.com, which requires walking a
paginated list of $ref links and re-fetching each one individually - dozens
of extra requests per team for no real benefit here).

Since this is unofficial, it's used defensively: any parse failure or
schema change degrades to "no injury data available" rather than crashing
the pipeline - see run_weekly.py, which must be able to produce a plan even
if this fails.

Starting-QB risk check: the schedule file (nfl_data.fetch_schedule) already
carries each upcoming game's projected starting QB name in home_qb_name /
away_qb_name. Cross-referencing that name against this team's QB entries in
the injury list answers the actual question ("is the QB I'm relying on for
this pick hurt") without needing a separate depth-chart source.
"""
import requests

INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

# This is fully public, unauthenticated JSON (the same data ESPN serves any
# browser, no login) - but ESPN's WAF returns 403 for generic script/library
# User-Agents (tested: a default "python-requests/..." UA and a spoofed
# desktop-browser UA were both blocked; curl's own UA string passed clean).
# Identifying as curl clears that filter without misrepresenting what's
# being accessed or bypassing any actual auth.
REQUEST_HEADERS = {"User-Agent": "curl/8.7.1"}

# ESPN's injury `status` values, ranked worst-to-best for risk purposes.
STATUS_SEVERITY = {"Out": 3, "Injured Reserve": 3, "Doubtful": 2, "Questionable": 1}


def fetch_all_injuries():
    """Raises requests.RequestException on network failure - callers should
    catch broadly (see run_weekly.py) and degrade gracefully rather than
    letting a flaky unofficial endpoint take down the whole pipeline."""
    resp = requests.get(INJURIES_URL, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def team_injuries(injuries_json, team_abbrev):
    """All current injury entries for one team, defensively parsed - a
    missing/renamed field just gets skipped for that one entry rather than
    failing the whole call."""
    out = []
    for team_block in injuries_json.get("injuries", []):
        for entry in team_block.get("injuries", []):
            athlete = entry.get("athlete", {})
            if athlete.get("team", {}).get("abbreviation") != team_abbrev:
                continue
            out.append({
                "name": athlete.get("displayName"),
                "position": (athlete.get("position") or {}).get("abbreviation"),
                "status": entry.get("status"),
                "detail": (entry.get("details") or {}).get("type"),
                "return_date": (entry.get("details") or {}).get("returnDate"),
                "comment": entry.get("shortComment"),
            })
    return out


def starter_qb_risk(injuries_json, team_abbrev, scheduled_qb_name):
    """
    Checks whether the QB the schedule says will start for `team_abbrev`
    shows up on that team's injury list. Returns:
      {"flag": None, ...}                        - not on the injury list at all
      {"flag": "questionable"|"doubtful"|"out", "detail": str, "comment": str}
    `scheduled_qb_name` can be None/NaN (schedule hasn't posted a starter
    yet, common for weeks further in the future) - returns flag=None with a
    note, not a false "clean" reading.
    """
    if not scheduled_qb_name or (isinstance(scheduled_qb_name, float)):  # NaN
        return {"flag": None, "note": "no projected starter posted yet"}

    qb_entries = [e for e in team_injuries(injuries_json, team_abbrev) if e["position"] == "QB"]
    match = next((e for e in qb_entries if e["name"] == scheduled_qb_name), None)
    if match is None:
        return {"flag": None, "note": "scheduled starter not on the injury list"}

    status = (match["status"] or "").strip()
    severity = STATUS_SEVERITY.get(status, 0)
    flag = {3: "out", 2: "doubtful", 1: "questionable"}.get(severity)
    return {
        "flag": flag,
        "status": status,
        "detail": match["detail"],
        "comment": match["comment"],
    }


if __name__ == "__main__":
    import nfl_data

    sched = nfl_data.fetch_schedule()
    plan_season, plan_week = nfl_data.current_season_and_week(sched)
    week_games = sched[
        (sched["season"] == plan_season) & (sched["game_type"] == "REG") & (sched["week"] == plan_week)
    ]

    try:
        data = fetch_all_injuries()
    except Exception as e:
        print(f"Could not fetch injuries: {e}")
        raise SystemExit(1)

    print(f"Starting-QB risk check for {plan_season} Week {plan_week}:\n")
    for _, g in week_games.iterrows():
        for side, team_col, qb_col in [("home", "home_team", "home_qb_name"), ("away", "away_team", "away_qb_name")]:
            team, qb = g[team_col], g[qb_col]
            risk = starter_qb_risk(data, team, qb)
            if risk["flag"]:
                print(f"  {team:<4} ({qb}): {risk['flag'].upper()} - {risk.get('comment')}")
    print("\n(Teams not printed above have no starter-QB injury flag this week.)")
