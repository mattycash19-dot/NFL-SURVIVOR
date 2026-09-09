"""
Running log of every pick actually made (not just recommended) and its
eventual result - accountability + the raw material for Phase 3's
calibration check. Same append-only-JSONL pattern as MLB Edge's
predictions_log.jsonl.

Logging a pick here is a deliberate, separate step (`run_weekly.py --lock`)
from generating a recommendation - the plan changes every week as data
updates, but a pick logged here represents what was actually submitted to
the pool and should never be silently rewritten.
"""
import json
import os

LOG_PATH = os.path.join(os.path.dirname(__file__), "picks_log.jsonl")


def _read_all():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_all(records):
    with open(LOG_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def log_pick(season, week, team, win_prob, flags, reasoning, locked_at):
    """Appends one committed pick. Refuses to double-log the same
    season+week (use a fresh call only if you genuinely need to correct an
    entry - do that by hand, not silently here, since this is meant to be
    an honest record of what was actually picked when)."""
    records = _read_all()
    if any(r["season"] == season and r["week"] == week for r in records):
        raise ValueError(f"Season {season} Week {week} is already logged - edit picks_log.jsonl by hand if this needs correcting.")
    records.append({
        "season": season,
        "week": week,
        "team": team,
        "win_prob_at_pick": win_prob,
        "flags": flags,
        "reasoning": reasoning,
        "locked_at": locked_at,
        "result": None,  # filled in by mark_result() once the game's final
    })
    _write_all(records)


def load_locked_picks(season):
    """{week: team} for every pick already logged this season - these are
    forced into every remaining-season optimization (see run_weekly.py)."""
    return {r["week"]: r["team"] for r in _read_all() if r["season"] == season}


def mark_result(season, week, won):
    records = _read_all()
    found = False
    for r in records:
        if r["season"] == season and r["week"] == week:
            r["result"] = "win" if won else "loss"
            found = True
    if not found:
        raise ValueError(f"No logged pick for season {season} week {week}.")
    _write_all(records)


def season_record(season):
    """(wins, losses, still_alive) for the logged picks so far this
    season - "still alive" is False as soon as one loss is logged, since
    that's the whole survivor-pool stakes."""
    records = [r for r in _read_all() if r["season"] == season]
    wins = sum(1 for r in records if r["result"] == "win")
    losses = sum(1 for r in records if r["result"] == "loss")
    return wins, losses, losses == 0
