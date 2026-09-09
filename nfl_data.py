"""
Data ingestion: NFL schedule + play-by-play, both free/no-key from the
nflverse project.

Sources:
- Schedule: raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
  (Lee Sharpe's schedule file, mirrored into nflverse - covers every season
  1999-present, including the current season's full schedule posted before
  it's played, with result columns NA until games are final. This single
  file gives us schedule, byes, rest days, and market lines all at once.)
- Play-by-play: github.com/nflverse/nflverse-data releases (one parquet per
  season), used here for the EPA/success-rate inputs to ratings.py.

Both cached to data/ (gitignored) so repeated runs don't re-download; pass
force_refresh=True to bypass the cache (do this weekly during the season -
see run_weekly.py, added in Phase 2).
"""
import os
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

SCHEDULE_CACHE = os.path.join(DATA_DIR, "schedule.csv")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_schedule(force_refresh=False):
    """
    Full schedule, 1999-present, one row per game. Columns of note for this
    project: season, week, game_type (REG/WC/DIV/CON/SB), away_team,
    home_team, away_score, home_score (NaN until the game is final),
    gameday, weekday, gametime, away_rest, home_rest, div_game, roof,
    spread_line, home_moneyline/away_moneyline (market lines - used in
    Phase 2 as a cross-check, not here).
    """
    _ensure_data_dir()
    if force_refresh or not os.path.exists(SCHEDULE_CACHE):
        resp = requests.get(SCHEDULE_URL, timeout=30)
        resp.raise_for_status()
        with open(SCHEDULE_CACHE, "wb") as f:
            f.write(resp.content)
    return pd.read_csv(SCHEDULE_CACHE, low_memory=False)


def fetch_pbp(season, force_refresh=False):
    """
    Play-by-play for one season, trimmed to the columns ratings.py actually
    needs (the full file has 300+ columns and is 15-25MB per season - no
    reason to cache or load more than this). Only REG-season rows are kept
    here since ratings.py filters to REG anyway and this keeps the cache
    smaller; pass season=None-filtering elsewhere if postseason is ever
    wanted.
    """
    _ensure_data_dir()
    cache_path = os.path.join(DATA_DIR, f"pbp_{season}.parquet")
    if force_refresh or not os.path.exists(cache_path):
        url = PBP_URL_TMPL.format(season=season)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        raw_path = os.path.join(DATA_DIR, f"_pbp_{season}_raw.parquet")
        with open(raw_path, "wb") as f:
            f.write(resp.content)
        cols = [
            "season", "week", "season_type", "game_id",
            "posteam", "defteam", "play_type", "epa", "success",
        ]
        df = pd.read_parquet(raw_path, columns=cols)
        df.to_parquet(cache_path)
        os.remove(raw_path)
    return pd.read_parquet(cache_path)


def latest_completed_season(schedule_df):
    """
    The most recent season where every scheduled REG-season game has a
    final score - i.e. a season we can safely build ratings from. Deliberately
    doesn't assume "current calendar year minus 1"; a season with games still
    in progress is excluded even if most of it is done, since fetch_pbp()
    pulls a whole-season file and a partial season's ratings should come from
    in-season play-by-play (Phase 2), not this whole-file fetch.
    """
    reg = schedule_df[schedule_df["game_type"] == "REG"]
    by_season = reg.groupby("season")["home_score"].apply(lambda s: s.notna().all())
    complete_seasons = by_season[by_season].index
    if len(complete_seasons) == 0:
        raise RuntimeError("No fully-completed REG season found in schedule data.")
    return int(complete_seasons.max())


def current_season_and_week(schedule_df):
    """
    The season/week to build a survivor plan for: the earliest season+week
    combination in the schedule with at least one REG-season game that
    hasn't been played yet (home_score is NaN). Assumes the schedule file is
    reasonably current - it's re-fetched fresh (force_refresh=True) at the
    start of every real run in practice, only the dev/test cache skips this.
    """
    reg = schedule_df[schedule_df["game_type"] == "REG"].copy()
    unplayed = reg[reg["home_score"].isna()]
    if unplayed.empty:
        raise RuntimeError("No unplayed REG-season games found - schedule may be stale.")
    season = int(unplayed["season"].min())
    week = int(unplayed[unplayed["season"] == season]["week"].min())
    return season, week


def season_schedule(schedule_df, season, game_type="REG"):
    """This season's schedule only, REG by default (byes/optimizer only
    care about the regular season - a survivor pool doesn't run into
    playoffs)."""
    return schedule_df[
        (schedule_df["season"] == season) & (schedule_df["game_type"] == game_type)
    ].copy()


def compute_byes(season_df):
    """week -> sorted list of team abbrevs on bye that week, derived by
    elimination (every team not playing that week) rather than trusting any
    explicit bye column, since games.csv doesn't have one."""
    all_teams = sorted(set(season_df["home_team"]) | set(season_df["away_team"]))
    weeks = sorted(season_df["week"].unique())
    byes = {}
    for wk in weeks:
        playing = set(season_df[season_df["week"] == wk]["home_team"]) | set(
            season_df[season_df["week"] == wk]["away_team"]
        )
        byes[int(wk)] = sorted(set(all_teams) - playing)
    return byes


if __name__ == "__main__":
    sched = fetch_schedule()
    print(f"Schedule loaded: {len(sched)} games, seasons {sched['season'].min()}-{sched['season'].max()}")

    rating_season = latest_completed_season(sched)
    print(f"Latest fully-completed season (for ratings input): {rating_season}")

    plan_season, plan_week = current_season_and_week(sched)
    print(f"Building a plan for: season {plan_season}, starting week {plan_week}")

    this_season = season_schedule(sched, plan_season)
    print(f"{plan_season} REG schedule: {len(this_season)} games, "
          f"{this_season['week'].nunique()} weeks, "
          f"{len(set(this_season['home_team']) | set(this_season['away_team']))} teams")

    byes = compute_byes(this_season)
    bye_weeks_used = {wk: teams for wk, teams in byes.items() if teams}
    print(f"Bye weeks: {len(bye_weeks_used)} weeks have byes "
          f"({sum(len(t) for t in bye_weeks_used.values())} team-byes total)")
    for wk in sorted(bye_weeks_used):
        print(f"  Week {wk}: {', '.join(bye_weeks_used[wk])}")

    pbp = fetch_pbp(rating_season)
    print(f"\nPBP for {rating_season}: {len(pbp)} plays cached at data/pbp_{rating_season}.parquet")
