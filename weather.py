"""
Outdoor-game weather via the National Weather Service's public API
(api.weather.gov) - official US government source, completely free, no
signup, no API key. Requests just need a descriptive User-Agent identifying
the app (NWS usage policy, not an auth mechanism).

Only covers US locations and only forecasts ~7 days out, both of which are
fine for this project's actual need ("weather for outdoor games THIS
week"). Two real gaps, both handled by returning an explicit "unavailable"
result rather than guessing:
  - International/neutral-site games (schedule `location` != "Home") -
    outside NWS's coverage entirely.
  - Dome/fixed-roof stadiums, and retractable-roof stadiums whose roof
    state for that specific game isn't decided/recorded yet - weather
    doesn't apply, or doesn't apply yet.
"""
import requests

import stadiums

BASE = "https://api.weather.gov"
HEADERS = {"User-Agent": "nfl-survivor-model (mattycash19@gmail.com)"}


def _get(url):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def is_outdoor_this_game(home_team, roof_value):
    """
    Decide whether THIS specific game is actually exposed to weather.
    roof_value: the schedule's `roof` column for this game (a string like
    "outdoors"/"closed"/"open", or NaN/None if not decided/recorded yet).
    """
    info = stadiums.STADIUMS.get(home_team)
    if info is None:
        return False  # unknown venue (shouldn't happen for a "Home" game)
    if info["roof_type"] == "dome":
        return False
    if info["roof_type"] == "outdoor":
        return True
    # retractable: trust the actual recorded state if we have one
    if isinstance(roof_value, str):
        return roof_value.strip().lower() in ("outdoors", "open")
    return None  # unknown yet - not "not outdoor", genuinely undecided


def forecast_for_game(home_team, gameday, location="Home"):
    """
    Returns a dict describing the forecast, or a dict with "available":
    False and a `reason` if we can't get one. `gameday` is only used to
    pick the closest matching forecast period (NWS gives ~7 daily periods,
    not a specific date query) - if the game is further out than the
    forecast covers, this returns unavailable rather than showing a
    different day's weather mislabeled as game day.
    """
    if location != "Home":
        return {"available": False, "reason": "international/neutral-site game - outside NWS coverage"}
    if not stadiums.is_us_venue(home_team):
        return {"available": False, "reason": "no stadium coordinates on file for this venue"}

    info = stadiums.STADIUMS[home_team]
    points = _get(f"{BASE}/points/{info['lat']},{info['lon']}")
    forecast_url = points["properties"].get("forecast")
    if not forecast_url:
        return {"available": False, "reason": "NWS did not return a forecast URL for this location"}

    fc = _get(forecast_url)
    periods = fc["properties"]["periods"]

    match = next((p for p in periods if p["startTime"][:10] == str(gameday)), None)
    if match is None:
        return {"available": False, "reason": f"game date {gameday} is outside NWS's ~7-day forecast window"}

    return {
        "available": True,
        "period_name": match["name"],
        "temperature_f": match["temperature"],
        "wind": match["windSpeed"],
        "wind_direction": match.get("windDirection"),
        "short_forecast": match["shortForecast"],
        "precipitation_chance": (match.get("probabilityOfPrecipitation") or {}).get("value"),
    }


if __name__ == "__main__":
    import nfl_data

    sched = nfl_data.fetch_schedule()
    plan_season, plan_week = nfl_data.current_season_and_week(sched)
    week_games = sched[
        (sched["season"] == plan_season) & (sched["game_type"] == "REG") & (sched["week"] == plan_week)
    ]

    print(f"Weather check for {plan_season} Week {plan_week} ({len(week_games)} games):\n")
    for _, g in week_games.iterrows():
        home = g["home_team"]
        outdoor = is_outdoor_this_game(home, g.get("roof"))
        if outdoor is False:
            print(f"  {g['away_team']} @ {home}: indoor/dome, no weather factor")
            continue
        if outdoor is None:
            print(f"  {g['away_team']} @ {home}: retractable roof, state not decided yet")
            continue
        fc = forecast_for_game(home, g["gameday"], g.get("location", "Home"))
        if not fc["available"]:
            print(f"  {g['away_team']} @ {home}: no forecast ({fc['reason']})")
        else:
            print(f"  {g['away_team']} @ {home}: {fc['short_forecast']}, {fc['temperature_f']}F, "
                  f"wind {fc['wind']}, precip {fc['precipitation_chance']}%")
