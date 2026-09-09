"""
Betting odds from The Odds API - used here strictly as a SECONDARY
cross-check against the power-rating model, never as the primary signal
(see README - this project is not MLB Edge; there's no "beat the market"
goal, just a sanity check on the model's own numbers). Same config pattern
as MLB Edge: ODDS_API_KEY environment variable, or {"odds_api_key": "..."}
in config.json. Same key works across both projects (Odds API keys are
account-scoped, not project-scoped) - they do share one subscription's
monthly request quota, worth knowing if both projects poll frequently.
"""
import json
import os

import requests

API_BASE = "https://api.the-odds-api.com/v4"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
NFL_SPORT_KEY = "americanfootball_nfl"


class NoApiKeyError(RuntimeError):
    pass


def _load_key_from_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return (json.load(f) or {}).get("odds_api_key") or None
    return None


def get_api_key(api_key=None):
    return api_key or os.environ.get("ODDS_API_KEY") or _load_key_from_config()


def moneyline_to_implied_prob(ml):
    ml = float(ml)
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def devig_two_way(p_a, p_b):
    """Removes the vig proportionally so the two sides sum to 1."""
    total = p_a + p_b
    return p_a / total, p_b / total


def get_nfl_odds(api_key=None, regions="us", markets="h2h"):
    """Returns The Odds API's list of upcoming/live NFL events with
    bookmaker moneylines. Raises NoApiKeyError if no key is configured
    anywhere - callers should catch this and degrade gracefully (the
    optimizer's primary signal doesn't depend on this working)."""
    key = get_api_key(api_key)
    if not key:
        raise NoApiKeyError(
            "No Odds API key configured. Set ODDS_API_KEY in the environment, "
            "or put {\"odds_api_key\": \"...\"} in config.json (copy config.example.json)."
        )
    params = {"apiKey": key, "regions": regions, "markets": markets, "oddsFormat": "american"}
    resp = requests.get(f"{API_BASE}/sports/{NFL_SPORT_KEY}/odds/", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def consensus_devigged_probs(event):
    """Average devigged home/away win probability across every bookmaker
    listed for one event. Returns (home_prob, away_prob, num_books)."""
    home_name, away_name = event.get("home_team"), event.get("away_team")
    home_probs, away_probs = [], []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if home_name not in outcomes or away_name not in outcomes:
                continue
            raw_home = moneyline_to_implied_prob(outcomes[home_name])
            raw_away = moneyline_to_implied_prob(outcomes[away_name])
            h, a = devig_two_way(raw_home, raw_away)
            home_probs.append(h)
            away_probs.append(a)
    if not home_probs:
        return None, None, 0
    return sum(home_probs) / len(home_probs), sum(away_probs) / len(away_probs), len(home_probs)


def match_event_to_teams(events, team_abbrev_to_full_name):
    """The Odds API identifies teams by full name ("Kansas City Chiefs"),
    the schedule/ratings use abbreviations ("KC"). Builds {abbrev:
    devigged_prob} from a raw events list plus an abbrev->full-name map so
    callers don't have to do this matching themselves."""
    full_to_abbrev = {v: k for k, v in team_abbrev_to_full_name.items()}
    out = {}
    for event in events:
        home_prob, away_prob, n_books = consensus_devigged_probs(event)
        if home_prob is None:
            continue
        home_abbrev = full_to_abbrev.get(event.get("home_team"))
        away_abbrev = full_to_abbrev.get(event.get("away_team"))
        if home_abbrev:
            out[home_abbrev] = {"market_prob": home_prob, "n_books": n_books}
        if away_abbrev:
            out[away_abbrev] = {"market_prob": away_prob, "n_books": n_books}
    return out


if __name__ == "__main__":
    try:
        key = get_api_key()
        if not key:
            raise NoApiKeyError("no key configured")
        events = get_nfl_odds()
        print(f"Fetched {len(events)} NFL events with odds.")
        for e in events[:5]:
            h, a, n = consensus_devigged_probs(e)
            if h is not None:
                print(f"  {e['away_team']} @ {e['home_team']}: home {h:.1%} / away {a:.1%} ({n} books)")
    except NoApiKeyError as ex:
        print(f"Skipped live check: {ex}")
