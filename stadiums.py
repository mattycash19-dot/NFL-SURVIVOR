"""
Static per-team home-stadium reference: coordinates (for the weather API)
and roof type. Same pattern as MLB Edge's park_factors.py - a small,
stable, publicly-verifiable table that doesn't need a live API. Coordinates
are approximate (stadium centroid, plenty precise for a weather lookup);
roof_type is the structural fact (can this venue ever be open to weather),
which is intentionally kept separate from the schedule's own `roof` column
(actual open/closed state for a specific game) - see weather.py for how the
two combine. Spot-check before trusting blindly if a franchise moves or
renames its stadium; not re-verified live each run.

roof_type values: "outdoor" (always exposed), "dome" (fixed, never
exposed), "retractable" (can go either way - actual state for a given game
comes from the schedule's `roof` column when it's been decided, which for
retractable venues is often NOT populated until close to game day).
"""

STADIUMS = {
    "ARI": {"lat": 33.5276, "lon": -112.2626, "roof_type": "retractable"},
    "ATL": {"lat": 33.7554, "lon": -84.4008, "roof_type": "retractable"},
    "BAL": {"lat": 39.2780, "lon": -76.6227, "roof_type": "outdoor"},
    "BUF": {"lat": 42.7738, "lon": -78.7870, "roof_type": "outdoor"},
    "CAR": {"lat": 35.2258, "lon": -80.8528, "roof_type": "outdoor"},
    "CHI": {"lat": 41.8623, "lon": -87.6167, "roof_type": "outdoor"},
    "CIN": {"lat": 39.0954, "lon": -84.5160, "roof_type": "outdoor"},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "roof_type": "outdoor"},
    "DAL": {"lat": 32.7473, "lon": -97.0945, "roof_type": "retractable"},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "roof_type": "outdoor"},
    "DET": {"lat": 42.3400, "lon": -83.0456, "roof_type": "dome"},
    "GB":  {"lat": 44.5013, "lon": -88.0622, "roof_type": "outdoor"},
    "HOU": {"lat": 29.6847, "lon": -95.4107, "roof_type": "retractable"},
    "IND": {"lat": 39.7601, "lon": -86.1639, "roof_type": "retractable"},
    "JAX": {"lat": 30.3239, "lon": -81.6373, "roof_type": "outdoor"},
    "KC":  {"lat": 39.0489, "lon": -94.4839, "roof_type": "outdoor"},
    "LA":  {"lat": 33.9535, "lon": -118.3392, "roof_type": "dome"},   # SoFi - fixed roof, effectively indoor
    "LAC": {"lat": 33.9535, "lon": -118.3392, "roof_type": "dome"},   # shares SoFi with LA
    "LV":  {"lat": 36.0909, "lon": -115.1833, "roof_type": "dome"},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "roof_type": "outdoor"},
    "MIN": {"lat": 44.9737, "lon": -93.2577, "roof_type": "dome"},
    "NE":  {"lat": 42.0909, "lon": -71.2643, "roof_type": "outdoor"},
    "NO":  {"lat": 29.9511, "lon": -90.0812, "roof_type": "dome"},
    "NYG": {"lat": 40.8135, "lon": -74.0745, "roof_type": "outdoor"},  # MetLife - shares with NYJ
    "NYJ": {"lat": 40.8135, "lon": -74.0745, "roof_type": "outdoor"},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "roof_type": "outdoor"},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "roof_type": "outdoor"},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "roof_type": "outdoor"},  # partial canopy over seats, field is open
    "SF":  {"lat": 37.4030, "lon": -121.9700, "roof_type": "outdoor"},
    "TB":  {"lat": 27.9759, "lon": -82.5033, "roof_type": "outdoor"},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "roof_type": "outdoor"},
    "WAS": {"lat": 38.9077, "lon": -76.8645, "roof_type": "outdoor"},
}


def is_us_venue(team_or_location):
    """True if `team_or_location` is a stadium we have US coordinates for.
    Used to decide whether weather.py can even attempt a forecast - a
    neutral-site international game (London, Mexico City, Madrid, etc, all
    of which show up in the schedule's `location` column as "Neutral") has
    no entry here and should be flagged, not guessed at."""
    return team_or_location in STADIUMS
