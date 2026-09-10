"""
Shared plan-assembly helpers used by both run_baseline.py and
run_weekly.py, so the two entry points can't drift apart.

`label` throughout is an int NFL week for a normal weekly pick, or a Circa
leg name (str) for a holiday-leg pick.
"""


def game_lookup(schedule_df, legs=()):
    """{(label, team): {opponent, is_home, div_game}} for every team in
    every game - weekly rows keyed by int week, Circa-leg rows keyed by the
    leg name."""
    lut = {}
    for _, g in schedule_df.iterrows():
        wk, home, away, div = int(g["week"]), g["home_team"], g["away_team"], bool(g["div_game"])
        lut[(wk, home)] = {"opponent": away, "is_home": True, "div_game": div}
        lut[(wk, away)] = {"opponent": home, "is_home": False, "div_game": div}
    for s in legs:
        for _, g in s["games"].iterrows():
            home, away, div = g["home_team"], g["away_team"], bool(g["div_game"])
            lut[(s["leg"], home)] = {"opponent": away, "is_home": True, "div_game": div}
            lut[(s["leg"], away)] = {"opponent": home, "is_home": False, "div_game": div}
    return lut


def schedule_slate(matrix, schedule_df, legs=()):
    """
    {label: [ {away, home, away_prob, home_prob, day} ]} - the full game
    slate for every week and every Circa leg, with each team's win
    probability read straight off `matrix` (which must carry a row for
    every label; pass the Circa-extended matrix). Powers the dashboard's
    "see all games / other picks" expander.
    """
    slate = {}
    for wk in sorted(schedule_df["week"].unique()):
        rows = []
        wk_games = schedule_df[schedule_df["week"] == wk].sort_values("gameday")
        for _, g in wk_games.iterrows():
            away, home = g["away_team"], g["home_team"]
            rows.append({
                "away": away, "home": home,
                "away_prob": _cell(matrix, int(wk), away),
                "home_prob": _cell(matrix, int(wk), home),
                "day": str(g.get("weekday") or ""),
                "div_game": bool(g["div_game"]),
            })
        slate[int(wk)] = rows

    for s in legs:
        rows = []
        for _, g in s["games"].sort_values("gameday").iterrows():
            away, home = g["away_team"], g["home_team"]
            rows.append({
                "away": away, "home": home,
                "away_prob": _cell(matrix, s["leg"], away),
                "home_prob": _cell(matrix, s["leg"], home),
                "day": f'{g.get("weekday") or ""} {str(g.get("gameday") or "")}'.strip(),
                "div_game": bool(g["div_game"]),
            })
        slate[s["leg"]] = rows
    return slate


def _cell(matrix, label, team):
    try:
        v = float(matrix.loc[label, team])
        return v if v == v else None  # NaN -> None
    except (KeyError, TypeError):
        return None
