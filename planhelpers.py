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


def build_leg_specs(matrix, schedule_df, legs, current_week, real_popularity):
    """
    Ordered leg specs for field_model.simulate() - one entry per NFL week
    plus one per Circa leg, each with its games as (away, p_away, home,
    p_home) and the field's expected pick distribution over the teams
    playing it. The current week uses real survivorgrid popularity; every
    other leg uses the win-probability-derived heuristic (see
    popularity.field_pick_distribution).
    """
    import popularity

    def _leg(label, games_df, is_current):
        games, wp = [], {}
        for _, g in games_df.iterrows():
            a, h = g["away_team"], g["home_team"]
            pa, ph = _cell(matrix, label, a), _cell(matrix, label, h)
            if pa is None or ph is None:
                continue
            games.append((a, pa, h, ph))
            wp[a], wp[h] = pa, ph
        if not games:
            return None
        rp = real_popularity if is_current else None
        return {"label": label, "games": games,
                "field_dist": popularity.field_pick_distribution(wp, rp)}

    specs = []
    for wk in sorted(schedule_df["week"].unique()):
        spec = _leg(int(wk), schedule_df[schedule_df["week"] == wk], int(wk) == current_week)
        if spec:
            specs.append(spec)
    for s in legs:
        spec = _leg(s["leg"], s["games"], is_current=False)
        if spec:
            specs.append(spec)
    return specs


def apply_payout_delta(matrix, deltas):
    """Adds field_model's small capped win-probability deltas to `matrix`,
    clamped to (0.01, 0.99). Returns a new matrix - the caller keeps the
    original for displaying true win probabilities."""
    out = matrix.copy()
    for (lb, tm), d in deltas.items():
        if lb in out.index and tm in out.columns:
            v = _cell(out, lb, tm)
            if v is not None:
                out.loc[lb, tm] = min(0.99, max(0.01, v + d))
    return out


def payout_blend_meta(pure_picks, blended_picks, true_matrix, weight, cap, field_after, field_size):
    """Summary of what the payout-share blend actually did to the top Circa
    plan, for the dashboard/README - how many picks it changed vs. the pure
    win-probability plan, and the biggest true-win-probability sacrifice on
    any changed pick."""
    changed = []
    for lb, pt in pure_picks.items():
        bt = blended_picks.get(lb)
        if bt and bt != pt:
            pp, bp = _cell(true_matrix, lb, pt), _cell(true_matrix, lb, bt)
            changed.append({"leg": str(lb), "pure": pt, "blended": bt,
                            "winprob_sacrificed": round((pp - bp), 4) if (pp and bp) else None})
    worst = max((c["winprob_sacrificed"] or 0) for c in changed) if changed else 0.0
    return {
        "weight": weight, "cap": cap,
        "picks_changed": len(changed), "changes": changed,
        "max_winprob_sacrificed": round(worst, 4),
        "field_alive_after": {str(k): round(v, 5) for k, v in field_after.items()},
        "field_size": field_size,
    }


def true_survival_prob(picks, matrix):
    """Product of the *true* (unblended) win probabilities across a plan's
    picks - what the plan's headline survival number should report even
    when the optimizer solved on a popularity-blended matrix."""
    import math
    total = 0.0
    for slot, team in picks.items():
        p = _cell(matrix, slot, team)
        if p and p > 0:
            total += math.log(p)
    return math.exp(total)
