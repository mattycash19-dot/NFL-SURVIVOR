"""
Season-long survivor pick optimizer.

Framed as a rectangular assignment problem: 18 weeks (rows) x 32 teams
(columns), cost[week, team] = -log(win probability), solved with
scipy.optimize.linear_sum_assignment (Hungarian/Jonker-Volgenant algorithm).
Minimizing summed cost is exactly maximizing sum(log win_prob), which is
exactly maximizing the probability of winning every single game picked
(product of independent-ish weekly win probabilities) - i.e. maximizing the
probability of running the whole table, which is the actual survivor-pool
objective (winner-take-all, no partial credit for surviving 15 of 18 weeks).
linear_sum_assignment on an 18x32 matrix returns exactly one column (team)
per row (week), each column used at most once - which already encodes both
survivor-pool constraints (one pick per week, no team reused) without any
extra constraint-modeling. Byes (and any other forbidden week/team pair -
see `forced`/`forbidden` below) are encoded as a very large cost, so the
solver will only ever pick one if literally nothing else is feasible.

For the "3-5 near-optimal alternates" requirement, a single call to
linear_sum_assignment only gives the single global optimum - it doesn't
enumerate runner-up solutions. k_best_assignments() implements Murty's
algorithm on top of it: partition the solution space by forcing/forbidding
individual (week, team) pairs and re-solving, keeping a priority queue of
partitions ranked by their best achievable cost. This is the standard exact
method for k-best assignment problems (not a heuristic) - each successive
plan really is the next-best distinct full-season sequence, not just "swap
one pick and see what happens."
"""
import heapq
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import ratings as ratings_mod

LARGE_COST = 1e6

# Circa Survivor's 2026 structure is 20 "legs": the 18 NFL weeks plus two
# standalone selection events - a Thanksgiving / Black Friday leg and a
# Christmas leg - each its own single life-or-death winning pick from that
# leg's whole multi-day slate, with no team reused anywhere across the 20.
# (Confirmed against Circa's official 2026 rules - see the project's vault
# note.) Dates are derived from the schedule, not hardcoded (Thanksgiving is
# Week 12 in 2026). Edit CIRCA_LEGS if the contest's slate changes.
CIRCA_LEGS = (
    {"name": "Thanksgiving / Black Friday", "days": ("thanksgiving_eve", "thanksgiving", "black_friday")},
    {"name": "Christmas", "days": ("christmas_eve", "christmas")},
)
_LEG_SUBORDER = {"Thanksgiving / Black Friday": 0.5, "Christmas": 0.5}


def _leg_day_dates(df):
    """Resolve each named holiday day to an actual date present in `df`
    (a schedule with a parsed `_d` column). Thanksgiving = the November
    Thursday carrying the most games; its eve/Friday are the days on either
    side. Christmas = Dec 25, its eve = Dec 24. Missing days -> None."""
    nov_thu = df[(df["_d"].dt.month == 11) & (df["_d"].dt.weekday == 3)]
    tg = nov_thu.groupby(nov_thu["_d"]).size().idxmax() if not nov_thu.empty else None
    xmas = pd.Timestamp(year=int(df["_d"].dt.year.mode().iloc[0]), month=12, day=25)
    return {
        "thanksgiving_eve": tg - pd.Timedelta(days=1) if tg is not None else None,
        "thanksgiving": tg,
        "black_friday": tg + pd.Timedelta(days=1) if tg is not None else None,
        "christmas_eve": xmas - pd.Timedelta(days=1),
        "christmas": xmas,
    }


def circa_legs(season_df, legs=CIRCA_LEGS):
    """
    The Circa standalone legs present in `season_df`, as an ordered list of
    {leg, dates (list of str), week, games (DataFrame - every game across
    the leg's days)}. A leg with no games in this schedule (already past)
    is omitted.
    """
    df = season_df.copy()
    df["_d"] = pd.to_datetime(df["gameday"])
    day_dates = _leg_day_dates(df)
    out = []
    for leg in legs:
        wanted = [day_dates[d] for d in leg["days"] if day_dates.get(d) is not None]
        games = df[df["_d"].isin(wanted)]
        if games.empty:
            continue
        out.append({
            "leg": leg["name"],
            "dates": sorted(str(d.date()) for d in games["_d"].unique()),
            "week": int(games["week"].min()),
            "games": games.drop(columns=["_d"]),
        })
    out.sort(key=lambda s: s["week"] + _LEG_SUBORDER.get(s["leg"], 0.5))
    return out


def append_circa_legs(win_prob_matrix, legs, team_ratings, b0, b1):
    """
    Appends one row per Circa leg (row label = the leg's name) to a weekly
    win-prob matrix - the row carries a win probability for every team
    playing on any of that leg's days. Returns (extended_matrix,
    slot_order); slot_order maps every row label, int weeks included, to a
    float for chronological sorting.
    """
    ext = win_prob_matrix.copy()
    order = {w: float(w) for w in win_prob_matrix.index}
    for s in legs:
        label = s["leg"]
        row = pd.Series(index=ext.columns, dtype=float)
        for _, g in s["games"].iterrows():
            home, away = g["home_team"], g["away_team"]
            if home not in team_ratings.index or away not in team_ratings.index:
                continue
            p_home = ratings_mod.win_probability(team_ratings[home], team_ratings[away], b0, b1)
            row[home] = p_home
            row[away] = 1.0 - p_home
        ext.loc[label] = row
        order[label] = s["week"] + _LEG_SUBORDER.get(label, 0.5)
    return ext, order


def build_win_prob_matrix(season_df, team_ratings, b0, b1):
    """
    DataFrame indexed by week (1..N), one column per team, value = that
    team's win probability in its game that week. NaN where the team is on
    bye that week (no game to pick).
    """
    teams = sorted(team_ratings.index)
    weeks = sorted(season_df["week"].unique())
    mat = pd.DataFrame(index=weeks, columns=teams, dtype=float)
    for _, g in season_df.iterrows():
        wk, home, away = int(g["week"]), g["home_team"], g["away_team"]
        if home not in team_ratings.index or away not in team_ratings.index:
            continue
        p_home = ratings_mod.win_probability(team_ratings[home], team_ratings[away], b0, b1)
        mat.loc[wk, home] = p_home
        mat.loc[wk, away] = 1.0 - p_home
    return mat


def _prob_matrix_to_cost(win_prob_matrix):
    """DataFrame of probabilities -> numpy cost matrix (-log p), NaN/0 -> LARGE_COST."""
    weeks = win_prob_matrix.index.tolist()
    teams = win_prob_matrix.columns.tolist()
    cost = np.full((len(weeks), len(teams)), LARGE_COST)
    vals = win_prob_matrix.values
    valid = np.isfinite(vals) & (vals > 0)
    cost[valid] = -np.log(vals[valid])
    return cost, weeks, teams


def _solve_sub(cost_matrix, forced_pairs, forbidden_pairs, n_rows, n_cols, large):
    """One Murty partition: `forced_pairs` must all be in the solution,
    `forbidden_pairs` must not be. Returns (assignment_list, total_cost) or
    None if infeasible (a forced pair itself costs >= large, or no feasible
    completion exists for the remaining free rows/cols)."""
    forced_rows = {r for r, c in forced_pairs}
    forced_cols = {c for r, c in forced_pairs}
    forced_cost = sum(cost_matrix[r, c] for r, c in forced_pairs)
    if forced_cost >= large:
        return None

    free_rows = [r for r in range(n_rows) if r not in forced_rows]
    free_cols = [c for c in range(n_cols) if c not in forced_cols]
    if not free_rows:
        return list(forced_pairs), forced_cost

    sub = cost_matrix[np.ix_(free_rows, free_cols)].copy()
    row_pos = {r: i for i, r in enumerate(free_rows)}
    col_pos = {c: i for i, c in enumerate(free_cols)}
    for (r, c) in forbidden_pairs:
        if r in row_pos and c in col_pos:
            sub[row_pos[r], col_pos[c]] = large

    rr, cc = linear_sum_assignment(sub)
    sub_cost = sub[rr, cc].sum()
    if sub_cost >= large:
        return None
    assignment = list(forced_pairs) + [(free_rows[i], free_cols[j]) for i, j in zip(rr, cc)]
    return assignment, forced_cost + sub_cost


def k_best_assignments(cost_matrix, k, forced=(), large=LARGE_COST):
    """
    Murty's algorithm. `forced` pre-fixes any (row, col) pairs that must be
    in every returned solution (used to lock already-made picks in Phase 2's
    weekly re-optimization). Returns up to k (assignment, total_cost) tuples,
    best first, all distinct as full assignments.
    """
    n_rows, n_cols = cost_matrix.shape
    forced = tuple(forced)
    counter = itertools.count()
    heap = []
    results = []
    seen = set()

    root = _solve_sub(cost_matrix, list(forced), [], n_rows, n_cols, large)
    if root is None:
        return []
    heapq.heappush(heap, (root[1], next(counter), root[0], forced, ()))

    while heap and len(results) < k:
        cost, _, assignment, node_forced, node_forbidden = heapq.heappop(heap)
        key = tuple(sorted(assignment))
        if key in seen:
            continue
        seen.add(key)
        results.append((assignment, cost))

        # Partition on the edges this solution added beyond node_forced,
        # in row order, for a canonical (non-overlapping) split of the
        # remaining solution space - the standard Murty partition.
        branchable = sorted((e for e in assignment if e not in node_forced), key=lambda e: e[0])
        cur_forced = list(node_forced)
        for edge in branchable:
            new_forbidden = node_forbidden + (edge,)
            sub = _solve_sub(cost_matrix, cur_forced, list(new_forbidden), n_rows, n_cols, large)
            if sub is not None:
                heapq.heappush(heap, (sub[1], next(counter), sub[0], tuple(cur_forced), new_forbidden))
            cur_forced = cur_forced + [edge]

    return results


def top_plans(win_prob_matrix, k=5, locked=None, slot_order=None):
    """
    High-level entry point. `locked`: dict {slot: team} for picks already
    made (Phase 2) - forced into every returned plan. `slot_order`: optional
    {row_label: float} to sort picks chronologically when the matrix has
    non-integer rows (Circa legs - see append_circa_legs).
    Returns a list of dicts, best first:
      {"picks": {slot: team, ...}, "survival_prob": float, "log_prob": float}
    survival_prob is the probability of winning every single picked game
    (product across picks) - the actual quantity being maximized.
    """
    cost, weeks, teams = _prob_matrix_to_cost(win_prob_matrix)
    forced_pairs = []
    if locked:
        for wk, team in locked.items():
            if wk not in weeks or team not in teams:
                raise ValueError(f"Locked pick week={wk} team={team} not in this matrix.")
            forced_pairs.append((weeks.index(wk), teams.index(team)))

    raw = k_best_assignments(cost, k, forced=forced_pairs)
    plans = []
    for assignment, total_cost in raw:
        picks = {weeks[r]: teams[c] for r, c in assignment}
        if slot_order:
            picks = dict(sorted(picks.items(), key=lambda kv: slot_order.get(kv[0], 1e9)))
        else:
            picks = dict(sorted(picks.items()))
        plans.append({
            "picks": picks,
            "survival_prob": float(np.exp(-total_cost)),
            "log_prob": float(-total_cost),
        })
    return plans


if __name__ == "__main__":
    import nfl_data

    sched = nfl_data.fetch_schedule()
    rating_season = nfl_data.latest_completed_season(sched)
    plan_season, plan_week = nfl_data.current_season_and_week(sched)
    pbp = nfl_data.fetch_pbp(rating_season)
    rating_input_schedule = nfl_data.season_schedule(sched, rating_season)
    plan_schedule = nfl_data.season_schedule(sched, plan_season)

    team_ratings, b0, b1 = ratings_mod.build_preseason_ratings(pbp, rating_input_schedule)
    matrix = build_win_prob_matrix(plan_schedule, team_ratings, b0, b1)

    plans = top_plans(matrix, k=5)
    print(f"Season {plan_season} - top {len(plans)} plans (of {matrix.shape[0]} weeks x {matrix.shape[1]} teams):\n")
    for i, plan in enumerate(plans, 1):
        print(f"Plan #{i} - survival probability: {plan['survival_prob']:.2%}")
        for wk, team in plan["picks"].items():
            p = matrix.loc[wk, team]
            print(f"  Week {wk:>2}: {team:<4} ({p:.1%})")
        print()
