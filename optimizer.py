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


def top_plans(win_prob_matrix, k=5, locked=None):
    """
    High-level entry point. `locked`: dict {week: team} for picks already
    made (Phase 2) - forced into every returned plan. Returns a list of
    dicts, best first:
      {"picks": {week: team, ...}, "survival_prob": float, "log_prob": float}
    survival_prob is the probability of winning every single picked game
    (product across weeks) - the actual quantity being maximized.
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
