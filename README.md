# NFL Survivor Pool Optimizer

A season-long pick-sequencing tool for a ~20-person winner-take-all NFL
survivor pool: pick one team per week to win straight-up, no repeats, out
for the season if your pick loses. No partial payouts, so the objective is
**pure survival-probability maximization** - not finding betting value.
This is a different problem from [MLB Edge](../MLB-EDGE) on purpose: there's
no "beat the market" concept here, just "maximize the odds of picking 18
correct winners in a row without reusing a team."

## What it does (Phase 1, built so far)

1. **Ingests the full schedule** - all 32 teams, all 18 weeks, byes derived
   by elimination - from nflverse's free public schedule file (no API key).
2. **Builds a win-probability matrix** (every team x every week it plays)
   from an independent power-rating model: team EPA/play + success rate on
   offense and defense (from nflverse play-by-play, prior completed season),
   blended into a composite rating, fit against real game outcomes via
   logistic regression (an Elo-style win-probability formula with
   empirically-fit coefficients rather than borrowed chess constants), then
   regressed toward the league mean for the new season - see `ratings.py`'s
   module docstring for the full methodology and the specific numbers
   currently assumed (70/30 EPA/success blend, 0.72 shrinkage factor).
3. **Solves the season-long assignment**: exactly one team per week, each
   team at most once, no byes, maximizing sum(log win probability) - which
   is exactly maximizing the probability of winning every single picked
   game (running the table). Framed as a real assignment problem
   (`scipy.optimize.linear_sum_assignment` on an 18-week x 32-team cost
   matrix), not a greedy "biggest favorite each week" heuristic - a greedy
   pick can burn a great team on an easy week and regret it later; this
   looks at the whole season at once.
4. **Outputs the top plan plus near-optimal alternates** via a real Murty's
   algorithm k-best implementation (`optimizer.k_best_assignments`) - each
   alternate is genuinely the next-best distinct full-season sequence, not
   a heuristic "swap one pick" variant.

## Setup

1. Needs Python 3 with `numpy`, `scipy`, `pandas`, `pyarrow`, `requests`
   (`pip install numpy scipy pandas pyarrow requests`).
2. No API key needed for Phase 1 - nflverse's schedule and play-by-play
   data are both free, public, no signup.
3. Phase 2 (not built yet) will add The Odds API as a secondary cross-check
   - copy `config.example.json` to `config.json` and add `odds_api_key`
   (same key as MLB Edge - Odds API keys aren't project-scoped), or set
   `ODDS_API_KEY` in the environment. A weather API key for Phase 2's
   outdoor-game weather check hasn't been decided/provided yet.

## Running it

```
python run_baseline.py
```
Writes `plan.json` (structured plan data) and `dashboard.html` (open in a
browser). Also prints the top plan to the console.

```
python nfl_data.py      # just the data pipeline, sanity-checks schedule/byes/pbp fetch
python ratings.py       # just the rating model, prints the fitted coefficients + team rankings
python optimizer.py     # just the optimizer, prints top 5 plans
```

## Current status

**Phase 1 (baseline full-season plan): done.** Ingestion, rating model, and
season optimizer all run end to end against real live data.

**Phase 2 (weekly re-optimization) and Phase 3 (historical calibration,
trap-game detection): not built yet.** Phase 2 needs decisions on exactly
which injury-report and weather sources to use and, for weather, an API key
that hasn't been provided - see the project's open questions before
building it. Phase 2 will also add: updated in-season ratings (not just the
preseason prior used today), the odds-API cross-check, a `run_weekly.py`
that locks only the current week's pick and re-solves everything after it,
and `picks_log.jsonl` for tracking actual picks made and their results
(same pattern as MLB Edge's `predictions_log.jsonl`).

## Design notes / things deliberately not done yet

- **No live/automated pick submission.** This produces a recommendation;
  making the actual pick in the pool is a manual step, on purpose.
- **No contrarian/differentiation strategy.** In a ~20-person pool, being
  the only survivor left doesn't pay any differently than splitting with
  others - so there's no reason to weight "how many other people picked
  this team" the way a large-field survivor pool strategy would. If pool
  size ever changes meaningfully, revisit this assumption.
- **Ratings use only the most recent fully-completed season.** Blending in
  multiple prior seasons (with recency weighting) is a reasonable future
  improvement once there's a calibration check (Phase 3) to justify it -
  not done speculatively.
- **The rating model's blend weights and shrinkage factor are documented
  starting assumptions, not fitted values** - see `ratings.py`. Phase 3's
  historical calibration check is the intended way to revisit them with
  evidence instead of guessing further.
