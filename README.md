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
2. No API key needed for schedule/play-by-play/weather/injuries - all four
   are free, public, no signup (see "Data sources" below for the specifics
   and the one header-quirk workaround injuries.py needs).
3. The Odds API cross-check is optional - copy `config.example.json` to
   `config.json` and add `odds_api_key` (same key as MLB Edge works fine;
   Odds API keys are account-scoped, not project-scoped - just note both
   projects then share one subscription's request quota), or set
   `ODDS_API_KEY` in the environment. Runs fine with no key configured at
   all; the cross-check just silently skips itself.

## Running it

```
python run_baseline.py     # Phase 1: from-scratch full-season plan (top + alternates)
python run_weekly.py       # Phase 2: re-optimize remaining weeks with fresh data
python run_weekly.py --lock KC   # commit KC as this week's actual pick, log it, re-run
```
Both write `plan.json` (structured plan data) and `dashboard.html` (open in
a browser); `run_weekly.py` also prints the current week's recommendation
with its risk flags and reasoning front and center.

**Normal vs. Circa:** every run produces two plans, shown as tabs on the
dashboard. *Normal* is one pick per week, 18 picks. *Circa Survivor's 2026
structure is 20 legs* - the 18 NFL weeks plus a standalone **Thanksgiving /
Black Friday** leg and a standalone **Christmas** leg, each its own
no-repeat winning pick from that leg's whole multi-day slate (5 games /
10 teams for Thanksgiving-BF, 4 games / 8 teams for Christmas this year).
Leg dates are derived from the schedule, not hardcoded (Thanksgiving is
Week 12 in 2026); `optimizer.CIRCA_LEGS` is the one place to edit if the
contest slate changes. Each pick card on the dashboard has a "see all
games" expander showing that week's / leg's full board with every team's
win probability, so you can see what else you could pick. Known limitation:
the QB-injury haircut applies to Circa leg picks, but the
rest/short-week/weather flags currently only annotate weekly picks.

```
python nfl_data.py      # just the data pipeline, sanity-checks schedule/byes/pbp fetch
python ratings.py       # just the rating model, prints the fitted coefficients + team rankings
python optimizer.py     # just the optimizer, prints top 5 plans (Phase 1 ratings only)
python weather.py       # just this week's outdoor-game forecasts
python injuries.py      # just this week's starting-QB injury flags
python odds_data.py     # just the odds cross-check (prints "skipped" with no key configured)
python calibration.py   # Phase 3: market vs. model calibration check against real history
python trap_games.py    # Phase 3: tests letdown/lookahead/bye/divisional patterns against real history
python popularity.py    # current-week real pick popularity from survivorgrid.com
python field_model.py   # (importable) Circa field-attrition Monte Carlo + payout-share scoring
```

## Data sources

| Source | What it's for | Key needed? |
|---|---|---|
| `raw.githubusercontent.com/nflverse/nfldata` | Schedule, byes, rest days, div games | No |
| `github.com/nflverse/nflverse-data` (pbp releases) | Play-by-play for the rating model | No |
| `api.weather.gov` (NWS) | Forecast for this week's outdoor US games | No - just a descriptive User-Agent header |
| `site.api.espn.com` (unofficial) | League-wide injury reports, incl. starting QB status | No, but requires a `curl`-like User-Agent - see `injuries.py`'s header comment, ESPN's WAF blocks generic script/library User-Agents on this public endpoint |
| The Odds API | Secondary market cross-check only | Yes, optional (see Setup) |
| `survivorgrid.com/picks` | Real current-week survivor pick popularity (Circa payout-share layer) | No - scrapes cleanly, no UA workaround needed |

## Current status

**Phase 1 (baseline full-season plan): done.** Ingestion, rating model, and
season optimizer all run end to end against real live data.

**Phase 2 (weekly re-optimization): done.** `run_weekly.py` re-solves the
remaining season each run with: in-season-updated ratings
(`ratings.build_inseason_ratings` - blends the preseason prior with this
season's own play-by-play, weighted by how much of the season has been
played), a starting-QB injury check applied as an actual probability
haircut (checked against every remaining week's scheduled starter, not
just the current week), the odds-API cross-check, and rest/short-week/
international/divisional/bye-adjacent flags. `picks_log.jsonl` (not created
until the first real `--lock`) tracks what was actually picked and its
eventual result.

**Design choice, stated plainly:** only the starting-QB check moves the
model's actual win probabilities. Rest/travel/weather/divisional-volatility
are surfaced as flags on the reasoning, not silently folded into the
number - see `risk.py`'s module docstring for why (turning those into a
precise multiplier without backtested evidence would be fabricated
precision). Whether any of them deserve a quantified adjustment is exactly
what Phase 3 is for.

**Phase 3 (historical calibration, trap-game detection): done, and it found
a real bug.**

`calibration.py` checked whether "win probability" actually means what it
claims, against real history:
- **Market (devigged closing moneylines, 2010-2025, n=4,161 games): well
  calibrated.** Brier score 0.211, predicted-vs-actual tracks closely
  across every bucket up to 90-95%. The market cross-check in `risk.py`
  can be trusted at face value.
- **This project's own model: was badly overconfident.** The original
  `fit_logistic()` fit b0/b1 against the SAME season the ratings were
  computed from - which turned out to systematically overstate the
  rating-to-win-probability relationship (a team's own EPA numbers in a
  season are partly a *consequence* of that season's own wins). Checked
  against real held-out seasons (2019-2025), predicted 80%+ buckets were
  actually winning only ~65-67% of the time. **Fixed**: `ratings.py` now
  uses `FITTED_B0`/`FITTED_B1`, fit genuinely out-of-sample by
  `calibration.fit_out_of_sample_logistic()` (pooling shrunk prior-season
  ratings against the *next* season's real outcomes across every
  transition from 2010 through 2025). This closed most of the
  overconfidence and is what `run_baseline.py`/`run_weekly.py` actually
  use now - not a hypothetical, the season plan's own numbers changed
  (survival probability dropped from an overconfident 2.78% to a more
  honest 0.21% for the same Week 1 slate). Full details, the validation
  numbers, and the honest remaining limitation (even fixed, a pure
  preseason model still doesn't match the market's accuracy - Brier ~0.239
  vs. 0.211 - rosters change too much year over year for last season's
  play-by-play alone to fully capture) are in `ratings.py`'s `FITTED_B0`
  docstring.

`trap_games.py` tested the classic "trap game" narratives against 16
seasons of real results (2010-2025, n=8,322 team-games) instead of just
asserting them: letdown spots, lookahead spots, coming off a bye, facing a
bye-rested opponent, and divisional-game volatility. **None showed a
statistically distinguishable effect from the market's own baseline
price** - divisional games in particular came back with *exactly* zero
mean bias and slightly *lower* variance than non-divisional games, the
opposite of the popular claim. Deliberately not wired into `risk.py` as
flags as a result - flagging something as risk after checking the data
says it isn't one would be the same fabricated-precision mistake this
project has tried to avoid elsewhere. `short_rest` and
`international/neutral-site` stay as flags but are explicitly labeled
"not yet Phase-3-validated" rather than asserted as proven.

## Payout-share / pick-popularity layer (Circa only) - built, and it moves nothing for one entry

For the real-money **Circa** entry the pot splits equally among all
surviving entries, so a correct pick *off* the popular team eliminates
other entries and grows your share. This got built as a signal layered on
top of win probability, never a replacement:

- **`popularity.py`** - scrapes real current-week pick popularity from
  `survivorgrid.com/picks` (their consensus of Yahoo / ESPN / USA Football
  Pools data). No WAF blocking (unlike ESPN's injuries endpoint); degrades
  to "skip the adjustment" on any failure. Future legs, which have no
  published data yet, use a chalk-seeking heuristic (share proportional to
  `win_prob ** 3`).
- **`field_model.py`** - a season-long Monte Carlo of Circa field
  attrition. Each trajectory draws every remaining leg's game outcomes and
  multiplies the alive fraction by the share of the field that picked a
  winner. `ev_share(team, leg)` = expected `1{team wins} / (1 + FIELD_SIZE
  * field_alive_at_end)` - high when a team wins often *and* its winning
  tends to coincide with a thinned field.
- **The blend** turns `ev_share` into a win-probability delta capped at
  **&plusmn;3 points** (`field_model.POPULARITY_BLEND_WEIGHT` /
  `_CAP`), added before the optimizer re-solves the Circa plan. The
  displayed win probability and survival probability stay the *true*
  (unblended) numbers; popularity and the payout lean are shown separately
  per pick.

**Verified finding, stated plainly (same standard as the calibration and
trap-game work): for a single entry, this changes zero picks.** At the
intended weight it moved 0 of 20. It only starts moving picks at ~10x that
weight - and even then most "changes" are the model swapping to a team
that's *better* on win probability (i.e. resource re-timing, which the
season-wide optimizer already does), not manufactured contrarianism.
Genuine contrarian moves (giving up real win probability) only appear at
weights far past anything the winner research supports. Why: (1) the
global assignment solve already handles resource timing, which is what
actually won Circa 2023-2025; (2) 65% of real winners' picks *were* the
top-1/top-2 popular option - the optimizer's favorites are usually the
popular picks anyway; (3) for 1 entry among ~15,000, the field-thinning
gain from any one leg is smaller than the survival cost of a worse team.

So the layer is kept as a **transparent, capped tiebreaker plus a useful
popularity display** - not a silent no-op, but honestly documented as
changing nothing at safe settings. It would matter more across multiple
diversified entries (Matt is starting with one) or if the published
popularity data shifts hard against a solid team.

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
