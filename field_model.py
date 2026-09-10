"""
Season-long Monte Carlo of Circa field attrition, and a per-(leg, team)
expected-payout-share score built from it.

The payout only resolves at the end (Circa's $20M splits equally among all
entries alive when the legs run out), so the value of differentiating on
any one leg depends on how thin the whole field gets by the end - which
means you have to simulate the rest of the season, not just the next leg.

Method, per simulated season trajectory:
  - Start the field at fraction 1.0 alive.
  - For each remaining leg, draw each game's outcome from the model's win
    probabilities, then multiply the alive fraction by the share of the
    (still-alive) field that had picked a winning team, where "share" comes
    from popularity.field_pick_distribution.
  - Record the field's alive fraction at the end.
For a candidate pick T on leg L: your entry survives L iff T's game is a
win in that trajectory; conditional on that, your eventual payout share is
~ 1 / (1 + FIELD_SIZE * field_alive_fraction_at_end). Averaging
1{T won} / (1 + FIELD_SIZE * alive_end) over trajectories gives
ev_share(T, L) - it is high when T wins often AND T winning tends to
coincide with a thinned field (i.e. T is off the chalk and the chalk is
beatable). That is exactly the "correct pick against the popular team"
payoff, quantified.

Simplifications (deliberate, documented): the field's per-leg pick
distribution is treated as independent leg to leg - real for the current
leg (survivorgrid data reflects the field's actual behavior including
teams already burned), a chalk-seeking heuristic beyond it. FIELD_SIZE is
a fixed nominal (~Circa's real entry count), not simulated to grow/shrink.
Neither materially changes the one question this is here to answer: for a
single entry, does differentiation move any pick vs. pure win-probability
maximization? (see the README write-up).
"""
import random

FIELD_SIZE = 15000          # nominal Circa entry count (2024: ~14,266; 2026: 12k+ and climbing)
DEFAULT_SIMS = 4000

# The payout-share signal is a strict tiebreaker, never an override. At
# these values it verifiably changes ZERO picks for a single entry (see the
# README write-up and run_baseline's payout_blend meta) - the season-wide
# optimizer already does the resource-timing that actually wins Circa, and
# for one entry among ~15k the field-thinning gain from any single leg is
# smaller than the survival cost of a worse team. Kept as a transparent,
# capped mechanism (and a useful popularity display) that would only bite
# if run across multiple diversified entries or if the field data shifts
# hard. Raise WEIGHT knowingly, not by default.
POPULARITY_BLEND_WEIGHT = 0.10
POPULARITY_BLEND_CAP = 0.03   # max win-probability shift (points/100) the blend may apply either way


def simulate(legs, n_sims=DEFAULT_SIMS, field_size=FIELD_SIZE, seed=0):
    """
    `legs`: ordered list of dicts, each:
        {"label": <int week or leg name>,
         "games": [(team_a, p_a, team_b, p_b), ...],   # p_a + p_b == 1
         "field_dist": {team: pick_share}}             # over teams playing this leg, sums ~1
    Returns:
        ev_share:      {(label, team): float}   expected end-of-season payout share
        p_win:         {(label, team): float}   raw model win probability (passthrough, for display)
        field_after:   {label: float}           mean field fraction alive AFTER that leg
    """
    rng = random.Random(seed)
    labels = [lg["label"] for lg in legs]

    won_share_sum = {}   # (label, team) -> sum of 1/(1+N*alive_end) over sims where team won that leg
    won_count = {}       # (label, team) -> sim count where team won that leg
    p_win = {}
    field_after_sum = {lb: 0.0 for lb in labels}

    for lg in legs:
        for (a, pa, b, pb) in lg["games"]:
            p_win[(lg["label"], a)] = pa
            p_win[(lg["label"], b)] = pb
            won_share_sum.setdefault((lg["label"], a), 0.0)
            won_share_sum.setdefault((lg["label"], b), 0.0)
            won_count.setdefault((lg["label"], a), 0)
            won_count.setdefault((lg["label"], b), 0)

    for _ in range(n_sims):
        alive = 1.0
        winners_by_leg = []
        for lg in legs:
            surviving_share = 0.0
            leg_winners = []
            for (a, pa, b, pb) in lg["games"]:
                if rng.random() < pa:
                    w, l = a, b
                else:
                    w, l = b, a
                leg_winners.append(w)
                surviving_share += lg["field_dist"].get(w, 0.0)
            alive *= max(surviving_share, 1.0 / field_size)  # never below one entry
            winners_by_leg.append(leg_winners)
            field_after_sum[lg["label"]] += alive

        share_if_alive = 1.0 / (1.0 + field_size * alive)
        for lg, leg_winners in zip(legs, winners_by_leg):
            for w in leg_winners:
                won_share_sum[(lg["label"], w)] += share_if_alive
                won_count[(lg["label"], w)] += 1

    ev_share = {k: won_share_sum[k] / n_sims for k in won_share_sum}
    field_after = {lb: field_after_sum[lb] / n_sims for lb in labels}
    return ev_share, p_win, field_after


def blend_multipliers(ev_share, legs, cap=POPULARITY_BLEND_CAP, weight=POPULARITY_BLEND_WEIGHT):
    """
    Turns ev_share into a small, capped win-probability *delta* per
    (label, team), to be added to the model's win probability before the
    optimizer re-solves.

    delta = weight * log(ev_share(T) / median ev_share on that leg),
    clamped to +/- `cap`. So popularity can shift an option's effective win
    probability by at most `cap` (default 3 points) either way - enough to
    reorder genuinely close calls, never enough to prefer a meaningfully
    worse team. `weight` scales how aggressively within that cap.
    Returns {(label, team): delta}.
    """
    import math
    by_leg = {}
    for (lb, tm), v in ev_share.items():
        by_leg.setdefault(lb, []).append(v)
    med = {lb: sorted(vs)[len(vs) // 2] for lb, vs in by_leg.items()}

    out = {}
    for (lb, tm), v in ev_share.items():
        base = med.get(lb) or 1e-9
        raw = weight * math.log(max(v, 1e-9) / base)
        out[(lb, tm)] = max(-cap, min(cap, raw))
    return out
