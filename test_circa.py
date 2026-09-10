"""
Verification tests for the Circa-mode refinements (2026-09-10 strategy pass).
Run: python test_circa.py   (plain asserts + prints, no pytest needed)

Covers:
  1. Holiday-leg single-use scarcity: when an elite team is a strong option
     in BOTH the Thanksgiving and Christmas legs, the optimizer assigns it
     to the leg where it's the bigger favorite and finds a genuine
     next-best team for the other leg.
  2. The expected-legs-survived tiebreaker: among plans tied on total
     survival probability, the one that front-loads its safest picks wins,
     and the tiebreak never lowers total survival probability.
  3. future_value_teams(): identifies multi-leg favorites from the matrix,
     not a hardcoded list.
  4. A real-2026-data smoke check that the refinements don't move today's
     top Circa pick set.
"""
import numpy as np
import pandas as pd

import optimizer
import planhelpers


def _matrix(rows):
    """rows = {label: {team: prob}} -> DataFrame with every team as a column."""
    teams = sorted({t for r in rows.values() for t in r})
    m = pd.DataFrame(index=list(rows), columns=teams, dtype=float)
    for lab, r in rows.items():
        for t, p in r.items():
            m.loc[lab, t] = p
    return m


def test_holiday_scarcity_synthetic():
    # Isolate the holiday-leg choice: the two regular weeks each have exactly
    # one eligible team (so they're forced and don't distort the trade-off),
    # leaving the only real decision as ELITE vs NB across the two holiday
    # legs. ELITE is a strong option in BOTH; it's a bigger favorite in the
    # Thanksgiving leg (0.82) than at Christmas (0.71). NB is the genuine
    # next-best holiday option; SCRUB is a distractor that must NOT be chosen.
    m = _matrix({
        1: {"W1T": 0.70},
        2: {"W2T": 0.70},
        "Thanksgiving / Black Friday": {"ELITE": 0.82, "NB": 0.60, "SCRUB": 0.45},
        "Christmas":                   {"ELITE": 0.71, "NB": 0.66, "SCRUB": 0.40},
    })
    order = {1: 1, 2: 2, "Thanksgiving / Black Friday": 2.5, "Christmas": 3.5}
    plan = optimizer.top_plans(m, k=1, slot_order=order)[0]["picks"]

    used = list(plan.values())
    assert len(used) == len(set(used)), f"a team was reused: {plan}"

    tg, xm = plan["Thanksgiving / Black Friday"], plan["Christmas"]
    assert tg == "ELITE", f"ELITE should take its stronger holiday leg (Thanksgiving 0.82 > Christmas 0.71); got {tg}"
    assert xm == "NB", f"the other holiday leg should get the genuine next-best (NB=0.66), not SCRUB; got {xm}"
    # sanity: ELITE-in-weaker-leg alternative really is worse
    alt = 0.71 * 0.60   # ELITE->Christmas, NB->Thanksgiving
    chosen = m.loc["Thanksgiving / Black Friday", "ELITE"] * m.loc["Christmas", "NB"]
    assert chosen > alt, "the chosen holiday split should beat the flipped one"
    print("  [ok] holiday scarcity (synthetic): ELITE -> Thanksgiving (bigger edge), Christmas -> next-best NB")


def test_holiday_scarcity_real_2026():
    import nfl_data, ratings as R
    sched = nfl_data.fetch_schedule()
    pbp = nfl_data.fetch_pbp(nfl_data.latest_completed_season(sched))
    psn, _ = nfl_data.current_season_and_week(sched)
    ps = nfl_data.season_schedule(sched, psn)
    tr, b0, b1 = R.build_preseason_ratings(pbp)
    legs = optimizer.circa_legs(ps)
    cm, order = optimizer.build_circa_matrix(ps, legs, tr, b0, b1)

    tg_pool = set(cm.loc["Thanksgiving / Black Friday"].dropna().index)
    xm_pool = set(cm.loc["Christmas"].dropna().index)
    both = tg_pool & xm_pool
    assert both, "expected some teams eligible for both holiday legs in 2026"

    plan = optimizer.top_plans(cm, k=1, slot_order=order)[0]["picks"]
    tg_pick, xm_pick = plan["Thanksgiving / Black Friday"], plan["Christmas"]
    assert tg_pick != xm_pick, "same team assigned to both holiday legs"

    # whichever both-legs team the plan uses in a holiday leg, it must be in
    # that team's stronger holiday leg (or a regular week, which is also fine)
    for t in both:
        legs_used_in = [s for s, tm in plan.items() if tm == t]
        for s in legs_used_in:
            if s in ("Thanksgiving / Black Friday", "Christmas"):
                p_here = cm.loc[s, t]
                p_other = cm.loc["Christmas" if s == "Thanksgiving / Black Friday" else "Thanksgiving / Black Friday", t]
                assert p_here >= p_other - 1e-9, (
                    f"{t} placed in its weaker holiday leg ({s} {p_here:.3f} < {p_other:.3f})")

    # neither holiday leg got stuck with a below-median forced pick
    for leg in ("Thanksgiving / Black Friday", "Christmas"):
        avail = cm.loc[leg].dropna().sort_values(ascending=False)
        pick_p = cm.loc[leg, plan[leg]]
        assert pick_p >= avail.median() - 1e-9, (
            f"{leg} pick {plan[leg]} ({pick_p:.3f}) is below the leg's median option {avail.median():.3f}")
    print(f"  [ok] holiday scarcity (real 2026): TG->{tg_pick}, XMAS->{xm_pick}; both-legs teams {sorted(both)} resolved to stronger side")


def test_tiebreaker_frontloads_safety():
    # Global optimum is unique (A@1, D@2 = 0.81). The 2nd/3rd plans are a
    # genuine tie at 0.63: {A@1,C@2} and {B@1,D@2}. The tiebreak must put
    # the front-loaded-safe one ({A@1=0.9 first}) ahead.
    m = _matrix({
        1: {"A": 0.90, "B": 0.70},
        2: {"C": 0.70, "D": 0.90},
    })
    plans = optimizer.top_plans(m, k=3)
    sp = [round(p["survival_prob"], 6) for p in plans]
    assert sp[0] == 0.81, f"expected unique optimum 0.81 first, got {sp}"
    assert sp[1] == sp[2] == 0.63, f"expected a 0.63 tie in slots 2-3, got {sp}"
    assert plans[1]["picks"] == {1: "A", 2: "C"}, (
        f"tiebreak should front-load safety (A=0.9 in wk1); got {plans[1]['picks']}")
    assert plans[1]["expected_legs_survived"] > plans[2]["expected_legs_survived"]
    # tiebreak never sacrifices total probability
    assert all(sp[i] >= sp[i + 1] for i in range(len(sp) - 1)), "plans not in non-increasing survival order"
    print("  [ok] tiebreaker: tied plans ordered by front-loaded safety, total probability never lowered")


def test_future_value_from_matrix():
    # X is a top-3 favorite in 5 legs, Y in only 3.
    rows = {}
    for i in range(1, 6):
        rows[i] = {"X": 0.80, "P": 0.40, "Q": 0.30, "R": 0.20}   # X top-3 (actually top-1) all 5
    for i in range(6, 9):
        rows[i] = {"Y": 0.75, "P": 0.35, "Q": 0.25, "R": 0.15}   # Y top-3 in 3
    m = _matrix(rows)
    fv = planhelpers.future_value_teams(m, plan_picks={3: "X"}, min_legs=4, top_n=3)
    teams = {d["team"]: d for d in fv}
    assert "X" in teams and teams["X"]["leg_count"] == 5, fv
    assert "Y" not in teams, f"Y (3 legs) is below the 4-leg threshold: {fv}"
    assert teams["X"]["spent_at"] == "3", f"spent_at should report where the plan uses X: {teams['X']}"
    print("  [ok] future_value_teams: multi-leg favorites derived from the matrix, threshold + spent_at honored")


def test_refinements_dont_move_todays_circa_picks():
    import nfl_data, ratings as R
    sched = nfl_data.fetch_schedule()
    pbp = nfl_data.fetch_pbp(nfl_data.latest_completed_season(sched))
    psn, _ = nfl_data.current_season_and_week(sched)
    ps = nfl_data.season_schedule(sched, psn)
    tr, b0, b1 = R.build_preseason_ratings(pbp)
    legs = optimizer.circa_legs(ps)
    cm, order = optimizer.build_circa_matrix(ps, legs, tr, b0, b1)

    top = optimizer.top_plans(cm, k=5, slot_order=order)
    # the tiebreak only reorders plans whose survival prob rounds equal to 5
    # decimals; with 20 real-valued probs the top plan's product is unique,
    # so plan[0] is the same pick set a plain product-max solve would give.
    best_sp = max(p["survival_prob"] for p in top)
    tied_with_best = [p for p in top if round(p["survival_prob"], 5) == round(best_sp, 5)]
    assert top[0]["survival_prob"] == best_sp, "plan[0] is not the max-survival plan"
    assert len(tied_with_best) == 1, (
        f"today's top Circa plan is in a {len(tied_with_best)}-way tie; the tiebreak *could* reorder it - "
        f"review before trusting plan[0] blindly")
    print(f"  [ok] today's top Circa plan is a unique optimum ({best_sp:.4%}) - refinements don't move it")


if __name__ == "__main__":
    tests = [
        test_holiday_scarcity_synthetic,
        test_tiebreaker_frontloads_safety,
        test_future_value_from_matrix,
        test_holiday_scarcity_real_2026,
        test_refinements_dont_move_todays_circa_picks,
    ]
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print("\nall Circa refinement tests passed")
