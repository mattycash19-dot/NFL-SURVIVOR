"""
Real survivor-pool pick popularity for the current week/leg, scraped from
survivorgrid.com/picks (their public "consensus picks" table blends real
Yahoo / ESPN / USA Football Pools entry data with a projected consensus
column). Used only for the current selection - future legs fall back to a
win-probability-derived estimate (see field_pick_distribution).

Why popularity matters here and not for the private pool: Circa's $20M
splits equally among all surviving entries, so a correct pick off the
popular team eliminates other entries and grows your payout share. Against
a field of thousands that's a real second-order signal (see the vault's
"Circa Survivor Blueprint" research note). It is strictly a tiebreaker
between already-comparable options - never a reason to take a worse team
(field_model.py enforces the cap).

Degrades cleanly: any fetch/parse failure returns {} and logs a warning,
same contract as odds_data / injuries - the pipeline still produces a plan.

Access note: unlike ESPN's injuries endpoint, survivorgrid does NOT block
scripted requests (tested: default python-requests UA and curl both get
200). No User-Agent workaround needed. If that changes, add one the way
injuries.py does.
"""
import re
import sys

import requests

PICKS_URL = "https://www.survivorgrid.com/picks"

# survivorgrid abbrevs -> this project's convention (nflverse). Everything
# else matches; only these two differ.
_ALIAS = {"LAR": "LA", "WSH": "WAS"}


def _norm(team_cell_text):
    ab = re.split(r"\s", team_cell_text.strip(), maxsplit=1)[0].strip().upper()
    return _ALIAS.get(ab, ab)


def _pct(text):
    m = re.search(r"([\d.]+)\s*%", text)
    return float(m.group(1)) / 100.0 if m else None


def fetch_current_popularity():
    """
    {team_abbrev: projected_pick_share (0-1)} for the current NFL week, from
    survivorgrid's `projected` consensus column. Returns {} on any failure.
    Also returns a small dict of source metadata via the second tuple slot.
    """
    try:
        resp = requests.get(PICKS_URL, timeout=20)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"popularity: fetch failed ({e}) - skipping the payout-share adjustment", file=sys.stderr)
        return {}, {"ok": False, "reason": str(e)}

    i = html.find('<table id="picks"')
    j = html.find("</table>", i)
    if i < 0 or j < 0:
        print("popularity: survivorgrid page format changed (no #picks table) - skipping", file=sys.stderr)
        return {}, {"ok": False, "reason": "table not found"}

    week_m = re.search(r"Week\s+(\d+)\s+Consensus", html)
    header = re.findall(r'data-sort="([^"]+)"', html[i:j])
    try:
        proj_idx = header.index("projected")
    except ValueError:
        proj_idx = -2  # projected is the second-to-last column in every version seen

    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html[i:j], re.S)[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) <= abs(proj_idx):
            continue
        team = _norm(re.sub(r"<[^>]+>", "", cells[0]))
        share = _pct(re.sub(r"<[^>]+>", "", cells[proj_idx]))
        if team and share is not None:
            out[team] = share

    if not out:
        print("popularity: parsed 0 rows from survivorgrid - skipping", file=sys.stderr)
        return {}, {"ok": False, "reason": "0 rows parsed"}

    meta = {"ok": True, "source": "survivorgrid.com/picks",
            "week": int(week_m.group(1)) if week_m else None, "n_teams": len(out)}
    return out, meta


def field_pick_distribution(win_probs, real_popularity=None, concentration=3.0):
    """
    The field's expected pick share across the teams playing one leg.

    `win_probs`: {team: win_prob} for the teams available that leg.
    `real_popularity`: {team: share} from fetch_current_popularity() when
        this is the current leg; None for future legs.

    Current leg -> use the real numbers, renormalized over just the teams
    still available. Future legs -> the field has no published data yet, so
    model it as chalk-seeking: share proportional to win_prob**concentration
    (concentration>1 => the field piles onto favorites harder than raw win
    probability, which matches how real survivor pools behave). This is a
    documented heuristic, not fitted - it only feeds the field-attrition
    simulation, never a pick directly.
    """
    teams = list(win_probs)
    if not teams:
        return {}

    if real_popularity:
        raw = {t: max(real_popularity.get(t, 0.0), 1e-4) for t in teams}
    else:
        raw = {t: max(win_probs[t], 1e-4) ** concentration for t in teams}

    total = sum(raw.values())
    return {t: raw[t] / total for t in teams}


if __name__ == "__main__":
    pop, meta = fetch_current_popularity()
    print(f"source ok: {meta.get('ok')}  week: {meta.get('week')}  teams: {meta.get('n_teams')}")
    for t, s in sorted(pop.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {t:>4}  {s:5.1%}")
