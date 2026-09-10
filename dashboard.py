"""
Renders a plan result (run_baseline.py / run_weekly.py) into a single
self-contained dashboard.html.

Two tabs: Normal Survivor (18 weekly picks) and Circa Survivor (20 legs -
the 18 weeks plus a standalone Thanksgiving/Black Friday leg and a
standalone Christmas leg, each its own no-repeat winning pick). The Circa
tab appears only when circa plans are present.

Colors follow the data-viz skill's reference palette: one blue for the
probability meter (magnitude = bar width, hue held constant), the fixed
`critical` red reserved for genuinely alarming picks (sub-50% or an injury
flag) and always paired with an icon+label, violet identity accent for
Circa legs. Team identity is carried by ESPN CDN logos, not color, so
there is no categorical series palette to worry about. Light and dark are
both selected (their own steps), following the OS setting.

Each pick card has a native <details> "see all games" expander showing that
week's / leg's full slate with every team's win probability - so you can
see what else was on the board and which teams you could still pick.
"""
import os

LOGO_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{}.png"

CSS = """
:root {
  color-scheme: light;
  --plane:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ee;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --hair:rgba(11,11,11,.10); --grid:#e1e0d9;
  --fill:#2a78d6; --fill-track:#e6ecf3;
  --leg:#4a3aa7; --leg-wash:#efeaf7;
  --alarm:#d03b3b; --alarm-wash:#fbeceb;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19; --surface-2:#232322;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --hair:rgba(255,255,255,.12); --grid:#2c2c2a;
    --fill:#3987e5; --fill-track:#26313f;
    --leg:#9085e9; --leg-wash:#241f3a;
    --alarm:#e66767; --alarm-wash:#2e1c1c;
  }
}
* { box-sizing:border-box; }
body {
  margin:0; padding:32px 18px 72px; background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.45;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:780px; margin:0 auto; }
h1 { font-size:21px; margin:0 0 3px; letter-spacing:-.01em; }
.meta { color:var(--muted); font-size:12px; margin-bottom:20px; }
h2 { font-size:12px; color:var(--muted); font-weight:700; letter-spacing:.09em;
     text-transform:uppercase; margin:30px 0 12px; }

.tabs { display:flex; gap:4px; margin-bottom:20px; border-bottom:1px solid var(--hair); }
.tab-btn {
  appearance:none; background:none; border:0; cursor:pointer; font:inherit;
  font-weight:600; font-size:13.5px; color:var(--muted); padding:9px 13px;
  border-bottom:2px solid transparent; margin-bottom:-1px;
}
.tab-btn.active { color:var(--ink); border-bottom-color:var(--fill); }
.tab-panel { display:none; }
.tab-panel.active { display:block; }

.callout {
  background:var(--surface); border:1px solid var(--hair); border-radius:10px;
  padding:11px 14px; font-size:12.5px; color:var(--ink-2); margin-bottom:12px;
}
.callout b { color:var(--ink); }
.callout.leg { background:var(--leg-wash); border-color:transparent; }

.unused { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:4px; }
.chip {
  display:inline-flex; align-items:center; gap:5px; font-size:12px;
  background:var(--surface); border:1px solid var(--hair); border-radius:999px;
  padding:3px 9px 3px 5px; color:var(--ink-2);
}

.card {
  background:var(--surface); border:1px solid var(--hair); border-radius:12px;
  padding:13px 15px 6px 17px; margin-bottom:9px; position:relative;
}
.card::before {
  content:""; position:absolute; left:0; top:10px; bottom:10px; width:3px;
  border-radius:3px; background:var(--stripe,transparent);
}
.card.leg::before { background:var(--leg); }
.card.alarm::before { background:var(--alarm); }
.row1 { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:9px; }
.slot { font-size:11px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
.slot.leg { color:var(--leg); }
.pill { font-size:10px; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
        padding:2px 7px; border-radius:999px; }
.pill.now { background:var(--fill); color:#fff; }
.pill.locked { background:var(--ink); color:var(--surface); }

.pick { display:flex; align-items:center; gap:13px; }
.logo { width:44px; height:44px; object-fit:contain; flex:0 0 auto; }
.logo-xs { width:18px; height:18px; object-fit:contain; vertical-align:middle; }
.logo-missing { display:none; }
.pick-body { flex:1; min-width:0; }
.pick-name { font-size:15.5px; font-weight:700; letter-spacing:-.01em; }
.pick-name .lead { font-size:11.5px; font-weight:700; color:var(--muted); letter-spacing:.06em; }
.meter { display:flex; align-items:center; gap:9px; margin-top:6px; }
.track { flex:1; max-width:240px; height:7px; border-radius:4px; background:var(--fill-track); overflow:hidden; }
.track > i { display:block; height:100%; background:var(--fill); border-radius:4px; }
.pct { font-size:13px; font-weight:700; font-variant-numeric:tabular-nums; min-width:40px; }
.beat { margin-top:7px; font-size:12.5px; color:var(--ink-2); }
.beat b { color:var(--ink); font-weight:600; }
.beat .sub { color:var(--muted); }
.chips { margin-top:8px; display:flex; flex-wrap:wrap; gap:5px; }
.risk { font-size:11px; font-weight:600; background:var(--alarm-wash); color:var(--alarm);
        border-radius:5px; padding:2px 7px; }
.reason { margin-top:7px; font-size:11.5px; color:var(--muted); }
.metaline { margin-top:6px; display:flex; flex-wrap:wrap; gap:5px; }
.mchip { font-size:11px; background:var(--surface-2); color:var(--ink-2); border-radius:5px; padding:2px 7px; }
.mchip.lean { color:var(--leg); }

details { margin:8px 0 6px; }
summary { cursor:pointer; font-size:11.5px; color:var(--fill); list-style:none; padding:4px 0; }
summary::-webkit-details-marker { display:none; }
summary::before { content:"\\25B8 "; }
details[open] summary::before { content:"\\25BE "; }
.slate { width:100%; border-collapse:collapse; font-size:12px; margin-top:4px; }
.slate td { padding:5px 6px; border-top:1px solid var(--grid); vertical-align:middle; }
.slate .side { display:inline-flex; align-items:center; gap:5px; }
.slate .at { color:var(--muted); font-size:11px; padding:0 2px; }
.slate .p { color:var(--ink-2); font-variant-numeric:tabular-nums; font-size:11px; }
.slate .is-pick { font-weight:700; color:var(--ink); }
.slate .is-used { color:var(--muted); text-decoration:line-through; }
.slate .used-note { color:var(--muted); text-decoration:none; font-size:10px; margin-left:3px; }
.slate .daycol { color:var(--muted); font-size:10.5px; white-space:nowrap; }

.alt { background:var(--surface); border:1px solid var(--hair); border-radius:10px;
       padding:10px 14px; margin-bottom:7px; font-size:12.5px; }
.alt .hd { color:var(--muted); margin-bottom:3px; }
.alt .d { color:var(--fill); font-weight:600; }

.ratings { display:grid; grid-template-columns:repeat(auto-fill,minmax(108px,1fr)); gap:4px; }
.ratings .r { display:flex; align-items:center; gap:6px; background:var(--surface);
  border:1px solid var(--hair); border-radius:7px; padding:4px 8px; font-size:12px; }
.ratings .r b { margin-left:auto; font-variant-numeric:tabular-nums; }
.byes { border-collapse:collapse; font-size:12.5px; }
.byes td { padding:4px 12px 4px 0; border-bottom:1px solid var(--grid); }
.note { color:var(--muted); font-size:12px; }
"""

JS = """
document.querySelectorAll('.tab-btn').forEach(function(b){
  b.addEventListener('click', function(){
    document.querySelectorAll('.tab-btn').forEach(function(x){ x.classList.toggle('active', x===b); });
    document.querySelectorAll('.tab-panel').forEach(function(p){ p.classList.toggle('active', p.id===b.dataset.target); });
  });
});
"""


def _logo(ab, cls="logo"):
    if not ab:
        return ""
    return (f'<img class="{cls}" src="{LOGO_URL.format(ab.lower())}" alt="{ab}" '
            f'loading="lazy" onerror="this.classList.add(\'logo-missing\')">')


def _label(slot):
    return f"Week {slot}" if isinstance(slot, int) else str(slot)


def _pct(p):
    return f"{p*100:.0f}%" if isinstance(p, (int, float)) else "–"


def _slate_table(rows, pick_team, used_at, this_label, team_names):
    out = ['<table class="slate">']
    for g in rows:
        cells = []
        for side in ("away", "home"):
            ab = g[side]
            prob = g[f"{side}_prob"]
            klass = "side"
            note = ""
            if ab == pick_team:
                klass += " is-pick"
            elif ab in used_at and used_at[ab] != this_label:
                klass += " is-used"
                note = f'<span class="used-note">used {_label(used_at[ab])}</span>'
            cells.append(
                f'<span class="{klass}">{_logo(ab, "logo-xs")} {ab} '
                f'<span class="p">{_pct(prob)}</span></span>{note}')
        out.append(
            f'<tr><td>{cells[0]}</td><td class="at">@</td><td>{cells[1]}</td>'
            f'<td class="daycol">{g.get("day","")}</td></tr>')
    out.append("</table>")
    return "".join(out)


def _card(slot, d, schedule, used_at, team_names, plan_week):
    p = d["win_prob"]
    is_leg = d.get("holiday_slot") or not isinstance(slot, int)
    coinflip = isinstance(p, (int, float)) and p < 0.50
    injury = any(f.lower().startswith("qb ") for f in (d.get("flags") or []))
    alarm = coinflip or injury

    klass = "card" + (" leg" if is_leg else "") + (" alarm" if alarm else "")
    team_full = team_names.get(d["team"], d["team"])
    opp_full = team_names.get(d["opponent"], d["opponent"]) if d.get("opponent") else "?"

    pill = ""
    if d.get("locked"):
        pill = '<span class="pill locked">Locked</span>'
    elif slot == plan_week:
        pill = '<span class="pill now">This week</span>'

    where = "" if d.get("is_home") is None else (
        '<span class="sub"> &middot; at home</span>' if d["is_home"] else '<span class="sub"> &middot; on the road</span>')
    divtag = '<span class="sub"> &middot; divisional</span>' if d.get("div_game") else ""

    chips = []
    if coinflip:
        chips.append('<span class="risk">⚠ under 50% &mdash; near coin flip</span>')
    for f in (d.get("flags") or []):
        chips.append(f'<span class="risk">⚠ {f}</span>')
    chips_html = f'<div class="chips">{"".join(chips)}</div>' if chips else ""
    reason_html = f'<div class="reason">{d["reason"]}</div>' if d.get("reason") else (
        f'<div class="reason">{d["reasoning"]}</div>' if d.get("reasoning") else "")

    meta = []
    if isinstance(d.get("popularity"), (int, float)):
        meta.append(f'<span class="mchip">{d["popularity"]*100:.0f}% of pools pick {d["team"]}</span>')
    if isinstance(d.get("payout_delta"), (int, float)) and abs(d["payout_delta"]) >= 0.001:
        meta.append(f'<span class="mchip lean">{d["payout_delta"]*100:+.1f} pt payout-share lean</span>')
    meta_html = f'<div class="metaline">{"".join(meta)}</div>' if meta else ""

    slate_rows = schedule.get(slot) or schedule.get(str(slot)) or []
    expander = ""
    if slate_rows:
        expander = (
            f'<details><summary>see all {len(slate_rows)} games &mdash; other picks available</summary>'
            f'{_slate_table(slate_rows, d["team"], used_at, slot, team_names)}</details>')

    return f"""
    <div class="{klass}" style="--stripe:var(--hair)">
      <div class="row1"><span class="slot {'leg' if is_leg else ''}">{_label(slot)}</span>{pill}</div>
      <div class="pick">
        {_logo(d["team"])}
        <div class="pick-body">
          <div class="pick-name"><span class="lead">PICK</span> {team_full} <span class="lead">TO WIN</span></div>
          <div class="meter"><span class="track"><i style="width:{max(2, p*100 if isinstance(p,(int,float)) else 0):.0f}%"></i></span>
            <span class="pct">{_pct(p)}</span></div>
          <div class="beat">to beat {_logo(d["opponent"], "logo-xs")} <b>{opp_full}</b>{where}{divtag}</div>
          {meta_html}{chips_html}{reason_html}{expander}
        </div>
      </div>
    </div>"""


def _alt(plan, top, idx):
    diffs = []
    for slot, d in plan["detail"].items():
        was = top["detail"].get(slot, {}).get("team")
        if d["team"] != was:
            diffs.append(f'{_label(slot)}: <span class="d">{d["team"]}</span> (was {was})')
    body = ", ".join(diffs) if diffs else "identical to the top plan"
    behind = (top["survival_prob"] - plan["survival_prob"]) * 100
    return (f'<div class="alt"><div class="hd">Alternate #{idx} &middot; survival '
            f'<b>{plan["survival_prob"]:.2%}</b> ({behind:.2f} pts behind top)</div>'
            f'<div>Differs at: {body}</div></div>')


def _unused_chips(top, all_teams, team_names):
    used = {d["team"] for d in top["detail"].values()}
    left = [t for t in all_teams if t not in used]
    chips = "".join(f'<span class="chip">{_logo(t, "logo-xs")} {t}</span>' for t in left)
    return (f'<h2>Teams never used in this plan ({len(left)} of {len(all_teams)})</h2>'
            f'<div class="unused">{chips}</div>')


def _panel(pid, plans, schedule, team_names, all_teams, plan_week, active, legs=None, blend_meta=None):
    if not plans:
        return f'<div class="tab-panel{" active" if active else ""}" id="{pid}"><p class="note">No plan available.</p></div>'
    top = plans[0]
    used_at = {d["team"]: slot for slot, d in top["detail"].items()}

    leg_note = ""
    if legs:
        spans = ", ".join(f'<b>{l["leg"]}</b> ({" & ".join(l["dates"])})' for l in legs)
        leg_note = (f'<div class="callout leg">Circa is 20 legs: the 18 weeks plus {spans} &mdash; '
                    f'each its own winning pick, no team reused across any of them.</div>')

    blend_note = ""
    if blend_meta:
        fa = blend_meta.get("field_alive_after", {})
        wk6 = next((v for k, v in fa.items() if k in ("6", "Week 6")), None)
        thin = f" Field simulated to thin to ~{wk6*100:.0f}% alive by Week 6." if wk6 else ""
        n = blend_meta.get("picks_changed", 0)
        verdict = ("changed <b>no picks</b>" if n == 0 else
                   f"changed <b>{n} pick(s)</b>, at most {blend_meta.get('max_winprob_sacrificed',0)*100:.1f} pt of win probability given up")
        blend_note = (f'<div class="callout">Payout-share signal (pick popularity &times; field attrition, capped '
                      f'&plusmn;{blend_meta.get("cap",0.03)*100:.0f} pts): at the current weight it {verdict}.'
                      f'{thin} The season-wide optimizer already does the resource timing that actually wins Circa &mdash; '
                      f'popularity is shown per pick as context, not forced.</div>')

    cards = "".join(_card(slot, d, schedule, used_at, team_names, plan_week) for slot, d in top["detail"].items())
    alts = "".join(_alt(p, top, i) for i, p in enumerate(plans[1:], 2))
    return f"""
    <div class="tab-panel{' active' if active else ''}" id="{pid}">
      {leg_note}
      {blend_note}
      <div class="callout">Probability of winning <b>every</b> pick: <b>{top['survival_prob']:.2%}</b>
        &middot; {len(top['detail'])} picks.</div>
      {_unused_chips(top, all_teams, team_names)}
      <h2>The Plan</h2>
      {cards}
      <h2>Alternate Plans</h2>
      {alts or '<p class="note">No distinct alternates found.</p>'}
    </div>"""


def render(result, out_path=None):
    out_path = out_path or os.path.join(os.path.dirname(__file__), "dashboard.html")
    team_names = result.get("team_names", {})
    plan_week = result.get("plan_week")
    schedule = result.get("schedule", {})
    all_teams = list(result.get("team_ratings", {}).keys())
    circa_plans = result.get("circa_plans") or []
    legs = result.get("circa_legs") or []

    bits = []
    if "rating_season" in result:
        bits.append(f"ratings from {result['rating_season']} play-by-play")
    if "inseason_weight" in result:
        bits.append(f"{result['inseason_weight']:.0%} current-season weight")
    bits.append(f"home-field {result['home_field_logodds']:.3f} log-odds")

    tabs = '<button class="tab-btn active" data-target="p-normal">Normal Survivor</button>'
    normal = _panel("p-normal", result["plans"], schedule, team_names, all_teams, plan_week, active=True)
    circa = ""
    if circa_plans:
        tabs += '<button class="tab-btn" data-target="p-circa">Circa Survivor</button>'
        circa = _panel("p-circa", circa_plans, schedule, team_names, all_teams, plan_week, active=False,
                       legs=legs, blend_meta=result.get("payout_blend"))

    ratings_html = "".join(
        f'<div class="r">{_logo(t, "logo-xs")} {t} <b>{v:+.2f}</b></div>'
        for t, v in result.get("team_ratings", {}).items())
    byes_html = "".join(
        f"<tr><td>Week {wk}</td><td>{', '.join(teams)}</td></tr>"
        for wk, teams in sorted(result.get("byes", {}).items()) if teams)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Survivor Plan - {result['plan_season']}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>NFL Survivor Pool &mdash; {result['plan_season']}</h1>
<div class="meta">Generated {result['generated_at']} &middot; current week {plan_week} &middot; {' &middot; '.join(bits)}</div>
<div class="tabs">{tabs}</div>
{normal}
{circa}
<h2>Team Ratings (higher = stronger, regressed toward the mean)</h2>
<div class="ratings">{ratings_html}</div>
<h2>Bye Weeks</h2>
<table class="byes"><tbody>{byes_html}</tbody></table>
<h2>About</h2>
<p class="note">Independent power-rating model (EPA/play + success rate, logistic win-probability
fit validated out-of-sample). Solved as a season-long assignment maximizing the probability of
winning every pick. Only the current week's pick is locked; everything after is provisional and
recomputed each run. Phase-3 calibration found this preseason model still trails the betting
market's accuracy &mdash; treat early-season probabilities with real skepticism.</p>
</div><script>{JS}</script></body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
