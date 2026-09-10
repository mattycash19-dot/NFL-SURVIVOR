"""
Renders a plan result (see run_baseline.py / run_weekly.py) into a single
self-contained dashboard.html.

Two tabs: Normal Survivor and Circa Survivor. Circa adds a separate winning
pick for Thanksgiving Eve, Thanksgiving Day, Black Friday, and Christmas on
top of the normal weekly pick (all from the same no-repeat team pool) - see
optimizer.CIRCA_SLOTS. The Circa tab is only shown when circa plans are
present in the result.

Team logos are pulled from ESPN's public CDN
(a.espncdn.com/i/teamlogos/nfl/500/<abbrev>.png) - fine here because this
is a normal hosted page, not a sandboxed Claude artifact. Every logo has an
onerror fallback and the team name is always shown as text next to it, so a
missing image never breaks a card.
"""
import os

LOGO_URL = "https://a.espncdn.com/i/teamlogos/nfl/500/{}.png"

CSS = """
:root {
  --bg:#0b0e14; --panel:#141a26; --panel-2:#1b2333; --border:#28324a;
  --text:#eef1f7; --muted:#93a0b8; --faint:#5f6b83;
  --accent:#5b9dff; --good:#31c76a; --mid:#e8b23b; --bad:#e8564a;
  --holiday:#c07be0;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg:#f4f6fb; --panel:#ffffff; --panel-2:#f0f3f9; --border:#dde3ee;
    --text:#141a26; --muted:#5a6784; --faint:#8b97ad;
    --accent:#2f6fe0; --good:#1a9e50; --mid:#c98a12; --bad:#d13c30;
    --holiday:#9a3fc0;
  }
}
* { box-sizing:border-box; }
body {
  background:var(--bg); color:var(--text); margin:0;
  padding:28px 20px 60px;
  font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.45;
}
.wrap { max-width:820px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }
.meta { color:var(--muted); font-size:12.5px; margin-bottom:20px; }
h2 { font-size:13px; color:var(--muted); font-weight:700; letter-spacing:.08em;
     text-transform:uppercase; margin:28px 0 12px; }

.tabs { display:flex; gap:6px; margin-bottom:22px; border-bottom:1px solid var(--border); }
.tab-btn {
  appearance:none; background:none; border:none; cursor:pointer;
  font:inherit; font-weight:600; font-size:14px; color:var(--muted);
  padding:10px 14px; border-bottom:2px solid transparent; margin-bottom:-1px;
}
.tab-btn.active { color:var(--text); border-bottom-color:var(--accent); }
.tab-panel { display:none; }
.tab-panel.active { display:block; }

.summary {
  background:var(--panel-2); border:1px solid var(--border); border-radius:10px;
  padding:12px 16px; font-size:13px; color:var(--muted); margin-bottom:16px;
}
.summary b { color:var(--text); }

.card {
  background:var(--panel); border:1px solid var(--border); border-radius:12px;
  padding:14px 16px 14px 20px; margin-bottom:10px; position:relative; overflow:hidden;
}
.card::before {
  content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
  background:var(--stripe, var(--faint));
}
.card.holiday { border-color:color-mix(in srgb, var(--holiday) 45%, var(--border)); }
.card-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.slot { font-size:11.5px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); }
.slot.holiday { color:var(--holiday); }
.badge {
  font-size:10.5px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  padding:2px 8px; border-radius:999px;
}
.badge.now { background:var(--accent); color:#fff; }
.badge.locked { background:var(--good); color:#04240f; }

.pick-row { display:flex; align-items:center; gap:14px; }
.logo { width:46px; height:46px; object-fit:contain; flex-shrink:0; }
.logo-sm { width:20px; height:20px; object-fit:contain; vertical-align:middle; }
.logo-missing { display:none; }
.pick-main { flex:1; min-width:0; }
.pick-line { font-size:16px; font-weight:700; letter-spacing:-.01em; }
.pick-line .verb { color:var(--muted); font-weight:600; font-size:13px; }
.pick-line .team { font-size:18px; }
.prob-wrap { display:flex; align-items:center; gap:10px; margin-top:6px; }
.bar { flex:1; height:7px; border-radius:4px; background:var(--panel-2); overflow:hidden; max-width:260px; }
.bar > span { display:block; height:100%; border-radius:4px; }
.pct { font-variant-numeric:tabular-nums; font-weight:700; font-size:13.5px; min-width:46px; }
.beat { margin-top:8px; font-size:13px; color:var(--muted); }
.beat .opp { color:var(--text); font-weight:600; }
.tag-div { font-size:10.5px; color:var(--faint); border:1px solid var(--border); border-radius:4px; padding:0 5px; margin-left:6px; }
.flags { margin-top:9px; display:flex; flex-wrap:wrap; gap:5px; }
.flag { font-size:11px; background:color-mix(in srgb, var(--bad) 16%, transparent);
        color:var(--bad); border-radius:5px; padding:2px 7px; }
.reasoning { margin-top:8px; font-size:12px; color:var(--faint); }

.alt { background:var(--panel); border:1px solid var(--border); border-radius:10px;
       padding:11px 15px; margin-bottom:8px; font-size:13px; }
.alt .hd { color:var(--muted); margin-bottom:4px; }
.alt .diff { color:var(--accent); font-weight:600; }

.ratings { display:grid; grid-template-columns:repeat(auto-fill,minmax(112px,1fr)); gap:5px; }
.ratings .r { display:flex; align-items:center; gap:6px; background:var(--panel);
  border:1px solid var(--border); border-radius:6px; padding:4px 8px; font-size:12.5px; }
.ratings .r b { margin-left:auto; font-variant-numeric:tabular-nums; }
.byes { font-size:13px; }
.byes tr td { padding:4px 10px 4px 0; border-bottom:1px solid var(--border); }
.note { color:var(--muted); font-size:12.5px; }
table { border-collapse:collapse; }
"""

JS = """
document.querySelectorAll('.tab-btn').forEach(function(btn){
  btn.addEventListener('click', function(){
    var id = btn.dataset.target;
    document.querySelectorAll('.tab-btn').forEach(function(b){ b.classList.toggle('active', b===btn); });
    document.querySelectorAll('.tab-panel').forEach(function(p){ p.classList.toggle('active', p.id===id); });
  });
});
"""


def _stripe(p):
    return "var(--good)" if p >= 0.72 else "var(--mid)" if p >= 0.60 else "var(--bad)"


def _logo(abbrev, cls="logo"):
    if not abbrev:
        return ""
    return (f'<img class="{cls}" src="{LOGO_URL.format(abbrev.lower())}" alt="{abbrev}" '
            f'loading="lazy" onerror="this.classList.add(\'logo-missing\')">')


def _slot_label(slot):
    return f"Week {slot}" if isinstance(slot, int) else str(slot)


def _pick_card(slot, d, team_names, plan_week):
    p = d["win_prob"]
    is_holiday = d.get("holiday_slot") or not isinstance(slot, int)
    team_full = team_names.get(d["team"], d["team"])
    opp_full = team_names.get(d["opponent"], d["opponent"]) if d.get("opponent") else "?"

    badges = ""
    if d.get("locked"):
        badges += '<span class="badge locked">Locked</span>'
    elif slot == plan_week:
        badges += '<span class="badge now">This week</span>'

    div_tag = '<span class="tag-div">div</span>' if d.get("div_game") else ""
    flags_html = ""
    if d.get("flags"):
        flags_html = '<div class="flags">' + "".join(f'<span class="flag">{f}</span>' for f in d["flags"]) + "</div>"
    reasoning_html = f'<div class="reasoning">{d["reasoning"]}</div>' if d.get("reasoning") else ""

    return f"""
    <div class="card {'holiday' if is_holiday else ''}" style="--stripe:{_stripe(p)}">
      <div class="card-top">
        <span class="slot {'holiday' if is_holiday else ''}">{_slot_label(slot)}</span>
        <span>{badges}</span>
      </div>
      <div class="pick-row">
        {_logo(d["team"])}
        <div class="pick-main">
          <div class="pick-line"><span class="verb">PICK</span> <span class="team">{team_full}</span> <span class="verb">TO WIN</span></div>
          <div class="prob-wrap">
            <span class="bar"><span style="width:{p*100:.0f}%;background:{_stripe(p)}"></span></span>
            <span class="pct">{p:.0%}</span>
          </div>
          <div class="beat">to beat {_logo(d["opponent"], "logo-sm")} <span class="opp">{opp_full}</span>{div_tag}
            {"" if d.get("is_home") is None else (" &middot; at home" if d["is_home"] else " &middot; on the road")}</div>
          {flags_html}
          {reasoning_html}
        </div>
      </div>
    </div>"""


def _alt_block(plan, top_plan, idx):
    diffs = []
    for slot, d in plan["detail"].items():
        top_team = top_plan["detail"].get(slot, {}).get("team")
        if d["team"] != top_team:
            diffs.append(f'{_slot_label(slot)}: <span class="diff">{d["team"]}</span> (was {top_team})')
    body = ", ".join(diffs) if diffs else "identical to the top plan"
    behind = (top_plan["survival_prob"] - plan["survival_prob"]) * 100
    return f"""
    <div class="alt">
      <div class="hd">Alternate #{idx} &middot; survival probability <b>{plan['survival_prob']:.2%}</b>
        ({behind:.2f} pts behind top)</div>
      <div>Differs at: {body}</div>
    </div>"""


def _panel(panel_id, plans, team_names, plan_week, active, circa_slots=None):
    if not plans:
        return f'<div class="tab-panel{" active" if active else ""}" id="{panel_id}"><p class="note">No plan available.</p></div>'
    top = plans[0]
    intro = ""
    if circa_slots:
        days = ", ".join(f"{s['slot']} ({s['date']})" for s in circa_slots)
        intro = (f'<div class="summary">Circa adds a separate winning pick for: <b>{days}</b> &mdash; '
                 f'each from a team not used anywhere else in the plan.</div>')
    cards = "".join(_pick_card(slot, d, team_names, plan_week) for slot, d in top["detail"].items())
    alts = "".join(_alt_block(p, top, i) for i, p in enumerate(plans[1:], 2))
    return f"""
    <div class="tab-panel{' active' if active else ''}" id="{panel_id}">
      {intro}
      <div class="summary">Full-plan survival probability (win every pick): <b>{top['survival_prob']:.2%}</b>
        over {len(top['detail'])} picks.</div>
      <h2>The Plan</h2>
      {cards}
      <h2>Alternate Plans</h2>
      {alts or '<p class="note">No distinct alternates found.</p>'}
    </div>"""


def render(result, out_path=None):
    out_path = out_path or os.path.join(os.path.dirname(__file__), "dashboard.html")
    team_names = result.get("team_names", {})
    plan_week = result.get("plan_week")
    circa_plans = result.get("circa_plans") or []
    circa_slots = result.get("circa_slots") or []

    rating_bits = []
    if "rating_season" in result:
        rating_bits.append(f"ratings from {result['rating_season']} play-by-play")
    if "inseason_weight" in result:
        rating_bits.append(f"{result['inseason_weight']:.0%} current-season weight")
    rating_bits.append(f"home-field {result['home_field_logodds']:.3f} log-odds")

    tabs = '<button class="tab-btn active" data-target="panel-normal">Normal Survivor</button>'
    normal_panel = _panel("panel-normal", result["plans"], team_names, plan_week, active=True)
    circa_panel = ""
    if circa_plans:
        tabs += '<button class="tab-btn" data-target="panel-circa">Circa Survivor</button>'
        circa_panel = _panel("panel-circa", circa_plans, team_names, plan_week, active=False, circa_slots=circa_slots)

    ratings_html = "".join(
        f'<div class="r">{_logo(t, "logo-sm")} {t} <b>{v:+.2f}</b></div>'
        for t, v in result.get("team_ratings", {}).items())
    byes_html = "".join(
        f"<tr><td>Week {wk}</td><td>{', '.join(teams)}</td></tr>"
        for wk, teams in sorted(result.get("byes", {}).items()) if teams)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Survivor Plan - {result['plan_season']}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>NFL Survivor Pool &mdash; {result['plan_season']}</h1>
<div class="meta">Generated {result['generated_at']} &middot; current week {plan_week} &middot; {' &middot; '.join(rating_bits)}</div>

<div class="tabs">{tabs}</div>
{normal_panel}
{circa_panel}

<h2>Team Ratings (higher = stronger, regressed toward the mean)</h2>
<div class="ratings">{ratings_html}</div>

<h2>Bye Weeks</h2>
<table class="byes"><tbody>{byes_html}</tbody></table>

<h2>About</h2>
<p class="note">Independent power-rating model (EPA/play + success rate, logistic win-probability
fit validated out-of-sample &mdash; see the repo's <code>ratings.py</code> / <code>calibration.py</code>),
solved as a season-long assignment that maximizes the probability of winning every pick. Only the
current week's pick should be treated as locked; everything after is provisional and recomputed each run.
Phase-3 calibration found this preseason model still trails the betting market's accuracy &mdash; treat
early-season probabilities with real skepticism.</p>
</div>
<script>{JS}</script>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
