"""
Renders plan.json (see run_baseline.py) into a single self-contained
dashboard.html - no CDN dependencies, opens directly in a browser, same
pattern as MLB Edge's dashboard.py.
"""
import os

CSS = """
:root {
  --bg: #0b0e14; --panel: #131826; --border: #232a3d;
  --text: #e6e9f0; --muted: #8b93a7;
  --good: #22c55e; --mid: #eab308; --bad: #ef4444; --accent: #60a5fa;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text); margin: 0; padding: 32px;
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
}
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; color: var(--muted); font-weight: 600; margin: 32px 0 12px; text-transform: uppercase; letter-spacing: .04em; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 14px; }
th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; }
tr:last-child td { border-bottom: none; }
.prob-bar { display: inline-block; height: 6px; border-radius: 3px; background: var(--accent); vertical-align: middle; margin-right: 8px; }
.prob-cell { display: flex; align-items: center; gap: 8px; min-width: 140px; }
.tag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px; margin-left: 6px; }
.tag-div { background: #3730a3; color: #c7d2fe; }
.badge-top { background: var(--good); color: #06210e; font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
.alt-summary { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
.alt-picks { font-size: 13px; line-height: 1.7; }
.diff { color: var(--accent); font-weight: 600; }
.ratings-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 6px; font-size: 13px; }
.ratings-grid div { padding: 4px 8px; background: var(--bg); border-radius: 4px; border: 1px solid var(--border); }
.note { color: var(--muted); font-size: 13px; line-height: 1.6; }
.reasoning { color: var(--muted); font-size: 12px; margin-top: 2px; }
.flag { display: inline-block; background: #7c2d12; color: #fed7aa; font-size: 11px; padding: 1px 6px; border-radius: 4px; margin: 2px 4px 0 0; }
.locked-badge { background: var(--accent); color: #06210e; font-weight: 700; padding: 1px 8px; border-radius: 4px; font-size: 11px; margin-left: 6px; }
"""


def _prob_color(p):
    if p >= 0.75:
        return "var(--good)"
    if p >= 0.60:
        return "var(--mid)"
    return "var(--bad)"


def _pick_row(wk, d):
    p = d["win_prob"]
    loc = "vs" if d["is_home"] else "@"
    div_tag = '<span class="tag tag-div">DIV</span>' if d["div_game"] else ""
    locked_tag = '<span class="locked-badge">LOCKED</span>' if d.get("locked") else ""
    bar_width = int(p * 100)
    flags_html = "".join(f'<span class="flag">{f}</span>' for f in d.get("flags") or [])
    reasoning_html = f'<div class="reasoning">{d["reasoning"]}</div>' if d.get("reasoning") else ""
    return f"""
    <tr>
      <td>Week {wk}</td>
      <td>
        <b>{d['team']}</b> {loc} {d['opponent']}{div_tag}{locked_tag}
        {reasoning_html}
        {flags_html}
      </td>
      <td>
        <div class="prob-cell">
          <span class="prob-bar" style="width:{bar_width}px; background:{_prob_color(p)};"></span>
          {p:.1%}
        </div>
      </td>
      <td>{d['team_rating']:+.2f}</td>
    </tr>"""


def _plan_table(plan, label=None):
    rows = "".join(_pick_row(wk, d) for wk, d in plan["detail"].items())
    header = f'<span class="badge-top">{label}</span>' if label else ""
    return f"""
    <div class="panel">
      <div class="alt-summary">{header} Survival probability across all {len(plan['detail'])} picks: <b>{plan['survival_prob']:.2%}</b></div>
      <table>
        <tr><th>Week</th><th>Pick</th><th>Win prob.</th><th>Rating</th></tr>
        {rows}
      </table>
    </div>"""


def _alt_summary(plan, top_plan, idx):
    diffs = []
    for wk, d in plan["detail"].items():
        top_team = top_plan["detail"][wk]["team"]
        if d["team"] != top_team:
            diffs.append(f"Wk{wk}: <span class='diff'>{d['team']}</span> (was {top_team})")
    diff_str = ", ".join(diffs) if diffs else "identical to top plan"
    return f"""
    <div class="panel">
      <div class="alt-summary">Alternate #{idx} - survival probability <b>{plan['survival_prob']:.2%}</b>
        ({(top_plan['survival_prob'] - plan['survival_prob']) * 100:.2f} pts behind the top plan)</div>
      <div class="alt-picks">Differs from top plan at: {diff_str}</div>
    </div>"""


def render(result, out_path=None):
    out_path = out_path or os.path.join(os.path.dirname(__file__), "dashboard.html")
    plans = result["plans"]
    top = plans[0]

    ratings_html = "".join(
        f"<div>{team} <b>{val:+.2f}</b></div>"
        for team, val in result["team_ratings"].items()
    )

    alt_html = "".join(_alt_summary(p, top, i) for i, p in enumerate(plans[1:], 2))

    byes_html = "".join(
        f"<tr><td>Week {wk}</td><td>{', '.join(teams) if teams else '-'}</td></tr>"
        for wk, teams in sorted(result["byes"].items())
        if teams
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>NFL Survivor Plan - {result['plan_season']}</title>
<style>{CSS}</style></head><body>
<h1>NFL Survivor Pool - Season {result['plan_season']} Plan</h1>
<div class="meta">
  Generated {result['generated_at']}
  {f" &middot; ratings built from {result['rating_season']} season play-by-play" if 'rating_season' in result else ""}
  {f" &middot; current-season weight in ratings: {result['inseason_weight']:.0%}" if 'inseason_weight' in result else ""}
  &middot; home-field edge (fitted): {result['home_field_logodds']:.3f} log-odds
  &middot; current week: {result['plan_week']}
</div>

<h2>Top Plan</h2>
{_plan_table(top, label="BEST")}

<h2>Alternate Plans ({len(plans) - 1})</h2>
{alt_html}

<h2>Team Ratings ({result.get('rating_season', 'prior season + in-season')} EPA/success, regressed toward mean)</h2>
<div class="panel"><div class="ratings-grid">{ratings_html}</div></div>

<h2>Bye Weeks</h2>
<div class="panel"><table><tr><th>Week</th><th>Teams on bye</th></tr>{byes_html}</table></div>

<h2>About this plan</h2>
<div class="panel note">
  This is the Phase 1 baseline: one independent power-rating model (Elo-style,
  fit on real outcomes, regressed toward the mean for the new season), solved
  as a season-long assignment maximizing survival probability. It does not yet
  include Phase 2 signals (in-season rating updates, market cross-check,
  injury reports, rest/travel/weather, trap-game flags) - those reshuffle the
  plan week to week once wired in. Only the current week's pick should ever
  be treated as locked; everything else here is provisional.
</div>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
