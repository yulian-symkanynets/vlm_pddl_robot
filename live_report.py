#!/usr/bin/env python
"""Graphs and a one-file HTML report from live_bench.py results.

    python live_report.py results/live/<run>                  # rebuild one run's report
    python live_report.py results/live/<run_a> results/live/<run_b>    # compare runs (e.g. two models)

Reads only results.json (written by live_bench.py); every number on the page comes from there.

Charts (charts/*.png, also embedded in report.html)
    outcomes.png      per kind of task: share correct / wrong plan / asked / no plan / error
    latency.png       median seconds per kind of task, split into perception / goal / planning / rest
    scaling.png       planner seconds and plan length against the number of objects
    photo_resolution.png   per resolution: seconds, image tokens, how much of the scene was seen right
    photo_vs_json.png      the same task from the photo and from the JSON: seconds and bytes sent
    quality.png       plans proven shortest against plans that are only verified
    compare.png       (several runs) success and seconds per kind of task, side by side
"""
from __future__ import annotations

import base64
import html
import json
import statistics as st
import sys
from pathlib import Path

from run_report import GRID, INK, MUTED, STAGE_COLORS, SURFACE

OUTCOME_COLORS = {'correct': '#1baf7a', 'partial': '#e0b000', 'wrong plan': '#d03b3b', 'asked': '#2a78d6',
                  'no plan': '#b9b8b2', 'unexpected ok': '#8a3ffc', 'error': '#0b0b0b'}
OUTCOMES = list(OUTCOME_COLORS)
RES_ORDER = ['low', 'medium', 'high', 'full']
ROUTE_COLORS = {'json': '#2a78d6', 'photo': '#eb6834'}


def load(run: Path) -> dict:
    return json.loads((run / 'results.json').read_text())


def lab(r: dict) -> str:
    return f"{r['suite']}/{r['group']}" + (f" ({r['route']})" if r['suite'] == 'photo' else '')


def ordered(rows, key):
    seen = []
    for r in rows:
        k = key(r)
        if k not in seen:
            seen.append(k)
    return seen


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


# --------------------------------------------------------------------------- figure helpers
def _fig(w, h, ncols=1, sharey=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=150, sharey=sharey)
    fig.patch.set_facecolor(SURFACE)
    axes = [axes] if ncols == 1 else list(axes)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return plt, fig, axes


def _save(plt, fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _title(ax, text):
    ax.set_title(text, loc='left', fontsize=8.5, color=INK, fontweight='bold')


# --------------------------------------------------------------------------- charts
def chart_outcomes(rows, out: Path):
    labels = ordered(rows, lab)
    if not labels:
        return None
    plt, fig, (ax,) = _fig(7.8, 0.34 * len(labels) + 1.6)
    ax.grid(axis='y', visible=False)
    ys = list(range(len(labels)))[::-1]
    left = [0.0] * len(labels)
    for oc in OUTCOMES:
        vals = []
        for l in labels:
            rs = [r for r in rows if lab(r) == l]
            vals.append(100 * sum(r['outcome'] == oc for r in rs) / len(rs))
        if any(vals):
            ax.barh(ys, vals, left=left, color=OUTCOME_COLORS[oc], height=0.62, label=oc,
                    edgecolor=SURFACE, linewidth=0.8)
            left = [a + b for a, b in zip(left, vals)]
    for y, l in zip(ys, labels):
        rs = [r for r in rows if lab(r) == l]
        ax.text(101, y, f"{sum(r['passed'] for r in rs)}/{len(rs)}", va='center', fontsize=7.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xlabel('% of requests', fontsize=7.5, color=MUTED)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    _title(ax, 'What came back: correct = the plan works on the true scene and does the task')
    _save(plt, fig, out)
    return out


def chart_latency(rows, out: Path):
    rows = [r for r in rows if r['group'] != 'perceive' and r['wall_s'] is not None]
    labels = ordered(rows, lab)
    if not labels:
        return None
    plt, fig, (ax,) = _fig(7.8, 0.34 * len(labels) + 1.6)
    ax.grid(axis='y', visible=False)
    ys = list(range(len(labels)))[::-1]
    left = [0.0] * len(labels)
    stages = [('perception', 'perceive_s'), ('goal', 'goal_s'), ('planning', 'plan_s')]
    meds = {}
    for name, key in stages:
        meds[name] = [med([r[key] for r in rows if lab(r) == l]) or 0 for l in labels]
    meds['rest'] = [max(0, (med([r['wall_s'] for r in rows if lab(r) == l]) or 0)
                        - sum(meds[n][i] for n, _ in stages)) for i, l in enumerate(labels)]
    for name in STAGE_COLORS:
        ax.barh(ys, meds[name], left=left, color=STAGE_COLORS[name], height=0.62, label=name,
                edgecolor=SURFACE, linewidth=0.8)
        left = [a + b for a, b in zip(left, meds[name])]
    for y, x in zip(ys, left):
        ax.text(x + max(left) * 0.01, y, f'{x:.2f} s', va='center', fontsize=7.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.set_xlim(0, max(left) * 1.12)
    ax.set_xlabel('median seconds per request (as the client sees it)', fontsize=7.5, color=MUTED)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    _title(ax, 'Where the time goes (rest = network, upload, checking the plan)')
    _save(plt, fig, out)
    return out


def chart_scaling(rows, out: Path):
    sc = [r for r in rows if r['suite'] == 'scale' and r['n_objects']]
    model = [r for r in rows if r['suite'] == 'json' and r['group'] in ('sort', 'stack') and r['n_objects']]
    if not sc and not model:
        return None
    plt, fig, (a, b) = _fig(8.2, 3.4, 2)
    palette = ['#2a78d6', '#eb6834', '#1baf7a', '#8a3ffc']
    for i, g in enumerate(ordered(sc, lambda r: r['group'])):
        rs = sorted((r for r in sc if r['group'] == g), key=lambda r: r['n_objects'])
        ok = [r for r in rs if r['status'] == 'ok']
        a.plot([r['n_objects'] for r in ok], [max(r['plan_s'] or 0, 0.001) for r in ok], 'o-', color=palette[i % 4], label=g, ms=4)
        b.plot([r['n_objects'] for r in ok], [r['moves'] for r in ok], '-', color=palette[i % 4], lw=1, alpha=.6)
        for r in rs:
            if r['status'] == 'ok':
                b.plot(r['n_objects'], r['moves'], 'o', ms=5, color=palette[i % 4],
                       mfc=palette[i % 4] if r['optimal'] else SURFACE)
            else:
                a.plot(r['n_objects'], max(r['wall_s'], 0.001), 'x', color=palette[i % 4], ms=6)
    for i, g in enumerate(ordered(model, lambda r: r['group'])):
        rs = sorted((r for r in model if r['group'] == g and r['total_s']), key=lambda r: r['n_objects'])
        a.plot([r['n_objects'] for r in rs], [r['total_s'] for r in rs], 's--', color='#52514e', ms=3.5,
               lw=1, label=f'{g}: whole request with the model')
    a.set_yscale('log')
    a.set_xlabel('objects in the scene', fontsize=7.5, color=MUTED)
    a.set_ylabel('seconds (log; 0.001 = under a millisecond)', fontsize=7.5, color=MUTED)
    a.legend(fontsize=6.5, frameon=False)
    _title(a, 'Seconds against scene size (x = no plan)')
    b.set_xlabel('objects in the scene', fontsize=7.5, color=MUTED)
    b.set_ylabel('moves in the plan', fontsize=7.5, color=MUTED)
    _title(b, 'Moves (filled = proven shortest)')
    _save(plt, fig, out)
    return out


def chart_photo_resolution(rows, out: Path):
    ps = [r for r in rows if r['group'] == 'perceive' and r.get('req_resolution')]
    if not ps:
        return None
    levels = [x for x in RES_ORDER if any(r['req_resolution'] == x for r in ps)] + \
        sorted({r['req_resolution'] for r in ps} - set(RES_ORDER))
    plt, fig, axes = _fig(9.6, 3.2, 3)
    x = list(range(len(levels)))
    for ax, key, title in ((axes[0], 'wall_s', 'Seconds to read the photo (client)'),
                           (axes[1], 'approx_tokens', 'Image tokens the model had to read')):
        for i, lv in enumerate(levels):
            v = [r[key] for r in ps if r['req_resolution'] == lv and r[key] is not None]
            ax.scatter([i] * len(v), v, s=10, color='#2a78d6', alpha=.45)
            if v:
                ax.plot([i - .25, i + .25], [st.mean(v)] * 2, color=INK, lw=2)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=7.5, color=INK)
        _title(ax, title)
    ax = axes[2]
    for key, color, name in (('recall', '#1baf7a', 'objects found'), ('precision', '#eb6834', 'found objects real'),
                             ('support_acc', '#2a78d6', 'rests-on right')):
        ys = [st.mean([r[key] for r in ps if r['req_resolution'] == lv and r.get(key) is not None] or [0])
              for lv in levels]
        ax.plot(x, [100 * v for v in ys], 'o-', color=color, label=name, ms=4)
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=7.5, color=INK)
    ax.set_ylabel('% (mean over photos)', fontsize=7.5, color=MUTED)
    ax.legend(fontsize=6.5, frameon=False, loc='lower right')
    _title(ax, 'How much of the scene was read right')
    _save(plt, fig, out)
    return out


def chart_photo_vs_json(rows, out: Path):
    ps = [r for r in rows if r['group'] == 'photo_plan']
    if not ps:
        return None
    tasks = ordered(ps, lambda r: r['name'])
    plt, fig, axes = _fig(9.6, 0.3 * len(tasks) * 2 + 2, 2)
    for ax, key, title, log in ((axes[0], 'wall_s', 'Seconds for the same task', False),
                                (axes[1], 'sent_bytes', 'Bytes sent to the server (log)', True)):
        ax.grid(axis='y', visible=False)
        ys = list(range(len(tasks)))[::-1]
        for j, (route, off) in enumerate((('json', .2), ('photo', -.2))):
            vals, oks = [], []
            for t in tasks:
                rs = [r for r in ps if r['name'] == t and r['route'] == route and r[key] is not None]
                vals.append(med([r[key] for r in rs]) or 0)
                oks.append(all(r['passed'] for r in rs) if rs else None)
            ax.barh([y + off for y in ys], vals, height=0.36, color=ROUTE_COLORS[route], label=f'{route} route',
                    edgecolor=SURFACE)
            if key == 'wall_s':
                for y, v, ok in zip(ys, vals, oks):
                    if ok is not None:
                        ax.text(v, y + off, ' ✓' if ok else ' ✗', va='center', fontsize=7,
                                color=OUTCOME_COLORS['correct'] if ok else OUTCOME_COLORS['wrong plan'])
        if log:
            ax.set_xscale('log')
        ax.set_yticks(ys)
        ax.set_yticklabels(tasks if ax is axes[0] else [], fontsize=7, color=INK)
        _title(ax, title)
    axes[0].legend(fontsize=7, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    _save(plt, fig, out)
    return out


def chart_quality(rows, out: Path):
    rs = [r for r in rows if r['status'] == 'ok' and r['suite'] in ('json', 'scale') and r['moves'] is not None]
    labels = ordered(rs, lab)
    if not labels:
        return None
    plt, fig, (ax,) = _fig(7.2, 0.32 * len(labels) + 1.4)
    ax.grid(axis='y', visible=False)
    ys = list(range(len(labels)))[::-1]
    proven = [100 * sum(bool(r['optimal']) for r in rs if lab(r) == l) / sum(lab(r) == l for r in rs) for l in labels]
    ax.barh(ys, proven, color='#1baf7a', height=0.62, label='proven shortest')
    ax.barh(ys, [100 - p for p in proven], left=proven, color='#b9b8b2', height=0.62, label='verified, not proven shortest')
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel('% of plans that came back ok', fontsize=7.5, color=MUTED)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.16))
    _title(ax, 'Plan quality')
    _save(plt, fig, out)
    return out


def chart_compare(runs: list[dict], out: Path):
    labels = ordered([r for run in runs for r in run['rows']], lab)
    if len(runs) < 2 or not labels:
        return None
    plt, fig, (a, b) = _fig(10, 0.36 * len(labels) + 1.6, 2, sharey=True)
    palette = ['#2a78d6', '#eb6834', '#1baf7a', '#8a3ffc', '#e0b000']
    h = 0.8 / len(runs)
    ys = list(range(len(labels)))[::-1]
    for i, run in enumerate(runs):
        name = run['meta'].get('label') or run['meta']['health'].get('model', f'run {i + 1}')
        passed, times = [], []
        for l in labels:
            rs = [r for r in run['rows'] if lab(r) == l]
            passed.append(100 * sum(r['passed'] for r in rs) / len(rs) if rs else 0)
            times.append(med([r['wall_s'] for r in rs]) or 0)
        off = [y + (i - (len(runs) - 1) / 2) * h for y in ys]
        a.barh(off, passed, height=h * .9, color=palette[i % 5], label=name)
        b.barh(off, times, height=h * .9, color=palette[i % 5])
    for ax in (a, b):
        ax.grid(axis='y', visible=False)
    a.set_yticks(ys)
    a.set_yticklabels(labels, fontsize=7.5, color=INK)
    a.set_xlim(0, 100)
    a.legend(fontsize=7, frameon=False, loc='lower right')
    _title(a, '% correct')
    _title(b, 'median seconds')
    _save(plt, fig, out)
    return out


# --------------------------------------------------------------------------- headline numbers
def headline(rows) -> list[tuple[str, str, str]]:
    """(big number, what it is, small note): computed from the rows only."""
    out = []
    js = [r for r in rows if r['suite'] == 'json']
    if js:
        out.append((f"{sum(r['passed'] for r in js)}/{len(js)}", 'sentence + JSON tasks correct',
                    'plan replayed on the true scene and checked against a reference'))
    wrong = [r for r in rows if r['outcome'] == 'wrong plan']
    out.append((str(len(wrong)), "plans that were 'ok' but wrong", 'the dangerous kind; 0 is the goal'))
    goal = [r['goal_s'] for r in js if r['goal_s'] is not None]
    if goal:
        out.append((f'{med(goal):.2f} s', 'median model time for a sentence', f'max {max(goal):.2f} s'))
    pr = [r for r in rows if r['group'] == 'perceive' and r.get('req_resolution') == 'high' and r['perceive_s']]
    if pr:
        out.append((f"{med([r['perceive_s'] for r in pr]):.1f} s", 'median time to read a photo (high)',
                    f"{len(pr)} photos"))
    pp = [r for r in rows if r['group'] == 'photo_plan']
    js_b = [r['sent_bytes'] for r in pp if r['route'] == 'json' and r['sent_bytes']]
    ph_b = [r['sent_bytes'] for r in pp if r['route'] == 'photo' and r['sent_bytes']]
    if js_b and ph_b:
        out.append((f'{med(ph_b) / med(js_b):.0f}x', 'bytes: photo against scene JSON', f'{med(ph_b):,.0f} vs {med(js_b):,.0f}'))
    sc = [r for r in rows if r['suite'] == 'scale' and r['status'] == 'ok']
    if sc:
        big = max(r['n_objects'] for r in sc)
        out.append((str(big), 'most objects planned without a model', 'scale suite'))
    return out


# --------------------------------------------------------------------------- html
def _embed(p: Path) -> str:
    return 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()


CAPTIONS = {
    'outcomes.png': 'Each bar is one kind of task. Green = correct. Red = the server said ok but the plan does not do '
                    'the task (read the failures list). Blue/grey = the model asked a question or no plan exists; '
                    'for the refusal group that is the right answer and counts as correct.',
    'latency.png': 'Perception = the VLM reading a photo, goal = the model turning the sentence into a goal, planning = '
                   'the server searching moves. Rest is upload, network and checking.',
    'scaling.png': 'Left: seconds against the number of objects, without a model (solid) and with the model in the '
                   'loop (dashed). Right: moves in the plan; open dots are verified but not proven shortest.',
    'photo_resolution.png': 'Lower resolution is faster and uses fewer tokens; the right panel shows what it costs in '
                            'objects found and in "what rests on what" read correctly.',
    'photo_vs_json.png': 'The same task from the photo and from the true scene as JSON. ✓/✗ = the plan was correct. '
                         'The JSON route sends a few hundred bytes and skips the VLM.',
    'quality.png': 'Plans are always verified. "Proven shortest" needs a lower bound or an exact search; plans from '
                   'general goals are verified only.',
    'compare.png': 'Same cases, several runs side by side.',
}


def build(runs: list[Path], out: Path | None = None) -> Path:
    data = [load(r) for r in runs]
    primary = data[0]
    rows = primary['rows']
    out = out or runs[0]
    charts = out / 'charts'
    charts.mkdir(parents=True, exist_ok=True)
    made = [chart_outcomes(rows, charts / 'outcomes.png'), chart_latency(rows, charts / 'latency.png'),
            chart_scaling(rows, charts / 'scaling.png'), chart_photo_resolution(rows, charts / 'photo_resolution.png'),
            chart_photo_vs_json(rows, charts / 'photo_vs_json.png'), chart_quality(rows, charts / 'quality.png'),
            chart_compare(data, charts / 'compare.png')]
    made = [m for m in made if m]
    meta, health = primary['meta'], primary['meta'].get('health') or {}
    cards = ''.join(f'<div class="card"><b>{html.escape(v)}</b><span>{html.escape(k)}</span>'
                    f'<small>{html.escape(n)}</small></div>' for v, k, n in headline(rows))
    cached = sum(1 for r in rows if r.get('memory_goal') == 'hit' or r.get('memory_plan') == 'hit')
    notes = []
    if cached:
        notes.append(f'{cached} requests were answered from the experience database; their times are not model '
                     'times. Start the server with --no-memory for clean numbers.')
    if meta.get('memory') is not None and not cached:
        notes.append('The experience database was on but gave no hits in this run.')
    sections = ''.join(f'<h2>{html.escape(p.stem.replace("_", " "))}</h2><p>{CAPTIONS.get(p.name, "")}</p>'
                       f'<img src="{_embed(p)}" alt="{p.stem}">' for p in made)
    bad = [r for r in rows if not r['passed']]
    fail_rows = ''.join(
        f"<tr><td>{html.escape(lab(r))}</td><td>{html.escape(r['name'])}</td><td>{html.escape(r['outcome'])}</td>"
        f"<td>{html.escape((r['instruction'] or '')[:90])}</td><td>{html.escape(r['goal_text'][:160])}</td>"
        f"<td>{html.escape('; '.join(r['reasons'])[:160])}</td></tr>" for r in bad)
    all_rows = ''.join(
        f"<tr><td>{html.escape(lab(r))}</td><td>{html.escape(r['name'])}</td><td>{html.escape(str(r['status']))}</td>"
        f"<td>{html.escape(r['outcome'])}</td><td>{r['wall_s']}</td><td>{r['moves'] if r['moves'] is not None else ''}</td>"
        f"<td>{html.escape(str(r['planner'] or ''))}</td></tr>" for r in rows)
    summary = ''.join(
        f"<tr><td>{s['suite']}</td><td>{s['group']}</td><td>{s['route']}</td><td>{s['passed']}/{s['n']}</td>"
        f"<td>{s['median_s']}</td><td>{s['max_s']}</td></tr>" for s in primary['summary'])
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Live benchmark {html.escape(meta.get('label') or '')}</title>
<style>
body{{margin:0;background:{SURFACE};color:{INK};font:15px/1.55 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 60px}}
h1{{font-size:1.6rem;margin:0 0 4px}}h2{{font-size:1.1rem;margin:30px 0 4px;text-transform:capitalize}}
p{{margin:0 0 8px;color:{MUTED};max-width:75ch}}.meta{{color:{MUTED};font-size:.85rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:18px 0}}
.card{{border:1px solid {GRID};border-radius:8px;padding:10px 12px;display:grid;gap:1px}}
.card b{{font-size:1.5rem}}.card span{{font-size:.85rem}}.card small{{color:{MUTED};font-size:.75rem}}
img{{width:100%;height:auto;border:1px solid {GRID};border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:.8rem}}td,th{{border-bottom:1px solid {GRID};padding:5px 8px;text-align:left;vertical-align:top}}
th{{color:{MUTED};font-weight:600}}.wrap{{overflow-x:auto}}.note{{background:#fff6d6;color:#5c4400;padding:8px 12px;border-radius:6px}}
</style></head><body><main>
<h1>Live benchmark {html.escape(meta.get('label') or '')}</h1>
<div class="meta">model {html.escape(str(health.get('model')))} · {html.escape(str(health.get('quant') or health.get('quantization') or ''))}
 · {html.escape(meta.get('started', ''))} · {meta.get('seconds')} s · {len(rows)} requests · server {html.escape(meta.get('server', ''))}</div>
{''.join(f'<p class="note">{html.escape(n)}</p>' for n in notes)}
<div class="cards">{cards}</div>
<h2>Summary</h2><div class="wrap"><table><tr><th>suite</th><th>group</th><th>route</th><th>correct</th><th>median s</th><th>max s</th></tr>{summary}</table></div>
{sections}
<h2>What failed</h2><p>{len(bad)} of {len(rows)} requests. "Goal as understood" is the model's goal in plain words.</p>
<div class="wrap"><table><tr><th>kind</th><th>case</th><th>outcome</th><th>instruction</th><th>goal as understood</th><th>why</th></tr>{fail_rows}</table></div>
<h2>All requests</h2><details><summary>show {len(rows)} rows</summary><div class="wrap"><table><tr><th>kind</th><th>case</th><th>status</th><th>outcome</th><th>seconds</th><th>moves</th><th>planner</th></tr>{all_rows}</table></div></details>
</main></body></html>"""
    path = out / 'report.html'
    path.write_text(page)
    print(f'report: {path}')
    return path


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    runs = [Path(a) for a in sys.argv[1:]]
    out = runs[0] if len(runs) == 1 else runs[0].parent / ('compare_' + '_vs_'.join(r.name[16:] or r.name for r in runs))
    build(runs, out)


if __name__ == '__main__':
    main()
