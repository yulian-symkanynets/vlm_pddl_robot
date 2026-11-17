"""Graphs of what the server did: where the time goes, what the database saved, resolution.

    python run_report.py                      # from logs/requests -> logs/report/index.html
    python run_report.py --last 50            # only the 50 most recent requests
    open http://<server>:8010/dashboard       # the same page, rebuilt on every visit

Reads every logged request (logs/requests/<id>/request.json + response.json; the server
writes these for every call) and, if present, results/resolution_sweep_<model>.json.

Charts
    stages.png       time per request, split into perception / goal / planning / rest
    by_type.png      average time per stage for each kind of request
    memory.png       model calls and planner runs answered by the experience database
    resolution.png   perception time and image tokens per resolution level (image requests)
    sweep.png        resolution sweep: plan works / shortest, and seconds, per level
    models.png       every model from scripts/compare_models.sh: seconds against plans that work
    outcomes.png     request outcomes and how the plan was proven shortest
"""
from __future__ import annotations

import argparse
import html
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE_COLORS = {"perception": "#2a78d6", "goal": "#eb6834", "planning": "#1baf7a", "rest": "#b9b8b2"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(log_dir: Path, last: int | None = None) -> list[dict]:
    rows = []
    for d in sorted(p for p in log_dir.iterdir() if p.is_dir() and (p / "response.json").exists()):
        try:
            res = json.loads((d / "response.json").read_text())
            req = json.loads((d / "request.json").read_text()) if (d / "request.json").exists() else {}
        except (json.JSONDecodeError, OSError):
            continue
        t = res.get("timings") or {}
        mem = res.get("memory") or {}
        kind = "image" if req.get("images") else "json"
        if not req.get("images") and not req.get("scene") and req.get("goal"):
            kind = "solve"
        perc, goal, plan = t.get("perceive_s") or 0.0, t.get("goal_s") or 0.0, t.get("plan_s") or 0.0
        total = t.get("total_s") or (perc + goal + plan)
        rows.append({
            "id": d.name, "kind": kind, "status": res.get("status"), "total": total,
            "perception": perc, "goal": goal, "planning": plan, "rest": max(total - perc - goal - plan, 0.0),
            "memory": mem, "memory_hit": any(v == "hit" for k, v in mem.items() if k in ("scene", "goal")),
            "resolution": (res.get("resolution") or {}).get("used"),
            "image_tokens": sum(x.get("approx_tokens", 0) for x in (res.get("resolution") or {}).get("tried", [])[-1:]),
            "proof": (res.get("quality") or {}).get("proof"), "optimal": (res.get("quality") or {}).get("optimal"),
            "bytes": res.get("received_bytes"), "model": res.get("model"), "decision": res.get("decision")})
    return rows[-last:] if last else rows


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return plt, fig, ax


def _save(plt, fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def label(r) -> str:
    tags = [r["kind"]]
    if r["resolution"]:
        tags.append(r["resolution"])
    if r["memory_hit"]:
        tags.append("memory")
    return f"{r['id'][9:15]} {' '.join(tags)}"


def chart_stages(rows, out: Path):
    rows = rows[-30:]
    plt, fig, ax = _fig(7.5, 0.25 * len(rows) + 1.3)
    y = list(range(len(rows)))[::-1]
    left = [0.0] * len(rows)
    for stage, col in STAGE_COLORS.items():
        vals = [r[stage] for r in rows]
        ax.barh(y, vals, left=left, color=col, height=0.62, label=stage, edgecolor=SURFACE, linewidth=0.8)
        left = [a + b for a, b in zip(left, vals)]
    for yy, r in zip(y, rows):
        ax.text(r["total"], yy, f" {r['total']:.2f} s", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([label(r) for r in rows], fontsize=6.5)
    ax.set_xlabel("seconds per request (most recent at the top)", fontsize=8, color=MUTED)
    ax.set_xlim(0, max(r["total"] for r in rows) * 1.18 or 1)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(plt, fig, out / "stages.png")


def request_type(r) -> str:
    return f"{r['kind']}{' + memory' if r['memory_hit'] else ''}"


def chart_by_type(rows, out: Path):
    groups = {}
    for r in rows:
        groups.setdefault(request_type(r), []).append(r)
    names = sorted(groups)
    plt, fig, ax = _fig(7.5, 0.45 * len(names) + 1.2)
    left = [0.0] * len(names)
    for stage, col in STAGE_COLORS.items():
        vals = [st.mean(r[stage] for r in groups[n]) for n in names]
        ax.barh(names, vals, left=left, color=col, height=0.55, label=stage, edgecolor=SURFACE, linewidth=0.8)
        left = [a + b for a, b in zip(left, vals)]
    for i, n in enumerate(names):
        ax.text(left[i], i, f" {left[i]:.2f} s  ({len(groups[n])} requests)", va="center", fontsize=7, color=INK)
    ax.set_xlim(0, max(left) * 1.45 or 1)
    ax.set_xlabel("average seconds per request", fontsize=8, color=MUTED)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(plt, fig, out / "by_type.png")


def chart_memory(rows, out: Path):
    with_mem = [r for r in rows if r["memory"]]
    if not with_mem:
        return False
    plt, fig, ax = _fig(7.5, 2.6)
    ends = []
    for kind, col, name, style, width in (("plan", "#1baf7a", "planner runs answered", "-", 3.2),
                                          ("goal", "#eb6834", "goal model calls answered", (0, (4, 2)), 1.8),
                                          ("scene", "#2a78d6", "perception model calls answered", "-", 1.8)):
        acc, ys = 0, []
        for r in with_mem:
            acc += r["memory"].get(kind) == "hit"
            ys.append(acc)
        ax.plot(range(1, len(ys) + 1), ys, color=col, linewidth=width, linestyle=style, label=f"{name}: {ys[-1]}")
        ends.append(ys[-1])
    ax.set_xlabel("requests with the database switched on", fontsize=8, color=MUTED)
    ax.set_ylabel("answered so far", fontsize=8, color=MUTED)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _save(plt, fig, out / "memory.png")
    return True


def chart_resolution(rows, out: Path):
    img = [r for r in rows if r["resolution"]]
    if not img:
        return False
    order = ["low", "medium", "high", "full"]
    levels = sorted({r["resolution"] for r in img}, key=lambda x: (order.index(x) if x in order else 9, x))
    plt, fig, ax = _fig(7.5, 2.6)
    for i, lv in enumerate(levels):
        vals = [r["perception"] for r in img if r["resolution"] == lv]
        ax.scatter([i] * len(vals), vals, s=14, color="#2a78d6", alpha=0.7, edgecolors="none")
        toks = [r["image_tokens"] for r in img if r["resolution"] == lv]
        ax.text(i, max(vals) * 1.05 + 0.01, f"median {st.median(vals):.2f} s\n~{st.mean(toks):.0f} tokens",
                ha="center", fontsize=6.8, color=INK)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(0, max(r["perception"] for r in img) * 1.35 + 0.05)
    ax.set_ylabel("perception seconds per request", fontsize=8, color=MUTED)
    ax.set_xlabel("resolution used", fontsize=8, color=MUTED)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _save(plt, fig, out / "resolution.png")
    return True


def chart_sweep(out: Path):
    files = sorted((ROOT / "results").glob("resolution_sweep_*.json"))
    files = [f for f in files if not f.name.endswith("_mock.json")]
    if not files:
        return False
    data = json.loads(files[-1].read_text())
    summ = data["summary"]
    levels = list(summ)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    works = [100 * summ[l]["works"] / summ[l]["runs"] for l in levels]
    short = [100 * summ[l]["shortest"] / summ[l]["runs"] for l in levels]
    ax.plot(levels, works, color="#2a78d6", marker="o", linewidth=2, label="plan works on the true scene")
    ax.plot(levels, short, color="#1baf7a", marker="o", linewidth=2, label="... and is the shortest")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of runs", fontsize=8, color=MUTED)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax = axes[1]
    secs = [summ[l]["mean_seconds"] for l in levels]
    ax.bar(levels, secs, color="#2a78d6", width=0.55)
    for i, (s, l) in enumerate(zip(secs, levels)):
        ax.text(i, s, f"{s:.1f} s\n~{summ[l]['mean_image_tokens']} tok", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(0, max(secs) * 1.35)
    ax.set_ylabel("seconds per request", fontsize=8, color=MUTED)
    for a in axes:
        _style(a)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.tick_params(axis="x", labelsize=7)
    fig.suptitle(f"resolution sweep, model {data['model']}", fontsize=8.5, color=MUTED)
    _save(plt, fig, out / "sweep.png")
    return True


def chart_models(out: Path):
    """All models from results/resolution_sweep_*.json: seconds per request against plans that work."""
    files = [f for f in sorted((ROOT / "results").glob("resolution_sweep_*.json")) if not f.name.endswith("_mock.json")]
    if not files:
        return False
    plt, fig, ax = _fig(7.5, 3.4)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for i, f in enumerate(files):
        data = json.loads(f.read_text())
        pts = [(v["mean_seconds"], 100 * v["works"] / v["runs"], lvl) for lvl, v in data["summary"].items()]
        pts.sort()
        col = palette[i % len(palette)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", markersize=5, linewidth=1.5,
                label=data["model"])
        for x, y, lvl in pts:
            ax.annotate(lvl, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.3, color=MUTED)
    ax.set_xlabel("seconds per request (lower is faster)", fontsize=8, color=MUTED)
    ax.set_ylabel("% of runs whose plan works on the true scene", fontsize=8, color=MUTED)
    ax.set_ylim(0, 105)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _save(plt, fig, out / "models.png")
    return True


def chart_outcomes(rows, out: Path):
    counts = {}
    for r in rows:
        key = r["status"] or "?"
        if r["status"] == "ok":
            key = "ok, shortest: " + ("planner search" if r["proof"] and "astar" in r["proof"]
                                      else "lower bound" if r["proof"] else "not proven")
        counts[key] = counts.get(key, 0) + 1
    names = sorted(counts, key=lambda k: -counts[k])
    plt, fig, ax = _fig(7.5, 0.32 * len(names) + 1.0)
    ax.barh(names[::-1], [counts[n] for n in names[::-1]], color="#2a78d6", height=0.55)
    for i, n in enumerate(names[::-1]):
        ax.text(counts[n], i, f" {counts[n]}", va="center", fontsize=7, color=INK)
    ax.set_xlabel("requests", fontsize=8, color=MUTED)
    _style(ax)
    _save(plt, fig, out / "outcomes.png")


def build(log_dir: Path = ROOT / "logs" / "requests", out: Path = ROOT / "logs" / "report",
          last: int | None = None, img_prefix: str = "") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    rows = load(log_dir, last) if log_dir.exists() else []
    parts = ["<!doctype html><meta charset='utf-8'><title>Planning server runs</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;"
             "color:#0b0b0b;background:#fcfcfb}img{max-width:100%;border:1px solid #e4e3df;margin:6px 0 18px}"
             "td,th{padding:3px 10px;border-bottom:1px solid #e4e3df;text-align:left;font-size:14px}"
             "h2{font-size:17px;margin-top:26px}p{color:#52514e}</style>",
             "<h1>Planning server runs</h1>"]
    if not rows:
        parts.append("<p>No logged requests yet. Send some requests to the server, then reload.</p>")
    else:
        totals = [r["total"] for r in rows]
        hits = {k: sum(1 for r in rows if r["memory"].get(k) == "hit") for k in ("scene", "goal", "plan")}
        with_mem = sum(1 for r in rows if r["memory"])
        summary = [("requests", len(rows)), ("ok", sum(r["status"] == "ok" for r in rows)),
                   ("median total", f"{st.median(totals):.2f} s"),
                   ("95th percentile", f"{sorted(totals)[int(0.95 * (len(totals) - 1))]:.2f} s"),
                   ("model", ", ".join(sorted({str(r['model']) for r in rows}))),
                   ("database on", f"{with_mem} requests"),
                   ("answered from the database", f"perception {hits['scene']}, goal {hits['goal']}, "
                                                  f"plan {hits['plan']}")]
        parts.append("<table>" + "".join(f"<tr><th>{html.escape(str(a))}</th><td>{html.escape(str(b))}</td></tr>"
                                         for a, b in summary) + "</table>")
        chart_stages(rows, out)
        chart_by_type(rows, out)
        chart_outcomes(rows, out)
        sections = [("Time per request", "stages.png", "Blue perception (image to scene), orange goal (instruction "
                     "to goal), green planning, grey everything else. Labels: time of the request, path, "
                     "resolution, and whether the database answered."),
                    ("Average time by kind of request", "by_type.png", "")]
        if chart_memory(rows, out):
            sections.append(("Answered from the experience database", "memory.png",
                             "Each answered perception or goal call is one model call that did not run."))
        if chart_resolution(rows, out):
            sections.append(("Image resolution", "resolution.png", "Perception time of image requests, by the "
                             "resolution that was finally used."))
        sections.append(("Outcomes", "outcomes.png", ""))
        for title, img, text in sections:
            parts.append(f"<h2>{title}</h2><p>{text}</p><img src='{img_prefix}{img}'>")
    if chart_models(out):
        parts.append(f"<h2>Models: speed against correct plans</h2><p>Each line is one model, each point one "
                     f"resolution level (scripts/compare_models.sh). Best is top left.</p><img src='{img_prefix}models.png'>")
    if chart_sweep(out):
        parts.append(f"<h2>Resolution sweep</h2><p>From scripts/resolution_sweep.py.</p><img src='{img_prefix}sweep.png'>")
    page = out / "index.html"
    page.write_text("\n".join(parts))
    return page


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=ROOT / "logs" / "requests")
    ap.add_argument("--out", type=Path, default=ROOT / "logs" / "report")
    ap.add_argument("--last", type=int)
    args = ap.parse_args()
    print("wrote", build(args.logs, args.out, args.last))


if __name__ == "__main__":
    main()
