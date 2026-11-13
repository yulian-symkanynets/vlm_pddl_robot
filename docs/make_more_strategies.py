"""Build docs/more_strategies.pdf: six further strategies, each with a diagram.

    python -m approaches.bench_extra      # writes approaches/results/bench_extra.json
    python docs/make_more_strategies.py

The diagrams are drawn from real planner output while the PDF is built.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from approaches import run, scenarios as SC
from approaches.bench_extra import true_stats
from approaches.contingent import build_tree
from approaches.cost_aware import CostAwareStrategy, move_success
from approaches.partial_order import deorder
from approaches.symbolic import GraphStrategy
from semplan.scene import apply, parse_scene

B = json.loads((ROOT / "approaches" / "results" / "bench_extra.json").read_text())
OUT = ROOT / "docs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SOFT_BLUE, SOFT_GREEN, SOFT_ORANGE, SOFT_GREY = "#e3eefb", "#e3f3ea", "#fdebe2", "#f1f1ee"


def ms(x):
    if x is None:
        return "n/a"
    return f"{x:.0f} ms" if x >= 100 else f"{x:.1f} ms" if x >= 10 else f"{x:.2f} ms" if x >= 1 \
        else f"{x:.3f} ms"


def pc(x, d=0):
    return f"{100 * x:.{d}f}%"


# ================================================================ drawing helpers

def canvas(w, h, xmax=100, ymax=60):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, face="white", edge="#8a8984", size=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, fontsize=size, color=INK, ha="center", va="center",
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, text="", col=MUTED, rad=0.0, dy=1.6, size=6.3, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, linewidth=1.1, color=col,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, fontsize=size, color=col,
                ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


def save(fig, name):
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / name, facecolor=fig.get_facecolor())
    plt.close(fig)


def short(m):
    return f"{m.obj}\n-> {m.dst}"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ================================================================ diagrams

def fig_overview():
    fig, ax = canvas(7.2, 3.1, 100, 44)
    y = 18
    stages = [("Scene message\n(version n)", SOFT_GREY), ("Confidence\ngate", SOFT_BLUE),
              ("Planner", SOFT_BLUE), ("Validator", SOFT_GREEN), ("Plan to\nexecutor", SOFT_GREY)]
    xs = [2, 22, 42, 62, 82]
    for (label, face), x in zip(stages, xs):
        box(ax, x, y, 16, 9, label, face, size=7.5, bold=True)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 16, y + 4.5), (b, y + 4.5))
    notes = [
        (22, 34, "6. Calibrated threshold\nreplaces a hand-set 0.8", SOFT_ORANGE, (30, 34), (30, 27)),
        (22, 2, "3. Contingent plan\nbranches prepared in advance", SOFT_ORANGE, (30, 11), (30, 18)),
        (41, 34, "2. Cost-aware\n5. Constraint solver\n4. Anytime", SOFT_ORANGE, (50, 34), (50, 27)),
        (80, 34, "1. Partial order\nplan as a graph", SOFT_ORANGE, (90, 34), (90, 27)),
    ]
    for x, yy, text, face, p, q in notes:
        box(ax, x, yy, 18, 9, text, face, edge="#d9a98f", size=6.6)
        arrow(ax, p, q, col=ORANGE)
    arrow(ax, (90, 18), (50, 18), "step feedback: success / failure statistics", col=AQUA, rad=-0.35, dy=-9.5)
    save(fig, "overview.png")


def fig_partial_order():
    scene = parse_scene(SC.DAG_EXAMPLE)
    moves = run(GraphStrategy(), scene).moves
    plan = deorder(scene, moves)
    edges = plan.edges()
    depth = {i: 0 for i in range(len(moves))}
    for _ in moves:
        for a, b in edges:
            depth[b] = max(depth[b], depth[a] + 1)
    fig, ax = canvas(7.2, 3.9, 100, 56)
    ax.text(2, 53.5, "Total order: one fixed sequence", fontsize=8, fontweight="bold", color=INK)
    w, h = 13.4, 7.5
    for i, m in enumerate(moves):
        x = 2 + i * 16.4
        box(ax, x, 43, w, h, f"{i + 1}. {short(m)}", SOFT_GREY, size=6)
        if i:
            arrow(ax, (x - 3.0, 46.75), (x, 46.75))
    ax.text(2, 36.5, "Partial order: only the dependencies that matter", fontsize=8, fontweight="bold",
            color=INK)
    layers: dict[int, list[int]] = {}
    for i, d in depth.items():
        layers.setdefault(d, []).append(i)
    pos = {}
    for d, items in layers.items():
        for k, i in enumerate(items):
            pos[i] = (6 + d * 24, 25 - k * 11.5)
    for a, b in edges:
        arrow(ax, (pos[a][0] + w, pos[a][1] + h / 2), (pos[b][0], pos[b][1] + h / 2), col=BLUE)
    for i, m in enumerate(moves):
        ready = not plan.before[i]
        box(ax, pos[i][0], pos[i][1], w, h, f"{i + 1}. {short(m)}", SOFT_GREEN if ready else "white",
            edge=AQUA if ready else "#8a8984", size=6)
    ax.text(80, 25, "green = can start now\n\nany order that follows\nthe arrows is valid\n\n"
            f"{pc(plan.flexibility())} of step pairs\nare left unordered", fontsize=7, color=MUTED, va="top")
    save(fig, "partial_order.png")
    return moves, plan


def fig_cost_aware():
    scene = parse_scene(SC.BLOCKED)
    stats = true_stats()
    plans = [("Fewest moves (graph planner)", run(GraphStrategy(), scene).moves, SOFT_GREY),
             ("Fewest expected attempts (cost-aware)", run(CostAwareStrategy(stats), scene).moves, SOFT_GREEN)]
    fig, ax = canvas(7.2, 3.0, 100, 44)
    totals = []
    for row, (title, moves, face) in enumerate(plans):
        y = 26 - row * 21
        ax.text(2, y + 12.5, title, fontsize=8, fontweight="bold", color=INK)
        s = scene.copy()
        total = 0.0
        for i, m in enumerate(moves):
            p = move_success(s, m, stats)
            total += 1 / p
            risky = p < 0.8
            x = 2 + i * 20.5
            box(ax, x, y, 17, 9, f"{short(m)}", SOFT_ORANGE if risky else face,
                edge=ORANGE if risky else "#8a8984", size=6.2)
            ax.text(x + 8.5, y - 2.2, f"p = {p:.2f}", fontsize=6.5, ha="center",
                    color=ORANGE if risky else MUTED)
            if i:
                arrow(ax, (x - 3.5, y + 4.5), (x, y + 4.5))
            apply(s, m)
        totals.append(total)
        ax.text(86, y + 4.5, f"{total:.2f}\nexpected\nattempts", fontsize=7.5, ha="left", va="center",
                color=INK, fontweight="bold")
    save(fig, "cost_aware.png")
    return plans, totals


def fig_contingent():
    probs = {"unknown_1": {"blue": 0.55, "green": 0.45}}
    tree = build_tree(parse_scene(SC.OCCLUDED), probs)
    fig, ax = canvas(7.2, 3.0, 100, 44)
    x = 2
    for m in tree["moves"]:
        box(ax, x, 18, 16, 9, short(m), SOFT_GREY, size=6.4)
        arrow(ax, (x + 16, 22.5), (x + 21, 22.5))
        x += 21
    box(ax, x, 18, 15, 9, f"OBSERVE\n{tree['observe']}", SOFT_BLUE, edge=BLUE, size=6.8, bold=True)
    ax.text(2, 31, "safe whatever the colour is", fontsize=6.8, color=MUTED)
    ox = x + 15
    for k, (colour, sub) in enumerate(tree["branches"].items()):
        y = 31 - k * 26
        arrow(ax, (ox, 22.5), (ox + 8, y + 4.5), f"if {colour} (p={probs['unknown_1'][colour]:.2f})",
              col=BLUE, dy=3.4 if k == 0 else -3.4)
        bx = ox + 8
        for i, m in enumerate(sub["moves"]):
            box(ax, bx, y, 16.5, 9, short(m), SOFT_GREEN, size=6.2)
            if i:
                arrow(ax, (bx - 3, y + 4.5), (bx, y + 4.5))
            bx += 19.5
    ax.text(ox + 9, 20, "the laptop continues down the matching branch\nwithout another call to the server",
            fontsize=6.6, color=MUTED, va="center")
    save(fig, "contingent.png")
    return tree


def fig_anytime():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=200, gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 44)
    ax.axis("off")
    stages = [("1. Constructive planner\nvalid plan immediately", SOFT_GREEN, 32),
              ("2. Remove redundant moves", SOFT_BLUE, 19),
              ("3. A* until the budget ends\nkeep the shortest plan found", SOFT_BLUE, 6)]
    for label, face, y in stages:
        box(ax, 3, y, 36, 8.5, label, face, size=6.8)
    arrow(ax, (21, 32), (21, 27.5))
    arrow(ax, (21, 19), (21, 14.5))
    arrow(ax, (44, 40), (44, 6), "time", col=MUTED, dy=0)
    ax = axes[1]
    for key, col, label in (("8", BLUE, "8 objects"), ("10", ORANGE, "10 objects")):
        rows = B["anytime"][key]
        xs = [max(r["budget_ms"], 0.1) for r in rows]
        ys = [100 * r["optimal"] / r["of"] for r in rows]
        ax.plot(xs, ys, color=col, marker="o", markersize=4.5, linewidth=2, label=label,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.5, 2, 10, 50])
    ax.set_xticklabels(["0", "0.5", "2", "10", "50"])
    ax.set_ylim(90, 100.6)
    ax.set_xlabel("time budget, ms", fontsize=8, color=MUTED)
    ax.set_ylabel("plans that are optimal, %", fontsize=8, color=MUTED)
    style_axes(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "anytime.png", facecolor="white")
    plt.close(fig)


def fig_cpsat():
    from approaches.cpsat import CpSatStrategy
    scene = parse_scene(SC.BLOCKED)
    strategy = CpSatStrategy(avoid=["tmp_1"])
    moves = run(strategy, scene).moves
    objs = sorted(scene.objects)
    states, s = [dict(scene.on)], scene.copy()
    for m in moves:
        apply(s, m)
        states.append(dict(s.on))
    fig, ax = canvas(7.2, 2.9, 100, 42)
    ax.text(2, 39.5, "One row per object, one column per time step. Each cell is a choice the solver makes: "
            "\"object rests on ...\"", fontsize=7, color=MUTED)
    cw, ch, x0, y0 = 15.5, 6.6, 22, 26
    for t in range(len(states)):
        ax.text(x0 + t * cw + cw / 2, y0 + ch + 2, f"t = {t}", fontsize=7.5, ha="center", color=INK,
                fontweight="bold")
    for r, o in enumerate(objs):
        y = y0 - r * (ch + 1.2)
        ax.text(x0 - 1.5, y + ch / 2, o, fontsize=7, ha="right", va="center", color=INK)
        for t, st in enumerate(states):
            changed = t > 0 and states[t - 1][o] != st[o]
            box(ax, x0 + t * cw + 0.6, y, cw - 1.2, ch, st[o], SOFT_ORANGE if changed else "white",
                edge=ORANGE if changed else "#c9c8c2", size=6.4)
    ax.text(2, 3.5, "orange = the one cell allowed to change in that step (one move per step).   "
            "Constraint added here: slot tmp_1 must stay empty.", fontsize=6.8, color=MUTED)
    save(fig, "cpsat.png")
    return moves, strategy.last_stats


def fig_calibration():
    ex = B["calibration"]["example"]
    curve = [c for c in ex["curve"] if c["n"] >= 15 and c["threshold"] >= 0.3]
    fig, ax = plt.subplots(figsize=(7.0, 2.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    xs = [c["threshold"] for c in curve]
    ax.plot(xs, [100 * c["empirical_error"] for c in curve], color=BLUE, linewidth=2,
            label="error among accepted detections (in the log)")
    ax.plot(xs, [100 * c["error_bound"] for c in curve], color=ORANGE, linewidth=2,
            label="upper bound on the true error (90% confidence)")
    ax.axhline(100 * ex["alpha"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.305, 100 * ex["alpha"] + 0.7, f"target: at most {pc(ex['alpha'])} wrong", fontsize=7.5, color=MUTED)
    if ex["threshold"] is not None:
        ax.axvline(ex["threshold"], color=AQUA, linewidth=1.5)
        ax.text(ex["threshold"] + 0.008, 17, f"chosen threshold {ex['threshold']:.2f}\n"
                f"accepts {pc(ex['accepted_fraction'])} of detections", fontsize=7.5, color=INK, va="top")
    ax.set_ylim(0, 22)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("confidence threshold (detections below it trigger another view)", fontsize=8, color=MUTED)
    ax.set_ylabel("wrong labels accepted, %", fontsize=8, color=MUTED)
    style_axes(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", facecolor=SURFACE)
    plt.close(fig)


# ================================================================ PDF helpers

S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10.5, leading=14.5,
                               textColor=colors.HexColor(MUTED), spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, spaceBefore=14,
                         spaceAfter=6, textColor=colors.HexColor("#123a6b"), keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.4, leading=14, spaceBefore=7,
                         spaceAfter=2, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.3, leading=13, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.3, leading=12.8, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.8, leading=9.8),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=7.8, leading=9.8),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7.9, leading=10.3,
                          textColor=colors.HexColor(MUTED), spaceAfter=8, spaceBefore=2),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.3, leading=9.1, leftIndent=4,
                           backColor=colors.HexColor("#f4f4f1"), borderPadding=5, spaceAfter=10,
                           spaceBefore=6),
    "box": ParagraphStyle("box", fontName="Helvetica", fontSize=9.3, leading=13,
                          backColor=colors.HexColor("#eef3fa"), borderPadding=7, spaceAfter=12,
                          spaceBefore=8, leftIndent=4, rightIndent=4),
}
story = []
H1 = lambda t: story.append(Paragraph(t, S["h1"]))
H2 = lambda t: story.append(Paragraph(t, S["h2"]))
P = lambda t: story.append(Paragraph(t, S["body"]))
BOX = lambda t: story.append(Paragraph(t, S["box"]))
CODE = lambda t: story.append(Preformatted(t.strip("\n"), S["code"]))


def BL(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="-"))
    story.append(Spacer(1, 3))


def FIGURE(name, width_mm, caption):
    img = Image(str(FIG / name))
    img.drawWidth, img.drawHeight = width_mm * mm, width_mm * mm * img.imageHeight / img.imageWidth
    story.append(KeepTogether([img, Paragraph(caption, S["cap"])]))


def TABLE(rows, widths, caption=None):
    data = [[Paragraph(str(c), S["cellb"] if i == 0 else S["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5d4cf")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#8fa3bd"))]))
    story.append(t)
    story.append(Paragraph(caption, S["cap"]) if caption else Spacer(1, 7))


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 7.5)
    canv.setFillColor(colors.HexColor(MUTED))
    canv.drawString(18 * mm, 11 * mm, "Further planning strategies for the sorting robot")
    canv.drawRightString(192 * mm, 11 * mm, f"{doc.page}")
    canv.restoreState()


# ================================================================ content

def build():
    fig_overview()
    po_moves, po_plan = fig_partial_order()
    ca_plans, ca_totals = fig_cost_aware()
    tree = fig_contingent()
    fig_anytime()
    cp_moves, cp_stats = fig_cpsat()
    fig_calibration()
    po, ca, ct, at, cp, cal = (B[k] for k in ("partial_order", "cost_aware", "contingent", "anytime",
                                               "cp_sat", "calibration"))

    story.append(Paragraph("Further planning strategies for the sorting robot", S["title"]))
    story.append(Paragraph("Six strategies beyond the first guide: how each works, a diagram, how to use it, "
                           f"and what was measured. Prepared for Yulian, {B['meta']['date']}.", S["subtitle"]))
    BOX("<b>Short answer: yes.</b> The first guide covered <i>how to find a plan</i> (graph search, PDDL, HTN, "
        "cases, language models, learned policies). The six strategies here answer different questions about "
        "the plan: in what order it may be executed, how likely it is to work, what to do when a fact is "
        "unknown, how to use spare time, how to add extra rules, and when a detection can be trusted. "
        "None of them needs a language model, a GPU or training. All six are implemented in "
        "<font face='Courier'>approaches/</font>, use the same scene and validator as the others, and were "
        "measured on synthetic scenes on one laptop "
        f"({B['meta']['system']}); nothing here was run on the real robot.")
    TABLE([
        ["#", "Strategy", "Question it answers", "File", "Main measured result"],
        ["1", "Partial-order plan", "Which steps really depend on each other?", "partial_order.py",
         f"{pc(po['6']['mean_flexibility'])} of step pairs are free to reorder at 6 objects; "
         f"{pc(po['6']['mean_travel_saved'])} less gripper travel"],
        ["2", "Failure-aware planning", "Which plan is most likely to work first time?", "cost_aware.py",
         f"{ca['6']['graph_expected_attempts']:.2f} to {ca['6']['cost_aware_expected_attempts']:.2f} expected "
         f"attempts at 6 objects; {ms(ca['6']['p50_ms'])}"],
        ["3", "Contingent plan", "Can the branches for an unknown object be prepared in advance?",
         "contingent.py", f"Tree for one unknown object in {ms(ct['1']['p50_ms'])}; "
         f"{ct['1']['invalid_paths']} invalid paths in {ct['1']['paths_checked']}"],
        ["4", "Anytime planning", "How to use spare milliseconds?", "anytime.py",
         f"Optimal plans at 10 objects: {at['10'][0]['optimal']}/{at['10'][0]['of']} with no budget, "
         f"{at['10'][-1]['optimal']}/{at['10'][-1]['of']} with 50 ms"],
        ["5", "Constraint solving (CP-SAT)", "How to add extra rules and still get a shortest plan?", "cpsat.py",
         f"Optimal in all {cp['6']['scenes']} test scenes; {ms(cp['6']['p50_ms'])} at 6 objects"],
        ["6", "Calibrated confidence threshold", "When can a detection be trusted?", "calibration.py",
         f"With a log of 1000 detections: threshold {cal['rows'][1]['mean_threshold']:.2f}, test error "
         f"{pc(cal['rows'][1]['mean_test_error'], 1)} against a 5% target"],
    ], [7, 34, 50, 28, 55], "Table 1. The six strategies.")
    FIGURE("overview.png", 165, "Figure 1. Where the six strategies attach to the existing pipeline.")
    CODE('''
# every strategy is used the same way as in the first guide
from approaches import run
from semplan.scene import parse_scene
scene = parse_scene(scene_message)
outcome = run(strategy, scene)     # .status .moves .ms .reason .info_requests
''')

    # ------------------------------------------------------------------ 1
    H1("1. Partial-order plan: the plan as a dependency graph")
    P("<b>Idea.</b> A normal plan is a list: step 1, step 2, step 3. Most of that order is accidental. "
      "Delivering the green cube and delivering the blue cylinder do not affect each other, so either may go "
      "first. A partial-order plan keeps only the orderings that are needed (a step that uncovers an object "
      "must come before the step that moves that object) and leaves the rest open.")
    FIGURE("partial_order.png", 165,
           "Figure 2. The same six moves as a sequence and as a dependency graph (real planner output for the "
           "scene DAG_EXAMPLE). Two moves can start immediately; the others wait only for the moves they "
           "actually depend on.")
    P("<b>How it works.</b> Take any valid plan. For each step, list the resources it touches: the object it "
      "moves, what that object stood on, and where it goes. Two steps are ordered only if they share a "
      "resource, and then the order of the original plan is kept. A zone with enough free poses for every "
      "placement of the plan does not count as a shared resource. Every ordering that follows the arrows is a "
      "valid plan; the tests check this by pushing random orderings through the validator.")
    CODE('''
from approaches.partial_order import deorder, travel_cost, PartialOrderStrategy
plan = deorder(scene, moves)            # moves: any valid plan
plan.edges()                            # [(0, 1), (0, 3), ...]  earlier step, later step
plan.ready(done={0})                    # steps that may run now
plan.affected_by({"green_1"})           # steps a change to this object can invalidate
order = plan.linearize(travel_cost(scene, plan, positions))   # nearest ready step first

run(PartialOrderStrategy(positions=positions), scene)         # as a drop-in strategy
''')
    rows = [["Objects", "Steps per plan", "Step pairs left unordered", "Steps ready at the start",
             "Gripper travel saved", "Time to build the graph"]]
    for n in ("4", "6", "8", "10"):
        r = po[n]
        rows.append([n, f"{r['mean_steps']:.1f}", pc(r["mean_flexibility"]), f"{r['mean_ready_at_start']:.1f}",
                     pc(r["mean_travel_saved"], 1), ms(r["deorder_p50_ms"])])
    TABLE(rows, [18, 26, 36, 34, 30, 30],
          "Table 2. Partial-order plans on 300 random scenes per size [measured]. Travel is the distance from "
          "each placement to the next pick-up, with random positions; 'saved' compares nearest-ready-step-first "
          "with the planner's original order.")
    P("<b>Use it when</b> the executor should choose the order itself: nearest object first, or skip an "
      "object that is momentarily out of reach and come back to it. It also tells the server which steps a "
      "scene change can touch. <b>Limits:</b> the graph is derived from one plan, so it cannot discover a "
      "different plan; the travel saving depends on the real layout and was measured with random positions only.")

    # ------------------------------------------------------------------ 2
    H1("2. Failure-aware planning: the plan most likely to work first time")
    P("<b>Idea.</b> The symbolic model treats every move as equally good. The robot does not: some grasps "
      "and some placements fail much more often than others. The server already counts successes and "
      "failures per kind of action from step feedback. This strategy plans with those counts.")
    FIGURE("cost_aware.png", 165,
           "Figure 3. Two valid plans for the 'blocked' scene, with the success probability of each move from "
           "the statistics used in the benchmark (temporary slots unreliable). Both have four moves; the "
           "lower one avoids the unreliable slot.")
    P("<b>How it works.</b> Each move costs its expected number of attempts, 1 / (p<sub>pick</sub> x "
      "p<sub>place</sub>), where the probabilities are the posterior means from the feedback counts for that "
      "kind of action (action, object type, what it is picked from or placed on). A* then minimises total "
      "expected attempts instead of the number of moves. Every object that still has to move costs at least "
      "one attempt, which gives an admissible heuristic. With no statistics all costs are equal and the "
      "result is the ordinary shortest plan.")
    CODE('''
from approaches.cost_aware import CostAwareStrategy, expected_attempts
strategy = CostAwareStrategy(service.stats)       # the statistics the server already collects
outcome = run(strategy, scene)
expected_attempts(scene, outcome.moves, service.stats)

# as the server's planner, learning while it runs:
#   STRATEGY=cost_aware uvicorn approaches.serve:app --port 8020
''')
    rows = [["Objects", "Expected attempts: graph planner", "Expected attempts: cost-aware", "Moves: graph / cost-aware",
             "Plans that differ", "Median time", "95th percentile"]]
    for n in ("4", "5", "6"):
        r = ca[n]
        rows.append([n, f"{r['graph_expected_attempts']:.2f}", f"{r['cost_aware_expected_attempts']:.2f}",
                     f"{r['graph_moves']:.2f} / {r['cost_aware_moves']:.2f}",
                     f"{r['plans_that_differ']}/{r['scenes']}", ms(r["p50_ms"]), ms(r["p95_ms"])])
    TABLE(rows, [16, 32, 30, 30, 24, 21, 21],
          "Table 3. Failure-aware planning on 200 scenes per size [measured]. Assumed statistics: placing in a "
          "temporary slot succeeds 70% of the time, placing on a cylinder 80%, picking a prism 75%, everything "
          "else 97%. The size of the gain depends entirely on how unequal the real probabilities are.")
    P(f"<b>Reading the result.</b> The gain is modest here (about "
      f"{pc(1 - ca['6']['cost_aware_expected_attempts'] / ca['6']['graph_expected_attempts'], 1)} fewer "
      "expected attempts at 6 objects) because most scenes offer no alternative to the risky move. "
      "It costs more time than the graph planner and grows quickly with scene size, so it falls back to the "
      "graph planner when its search budget runs out. <b>Use it when</b> real failure rates differ clearly "
      "between action kinds. <b>Limits:</b> the probabilities are per action kind, not per pose; the numbers "
      "above are assumed, not measured on the robot.")

    # ------------------------------------------------------------------ 3
    H1("3. Contingent plan: the branches prepared in advance")
    P("<b>Idea.</b> With an unidentified object the usual flow needs two round trips: plan what is safe, ask "
      "for another view, receive the new scene, plan again. A contingent plan does the second planning in "
      "advance and sends the laptop a small tree.")
    FIGURE("contingent.png", 165,
           "Figure 4. Contingent plan for the 'occluded' scene (real output). One move is right under both "
           "colour hypotheses; after observing, the laptop follows the branch that matches.")
    P("<b>How it works.</b> The server plans once per colour hypothesis and keeps the moves all plans share "
      "from the start. Where the plans begin to differ it inserts an OBSERVE step for the object that causes "
      "the difference and builds one subtree per possible colour, recursively. Every path from the root to a "
      "leaf is a complete, valid plan for its hypothesis.")
    CODE('''
from approaches.contingent import build_tree, follow, render, ContingentStrategy
tree = build_tree(scene, {"unknown_1": {"blue": 0.55, "green": 0.45}})
print(render(tree))
#   green_cube_1 -> green_zone
#   OBSERVE unknown_1
#     if blue:   unknown_1 -> blue_zone,    red_cube_1 -> red_zone
#     if green:  unknown_1 -> green_cube_1, red_cube_1 -> red_zone
follow(tree, {"unknown_1": "blue"})       # the moves executed if it turns out blue

strategy = ContingentStrategy(color_probs); run(strategy, scene); strategy.tree
''')
    rows = [["Unknown objects", "Leaves in the tree", "Moves before the first observation", "Median time",
             "95th percentile", "True-colour path valid"]]
    for k in ("1", "2"):
        r = ct[k]
        rows.append([k, f"{r['mean_leaves']:.0f}", f"{r['mean_moves_before_observing']:.2f}", ms(r["p50_ms"]),
                     ms(r["p95_ms"]), f"{r['paths_checked'] - r['invalid_paths']}/{r['paths_checked']}"])
    TABLE(rows, [28, 28, 40, 24, 26, 28],
          "Table 4. Contingent plans for 6-object scenes with three colours [measured, 200 scenes each].")
    P("<b>Use it when</b> the link to the server is slow or unreliable and one or two objects are uncertain: "
      "the laptop can finish without another request. <b>Limits:</b> the tree has (colours) to the power "
      "(unknown objects) leaves, so it is for one or two unknowns. A branch for a colour that would make the "
      "scene unsolvable is marked incomplete. The response format with a tree is not yet part of the HTTP "
      "contract; the tree is available as <font face='Courier'>strategy.tree</font>.")

    # ------------------------------------------------------------------ 4
    H1("4. Anytime planning: a valid plan now, a better one if there is time")
    P("<b>Idea.</b> The answer must never be late, but the robot is often busy for seconds with the current "
      "step. Anytime planning returns a valid plan at once and spends whatever time is allowed on making it "
      "shorter.")
    FIGURE("anytime.png", 165,
           "Figure 5. Left: the three stages. Right: share of plans that are optimal as the time budget "
           "grows [measured, 200 scenes per size].")
    P("<b>How it works.</b> Stage 1 is the constructive planner. Stage 2 removes moves whose removal leaves "
      "the plan valid, and merges 'park it, later move it' pairs when the object can go straight to its final "
      "place. Stage 3 runs A* until the budget ends and keeps its result if it is shorter. The plan returned "
      "is always valid; only its length depends on the budget.")
    CODE('''
from approaches.anytime import AnytimeStrategy
strategy = AnytimeStrategy(budget_ms=10)
outcome = run(strategy, scene)
strategy.stages          # [("constructive", 0.03, 9), ("A* improvement", 4.1, 8)]  stage, ms, plan length

#   STRATEGY=anytime BUDGET_MS=10 uvicorn approaches.serve:app --port 8020
''')
    rows = [["Budget", "8 objects: optimal", "8 objects: median / p95 / max time", "10 objects: optimal",
             "10 objects: median / p95 / max time"]]
    for a, b in zip(at["8"], at["10"]):
        rows.append([f"{a['budget_ms']:g} ms", f"{a['optimal']}/{a['of']}",
                     f"{ms(a['p50_ms'])} / {ms(a['p95_ms'])} / {ms(a['max_ms'])}",
                     f"{b['optimal']}/{b['of']}",
                     f"{ms(b['p50_ms'])} / {ms(b['p95_ms'])} / {ms(b['max_ms'])}"])
    TABLE(rows, [20, 28, 50, 28, 48], "Table 5. Anytime planning [measured].")
    P("<b>Reading the result.</b> The constructive plan is already optimal in about 95% of these scenes, so "
      "there is little left to gain; a few milliseconds recover most of it. The budget is not a hard limit: "
      "small budgets were exceeded by a few milliseconds (see the maximum column) because the search checks "
      "the clock only every 64 expansions. <b>Use it when</b> shorter plans matter (each saved move is "
      "seconds of robot time) and the server has idle time.")

    # ------------------------------------------------------------------ 5
    H1("5. Constraint solving: describe the plan instead of searching for it")
    P("<b>Idea.</b> Instead of searching step by step, write down what a legal plan looks like and let a "
      "constraint solver find the shortest one. The value is not speed. It is that an extra rule is one line, "
      "and the answer is still a provably shortest plan that obeys it.")
    FIGURE("cpsat.png", 165,
           "Figure 6. The solver's view of the 'blocked' scene with the extra rule 'do not use slot tmp_1' "
           "(real solution). The solver fills the table so that every column is a legal scene, at most one "
           "cell changes between neighbouring columns, and the last column satisfies the goal.")
    P("<b>How it works.</b> For a horizon of T steps there is one yes/no variable per (object, time, "
      "support). Constraints: every object has exactly one support at every time; at most one object on each "
      "object; nothing on a prism; zones and slots respect their capacity; at most one object changes its "
      "support per step; an object that changes must have nothing on it; nothing is put on the open table; at "
      "time T every goal object is in a stack standing in its zone. The solver minimises the number of "
      "changes. The horizon comes from the constructive plan, which is always long enough when no extra rules "
      "are active.")
    CODE('''
from approaches.cpsat import CpSatStrategy          # pip install ortools
run(CpSatStrategy(), scene)                                          # provably shortest plan
run(CpSatStrategy(avoid=["tmp_1"]), scene)                           # do not use this slot
run(CpSatStrategy(forbid_on=[("blue_cylinder_1", "red_cube_1")]), scene)   # never place that on this
run(CpSatStrategy(move_once={"glass_cube"}), scene)                  # fragile: at most one move
strategy.last_stats      # {"horizon": 7, "booleans": 204, "status": "OPTIMAL", ...}
''')
    rows = [["Objects", "Valid", "Same length as A*", "Median time", "95th percentile", "Yes/no variables"]]
    for n in ("4", "5", "6", "8"):
        r = cp[n]
        rows.append([n, f"{r['valid']}/{r['scenes']}", f"{r['same_length_as_astar']}/{r['scenes']}",
                     ms(r["p50_ms"]), ms(r["p95_ms"]), f"{r['mean_booleans']:.0f}"])
    TABLE(rows, [20, 24, 34, 28, 30, 34],
          "Table 6. Constraint solving without extra rules [measured, 40 scenes per size, one solver thread].")
    P(f"<b>Reading the result.</b> It is exact and agrees with A* on plan length in every scene, and it is "
      f"two to three orders of magnitude slower than the graph planner. The very first call in a process took "
      f"{cp['first_call_s']:.1f} s here to load the solver library, so warm it up at server start. "
      "If a rule makes the task impossible the solver says so (status INFEASIBLE) instead of timing out. "
      "<b>Use it when</b> the task gains rules that the hand-written planner does not know: forbidden "
      "stackings, fragile objects, reserved slots, limits on stack height. <b>Limits:</b> the model assumes an "
      "empty gripper at the start; with extra rules the horizon is the constructive length plus three steps, "
      "which may be too short for unusual rules.")

    # ------------------------------------------------------------------ 6
    H1("6. Calibrated confidence threshold: when to trust a detection")
    P("<b>Idea.</b> The server plans only for objects whose identity confidence is above a threshold and asks "
      "for another view otherwise. That threshold has been a hand-set 0.8. Too low, and objects are sorted "
      "into the wrong zone; too high, and the robot keeps looking when it does not need to. The threshold can "
      "be chosen from data so that the share of wrong labels among accepted detections stays below a target.")
    FIGURE("calibration.png", 160,
           "Figure 7. Choosing the threshold from a log of 1000 detections (synthetic, over-confident "
           "detector). The threshold is the lowest one whose upper bound is under the target.")
    P("<b>How it works.</b> Collect a log of detections with their confidence and whether the label was "
      "right. For each candidate threshold, count the errors among detections at or above it and compute an "
      "exact upper confidence bound on the true error rate (Clopper-Pearson). Take the lowest threshold whose "
      "bound is under the target. This is the same family of method that KnowNo uses with conformal "
      "prediction to decide when a robot should ask for help.")
    CODE('''
from approaches.calibration import calibrate
log = [(0.93, True), (0.71, False), ...]          # (confidence, label was correct) from the laptop
result = calibrate(log, alpha=0.05, delta=0.1)    # at most 5% wrong, with 90% confidence
result["threshold"]                               # e.g. 0.84; None if the log is too small

scene_message["min_confidence"] = result["threshold"]     # the server uses it per scene
''')
    rows = [["Detections in the log", "Threshold found (of 50 logs)", "Mean threshold", "Detections accepted",
             "Error on 50,000 new detections", "Logs that met the 5% target"]]
    for r in cal["rows"]:
        rows.append([f"{r['log_size']:,}", f"{r['threshold_found']}/50",
                     f"{r['mean_threshold']:.2f}" if r["mean_threshold"] else "-",
                     pc(r["mean_accepted"]) if r["mean_accepted"] else "-",
                     pc(r["mean_test_error"], 1) if r["mean_test_error"] is not None else "-",
                     f"{r['runs_within_target']}/{r['threshold_found']}"])
    TABLE(rows, [28, 30, 24, 28, 34, 30],
          "Table 7. Calibration on a synthetic detector [measured]. For comparison, the hand-set threshold 0.8 "
          f"accepts {pc(cal['fixed_0.8']['accepted'])} of detections with {pc(cal['fixed_0.8']['error'], 1)} "
          "wrong on the same test data.")
    P("<b>Reading the result.</b> With 200 logged detections a safe threshold could be found in fewer than "
      "half of the trials; with 1000 it always could. In a few trials the chosen threshold missed the target "
      "on new data. That is expected: the guarantee is stated with 90% confidence, and picking the lowest "
      "passing threshold after looking at all of them weakens it slightly. <b>Use it when</b> a labelled log "
      "from the real cameras exists. <b>Limits:</b> the log here is synthetic; the guarantee only holds while "
      "lighting, objects and camera stay like the log, so re-calibrate when they change.")

    # ------------------------------------------------------------------ combine
    H1("How they combine with what you already have")
    BL(["<b>Default path, unchanged:</b> graph planner, validator, validate-then-replan.",
        "<b>Add first:</b> the calibrated threshold (6), because it needs only a log and changes one number; "
        "and the partial-order graph (1), because it is computed from the plan you already have in a few "
        "microseconds and gives the executor freedom.",
        "<b>Add when failure data exists:</b> failure-aware planning (2). It is the only strategy here that "
        "gets better as the robot runs.",
        "<b>Add for uncertain objects on a slow link:</b> contingent plans (3), on top of the uncertainty "
        "gate from the first guide.",
        "<b>Add if plan length matters:</b> anytime planning (4).",
        "<b>Keep in reserve:</b> constraint solving (5), for the day the task gets rules the planner does "
        "not have."])
    CODE('''
python -m approaches.bench_extra                           # reproduces every table in this document
python -m pytest tests/test_more_strategies.py -q          # 7 tests
python docs/make_more_strategies.py                        # rebuilds this PDF
STRATEGY=cost_aware uvicorn approaches.serve:app --port 8020     # also: anytime, cp_sat, partial_order
''')

    H1("Other strategies that exist but are not implemented here")
    TABLE([
        ["Strategy", "What it is", "Why not implemented", "When it would be worth it"],
        ["Answer Set Programming (clingo)", "Declarative logic program; multi-shot solving can reuse work between replans", "Covers the same ground as CP-SAT and PDDL", "If the rules are easier to state as logic"],
        ["Monte Carlo Tree Search", "Simulate many random futures and pick the best first move", "The state space is small enough for exact search", "With stochastic outcomes and a simulator of the robot"],
        ["Probabilistic planning (MDP, LAO*)", "Optimal policy under action failure, including what to do after each failure", "Failure-aware planning with retry covers the simple case", "If a failed action changes the state (objects knocked over)"],
        ["Macro-actions", "Store frequent move sequences as single actions", "Planning is already fast", "With a slow planner or much larger scenes"],
        ["Problem decomposition", "Split a scene into independent parts, plan and cache each", "Needs care with shared zones and slots; open research direction", "To raise case-base hit rates on larger scenes"],
        ["Temporal and multi-robot planning", "Actions with durations, several arms in parallel", "One arm, one action at a time", "If a second arm is added; the partial-order graph is the starting point"],
        ["Simulation check (digital twin)", "Test the plan in a physics simulator before execution", "Geometry is the local side's job", "If stability of stacks becomes a problem"],
        ["Teleo-reactive rules", "Ordered condition-action rules evaluated continuously, no plan at all", "Same shape as the generated policy and the behaviour tree in the first guide", "For the local executor"],
    ], [34, 54, 46, 40],
        "Table 8. Further strategies. The 'why' and 'when' columns are judgements, not measurements.")

    H1("Limitations")
    BL(["All measurements are on synthetic scenes on one laptop. Nothing was run on the robot.",
        "The failure probabilities in section 2 and the detection log in section 6 are assumed. The methods "
        "are real; the size of their benefit on your robot is unknown until real data is used.",
        "Travel savings in section 1 use random positions, not your table layout.",
        "Contingent plans, partial-order graphs and solver side constraints are available in Python but are "
        "not yet fields of the HTTP response; the laptop client would need a small extension to use them.",
        "The six strategies were tested separately. Combinations (for example failure-aware planning inside "
        "a contingent tree) were not tested."])


def main():
    build()
    doc = SimpleDocTemplate(str(OUT / "more_strategies.pdf"), pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=17 * mm, bottomMargin=18 * mm,
                            title="Further planning strategies for the sorting robot", author="vlm_pddl")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT / "more_strategies.pdf")


if __name__ == "__main__":
    main()
