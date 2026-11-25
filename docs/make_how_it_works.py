"""Build docs/how_it_works.pdf: how the planning server works, from the request to the checked plan.

    python scripts/measure_how_it_works.py      # writes results/how_it_works.json (about a minute, no GPU)
    python docs/make_how_it_works.py            # writes docs/how_it_works.pdf

Every number in the PDF is read from results/how_it_works.json. The scene drawing is made from
live_cases.four_towers() (the photo of 20 November).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import live_cases

OUT = ROOT / "docs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
M = json.loads((ROOT / "results" / "how_it_works.json").read_text())

BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#c8332f"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S_BLUE, S_GREEN, S_ORANGE, S_GREY, S_YELLOW = "#e3eefb", "#e3f3ea", "#fdebe2", "#f1f1ee", "#fdf3d7"
BLOCK = {"red": "#c8332f", "yellow": "#e2b100", "blue": "#2b6fc2", "green": "#2f8a3c"}


# ================================================================ drawing helpers
def canvas(w, h, xmax=100, ymax=60):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, face="white", edge="#8a8984", size=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, fontsize=size, color=INK, ha="center", va="center",
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, text="", col=MUTED, rad=0.0, dx=0.0, dy=1.6, size=6.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, linewidth=1.1, color=col,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, fontsize=size, color=col,
                ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.4))


def save(fig, name):
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG / name, facecolor=fig.get_facecolor())
    plt.close(fig)


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=0)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ================================================================ figures
def fig_overview():
    fig, ax = canvas(7.2, 3.1, 100, 44)
    box(ax, 1, 14, 20, 17, "Robot arm\n+ camera on the arm\n+ external camera", S_GREY)
    box(ax, 29, 11, 27, 23, "Laptop (robot side)\n\nsimple steps itself\nuncertainty: can I do\nthis alone?", S_YELLOW)
    box(ax, 66, 11, 32, 23, "Server (HEKATE, GPU)\n\nmodel: words -> goal\n(photo -> scene)\nplanner: goal -> moves\nchecks every plan", S_BLUE, bold=False)
    arrow(ax, (21, 25), (29, 25), "images", dy=2.2)
    arrow(ax, (29, 18), (21, 18), "steps", dy=-2.2)
    arrow(ax, (56, 27), (66, 27), "scene JSON + task", dy=2.4)
    arrow(ax, (66, 17), (56, 17), "checked plan", dy=-2.4)
    ax.text(82, 4.5, "called only for the hard cases", fontsize=7.2, color=MUTED, ha="center", style="italic")
    ax.text(42.5, 4.5, "decides when to ask", fontsize=7.2, color=MUTED, ha="center", style="italic")
    save(fig, "hw_overview.png")


def fig_pipeline():
    fig, ax = canvas(7.4, 4.4, 100, 62)
    steps = [
        (1, 46, "1  Read the scene\nobjects, what stands on\nwhat, zones, limits", S_GREY),
        (26, 46, "2  Photo? (optional)\nthe VLM turns the\nphoto into a scene", S_ORANGE),
        (51, 46, "3  Understand the task\nthe model turns the\nsentence into a goal", S_ORANGE),
        (76, 46, "4  Check the goal\nnames exist, possible,\nconstraints backed", S_GREY),
        (76, 22, "5  Choose a planner\nand search moves\n(section 5)", S_BLUE),
        (51, 22, "6  Check the plan\nrobot rules, scene limits,\ngoal reached", S_GREEN),
        (26, 22, "7  Prove its quality\nlower bound or\nexact search", S_GREEN),
        (1, 22, "8  Answer\nsteps + quality\n(or a question)", S_GREY),
    ]
    for x, y, t, c in steps:
        box(ax, x, y, 22, 13, t, c, size=7.0)
    for a, b in (((23, 52.5), (26, 52.5)), ((48, 52.5), (51, 52.5)), ((73, 52.5), (76, 52.5)),
                 ((87, 46), (87, 35)), ((76, 28.5), (73, 28.5)), ((51, 28.5), (48, 28.5)), ((26, 28.5), (23, 28.5))):
        arrow(ax, a, b)
    ax.text(50, 12.5, "orange = a model is used     blue = search     green = checks     grey = plain code",
            fontsize=7.2, color=MUTED, ha="center")
    ax.text(50, 6, "Without a photo step 2 is skipped. With a goal in the request step 3 is skipped (no model at all).",
            fontsize=7.2, color=MUTED, ha="center", style="italic")
    save(fig, "hw_pipeline.png")


def fig_routing():
    fig, ax = canvas(7.4, 4.6, 100, 66)
    box(ax, 36, 54, 28, 9, "goal from step 4", S_GREY, bold=True)
    rows = [
        ("ordered:\ntrue", "ordered\nsearch\n(one A*)", S_BLUE),
        ("formula", "general\nsearch\n(weighted A*)", S_BLUE),
        ("stack_in\n_zones", "zone towers\nexact A*,\nthen greedy", S_BLUE),
        ("height or\nsize limits", "general\nsearch +\nlower bound", S_BLUE),
        ("stack_by\n_color", "colour\nstacks\nexact A*", S_BLUE),
        ("plain on /\nin / table", "PDDL +\nFast\nDownward", S_BLUE),
    ]
    for i, (cond, planner, col) in enumerate(rows):
        x = 1 + i * 16.5
        box(ax, x, 33, 15.3, 10, cond, S_YELLOW, size=6.8)
        box(ax, x, 11, 15.3, 15, planner, col, size=6.8)
        arrow(ax, (50, 54), (x + 7.6, 43.2), rad=0.0)
        arrow(ax, (x + 7.6, 33), (x + 7.6, 26.2))
    ax.text(50, 47.5, "first match from the left", fontsize=7.0, color=MUTED, ha="center", style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.text(50, 4, "Every path ends in the same three checks (section 6). The plan only comes back if all pass.",
            fontsize=7.2, color=MUTED, ha="center")
    save(fig, "hw_routing.png")


def tower_state(ax, support, colour, x0, y0, title, w=5.2, h=3.0, gap=1.6):
    """Draw towers bottom to top from a support map."""
    above = {y: x for x, y in support.items() if y in support}
    bases = [x for x, y in support.items() if y not in support]
    x = x0
    for b in bases:
        cur, level = b, 0
        while cur is not None:
            ax.add_patch(Rectangle((x, y0 + level * h), w, h * 0.9, facecolor=BLOCK.get(colour[cur], "#999"),
                                   edgecolor="white", linewidth=0.6))
            cur, level = above.get(cur), level + 1
        lab = support[b] if support[b] != "table" else ""
        if lab:
            ax.text(x + w / 2, y0 - 2.2, lab.replace("_zone", ""), fontsize=5.2, color=MUTED, ha="center")
        x += w + gap
    ax.plot([x0 - 1, x], [y0 - 0.3, y0 - 0.3], color=MUTED, linewidth=0.8)
    ax.text(x0 - 1, y0 + 16, title, fontsize=6.5, color=INK, fontweight="bold")


def fig_scene():
    four = live_cases.four_towers()
    sup = {o["id"]: o["on"] for o in four["objects"]}
    col = {o["id"]: o["color"] for o in four["objects"]}
    fig, ax = canvas(7.2, 1.9, 100, 22)
    tower_state(ax, sup, col, 3, 3, "The scene of 20 November: 16 objects in 4 towers (bottom to top)")
    ax.text(40, 9, "In JSON: one line per object,\nfor example\n"
                   '{"id": "red_block_1", "color": "red",\n "shape": "block", "on": "yellow_block_1"}',
            fontsize=6.0, color=INK, va="center", family="monospace")
    ax.text(80, 9, "plus the zones:\n" + '{"id": "red_zone", "color": "red",\n "capacity": 4}', fontsize=6.0,
            color=INK, va="center", family="monospace")
    save(fig, "hw_scene.png")


def fig_search():
    fig, ax = canvas(7.4, 3.5, 100, 50)
    box(ax, 38, 38, 24, 9, "start arrangement\ng = 0 moves, h = 3", S_GREY, size=7.0)
    kids = [(4, "move blue_cube_1\nto the table\ng=1, h=3", S_GREY), (36, "move blue_cube_1\nonto blue_block_2\ng=1, h=2", S_GREEN),
            (68, "move green_cube_2\nto the green zone\ng=1, h=2", S_GREEN)]
    for x, t, c in kids:
        box(ax, x, 20, 28, 10, t, c, size=7.0)
        arrow(ax, (50, 38), (x + 14, 30.2))
    box(ax, 36, 2, 28, 9, "... until the goal holds\n(h = 0 and the goal test passes)", S_BLUE, size=7.0)
    arrow(ax, (50, 20), (50, 11.2))
    ax.text(84, 42, "g = moves so far\nh = estimate of moves still needed\nnext: the arrangement with\nthe smallest g + w*h",
            fontsize=7.0, color=MUTED, ha="center")
    save(fig, "hw_search.png")


def fig_replan():
    fig, ax = canvas(7.2, 2.8, 100, 40)
    box(ax, 2, 14, 20, 13, "/plan\nfirst plan", S_BLUE)
    box(ax, 30, 14, 20, 13, "robot does\nthe next step", S_GREY)
    box(ax, 58, 14, 20, 13, "/replan\nnew scene\n(+ failed steps)", S_BLUE)
    box(ax, 82, 24, 17, 10, "kept: rest of\nthe old plan", S_GREEN, size=6.0)
    box(ax, 82, 6, 17, 10, "replanned:\nnew plan", S_YELLOW, size=6.0)
    arrow(ax, (22, 20.5), (30, 20.5))
    arrow(ax, (50, 20.5), (58, 20.5))
    arrow(ax, (78, 22), (82, 28))
    arrow(ax, (78, 18), (82, 12))
    arrow(ax, (90, 6), (40, 14), rad=-0.35, text="next step", dy=-6)
    save(fig, "hw_replan.png")


def fig_scaling():
    rows = M["scaling"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.7), dpi=200)
    for ax in (a, b):
        style_axes(ax)
    for name, colr in (("sort_by_color", BLUE), ("stack_by_color", ORANGE), ("stack as formula", AQUA)):
        rs = sorted((r for r in rows if r["goal"] == name and r["status"] == "ok"), key=lambda r: r["n"])
        a.plot([r["n"] for r in rs], [max(r["wall_s"], 0.01) for r in rs], "o-", color=colr, ms=3.5, label=name)
        b.plot([r["n"] for r in rs], [r["moves"] for r in rs], "-", color=colr, lw=1, alpha=0.6)
        for r in rs:
            b.plot(r["n"], r["moves"], "o", ms=4, color=colr, mfc=colr if r["optimal"] else "white")
    a.set_yscale("log")
    a.set_xlabel("objects", fontsize=7, color=MUTED)
    a.set_ylabel("seconds (log)", fontsize=7, color=MUTED)
    a.legend(fontsize=6.3, frameon=False)
    a.set_title("Time for one request (no model)", fontsize=7.5, loc="left")
    b.set_xlabel("objects", fontsize=7, color=MUTED)
    b.set_ylabel("moves", fontsize=7, color=MUTED)
    b.set_title("Plan length (filled = proven shortest)", fontsize=7.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "hw_scaling.png")
    plt.close(fig)


def fig_greedy_exact():
    rows = M["greedy_vs_exact"]
    fig, ax = plt.subplots(figsize=(7.2, 2.3), dpi=200)
    style_axes(ax)
    xs = list(range(len(rows)))
    ax.bar([x - 0.2 for x in xs], [r["greedy_moves"] or 0 for r in rows], 0.38, color=GRID, label="greedy only (before)")
    ax.bar([x + 0.2 for x in xs], [r["exact_moves"] or 0 for r in rows], 0.38, color=AQUA, label="exact phase (now)")
    ax.plot([x - 0.45 for x in xs] + [xs[-1] + 0.45], [rows[0]["lower_bound"]] * (len(xs) + 1), "--", color=MUTED,
            lw=0.8, label="lower bound")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"scene {r['seed'] + 1}" for r in rows], fontsize=6.5)
    ax.set_ylabel("moves", fontsize=7, color=MUTED)
    ax.legend(fontsize=6.3, frameon=False, ncol=3, loc="upper right")
    ax.set_title("Towers in zones without the table, kept bases, height and size limits (16 objects)",
                 fontsize=7.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "hw_greedy_exact.png")
    plt.close(fig)


# ================================================================ PDF helpers
S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20, leading=24, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10.5, leading=14.5,
                               textColor=colors.HexColor(MUTED), spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, spaceBefore=14,
                         spaceAfter=6, textColor=colors.HexColor("#123a6b"), keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.4, leading=14, spaceBefore=7,
                         spaceAfter=2, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.3, leading=13, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.3, leading=12.8, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.7, leading=9.7),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=7.7, leading=9.7),
    "cellc": ParagraphStyle("cellc", fontName="Courier", fontSize=7.2, leading=9.3),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7.9, leading=10.3,
                          textColor=colors.HexColor(MUTED), spaceAfter=8, spaceBefore=2),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=9, leftIndent=4,
                           backColor=colors.HexColor("#f4f4f1"), borderPadding=5, spaceAfter=10, spaceBefore=6),
    "box": ParagraphStyle("box", fontName="Helvetica", fontSize=9.3, leading=13,
                          backColor=colors.HexColor("#eef3fa"), borderPadding=7, spaceAfter=12,
                          spaceBefore=8, leftIndent=4, rightIndent=4),
}
story = []
H1 = lambda t: story.append(Paragraph(t, S["h1"]))          # noqa: E731
H2 = lambda t: story.append(Paragraph(t, S["h2"]))          # noqa: E731
P = lambda t: story.append(Paragraph(t, S["body"]))         # noqa: E731
BOX = lambda t: story.append(Paragraph(t, S["box"]))        # noqa: E731
CODE = lambda t: story.append(Preformatted(t.strip("\n"), S["code"]))   # noqa: E731


def BL(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="-"))
    story.append(Spacer(1, 3))


def FIGURE(name, width_mm, caption):
    img = Image(str(FIG / name))
    img.drawWidth, img.drawHeight = width_mm * mm, width_mm * mm * img.imageHeight / img.imageWidth
    story.append(KeepTogether([img, Paragraph(caption, S["cap"])]))


def TABLE(rows, widths, caption=None, code_cols=()):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), S["cellb"] if i == 0 else S["cellc"] if j in code_cols else S["cell"])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5d4cf")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#8fa3bd"))]))
    story.append(t)
    story.append(Paragraph(caption, S["cap"]) if caption else Spacer(1, 7))


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Helvetica", 7.5)
    canv.setFillColor(colors.HexColor(MUTED))
    canv.drawString(18 * mm, 11 * mm, "How the planning server works")
    canv.drawRightString(192 * mm, 11 * mm, f"{doc.page}")
    canv.restoreState()


def yes(v):
    return "yes" if v else "no"


def secs(v):
    return "-" if v is None else "under 0.01 s" if v < 0.01 else f"{v:.2f} s" if v < 10 else f"{v:.0f} s"


# ================================================================ content
def build():
    for f in (fig_overview, fig_pipeline, fig_routing, fig_scene, fig_search, fig_replan, fig_scaling,
              fig_greedy_exact):
        f()
    ex, gt, wire, rp, ck = M["example"], {g["name"]: g for g in M["goal_types"]}, M["wire"], M["replanning"], M["checking"]
    ge = M["greedy_vs_exact"]
    sc = {(r["n"], r["goal"]): r for r in M["scaling"]}

    story.append(Paragraph("How the planning server works", S["title"]))
    story.append(Paragraph("From a scene and a sentence to a checked, shortest list of pick and place steps: what "
                           "the server does, which part uses a model, how a plan is searched and proven, and "
                           "how the robot talks to it. November 2026.", S["subtitle"]))
    BOX("<b>In short.</b> The robot's laptop sends the server a description of the table (which objects there "
        "are and what stands on what, as a small JSON) and the task in words. A model reads only the words and "
        "writes the <b>goal</b>: what must be true at the end. Plain code then <b>searches</b> for the moves "
        "that reach that goal, <b>checks</b> them against the robot's rules three times, and, where it can, "
        "<b>proves</b> that no shorter plan exists. When the robot moves something, or a step fails, the server "
        "is asked again and keeps or repairs the plan without the model.")
    P(f"<b>Where the numbers come from.</b> Every number below was measured by "
      f"<font face='Courier'>scripts/measure_how_it_works.py</font> on {escape(M['machine'])} "
      f"({M['measured'][:10]}), with the goal given or read by the rule-based stand-in model, so no "
      "vision-language model was involved. They are planner and server times. Model times are measured on "
      "the GPU machine with the live benchmark (section 10).")

    # ------------------------------------------------------------------ 1
    H1("1. The big picture")
    FIGURE("hw_overview.png", 160, "Figure 1. Who does what. The robot side does the simple steps itself and asks "
           "the server only when a situation is too hard (stacked objects, a new task, a failure).")
    TABLE([["Part", "Runs on", "Job"],
           ["Robot arm and cameras", "robot", "grasp and place objects, take pictures"],
           ["Laptop", "next to the robot", "simple steps, uncertainty: can I do this alone? builds the scene JSON, "
                                          "executes the steps it gets back"],
           ["Model (VLM)", "server GPU", "turns the sentence into a goal; optionally turns a photo into a scene"],
           ["Planner and checks", "server CPU", "finds the moves, checks them, proves how good they are"]],
          [36, 30, 108])
    P("The important split: <b>the model understands, the code plans</b>. A small model is good at reading a "
      "sentence like \"build one tower per colour in the zones\" in any wording or language, but bad at writing "
      "thirty correct steps in a row. Search code is the opposite: it never invents a step, and it can prove "
      "its plan is the shortest. So the model writes only the goal, and the code does the rest.")

    # ------------------------------------------------------------------ 2
    H1("2. What the server receives")
    FIGURE("hw_scene.png", 165, "Figure 2. The example used throughout: the photo of 20 November as a scene.")
    P(f"A request is one JSON with the <b>scene</b> and the <b>task</b>. For the scene above it is "
      f"{ex['request_bytes']:,} bytes; the photos of the project are about {ex['photo_bytes_median'] / 1000:.0f} kB "
      f"each (median of {ex['photos']}). Sending the scene instead of the photo is the <i>semantic "
      "communication</i> part of the project: the laptop sends what matters, not pixels.")
    TABLE([["Field", "What it says"],
           ["objects", "id, colour, shape, and <b>on</b>: what the object stands on (table, another object, a zone). "
                       "Optional: height_mm, size_mm (footprint)"],
           ["zones (places)", "id, colour, capacity (slots), optional max_height_mm and max_top (objects per tower)"],
           ["rules", "optional, for the whole scene: max_height_mm, max_objects, no_larger_on_smaller"],
           ["instruction", "the task in words, any language: the model reads it"],
           ["goal", "instead of the instruction: the goal itself, then no model is called"],
           ["constraints", "no_table, keep, never_on, first, max_stack_height"],
           ["images_b64", "instead of the scene: photos; the VLM reads them first (slower, see section 3)"]],
          [32, 142])
    P("The server first cleans the scene: it gives repeated ids their own names, reads colour and shape also "
      "from ids like <font face='Courier'>red_block_2</font>, refuses impossible scenes (an object standing on "
      "itself, two objects on the same top) and answers <i>need_observation</i> when it does not know what an "
      "object stands on.")

    # ------------------------------------------------------------------ 3
    H1("3. One request, step by step")
    FIGURE("hw_pipeline.png", 166, "Figure 3. The path of a /plan request. Only steps 2 and 3 use a model.")
    BL(["<b>1 Read the scene.</b> As in section 2.",
        "<b>2 Photo (only if no scene was sent).</b> The VLM looks at the image and writes the scene JSON. The "
        "request can choose the image size (low, medium, high, full, or auto: start small, go bigger only if the "
        "scene is unclear) and a crop of the work area.",
        "<b>3 Understand the task.</b> The model gets the list of objects and zones and the sentence, but "
        "<i>not</i> what stands on what, so it cannot copy the current arrangement as the goal. It answers with "
        "a goal (section 4).",
        "<b>4 Check the goal.</b> Every name must exist; a constraint is kept only if the model quotes words that "
        "are really in the sentence; things that are impossible at first sight (more objects than a zone holds, "
        "a tower taller than its limit) are answered at once with the reason.",
        "<b>5 Plan.</b> The server picks the right planner for the goal and searches the moves (section 5).",
        "<b>6 Check.</b> Three independent checks (section 6).",
        "<b>7 Quality.</b> How many moves, the lower bound, and whether the plan is proven shortest.",
        "<b>8 Answer.</b> Status <i>ok</i> with the steps, or <i>clarify</i> (a question back), "
        "<i>need_observation</i>, <i>no_plan</i> (with the reason) or <i>error</i>."])
    P(f"For the example (sort the 16 objects by colour into the zones) the stand-in model answered "
      f"<font face='Courier'>{escape(str(ex['goal_answer']))}</font>; the server planned "
      f"{ex['moves']} moves with a lower bound of {ex['lower_bound']}, so the plan is proven shortest, in "
      f"{secs(ex['plan_s'])} of planning. The first steps:")
    CODE("\n".join(ex["first_steps"]) + f"\n... {ex['steps']} steps in total (every move is one pick and one place)")

    # ------------------------------------------------------------------ 4
    H1("4. What the model writes: the goal")
    P("The model never writes moves. It writes what must be true at the end, in a small fixed vocabulary. "
      "Frequent tasks have short forms (fast, and proven shortest); anything else can be written as a formula.")
    TABLE([["Goal entry", "Means", "Example sentence"],
           ['{"object": "a", "on": "b"}', "a stands directly on b", "put the red cube on the blue block"],
           ['{"object": "a", "in": "red_zone"}', "a stands directly in a slot of the zone", "put a in the red zone"],
           ['{"object": "a", "on": "table"}', "a stands on the table", "take a off the tower"],
           ['{"sort_by_color": true}', "every object into the zone of its colour", "sort by colour"],
           ['{"stack_by_color": true}', "one stack per colour, anywhere", "every stack one colour"],
           ['{"stack_in_zones": true}', "one tower per colour, in its zone", "build the towers in the zones"],
           ['{"formula": F}', "anything else, as logic (below)", "no two reds on each other"],
           ['"ordered": true', "the entries are a sequence", "first ..., then ..."]],
          [50, 58, 66], code_cols=(0,))
    P("<b>Formulas</b> are built from <i>for every / there is / and / or / not / if-then</i> and the facts "
      "<i>on, on_table, in, clear, above, color, shape, same_color, same_shape, is</i>. Example: \"every cube "
      "into the red zone\" is <font face='Courier'>forall x object, where shape(x, cube): in(x, red_zone)</font>. "
      "The server expands the formula for the objects of the scene, checks every name, and repeats it in plain "
      "words in the answer, so a person can see what the model understood. If the model writes something the "
      "server cannot use, it is asked again (up to three times); if the sentence is unclear it should ask back "
      "(<i>clarify</i>).")
    P("<b>Constraints</b> come with the goal: <i>no_table</i>, <i>keep</i> (do not move), <i>never_on</i> "
      "(never put X on Y), <i>first</i> (move X first), <i>max_stack_height</i>. The model must quote the words "
      "of the sentence that say each constraint; a constraint without such words is dropped, so the model "
      "cannot invent restrictions.")

    # ------------------------------------------------------------------ 5
    H1("5. How a plan is made")
    H2("5.1 The idea: search")
    P("The planner sees the table as an <b>arrangement</b>: for every object, what it stands on. A <b>move</b> "
      "takes an object with nothing on top and puts it on the table, on another free object, or into a zone "
      "with a free slot, if the rules allow it. The <b>goal test</b> says whether an arrangement is finished. "
      "Planning is searching from the start arrangement through moves until the goal test passes.")
    FIGURE("hw_search.png", 158, "Figure 4. The search keeps a list of arrangements and always continues with the "
           "most promising one: few moves so far (g) plus few moves still needed (h, an estimate).")
    P("The estimate h is what makes the search fast and what makes the proof possible. The server mostly uses "
      "<b>objects that must still move</b>: an object not yet where the goal wants it, or standing on such an "
      "object, needs at least one more move. This estimate <b>never overestimates</b>. With weight w = 1 "
      "(plain A*) the first plan found is therefore the shortest one; with a larger w the search is greedier: "
      "much faster, but not guaranteed shortest.")
    H2("5.2 Which planner")
    FIGURE("hw_routing.png", 166, "Figure 5. The goal decides the planner. All of them produce the same kind of "
           "steps and go through the same checks.")
    TABLE([["Planner", "Used for", "How", "Shortest?"],
           ["PDDL + Fast Downward", "plain on / in / table goals", "the scene becomes a PDDL problem for a fixed "
            "domain; a fast planner, then the optimal one (5 s) if the fast plan is not provably shortest",
            "proven by the lower bound or by optimal A*"],
           ["Colour-stack search", "stack_by_color, up to 18 objects", "exact A* on stacks", "proven"],
           ["Zone-tower search", "stack_in_zones", "exact A* first (within a fixed budget), then greedy",
            "proven when the exact phase finishes"],
           ["General search", "formulas, and any goal with height or size limits", "weighted A* (greedy first, then "
            "smaller weights to improve), estimate = requirements still false", "proven only if moves = lower bound"],
           ["Ordered search", "ordered: true", "one A* over the whole sequence (section 5.4)", "proven when it finishes"]],
          [32, 38, 72, 32])
    P("Budgets are counted in arrangements looked at, not in seconds, and shrink with the size of the scene. "
      "So the same request always gets the same answer, also on a busy machine; a time guard of 30 s only stops "
      "a runaway search.")
    H2("5.3 Proof of shortest")
    P("A plan comes back with <i>quality</i>: <i>moves</i>, <i>lower_bound</i>, <i>must_move</i> (the objects "
      "the bound counts) and <i>optimal</i> with the reason. The bound is the number of objects that must move "
      "at least once, so no plan can be shorter. If the plan has exactly that many moves, it is proven "
      "shortest without any further search. Otherwise the proof comes from an exact search (A* with an "
      "estimate that never overestimates) that finished. If neither holds, the answer says "
      "<i>verified, not proven shortest</i>.")
    H2("5.4 Ordered goals")
    P("With <i>ordered: true</i> the state of the search is the arrangement plus how many goals are already "
      "reached. A move may not undo a reached goal and may not reach a later goal before its turn (a goal that "
      "is already true at the start may wait). Because the whole sequence is one search, the plan is the "
      "shortest in that order, and the answer lists after which move each goal was reached "
      f"(<i>order.achieved_at_move</i>). Example: {escape(gt['ordered (3 goals)']['words'])}: "
      f"{gt['ordered (3 goals)']['moves']} moves, proven: {yes(gt['ordered (3 goals)']['optimal'])}.")
    H2("5.5 Limits of the real robot")
    P("The search follows the same rules the robot needs: a larger object (size_mm) never goes on a smaller one, "
      "a tower never exceeds the height limit of its zone or of the scene (whole tower, base included), never "
      "more objects per tower than allowed, nothing on objects without a flat top, zone capacity, closed lids, "
      "and the constraints. These rules hold for every intermediate arrangement too, not only at the end. "
      "Limits that make the goal impossible (for example four 30 mm objects in a 90 mm zone) are found before "
      "searching and answered with the reason.")
    H2("5.6 Measured on the example scene")
    TABLE([["Goal", "Planner", "Moves", "Lower bound", "Proven", "Planning"]] +
          [[escape(g["name"]), escape(str(g["planner"])), g["moves"],
            "-" if g["lower_bound"] is None else g["lower_bound"], yes(g["optimal"]),
            secs(g["plan_s"])] for g in M["goal_types"]],
          [52, 28, 18, 24, 18, 22], caption="Table 1. The 16-object scene of figure 2, goal given (no model). "
          "Planning = time of the planner inside the server.")

    # ------------------------------------------------------------------ 6
    H1("6. Checking every plan")
    P("A plan is only sent back when it passes three independent checks:")
    BL(["<b>The robot's rules.</b> The steps are executed one by one on the PDDL model of the robot (a fixed "
        "domain file with the actions pick-from-table, pick-from-item, pick-from-place, place-on-table, "
        "place-on-item, place-in, observe, open-lid) together with the constraints.",
        "<b>The scene's limits.</b> Every move is replayed with the size, height, capacity and constraint rules "
        "of section 5.5.",
        "<b>The goal.</b> The goal (as a formula) is evaluated on the final arrangement."])
    tbl = ck["table"]
    P("The same checks are available as <font face='Courier'>/check</font>: send a plan made somewhere else "
      "(for example by the robot's own planner) and the server says <i>ok</i> or names the first step it "
      f"rejects and why. Example: the plan <font face='Courier'>{escape(' ; '.join(ck['table_plan']))}</font> "
      f"with <i>no_table</i> is rejected at step {tbl.get('step')}: <i>{escape(str(tbl.get('reason')))}</i>.")

    # ------------------------------------------------------------------ 7
    H1("7. When the scene changes")
    FIGURE("hw_replan.png", 162, "Figure 6. The loop between the robot and the server.")
    kept, ch, fl = rp["kept"], rp["changed"], rp["failed"]
    P(f"After a step the laptop sends the new scene to <font face='Courier'>/replan</font> with the request id. "
      f"The goal and constraints of the first request are reused, so <b>no model is called</b>. If the new "
      f"scene is one the old plan passes through, the rest of the old plan comes back (<i>kept</i>): in the "
      f"example, after the first move of the {rp['first_moves']}-move sort, {kept['moves_left']} moves were left, "
      f"answered in {secs(kept['seconds'])}. Anything else gives a new plan (<i>replanned</i>): when a person "
      f"put a cube on the table, the server found {ch['moves']} moves in {secs(ch['seconds'])}.")
    P("If the robot <b>could not do a step</b>, it reports it in <i>failed</i>. A failed place becomes "
      "<i>never_on</i> (never put this object there again), a failed pick becomes <i>keep</i> (leave this object "
      "where it is); these rules stay for all later replans. Example: the plan "
      f"<font face='Courier'>{escape(' '.join(fl['first_plan'][:2]))}</font> ... reported "
      f"<font face='Courier'>{escape(fl['reported'])}</font> as failed; the new plan is "
      f"<font face='Courier'>{escape(' '.join(fl['new_plan']))}</font>.")

    # ------------------------------------------------------------------ 8
    H1("8. Talking to the server")
    TABLE([["Endpoint", "Use"],
           ["POST /plan", "scene (or photos) + instruction (or goal): the main call"],
           ["POST /solve", "scene + goal: planner only, no model"],
           ["POST /replan", "new scene (+ failed steps) after a step"],
           ["POST /check", "check a plan made elsewhere"],
           ["POST /perceive", "photo to scene only (to test the VLM)"],
           ["POST /camera/plan, /camera/replan", "the same from a streamed camera"],
           ["GET /health, /dashboard, /requests/{id}", "status, graphs of past requests, the full logged answer"]],
          [62, 112], code_cols=(0,))
    P(f"<b>Small and fast on the wire.</b> With <i>compact: true</i> the answer has only the request id, the "
      f"status, the steps as short strings, the quality and the server time: {wire['answer_compact_bytes']:,} "
      f"bytes instead of {wire['answer_full_bytes']:,} for the example. Requests and answers can be gzipped: the "
      f"request then shrinks from {wire['request_bytes']:,} to {wire['request_gzip_bytes']:,} bytes and the "
      f"compact answer to {wire['answer_compact_gzip_bytes']:,} bytes. Every answer carries the header "
      "<i>X-Server-Ms</i>, so the robot can split its waiting time into server time and network time. The full "
      "answer is always logged on the server, so <i>/replan</i> works after a compact answer too.")
    cx = wire["compact_example"]
    lines = ["{", f'  "request_id": "{cx["request_id"]}", "status": "{cx["status"]}",', '  "plan": [']
    lines += [f'    "{st}",' for st in cx["plan"][:4]] + [f'    ... {len(cx["plan"])} steps'] + ['  ],']
    lines += [f'  "quality": {json.dumps(cx["quality"])}, "total_s": {cx.get("total_s")}', "}"]
    CODE("\n".join(lines))

    # ------------------------------------------------------------------ 9
    H1("9. How far it scales")
    FIGURE("hw_scaling.png", 165, "Figure 7. Generated scenes of 4 to 48 objects in towers, goal given. "
           "Up to about 16-18 objects the colour stacks are proven shortest; beyond, the general search returns "
           "a verified plan within seconds.")
    s16, s48 = sc.get((16, "stack_by_color")), sc.get((48, "stack_by_color"))
    if s16 and s48:
        P(f"One stack per colour: {s16['moves']} moves for 16 objects in {secs(s16['wall_s'])}, "
          f"{s48['moves']} moves for 48 objects in {secs(s48['wall_s'])}. Sorting into zones stays proven "
          "shortest at every size, because every object only has to move once.")
    FIGURE("hw_greedy_exact.png", 165, "Figure 8. Why the exact phase was added: on scenes like the robot "
           "team's farby test the greedy search alone needed several moves more.")
    P(f"On these {len(ge)} scenes the exact phase found plans between "
      f"{min(r['exact_moves'] for r in ge)} and {max(r['exact_moves'] for r in ge)} moves, all proven "
      f"shortest: {yes(all(r['exact_optimal'] for r in ge))}, all accepted by /check: "
      f"{yes(all(r['check'] == 'ok' for r in ge))}; the greedy search alone needed "
      f"{min(r['greedy_moves'] for r in ge)} to {max(r['greedy_moves'] for r in ge)}.")

    # ------------------------------------------------------------------ 10
    H1("10. How it is tested")
    BL(["<b>Offline tests</b> (<font face='Courier'>python -m pytest tests</font>): every endpoint with the "
        "rule-based stand-in model, plans compared with brute-force searches on small random scenes, and the "
        "rule explanations compared with the search rules.",
        "<b>Live benchmark</b> (<font face='Courier'>python live_bench.py</font> on HEKATE): real requests to the "
        "running server with the real model: sentences of every kind (also new tasks and other languages), scene "
        "sizes, photos at every resolution, photo against JSON. Each plan is replayed on the true scene with the "
        "benchmark's own rules and judged against a reference written by a person; the report has graphs. "
        "<font face='Courier'>scripts/live_compare.sh</font> runs it for several models.",
        "<b>The robot team's simulation</b> ran the server's plans for their scenarios and checked them with "
        "their own rules."])

    # ------------------------------------------------------------------ 11
    H1("11. Limits and what is not done yet")
    BL(["The planner works with what stands on what. Positions, rotations, pushing and \"in a row\" are not "
        "part of the model of the robot.",
        "The model only writes the goal. A mode where the model writes the steps itself (and the server only "
        "checks them) is planned for comparison, not built yet.",
        "The new goal types, formulas and photo understanding have not yet been measured with the real model on "
        "the GPU machine; that is what the live benchmark is for.",
        "General goals and scenes with tight limits are verified but not always proven shortest.",
        "Nothing here is a test on the real robot; the robot team's results come from their simulation."])

    # ------------------------------------------------------------------ appendix
    H1("Appendix: the files")
    TABLE([["File", "What is in it"],
           ["server.py", "the HTTP endpoints, compact answers, gzip, X-Server-Ms"],
           ["openworld.py", "the request pipeline: scene cleaning, prompts, goals, choice of planner, checks, replan"],
           ["goal_logic.py", "the goal formula language: check, expand, evaluate, plain words"],
           ["formula_plan.py", "the general, zone-tower and ordered searches; the move rules and their explanations"],
           ["colour_stacks.py", "exact one-stack-per-colour search; the zone-tower lower bound"],
           ["pddl_task.py, domain/tabletop.pddl", "the PDDL model of the robot and the constraints in it"],
           ["plan_quality.py", "lower bound and proof of shortest"],
           ["image_resolution.py", "image sizes and crop for photo requests"],
           ["live_bench.py, live_cases.py, live_score.py, live_report.py", "the live benchmark"],
           ["docs/api.html", "the API reference (also published as a web page)"]],
          [62, 112], code_cols=(0,))


def main():
    build()
    doc = SimpleDocTemplate(str(OUT / "how_it_works.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=17 * mm, bottomMargin=18 * mm, title="How the planning server works",
                            author="vlm_pddl")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT / "how_it_works.pdf")


if __name__ == "__main__":
    main()
