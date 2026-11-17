"""Build docs/system_guide.pdf: how the planning server works now, how to use it, what each
file is for, the image-resolution option, and what else would make it faster or shorter.

    python docs/make_system_guide.py

Numbers are read from approaches/results/*.json:
    live_planner.json        planner timing on the live path   (scratch measurement, see section 4)
    speculative.json         streamed moves with gates          (python -m approaches.speculative)
    resolution_levels.json   bytes and image tokens per level  (computed from images/*.jpg, no model)
    memory_session.json      example session with the experience database (stand-in model)
and charts from docs/example_run/report/ (python run_report.py on that session's logs).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "approaches" / "results"
OUT = ROOT / "docs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

LIVE = json.loads((RES / "live_planner.json").read_text())
SPEC = json.loads((RES / "speculative.json").read_text())
LEVELS = json.loads((RES / "resolution_levels.json").read_text())
SESSION = json.loads((RES / "memory_session.json").read_text())
EXAMPLE_RUN = ROOT / "docs" / "example_run"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S_BLUE, S_GREEN, S_ORANGE, S_GREY, S_YELLOW = "#e3eefb", "#e3f3ea", "#fdebe2", "#f1f1ee", "#fdf3d7"


# ================================================================ drawing helpers

def canvas(w, h, xmax=100, ymax=60):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, face="white", edge="#8a8984", size=6.8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor=face, edgecolor=edge, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, fontsize=size, color=INK, ha="center", va="center",
            fontweight="bold" if bold else "normal")


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


def fig_overview():
    fig, ax = canvas(7.2, 3.3, 100, 46)
    box(ax, 2, 30, 24, 12, "Robot arm + cameras", S_GREY, bold=True)
    box(ax, 2, 12, 24, 12, "Laptop (Dávid)\nperception, uncertainty,\nsimple cases itself", S_GREY)
    box(ax, 40, 6, 56, 36, "", S_BLUE, edge="#9fb7d6")
    ax.text(68, 39.5, "Server (Yulian)", fontsize=8, fontweight="bold", color=INK, ha="center")
    box(ax, 43, 26, 24, 9, "understand\nscene + task\n(VLM if needed)", "white")
    box(ax, 70, 26, 23, 9, "plan\n(PDDL planner)", "white")
    box(ax, 43, 10, 24, 9, "verify on the scene\n+ prove shortest", "white")
    box(ax, 70, 10, 23, 9, "replan when the\nscene changes", "white")
    arrow(ax, (67, 30.5), (70, 30.5))
    arrow(ax, (81.5, 26), (55, 19))
    arrow(ax, (14, 30), (14, 24))
    arrow(ax, (26, 21), (43, 30), "hard case: scene JSON\nor image + instruction", col=BLUE, dy=3.5)
    arrow(ax, (43, 13), (26, 15), "PICK / PLACE steps", col=AQUA, dy=-2.5)
    arrow(ax, (14, 12), (81.5, 10), col=ORANGE, rad=0.22)
    ax.text(47, 1.2, "scene changed / step failed: new scene to the server", fontsize=6.0, color=ORANGE,
            ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    save(fig, "sg_overview.png")


def fig_pipeline():
    fig, ax = canvas(7.2, 4.6, 100, 64)
    box(ax, 2, 52, 26, 8, "request\ninstruction + scene JSON", S_GREY, size=6.4)
    box(ax, 2, 40, 26, 8, "request\ninstruction + image(s)", S_GREY, size=6.4)
    box(ax, 34, 40, 20, 8, "resolution / crop\nimage_resolution.py", S_YELLOW, size=6.2)
    box(ax, 60, 40, 22, 8, "VLM: image -> scene JSON\n(perceive_prompt)", S_BLUE, size=6.2)
    box(ax, 34, 52, 48, 8, "normalize + check the scene  (openworld.normalize_scene)", "white", size=6.4)
    box(ax, 86, 46, 12, 14, "unknown\nsupport?\n-> ask for\na view", S_ORANGE, size=5.8)
    box(ax, 34, 28, 48, 8, "VLM: instruction -> goal + constraints  (goal_prompt; skipped if goal given)",
        S_BLUE, size=6.2)
    box(ax, 34, 16, 22, 8, "impossible?\ncertain_no_plan", S_ORANGE, size=6.2)
    box(ax, 60, 16, 22, 8, "PDDL problem\nmake_problem / pddl_task", "white", size=6.2)
    box(ax, 34, 3, 22, 9, "fast planner, then\nlower bound / optimal\n(plan_for, plan_quality)", S_GREEN,
        size=5.9)
    box(ax, 60, 3, 22, 9, "simulate the plan\non the problem\n(planning.simulate)", S_GREEN, size=5.9)
    box(ax, 86, 3, 12, 9, "response\nsteps +\nquality", S_GREY, size=6.0)
    arrow(ax, (28, 56), (34, 56))
    arrow(ax, (28, 44), (34, 44))
    arrow(ax, (54, 44), (60, 44))
    arrow(ax, (71, 48), (62, 52))
    arrow(ax, (82, 56), (86, 54))
    arrow(ax, (58, 52), (58, 36))
    arrow(ax, (45, 28), (45, 24))
    arrow(ax, (56, 20), (60, 20))
    arrow(ax, (71, 16), (52, 12))
    arrow(ax, (56, 7.5), (60, 7.5))
    arrow(ax, (82, 7.5), (86, 7.5))
    ax.text(1, 21, "no plan ->\nanswer at once\nwith the reason", fontsize=5.8, color=ORANGE, va="center")
    arrow(ax, (34, 20), (17, 20), col=ORANGE)
    save(fig, "sg_pipeline.png")


def fig_quality():
    fig, ax = canvas(7.2, 2.7, 100, 38)
    box(ax, 1, 14, 17, 10, "fast planner\n(about 0.1 s)", S_BLUE)
    box(ax, 23, 14, 21, 10, "moves == lower bound?\n(objects that must move)", S_YELLOW, size=6.3)
    box(ax, 50, 25, 22, 9, "proven shortest\nno search needed", S_GREEN)
    box(ax, 50, 3, 22, 9, "optimal search\nastar(lmcut), up to 5 s", S_BLUE)
    box(ax, 78, 15, 21, 9, "proven shortest", S_GREEN)
    box(ax, 78, 1, 21, 9, "keep fast plan\nsay max extra moves", S_ORANGE, size=6.3)
    arrow(ax, (18, 19), (23, 19))
    arrow(ax, (44, 21), (50, 29), "yes", col=AQUA, dy=1.8)
    arrow(ax, (44, 17), (50, 8), "no", col=ORANGE, dy=-1.8)
    arrow(ax, (72, 8.5), (78, 19), "finished", col=AQUA, dy=1.6, size=5.6)
    arrow(ax, (72, 6), (78, 5.5), "timeout", col=ORANGE, dy=-1.8, size=5.6)
    save(fig, "sg_quality.png")


def fig_resolution():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=200, gridspec_kw={"width_ratios": [1.05, 1]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 44)
    ax.axis("off")
    for i, (name, face) in enumerate((("low  448 px", S_GREEN), ("medium  768 px", S_YELLOW),
                                      ("high  1280 px", S_ORANGE))):
        y = 33 - i * 14
        box(ax, 1, y, 17, 8, name, face, size=6.4)
        box(ax, 23, y, 20, 8, "scene\nreliable?", "white", size=6.2)
        arrow(ax, (18, y + 4), (23, y + 4))
        arrow(ax, (43, y + 4), (52, y + 4), "yes" if i < 2 else "", col=AQUA, dy=1.6, size=5.6)
        if i < 2:
            arrow(ax, (33, y), (9.5, y - 6), "no", col=ORANGE, dy=-0.6, dx=4, size=5.6)
    box(ax, 52, 5, 17, 36, "plan with\nthis scene", S_GREEN, size=6.6)
    ax.text(35, 0.8, "no = unknown support, hidden object, or unreadable answer", fontsize=5.4, color=ORANGE,
            ha="center")
    ax = axes[1]
    names = ["low", "medium", "high", "full"]
    toks = [LEVELS[n]["mean_tokens"] for n in names]
    kb = [LEVELS[n]["mean_bytes"] / 1000 for n in names]
    bars = ax.bar(names, toks, color=BLUE, width=0.55)
    for b, t, k in zip(bars, toks, kb):
        ax.text(b.get_x() + b.get_width() / 2, t + 20, f"{t:.0f} tokens\n{k:.0f} kB", ha="center", fontsize=6.4,
                color=INK)
    ax.set_ylim(0, max(toks) * 1.3)
    ax.set_ylabel("image tokens the model reads", fontsize=7.5, color=MUTED)
    ax.set_title("your 12 photos, per image (computed, no model)", fontsize=7.5, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "sg_resolution.png", facecolor="white")
    plt.close(fig)


def fig_replan():
    fig, ax = canvas(7.2, 2.4, 100, 34)
    box(ax, 1, 12, 18, 10, "new scene\n(JSON or camera)", S_GREY)
    box(ax, 24, 12, 24, 10, "is it a state the old\nplan passes through?", S_YELLOW, size=6.4)
    box(ax, 54, 22, 22, 9, "kept: rest of the plan\n(still proven shortest)", S_GREEN, size=6.2)
    box(ax, 54, 2, 22, 9, "replanned: new plan\nsame goal, no VLM call", S_BLUE, size=6.2)
    box(ax, 82, 12, 17, 10, "steps to\nthe robot", S_GREY)
    arrow(ax, (19, 17), (24, 17))
    arrow(ax, (48, 19), (54, 26), "yes", col=AQUA, dy=1.6)
    arrow(ax, (48, 15), (54, 7), "no", col=ORANGE, dy=-1.6)
    arrow(ax, (76, 26), (82, 19))
    arrow(ax, (76, 7), (82, 15))
    save(fig, "sg_replan.png")


def fig_memory():
    fig, ax = canvas(7.2, 3.1, 100, 44)
    stages = [("perception\nimage -> scene", "key: model, prompt,\ntoken budget, image bytes", 2),
              ("goal\ninstruction -> goal", "key: model + whole goal\nprompt (no arrangement)", 36),
              ("planning\nproblem -> plan", "key: PDDL problem,\ndomain, planner settings", 70)]
    for name, key, x in stages:
        box(ax, x, 28, 26, 10, name, S_BLUE, size=6.6)
        ax.text(x + 13, 22.5, key, fontsize=5.8, color=MUTED, ha="center", va="center")
        arrow(ax, (x + 13, 19.5), (x + 13, 13.5), "", col=ORANGE)
        arrow(ax, (x + 17, 13.5), (x + 17, 19.5), "", col=AQUA)
    for a, b in ((28, 36), (62, 70)):
        arrow(ax, (a, 33), (b, 33))
    box(ax, 2, 2, 94, 11, "experience database (SQLite: logs/experience.sqlite)\n"
        "hit: use the stored answer, skip the model / planner        miss: run it, store the answer "
        "if the request ends ok", S_YELLOW, size=6.4)
    ax.text(8, 16.5, "look up", fontsize=5.8, color=ORANGE)
    ax.text(21, 16.5, "answer", fontsize=5.8, color=AQUA)
    ax.text(50, 42, "every cached plan is still replayed on the scene before it is returned", fontsize=6.4,
            color=INK, ha="center")
    save(fig, "sg_memory.png")


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
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.7, leading=9.7),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=7.7, leading=9.7),
    "cellc": ParagraphStyle("cellc", fontName="Courier", fontSize=7.3, leading=9.4),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7.9, leading=10.3,
                          textColor=colors.HexColor(MUTED), spaceAfter=8, spaceBefore=2),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=9, leftIndent=4,
                           backColor=colors.HexColor("#f4f4f1"), borderPadding=5, spaceAfter=10, spaceBefore=6),
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
    img = Image(str(name if isinstance(name, Path) else FIG / name))
    img.drawWidth, img.drawHeight = width_mm * mm, width_mm * mm * img.imageHeight / img.imageWidth
    story.append(KeepTogether([img, Paragraph(caption, S["cap"])]))


def TABLE(rows, widths, caption=None, code_col=None):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), S["cellb"] if i == 0 else S["cellc"] if j == code_col else S["cell"])
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
    canv.drawString(18 * mm, 11 * mm, "Planning server: how it works and how to use it")
    canv.drawRightString(192 * mm, 11 * mm, f"{doc.page}")
    canv.restoreState()


# ================================================================ content

def build():
    for f in (fig_overview, fig_pipeline, fig_quality, fig_resolution, fig_replan, fig_memory):
        f()
    by_n = {}
    for r in LIVE:
        by_n.setdefault(r["n"], []).append(r)
    proven_fast = sum(1 for r in LIVE if r["sat_moves"] is not None and r["sat_moves"] == r["lower_bound"])
    opt_timeouts = sum(1 for r in LIVE if r["opt_moves"] is None)
    sat_med = st.median(r["sat_s"] for r in LIVE)
    g_none, g_legal, g_opt = (SPEC[f"thinking=0 gate={g}"] for g in ("none", "legal", "optimal"))
    think = SPEC["thinking=400 gate=optimal"]
    lo, hi, full = LEVELS["low"], LEVELS["high"], LEVELS["full"]

    story.append(Paragraph("Planning server: how it works and how to use it", S["title"]))
    story.append(Paragraph("What the server does with a request, why its plans are the shortest, how to "
                           "choose the image resolution, what every file is for, and what to build next. "
                           "Written for Yulian, November 2026.", S["subtitle"]))
    BOX("<b>In short.</b> The laptop calls the server only when it cannot solve a situation itself. It sends "
        "what it sees (a compact scene JSON, or an image) and the task in words. The server turns that into "
        "a goal, builds a planning problem, finds a sequence of PICK and PLACE steps, checks the sequence on "
        "the scene, and proves that no shorter sequence exists. When the scene changes it keeps or replans "
        "without calling the model again. New in this version: plans carry a <i>quality</i> certificate, "
        "impossible goals are answered at once, the planner runs fast-first, every image request can "
        "choose its resolution (or let the server choose with <i>auto</i>), an experience database answers "
        "repeated questions without the model, and <i>/dashboard</i> shows graphs of where the time goes.")
    P("<b>What is measured and what is not.</b> Planner timings and plan lengths were measured on the "
      "development laptop (no GPU) on generated scenes. Image sizes and token counts were computed from your "
      "12 photographs without running a model. No vision-language model was run for this document, so the "
      "time a real model needs at each resolution, and whether its scene is still right, are not known yet; "
      "section 5.4 gives the script that measures them on the GPU machine.")

    # ------------------------------------------------------------------ 1
    H1("1. The big picture")
    FIGURE("sg_overview.png", 160, "Figure 1. Who does what. The laptop handles simple cases; the server is "
           "asked for the hard ones and returns steps the robot can execute deterministically.")
    TABLE([
        ["Part", "Where", "Does"],
        ["Robot arm, wrist camera, external camera", "lab", "moves objects, takes pictures"],
        ["Laptop (Dávid)", "lab", "detects objects, estimates uncertainty, solves simple cases, calls the "
                                   "server for hard ones, executes the returned steps one by one"],
        ["Server (Yulian)", "GPU machine, port 8010", "understands scene and task, plans, verifies, proves "
                                                       "the plan is shortest, replans when the scene changes"],
    ], [52, 36, 86])

    # ------------------------------------------------------------------ 2
    H1("2. What happens to one request")
    FIGURE("sg_pipeline.png", 168, "Figure 2. Inside the server (openworld.run). Yellow: new resolution step. "
           "Blue: model calls. Green: planning and checking. Orange: early answers.")
    BL(["<b>1. Scene.</b> With a scene JSON from the laptop, no model is needed for the scene. With an image, "
        "the image is first resized or cropped (section 5) and the VLM writes the scene JSON "
        "(objects, places, what stands on what).",
        "<b>2. Check the scene.</b> <font face='Courier'>normalize_scene</font> cleans ids and checks that the "
        "scene is physically possible. If an object's support is unknown, the answer is "
        "<font face='Courier'>need_observation</font>: the laptop should take another view.",
        "<b>3. Goal.</b> A second, text-only VLM call turns the instruction into goal facts and constraints "
        "(\"don't use the table\", \"keep X\", \"never on Y\", \"move X first\"). The model is shown which objects "
        "exist but not how they are arranged, so it cannot copy the current state as the goal. If the request "
        "already contains a <font face='Courier'>goal</font>, this call is skipped.",
        "<b>4. Impossible?</b> <font face='Courier'>certain_no_plan</font> checks, in microseconds, reasons "
        "that make the goal impossible (a zone with fewer slots than objects that must go there, something on "
        "a prism, too tall for a shelf). Before this version these were found only after all planners timed out.",
        "<b>5. Plan.</b> Python writes a PDDL problem (the domain file is fixed; constraints add a few lines), a "
        "planner finds the steps, and the plan quality is proven (section 4).",
        "<b>6. Verify.</b> The plan is executed step by step on the same problem. Only a plan that reaches the "
        "goal is returned. Every request is logged in <font face='Courier'>logs/requests/&lt;id&gt;/</font>."])
    H2("Replanning")
    FIGURE("sg_replan.png", 150, "Figure 3. /replan and /camera/replan reuse the goal already understood. "
           "No model call is needed for a JSON scene.")
    P("If the new scene is one the old plan passes through (the robot simply made progress), the rest of the "
      "old plan is kept. Anything else (a person moved something, an object appeared or vanished, a step "
      "failed) gives a new plan from the new scene, with the same goal and constraints.")

    # ------------------------------------------------------------------ 3
    H1("3. How to use it")
    H2("Start")
    CODE("""
cd vlm_pddl
python server.py --model qwen3-vl-8b                 # GPU machine (or: ./serve.sh start qwen3-vl-8b 4bit)
python server.py --model mock                        # no GPU: rule-based stand-in, for testing the API
open http://127.0.0.1:8010/          (web page)     http://127.0.0.1:8010/docs   (all endpoints)
""")
    H2("Requests")
    CODE("""
# 1. scene JSON from the laptop (main path, no image)
curl -s localhost:8010/plan -H 'Content-Type: application/json' -d '{
  "instruction": "put the red cube on the green cube, but do not use the table",
  "scene": {"objects": [{"id": "red_cube", "label": "red cube"}, {"id": "blue_cylinder", "label": "blue cylinder"},
                        {"id": "green_cube", "label": "green cube"}],
            "places": [{"id": "red_zone", "kind": "zone", "capacity": 2}],
            "support": {"red_cube": "table", "blue_cylinder": "red_cube", "green_cube": "table"}}}'

# 2. image instead of JSON, with the resolution chosen per request
curl -s localhost:8010/plan -H 'Content-Type: application/json' -d '{
  "instruction": "sort the objects by colour", "images_b64": ["<base64 jpeg>"],
  "resolution": "auto", "crop": [0.15, 0.25, 0.85, 1.0]}'

# 3. the same as a form upload
curl -s localhost:8010/plan/upload -F instruction='sort the objects by colour' -F files=@scene.jpg \\
     -F resolution=medium -F crop='[0.15, 0.25, 0.85, 1.0]'

# 4. the scene changed: send it with the request_id of the plan
curl -s localhost:8010/replan -H 'Content-Type: application/json' -d '{"request_id": "<id>", "scene": {...}}'

# 5. no model at all: scene + goal -> plan (to test the planner)
curl -s localhost:8010/solve -H 'Content-Type: application/json' -d '{"scene": {...},
     "goal": [{"object": "red_cube", "on": "green_cube"}]}'

# ready-made examples
scripts/try_requests.sh      scripts/try_replan.sh      python scripts/check_response.py
""")
    TABLE([
        ["Endpoint", "Send", "Get"],
        ["POST /plan", "instruction + scene, or instruction + images_b64; optional goal, constraints, "
                       "resolution, crop, planner, attempts", "plan, goal, quality, resolution, timings"],
        ["POST /plan/upload", "form: instruction, files or scene, resolution, crop", "same"],
        ["POST /perceive", "form: files, resolution, crop", "scene JSON only"],
        ["POST /replan", "request_id + new scene", "decision kept / replanned, plan, quality"],
        ["POST /solve", "scene + goal", "plan without any model"],
        ["POST /camera/plan, /camera/replan", "camera_id, instruction or request_id, resolution, crop",
         "as /plan, from the latest streamed frame"],
        ["GET /requests/{id}", "", "the logged request and response"],
    ], [42, 76, 56], "Table 1. Endpoints. Full list: the docstring at the top of server.py or /docs.")
    TABLE([
        ["Response field", "Meaning"],
        ["status", "ok | need_observation (take another view) | clarify (question in message) | no_plan "
                   "(reason in message) | error"],
        ["plan", "steps: pick-from-table / pick-from-item / pick-from-place, place-on-table / place-on-item / "
                 "place-in, observe; each with args"],
        ["quality", "moves, lower_bound, optimal, proof, max_extra_moves, moved_more_than_once (section 4)"],
        ["resolution", "requested, used, crop, tried: level, approx_tokens, image_bytes, seconds, "
                       "not_reliable_because (section 5)"],
        ["goal, constraints", "what the server understood from the instruction"],
        ["timings", "perceive_s, goal_s, plan_s, total_s"],
        ["request_id", "needed for /replan"],
    ], [34, 140], "Table 2. The fields to read in a response.")

    # ------------------------------------------------------------------ 4
    H1("4. Why the plan is the shortest one")
    P("A returned plan is always <i>valid</i>: it was executed on the scene before it was sent. Whether it is "
      "also the <i>shortest</i> is now proven in every response, in the field "
      "<font face='Courier'>quality</font>.")
    CODE("""
"quality": {"moves": 3, "lower_bound": 3, "optimal": true,
            "proof": "the plan moves only objects that must move, each once",
            "max_extra_moves": 0, "moved_more_than_once": []}
""")
    P("<b>The lower bound.</b> Count the objects that must be picked up at least once: those whose goal is not "
      "met yet, those standing where another object has to go, and everything stacked on top of those. No plan "
      "can have fewer moves. If the plan has exactly that many, it is proven shortest. A plan such as \"blue on "
      "red, then blue has to move again\" has more moves than the bound, so it can never carry this proof.")
    FIGURE("sg_quality.png", 160, "Figure 4. The order of planning since this version (openworld.plan_for).")
    rows = [["Objects", "Scenes", "Fast plan: median time", "Fast plan proven shortest",
             "Optimal search alone: timed out at 5 s"]]
    for n, rs in sorted(by_n.items()):
        rows.append([n, len(rs), f"{st.median(r['sat_s'] for r in rs):.2f} s",
                     f"{sum(1 for r in rs if r['sat_moves'] == r['lower_bound'])}/{len(rs)}",
                     f"{sum(1 for r in rs if r['opt_moves'] is None)}/{len(rs)}"])
    TABLE(rows, [20, 18, 40, 44, 52],
          f"Table 3. Colour sorting on generated scenes, measured on the laptop [measured]. In "
          f"{proven_fast}/{len(LIVE)} scenes the fast plan (median {sat_med:.2f} s) was already proven shortest. "
          f"The old order (optimal search first) hit its 5 s timeout in {opt_timeouts}/{len(LIVE)} scenes.")
    BL(["<b>When an object has to move twice.</b> Red stands in the blue zone and blue in the red zone, each "
        "zone holds one object. The fast plan cannot be proven by the bound (3 moves, bound 2), so the optimal "
        "search runs (Fast Downward <font face='Courier'>astar(lmcut())</font>, an admissible heuristic) and "
        "proves that 3 is the minimum. <font face='Courier'>moved_more_than_once</font> then lists blue, and the "
        "proof says why that is unavoidable.",
        "<b>After a scene change.</b> A kept plan keeps its proof: the rest of a shortest plan, from a state on "
        "that plan, is itself shortest. A new plan gets its own proof.",
        "<b>What \"shortest\" means.</b> Fewest pick-and-place moves for the scene as the server understood it. "
        "If the scene is wrong, the plan can be shortest for the wrong scene; the per-step check on the laptop "
        "and replanning catch that. Distance travelled by the arm is not part of the count (section 7).",
        "<b>Tasks other than sorting.</b> With orders and constraints the bound is reached less often, so the "
        "optimal search runs more often. Only colour sorting was measured."])

    # ------------------------------------------------------------------ 5
    H1("5. Choosing the image resolution")
    P("A vision-language model reads an image as tokens: Qwen-VL turns every 32 x 32 pixel block of the "
      "resized image into one token. Fewer pixels means fewer tokens to read, which is the part of perception "
      "that grows with the image. The model still has to write the same scene JSON, so the total time does not "
      "shrink in proportion; how much is saved has to be measured on the GPU (5.4).")
    FIGURE("sg_resolution.png", 165, "Figure 5. Left: the auto rule. Right: image tokens and JPEG size per image "
           "for your 12 photographs at each level [computed with image_resolution.prepare, no model].")
    rows = [["Level", "Longest side", "Token budget", "Tokens per photo (computed)", "JPEG size per photo"]]
    for name, side in (("low", "448 px"), ("medium", "768 px"), ("high", "1280 px"), ("full", "original")):
        v = LEVELS[name]
        rows.append([name, side, v["budget"], f"{v['mean_tokens']:.0f}", f"{v['mean_bytes'] / 1000:.0f} kB"])
    v = LEVELS["low + crop"]
    rows.append(["low + crop", "448 px", v["budget"], f"{v['mean_tokens']:.0f}", f"{v['mean_bytes'] / 1000:.0f} kB"])
    TABLE(rows, [26, 26, 26, 50, 46],
          f"Table 4. Levels. 'full' keeps the photograph as sent (on average {full['mean_width']:.0f} x "
          f"{full['mean_height']:.0f} px); the model's own budget still limits what it reads. Low reads about "
          f"{hi['mean_tokens'] / lo['mean_tokens']:.0f} times fewer image tokens than high.")
    H2("5.1 How to set it")
    BL(["<font face='Courier'>\"resolution\": \"low\" | \"medium\" | \"high\" | \"full\"</font>: a fixed level.",
        "<font face='Courier'>\"resolution\": 640</font>: any longest side in pixels (64 to 4096); the token "
        "budget is chosen to match.",
        "<font face='Courier'>\"resolution\": \"auto\"</font>: low first; medium only if the low result is "
        "unreliable; then high. Unreliable means: the scene JSON could not be read, an object's support is "
        "unknown, an object is reported hidden, or the instruction names an object the model did not see.",
        "<font face='Courier'>\"crop\": [x0, y0, x1, y1]</font>: keep only the work area (fractions of the "
        "image). With a fixed camera this is set once. The objects then fill more of the same token budget.",
        "Without these fields nothing changes: the server's start-up setting "
        "(<font face='Courier'>--max-visual-tokens</font>, default 1024) is used, as before."])
    H2("5.2 What the response says")
    CODE("""
"resolution": {"requested": "auto", "used": "medium", "crop": null,
  "tried": [{"resolution": "low", "approx_tokens": 117, "image_bytes": 18342, "status": "need_observation",
             "seconds": 4.1, "not_reliable_because": "an object's support is unknown"},
            {"resolution": "medium", "approx_tokens": 348, "image_bytes": 42655, "status": "ok", "seconds": 6.3}]}
""")
    P("(Example shape; the seconds are invented for illustration.) Auto costs one extra model call when the low "
      "level is not enough, so it pays off only if low is enough most of the time. The sweep below tells.")
    H2("5.3 Where it is done")
    P("On the server, after the image arrives. So the robot side needs no change, but the bytes on the network "
      "do not go down. If the laptop also sends smaller pictures, they do: "
      f"{lo['mean_bytes'] / 1000:.0f} kB at low against {full['mean_bytes'] / 1000:.0f} kB for the original photos.")
    H2("5.4 Measuring it (GPU machine)")
    CODE("""
python scripts/resolution_sweep.py --model qwen3-vl-8b
python scripts/resolution_sweep.py --model qwen3-vl-8b --levels low 384 512 medium high full auto --crop 0.15 0.25 0.85 1.0
""")
    P("For each photographed scene in <font face='Courier'>scenes/scenes.yaml</font>, each view and each level, "
      "it runs the whole pipeline and replays the plan on the <i>true</i> scene: <i>works</i> (reaches the goal), "
      "<i>shortest</i> (as many moves as the optimal ground-truth plan), and the share of objects whose support "
      "the model got right, with time and tokens. The result is a curve of resolution against plan "
      "correctness and time: the main figure for the semantic-communication article.")

    # ------------------------------------------------------------------ 6
    # ------------------------------------------------------------------ 6 database
    H1("6. Answers from the experience database")
    P("Many requests repeat something the server has already worked out: the same instruction about the "
      "same objects, the same photo sent again after a failed step, the same scene and goal after a retry. "
      "The experience database (<font face='Courier'>experience_db.py</font>, one SQLite file) keeps the "
      "answers of requests that ended <i>ok</i> and gives them back instead of asking the model or the "
      "planner again.")
    FIGURE("sg_memory.png", 165, "Figure 6. One lookup before each expensive stage. A hit skips the stage; a "
           "miss runs it and the answer is stored once the whole request has ended ok.")
    TABLE([
        ["Stored", "Looked up by", "A hit skips", "Why reusing it is safe"],
        ["scene answer", "model + perception prompt + token budget + the exact image bytes",
         "the perception model call", "same input to the same model gives the same answer (greedy decoding)"],
        ["goal answer", "model + the complete goal prompt", "the goal model call",
         "the goal prompt lists the objects and places but not how they are arranged, so the same "
         "instruction about the same objects gets the same goal even after objects were moved"],
        ["plan", "the exact PDDL problem + domain + planner settings", "planning",
         "same problem gives the same plan; it is still replayed on the scene, and keeps its proof"],
    ], [24, 52, 32, 66], "Table 6. What the database keeps. A changed prompt, model or setting changes the "
       "key, so an old answer is never used for a different question.")
    BL(["<b>Only good answers are kept.</b> A request that ends in a question, no plan or an error stores "
        "nothing. A retry with sampling (after an unusable answer) never reads from the database.",
        "<b>Similar instructions (optional).</b> With <font face='Courier'>--memory-examples 3</font> the three "
        "earlier instructions that share most words with the new one are put into the goal prompt with the "
        "answers that worked (retrieval-augmented). This changes what the model sees, so it is off by default "
        "and should be compared with and without on the GPU.",
        "<b>Exact images only.</b> A live camera rarely produces the same bytes twice, so scene hits come from "
        "repeated photos (retries, the web page, benchmarks), not from a moving camera."])
    m = SESSION
    P(f"<b>Example session [measured with the stand-in model].</b> Your 14 example requests, three times in "
      f"shuffled order, plus 8 image requests at different resolutions: {m['requests']} requests. Without "
      f"the database they would need {m['model_calls_without_database']} model calls; "
      f"{m['model_calls_made']} were made, so {m['model_calls_without_database'] - m['model_calls_made']} "
      f"({100 * (1 - m['model_calls_made'] / m['model_calls_without_database']):.0f}%) were answered from the "
      f"database. {m['statuses'].get('clarify', 0)} requests ended in a question (the stand-in model "
      f"understands only simple sentences); those are never stored, so they are asked again every time. "
      f"The time saved per hit is one model call; with the stand-in model that time is zero, so this session "
      f"shows counts, not seconds.")
    FIGURE(EXAMPLE_RUN / "report" / "memory.png", 160, "Figure 7. The same session: model calls and planner "
           "runs answered from the database so far, request by request.")
    H2("How to use it")
    CODE("""
python server.py --model qwen3-vl-8b                       # on by default: logs/experience.sqlite
python server.py --model qwen3-vl-8b --no-memory           # off (for clean benchmark runs)
python server.py --model qwen3-vl-8b --memory-examples 3   # plus similar instructions in the goal prompt
curl -s localhost:8010/memory                              # entries and hits per kind
curl -s -X POST 'localhost:8010/memory/clear?kind=goal'    # forget goals (or everything without kind)
""")
    P("Every response that used the database has a <font face='Courier'>memory</font> field, for example "
      "<font face='Courier'>{\"scene\": \"hit\", \"goal\": \"hit\", \"plan\": \"miss\"}</font>. Switch it off "
      "(<font face='Courier'>--no-memory</font>) when you measure model speed or accuracy, otherwise repeated "
      "benchmark inputs are answered from the database and look instant.")

    # ------------------------------------------------------------------ 7 graphs
    H1("7. Graphs: where the time goes")
    P("Every request is logged with its timings. <font face='Courier'>run_report.py</font> turns the logs "
      "into charts; the server shows the same page at <font face='Courier'>/dashboard</font>, rebuilt on every "
      "visit.")
    CODE("""
open http://<server>:8010/dashboard               # in a browser, while or after the server runs
open http://<server>:8010/dashboard?last=50       # only the 50 most recent requests
python run_report.py                              # the same as files: logs/report/index.html
python run_report.py --logs docs/example_run/requests --out /tmp/report   # any log folder
""")
    TABLE([
        ["Chart", "Shows", "Use it to see"],
        ["Time per request", "one bar per request: perception, goal, planning, rest", "which stage is slow"],
        ["Average time by kind", "image vs JSON, with and without database hits", "what JSON and the database save"],
        ["Answered from the database", "model calls and planner runs answered so far", "whether the database pays off"],
        ["Image resolution", "perception time and image tokens per level used", "which resolution to choose"],
        ["Resolution sweep", "plan works / shortest and seconds per level (after resolution_sweep.py)",
         "the trade-off for the article"],
        ["Outcomes", "ok (and how proven shortest), clarify, no plan, errors", "how often the server succeeds"],
    ], [38, 74, 62], "Table 7. The charts on /dashboard.")
    FIGURE(EXAMPLE_RUN / "report" / "by_type.png", 160, "Figure 8. Average time by kind of request in the "
           "example session [stand-in model: model stages are 0 s here; with a real model the perception and "
           "goal bars are the long ones].")

    H1("8. Starting moves before the model has finished")
    P("In this server the VLM never writes the moves: it writes the scene and the goal, and the planner turns "
      "them into moves about 0.1 s after the VLM finishes. Moving the arm earlier would mean acting on a "
      "half-read scene or instruction (a later \"don't use the table\" could invalidate a move already made), "
      "so the live server does not do it.")
    P("Where a model does write the moves (the comparison mode in <font face='Courier'>approaches/</font>), "
      "<font face='Courier'>approaches/speculative.py</font> releases each streamed move only if the planner "
      "proves it lies on a shortest plan from the current state. Replay of 200 model answers with injected "
      "mistakes [measured with scripted answers]:")
    TABLE([
        ["Check before a streamed move is executed", "Wasted moves executed", "Illegal moves sent",
         "First move, no thinking", "First move, thinking model"],
        ["none", g_none["extra_moves_total"], g_none["illegal_moves_sent"], f"{g_none['first_move_s']:.1f} s",
         f"{SPEC['thinking=400 gate=none']['first_move_s']:.1f} s"],
        ["legal moves only", g_legal["extra_moves_total"], g_legal["illegal_moves_sent"],
         f"{g_legal['first_move_s']:.1f} s", f"{SPEC['thinking=400 gate=legal']['first_move_s']:.1f} s"],
        ["on a shortest plan", g_opt["extra_moves_total"], g_opt["illegal_moves_sent"],
         f"{g_opt['first_move_s']:.1f} s", f"{think['first_move_s']:.1f} s"],
    ], [58, 30, 28, 30, 30],
        f"Table 5. With a thinking model the first move only comes after the thinking: {think['first_move_s']:.1f} s "
        f"of {think['wait_for_full_answer_s']:.1f} s. Writing moves is the fast part, so streaming saves little; "
        "the model's thinking is what to cut.")

    # ------------------------------------------------------------------ 7
    H1("9. What else can be built")
    P("Ordered by how much they are likely to help. <i>Live</i> = in the running server; <i>approaches</i> = "
      "implemented and tested in <font face='Courier'>approaches/</font> but not connected to the live server; "
      "<i>idea</i> = not implemented. Effects marked 'expected' are not measured.")
    H2("7.1 Faster")
    TABLE([
        ["What", "Why it helps", "State", "Effort"],
        ["Send JSON from the laptop instead of images", "No perception call at all; only the goal call remains",
         "live", "none on the server"],
        ["Resolution levels, auto, crop", "Fewer image tokens per perception call", "live (this version)",
         "measure with the sweep"],
        ["Use a non-thinking model, or cap its thinking", "Thinking tokens are most of the model time "
         "(table 5)", "idea: models.yaml has both variants", "small"],
        ["Experience database", "Repeated perception, goal and planning answered without the model",
         "live (this version)", "done; measure the hit rate on real sessions"],
        ["One model call for scene + goal on images", "One image read instead of two calls",
         "idea; risk: the model copies the scene as goal (seen before)", "medium"],
        ["Fast-first planning + proof", "Skips the 5 s optimal search when not needed", "live (this version)",
         "done"],
        ["Precheck of impossible goals", "Answer in microseconds instead of after timeouts", "live (this version)",
         "done"],
        ["Keep the planner process warm", "Fast Downward starts a new process per call (about 0.1 s)",
         "idea", "medium"],
        ["Graph planner for sorting-only tasks", "0.04 ms instead of about 0.15 s, same proof",
         "approaches (GraphStrategy, task server)", "small to connect"],
        ["Streamed moves with the shortest-plan gate", "Robot starts while a model writes moves",
         "approaches/speculative.py; only for model-written plans", "done; little gain with thinking models"],
    ], [52, 56, 44, 22], "Table 6. Speed.")
    H2("7.2 Fewer moves, less robot time")
    TABLE([
        ["What", "Why it helps", "State"],
        ["Proven shortest plans", "Fewest pick-and-place moves for the scene", "live (this version)"],
        ["Shortest gripper travel", "Among equally short plans, choose the one with the least arm movement "
         "(needs object positions)", "approaches/partial_order.py (order only); idea for full plans"],
        ["Failure-aware plans", "Avoid places where grasps or placements often fail (learned from feedback)",
         "approaches/cost_aware.py"],
        ["Look only when it matters", "Request another view only if the next move depends on the unknown "
         "object", "approaches/active_perception.py, contingent.py"],
        ["Keep the plan when the robot made progress", "No wasted replanning or re-ordering", "live"],
        ["Executor picks the order", "Steps that do not depend on each other can run in any order",
         "approaches/partial_order.py"],
    ], [50, 80, 44], "Table 7. Fewer moves and less time on the robot.")
    P("<b>Recommended next steps:</b> (1) run the resolution sweep on the GPU machine (it calls the pipeline directly and "
      "never uses the database); (2) compare a thinking and a non-thinking model at the chosen "
      "resolution; (3) watch /dashboard during real sessions to see the database hit rate; (4) try "
      "<font face='Courier'>--memory-examples 3</font> against 0 on the same instructions; (5) connect "
      "travel-aware ordering once the laptop sends positions.")

    # ------------------------------------------------------------------ 8
    H1("10. What every file is for")
    H2("8.1 The live server (repository root)")
    TABLE([
        ["File", "What it does"],
        ["server.py", "FastAPI app: endpoints, request logging, one-GPU lock, resolution per request"],
        ["openworld.py", "the pipeline: scene check, prompts, goal grounding, precheck, planning order, replan"],
        ["image_resolution.py", "NEW: resolution levels, crop, auto escalation, token estimate"],
        ["plan_quality.py", "NEW: lower bound on moves and the shortest-plan certificate"],
        ["experience_db.py", "NEW: experience database (scene, goal, plan answers; similar instructions)"],
        ["run_report.py", "NEW: graphs of the logged requests; also served at /dashboard"],
        ["pddl_task.py", "constraints (no table, keep, never on, first) as changes to the PDDL"],
        ["scenes.py", "PDDL problem generator (make_problem) and the benchmark scene config"],
        ["planning.py", "parse PDDL, run planners (solve), execute a plan on a problem (simulate)"],
        ["vlm.py", "loads a model from models.yaml; generate(); NEW: stream() for streamed answers"],
        ["domain/tabletop.pddl", "the fixed PDDL domain: what the robot can do"],
        ["models.yaml", "the models that can be loaded, with their settings"],
        ["web/index.html", "test page at http://…:8010/"],
        ["SERVER.md, serve.sh, setup.sh, docs/WSL_SETUP.md", "start the server, GPU machine setup"],
        ["logs/requests/, logs/experience.sqlite, logs/report/", "request logs, the database, the graphs "
                                                                  "(not in git)"],
    ], [52, 122], code_col=0)
    H2("8.2 Scripts and examples")
    TABLE([
        ["File", "What it does"],
        ["scripts/resolution_sweep.py", "NEW: resolution against plan correctness and time (GPU)"],
        ["scripts/try_requests.sh, try_replan.sh", "send the example requests in examples/"],
        ["scripts/check_response.py", "print a response and check it against the example's expectations"],
        ["scripts/camera_client.py", "streams a local camera to the server and asks for plans"],
        ["client.py", "small command-line client"],
        ["examples/requests/, examples/replan/", "ready requests: tasks, constraints, scene changes"],
    ], [64, 110], code_col=0)
    H2("8.3 Research and comparison (not needed to run the server)")
    TABLE([
        ["File", "What it does"],
        ["benchmark.py, evaluate.py, summarize.py, inspect_runs.py, rescore.py, prompts.py",
         "the VLM benchmark on photographed scenes (modes pddl, json, plan, split)"],
        ["approaches/", "interchangeable planning strategies, the task server (port 8020), experiments: "
                        "semcom.py, gnn_policy.py, speculative.py, model_bench.py; guide in approaches/README.md"],
        ["semplan/", "copy of the stand-alone planner core used by approaches/"],
        ["docs/*.pdf, docs/make_*.py", "guides; each PDF is rebuilt by its make_ script from the results files"],
        ["tests/", "offline tests: python -m pytest tests -q"],
    ], [64, 110], code_col=0)

    # ------------------------------------------------------------------ 9
    H1("11. Not measured yet, and known gaps")
    BL(["No vision-language model was run for this document. Model time per resolution, and whether the scene "
        "stays right at low resolution, are open until <font face='Courier'>resolution_sweep.py</font> runs "
        "on the GPU.",
        "The 'auto' rule decides from what the model says (unknown support, hidden objects, unreadable output). "
        "A model that is confidently wrong at low resolution is not caught by it; the sweep measures how often "
        "that happens.",
        "Planner timings are from the laptop; the GPU machine's CPU may differ. Running several planners at the "
        "same time slows each one down (one 5 s timeout during testing came from that).",
        "Only colour sorting was used to measure the planner order; tasks with orders and constraints were "
        "tested for correctness, not timed.",
        "<font face='Courier'>vlm.HFVLM.stream()</font> was written without a GPU and has never run.",
        "The database's benefit was counted with a stand-in model. How often real sessions repeat a request, "
        "and how much time a hit saves, shows on /dashboard once the server runs with a real model.",
        "One older test, <font face='Courier'>tests/test_scene_understanding.py::test_images_go_through_the_model"
        "_then_the_same_reader</font>, fails: it expects the task server to understand \"sort by colour\" without "
        "a model, but understanding now always goes through the model. It needs a model answer in the test or a "
        "decision about the rule path."])


def main():
    build()
    doc = SimpleDocTemplate(str(OUT / "system_guide.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=17 * mm, bottomMargin=18 * mm,
                            title="Planning server: how it works and how to use it", author="vlm_pddl")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT / "system_guide.pdf")


if __name__ == "__main__":
    main()
