"""Build docs/hybrid_planning_guide.pdf.

    python docs/make_guide.py

Measured numbers are read from approaches/results/bench.json (python -m approaches.bench).
The scenario walk-throughs are produced by running the strategies while the PDF is built.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from approaches import run, scenarios as SC
from approaches.active_perception import ActivePerceptionStrategy
from approaches.adaptive import AdaptivePlanner
from approaches.htn import HtnStrategy
from approaches.llm import MockLLM
from approaches.llm_online import LLMModuloStrategy, LLMPlusPStrategy
from approaches.mock import goal_oracle, planner_oracle
from approaches.replanning import FullReplan, PrefixRepair, ValidateThenReplan
from approaches.symbolic import GraphStrategy, PddlStrategy
from semplan.scene import Move, parse_scene

B = json.loads((ROOT / "approaches" / "results" / "bench.json").read_text())
ST = B["strategies"]
OUT = ROOT / "docs"
BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def ms(x):
    if x is None:
        return "n/a"
    return f"{x:.0f} ms" if x >= 100 else f"{x:.1f} ms" if x >= 10 else f"{x:.2f} ms" if x >= 1 \
        else f"{x:.3f} ms"


def pc(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


def row(label, n=6):
    return ST[label][str(n)]


# ================================================================ figure

def fig_latency():
    labels = [l for l in ST if not l.startswith("llm_direct") and not l.startswith("llm_modulo")]
    labels.sort(key=lambda l: row(l)["p50_ms"])
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, l in enumerate(labels):
        r = row(l)
        ax.plot([r["p50_ms"], r["p95_ms"]], [i, i], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.plot([r["p50_ms"]], [i], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1)
        ax.text(r["p95_ms"] * 1.35, i, ms(r["p50_ms"]), va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(None, max(row(l)["p95_ms"] for l in labels) * 6)
    ax.set_xlabel("planning time per request at 6 objects, ms (log scale): dot = median, bar to 95th percentile",
                  fontsize=8, color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "guide_latency.png", facecolor=SURFACE)
    plt.close(fig)


# ================================================================ PDF helpers

S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=19, leading=23, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10.5, leading=14.5,
                               textColor=colors.HexColor(MUTED), spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, spaceBefore=14,
                         spaceAfter=6, textColor=colors.HexColor("#123a6b"), keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.6, leading=14, spaceBefore=9,
                         spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.3, leading=13, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.3, leading=12.8, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.5, leading=9.4),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=7.5, leading=9.4),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7.9, leading=10.3,
                          textColor=colors.HexColor(MUTED), spaceAfter=8, spaceBefore=2),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=9, leftIndent=4,
                           backColor=colors.HexColor("#f4f4f1"), borderPadding=5, spaceAfter=10,
                           spaceBefore=6),
    "ref": ParagraphStyle("ref", fontName="Helvetica", fontSize=8, leading=10.5, leftIndent=14,
                          firstLineIndent=-14, spaceAfter=2.5),
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
CAP = lambda t: story.append(Paragraph(t, S["cap"]))


def BL(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="-"))
    story.append(Spacer(1, 3))


def TABLE(rows, widths, caption=None):
    data = [[Paragraph(str(c), S["cellb"] if i == 0 else S["cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5d4cf")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#8fa3bd"))]))
    story.append(t)
    if caption:
        CAP(caption)
    else:
        story.append(Spacer(1, 7))


def A(arxiv_id):
    return f'<a href="https://arxiv.org/abs/{arxiv_id}" color="#1c5cab">arXiv:{arxiv_id}</a>'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(18 * mm, 11 * mm, "Hybrid planning architectures for low-latency robotic manipulation")
    canvas.drawRightString(192 * mm, 11 * mm, f"{doc.page}")
    canvas.restoreState()


def plan_text(moves):
    return ", ".join(f"{m.obj}->{m.dst}" for m in moves) or "-"


def out_line(o):
    extra = f" llm_calls={o.llm_calls}" if o.llm_calls else ""
    line = f"{o.strategy:<18} {o.status:<8} {o.ms:8.3f} ms{extra}\n    {plan_text(o.moves)}"
    for r in o.info_requests:
        line += f"\n    -> {r['action']}({r['target']})"
    if o.reason:
        line += f"\n    reason: {o.reason}"
    return line


# ================================================================ content

def build():
    g, fd, pyp, htn = row("graph"), row("pddl:fast-downward"), row("pddl:pyperplan"), row("htn")
    prog, heur, lrn, lrnf = row("llm_program"), row("llm_heuristic"), row("learned"), row("learned+symbolic")
    cb, ad, direct, modulo = row("case_based"), row("adaptive"), row("llm_direct"), row("llm_modulo")
    off = B["offline"]
    rep = {r["method"]: r for r in B["replanning"]}
    loop = B["closed_loop"]

    story.append(Paragraph("Hybrid planning architectures for low-latency robotic manipulation", S["title"]))
    story.append(Paragraph("How each approach works, how to use it in this repository, and which one to "
                           f"build on. Prepared for Yulian, {B['meta']['date']}.", S["subtitle"]))
    BOX("<b>Summary.</b> The published hybrid systems differ mainly in <i>where the language model sits</i>: "
        "inside the loop on every step, once per task as a translator, or only offline as a programmer. "
        "Only the last group, together with purely symbolic planners, runs without model inference at "
        "request time. For this robot the server already receives a symbolic scene, so the translation step "
        "that LLM+P and the VLM+PDDL systems exist for is not needed. "
        f"Measured here on the same scenes: the graph planner answers in {ms(g['p50_ms'])} (median, 6 objects), "
        f"a small HTN planner in {ms(htn['p50_ms'])}, PDDL with Fast Downward in {ms(fd['p50_ms'])}. "
        "Recommendation: a symbolic core (graph planner, validator, validate-then-replan) with three optional "
        "attachments, each of which keeps the model out of the request path: an instruction-to-goal "
        "translator, offline program or heuristic synthesis, and an uncertainty gate that decides when to "
        "look again. All of this is implemented in <font face='Courier'>approaches/</font> behind one interface.")
    P("<b>Evidence labels used in this guide.</b> "
      "<b>[S]</b> found in this session's literature search; bibliographic data and abstract-level claims were "
      "checked, full texts were not read. "
      "<b>[M]</b> cited from memory, not re-checked. "
      "<b>[measured]</b> from <font face='Courier'>python -m approaches.bench</font> on one machine "
      f"({B['meta']['system']}), synthetic scenes, simulated robot. "
      "<b>[mock]</b> run with scripted model answers: shows control flow and code overhead only, "
      "and says nothing about a real model's accuracy or speed. "
      "No VLM or LLM was run for this guide, and no published latency figure for the systems below was found "
      "in the search, so the guide contains no latency numbers for other people's systems.")

    # ------------------------------------------------------------------ Part 1
    H1("Part 1. Research landscape")
    TABLE([
        ["Pattern", "What the model does", "Model is called", "Examples", "Online inference"],
        ["1. Model plans, open loop", "Writes the whole plan or program for the task", "Once per task", "ProgPrompt, Code as Policies", "Yes"],
        ["2. Model plans, closed loop", "Chooses the next action from feedback text", "Every step", "SayCan, Inner Monologue, SayPlan", "Yes, repeatedly"],
        ["3. Model translates", "Turns language or images into a PDDL problem; a planner solves it", "Once per task (plus retries)", "LLM+P, ViLaIn, Image2PDDL, DELTA", "Yes"],
        ["4. Model grounds and monitors", "Answers predicate questions about images; checks preconditions and effects", "Every step", "ViPlan (grounder mode), DKPROMPT, VAP-TAMP", "Yes, repeatedly"],
        ["5. Model guides TAMP", "Proposes subgoals or constraints that a task-and-motion planner refines", "Per task, again on failure", "VLM-TAMP, OWL-TAMP", "Yes"],
        ["6. Model proposes, critics verify", "Candidate plans checked by sound verifiers, errors fed back", "1 to k times per task", "LLM-Modulo", "Yes"],
        ["7. Model programs offline", "Writes the domain model, a policy, a heuristic or search components once", "Before deployment only", "Guan et al., Silver et al., Thought of Search, Correa et al., VisualPredicator", "<b>No</b>"],
        ["8. Learned planner, no language", "Trained policy or heuristic over a graph of the state", "Forward pass per state", "GNN policies, WL-GOOSE", "Small model"],
        ["9. Experience retrieval", "Finds a similar past solution and adapts it", "Often an LLM adapts", "Memory Transfer Planning, CoRAL, classical case-based planning", "Depends"],
        ["10. Uncertainty-aware", "Decides when to gather information or ask", "Varies", "TAMPURA, KnowNo", "Depends"],
        ["11. No model", "Classical search, HTN, behaviour trees", "Never", "Fast Downward, GTPyhop, py_trees", "<b>No</b>"],
    ], [34, 50, 28, 42, 20],
        "Table 1. Hybrid planning patterns, ordered by how much the model is involved at run time.")
    P("Two observations organise everything that follows. First, the patterns with a model in the loop "
      "(1 to 6) were designed for inputs that are <i>not yet symbolic</i>: language instructions, raw images, "
      "open vocabularies, household-scale environments. Second, patterns 7 and 11 are the only ones whose "
      "request-time cost is independent of a model. This robot has a small closed domain and a laptop that "
      "already produces symbols, which is the setting where patterns 7 and 11 apply directly.")

    # ------------------------------------------------------------------ Part 2
    H1("Part 2. Existing papers and implementations")
    papers = [
        ["1", "Thought of Search (Katz et al., NeurIPS 2024)", A("2404.11833"), "not checked",
         "The LLM writes the successor and goal-test functions as code; ordinary search then solves every instance. Sound and complete if the code is right.", "24 Game, crosswords, BlocksWorld, PrOntoQA", "<b>High.</b> Same idea as offline synthesis here", "S"],
        ["2", "Generalized Planning in PDDL Domains with Pretrained LLMs (Silver et al., AAAI 2024)", A("2305.11014"), "github.com/tomsilver/llm-genplan",
         "GPT-4 summarises the domain, proposes a strategy and writes a Python program that solves all tasks of the domain; automated debugging feeds failures back.", "Seven PDDL domains", "<b>High.</b> Implemented as synthesize_policy", "S"],
        ["3", "Classical Planning with LLM-Generated Heuristics (Correa, Pereira, Seipp, 2025)", A("2503.18809"), "built on Pyperplan; repo not checked",
         "The LLM writes several domain-specific heuristics in Python; the best on training tasks is used in greedy best-first search.", "Classical planning benchmark domains", "<b>High.</b> Implemented as synthesize_heuristic", "S"],
        ["4", "LLM+P (Liu et al., 2023)", A("2304.11477"), "github.com/Cranial-XIX/llm-pddl",
         "The LLM translates a language problem into a PDDL problem for a given domain; Fast Downward solves it; the LLM translates the plan back.", "Seven planning domains incl. blocksworld, grippers", "Medium. Only the goal needs translating here", "S"],
        ["5", "LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks (Kambhampati et al., 2024)", A("2402.01817"), "n/a (position paper)",
         "LLM as candidate generator and knowledge source; external sound critics verify and send errors back.", "Conceptual, with case studies", "<b>High</b> as the pattern for any model fallback", "S"],
        ["6", "ViPlan (Merler et al., 2025)", A("2505.13180"), "open-source benchmark",
         "Compares VLM-as-grounder (fills PDDL predicates for a planner) with VLM-as-planner. Reports no consistent benefit from chain-of-thought.", "Visual Blocksworld; household simulator (iGibson)", "High as a benchmark design to copy", "S"],
        ["7", "Planetarium (Zuo et al., 2024)", A("2407.03321"), "github.com/BatsResearch/planetarium",
         "Benchmark for text-to-PDDL translation. Shows that parseable and solvable PDDL is often semantically wrong.", "Blocks World and Gripper, 132k pairs", "High: reason to keep models out of state translation", "S"],
        ["8", "DKPROMPT (Zhang et al., 2024)", A("2406.17659"), "project page dkprompt.github.io",
         "Uses the PDDL preconditions and effects of the current action as yes/no questions to a VLM; replans with the classical planner when a check fails.", "Household tasks, simulation", "Medium: the same checks are done locally here", "S"],
        ["9", "VAP-TAMP: Robot Planning and Situation Handling with Active Perception (2026)", A("2604.26988"), "not checked",
         "Scene graph as symbolic state; VLMs select viewpoints and monitor plans; separates perception errors, world changes and execution failures.", "Service tasks in simulation and on a mobile manipulator", "<b>High</b> for the failure taxonomy", "S"],
        ["10", "ViLaIn (Shirai et al., ICRA 2024)", A("2311.00967"), "project page",
         "Detector + VLM + LLM generate a PDDL problem from an image and an instruction; planner errors are used to re-prompt.", "Cooking, blocks world, Hanoi (ProDG dataset)", "Medium: this is what vlm_pddl already does", "S"],
        ["11", "Image2PDDL: Planning with VLMs and a Use Case in Robot-Assisted Teaching (Dang et al., 2025)", A("2501.17665"), "not checked",
         "GPT-4o converts an image of the initial state and a goal description into a PDDL problem in three prompting steps.", "Blocksworld, sliding-tile puzzles", "Medium", "S"],
        ["12", "DELTA (Liu et al., ICRA 2025)", A("2404.03275"), "project page delta-llm.github.io",
         "LLM turns a scene graph into PDDL domain and problem, prunes the graph, splits the goal into sub-goals solved one after another by a classical planner.", "Large household scene graphs", "Medium: decomposition matters for large scenes only", "S"],
        ["13", "SayPlan (Rana et al., CoRL 2023)", A("2307.06135"), "project page",
         "LLM searches a collapsed 3D scene graph, plans, and repairs the plan with feedback from a scene-graph simulator.", "Two buildings, up to 37 rooms", "Low: scale problem this robot does not have", "S"],
        ["14", "Leveraging Pre-trained LLMs to Construct World Models (Guan et al., NeurIPS 2023)", A("2305.14909"), "github.com/GuanSuns/LLMs-World-Models-for-Planning",
         "LLM writes the PDDL domain; humans correct it through language feedback; a classical planner then plans.", "Three domains, 40+ actions", "Medium: useful when the domain grows", "S"],
        ["15", "TAMPURA (Curtis et al., RSS 2024)", A("2403.10454"), "not checked",
         "Solves determinised symbolic problems to build a small MDP, then plans with uncertainty and risk awareness (LAO*).", "Simulation and two real robot tasks", "Medium: principled version of the uncertainty gate", "S"],
        ["16", "KnowNo: Robots That Ask For Help (Ren et al., CoRL 2023)", A("2307.01928"), "project page robot-help.github.io",
         "Conformal prediction over the LLM's next-step options; the robot asks when the prediction set has more than one option.", "Tabletop and mobile manipulation", "Medium: calibrated 'when to ask'", "S"],
        ["17", "Return to Tradition / WL-GOOSE (Chen et al., ICAPS 2024)", A("2403.16508"), "not checked",
         "Weisfeiler-Leman graph features with classical regressors as learned heuristics; far smaller and faster to train than GNN heuristics.", "International Planning Competition learning-track domains", "Medium: the right way to try 'GNN-like' learning", "S"],
        ["18", "VLM-TAMP (Yang et al., 2024)", A("2410.02193"), "project page",
         "VLM proposes intermediate subgoals; a TAMP solver refines each; the VLM is re-queried when refinement fails.", "Kitchen tasks, 30 to 50 actions, simulation", "Low: needs a full TAMP stack on the server", "S"],
        ["19", "OWL-TAMP (Kumar et al., RA-L)", A("2411.08253"), "project page",
         "VLM generates discrete and continuous constraints that a TAMP system must satisfy.", "Simulation and real robot", "Low for the same reason", "S"],
        ["20", "VisualPredicator (ICLR 2025) and Pix2Pred", A("2410.23156") + ", " + A("2501.00296"), "not checked",
         "VLM proposes predicates; the useful ones are selected and an abstract world model is learned for planning.", "Simulated robot domains; Spot robot", "Low now; relevant if predicates must be discovered", "S"],
        ["21", "Memory Transfer Planning (2025) and CoRAL (2026)", A("2509.24160") + ", " + A("2605.02600"), "not checked",
         "Retrieve successful earlier code or plans from memory and let an LLM adapt them to the new setting.", "RLBench, CALVIN, UR5; contact-rich manipulation", "Medium: LLM-side counterpart of the case base", "S"],
        ["22", "BETR-XP-LLM (2024) and BTGenBot-2 (2026)", A("2409.13356") + ", " + A("2602.01870"), "not checked",
         "Planner builds a behaviour tree from goals and skills; an LLM proposes missing preconditions after failures. BTGenBot-2 is a small model that emits behaviour trees.", "Robot manipulation", "Medium for the executor side", "S"],
        ["23", "Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning (Hao et al., 2025)", A("2510.03182"), "not checked",
         "Already in your folder. Not read for this guide; listed for completeness.", "", "Read next to ViPlan", "title only"],
        ["24", "SayCan, Inner Monologue, ProgPrompt, Code as Policies", A("2204.01691") + ", " + A("2207.05608") + ", " + A("2209.11302") + ", " + A("2209.07753"), "project pages",
         "The foundational model-in-the-loop systems; see Part 3.", "Kitchen and tabletop robots; VirtualHome", "Low: every step or every task needs a model call", "M"],
    ]
    TABLE([["#", "Paper", "Link", "Code", "Algorithm in one sentence", "Evaluated on", "Relevance here", "Ev."]] + papers,
          [6, 30, 25, 20, 44, 21, 21, 7],
          "Table 2. Papers and systems, ranked by relevance to this robot. 'Not checked' means the code "
          "availability was not verified in this session.")
    P("<b>Not found.</b> No paper titled 'Adaptive Task Planning with Dynamic Scene Graphs and Video Priors' "
      "turned up. The closest results were LookPlanGraph (" + A("2512.21243") + "), Scene Graph-Guided "
      "Proactive Replanning (" + A("2508.11286") + ") and ViReSkill (" + A("2509.24219") + "). "
      "If you have the link, it should be added to the table. "
      "<b>Naming.</b> 'ViPlan' is a benchmark comparing VLM grounding with VLM planning, not a PDDL-plus-"
      "behaviour-tree system; 'VAP-TAMP' is the short name of arXiv:2604.26988.")

    # ------------------------------------------------------------------ Part 3
    H1("Part 3. How the named systems work")
    TABLE([
        ["System", "Model does", "Symbolic part does", "Model on every request?", "Invalid plans caught by", "Replanning", "Where the time goes"],
        ["LLM+P", "Translates problem text to PDDL using one worked example; translates the plan back", "Fast Downward solves the PDDL problem", "Yes, twice per task", "Nothing checks the translation; only planner failure shows", "Not addressed", "Model calls; planner start-up"],
        ["Generalized planning (Silver et al.)", "Writes one Python program per domain, debugged against training tasks", "A validator checks the program's plans during synthesis", "<b>No.</b> Offline only", "Validator during synthesis; nothing at run time unless you add it", "Run the program again on the new state", "Program execution only"],
        ["ProgPrompt", "Completes a Python-like task program from a prompt listing actions, objects and examples", "Assertions in the program test preconditions at run time and trigger recovery actions", "Once per task", "Assertions", "Recovery branches written into the program", "One model call per task"],
        ["SayCan", "Scores every available skill for usefulness at every step", "Learned value functions score feasibility; the product picks the skill", "Yes, every step, every skill", "Affordance scores only", "Implicit: next step is chosen from the current state", "Many scoring calls per step"],
        ["Inner Monologue", "Chooses the next action; success detectors and scene descriptions are appended to its prompt", "None", "Yes, every step", "Success detectors after execution", "Built in: the model reacts to feedback text", "One call per step plus perception models"],
        ["ViLaIn", "Detector, VLM and LLM produce the objects, initial state and goal of a PDDL problem", "Planner solves; its error messages are used for re-prompting", "Yes, per task", "Parser and planner errors", "Regenerate the problem", "Three model stages plus retries"],
        ["ViPlan (grounder mode)", "Answers yes/no questions about predicates in the image", "Planner plans; action preconditions and effects are verified through the VLM", "Yes, every step", "Predicate checks", "Replan when a check fails", "Many VLM questions per step"],
        ["DKPROMPT", "Answers precondition and effect questions for the current action", "Classical planner replans from the updated state", "Yes, every action", "Failed precondition or effect check", "Classical replanning", "VLM questions per action"],
        ["VLM-TAMP", "Proposes subgoal sequences", "TAMP refines each subgoal into motions", "Per task and on failure", "TAMP failing to refine", "Re-query the VLM", "TAMP solving and VLM calls"],
        ["VAP-TAMP", "Chooses viewpoints, monitors execution", "Scene graph state; planner adapts", "During monitoring", "Discrepancy between expected and observed scene graph", "Diagnose cause, then observe, adapt or recover", "VLM monitoring"],
    ], [22, 34, 30, 21, 26, 22, 22],
        "Table 3. Internals of the systems named in the task. Source: abstracts and project pages [S], and "
        "memory for SayCan, Inner Monologue and ProgPrompt [M].")
    P("<b>Adaptation to this robot.</b> In every model-in-the-loop system the model does one of two jobs: it "
      "turns something unstructured into symbols, or it stands in for a planner. The laptop already does the "
      "first job, and a planner does the second better. What remains transferable is the structure around the "
      "model: the validator-in-the-loop of LLM-Modulo, the precondition and effect checks of DKPROMPT (done "
      "by the laptop from the <font face='Courier'>expect</font> field of each step), the failure taxonomy of "
      "VAP-TAMP, and above all the offline-synthesis idea of pattern 7.")
    P("<b>Should the server run a VLM when the scene is already structured?</b> As planner: no. As state "
      "translator: no, and Planetarium shows why the risk is real even when the output parses [S]. "
      "As goal interpreter for spoken or typed instructions: yes, optionally. "
      "As exception handler for an object the local perception cannot classify at all: possibly, as the last "
      "step. As heuristic provider or repair assistant: no benefit when planning takes "
      f"{ms(g['p50_ms'])}. Your own earlier result (plan mode 0/5 on the swap scene) points the same way.")

    # ------------------------------------------------------------------ Part 4
    H1("Part 4. The approaches in this repository: how each works and how to use it")
    P("All strategies share the scene model, the action <font face='Courier'>Move(object, destination)</font>, "
      "the goal representation and the validator in <font face='Courier'>semplan/</font>. "
      "<font face='Courier'>run(strategy, scene)</font> times the strategy and validates its answer.")
    CODE('''
from approaches import run
from semplan.scene import parse_scene

scene = parse_scene({
    "scene_version": 15,
    "objects": [{"id": "cube_1", "color": "red", "type": "cube"},
                {"id": "cylinder_1", "color": "blue", "type": "cylinder", "on": "cube_1"}],
    "zones": [{"id": "red_zone", "color": "red", "capacity": 1},
              {"id": "blue_zone", "color": "blue", "capacity": 1}],
    "free_slots": ["tmp_1"],
    # optional; without "goal" every object goes to the zone of its colour
    "goal": {"object": "cube_1", "destination": "red_zone"},
})
outcome = run(strategy, scene)     # .status .moves .ms .reason .llm_calls .info_requests
''')

    H2("4.1 Graph planner (GraphStrategy)")
    P("<b>How it works.</b> The scene is a forest of supported-by edges. The planner repeats two rules: if a "
      "clear object can go to a place it never has to leave (its zone floor, or a finished stack in its zone), "
      "move it; otherwise park the object whose removal uncovers something useful soonest. If it gets stuck, "
      "A* with an admissible heuristic (objects still to move) takes over. Unsolvable scenes are rejected by "
      "a counting check before any search.")
    CODE('''
from approaches.symbolic import GraphStrategy
run(GraphStrategy(), scene)                 # fast, near-optimal
run(GraphStrategy(optimal=True), scene)     # A*: shortest plan
''')
    P(f"<b>Measured.</b> {g['valid']}/{g['scenes']} valid at 6 objects, median {ms(g['p50_ms'])}, "
      f"95th percentile {ms(g['p95_ms'])}, {g['moves_vs_optimal']:.3f} times the optimal length. "
      "<b>Use when:</b> always, as the default.")

    H2("4.2 Scene graph to PDDL (PddlStrategy)")
    P("<b>How it works.</b> Code, not a model, writes the PDDL problem from the scene: every ground pose "
      "becomes a slot object, supports become <font face='Courier'>on</font> and "
      "<font face='Courier'>at</font> facts, goals become <font face='Courier'>in</font> facts. "
      "unified-planning parses it and pyperplan or Fast Downward solves it. The plan is mapped back to moves.")
    CODE('''
from approaches.symbolic import PddlStrategy, scene_to_pddl
text, slots = scene_to_pddl(scene)                 # inspect the generated problem
strategy = PddlStrategy("fast-downward")           # or "pyperplan", "fast-downward-opt"
run(strategy, scene); strategy.last_timing         # generate / parse / solve in ms
''')
    P(f"<b>Measured.</b> Fast Downward: median {ms(fd['p50_ms'])}, plans {fd['moves_vs_optimal']:.2f} times "
      f"optimal with the default configuration. pyperplan: median {ms(pyp['p50_ms'])}, "
      f"95th percentile {ms(pyp['p95_ms'])}. All plans valid. "
      "<b>Use when:</b> a new kind of goal or action is easier to state in PDDL than to program, "
      "and as an independent reference for tests.")

    H2("4.3 HTN planner (HtnStrategy)")
    P("<b>How it works.</b> Tasks are decomposed by hand-written methods: <i>sort_all</i> picks an object and "
      "calls <i>deliver</i>; <i>deliver</i> either moves it home, or first calls <i>get_out_of_the_way</i> "
      "for what is on top of it or what blocks its zone. Search is depth-first with backtracking over "
      "alternative decompositions. The planner knows exactly what the methods say and nothing else.")
    CODE('''
from approaches.htn import HtnStrategy, METHODS    # edit METHODS to change the domain knowledge
run(HtnStrategy(), scene)
''')
    P(f"<b>Measured.</b> {htn['valid']}/{htn['scenes']} valid ({pc(htn['valid'], htn['scenes'])}), median "
      f"{ms(htn['p50_ms'])}, 95th percentile {ms(htn['p95_ms'])}. The unsolved scenes are cases the three "
      "methods do not cover; it never returned an invalid plan. "
      "<b>Use when:</b> the team wants the strategy to be readable as a hierarchy, or as a baseline.")

    H2("4.4 Case-based planning (CaseBasedStrategy)")
    P("<b>How it works.</b> The scene is reduced to a canonical form in which ids, colour names and stack "
      "order are replaced by indices. Equal forms mean isomorphic problems, so a stored plan transfers by "
      "substituting the current ids. Every retrieved plan is validated again; on a miss the fallback strategy "
      "plans and the result is stored.")
    CODE('''
from approaches.symbolic import CaseBasedStrategy, PddlStrategy
cbp = CaseBasedStrategy(fallback=PddlStrategy("fast-downward"))   # memory in front of a slow planner
run(cbp, scene).source       # "miss->fast-downward" the first time, "case" afterwards
''')
    P(f"<b>Measured.</b> In front of the graph planner: median {ms(cb['p50_ms'])} against {ms(g['p50_ms'])} "
      f"without it, and {cb['case_stats']['hit']} hit in {cb['scenes']} random 6-object scenes. "
      "With random scenes of this size, memory costs more than it saves. "
      "<b>Use when:</b> the planner behind it is slow (PDDL, a model), or scenes repeat.")

    H2("4.5 Model writes the plan (LLMDirectStrategy) and validator-in-the-loop (LLMModuloStrategy)")
    P("<b>How it works.</b> Direct: the scene JSON and the rules go into one prompt, the answer is parsed as "
      "MOVE lines. Modulo: the same, but the validator's error message ('step 0: red_cube_1 is not clear') is "
      "appended and the model is asked again, up to k rounds. Only a plan that validates is returned.")
    CODE('''
from approaches.llm import RepoVLM, MockLLM
from approaches.llm_online import LLMDirectStrategy, LLMModuloStrategy
llm = RepoVLM("qwen3vl-8b", quant="4bit")          # GPU machine; a key from models.yaml
run(LLMModuloStrategy(llm, max_rounds=3), scene)   # .llm_calls tells how many rounds were needed
''')
    P(f"<b>Measured [mock].</b> With a scripted model that makes the 'picks a covered object' mistake in 30% "
      f"of first answers: direct {direct['valid']}/{direct['scenes']} valid, modulo "
      f"{modulo['valid']}/{modulo['scenes']} valid with {modulo['llm_calls_per_request']:.2f} calls per request. "
      "This demonstrates the mechanism only. With a real model, time per request is model generation time "
      "multiplied by the number of rounds. <b>Use when:</b> as the last fallback, never as the default.")

    H2("4.6 LLM+P for this robot (LLMPlusPStrategy)")
    P("<b>How it works.</b> The model translates a natural-language instruction into a goal mapping, nothing "
      "else. The mapping is checked against the scene (unknown object or zone names are rejected), then the "
      "symbolic planner plans. Without an instruction the model is not called.")
    CODE('''
from approaches.llm_online import LLMPlusPStrategy
s = LLMPlusPStrategy(llm, instruction="put the red cube in the green zone")
run(s, scene)              # 1 model call for the goal, 0 for the plan
''')
    P("<b>Use when:</b> a person gives instructions in language. This is the split mode you already found to "
      "work best in vlm_pddl, with the scene coming from the laptop instead of from the model.")

    H2("4.7 Offline synthesis: the model writes code once (llm_offline.py)")
    P("<b>How it works.</b> <i>Policy synthesis:</i> the model is shown a small read-only API of the scene and "
      "asked for <font face='Courier'>policy(v)</font> returning the next move. The policy is run on training "
      "scenes; the first failure, with the scene described in words, goes back into the prompt. "
      "<i>Heuristic synthesis:</i> the model writes several <font face='Courier'>h(v)</font> functions; the one "
      "that solves the training scenes with the fewest expansions is kept and used in greedy best-first "
      "search. Both are saved as plain Python in <font face='Courier'>approaches/generated/</font>.")
    CODE('''
from approaches.llm_offline import synthesize_policy, ProgramStrategy
code, log = synthesize_policy(llm, train_scenes, max_rounds=4, accept=0.9)   # once, offline
run(ProgramStrategy(), scene)          # loads generated/policy.py; no model call
''')
    last = off["policy_rounds"][-1]
    first = off["policy_rounds"][0]
    P(f"<b>Measured [mock model, real execution].</b> With three canned 'model answers' the feedback loop "
      f"took the policy from {first['solved']}/{first['of']} to {last['solved']}/{last['of']} training scenes. "
      f"That policy then solved {prog['valid']}/{prog['scenes']} unseen 6-object scenes "
      f"({pc(prog['valid'], prog['scenes'])}) in {ms(prog['p50_ms'])}; the rest were reported as no plan, "
      f"never as an invalid plan. The generated-heuristic search solved {heur['valid']}/{heur['scenes']} in "
      f"{ms(heur['p50_ms'])} (95th percentile {ms(heur['p95_ms'])}). "
      "How good a policy a real model writes is the open question and the experiment worth running. "
      "<b>Security:</b> generated code runs through exec() with a reduced namespace, which is not a sandbox; "
      "review the file before use. <b>Use when:</b> you want model knowledge with zero request-time inference.")

    H2("4.8 Learned policy with symbolic fallback (LearnedPolicyStrategy)")
    P("<b>How it works.</b> Scenes are solved optimally with A*; every state on those plans yields one "
      "training example (all legal moves, the optimal one marked). A ranker is trained to put the optimal "
      "move first. At run time the ranker is rolled out greedily, the result is validated, and the symbolic "
      "planner takes over if it is not a valid plan. The ranker here is linear over 12 hand-made relational "
      "features; a GNN would replace the feature function, everything else stays.")
    CODE('''
from approaches.learned_policy import make_dataset, train, LearnedPolicyStrategy
weights, accuracy = train(make_dataset(400))           # seconds, CPU
run(LearnedPolicyStrategy(weights), scene)             # with symbolic fallback
''')
    P(f"<b>Measured.</b> Trained on {off['learned']['train_scenes']} scenes "
      f"({off['learned']['train_states']} states) in {off['learned']['data_and_training_s']:.1f} s, top-1 "
      f"accuracy {off['learned']['top1_accuracy']:.2f}. Alone it solved {lrn['valid']}/{lrn['scenes']} "
      f"({pc(lrn['valid'], lrn['scenes'])}) at 6 objects in {ms(lrn['p50_ms'])}. With fallback: all valid, "
      f"fallback used {lrnf['fallback_used']} times, {ms(lrnf['p50_ms'])}. It is slower and less reliable "
      "than the planner it imitates. A GNN was not trained; this result is for the linear stand-in only.")

    H2("4.9 Deciding when to look again (ActivePerceptionStrategy)")
    P("<b>How it works.</b> For each unidentified object the laptop sends colour probabilities. The server "
      "plans once per hypothesis and compares the plans. Moves that all plans share from the start are safe "
      "now. If the plans differ on the very first move, the missing fact matters immediately and the server "
      "requests a view of the object that splits the plans most. This is determinisation with a plan-agreement "
      "test, a light alternative to belief-space planning as in TAMPURA.")
    CODE('''
from approaches.active_perception import decide
decide(scene, {"unknown_1": {"blue": 0.55, "green": 0.45}})
# {"action": "EXECUTE", "moves": [...agreed prefix...]}  or
# {"action": "REQUEST_NEW_VIEW", "target": "unknown_1"}
''')

    H2("4.10 Replanning (replanning.py) and behaviour tree (behavior_tree.py)")
    P("<b>How they work.</b> <i>FullReplan</i> plans from the observed scene. <i>ValidateThenReplan</i> first "
      "replays the remaining plan on the observed scene and keeps it if it still reaches the goal. "
      "<i>PrefixRepair</i> keeps the longest executable prefix and plans the rest. "
      "<i>Receding horizon</i> plans, executes one move, observes, and plans again. "
      "The <i>behaviour tree</i> re-checks 'goal reached', 'everything identified' and 'plan valid' on every "
      "tick, so a disturbance simply makes a condition fail and the tree falls through to 'replan'.")
    CODE('''
from approaches.replanning import ValidateThenReplan
r = ValidateThenReplan().replan(observed_scene, remaining_moves)
r.decision, r.moves, r.based_on_version        # "kept" | "replanned", plan, scene version

from approaches.behavior_tree import run_tree
run_tree(robot)                                # robot: semplan.sim.SimRobot or your own client
''')
    rows = [["Method", "Median time", "95th percentile", "Old plan kept", "Moves in new plan", "Old moves reused"]]
    for name in ("full_replan", "validate_then_replan", "prefix_repair"):
        r = rep[name]
        rows.append([name, ms(r["p50_ms"]), ms(r["p95_ms"]), f"{r['kept']}/{r['cases']}",
                     f"{r['mean_moves']:.2f}", f"{r['mean_reused']:.2f}"])
    TABLE(rows, [42, 24, 26, 26, 28, 28],
          "Table 4. Replanning after a random disturbance in mid-plan, same cases for all methods [measured].")
    bt, rh = loop["behavior_tree"], loop["receding_horizon"]
    P(f"<b>Measured.</b> Prefix repair gives longer plans than full replanning "
      f"({rep['prefix_repair']['mean_moves']:.2f} against {rep['full_replan']['mean_moves']:.2f} moves) at the "
      "same cost. In the simulated closed loop with 10% failed grasps the behaviour tree reached the goal in "
      f"{bt['goal']}/{bt['episodes']} episodes with {bt['planner_calls_per_episode']:.2f} planner calls per "
      f"episode; receding horizon reached {rh['goal']}/{rh['episodes']} with "
      f"{rh['planner_calls_per_episode']:.2f} calls. Receding horizon is the simplest correct loop and is "
      "affordable only because a planner call is cheap.")

    H2("4.11 Adaptive planner (AdaptivePlanner) and the server")
    P("<b>How it works.</b> A fixed order of stages: instruction to goal (only if an instruction came), "
      "uncertainty gate, case lookup, graph planner, optional second symbolic planner, optional model "
      "fallback. Each stage answers or passes on, and the response carries a trace of stages and times.")
    CODE('''
from approaches.adaptive import AdaptivePlanner
from approaches.llm_online import LLMModuloStrategy
planner = AdaptivePlanner(llm_fallback=LLMModuloStrategy(llm))
o = run(planner, scene); o.source      # e.g. "graph | case+graph(miss->greedy)"

# same HTTP contract for the robot, strategy chosen at start-up:
#   STRATEGY=adaptive uvicorn approaches.serve:app --port 8020
#   STRATEGY=htn      uvicorn approaches.serve:app --port 8020
''')
    P(f"<b>Measured.</b> {ad['valid']}/{ad['scenes']} valid, median {ms(ad['p50_ms'])}; in the tests the model "
      "fallback was never reached on solvable scenes.")

    # ------------------------------------------------------------------ Part 5
    H1("Part 5. Comparison of architectures A to O")
    img = Image(str(OUT / "guide_latency.png"))
    img.drawWidth, img.drawHeight = 160 * mm, 160 * mm * img.imageHeight / img.imageWidth
    story.append(KeepTogether([img, Paragraph(
        "Figure 1. Planning time per request for the strategies that run without a model [measured, "
        "6 objects]. Strategies using a mock model are left out because their time excludes inference.",
        S["cap"])]))

    def m(label):
        r = row(label)
        return f"{pc(r['valid'], r['scenes'])} valid; {ms(r['p50_ms'])} / {ms(r['p95_ms'])}"
    TABLE([
        ["ID", "Architecture", "Measured here (valid; median / p95 at 6 objects)", "Training", "GPU at run time", "Deterministic", "Replanning", "Reuses earlier results", "Verdict"],
        ["A", "Pure VLM planner", "Not run. Your earlier run: 0/5 on the swap scene", "None", "Yes", "No", "New model call", "No", "Reject"],
        ["B", "Pure LLM planner", f"[mock] {m('llm_direct')}; time excludes the model", "None", "Yes", "No", "New model call", "No", "Reject"],
        ["C", "Classical PDDL planner", m("pddl:fast-downward"), "None", "No", "Yes", "Full replan", "No", "Keep as reference"],
        ["D", "LLM+P", "Goal translation [mock] + planner as in H", "None", "Only with an instruction", "Planner part yes", "Planner only; goal stays", "Goal is reused", "Use for instructions"],
        ["E", "VLM + PDDL", "Not run here; this is vlm_pddl split mode", "None", "Yes", "No", "New perception + planner", "No", "Superseded by F when the laptop sends symbols"],
        ["F", "Scene graph + PDDL", m("pddl:fast-downward") + " (Fast Downward); " + m("pddl:pyperplan") + " (pyperplan)", "None", "No", "Yes", "Full replan", "No", "<b>Second choice</b>"],
        ["G", "Scene graph + HTN", m("htn"), "None", "No", "Yes", "Full replan", "No", "Viable; methods incomplete"],
        ["H", "Scene graph + A* / graph planner", m("graph") + " (graph); " + m("astar (optimal)") + " (A*)", "None", "No", "Yes", "Validate, then full replan", "Old plan kept if still valid", "<b>First choice</b>"],
        ["I", "GNN + symbolic planner", "Linear stand-in: " + m("learned") + " alone; " + m("learned+symbolic") + " with fallback. No GNN trained", "Supervised, from A* plans", "Optional", "Yes after training", "New rollout", "No", "No gain over H"],
        ["J", "Case-based + symbolic fallback", m("case_based"), "None (accumulates)", "No", "Yes", "Lookup, then fallback", "Yes", "Only in front of a slow planner"],
        ["K", "RAG + LLM + validator", f"Validator loop without retrieval [mock]: {m('llm_modulo')}", "None", "Yes", "No", "New model call", "Retrieved examples", "Last fallback only"],
        ["L", "LLM-generated planning program", "Canned program: " + m("llm_program") + "; heuristic search: " + m("llm_heuristic"), "None (offline synthesis)", "<b>No</b>", "Yes", "Run the program again", "The program is the reuse", "<b>Most interesting research direction</b>"],
        ["M", "VLM-guided TAMP", "Not run; needs geometric TAMP on the server", "None", "Yes", "No", "Re-query VLM", "No", "Out of scope"],
        ["N", "Behaviour tree + symbolic recovery", f"Closed loop: goal in {bt['goal']}/{bt['episodes']} simulated episodes", "None", "No", "Yes", "Built into the tree", "Plan kept while valid", "Use for the executor"],
        ["O", "Adaptive: gate, memory, graph planner, optional fallbacks", m("adaptive"), "None", "Only in the fallback", "Yes without fallback", "As H", "Yes", "<b>Recommended shell around H</b>"],
    ], [7, 27, 50, 18, 16, 15, 18, 17, 22],
        "Table 5. Architectures A to O. Columns without numbers are qualitative judgements. "
        "Rows marked 'not run' have no measurement in this guide.")
    P("<b>What cannot be concluded from this table.</b> Accuracy of real VLMs and LLMs on this task, the "
      "quality of a policy written by a real model, and anything about a trained GNN. "
      "<b>Experiments that would settle them:</b> run B, D, K and L with Qwen on the GPU machine using "
      "<font face='Courier'>python -m approaches.run_examples --model &lt;key&gt;</font> and the same scenes; "
      "report validity before and after the validator loop, calls per request and wall time per request.")

    # ------------------------------------------------------------------ Part 6
    H1("Part 6. Top five solutions and what was rejected")
    TABLE([
        ["Rank", "Solution", "Why", "Cost"],
        ["1", "Graph planner + validator + validate-then-replan (H)", "Fastest measured, near-optimal, deterministic, every plan checked against the current scene version", "Hand-written for this domain"],
        ["2", "Scene graph to PDDL with Fast Downward (F)", "General and sound; a new goal type is a domain edit; independent check of H", f"About {ms(fd['p50_ms'])} per call"],
        ["3", "Offline program or heuristic synthesis (L)", "Model knowledge without request-time inference; little studied on real robots; cheap to test", "Generated code must be reviewed and stays behind the validator"],
        ["4", "LLM+P reduced to goal translation (D)", "Gives language instructions without letting the model touch state or plan", "One model call per instruction"],
        ["5", "Uncertainty gate + behaviour-tree executor (4.9, N)", "Decides when another view is worth it; recovery without special cases", "Needs colour probabilities from the laptop"],
    ], [10, 52, 76, 42], "Table 6. The five solutions to build on.")
    BL(["<b>Rejected: pure VLM or LLM planning (A, B).</b> No guarantee, model time on every request, and your "
        "own measurements were poor.",
        "<b>Rejected: SayCan-style closed loop.</b> Needs learned value functions per skill and a model call "
        "per skill per step.",
        "<b>Rejected: VLM-guided TAMP (M).</b> The server is not supposed to do geometry.",
        "<b>Rejected for now: GNN planning (I).</b> The imitation target is a planner that is already faster "
        "than a forward pass; nothing to gain in speed, and reliability needs the same fallback.",
        "<b>Rejected: RAG with an LLM (K).</b> Retrieval does not remove generation time or the need for "
        "validation; exact symbolic retrieval already covers reuse.",
        "<b>Not adopted: learned strategy selection.</b> The cheap stages cost so little that a selector has "
        "nothing to save."])

    # ------------------------------------------------------------------ Part 7
    H1("Part 7. Best overall architecture and system design")
    CODE('''
LOCAL (robot + laptop)                            SERVER
----------------------                            ------------------------------------------------
cameras -> detection + uncertainty                approaches/serve.py   (FastAPI, STRATEGY=...)
        -> scene message (JSON, version n)  --->  semplan/service.py    session, version gate (409 if stale)
           objects, supports, zones, slots,         parse + consistency check (422 if impossible)
           confidences, colour probabilities        +-- AdaptivePlanner -------------------------+
                                                    | 0 instruction -> goal   (model, optional) |
deterministic executor                              | 1 uncertainty gate      (act or look)     |
   one step at a time, checks "expect"              | 2 case lookup           (canonical form)  |
        <--- steps: PICK / PLACE / REQUEST_NEW_VIEW | 3 graph planner         (A* inside)       |
             plan_id, based_on_version              | 4 PDDL planner          (optional)        |
                                                    | 5 model + validator     (optional, last)  |
step feedback (+ observation)               --->    +--------------------------------------------+
scene update (full or delta)                --->    validate on current scene -> respond
                                                    feedback: retry / quarantine / verify / replan
''')
    BL(["<b>Shared by all strategies:</b> scene representation (<font face='Courier'>semplan.scene.Scene</font>), "
        "action vocabulary (<font face='Courier'>Move</font>, expanded to PICK and PLACE steps), goal "
        "representation (colour sorting by default, explicit object-to-zone goals otherwise), validator, "
        "replanning interface, feedback protocol.",
        "<b>Switching strategy:</b> set <font face='Courier'>STRATEGY</font> when starting the server, or pass "
        "<font face='Courier'>strategy=</font> to <font face='Courier'>PlanningService</font>. The endpoints "
        "and message formats do not change; a test runs the same client against three strategies.",
        "<b>Representation:</b> structured JSON on the wire, a support forest in memory. PDDL is generated "
        "from it when a PDDL planner is selected. Graph embeddings are not used; nothing here needs them.",
        "<b>Incremental updates:</b> the scene endpoint accepts deltas by object id against the last version.",
        "<b>Uncertainty on the wire:</b> per-object identity confidence, pose confidence, reachability, and "
        "optional colour probabilities for the uncertainty gate."])

    # ------------------------------------------------------------------ Part 8
    H1("Part 8. Three scenarios, three architectures")
    P("The outputs below were produced while building this PDF. The three architectures are the adaptive "
      "symbolic planner (H inside O), scene graph to PDDL (F), and LLM+P with goal translation (D, mock model).")
    blocked = parse_scene(SC.BLOCKED)
    probs = {o["id"]: o["color_probs"] for o in SC.OCCLUDED["objects"] if "color_probs" in o}
    adaptive = AdaptivePlanner(color_probs=probs)
    pddl = PddlStrategy("fast-downward")
    llmp = LLMPlusPStrategy(MockLLM(goal_oracle), instruction="sort everything; put the red cube in the red zone")
    H2("Scenario 1. An object blocks another")
    P("The blue cylinder stands on the red cube. The blue zone holds a misplaced red prism and has one "
      "ground pose, so the cylinder cannot go home directly. One free slot exists.")
    CODE("\n".join(out_line(run(s, blocked)) for s in (adaptive, pddl, HtnStrategy())))
    P("All three find a valid plan; they differ in which object they park. With an explicit instruction, "
      "LLM+P translates only the goal:")
    CODE(out_line(run(LLMPlusPStrategy(MockLLM(goal_oracle), instruction="put the red cube in the green zone"),
                      blocked)))
    H2("Scenario 2. An object cannot be identified")
    occluded = parse_scene(SC.OCCLUDED)
    CODE("\n".join([
        "# graph planner: moves the unknown object aside, sorts the rest, reports what is missing",
        out_line(run(GraphStrategy(), occluded)),
        "# uncertainty gate: only the move that both colour hypotheses agree on, then look",
        out_line(run(ActivePerceptionStrategy(probs), occluded)),
        "# PDDL: same partial plan from the same goals",
        out_line(run(pddl, occluded)),
        "# after REQUEST_NEW_VIEW the external camera reports blue (scene_version 2)",
        out_line(run(adaptive, parse_scene(SC.OCCLUDED_RESOLVED)))]))
    P("An LLM-based architecture adds nothing here: the missing fact is visual and is resolved by the laptop "
      "with another camera view. The server's job is to notice that it matters and to ask.")
    H2("Scenario 3. The scene changes during execution")
    before, after = parse_scene(SC.CHANGE_BEFORE), parse_scene(SC.CHANGE_AFTER)
    plan = run(GraphStrategy(), before).moves
    remaining = [mv for mv in plan if mv != Move("blue_cylinder_1", "blue_zone")]
    lines = [f"plan on scene_version 1:   {plan_text(plan)}",
             "blue cylinder delivered; a person puts the red cube on the green cube -> scene_version 2",
             f"remaining old plan:        {plan_text(remaining)}   (first step no longer executable)"]
    for method in (ValidateThenReplan(), FullReplan(), PrefixRepair(), ValidateThenReplan(pddl)):
        r = method.replan(after, remaining)
        name = method.name + (" + PDDL" if method.planner is pddl else "")
        lines.append(f"{name:<28} {r.decision:<10} {r.ms:8.3f} ms  v{r.based_on_version}  {plan_text(r.moves)}")
    CODE("\n".join(lines))
    P("The server never assumes the old steps succeeded: the input is the observed scene with its version, "
      "and the response names the version it is based on. Feedback that refers to the old plan is refused "
      "with HTTP 409.")

    # ------------------------------------------------------------------ Part 9
    H1("Part 9. Performance optimisation")
    TABLE([
        ["Bottleneck", "Remedy", "Expected effect", "Evidence"],
        ["Model generation on every request", "Remove the model from the request path: symbolic planner, or offline synthesis", "Largest single effect", "Structural; model time not measured here"],
        ["Repeated model calls per step (grounding, scoring)", "Do perception locally; send symbols", "Removes all server-side visual inference in normal operation", "Design of this system"],
        ["PDDL start-up and grounding", "Task-specific planner; or keep the PDDL planner for rare cases only", f"{ms(fd['p50_ms'])} to {ms(g['p50_ms'])}", "measured"],
        ["Pure-Python PDDL grounding", "Use Fast Downward for larger scenes", f"pyperplan p95 at 8 objects: {ms(row('pddl:pyperplan', 8)['p95_ms'])}", "measured"],
        ["Search on unsolvable scenes", "Cheap infeasibility precheck, time-bounded A*, negative cases", "Flat tail latency", "measured in semplan"],
        ["Replanning from scratch when nothing relevant changed", "Validate the remaining plan first", f"{rep['validate_then_replan']['kept']}/{rep['validate_then_replan']['cases']} disturbances needed no new plan", "measured"],
        ["Experience retrieval", "Use only in front of slow planners", "No gain in front of the graph planner", "measured"],
        ["Small or quantised models", "Helps only where a model is still needed (goal translation, fallback)", "Unknown here", "not measured"],
        ["Incremental state updates", "Deltas by object id", "Smaller messages; no effect on planning time at this size", "payload measured in semplan"],
        ["Transport", "Persistent connection; one request per event", "Network dominates once planning is sub-millisecond", "loopback measured in semplan"],
    ], [42, 52, 46, 40], "Table 7. Latency bottlenecks and remedies.")

    # ------------------------------------------------------------------ Part 10
    H1("Part 10. Implementation roadmap")
    BL(["<b>1. Baseline (now).</b> Start <font face='Courier'>approaches.serve</font> with STRATEGY=graph; "
        "connect the laptop client; execute one step at a time; send an observation with each feedback.",
        "<b>2. Benchmark on the robot.</b> The three scenarios, then the scenario ladder from the semplan "
        "report; log timing_ms, decision and status for every request.",
        "<b>3. Comparison runs.</b> Same scenes with STRATEGY=pddl:fast-downward and STRATEGY=htn.",
        "<b>4. Model experiments on the GPU machine.</b> run_examples with --model: direct, modulo, LLM+P goal "
        "translation, policy synthesis. Record validity, calls and seconds per request.",
        "<b>5. Uncertainty gate.</b> Have the laptop send colour probabilities; measure how many extra views "
        "the gate saves compared with always looking.",
        "<b>6. Only then:</b> offline synthesis as a research study (how good are policies written by different "
        "models, how many feedback rounds, how do they generalise to larger scenes), learned action costs from "
        "failure statistics, and, if still of interest, a WL-feature or GNN policy as a negative or positive result."])

    # ------------------------------------------------------------------ Part 11
    H1("Part 11. Final technical decisions")
    QA = [
        ("1. Is LLM+P suitable?", "In its original form, no: its job is translating a language problem into "
         "PDDL, and here the state is already symbolic. Reduced to translating an instruction into a goal, yes."),
        ("2. Is VLM+PDDL better than direct VLM planning?", "For a structured domain like this one, yes: a "
         "planner guarantees the plan given the state, and your own split-mode result points that way. "
         "ViPlan is the benchmark built for exactly this question; as I recall its abstract [M], grounding "
         "plus planner wins in Blocksworld and direct VLM planning wins in the household tasks, so the answer "
         "depends on the domain. The weak point of VLM+PDDL is the state translation (Planetarium), which "
         "this system moves to the laptop."),
        ("3. Can a VLM or LLM be only a fallback?", "Yes. In the adaptive planner the model is reached only "
         "after every symbolic stage failed, and its answer still has to pass the validator. On solvable test "
         "scenes it was never called."),
        ("4. Can a semantic scene graph eliminate most server-side VLM inference?", "Yes, all of it in normal "
         "operation. The remaining uses are instruction understanding and objects the laptop cannot classify."),
        ("5. Would GNN-based planning be faster or more reliable?", "Neither, on current evidence. The linear "
         f"stand-in was slower ({ms(lrn['p50_ms'])} against {ms(g['p50_ms'])}) and solved "
         f"{pc(lrn['valid'], lrn['scenes'])} alone. A real GNN was not trained, so this is a prediction for it, "
         "supported by the literature on expressiveness limits and by WL-GOOSE."),
        ("6. Could case-based planning and RAG reduce repeated reasoning?", "They reduce repeated <i>slow</i> "
         "reasoning. In front of the graph planner there is nothing to save. In front of Fast Downward or a "
         "model, exact case retrieval is worthwhile. RAG with an LLM keeps the generation cost."),
        ("7. Which architecture supports the fastest reliable replanning?", "Validate-then-replan on the graph "
         f"planner: {ms(rep['validate_then_replan']['p50_ms'])} median, and "
         f"{rep['validate_then_replan']['kept']}/{rep['validate_then_replan']['cases']} disturbances needed no "
         "new plan."),
        ("8. Which approach needs the least training?", "All symbolic ones need none. Offline synthesis needs "
         "no training either, only a handful of test scenes."),
        ("9. Which offers the most interesting research contribution?", "Two candidates. (a) Offline synthesis "
         "of planning code for a physical robot, compared against the hand-written planner and against "
         "model-in-the-loop planning on the same scenes. (b) Semantic communication with an uncertainty gate: "
         "how much perception and bandwidth are saved by looking only when the plan depends on it."),
        ("10. What exactly to implement first?", "The adaptive server with the graph planner as it stands in "
         "this repository, connected to the real laptop client, with per-step verification. Everything else "
         "is a switch of STRATEGY away."),
    ]
    for q, a in QA:
        story.append(KeepTogether([Paragraph(q, S["h2"]), Paragraph(a, S["body"])]))

    H1("Limitations")
    BL(["No language or vision model was run. Every statement about model behaviour here is from the "
        "literature or is a mechanism demonstration with scripted answers.",
        "The literature search read abstracts and summaries, not full papers. Check each paper before citing "
        "it, in particular the entries marked [M] and the 2026 preprints.",
        "No published latency numbers were found for the listed systems, so no cross-paper latency comparison "
        "is offered.",
        "All measurements are on synthetic scenes and a simulated robot on one laptop.",
        "The 'canned' generated policy and heuristics were written to resemble model output; they are not "
        "model output.",
        "The learned-policy result is for a linear ranker, not a GNN.",
        "The HTN methods and the generated policy are incomplete by construction; the validator turns their "
        "gaps into 'no plan' rather than wrong plans."])

    H1("References")
    refs = [
        "Katz, M., Kokel, H., Srinivas, K., Sohrabi, S. (2024). Thought of Search: Planning with Language Models Through the Lens of Efficiency. NeurIPS. " + A("2404.11833"),
        "Silver, T., Dan, S., Srinivas, K., Tenenbaum, J., Kaelbling, L., Katz, M. (2024). Generalized Planning in PDDL Domains with Pretrained Large Language Models. AAAI. " + A("2305.11014"),
        "Correa, A., Pereira, A., Seipp, J. (2025). Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code. " + A("2503.18809"),
        "Liu, B., Jiang, Y., Zhang, X., Liu, Q., Zhang, S., Biswas, J., Stone, P. (2023). LLM+P: Empowering Large Language Models with Optimal Planning Proficiency. " + A("2304.11477"),
        "Kambhampati, S. et al. (2024). LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks. " + A("2402.01817"),
        "Merler, M. et al. (2025). ViPlan: A Benchmark for Visual Planning with Symbolic Predicates and Vision-Language Models. " + A("2505.13180"),
        "Zuo, M. et al. (2024). Planetarium: A Rigorous Benchmark for Translating Text to Structured Planning Languages. " + A("2407.03321"),
        "Zhang, X. et al. (2024). DKPROMPT: Domain Knowledge Prompting Vision-Language Models for Open-World Planning. " + A("2406.17659"),
        "Robot Planning and Situation Handling with Active Perception (VAP-TAMP) (2026). " + A("2604.26988"),
        "Shirai, K. et al. (2024). Vision-Language Interpreter for Robot Task Planning. ICRA. " + A("2311.00967") + "; ViLaIn-TAMP: " + A("2506.03270"),
        "Dang, X., Kudlackova, L., Edelkamp, S. (2025). Planning with Vision-Language Models and a Use Case in Robot-Assisted Teaching (Image2PDDL). " + A("2501.17665"),
        "Liu, Y., Palmieri, L., Koch, S., Georgievski, I., Aiello, M. (2025). DELTA: Decomposed Efficient Long-Term Robot Task Planning using Large Language Models. ICRA. " + A("2404.03275"),
        "Rana, K. et al. (2023). SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning. CoRL. " + A("2307.06135"),
        "Guan, L., Valmeekam, K., Sreedharan, S., Kambhampati, S. (2023). Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning. NeurIPS. " + A("2305.14909"),
        "Curtis, A. et al. (2024). Partially Observable Task and Motion Planning with Uncertainty and Risk Awareness. RSS. " + A("2403.10454"),
        "Ren, A. et al. (2023). Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners. CoRL. " + A("2307.01928"),
        "Chen, D., Trevizan, F., Thiebaux, S. (2024). Return to Tradition: Learning Reliable Heuristics with Classical Machine Learning. ICAPS. " + A("2403.16508"),
        "Yang, Z., Garrett, C., Fox, D., Lozano-Perez, T., Kaelbling, L. (2024). Guiding Long-Horizon Task and Motion Planning with Vision Language Models. " + A("2410.02193"),
        "Kumar, N. et al. Open-World Task and Motion Planning via Vision-Language Model Generated Constraints. RA-L. " + A("2411.08253"),
        "VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning. ICLR 2025. " + A("2410.23156") + "; From Pixels to Predicates. " + A("2501.00296"),
        "Memory Transfer Planning (2025). " + A("2509.24160") + "; CoRAL (2026). " + A("2605.02600"),
        "Automatic Behavior Tree Expansion with LLMs for Robotic Manipulation (2024). " + A("2409.13356") + "; BTGenBot-2 (2026). " + A("2602.01870"),
        "Hao, Y., Chen, Y., Fan, C., Zhang, Y. (2025). Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning. " + A("2510.03182"),
        "[M] Ahn, M. et al. (2022). SayCan. " + A("2204.01691") + "; Huang, W. et al. (2022). Inner Monologue. " + A("2207.05608") + "; Singh, I. et al. (2022). ProgPrompt. " + A("2209.11302") + "; Liang, J. et al. (2022). Code as Policies. " + A("2209.07753"),
    ]
    for r in refs:
        story.append(Paragraph(r, S["ref"]))


def main():
    random.seed(0)
    fig_latency()
    build()
    doc = SimpleDocTemplate(str(OUT / "hybrid_planning_guide.pdf"), pagesize=A4, leftMargin=18 * mm,
                            rightMargin=18 * mm, topMargin=17 * mm, bottomMargin=18 * mm,
                            title="Hybrid planning architectures for low-latency robotic manipulation",
                            author="vlm_pddl")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT / "hybrid_planning_guide.pdf")


if __name__ == "__main__":
    main()
