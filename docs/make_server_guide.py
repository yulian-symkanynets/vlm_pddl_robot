"""Build docs/vlm_planning_server_guide.pdf: how the VLM + PDDL planning server works, how to
serve it on WSL with port forwarding, how to test it, and which quantized model to use.

    python docs/make_server_guide.py

The PDDL examples in the guide are produced by running the real code while the PDF is built.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

import openworld
import pddl_task

OUT = ROOT / "docs" / "vlm_planning_server_guide.pdf"
FIG = ROOT / "docs" / "figures" / "server_pipeline.png"
REQ = ROOT / "examples" / "requests"

# ---------------------------------------------------------------- styles
INK, ACCENT, MUTED = colors.HexColor("#1f2933"), colors.HexColor("#1d4ed8"), colors.HexColor("#52606d")
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=8)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=ACCENT,
                    spaceBefore=12, spaceAfter=5)
H3 = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=INK,
                    spaceBefore=8, spaceAfter=3)
P = ParagraphStyle("p", fontName="Helvetica", fontSize=9.5, leading=13.2, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("s", parent=P, fontSize=8.5, leading=11, textColor=MUTED)
CELL = ParagraphStyle("c", parent=P, fontSize=8.3, leading=10.5, spaceAfter=0)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold")
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.6, leading=9.4, textColor=INK,
                      backColor=colors.HexColor("#f3f4f6"), borderPadding=5, spaceBefore=3, spaceAfter=7)


def para(t, st=P):
    return Paragraph(t, st)


def code(t, width=104):
    """Preformatted block; long lines are wrapped so nothing runs off the page."""
    import textwrap
    out = []
    for line in t.strip("\n").splitlines():
        if len(line) <= width:
            out.append(line)
            continue
        ind = len(line) - len(line.lstrip())
        out += textwrap.wrap(line, width, subsequent_indent=" " * (ind + 4), break_long_words=True,
                             break_on_hyphens=False)
    return Preformatted("\n".join(out), CODE)


def table(rows, widths, head=True):
    data = [[Paragraph(c if isinstance(c, str) else str(c), CELLB if (head and i == 0) else CELL)
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e7eb")))
    t.setStyle(TableStyle(st))
    return t


def bullets(items):
    return [para("&bull;&nbsp; " + i) for i in items]


# ---------------------------------------------------------------- pipeline figure
def pipeline_figure():
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.1))
    ax.set_xlim(0, 100), ax.set_ylim(0, 42), ax.axis("off")

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec="#52606d", lw=1))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=10, weight="bold",
                color="#1f2933")
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=7.6, color="#3e4c59")

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     color="#52606d", lw=1.1))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.4, label, ha="center", fontsize=7, color="#52606d")

    vlm, code_c, plan_c = "#dbeafe", "#e5e7eb", "#dcfce7"
    box(1, 29, 16, 10, "Image(s)", "optional", "#fff7ed")
    box(1, 15, 16, 10, "Scene JSON", "from the laptop", "#fff7ed")
    box(1, 1, 16, 10, "Sentence", "any wording, any task", "#fff7ed")
    box(23, 29, 18, 10, "VLM: describe", "image -> scene JSON", vlm)
    box(23, 15, 18, 10, "Scene reader", "code: who rests on what", code_c)
    box(23, 1, 18, 10, "VLM: understand", "goal + constraints + quotes", vlm)
    box(47, 8, 18, 12, "Checks", "ids exist, quotes are in\nthe sentence, goal possible", code_c)
    box(47, 25, 18, 12, "PDDL builder", "domain + problem\nfor THIS request", code_c)
    box(71, 25, 13, 12, "Planner", "Fast Downward\n(shortest plan)", plan_c)
    box(71, 8, 13, 12, "Verify", "replay every\naction", code_c)
    box(88, 15, 11, 12, "Plan", "pick / place\nsteps", "#fef9c3")
    arrow(17, 34, 23, 34)
    arrow(32, 29, 32, 25, "")
    arrow(17, 20, 23, 20)
    arrow(17, 6, 23, 6)
    arrow(41, 20, 47, 28)
    arrow(41, 6, 47, 12)
    arrow(56, 20, 56, 25)
    arrow(65, 31, 71, 31)
    arrow(77.5, 25, 77.5, 20)
    arrow(84, 14, 88, 19)
    ax.text(50, 41, "blue = vision-language model     grey = deterministic code     green = planner",
            ha="center", fontsize=8, color="#52606d")
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- real PDDL for one request
def example_pddl():
    req = json.loads((REQ / "03_three_stacks_no_table.json").read_text())
    sc = openworld.normalize_scene(req["scene"])
    goal = [{"object": "blue_cylinder", "on": "yellow_block"}, {"object": "green_block", "on": "red_cylinder"}]
    grounded, _ = openworld.resolve_goal(goal, sc)
    cons, _ = pddl_task.normalize_constraints({"no_table": True}, sc)
    with_c = openworld.plan_for(sc, grounded, "auto", cons)
    without = openworld.plan_for(sc, grounded, "auto", {})
    first, _ = pddl_task.normalize_constraints({"first": ["green_block"], "keep": ["purple_cylinder"]}, sc)
    dom_first = pddl_task.make_domain(first, sc)
    pick = dom_first[dom_first.index("(:action pick-first-from-item"):]
    pick = pick[:pick.index("\n\n")] if "\n\n" in pick else pick[:600]
    return req, with_c, without, pick


# ---------------------------------------------------------------- document
def build():
    pipeline_figure()
    req, with_c, without, pick_first = example_pddl()
    s = []
    s += [para("VLM + PDDL planning server", H1),
          para("How it works, how to serve it on WSL and reach it from other machines, how to test it "
               "with a real model, and which quantized model to use. Server side only: the robot / "
               "laptop side is not changed.", SMALL),
          Spacer(1, 4)]

    # 1 ----------------------------------------------------------------
    s += [para("1. The idea in one paragraph", H2),
          para("A vision-language model (VLM) is good at <b>understanding</b>: it can describe a picture "
               "and translate a sentence into a precise statement. It is bad at <b>reasoning about moves</b>: "
               "planning alone, Qwen3-VL-8B picked blocks that were still covered (0 of 5 correct in the "
               "earlier test). So the server splits the work. The VLM only translates: image to scene, "
               "sentence to goal and constraints. Code turns that translation into a PDDL problem built for "
               "this one request, and a classical planner finds the moves. The task is never written in "
               "advance: it comes from the user's sentence at run time."),
          Image(str(FIG), width=178 * mm, height=73 * mm),
          table([["Job", "Done by", "Why"],
                 ["Image to scene description", "VLM", "only a vision model can read a photo"],
                 ["Scene JSON to facts (who rests on what)", "code", "the JSON is already exact; a model could only add mistakes"],
                 ["Sentence to goal + constraints", "VLM (text-only call, no image)", "free wording, any language"],
                 ["Goal + constraints to PDDL", "code", "exact translation, always valid PDDL"],
                 ["PDDL to moves", "planner", "search: never picks a covered object, shortest plan"],
                 ["Check the plan", "code", "replays every action before it is returned"]],
                [62, 45, 71])]

    # 2 ----------------------------------------------------------------
    s += [para("2. What happens to one request", H2),
          para("Example: test file <font face='Courier'>03_three_stacks_no_table.json</font>. Three stacks, the "
               "user asks to rearrange two of them and forbids the table."),
          para("Step 1: the request", H3),
          code('{"scene": {"objects": [\n'
               + ",\n".join("    " + json.dumps(o) for o in req["scene"]["objects"][:3])
               + ',\n    ... 3 more objects]},\n "instruction": ' + json.dumps(req["instruction"]) + "}"),
          para("Step 2: the VLM reads the sentence", H3),
          para("The VLM gets the object list, the sentence and the answer format (see <font face='Courier'>"
               "goal_prompt</font> in <font face='Courier'>openworld.py</font>). It never sees a word list. It answers "
               "with the final arrangement, the limits it found, and for every limit the exact words of the "
               "sentence that state it:"),
          code('{"goal": [{"object": "blue_cylinder", "on": "yellow_block"},\n'
               '          {"object": "green_block", "on": "red_cylinder"}],\n'
               ' "constraints": {"no_table": true},\n'
               ' "evidence": {"no_table": "do not put anything on the table"}}'),
          para("Step 3: checks on the VLM answer", H3)]
    s += bullets([
        "every id must exist in the scene, otherwise status <b>clarify</b> with a question;",
        "a constraint is kept only if its quoted words really occur in the sentence. This removes limits the "
        "model invents (seen in test 09: an unrequested <i>no_table</i>). It needs no word list and works in any language;",
        "simple impossibilities are explained (two objects on one top, circular stacking, something on a prism).",
    ])
    s += [para("Step 4: the PDDL for this request", H3),
          para("<font face='Courier'>pddl_task.py</font> starts from <font face='Courier'>domain/tabletop.pddl</font> "
               "and changes only what the constraints require. With <i>no_table</i> the action that puts an "
               "object on the table is removed, so no plan can use it:"),
          code("\n".join(l for l in with_c["domain_pddl"].splitlines()
                         if "place-on-table removed" in l or "(:action place-on-item" in l)
               or "; place-on-table removed: this task forbids putting objects on the table"),
          para("The problem file is built from the scene (who rests on what) and the goal:"),
          code(with_c["problem_pddl"]),
          para("Step 5: plan and verification", H3),
          table([["Same scene and goal", "Plan"],
                 ["table allowed (test 04)", "<br/>".join(a["pddl"] for a in without["plan"])],
                 ["no table (test 03)", "<br/>".join(a["pddl"] for a in with_c["plan"])]], [45, 133]),
          para("Without the table the planner parks the red cylinder on the purple cylinder by itself. The "
               "planner is Fast Downward in optimal mode (shortest plan) with a 5 s limit; if that limit is "
               "reached, the fast non-optimal mode is used. Every plan is replayed on the problem before it is "
               "returned.", SMALL)]

    # 3 ----------------------------------------------------------------
    s += [para("3. Constraints and how they change the PDDL", H2),
          para("The kinds of limit the PDDL can express are fixed, like the predicates of any PDDL domain. The "
               "wording is not: the VLM maps any sentence onto them. A new kind of limit needs a few lines in "
               "<font face='Courier'>pddl_task.py</font> and one line in the prompt."),
          table([["Constraint (from the VLM)", "Example wording", "Domain change", "Problem facts"],
                 ["none", "put blue on yellow", "none: tabletop.pddl as is", "-"],
                 ["<font face='Courier'>no_table</font>", "do not put anything on the table; keep the table free",
                  "action place-on-table removed", "-"],
                 ["<font face='Courier'>keep: [x]</font>", "leave the purple cylinder where it is",
                  "every pick needs (movable ?x)", "(movable o) for every other object"],
                 ["<font face='Courier'>never_on: [[a|*, y]]</font>", "never put anything on the purple cylinder",
                  "place-on-item needs (can-stack ?x ?y)", "(can-stack a b) for every allowed pair"],
                 ["object without flat top", "(a prism, cone, ball in the scene)",
                  "place-on-item needs (can-stack ?x ?y)", "no pair with that object as the support"],
                 ["<font face='Courier'>first: [x]</font>", "but first move the green block",
                  "picks need (unlocked); extra pick-first-* actions", "(first-object x)"],
                 ["<font face='Courier'>max_stack_height</font>", "no stack higher than 2",
                  "not expressible in plain PDDL", "ignored with a warning; the task server (port 8020) supports it"]],
                [36, 48, 50, 44]),
          para("How <i>first</i> works", H3),
          para("All normal pick actions require <font face='Courier'>(unlocked)</font>, which is false at the start. "
               "A copy of each pick action may only take the object marked <font face='Courier'>(first-object x)</font> "
               "and makes <font face='Courier'>(unlocked)</font> true. The robot receives the normal action name "
               "(the copy is renamed back), so nothing changes on the robot side. Generated action:"),
          code(pick_first)]

    # 4 ----------------------------------------------------------------
    s += [para("4. Request and response", H2),
          para("<b>POST /plan</b> on port 8010. Send a scene (JSON) <i>or</i> images, plus a sentence."),
          table([["Field", "Meaning"],
                 ["scene", "objects with <font face='Courier'>id</font>, <font face='Courier'>color</font>, "
                           "<font face='Courier'>shape</font>, <font face='Courier'>on</font> (an object id or \"table\"); "
                           "optional <font face='Courier'>zones</font>. Also accepted: a <font face='Courier'>support</font> "
                           "map or a <font face='Courier'>relations</font> list"],
                 ["instruction", "the user's sentence, any wording"],
                 ["images_b64 / POST /plan/upload", "instead of a scene: the VLM describes the photos first"],
                 ["constraints (optional)", "force constraints instead of reading them from the sentence"],
                 ["planner (optional)", "auto (shortest plan, fallback fast), fast-downward, fast-downward-opt, pyperplan"]],
                [42, 136]),
          para("Response fields: <b>status</b> (ok | clarify | no_plan | need_observation | error), <b>message</b>, "
               "<b>goal</b> and <b>constraints</b> as understood, <b>plan</b> (list of PDDL actions), "
               "<b>problem_pddl</b>, <b>domain_pddl</b> (only when the domain was changed), <b>warnings</b>, "
               "<b>timings</b> (goal_s = model time).")]

    # 5 ----------------------------------------------------------------
    s += [PageBreak(), para("5. Serving it on the GPU machine (WSL)", H2),
          para("First time on a new Windows machine: GPU and DNS checks, clone, install (details in "
               "<font face='Courier'>docs/WSL_SETUP.md</font>):"),
          code("$ nvidia-smi                       # GPU visible inside WSL?\n"
               "$ git clone https://github.com/yulian-symkanynets/vlm_pddl_robot.git ~/vlm_pddl_robot\n"
               "$ cd ~/vlm_pddl_robot && ./setup.sh && source .venv/bin/activate\n"
               "$ python check_env.py               # CUDA, bitsandbytes, planners\n"
               "$ python download_models.py qwen3-vl-8b"),
          para("Every time:", H3),
          code("$ cd ~/vlm_pddl_robot && git pull\n"
               "$ ./serve.sh start qwen3-vl-8b 4bit     # background; waits until the model is loaded\n"
               "$ ./serve.sh status                     # running? which model?\n"
               "$ ./serve.sh logs                       # follow the log (Ctrl+C leaves, server keeps running)\n"
               "$ ./serve.sh restart qwen3-vl-8b 8bit   # switch model or quantization\n"
               "$ ./serve.sh stop\n"
               "$ PORT=8020 ./serve.sh start ...        # other port if 8010 is taken"),
          para("Check: <font face='Courier'>curl http://localhost:8010/health</font>. Browser on the same PC: "
               "<font face='Courier'>http://127.0.0.1:8010/docs</font> (interactive API) or <font face='Courier'>"
               "http://127.0.0.1:8010/</font> (test page)."),
          para("6. Port forwarding: reach WSL from the laptop / robot PC", H2),
          para("WSL2 runs behind its own virtual network, so other machines cannot see port 8010 until Windows "
               "forwards it. Choose <b>one</b> of these."),
          para("A. Mirrored networking (Windows 11 22H2 or newer, survives restarts)", H3),
          code("# C:\\Users\\<you>\\.wslconfig\n[wsl2]\nnetworkingMode=mirrored\n\n"
               "PS> wsl --shutdown\n# PowerShell as Administrator:\n"
               "PS> New-NetFirewallRule -DisplayName \"WSL planner 8010\" -Direction Inbound -Protocol TCP "
               "-LocalPort 8010 -Action Allow\n"
               "PS> New-NetFirewallHyperVRule -Name \"WSLPlanner8010\" -DisplayName \"WSL planner 8010\" "
               "-Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8010"),
          para("Reopen Ubuntu, start the server. Address for other machines: the Windows IP from "
               "<font face='Courier'>ipconfig</font>."),
          para("B. Port forwarding script (any Windows; run again after every restart)", H3),
          code("# server running in WSL first, then PowerShell AS ADMINISTRATOR:\n"
               "PS> cd \\\\wsl.localhost\\Ubuntu-24.04\\home\\<user>\\vlm_pddl_robot\n"
               "PS> powershell -ExecutionPolicy Bypass -File scripts\\wsl_portproxy.ps1             # port 8010\n"
               "PS> powershell -ExecutionPolicy Bypass -File scripts\\wsl_portproxy.ps1 -Port 8020  # task server\n"
               "PS> powershell -ExecutionPolicy Bypass -File scripts\\wsl_portproxy.ps1 -Remove     # undo"),
          para("The script finds the current WSL IP, adds <font face='Courier'>netsh portproxy</font> 0.0.0.0:8010 to "
               "WSL:8010, opens the Windows firewall, prints the addresses other machines should use and checks "
               "that the server answers. The WSL IP changes after each Windows restart, so run it again then."),
          para("C. Tailscale (network blocks device-to-device traffic, or access from home)", H3),
          code("$ curl -fsSL https://tailscale.com/install.sh | sh\n$ sudo tailscale up\n$ tailscale ip -4      # 100.x.y.z"
               "\n# laptop: install Tailscale with the same account, then http://100.x.y.z:8010/"),
          para("Check from the other machine", H3),
          code("curl http://<windows-ip>:8010/health\n"
               "URL=http://<windows-ip>:8010 scripts/try_requests.sh 01"),
          table([["Problem", "Fix"],
                 ["works on the PC, not from the laptop", "run the forwarding script as admin (again after a restart), or use A"],
                 ["browser shows another app", "port taken: PORT=8020 ./serve.sh restart qwen3-vl-8b 4bit and forward 8020"],
                 ["./serve.sh start says the server exited", "./serve.sh logs; missing package: pip install -r requirements.txt"],
                 ["Could not resolve host in WSL", "DNS fix in SERVER.md (write /etc/resolv.conf, generateResolvConf = false)"],
                 ["slow or out of GPU memory", "smaller model or 4-bit, section 8"]], [62, 116])]

    # 7 ----------------------------------------------------------------
    s += [para("7. Testing with the real model", H2),
          para("<font face='Courier'>examples/requests/</font> holds 14 requests in the plain format. Each has the "
               "sentence, the scene, a human note (<font face='Courier'>_expected</font>) and exact pass conditions "
               "(<font face='Courier'>_check</font>). The script sends them, replays every plan independently and "
               "prints PASS / FAIL plus a summary with the mean model time."),
          code("$ scripts/try_requests.sh            # all 14\n$ scripts/try_requests.sh 03 08      # some"),
          table([["#", "Tests", "Pass condition"],
                 ["01, 04", "plain rearrangement", "goal reached, no constraints, 3 moves"],
                 ["02, 03", "\"do not put anything on the table\" in two scenes", "no_table read, nothing placed on the table, 3 moves"],
                 ["05", "\"but first move the green block\"", "first pick is green, 4 moves"],
                 ["06", "\"leave the purple cylinder where it is\"", "keep read, purple never picked"],
                 ["07", "\"never put anything on the purple cylinder\"", "never_on read, nothing on purple"],
                 ["08", "unseen wording: \"keep the table free\"", "no_table read anyway"],
                 ["09", "a tower of three, no limits", "no invented constraint, 4 moves"],
                 ["10", "sort onto the matching papers", "every object in its zone"],
                 ["11", "something on a prism", "refused (no_plan or clarify)"],
                 ["12, 13", "unknown object; \"rearrange these blocks\"", "a question back (clarify)"],
                 ["14", "the same task in Slovak", "same goal and plan as 01"]], [14, 82, 82]),
          para("Add a test: copy a file, change the sentence and the <font face='Courier'>_check</font> block "
               "(keys: status, constraints, on, in, no_place_on, never_pick, first_pick, max_moves)."),
          para("First run with Qwen3-VL-8B (4-bit), before the fixes", H3),
          table([["Result", "Tests", "Note"],
                 ["correct", "01, 03, 04, 05, 06, 07, 08, 10, 11, 12, 13, 14",
                  "all constraints read correctly, including the unseen wording (08) and Slovak (14); 2 to 5.7 s per request"],
                 ["valid but too long", "02", "12 actions instead of 6: the fast planner returns the first plan it finds. "
                                              "Fixed: optimal planner first"],
                 ["invented constraint", "09", "the VLM added no_table although the sentence has no limit: 7 moves instead of 4. "
                                               "Fixed: a constraint needs quoted words from the sentence"]],
                [34, 52, 92]),
          para("No plan in that run broke a rule; the two problems were plan length and an invented limit. "
               "The fixes are in the code but were not yet re-measured with the model: run "
               "<font face='Courier'>scripts/try_requests.sh</font> again after <font face='Courier'>git pull</font>.", SMALL)]

    # 7b ---------------------------------------------------------------
    s += [para("7b. The scene changed after the plan: POST /replan", H2),
          para("While the robot works, the scene can change: the robot made progress, a person moved or added "
               "an object, an object vanished. The laptop sends the scene as it is now together with the "
               "<font face='Courier'>request_id</font> of the earlier answer:"),
          code('POST /replan\n{"request_id": "20261113_101500_ab12cd", "scene": { ...the scene NOW... }}'),
          para("The server reuses the goal and constraints it understood the first time, so <b>the VLM is not "
               "called again</b> (replanning takes only planner time). It lists what changed and decides:"),
          table([["Decision", "When", "Answer"],
                 ["kept", "the new scene is a state the old plan passes through (the robot made progress)",
                  "the rest of the old plan"],
                 ["replanned", "anything else (a person moved something, a new object, a failed step)",
                  "a new plan from the new scene, same goal, same constraints"],
                 ["clarify", "an object the task names is gone", "a question; nothing is guessed"]], [24, 92, 62]),
          para("A \"first move X\" rule counts as done once X has moved. A replan answer has its own request_id, "
               "so the next change can be sent against it."),
          code("$ scripts/try_replan.sh          # 7 cases: /plan, then /replan with a changed scene\n"
               "$ scripts/try_replan.sh R3 R5"),
          table([["Case", "What changed", "Must happen"],
                 ["R1", "the robot did the first move of the plan", "kept: the remaining moves"],
                 ["R2", "a person put the red cylinder on the green block", "replanned, goal reached, 3 moves"],
                 ["R3", "a new white block appeared on top of the red cylinder", "replanned: white removed first, 4 moves"],
                 ["R4", "a person did the whole task", "nothing left to do"],
                 ["R5", "a person moved the purple cylinder to the table (no-table task)", "replanned, still nothing put on the table"],
                 ["R6", "the green block was already moved (\"first move green\" task)", "replanned without the first rule, 3 moves"],
                 ["R7", "the green block was taken away", "clarify"]], [12, 92, 74]),
          para("All 7 cases pass with the offline stand-in for the VLM (replanning itself never uses the model).", SMALL)]

    # 8 ----------------------------------------------------------------
    s += [para("8. Which quantized model", H2),
          para("The server already uses a quantized model: <font face='Courier'>./serve.sh start qwen3-vl-8b 4bit</font> "
               "loads Qwen3-VL-8B with 4-bit NF4 weights (bitsandbytes, double quantization). Choices "
               "(<font face='Courier'>models.yaml</font>, <font face='Courier'>vlm.py</font>):"),
          table([["Setting", "Memory (weights)", "Expected effect", "When"],
                 ["qwen3-vl-8b 4bit (now)", "about 5-6 GB, about 8 GB with 2 images",
                  "good understanding; measured 2-5.7 s per sentence", "default"],
                 ["qwen3-vl-8b 8bit", "about 9-10 GB", "a little closer to full precision; 8-bit in bitsandbytes is "
                  "often slower than 4-bit", "if 4-bit misreads sentences"],
                 ["qwen3-vl-8b bf16", "about 17 GB", "full precision, reference quality; tight on a 20 GB RTX A4500 "
                  "with images", "as the upper bound for the paper"],
                 ["qwen3-vl-4b 4bit", "about 3 GB, about 5 GB with 2 images", "faster, weaker in rare wordings and "
                  "other languages", "if speed matters most"]],
                [36, 38, 70, 34]),
          para("Memory figures are estimates from the model size; measure them on the machine "
               "(<font face='Courier'>nvidia-smi</font>). Quantization mostly affects the understanding step; the "
               "planning step is the same for every model. So compare the settings on the same 14 requests:"),
          code("$ ./serve.sh restart qwen3-vl-8b 4bit && scripts/try_requests.sh | tail -5\n"
               "$ ./serve.sh restart qwen3-vl-8b 8bit && scripts/try_requests.sh | tail -5\n"
               "$ ./serve.sh restart qwen3-vl-4b 4bit && scripts/try_requests.sh | tail -5\n"
               "$ ./serve.sh restart qwen3-vl-8b bf16 && scripts/try_requests.sh | tail -5"),
          para("The last line of each run gives passed tests and mean model time. Pick the smallest setting "
               "that passes all tests. Other quantization formats (AWQ, GPTQ, GGUF in llama.cpp or vLLM) need "
               "a different loader and are not wired in yet.", SMALL)]

    # 9 ----------------------------------------------------------------
    s += [para("9. Limits", H2)]
    s += bullets([
        "The PDDL path cannot express \"no stack higher than N\" and enforces only the first object of a "
        "\"first\" order. The task server (port 8020) supports both and also replans after failures and scene changes.",
        "A new <i>kind</i> of limit (for example \"keep the red zone empty\") needs a new constraint type: a few "
        "lines in pddl_task.py and one line in the prompt. New wordings of existing kinds need nothing.",
        "Image to scene with the real VLM has to be measured separately (POST /plan/upload with the photos).",
        "This PDDL server plans once per request; for execution with feedback and replanning use the task server.",
    ])
    s += [para("10. Files", H2),
          table([["File", "Role"],
                 ["server.py, serve.sh", "HTTP server (port 8010) and start/stop script"],
                 ["openworld.py", "the request pipeline: scene reading, VLM prompts, checks, plan_for"],
                 ["pddl_task.py", "domain and problem changes for each request (constraints)"],
                 ["domain/tabletop.pddl", "the robot's general skills, written once"],
                 ["planning.py", "PDDL parsing, planners (Fast Downward, pyperplan), plan replay"],
                 ["examples/requests/*.json", "the 14 test requests with pass conditions"],
                 ["examples/replan/*.json, scripts/try_replan.sh", "scene-change cases for POST /replan"],
                 ["scripts/try_requests.sh, scripts/check_response.py", "send, check and summarize"],
                 ["scripts/wsl_portproxy.ps1, docs/WSL_SETUP.md", "port forwarding and WSL setup"],
                 ["docs/make_server_guide.py", "builds this PDF"]], [70, 108])]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm, title="VLM + PDDL planning server",
                            author="vlm_pddl_robot")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 8 * mm, "VLM + PDDL planning server: how it works, serving, testing")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, str(d.page))
        canvas.restoreState()

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
