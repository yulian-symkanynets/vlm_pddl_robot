"""Print one /plan response and check it against the request's "_check" block.

    curl ... | python3 scripts/check_response.py examples/requests/03_*.json [summary.jsonl]

_check keys: status (str or list), constraints (exact dict the model must produce; {} = none),
on / in (final relations), no_place_on, never_pick, first_pick, max_moves (picks).
The plan is replayed here: the final arrangement is computed from the actions themselves.
"""
import json
import sys


def replay(scene, plan):
    on = {o["id"]: o.get("on", "table") for o in scene["objects"]}
    picks, placed_on = [], []
    for a in plan:
        name, args = a["action"], a["args"]
        if name.startswith("pick"):
            picks.append(args[0])
        elif name == "place-on-table":
            on[args[0]] = "table"; placed_on.append("table")
        elif name == "place-on-item":
            on[args[0]] = args[1]; placed_on.append(args[1])
        elif name == "place-in":
            on[args[0]] = args[2]; placed_on.append(args[2])
    return on, picks, placed_on


def base(on, x):
    while on.get(x) in on:
        x = on[x]
    return on.get(x)


def main():
    req = json.load(open(sys.argv[1]))
    raw = sys.stdin.read()
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        print("!! server did not answer with JSON:", raw[:300])
        return
    chk = req.get("_check", {})
    print("instruction :", req.get("instruction"))
    print("expected    :", req.get("_expected"))
    print("status      :", r.get("status"), "-", r.get("message") or "")
    if r.get("decision"):
        print("decision    :", r["decision"], "| changes:", r.get("changes"))
    print("goal (VLM)  :", r.get("goal"))
    print("constraints :", r.get("constraints"))
    if r.get("warnings"):
        print("warnings    :", r["warnings"])
    print("domain      :", "built for this request" if r.get("domain_pddl") else "standard tabletop.pddl",
          "| planner:", r.get("planner"))
    if r.get("status") == "ok" and not r.get("plan"):
        print("  (nothing to do)")
    for i, a in enumerate(r.get("plan") or [], 1):
        print("  %2d. %s" % (i, a["pddl"]))
    t = r.get("timings", {})
    print("time        :", t.get("total_s"), "s (goal model", t.get("goal_s"), "s)")

    fails = []
    if chk.get("decision") and r.get("decision") != chk["decision"]:
        fails.append(f"decision {r.get('decision')} != {chk['decision']}")
    want = chk.get("status")
    if want is not None and r.get("status") not in (want if isinstance(want, list) else [want]):
        fails.append(f"status {r.get('status')} != {want}")
    if "constraints" in chk and r.get("status") == "ok":
        got = {k: v for k, v in (r.get("constraints") or {}).items()}
        if got != chk["constraints"]:
            fails.append(f"constraints {got} != {chk['constraints']}")
    if r.get("status") == "ok" and (r.get("plan") or "on" in chk or "in" in chk):
        on, picks, placed = replay(req["scene"], r["plan"])
        for a, b in chk.get("on", []):
            if on.get(a) != b:
                fails.append(f"{a} ends on {on.get(a)}, not {b}")
        for a, z in chk.get("in", []):
            if base(on, a) != z:
                fails.append(f"{a} ends in {base(on, a)}, not {z}")
        for d in chk.get("no_place_on", []):
            if d in placed:
                fails.append(f"something was placed on {d}")
        for x in chk.get("never_pick", []):
            if x in picks:
                fails.append(f"{x} was moved")
        if chk.get("first_pick") and (not picks or picks[0] != chk["first_pick"]):
            fails.append(f"first pick {picks[0] if picks else None} != {chk['first_pick']}")
        if chk.get("max_moves") is not None and len(picks) > chk["max_moves"]:
            fails.append(f"{len(picks)} moves, the shortest plan has {chk['max_moves']}")
    print("RESULT      :", "PASS" if not fails else "FAIL  " + " | ".join(fails))
    if len(sys.argv) > 2:
        with open(sys.argv[2], "a") as f:
            f.write(json.dumps({"file": sys.argv[1], "pass": not fails, "fails": fails,
                                "goal_s": t.get("goal_s"), "total_s": t.get("total_s")}) + "\n")


if __name__ == "__main__":
    main()
