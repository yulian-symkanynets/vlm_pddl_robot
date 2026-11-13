"""Run every strategy on the three scenarios and print what each one does.

    python -m approaches.run_examples            # everything that needs no GPU
    python -m approaches.run_examples --model qwen3vl-8b   # use a real model from models.yaml

Without --model the language-model strategies use scripted answers (approaches/mock.py),
which shows their control flow but not how a real model behaves.
"""
from __future__ import annotations

import argparse
import random

from . import run, scenarios as S
from .active_perception import ActivePerceptionStrategy
from .adaptive import AdaptivePlanner
from .behavior_tree import run_tree
from .htn import HtnStrategy
from .llm import MockLLM
from .llm_offline import ProgramStrategy, synthesize_policy
from .llm_online import LLMDirectStrategy, LLMModuloStrategy, LLMPlusPStrategy
from .mock import POLICY_V1, POLICY_V2, POLICY_V3, goal_oracle, planner_oracle
from .replanning import FullReplan, PrefixRepair, ValidateThenReplan
from .symbolic import CaseBasedStrategy, GraphStrategy
from semplan.gen import random_scene
from semplan.planner import quick_infeasible
from semplan.scene import Move, parse_scene
from semplan.sim import SimRobot


def show(o, indent="   "):
    plan = ", ".join(f"{m.obj}->{m.dst}" for m in o.moves) or "-"
    extra = f"  llm_calls={o.llm_calls}" if o.llm_calls else ""
    print(f"{indent}{o.strategy:<18} {o.status:<17} {o.ms:8.3f} ms{extra}  [{plan}]")
    if o.reason:
        print(f"{indent}{'':<18} reason: {o.reason}")
    for r in o.info_requests:
        print(f"{indent}{'':<18} -> {r['action']}({r['target']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model key from models.yaml (needs the GPU machine)")
    ap.add_argument("--quant", default="4bit")
    args = ap.parse_args()
    if args.model:
        from .llm import RepoVLM
        real = RepoVLM(args.model, args.quant)
        plan_llm = goal_llm = real
        print(f"language model: {args.model} ({args.quant})")
    else:
        plan_llm, goal_llm = MockLLM(planner_oracle(error_rate=1.0)), MockLLM(goal_oracle)
        print("language model: MOCK (scripted answers; first plan always has the "
              "'picks a covered object' mistake)")

    rng = random.Random(0)
    train = []
    while len(train) < 40:
        s = random_scene(rng, rng.randint(3, 5), n_colors=3, n_buffers=rng.choice([1, 2]))
        if not quick_infeasible(s) and run(GraphStrategy(), s).status == "ok":
            train.append(s)
    synth_llm = plan_llm if args.model else MockLLM([POLICY_V1, POLICY_V2, POLICY_V3])
    code, log = synthesize_policy(synth_llm, train, accept=0.9)
    print("\noffline policy synthesis (once, before any request):")
    for l in log:
        print(f"   round {l['round']}: solved {l['solved']}/{l['of']}"
              + (f"   first failure: {l['error'][:70]}" if l["error"] else ""))

    strategies = [GraphStrategy(), GraphStrategy(optimal=True), HtnStrategy(), CaseBasedStrategy(),
                  LLMDirectStrategy(plan_llm), LLMModuloStrategy(plan_llm), LLMPlusPStrategy(goal_llm)]
    if code:
        strategies.append(ProgramStrategy(code))
    try:
        from .symbolic import PddlStrategy
        strategies[3:3] = [PddlStrategy("pyperplan"), PddlStrategy("fast-downward")]
    except Exception as e:
        print(f"(PDDL strategies skipped: {e})")

    # ------------------------------------------------------------ scenario 1
    print("\n=== Scenario 1: an object blocks another (blue zone is occupied, one free slot)")
    scene = parse_scene(S.BLOCKED)
    for stg in strategies:
        show(run(stg, scene))
    print("   with an instruction, LLM+P style: 'put the red cube in the green zone'")
    show(run(LLMPlusPStrategy(goal_llm, instruction="put the red cube in the green zone"), scene))

    # ------------------------------------------------------------ scenario 2
    print("\n=== Scenario 2: the object on top cannot be identified")
    scene = parse_scene(S.OCCLUDED)
    probs = {o["id"]: o["color_probs"] for o in S.OCCLUDED["objects"] if "color_probs" in o}
    print("   symbolic planners move the unknown object aside but do not sort it (status partial):")
    show(run(GraphStrategy(), scene))
    print("   active perception: execute only what every colour hypothesis agrees on, then look:")
    show(run(ActivePerceptionStrategy(probs), scene))
    print("   after REQUEST_NEW_VIEW the external camera reports blue (scene_version 2):")
    show(run(GraphStrategy(), parse_scene(S.OCCLUDED_RESOLVED)))

    # ------------------------------------------------------------ scenario 3
    print("\n=== Scenario 3: the scene changes during execution")
    before, after = parse_scene(S.CHANGE_BEFORE), parse_scene(S.CHANGE_AFTER)
    plan = run(GraphStrategy(), before).moves
    print("   plan on scene_version 1: ", ", ".join(f"{m.obj}->{m.dst}" for m in plan))
    remaining = [m for m in plan if m != Move("blue_cylinder_1", "blue_zone")]
    print("   blue cylinder delivered; a person puts the red cube on the green cube (version 2)")
    for method in (FullReplan(), ValidateThenReplan(), PrefixRepair()):
        r = method.replan(after, remaining)
        print(f"   {method.name:<22} {r.decision:<10} {r.reused} old move(s) in new plan {r.ms:7.3f} ms  "
              f"[{', '.join(f'{m.obj}->{m.dst}' for m in r.moves)}]  based_on_version={r.based_on_version}")
    print("   behaviour tree on a simulated robot, same disturbance injected after the first move:")
    robot = SimRobot(before, random.Random(1))
    def disturb(tick):
        if tick == 1 and robot.truth.held() is None:
            robot.truth.on["red_cube_1"] = "green_cube_1"
    out = run_tree(robot, disturb=disturb)
    print(f"   goal={out['goal']} ticks={out['ticks']} replans={out['replans']} robot_steps={out['steps']}")

    # ------------------------------------------------------------ adaptive
    print("\n=== Adaptive planner: which stage answers")
    adaptive = AdaptivePlanner(llm_fallback=LLMModuloStrategy(plan_llm), color_probs=probs)
    for name in ("BLOCKED", "BLOCKED", "OCCLUDED", "CHANGE_AFTER"):
        o = run(adaptive, parse_scene(getattr(S, name)))
        print(f"   {name:<13} {o.status:<8} {o.ms:7.3f} ms  {o.source}")


if __name__ == "__main__":
    main()
