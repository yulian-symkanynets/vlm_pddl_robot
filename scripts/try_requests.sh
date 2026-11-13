#!/usr/bin/env bash
# Send every request in examples/requests/ to the running server and print the result.
#   ./serve.sh start qwen3-vl-8b 4bit
#   scripts/try_requests.sh                 # all files
#   scripts/try_requests.sh 03 05           # only files starting with 03 or 05
# Env: URL (default http://localhost:8010)
cd "$(dirname "$0")/.." || exit 1
URL=${URL:-http://localhost:8010}
files=()
if [ $# -eq 0 ]; then files=(examples/requests/*.json)
else for p in "$@"; do files+=(examples/requests/"$p"*.json); done; fi
for f in "${files[@]}"; do
  echo "================ $(basename "$f")"
  curl -s -X POST "$URL/plan" -H 'Content-Type: application/json' --data-binary "@$f" |
  python3 -c '
import json, sys
req = json.load(open(sys.argv[1]))
r = json.load(sys.stdin)
print("instruction :", req["instruction"])
print("expected    :", req.get("_expected"))
print("status      :", r.get("status"), "-", r.get("message") or "")
print("goal (VLM)  :", r.get("goal"))
print("constraints :", r.get("constraints"))
print("domain      :", "built for this request" if r.get("domain_pddl") else "standard tabletop.pddl")
for i, a in enumerate(r.get("plan") or [], 1):
    print(f"  {i:2}. {a[\"pddl\"]}")
t = r.get("timings", {})
print("time        :", t.get("total_s"), "s (goal model", t.get("goal_s"), "s)")
' "$f"
done
