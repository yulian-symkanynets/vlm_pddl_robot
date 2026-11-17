#!/usr/bin/env bash
# Scene changes after a plan was made: /plan first, then /replan with the changed scene.
#   ./serve.sh start qwen3-vl-8b 4bit
#   scripts/try_replan.sh            # all cases in examples/replan/
#   scripts/try_replan.sh R2 R5      # some
# Env: URL (default http://localhost:8010)
cd "$(dirname "$0")/.." || exit 1
URL=${URL:-http://localhost:8010}
files=()
if [ $# -eq 0 ]; then files=(examples/replan/*.json)
else for p in "$@"; do files+=(examples/replan/"$p"*.json); done; fi
SUMMARY=$(mktemp); TMP=$(mktemp -d)
for f in "${files[@]}"; do
  echo "================ $(basename "$f")"
  first=examples/requests/$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["first_request"])' "$f")
  curl -s -X POST "$URL/plan" -H 'Content-Type: application/json' --data-binary "@$first" > "$TMP/first.json"
  python3 - "$f" "$first" "$TMP/first.json" "$TMP/req.json" "$TMP/check.json" <<'PY'
import copy, json, sys
case, first, r1 = (json.load(open(a)) for a in sys.argv[1:4])
print("first plan  :", r1.get("status"), "|", " ; ".join(a["pddl"] for a in r1.get("plan") or []))
after = case["scene_after"]
if after == "AFTER_FIRST_MOVE":            # the robot did the first pick+place of the actual plan
    after = copy.deepcopy(first["scene"])
    place = next(a for a in r1["plan"] if a["action"].startswith("place"))
    x, args = place["args"][0], place["args"]
    dst = "table" if place["action"] == "place-on-table" else (args[1] if place["action"] == "place-on-item" else args[2])
    for o in after["objects"]:
        if o["id"] == x:
            o["on"] = dst
print("what changed:", case["what_changed"])
json.dump({"request_id": r1.get("request_id", ""), "scene": after}, open(sys.argv[4], "w"))
json.dump({"scene": after, "instruction": first["instruction"],
           "_expected": case["what_changed"], "_check": case["_check"]}, open(sys.argv[5], "w"))
PY
  curl -s -X POST "$URL/replan" -H 'Content-Type: application/json' --data-binary "@$TMP/req.json" |
    python3 scripts/check_response.py "$TMP/check.json" "$SUMMARY"
done
python3 - "$SUMMARY" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"\n==== {sum(r['pass'] for r in rows)}/{len(rows)} replan cases passed")
for r in rows:
    if not r["pass"]:
        print("  FAIL:", "; ".join(r["fails"]))
PY
rm -rf "$SUMMARY" "$TMP"
