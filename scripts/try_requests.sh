#!/usr/bin/env bash
# Send every request in examples/requests/ to the running server, check and summarize.
#   ./serve.sh start qwen3-vl-8b 4bit
#   scripts/try_requests.sh                 # all files
#   scripts/try_requests.sh 03 05           # only files starting with 03 or 05
# Env: URL (default http://localhost:8010)
cd "$(dirname "$0")/.." || exit 1
URL=${URL:-http://localhost:8010}
files=()
if [ $# -eq 0 ]; then files=(examples/requests/*.json)
else for p in "$@"; do files+=(examples/requests/"$p"*.json); done; fi
SUMMARY=$(mktemp)
echo "server: $(curl -s "$URL/health")"
for f in "${files[@]}"; do
  echo "================ $(basename "$f")"
  curl -s -X POST "$URL/plan" -H 'Content-Type: application/json' --data-binary "@$f" |
    python3 scripts/check_response.py "$f" "$SUMMARY"
done
python3 - "$SUMMARY" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
if rows:
    ok = sum(r["pass"] for r in rows)
    g = [r["goal_s"] for r in rows if r.get("goal_s")]
    print(f"\n==== {ok}/{len(rows)} passed | mean model time {sum(g)/max(len(g),1):.2f} s")
    for r in rows:
        if not r["pass"]:
            print("  FAIL", r["file"].split("/")[-1], ":", "; ".join(r["fails"]))
PY
rm -f "$SUMMARY"
