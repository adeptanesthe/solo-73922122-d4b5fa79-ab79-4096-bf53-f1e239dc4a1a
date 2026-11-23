#!/usr/bin/env bash
# Demo: pilot batches while commissioning a repaired fan (F3).  Run after
# demo.sh, or standalone -- this script onboards the factories itself.
set -euo pipefail

BASE="${BASE_URL:-http://localhost:8000}"
cd "$(dirname "$0")"

fmt() { if command -v jq >/dev/null 2>&1; then jq .; else cat; echo; fi; }
field() { python3 -c "import json,sys; print(json.load(sys.stdin)$1)"; }
post() { curl -sS -X POST "$BASE$1" -H 'Content-Type: application/json' --data @"$2"; }
get() { curl -sS "$BASE$1"; }
pins() {
  python3 -c '
import json, sys
j = json.load(sys.stdin)
r = j.get("report") or {}
print("   pins:", {k: j[k] for k in ("version", "correction", "offset")},
      "| stored report n:", r.get("n"))'
}

echo "=== 0. onboard 4 factories -> POST /versions"
CREATE=$(post /versions demo/create_version.json)
V1=$(echo "$CREATE" | field "['version']")
echo "-> current version: $V1 (F3 reported correction 0)"

echo
echo "=== 1. B1 opens for F3 before any report -> pinned offset = 0"
post /pilot/batches demo/pilot_open_b1.json | fmt

echo
echo "=== 2. B1 reports a trial-run summary -> MSE before/after calibration"
post /pilot/batches/F3/B1/reports demo/pilot_report_b1_first.json | fmt
echo "    offset was 0, so before == after; the summary is now committed"

echo
echo "=== 3. B2 opens now -> its offset is the mean residual of B1's"
echo "    committed summary (actual - shared prediction)"
post /pilot/batches demo/pilot_open_b2.json | fmt

echo
echo "=== 4. predict inside B2: pinned version + correction + offset"
post /predict demo/pilot_predict_b2.json | fmt

echo
echo "=== 5. B2 reports out of order (B1 is still the only other report):"
echo "    its own MSEs use B2's pins, not B1's"
post /pilot/batches/F3/B2/reports demo/pilot_report_b2.json | fmt

echo
echo "=== 6. B1 replaces its summary -> B1's MSEs and the factory totals"
echo "    change; B2 keeps the offset it was opened with"
post /pilot/batches/F3/B1/reports demo/pilot_report_b1_fixed.json | fmt
get /pilot/batches/F3/B1 | pins
get /pilot/batches/F3/B2 | pins

echo
echo "=== 7. a broken replacement is rejected -> HTTP 422, old summary kept"
echo "    (sum of squares too small: n*sum(x^2) < sum(x)^2)"
curl -sS -o /dev/null -w "HTTP %{http_code}\n" -X POST \
  "$BASE/pilot/batches/F3/B1/reports" -H 'Content-Type: application/json' \
  --data @demo/pilot_report_b2_broken.json
get /pilot/batches/F3/B1 | pins
echo "    (B1's stored n is still the replacement from step 6)"

echo
echo "=== 8. shared model re-published: F3 correction 0 -> 500"
NEW=$(post "/versions/$V1/correction" demo/correction_update.json)
V2=$(echo "$NEW" | field "['version']")
echo "-> current version: $V2 | F3 correction: $(echo "$NEW" | field "['corrections']['F3']")"

echo
echo "=== 9. B1 still predicts on version $V1 with its old correction/offset"
post /predict demo/pilot_predict_b1_after_release.json | fmt

echo
echo "=== 10. B3 opens after the re-publish -> pins version $V2 + new"
echo "     correction; offset computed from B1+B2 summaries under the new model"
post /pilot/batches demo/pilot_open_b3.json | fmt

echo
echo "=== 11. predict inside B3 (new model + new offset)"
post /predict demo/pilot_predict_b3.json | fmt

echo
echo "=== 12. re-open B3 -> original pins returned unchanged (idempotent)"
post /pilot/batches demo/pilot_open_b3.json | fmt
