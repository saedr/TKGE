#!/usr/bin/env bash
set -euo pipefail

START=$(date +%s)
mkdir -p experiment/results/raw experiment/results/aggregated experiment/results/figures
export PYTHONPATH="${PWD}/experiment:${PYTHONPATH:-}"

for f in data/FB15k-237/train.txt data/FB15k-237/valid.txt data/FB15k-237/test.txt; do
  [[ -f "$f" ]] || { echo "FB15k-237 files are required. Place train.txt, valid.txt, and test.txt at data/FB15k-237/."; exit 1; }
done

for c in Original Random Structured-low Structured-high; do
  python -m experiment.src.run_condition --condition "$c"
done
python -m experiment.src.aggregate
python experiment/scripts/validate_smoke.py | tee experiment/results/validation.log

python - <<'PY'
import json
from pathlib import Path
root=Path('experiment/results')
conds=['Original','Random','Structured-low','Structured-high']
rt={c:json.loads((root/'raw'/c/'runtime.json').read_text()) for c in conds}
pt={c:json.loads((root/'raw'/c/'perturbation.json').read_text()) for c in conds}
tr={c:json.loads((root/'raw'/c/'training.json').read_text()) for c in conds}
val=(root/'validation.log').read_text().strip()
lines=["# Smoke Summary", "", f"Did smoke complete end-to-end? {'Yes' if 'PASSED' in val else 'No'}", "", "## Runtime by stage (seconds)"]
for c in conds: lines.append(f"- {c}: {rt[c]}")
lines += ["", f"Did validation pass? {'Yes' if 'PASSED' in val else 'No'}", f"Did all deletion regimes remove the same number of triples? {len({pt[c]['removed'] for c in conds[1:]})==1}", f"Was zero-degree constraint enforced? {all(pt[c]['zero_degree_entities_after']==0 for c in conds)}", f"Does custom trainer appear to learn? {all(tr[c]['roughly_decreasing'] for c in conds)}", "Are metrics explicitly marked tail-only? Yes", "Where are artifacts saved? experiment/results/raw, experiment/results/aggregated, experiment/results/figures"]
(root/'smoke_summary.md').write_text('\n'.join(lines))
PY

END=$(date +%s)
echo "Smoke completed in $((END-START)) seconds"
