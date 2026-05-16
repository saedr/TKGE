#!/usr/bin/env python3
import subprocess
from pathlib import Path

SEEDS=[11,22,33]
CONDS=['Original','Random','Structured-low','Structured-high']
ROOT=Path('experiment/results/raw/full_grid/DistMult/budget_10')
REQ=['metrics.json','perturbation.json','runtime.json','training.json','top10_tail_detailed.json']

for s in SEEDS:
  for c in CONDS:
    d=ROOT/f'seed_{s}'/c
    missing=[f for f in REQ if not (d/f).exists()]
    if not missing:
      print(f'SKIP complete seed={s} condition={c}')
      continue
    print(f'RUN missing seed={s} condition={c}: {missing}')
    cmd=[
      'python','-m','experiment.src.run_condition',
      '--config','experiment/configs/smoke.yaml',
      '--condition',c,
      '--results-root','experiment/results/raw',
      '--seed',str(s),
      '--budget','0.10',
      '--model','DistMult',
      '--tag',f'full_grid/DistMult/budget_10/seed_{s}'
    ]
    subprocess.run(cmd, check=True)
print('RESUME_DONE')
