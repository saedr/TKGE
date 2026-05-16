#!/usr/bin/env python3
import subprocess
from pathlib import Path

SEEDS=[11,22,33]
CONDS=['Original','Random','Structured-low','Structured-high']
REQ=['metrics.json','perturbation.json','runtime.json','training.json','top10_tail_detailed.json']
for budget in ['10','30']:
  root=Path(f'experiment/results/raw/full_grid/ComplEx/budget_{budget}')
  for s in SEEDS:
    for c in CONDS:
      d=root/f'seed_{s}'/c
      missing=[f for f in REQ if not (d/f).exists()]
      if not missing:
        print(f'SKIP budget={budget} seed={s} cond={c}')
        continue
      print(f'RUN budget={budget} seed={s} cond={c} missing={missing}')
      subprocess.run([
        'python','-m','experiment.src.run_condition',
        '--config','experiment/configs/smoke.yaml',
        '--condition',c,
        '--results-root','experiment/results/raw',
        '--seed',str(s),
        '--budget',f'0.{budget}',
        '--model','ComplEx',
        '--tag',f'full_grid/ComplEx/budget_{budget}/seed_{s}'
      ], check=True)
print('COMPLEX_CHUNKS_DONE')
