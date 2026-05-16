#!/usr/bin/env python3
import itertools, json
from pathlib import Path

ROOT=Path('experiment/results/raw/pilot_distmult_30pct')
SEEDS=[11,22,33]
CONDS=['Original','Random','Structured-low','Structured-high']


def loadj(p):
    return json.loads(Path(p).read_text())


def main():
    for s in SEEDS:
        for c in CONDS:
            d=ROOT/f'seed_{s}'/c
            for f in ['metrics.json','perturbation.json','training.json','runtime.json','top10_tail_ids.npy','top10_tail_detailed.json']:
                assert (d/f).exists(), f'missing {d/f}'
            m=loadj(d/'metrics.json')
            assert m['evaluation_kind']=='tail_only'
            assert m['shared_filtering']=='original_train_valid_test'
            p=loadj(d/'perturbation.json')
            assert p['zero_degree_entities_after']==0
            td=loadj(d/'top10_tail_detailed.json')
            assert td['model']=='DistMult' and abs(td['budget']-0.30)<1e-9 and td['condition']==c and td['seed']==s
            rows=td['rows']
            assert rows and all(k in rows[0] for k in ['query_idx','coverage_bin','relation_frequency_bin','top10_tail_ids'])

    for s in SEEDS:
        rem=[loadj(ROOT/f'seed_{s}'/c/'perturbation.json')['removed'] for c in CONDS[1:]]
        assert len(set(rem))==1, f'deletion mismatch seed {s}: {rem}'

    # stability scope + bin split sanity
    for c in CONDS:
        rows_by_seed=[]
        for s in SEEDS:
            td=loadj(ROOT/f'seed_{s}'/c/'top10_tail_detailed.json')['rows']
            rows_by_seed.append({r['query_idx']:r for r in td})
        qids=set(rows_by_seed[0].keys())
        assert all(set(r.keys())==qids for r in rows_by_seed)
        # verify bins are subsets and typically differ from overall pool
        cov_bins={b:{qid for qid in qids if rows_by_seed[0][qid]['coverage_bin']==b} for b in ['low','mid','high']}
        rel_bins={b:{qid for qid in qids if rows_by_seed[0][qid]['relation_frequency_bin']==b} for b in ['low-frequency','mid-frequency','high-frequency']}
        assert all(len(v)>0 for v in cov_bins.values())
        assert all(len(v)>0 for v in rel_bins.values())
        assert any(cov_bins[b]!=qids for b in cov_bins), f'coverage bins not separated for {c}'
        assert any(rel_bins[b]!=qids for b in rel_bins), f'relation bins not separated for {c}'

    print('PILOT VALIDATION PASSED')

if __name__=='__main__':
    main()
