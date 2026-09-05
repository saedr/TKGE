#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(indir):
    data = []
    for p in sorted(Path(indir).glob("*.json")):
        if p.name.endswith("mapping.json"):
            continue
        obj = json.load(open(p, "r", encoding="utf-8"))
        for run in obj["runs"]:
            data.append({
                "dataset": obj["dataset"], "model": obj["model"],
                "condition": run["condition"], "deletion_seed": run["deletion_seed"],
                "train_seed": run["train_seed"], "metrics": run["metrics"]
            })
    return data


def mean_metric(rows, key):
    vals=[r["metrics"][key] for r in rows if r["metrics"].get(key) is not None]
    return float(np.mean(vals)) if vals else None


def paired_bootstrap(rows, dataset, model_a, model_b, cond_a, cond_b, key, B=4000, seed=17):
    rng=np.random.default_rng(seed)
    idx={(r['model'],r['condition'],r['deletion_seed'],r['train_seed']):r for r in rows if r['dataset']==dataset}
    del_seeds=sorted({r['deletion_seed'] for r in rows if r['dataset']==dataset and r['condition']==cond_a})
    if cond_a=='Original': del_seeds=[0]
    train_seeds=sorted({r['train_seed'] for r in rows if r['dataset']==dataset})
    def get(model, cond, d, s):
        dd=0 if cond=='Original' else d
        return idx[(model,cond,dd,s)]['metrics'][key]
    # Difference in model gap across conditions: (A-B)_cond_b - (A-B)_cond_a
    draws=[]
    for _ in range(B):
        ss=rng.choice(train_seeds, size=len(train_seeds), replace=True)
        da=rng.choice(del_seeds, size=len(del_seeds), replace=True) if cond_a!='Original' else np.array([0])
        db=rng.choice(sorted({r['deletion_seed'] for r in rows if r['dataset']==dataset and r['condition']==cond_b}), size=len(del_seeds), replace=True)
        ga=[]; gb=[]
        for s in ss:
            for d in da:
                ga.append(get(model_a,cond_a,int(d),int(s))-get(model_b,cond_a,int(d),int(s)))
            for d in db:
                gb.append(get(model_a,cond_b,int(d),int(s))-get(model_b,cond_b,int(d),int(s)))
        draws.append(np.mean(gb)-np.mean(ga))
    q=np.quantile(draws,[.025,.5,.975])
    return [float(x) for x in q]


def main(args):
    rows=load_rows(args.indir)
    datasets=sorted({r['dataset'] for r in rows})
    models=sorted({r['model'] for r in rows})
    conditions=['Original','Random','Structured-low','Relation-low']
    keys=['overall_mrr','low_coverage_mrr','low_relation_mrr']
    summary={"datasets":datasets,"models":models,"conditions":conditions,"means":{},"interactions":[],"supported_reversals":[]}
    for ds in datasets:
        summary['means'][ds]={}
        for m in models:
            summary['means'][ds][m]={}
            base=[r for r in rows if r['dataset']==ds and r['model']==m and r['condition']=='Original']
            for c in conditions:
                rr=[r for r in rows if r['dataset']==ds and r['model']==m and r['condition']==c]
                summary['means'][ds][m][c]={k:mean_metric(rr,k) for k in keys}
                summary['means'][ds][m][c]['n_runs']=len(rr)
            # differential degradation vs random
            for c in ['Structured-low','Relation-low']:
                for k in keys:
                    b=mean_metric(base,k); r=mean_metric([x for x in rows if x['dataset']==ds and x['model']==m and x['condition']=='Random'],k); s=mean_metric([x for x in rows if x['dataset']==ds and x['model']==m and x['condition']==c],k)
                    summary['means'][ds][m][c][f'D_vs_random_{k}']=(b-s)-(b-r) if None not in (b,r,s) else None
        for i,a in enumerate(models):
            for b in models[i+1:]:
                for c in ['Structured-low','Relation-low']:
                    for k in keys:
                        ci=paired_bootstrap(rows,ds,a,b,'Random',c,k)
                        summary['interactions'].append({'dataset':ds,'model_a':a,'model_b':b,'structured_condition':c,'metric':k,'diff_in_model_gap_ci95':ci,'supported':ci[0]>0 or ci[2]<0})
                        # supported reversal uses condition-specific model gaps and conservative normal approximation across run pairs
                        ar=mean_metric([x for x in rows if x['dataset']==ds and x['model']==a and x['condition']=='Random'],k); br=mean_metric([x for x in rows if x['dataset']==ds and x['model']==b and x['condition']=='Random'],k)
                        ac=mean_metric([x for x in rows if x['dataset']==ds and x['model']==a and x['condition']==c],k); bc=mean_metric([x for x in rows if x['dataset']==ds and x['model']==b and x['condition']==c],k)
                        if None not in (ar,br,ac,bc) and np.sign(ar-br)*np.sign(ac-bc)<0 and (ci[0]>0 or ci[2]<0):
                            summary['supported_reversals'].append({'dataset':ds,'model_a':a,'model_b':b,'condition':c,'metric':k,'random_gap':ar-br,'structured_gap':ac-bc,'gap_change_ci95':ci})
    # Frozen continuation gate
    supported_by_ds=defaultdict(int)
    for x in summary['interactions']:
        if x['supported']:
            supported_by_ds[x['dataset']]+=1
    summary['gate']={
        'supported_interactions_by_dataset':dict(supported_by_ds),
        'num_supported_reversals':len(summary['supported_reversals']),
        'continue':sum(v>0 for v in supported_by_ds.values())>=1,
        'strong_paper_signal':len(summary['supported_reversals'])>0,
        'note':'Primary gate: at least one supported model×mechanism interaction on a new dataset; stronger signal requires uncertainty-supported reversal.'
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary,open(args.out,'w',encoding='utf-8'),indent=2)
    print(json.dumps(summary['gate'],indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--indir',required=True); ap.add_argument('--out',required=True); main(ap.parse_args())
