#!/usr/bin/env python3
"""Analyze frozen 10% replication with paired crossed hierarchical bootstrap."""
import argparse, csv, json
from itertools import combinations
from pathlib import Path
import numpy as np

DATASETS=["WN18RR","CoDEx-M"]
MODELS=["TransE","DistMult","ComplEx"]
CONDITIONS=["Original","Random","Structured-low","Relation-low"]
STRUCTURED=["Structured-low","Relation-low"]
TRAIN_SEEDS=[11,22,33]
DELETION_SEEDS=[101,202,303]
EXPECTED_RUNS=30
SLICES=["overall","coverage_low","coverage_mid","coverage_high","relation_low","relation_mid","relation_high"]


def ci(x):
    q=np.quantile(x,[.025,.5,.975]); return {"low":float(q[0]),"median":float(q[1]),"high":float(q[2])}
def supported(x): return x["low"]>0 or x["high"]<0
def write_csv(path,rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    if not rows: Path(path).write_text(""); return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def load_chunks(indir):
    chunks={}
    for p in Path(indir).glob("*.json"):
        if p.name.endswith(".mapping.json"): continue
        o=json.loads(p.read_text())
        k=(o.get("dataset"),o.get("model"))
        if k not in {(d,m) for d in DATASETS for m in MODELS}: continue
        if o.get("budget")!=0.10 or not o.get("complete") or len(o.get("runs",[]))!=EXPECTED_RUNS:
            raise RuntimeError(f"Invalid/incomplete 10pct chunk {p}")
        chunks[k]=o
    miss=[(d,m) for d in DATASETS for m in MODELS if (d,m) not in chunks]
    if miss: raise RuntimeError(f"Missing chunks {miss}")
    return chunks

def arrays(chunk):
    rm={(r["condition"],int(r["deletion_seed"]),int(r["train_seed"])):np.asarray(r["metrics"]["rr"],float) for r in chunk["runs"]}
    out={"Original":np.stack([[rm[("Original",0,s)] for s in TRAIN_SEEDS]])}
    for c in CONDITIONS[1:]: out[c]=np.stack([[rm[(c,d,s)] for s in TRAIN_SEEDS] for d in DELETION_SEEDS])
    return out

def slice_ids(chunk):
    ref=chunk["runs"][0]["metrics"]
    cov=np.asarray(ref["coverage_bins"]); rel=np.asarray(ref["relation_bins"])
    out={"overall":np.arange(len(cov))}
    for v in ["low","mid","high"]:
        out[f"coverage_{v}"]=np.flatnonzero(cov==v); out[f"relation_{v}"]=np.flatnonzero(rel==v)
    return out

def bootstrap(ds_arrays,qids,B=4000,seed=1701):
    rng=np.random.default_rng(seed); draws=np.empty((B,len(MODELS),len(CONDITIONS)))
    nq=len(qids)
    for b in range(B):
        q=rng.choice(qids,nq,replace=True); ts=rng.integers(0,3,3); ds=rng.integers(0,3,3)
        for mi,m in enumerate(MODELS):
            draws[b,mi,0]=ds_arrays[m]["Original"][np.ix_([0],ts,q)].mean()
            for cj,c in enumerate(CONDITIONS[1:],1): draws[b,mi,cj]=ds_arrays[m][c][np.ix_(ds,ts,q)].mean()
    return draws

def point(ds_arrays,qids):
    z=np.empty((len(MODELS),len(CONDITIONS)))
    for mi,m in enumerate(MODELS):
        for cj,c in enumerate(CONDITIONS): z[mi,cj]=ds_arrays[m][c][:,:,qids].mean()
    return z

def fci(x): return f"[{x['low']:.4f}, {x['high']:.4f}]"

def main(a):
    chunks=load_chunks(a.indir); means=[]; effects=[]; interactions=[]; reversals=[]; sanity={}
    for di,dataset in enumerate(DATASETS):
        da={m:arrays(chunks[(dataset,m)]) for m in MODELS}; ids=slice_ids(chunks[(dataset,MODELS[0])]); sanity[dataset]={}
        for m in MODELS:
            bad=sum(r["loss_end"]>=r["loss_start"] for r in chunks[(dataset,m)]["runs"]); sanity[dataset][m]={"non_decreasing_runs":bad,"total":30}
        for si,sname in enumerate(SLICES):
            qids=ids[sname]; dr=bootstrap(da,qids,4000,1701+di*100+si); pt=point(da,qids)
            for mi,m in enumerate(MODELS):
                for cj,c in enumerate(CONDITIONS):
                    x=ci(dr[:,mi,cj]); means.append({"dataset":dataset,"model":m,"condition":c,"slice":sname,"mean":float(pt[mi,cj]),"ci_low":x["low"],"ci_high":x["high"]})
                for c in STRUCTURED:
                    cj=CONDITIONS.index(c); rj=CONDITIONS.index("Random")
                    # Missingness effect E = condition-clean. Mechanism contrast M = E_structured-E_random = structured-random.
                    vals=dr[:,mi,cj]-dr[:,mi,rj]; x=ci(vals); pv=float(pt[mi,cj]-pt[mi,rj])
                    effects.append({"dataset":dataset,"model":m,"condition":c,"slice":sname,"mechanism_contrast":pv,"ci_low":x["low"],"ci_high":x["high"],"supported":supported(x)})
            for ma,mb in combinations(range(len(MODELS)),2):
                for c in STRUCTURED:
                    cj=CONDITIONS.index(c); rj=CONDITIONS.index("Random")
                    vals=(dr[:,ma,cj]-dr[:,ma,rj])-(dr[:,mb,cj]-dr[:,mb,rj]); x=ci(vals)
                    pv=(pt[ma,cj]-pt[ma,rj])-(pt[mb,cj]-pt[mb,rj])
                    interactions.append({"dataset":dataset,"model_a":MODELS[ma],"model_b":MODELS[mb],"condition":c,"slice":sname,"interaction":float(pv),"ci_low":x["low"],"ci_high":x["high"],"supported":supported(x)})
                # Strict reversals for Random vs each structured condition.
                rj=CONDITIONS.index("Random")
                for c in STRUCTURED:
                    cj=CONDITIONS.index(c); ga=dr[:,ma,rj]-dr[:,mb,rj]; gb=dr[:,ma,cj]-dr[:,mb,cj]; ca,cb=ci(ga),ci(gb)
                    rev=(ca["low"]>0 and cb["high"]<0) or (ca["high"]<0 and cb["low"]>0)
                    reversals.append({"dataset":dataset,"slice":sname,"model_a":MODELS[ma],"model_b":MODELS[mb],"condition_a":"Random","condition_b":c,"gap_a":float(pt[ma,rj]-pt[mb,rj]),"gap_a_ci_low":ca["low"],"gap_a_ci_high":ca["high"],"gap_b":float(pt[ma,cj]-pt[mb,cj]),"gap_b_ci_low":cb["low"],"gap_b_ci_high":cb["high"],"supported":bool(rev)})
    overall_rel=[x for x in interactions if x["slice"]=="overall" and x["condition"]=="Relation-low"]
    counts={d:sum(x["supported"] for x in overall_rel if x["dataset"]==d) for d in DATASETS}
    # Compare direction with 30% non-RotatE key contrasts from frozen report.
    prior={("WN18RR","TransE","DistMult"):0.050259,("WN18RR","TransE","ComplEx"):0.039010,("CoDEx-M","TransE","DistMult"):-0.014305,("CoDEx-M","TransE","ComplEx"):-0.018516}
    comp=[]
    for x in overall_rel:
        k=(x["dataset"],x["model_a"],x["model_b"])
        if k in prior:
            comp.append({**x,"interaction_30pct":prior[k],"same_direction":bool(np.sign(x["interaction"])==np.sign(prior[k]))})
    both=all(counts[d]>0 for d in DATASETS)
    compat={d:any(x["same_direction"] for x in comp if x["dataset"]==d) for d in DATASETS}
    if both: label="REPLICATES"
    elif any(counts.values()) and all(compat.values()): label="PARTIAL REPLICATION"
    elif not any(counts.values()) and all(any(x["same_direction"] for x in comp if x["dataset"]==d) for d in DATASETS): label="SEVERITY-DEPENDENT"
    else: label="FAILS TO REPLICATE"
    summary={"analysis":{"budget":0.10,"bootstrap_replications":4000,"bootstrap_seed_base":1701,"models":MODELS,"units":["deletion realization","training seed","test query within slice"]},"training_sanity":sanity,"mean_mrr":means,"mechanism_contrasts":effects,"interactions":interactions,"reversals":reversals,"comparison_30pct":comp,"decision":{"label":label,"supported_relation_low_overall":counts,"supported_reversals":sum(x["supported"] for x in reversals)}}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(summary,indent=2))
    td=Path(a.tables_dir); td.mkdir(parents=True,exist_ok=True); write_csv(td/"mean_mrr.csv",means); write_csv(td/"mechanism_contrasts.csv",effects); write_csv(td/"interactions.csv",interactions); write_csv(td/"reversals.csv",reversals)
    lines=["# 10% structured-missingness replication",f"\n## Decision: **{label}**",f"\nSupported overall Relation-low interactions: WN18RR {counts['WN18RR']}/3; CoDEx-M {counts['CoDEx-M']}/3.",f"Supported strict reversals across all slices: {summary['decision']['supported_reversals']}.","\n## Key Relation-low interactions vs 30%","\n| Dataset | Pair | 10% interaction | 95% CI | 30% interaction | Same direction |","|---|---|---:|---|---:|---|"]
    for x in comp: lines.append(f"| {x['dataset']} | {x['model_a']}−{x['model_b']} | {x['interaction']:.4f} | [{x['ci_low']:.4f}, {x['ci_high']:.4f}] | {x['interaction_30pct']:.4f} | {x['same_direction']} |")
    lines += ["\n## Frozen interpretation","The generic quantity is performance change / missingness effect. The mechanism contrast is structured minus random, so positive values mean structured missingness yields higher MRR than random and negative values mean lower MRR than random. The model × mechanism interaction is the difference in this contrast between model pairs.","\nThe 10% run uses the same datasets, seeds, training settings, filtering, fixed original-graph slices, and hierarchical resampling as the 30% confirmatory study, with RotatE excluded a priori because its WN18RR 30% training failed the sanity check."]
    Path(a.report).write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--indir",required=True); p.add_argument("--out",required=True); p.add_argument("--report",required=True); p.add_argument("--tables-dir",required=True); main(p.parse_args())
