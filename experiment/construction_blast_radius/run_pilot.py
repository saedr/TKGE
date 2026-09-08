#!/usr/bin/env python3
import argparse, csv, json, math, random, re, subprocess
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

SEED = 20260908
BOOTSTRAPS = 2000
RANDOM_RANKINGS = 1000
DATASETS = ["DB-YG-15K", "DB-WD-15K"]

class UnionFind:
    def __init__(self):
        self.parent, self.rank = {}, {}
    def add(self, x):
        if x not in self.parent:
            self.parent[x], self.rank[x] = x, 0
    def find(self, x):
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return
        if self.rank[a] < self.rank[b]: a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]: self.rank[a] += 1

def read_pairs(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                a, b = line.split("\t")[:2]
                out.append((a, b))
    return out

def read_triples(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                yield parts[0], parts[1], parts[2]

def find_dataset_dir(root, name):
    hits = [p for p in Path(root).rglob(name) if p.is_dir() and p.parent.name == "RealEA" and (p / "rel_triples_1").exists()]
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one RealEA/{name}; found {hits}")
    return hits[0]

def first_fold(ds):
    folds = sorted(p.name for p in (ds / "721_5folds").iterdir() if p.is_dir())
    if not folds: raise RuntimeError(f"No folds in {ds}/721_5folds")
    return folds[0]

def turn_yg(triples, triple_type="rel"):
    prefixes = {"dbp":"http://dbpedia.org/ontology/", "owl":"http://www.w3.org/2002/07/owl#", "rdf":"http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdfs":"http://www.w3.org/2000/01/rdf-schema#", "skos":"http://www.w3.org/2004/02/skos/core#", "xsd":"http://www.w3.org/2001/XMLSchema#"}
    base = "http://yago-knowledge.org/resource/"
    out = set()
    for s, p, o in triples:
        s, p, o = s.strip("<>"), p.strip("<>"), o.strip("<>")
        s = base + s
        if ":" in p and "EntityMatchers" not in p:
            pref, rest = p.split(":", 1); p = prefixes[pref] + rest
        elif "EntityMatchers" not in p:
            p = base + p
        if triple_type == "rel": o = base + o
        out.add((s, p, o))
    return out

def write_nt(ds, fold, out1, out2):
    rel1, attr1 = set(read_triples(ds/"rel_triples_1")), set(read_triples(ds/"attr_triples_1"))
    rel2, attr2 = set(read_triples(ds/"rel_triples_2")), set(read_triples(ds/"attr_triples_2"))
    if "YG" in ds.name:
        rel2, attr2 = turn_yg(rel2, "rel"), turn_yg(attr2, "attr")
    seed1, seed2 = [], []
    for e1, e2 in read_pairs(ds/"721_5folds"/fold/"train_links"):
        label = e1.split("/")[-1]
        seed1.append((e1, "EntityMatchers:label", f'"{label}"'))
        seed2.append((e2, "EntityMatchers:label", f'"{label}"'))
    if "YG" in ds.name: seed2 = list(turn_yg(seed2, "attr"))
    def write(path, rel, attr, seeds):
        with open(path, "w", encoding="utf-8") as f:
            for s,p,o in rel: f.write(f"<{s}> <{p}> <{o}> .\n")
            for s,p,o in attr:
                if not o.startswith('"'): o = '"' + o + '"'
                f.write(f"<{s}> <{p}> {o} .\n")
            for s,p,o in seeds: f.write(f"<{s}> <{p}> {o} .\n")
    write(out1, rel1, attr1, seed1); write(out2, rel2, attr2, seed2)

def run_paris(ds, fold, paris_jar, workdir):
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    kg1, kg2 = workdir/"kg1.nt", workdir/"kg2.nt"; write_nt(ds, fold, kg1, kg2)
    out = workdir/"paris"; (out/"output").mkdir(parents=True); (out/"log").mkdir()
    ini = out/"paris.ini"
    ini.write_text(f"resultTSV = {out.resolve()}/output\nfactstore1 = {kg1.resolve()}\nfactstore2 = {kg2.resolve()}\nhome = {out.resolve()}/log\n")
    cp = subprocess.run(["java","-Xmx6g","-jar",str(Path(paris_jar).resolve()),str(ini.resolve())], cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1200)
    (workdir/"paris_stdout.log").write_text(cp.stdout)
    if cp.returncode != 0: raise RuntimeError(f"PARIS failed rc={cp.returncode}")
    eqvs = []
    for p in (out/"output").glob("*_eqv.tsv"):
        m = re.match(r"(\d+)_eqv\.tsv$", p.name)
        if m and p.stat().st_size > 0: eqvs.append((int(m.group(1)), p))
    if not eqvs: raise RuntimeError("No nonempty PARIS equivalence output")
    return max(eqvs)[1]

def normalize_entity(x):
    if "dbp:" in x: x = x.replace("dbp:", "http://dbpedia.org/")
    if "y2:" in x: x = x.replace("y2:", "")
    return x

def parse_predictions(eqv, train):
    train, rows = set(train), {}
    with open(eqv, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            pair = (normalize_entity(parts[0]), normalize_entity(parts[1]))
            if pair in train: continue
            conf = None
            if len(parts) >= 3:
                try:
                    z = float(parts[2]); conf = z if math.isfinite(z) else None
                except Exception: pass
            if pair not in rows or (conf is not None and (rows[pair] is None or conf > rows[pair])): rows[pair] = conf
    return rows

def build_clean_graph(ds, gold):
    uf = UnionFind()
    for a,b in gold: uf.union(("1",a),("2",b))
    triples = []
    for src, fn in [("1","rel_triples_1"),("2","rel_triples_2")]:
        for s,r,t in read_triples(ds/fn):
            uf.add((src,s)); uf.add((src,t)); triples.append((src,s,r,t))
    groups = defaultdict(list)
    for x in list(uf.parent): groups[uf.find(x)].append(x)
    canon = {}
    for members in groups.values():
        label = "||".join(f"{src}:{ent}" for src,ent in sorted(members))
        for x in members: canon[x] = label
    edges = set()
    for src,s,r,t in triples:
        a,b = canon[(src,s)],canon[(src,t)]
        edges.add((a,f"kg{src}::{r}",b)); edges.add((b,f"kg{src}::INV::{r}",a))
    out, inc, und, out_rel, in_rel = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    for s,r,t in edges:
        out[s].add((r,t)); inc[t].add((r,s)); und[s].add(t); und[t].add(s); out_rel[(s,r)].add(t); in_rel[(r,t)].add(s)
    return canon, edges, out, inc, und, out_rel, in_rel

def radius2_size(node, und):
    seen, q = {node}, deque([(node,0)])
    while q:
        x,d = q.popleft()
        if d == 2: continue
        for y in und.get(x,()):
            if y not in seen: seen.add(y); q.append((y,d+1))
    return len(seen)-1

def clean_path_exists(s,r1,r2,t,out_rel,in_rel):
    a,b = out_rel.get((s,r1)), in_rel.get((r2,t))
    return bool(a and b and not a.isdisjoint(b))

def damage_for(u,v,out,inc,out_rel,in_rel):
    touched, cand = {u,v}, set()
    for r1,s in inc.get(u,()):
        if s in touched: continue
        for r2,t in out.get(v,()):
            if t not in touched: cand.add((s,r1,r2,t))
    for r1,s in inc.get(v,()):
        if s in touched: continue
        for r2,t in out.get(u,()):
            if t not in touched: cand.add((s,r1,r2,t))
    return sum(not clean_path_exists(*q,out_rel,in_rel) for q in cand)

def c20(vals):
    vals = np.asarray(vals,dtype=float); total = float(vals.sum())
    if not len(vals) or total <= 0: return 0.0
    return float(np.sort(vals)[::-1][:math.ceil(.2*len(vals))].sum()/total)

def bootstrap_c20(vals):
    vals=np.asarray(vals,dtype=float); rng=np.random.default_rng(SEED); n=len(vals)
    arr=[c20(vals[rng.integers(0,n,size=n)]) for _ in range(BOOTSTRAPS)]
    return float(np.quantile(arr,.025)), float(np.quantile(arr,.975))

def capture(vals,scores,keys):
    total=sum(vals)
    if total<=0:return 0.0
    order=sorted(range(len(vals)),key=lambda i:(-scores[i],keys[i])); k=math.ceil(.2*len(vals))
    return sum(vals[i] for i in order[:k])/total

def random_capture(vals):
    total=sum(vals)
    if total<=0:return {"mean":0,"median":0,"lo":0,"hi":0}
    rng=random.Random(SEED); idx=list(range(len(vals))); k=math.ceil(.2*len(vals)); x=[]
    for _ in range(RANDOM_RANKINGS): x.append(sum(vals[i] for i in rng.sample(idx,k))/total)
    return {"mean":float(np.mean(x)),"median":float(np.median(x)),"lo":float(np.quantile(x,.025)),"hi":float(np.quantile(x,.975))}

def analyze_dataset(ds,paris_jar,outdir):
    fold=first_fold(ds); fd=ds/"721_5folds"/fold
    train,valid,test=read_pairs(fd/"train_links"),read_pairs(fd/"valid_links"),read_pairs(fd/"test_links")
    gold=train+valid+test; gold_set=set(gold)
    preds=parse_predictions(run_paris(ds,fold,paris_jar,Path(outdir)/"paris_run"),train)
    pred_set=set(preds); eval_gold=set(valid+test); tp=len(pred_set & eval_gold)
    precision=tp/len(pred_set) if pred_set else 0; recall=tp/len(eval_gold) if eval_gold else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
    false=sorted(p for p in pred_set if p not in gold_set)
    canon,edges,out,inc,und,out_rel,in_rel=build_clean_graph(ds,gold)
    g=nx.DiGraph(); g.add_nodes_from(canon.values()); g.add_edges_from((s,t) for s,_,t in edges); pr=nx.pagerank(g,alpha=.85,max_iter=200)
    r2,rows={},[]
    for j,(a,b) in enumerate(false,1):
        if ("1",a) not in canon or ("2",b) not in canon: continue
        u,v=canon[("1",a)],canon[("2",b)]
        if u==v: continue
        du,dv=len(out.get(u,())),len(out.get(v,()))
        if u not in r2:r2[u]=radius2_size(u,und)
        if v not in r2:r2[v]=radius2_size(v,und)
        rows.append({"entity1":a,"entity2":b,"confidence":preds[(a,b)],"degree_u":du,"degree_v":dv,"degree_sum":du+dv,"degree_product":(du+1)*(dv+1),"pagerank_sum":pr.get(u,0)+pr.get(v,0),"radius2_sum":r2[u]+r2[v],"damage":damage_for(u,v,out,inc,out_rel,in_rel)})
        if j%100==0: print(ds.name,"damage",j,"/",len(false),flush=True)
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    fields=["entity1","entity2","confidence","degree_u","degree_v","degree_sum","degree_product","pagerank_sum","radius2_sum","damage"]
    with open(outdir/"per_error.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    vals=[r["damage"] for r in rows]; keys=[(r["entity1"],r["entity2"]) for r in rows]; C=c20(vals); lo,hi=bootstrap_c20(vals) if vals else (0,0); total=sum(vals); maxshare=max(vals)/total if vals and total else 0
    baselines={}
    for name in ["degree_sum","degree_product","pagerank_sum","radius2_sum"]:
        scores=[float(r[name]) for r in rows]; rho=float(spearmanr(vals,scores).statistic) if len(rows)>=2 else float("nan")
        baselines[name]={"capture20":capture(vals,scores,keys) if rows else 0,"spearman":rho if math.isfinite(rho) else None}
    conf_ok=bool(rows) and all(r["confidence"] is not None and math.isfinite(float(r["confidence"])) for r in rows)
    if conf_ok:
        scores=[1-float(r["confidence"]) for r in rows]; rho=float(spearmanr(vals,scores).statistic) if len(rows)>=2 else float("nan")
        baselines["paris_uncertainty"]={"capture20":capture(vals,scores,keys),"spearman":rho if math.isfinite(rho) else None}
    baselines["random"]=random_capture(vals)
    best=max((v["capture20"] for k,v in baselines.items() if k!="random" and "capture20" in v),default=0); best_ratio=best/C if C>0 else 1.0
    summary={"dataset":ds.name,"fold":fold,"num_predictions_after_train_removal":len(pred_set),"num_false_merges":len(rows),"paris_precision":precision,"paris_recall":recall,"paris_f1":f1,"total_damage":total,"zero_damage_fraction":sum(d==0 for d in vals)/len(vals) if vals else 1.0,"c20":C,"c20_bootstrap_95":[lo,hi],"max_single_share":maxshare,"best_simple_ratio":best_ratio,"confidence_available":conf_ok,"baselines":baselines,"graph_nodes":g.number_of_nodes(),"graph_labeled_edges":len(edges)}
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)); return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); ap.add_argument("--paris-jar",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); summaries=[]
    for name in DATASETS: summaries.append(analyze_dataset(find_dataset_dir(a.data_root,name),a.paris_jar,out/name))
    feasible=all(s["num_false_merges"]>=100 for s in summaries)
    concentrated=feasible and all(s["total_damage"]>0 and s["c20"]>=.60 and s["c20_bootstrap_95"][0]>=.50 and s["max_single_share"]<=.25 for s in summaries)
    decision="KILL-BENCHMARK" if not feasible else "KILL-PHENOMENON" if not concentrated else "STOP-METHOD" if any(s["best_simple_ratio"]>=.90 for s in summaries) else "CONTINUE"
    result={"decision":decision,"frozen_thresholds":{"min_false_merges":100,"c20":.60,"bootstrap_lower":.50,"max_single_share":.25,"simple_ratio":.90},"datasets":summaries}
    (out/"RESULT.json").write_text(json.dumps(result,indent=2,sort_keys=True)); print("FINAL_DECISION="+decision); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
