#!/usr/bin/env python3
"""Frozen FactGate sparsity phenomenon pilot on tau-bench airline trajectories.

The pilot asks whether a small subset of structured tool-return facts controls
the next consequential tool decision. It uses successful historical GPT-4o
airline trajectories only as states/tasks; Qwen3-1.7B is the fixed probe.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, re, urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCE_URL = "https://raw.githubusercontent.com/sierra-research/tau-bench/59a200c6d575d595120f1cb70fea53cef0632f6b/historical_trajectories/gpt-4o-airline.json"
MODEL = "Qwen/Qwen3-1.7B"
SEED = 42
MAX_CANDIDATES = 60
MAX_FACTS = 12
N_ALTS = 2
N_SHARDS = 5
MIN_BASELINE_CORRECT = 20
MIN_SENSITIVE = 10
SPARSITY_GATE = 0.20

WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}
ALL_TOOLS = [
    "book_reservation","calculate","cancel_reservation","get_reservation_details",
    "get_user_details","list_all_airports","search_direct_flight",
    "search_onestop_flight","send_certificate","think",
    "transfer_to_human_agents","update_reservation_baggages",
    "update_reservation_flights","update_reservation_passengers",
]

def download_source(path: Path):
    if not path.exists():
        print("Downloading pinned tau-bench historical trajectories", flush=True)
        urllib.request.urlretrieve(SOURCE_URL, path)

def parse_tool_json(msg):
    try:
        return json.loads(msg.get("content") or "")
    except Exception:
        return None

def flatten(obj, prefix=()):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten(v, prefix + (str(k),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(flatten(v, prefix + (str(i),)))
    elif isinstance(obj, (str, int, float, bool)) and obj is not None:
        if isinstance(obj, str) and len(obj) > 120:
            return []
        out.append((prefix, obj))
    return out

def value_type(v):
    if isinstance(v, bool): return "bool"
    if isinstance(v, int) and not isinstance(v, bool): return "int"
    if isinstance(v, float): return "float"
    return "str"

def set_path(obj, path, value):
    if not path:
        raise ValueError("Cannot set an empty JSON path")
    cur = obj
    for p in path[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    last = path[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value

def assistant_tool_name(msg):
    calls = msg.get("tool_calls") or []
    if len(calls) != 1:
        return None
    f = calls[0].get("function") or {}
    return f.get("name")

def fact_key(tool_name, path):
    semantic = tuple(p for p in path if not p.isdigit())
    return (tool_name, semantic)

def build_pool(data):
    pool = defaultdict(list)
    for ep in data:
        for msg in ep.get("traj", []):
            if msg.get("role") != "tool":
                continue
            obj = parse_tool_json(msg)
            if obj is None:
                continue
            tname = str(msg.get("name") or "tool")
            for path, v in flatten(obj):
                if not path:
                    continue
                key = (fact_key(tname, path), value_type(v))
                if v not in pool[key]:
                    pool[key].append(v)
    return pool

def donor_values(pool, tool_name, path, original):
    vals = pool.get((fact_key(tool_name, path), value_type(original)), [])
    alts = [v for v in vals if v != original]
    alts.sort(key=lambda v: hashlib.sha256((repr(v)+"|42").encode()).hexdigest())
    return alts[:N_ALTS]

def candidate_from_call(ep, call_idx, msg, pool):
    target = assistant_tool_name(msg)
    if target not in WRITE_TOOLS:
        return None

    prior = ep.get("traj", [])[:call_idx]
    tool_positions = [i for i, m in enumerate(prior) if m.get("role") == "tool" and parse_tool_json(m) is not None]
    if not tool_positions:
        return None
    tool_positions = tool_positions[-2:]

    facts = []
    for pos in reversed(tool_positions):
        tm = prior[pos]
        obj = parse_tool_json(tm)
        tname = str(tm.get("name") or "tool")
        for path, v in flatten(obj):
            if not path:
                continue
            alts = donor_values(pool, tname, path, v)
            if len(alts) >= N_ALTS:
                facts.append({
                    "msg_pos": pos, "tool_name": tname, "path": list(path),
                    "original": v, "alts": alts,
                    "label": f"{tname}." + ".".join(path),
                })
    facts = facts[:MAX_FACTS]
    if len(facts) < 3:
        return None

    system = next((m.get("content","") for m in ep.get("traj", []) if m.get("role")=="system"), "")
    return {
        "task_id": ep.get("task_id"),
        "call_idx": call_idx,
        "target_tool": target,
        "system": system,
        "prior": copy.deepcopy(prior),
        "facts": facts,
    }

def prepare_candidates(data):
    pool = build_pool(data)
    cands = []
    for ep in data:
        try:
            reward = float(ep.get("reward", 0))
        except Exception:
            reward = 0
        if reward < 0.999:
            continue
        for i, msg in enumerate(ep.get("traj", [])):
            if msg.get("role") != "assistant":
                continue
            c = candidate_from_call(ep, i, msg, pool)
            if c:
                cands.append(c)
    cands.sort(key=lambda c: hashlib.sha256(f"{c['task_id']}|{c['call_idx']}|42".encode()).hexdigest())
    return cands[:MAX_CANDIDATES]

def render_context(cand, perturb=None):
    prior = copy.deepcopy(cand["prior"])
    if perturb is not None:
        fact, alt = perturb
        msg = prior[fact["msg_pos"]]
        obj = parse_tool_json(msg)
        set_path(obj, fact["path"], alt)
        msg["content"] = json.dumps(obj, ensure_ascii=False, separators=(",",":"))

    system = cand["system"][:6500]
    recent = prior[-10:]
    rows = []
    for m in recent:
        role = m.get("role","")
        if role == "tool":
            rows.append(f"TOOL RESULT [{m.get('name','tool')}]: {m.get('content','')}")
        elif role in ("user","assistant"):
            if m.get("content"):
                rows.append(f"{role.upper()}: {m.get('content')}")
            elif role == "assistant" and m.get("tool_calls"):
                nm = assistant_tool_name(m)
                if nm:
                    rows.append(f"ASSISTANT TOOL CALL: {nm}")
    return (
        "You are predicting the next tool call of an airline support agent.\n"
        "Use only the policy and conversation state below. Return exactly ONE tool name "
        "from the allowed list and nothing else.\n\n"
        f"ALLOWED TOOLS: {', '.join(ALL_TOOLS)}\n\n"
        f"POLICY:\n{system}\n\nRECENT STATE:\n" + "\n".join(rows) + "\n\nNEXT TOOL:"
    )

def norm_tool(text):
    s = str(text).strip().lower()
    if "</think>" in s:
        s = s.split("</think>",1)[1].strip()
    for t in ALL_TOOLS:
        if re.search(rf"\b{re.escape(t)}\b", s):
            return t
    s = re.sub(r"[^a-z0-9_]+","",s)
    return s

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok, model

def infer(tok, model, prompts, batch_size=4):
    outs = []
    for st in range(0, len(prompts), batch_size):
        chunk = prompts[st:st+batch_size]
        rendered = []
        for p in chunk:
            msgs = [{"role":"user","content":p}]
            try:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "\n/no_think"
            rendered.append(txt)
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=2300)
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=12, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
            )
        width = enc["input_ids"].shape[1]
        for i in range(len(chunk)):
            outs.append(tok.decode(gen[i, width:], skip_special_tokens=True).strip())
        if min(st+batch_size,len(prompts)) % 20 == 0 or st+batch_size >= len(prompts):
            print(f"INFER {min(st+batch_size,len(prompts))}/{len(prompts)}", flush=True)
    return outs

def run_shard(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    src = outdir / "gpt-4o-airline.json"
    download_source(src)
    data = json.loads(src.read_text(encoding="utf-8"))
    cands = prepare_candidates(data)
    print(f"CANDIDATES total={len(cands)} shard={args.shard}/{args.n_shards}", flush=True)
    subset = [(i,c) for i,c in enumerate(cands) if i % args.n_shards == args.shard]
    tok, model = load_model()

    baseline_prompts = [render_context(c) for _,c in subset]
    baseline_raw = infer(tok, model, baseline_prompts)
    rows = []
    for (idx,c), raw in zip(subset, baseline_raw):
        pred = norm_tool(raw)
        row = {
            "index": idx, "task_id": c["task_id"], "call_idx": c["call_idx"],
            "target_tool": c["target_tool"], "baseline_raw": raw,
            "baseline_pred": pred, "baseline_correct": pred == c["target_tool"],
            "n_facts": len(c["facts"]), "facts": [],
        }
        if row["baseline_correct"]:
            pp, meta = [], []
            for fi, fact in enumerate(c["facts"]):
                for ai, alt in enumerate(fact["alts"][:N_ALTS]):
                    pp.append(render_context(c, (fact, alt)))
                    meta.append((fi, ai, fact, alt))
            preds = infer(tok, model, pp) if pp else []
            fact_res = [{"label":f["label"],"path":f["path"],"tool_name":f["tool_name"],
                         "original":f["original"],"preds":[], "alts":[]} for f in c["facts"]]
            for raw2,(fi,ai,fact,alt) in zip(preds, meta):
                p2 = norm_tool(raw2)
                fact_res[fi]["preds"].append(p2)
                fact_res[fi]["alts"].append(alt)
            for fr in fact_res:
                fr["influence"] = float(np.mean([p != c["target_tool"] for p in fr["preds"]])) if fr["preds"] else 0.0
            row["facts"] = fact_res
        rows.append(row)
        print(f"DECISION idx={idx} target={c['target_tool']} baseline={pred} correct={row['baseline_correct']}", flush=True)

    p = outdir / f"factgate_shard_{args.shard}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"WROTE {p}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, default=N_SHARDS)
    ap.add_argument("--outdir", required=True)
    run_shard(ap.parse_args())
