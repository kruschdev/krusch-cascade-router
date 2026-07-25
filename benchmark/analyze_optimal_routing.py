#!/usr/bin/env python3
import os
import json

ROUTER_ARENA_DIR = "/home/krusch/homelab/projects/krusch-cascade-router/benchmark/RouterArena"
cached_dir = os.path.join(ROUTER_ARENA_DIR, "cached_results")

gpt = {}
gemini = {}
claude = {}

def load_cache(filename):
    res = {}
    path = os.path.join(cached_dir, filename)
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gidx = item.get("global_index")
            eval_res = item.get("evaluation_result") or {}
            score = eval_res.get("score", 0.0)
            cost = eval_res.get("inference_cost", 0.0)
            prefix = gidx.split("_")[0] if "_" in gidx else gidx
            res[gidx] = (score, cost, prefix)
    return res

gpt = load_cache("gpt-4o-mini.jsonl")
gemini = load_cache("gemini-2.0-flash-001.jsonl")
claude = load_cache("claude-3-haiku-20240307.jsonl")

domain_scores = {} # domain -> {gpt: [acc, cost], gemini: [acc, cost], claude: [acc, cost]}

for gidx, (score, cost, domain) in gpt.items():
    if domain not in domain_scores:
        domain_scores[domain] = {"gpt": [], "gemini": [], "claude": []}
    domain_scores[domain]["gpt"].append((score, cost))
    
    if gidx in gemini:
        domain_scores[domain]["gemini"].append((gemini[gidx][0], gemini[gidx][1]))
    if gidx in claude:
        domain_scores[domain]["claude"].append((claude[gidx][0], claude[gidx][1]))

print(f"{'Domain':<25} | {'Count':<6} | {'GPT-4o-mini Acc':<15} | {'Gemini-2.0 Acc':<15} | {'Claude-3-Haiku Acc':<18}")
print("-" * 85)

for domain, data in sorted(domain_scores.items(), key=lambda x: len(x[1]["gpt"]), reverse=True):
    count = len(data["gpt"])
    gpt_acc = sum(x[0] for x in data["gpt"]) / count if count else 0
    gem_acc = sum(x[0] for x in data["gemini"]) / len(data["gemini"]) if data["gemini"] else 0
    claude_acc = sum(x[0] for x in data["claude"]) / len(data["claude"]) if data["claude"] else 0
    print(f"{domain:<25} | {count:<6} | {gpt_acc*100:6.2f}%         | {gem_acc*100:6.2f}%         | {claude_acc*100:6.2f}%")
