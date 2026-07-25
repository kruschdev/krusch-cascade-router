#!/usr/bin/env python3
import os
import sys
import json
import math

ROUTER_ARENA_DIR = "/home/krusch/homelab/projects/krusch-cascade-router/benchmark/RouterArena"
cached_dir = os.path.join(ROUTER_ARENA_DIR, "cached_results")

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
            q = item.get("question", "")
            res[gidx] = (score, cost, q)
    return res

gpt = load_cache("gpt-4o-mini.jsonl")
gemini = load_cache("gemini-2.0-flash-001.jsonl")
claude = load_cache("claude-3-haiku-20240307.jsonl")

common_indices = set(gpt.keys()) & set(gemini.keys()) & set(claude.keys())
print(f"Common sub_10 indices across all 3 models: {len(common_indices)}")

def compute_arena_score(cost, accuracy, beta=0.1, c_max=200, c_min=0.0044):
    C_i = (math.log2(c_max) - math.log2(cost)) / (math.log2(c_max) - math.log2(c_min))
    return ((1 + beta) * accuracy * C_i) / (beta * accuracy + C_i)

# Test 1: pure GPT-4o-mini
acc1 = sum(gpt[idx][0] for idx in common_indices) / len(common_indices)
cost1 = sum(gpt[idx][1] for idx in common_indices) / len(common_indices) * 1000
score1 = compute_arena_score(cost1, acc1)
print(f"Pure GPT-4o-mini  : Acc={acc1*100:.2f}%, Cost/1K=${cost1:.4f}, Arena Score={score1*100:.2f}")

# Test 2: pure Gemini 2.0 Flash
acc2 = sum(gemini[idx][0] for idx in common_indices) / len(common_indices)
cost2 = sum(gemini[idx][1] for idx in common_indices) / len(common_indices) * 1000
score2 = compute_arena_score(cost2, acc2)
print(f"Pure Gemini Flash : Acc={acc2*100:.2f}%, Cost/1K=${cost2:.4f}, Arena Score={score2*100:.2f}")

# Test 3: pure Claude Haiku
acc3 = sum(claude[idx][0] for idx in common_indices) / len(common_indices)
cost3 = sum(claude[idx][1] for idx in common_indices) / len(common_indices) * 1000
score3 = compute_arena_score(cost3, acc3)
print(f"Pure Claude Haiku : Acc={acc3*100:.2f}%, Cost/1K=${cost3:.4f}, Arena Score={score3*100:.2f}")

# Test 4: Optimal Oracle Router
acc4 = sum(max(gpt[idx][0], gemini[idx][0], claude[idx][0]) for idx in common_indices) / len(common_indices)
cost4 = sum(gemini[idx][1] if gemini[idx][0] > gpt[idx][0] else gpt[idx][1] for idx in common_indices) / len(common_indices) * 1000
score4 = compute_arena_score(cost4, acc4)
print(f"Oracle Best Router: Acc={acc4*100:.2f}%, Cost/1K=${cost4:.4f}, Arena Score={score4*100:.2f}")
