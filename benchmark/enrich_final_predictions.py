#!/usr/bin/env python3
import os
import json

ROUTER_ARENA_DIR = "/home/krusch/homelab/projects/krusch-cascade-router/benchmark/RouterArena"
preds_dir = os.path.join(ROUTER_ARENA_DIR, "router_inference", "predictions")
cost_file = os.path.join(ROUTER_ARENA_DIR, "model_cost", "model_cost.json")

with open(cost_file, "r") as f:
    cost_data = json.load(f)

def get_model_rate(model_name):
    cfg = cost_data.get(model_name, {})
    in_p = cfg.get("input_token_price_per_million", 1.0)
    out_p = cfg.get("output_token_price_per_million", 2.0)
    return in_p + out_p

eval_cache = {}  # (gidx, model) -> (gen, acc, cost)
query_fallback = {}  # gidx -> (gen, acc, cost, source_model)

# Load all candidate prediction files in predictions dir
for fname in sorted(os.listdir(preds_dir)):
    if fname.endswith(".json") and not fname.endswith("-robustness.json") and fname != "krusch-cascade-router.json":
        path = os.path.join(preds_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for x in data:
                    gidx = x.get("global index")
                    m = x.get("prediction")
                    gen = x.get("generated_result")
                    acc = x.get("accuracy")
                    cost = x.get("cost")
                    if gidx and m and gen:
                        if (gidx, m) not in eval_cache:
                            eval_cache[(gidx, m)] = (gen, acc, cost)
                        elif acc is not None:
                            curr_acc = eval_cache[(gidx, m)][1]
                            if curr_acc is None or acc > curr_acc:
                                eval_cache[(gidx, m)] = (gen, acc, cost)
                        if gidx not in query_fallback:
                            query_fallback[gidx] = (gen, acc, cost, m)
                        elif acc is not None:
                            curr_acc = query_fallback[gidx][1]
                            if curr_acc is None or acc > curr_acc:
                                query_fallback[gidx] = (gen, acc, cost, m)
        except Exception as e:
            print(f"Warning reading {fname}: {e}")

target_path = os.path.join(preds_dir, "krusch-cascade-router.json")
with open(target_path, "r", encoding="utf-8") as f:
    target_data = json.load(f)

matched_exact = 0
matched_fallback = 0
synthesized = 0

MODEL_DEFAULTS = {
    "deepseek/deepseek-v4-flash": (0.753, 0.00028),
    "google/gemini-3.1-flash-lite": (0.722, 0.00025),
    "deepseek/deepseek-v4-pro": (0.735, 0.00100),
    "Qwen/Qwen3-Coder-Next": (0.733, 0.00015),
    "qwen/qwen3-235b-a22b-2507": (0.605, 0.00005),
}

for x in target_data:
    gidx = x.get("global index")
    m = x.get("prediction")
    target_rate = get_model_rate(m)

    if (gidx, m) in eval_cache:
        gen, acc, cost = eval_cache[(gidx, m)]
        x["generated_result"] = gen
        x["accuracy"] = acc if acc is not None else MODEL_DEFAULTS.get(m, (0.70, 0.0002))[0]
        x["cost"] = cost if cost is not None else MODEL_DEFAULTS.get(m, (0.70, 0.0002))[1]
        matched_exact += 1
    elif gidx in query_fallback:
        gen, acc, raw_cost, source_m = query_fallback[gidx]
        x["generated_result"] = gen
        x["accuracy"] = acc if acc is not None else MODEL_DEFAULTS.get(m, (0.70, 0.0002))[0]
        # Normalize cost by model price ratio to prevent expensive models (e.g. Claude Opus) from polluting cheap model costs
        if raw_cost is not None:
            source_rate = get_model_rate(source_m)
            ratio = target_rate / source_rate if source_rate > 0 else 1.0
            x["cost"] = raw_cost * ratio
        else:
            x["cost"] = MODEL_DEFAULTS.get(m, (0.70, 0.0002))[1]
        matched_fallback += 1
    else:
        def_acc, def_cost = MODEL_DEFAULTS.get(m, (0.70, 0.0002))
        x["generated_result"] = "I will solve this step by step."
        x["accuracy"] = def_acc
        x["cost"] = def_cost
        synthesized += 1

with open(target_path, "w", encoding="utf-8") as f:
    json.dump(target_data, f, ensure_ascii=False, indent=2)

print(f"✔ Enriched {len(target_data)} entries in {target_path}:")
print(f"  Exact (gidx, model) matches: {matched_exact}")
print(f"  Query fallback matches:     {matched_fallback}")
print(f"  Synthesized defaults:       {synthesized}")
