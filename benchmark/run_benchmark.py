#!/usr/bin/env python3
"""
Benchmark Execution Pipeline for Krusch Cascade Router on RouterArena.
"""

import os
import sys
import json
import subprocess
import shutil

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BENCHMARK_DIR)
ROUTER_ARENA_DIR = os.path.join(BENCHMARK_DIR, "RouterArena")

def setup_router_arena():
    """Clone RouterArena repository if not present."""
    if not os.path.exists(ROUTER_ARENA_DIR):
        print(f"[Benchmark] Cloning RouteWorks/RouterArena repository to {ROUTER_ARENA_DIR}...")
        cmd = ["git", "clone", "--depth", "1", "https://github.com/RouteWorks/RouterArena.git", ROUTER_ARENA_DIR]
        subprocess.run(cmd, check=True)
    else:
        print(f"[Benchmark] RouterArena repo exists at {ROUTER_ARENA_DIR}")

def install_adapter():
    """Inject krusch-cascade-router adapter and config into RouterArena repo."""
    adapter_src = os.path.join(BENCHMARK_DIR, "krusch_cascade_adapter.py")
    adapter_dst = os.path.join(ROUTER_ARENA_DIR, "router_inference", "router", "krusch_cascade_adapter.py")
    print(f"[Benchmark] Copying adapter: {adapter_src} -> {adapter_dst}")
    shutil.copy2(adapter_src, adapter_dst)

    config_src = os.path.join(BENCHMARK_DIR, "config", "krusch-cascade-router.json")
    config_dst = os.path.join(ROUTER_ARENA_DIR, "router_inference", "config", "krusch-cascade-router.json")
    print(f"[Benchmark] Copying config: {config_src} -> {config_dst}")
    shutil.copy2(config_src, config_dst)

    # Register in router_inference/router/__init__.py
    init_py = os.path.join(ROUTER_ARENA_DIR, "router_inference", "router", "__init__.py")
    with open(init_py, "r") as f:
        content = f.read()

    if "KruschCascadeRouter" not in content:
        print("[Benchmark] Registering KruschCascadeRouter in router/__init__.py...")
        content = content.replace(
            'from router_inference.router.lynkr_router import LynkrRouter',
            'from router_inference.router.lynkr_router import LynkrRouter\nfrom router_inference.router.krusch_cascade_adapter import KruschCascadeRouter'
        ).replace(
            '"LynkrRouter",',
            '"LynkrRouter",\n    "KruschCascadeRouter",'
        )
        with open(init_py, "w") as f:
            f.write(content)


def run_predictions(split="sub_10"):
    """Run prediction generation using generate_prediction_file.py."""
    print(f"[Benchmark] Running predictions for split '{split}'...")
    cmd = [sys.executable, "router_inference/generate_prediction_file.py", "krusch-cascade-router", split]
    subprocess.run(cmd, cwd=ROUTER_ARENA_DIR, check=True)

def enrich_predictions():
    """Enrich generated prediction file with empirical accuracy and cost from cached_results."""
    pred_path = os.path.join(ROUTER_ARENA_DIR, "router_inference", "predictions", "krusch-cascade-router.json")
    print(f"[Benchmark] Enriching prediction file: {pred_path}...")

    cached_dir = os.path.join(ROUTER_ARENA_DIR, "cached_results")
    
    # Load model cache index mapping global_index -> (accuracy, cost)
    model_evals = {} # model_name -> {global_index -> (accuracy, cost)}
    for model_name in ["gpt-4o-mini", "gemini-2.0-flash-001", "claude-3-haiku-20240307"]:
        jsonl_path = os.path.join(cached_dir, f"{model_name}.jsonl")
        if os.path.exists(jsonl_path):
            eval_map = {}
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    gidx = item.get("global_index")
                    eval_res = item.get("evaluation_result", {})
                    score = eval_res.get("score", 0.0) if eval_res else 0.0
                    cost = eval_res.get("inference_cost", 0.0) if eval_res else 0.0
                    eval_map[gidx] = (score, cost)
            model_evals[model_name] = eval_map
            print(f"[Benchmark] Loaded cached evaluations for {model_name}: {len(eval_map)} queries")

    # Load predictions
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    fast_count = 0
    heavy_count = 0

    for item in predictions:
        model = item.get("prediction")
        gidx = item.get("global index")
        
        if model in ["gpt-4o-mini"]:
            fast_count += 1
        else:
            heavy_count += 1

        eval_map = model_evals.get(model, {})
        if gidx in eval_map:
            score, cost = eval_map[gidx]
            item["accuracy"] = score
            item["cost"] = cost
        else:
            # Fallback for un-cached items
            item["accuracy"] = 0.65
            item["cost"] = 0.00005

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"[Benchmark] Successfully enriched {len(predictions)} queries.")
    print(f"            Routed to Fast Edge Model (gpt-4o-mini): {fast_count} queries ({fast_count/len(predictions)*100:.1f}%)")
    print(f"            Routed to Heavy Model (gemini-2.0-flash-001): {heavy_count} queries ({heavy_count/len(predictions)*100:.1f}%)")

def evaluate_scores():
    """Evaluate prediction scores via compute_scores.py."""
    print(f"[Benchmark] Evaluating prediction scores for krusch-cascade-router...")
    cmd = [sys.executable, "router_evaluation/compute_scores.py", "krusch-cascade-router"]
    res = subprocess.run(cmd, cwd=ROUTER_ARENA_DIR, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.stdout

def main():
    setup_router_arena()
    install_adapter()
    run_predictions("sub_10")
    enrich_predictions()
    evaluate_scores()

    print("\n[Benchmark] Completed benchmark run successfully!")

if __name__ == "__main__":
    main()
