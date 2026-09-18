#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the Krusch Cascade Router project
# SPDX-License-Identifier: MIT

"""
Execution and Evaluation runner for Arena-Hard-Auto benchmark.
Evaluates Krusch Cascade Router across Arena-Hard v0.1 (500 prompts) and v2.0 (750 prompts).
Produces cost vs win-rate curves, domain distributions, and APGR metrics.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARENA_HARD_DIR = Path(__file__).resolve().parent / "repo"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ARENA_HARD_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_HARD_DIR))

# Ensure OpenRouter environment is loaded
env_file = PROJECT_ROOT / ".env"
if not env_file.exists():
    env_file = Path("/home/krusch/homelab/.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

if "OPENROUTER_API_KEY" in os.environ and "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

from benchmark.arena_hard.krusch_arena_hard_adapter import KruschArenaHardAdapter


# Pricing Constants per 1M tokens ($)
MODEL_PRICING = {
    "gpt-3.5-turbo-0125": {"prompt": 0.50, "completion": 1.50},
    "gpt-4-0613": {"prompt": 30.00, "completion": 60.00},
    "gpt-4-0314": {"prompt": 30.00, "completion": 60.00},
    "qwq-32b": {"prompt": 0.15, "completion": 0.60},
    "deepseek-r1": {"prompt": 0.55, "completion": 2.19},
    "o3-mini-2025-01-31": {"prompt": 1.10, "completion": 4.40},
}


def load_judgments_scores(path: Path, weight: int = 3) -> Dict[str, float]:
    """
    Extract per-question win-rates against baseline using official Arena-Hard weighting.
    """
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }
    uid_scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            uid = data["uid"]
            games = data.get("games", [])
            if len(games) >= 2:
                g0 = games[0].get("score")
                g1 = games[1].get("score")
                if g0 and g1:
                    # g1: Assistant A = candidate, Assistant B = baseline
                    s1 = label_to_score.get(g1, [0.5])
                    # g0: Assistant A = baseline, Assistant B = candidate
                    s0 = [1 - s for s in label_to_score.get(g0, [0.5])]
                    uid_scores[uid] = float(np.mean(s1 + s0))
    return uid_scores


def load_model_answers_tokens(path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Estimate prompt and completion token counts from answer file.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens_map = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            uid = data["uid"]
            messages = data.get("messages", [])
            p_tok = 0
            c_tok = 0
            for m in messages:
                raw_content = m.get("content", "")
                if isinstance(raw_content, list):
                    content_str = " ".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in raw_content
                    )
                elif isinstance(raw_content, dict):
                    content_str = json.dumps(raw_content)
                else:
                    content_str = str(raw_content)

                tok_count = len(enc.encode(content_str, disallowed_special=()))
                if m.get("role") in ("user", "system"):
                    p_tok += tok_count
                elif m.get("role") == "assistant":
                    c_tok += tok_count
            tokens_map[uid] = (p_tok, c_tok)
    return tokens_map


def compute_query_cost(p_tok: int, c_tok: int, model: str) -> float:
    pricing = MODEL_PRICING.get(model, {"prompt": 1.0, "completion": 2.0})
    return (p_tok / 1_000_000.0) * pricing["prompt"] + (c_tok / 1_000_000.0) * pricing["completion"]


def evaluate_arena_hard_v01():
    print("\n" + "=" * 80)
    print("Evaluating Arena-Hard-Auto v0.1 (500 Prompts)")
    print("=" * 80)

    data_dir = ARENA_HARD_DIR / "data" / "arena-hard-v0.1"
    ques_file = data_dir / "question.jsonl"
    judge_dir = data_dir / "model_judgment" / "gpt-4-1106-preview"
    ans_dir = data_dir / "model_answer"

    # Load questions
    questions = []
    with open(ques_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} evaluation questions.")

    # Load judgments against baseline gpt-4-0314
    scores_weak = load_judgments_scores(judge_dir / "gpt-3.5-turbo-0125.jsonl")
    scores_strong = load_judgments_scores(judge_dir / "gpt-4-0613.jsonl")

    # Load answer tokens
    tokens_weak = load_model_answers_tokens(ans_dir / "gpt-3.5-turbo-0125.jsonl")
    tokens_strong = load_model_answers_tokens(ans_dir / "gpt-4-0613.jsonl")

    adapter = KruschArenaHardAdapter()

    # Domain classification & complexity analysis
    domain_counts = collections_counter = {}
    complexity_per_domain = {}
    for q in questions:
        dom = adapter.classify_domain(q["prompt"])
        comp = adapter.evaluate_complexity(q["prompt"])
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        complexity_per_domain.setdefault(dom, []).append(comp)

    print("\n--- Domain & Complexity Breakdown (500 Prompts) ---")
    for dom, cnt in sorted(domain_counts.items(), key=lambda x: -x[1]):
        avg_c = float(np.mean(complexity_per_domain[dom]))
        pct = (cnt / len(questions)) * 100
        print(f"  {dom:<18}: {cnt:>3} prompts ({pct:>5.1f}%) | Avg Complexity: {avg_c:.3f}")

    # Baseline calculations
    # Always weak (GPT-3.5)
    cost_weak_total = sum(compute_query_cost(tokens_weak[q["uid"]][0], tokens_weak[q["uid"]][1], "gpt-3.5-turbo-0125") for q in questions if q["uid"] in tokens_weak)
    score_weak_avg = float(np.mean([scores_weak[q["uid"]] for q in questions if q["uid"] in scores_weak])) * 100

    # Always strong (GPT-4)
    cost_strong_total = sum(compute_query_cost(tokens_strong[q["uid"]][0], tokens_strong[q["uid"]][1], "gpt-4-0613") for q in questions if q["uid"] in tokens_strong)
    score_strong_avg = float(np.mean([scores_strong[q["uid"]] for q in questions if q["uid"] in scores_strong])) * 100

    print(f"\nBaseline Performance:")
    print(f"  Always GPT-3.5-Turbo : Win-Rate = {score_weak_avg:.2f}% | Total Cost = ${cost_weak_total:.4f} (${cost_weak_total/len(questions)*1000:.2f}/1k)")
    print(f"  Always GPT-4-0613    : Win-Rate = {score_strong_avg:.2f}% | Total Cost = ${cost_strong_total:.4f} (${cost_strong_total/len(questions)*1000:.2f}/1k)")
    print(f"  Cost Ratio (GPT-4/GPT-3.5): {cost_strong_total/cost_weak_total:.1f}x")

    # Threshold Sweep
    thresholds = [0.15, 0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85]
    sweep_results = []

    print("\n--- Krusch Cascade Router Threshold Sweep ---")
    print(f"{'Threshold τ':<12}{'Win-Rate (%)':<15}{'Total Cost ($)':<16}{'Cost/1K ($)':<14}{'% GPT-4 Calls':<16}{'Cost Reduction (%)':<20}{'Quality Retained (%)':<20}")

    for tau in thresholds:
        routed_scores = []
        routed_costs = []
        strong_calls = 0

        t0 = time.perf_counter()
        for q in questions:
            uid = q["uid"]
            chosen_model, meta = adapter.route_binary(
                q["prompt"],
                weak_model="gpt-3.5-turbo-0125",
                strong_model="gpt-4-0613",
                threshold=tau,
            )
            if meta["escalated"]:
                strong_calls += 1
                routed_scores.append(scores_strong.get(uid, 0.5))
                pt, ct = tokens_strong.get(uid, (100, 300))
                routed_costs.append(compute_query_cost(pt, ct, "gpt-4-0613"))
            else:
                routed_scores.append(scores_weak.get(uid, 0.5))
                pt, ct = tokens_weak.get(uid, (100, 300))
                routed_costs.append(compute_query_cost(pt, ct, "gpt-3.5-turbo-0125"))
        eval_time = (time.perf_counter() - t0) * 1000 / len(questions)

        avg_score = float(np.mean(routed_scores)) * 100
        tot_cost = sum(routed_costs)
        cost_per_1k = (tot_cost / len(questions)) * 1000
        strong_pct = (strong_calls / len(questions)) * 100
        cost_reduction = (1.0 - (tot_cost / cost_strong_total)) * 100
        quality_retained = (avg_score / score_strong_avg) * 100

        sweep_results.append({
            "threshold": tau,
            "win_rate": avg_score,
            "total_cost": tot_cost,
            "cost_per_1k": cost_per_1k,
            "strong_calls_pct": strong_pct,
            "cost_reduction_pct": cost_reduction,
            "quality_retained_pct": quality_retained,
            "latency_ms": eval_time,
        })

        print(f"{tau:<12.2f}{avg_score:<15.2f}${tot_cost:<15.4f}${cost_per_1k:<13.2f}{strong_pct:<16.1f}%{cost_reduction:<19.2f}%{quality_retained:<19.2f}%")

    # APGR (Average Preference Grade Ratio) calculation
    # Normalized area under the cost-winrate curve relative to random router
    costs_norm = [(r["total_cost"] - cost_weak_total) / (cost_strong_total - cost_weak_total) for r in sweep_results]
    scores_norm = [(r["win_rate"] - score_weak_avg) / (score_strong_avg - score_weak_avg) for r in sweep_results]
    
    # Sort by cost
    sorted_pairs = sorted(zip(costs_norm, scores_norm))
    c_vals = [0.0] + [p[0] for p in sorted_pairs] + [1.0]
    s_vals = [0.0] + [p[1] for p in sorted_pairs] + [1.0]
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    apgr = float(trapezoid(s_vals, c_vals))
    print(f"\nKrusch Cascade Router Arena-Hard APGR: {apgr:.4f} (Ceiling: 1.0000)")

    return {
        "benchmark": "arena-hard-v0.1",
        "num_queries": len(questions),
        "score_weak": score_weak_avg,
        "score_strong": score_strong_avg,
        "cost_weak_total": cost_weak_total,
        "cost_strong_total": cost_strong_total,
        "apgr": apgr,
        "sweep_results": sweep_results,
        "domain_distribution": {dom: {"count": cnt, "avg_complexity": float(np.mean(complexity_per_domain[dom]))} for dom, cnt in domain_counts.items()},
    }


def evaluate_arena_hard_v20():
    print("\n" + "=" * 80)
    print("Evaluating Arena-Hard-Auto v2.0 (750 Prompts - Frontier Reasoning & Open Models)")
    print("=" * 80)

    data_dir = ARENA_HARD_DIR / "data" / "arena-hard-v2.0"
    ques_file = data_dir / "question.jsonl"
    judge_dir = data_dir / "model_judgment" / "gpt-4.1"
    ans_dir = data_dir / "model_answer"

    questions = []
    with open(ques_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions)} v2.0 evaluation questions.")

    # Load judgments against baseline o3-mini-2025-01-31
    scores_weak = load_judgments_scores(judge_dir / "qwq-32b.jsonl")
    scores_strong = load_judgments_scores(judge_dir / "deepseek-r1.jsonl")

    # Load answer tokens
    tokens_weak = load_model_answers_tokens(ans_dir / "qwq-32b.jsonl")
    tokens_strong = load_model_answers_tokens(ans_dir / "deepseek-r1.jsonl")

    adapter = KruschArenaHardAdapter()

    cost_weak_total = sum(compute_query_cost(tokens_weak[q["uid"]][0], tokens_weak[q["uid"]][1], "qwq-32b") for q in questions if q["uid"] in tokens_weak)
    score_weak_avg = float(np.mean([scores_weak[q["uid"]] for q in questions if q["uid"] in scores_weak])) * 100

    cost_strong_total = sum(compute_query_cost(tokens_strong[q["uid"]][0], tokens_strong[q["uid"]][1], "deepseek-r1") for q in questions if q["uid"] in tokens_strong)
    score_strong_avg = float(np.mean([scores_strong[q["uid"]] for q in questions if q["uid"] in scores_strong])) * 100

    print(f"\nBaseline Performance (v2.0 against o3-mini):")
    print(f"  Always QwQ-32B      : Win-Rate = {score_weak_avg:.2f}% | Total Cost = ${cost_weak_total:.4f} (${cost_weak_total/len(questions)*1000:.2f}/1k)")
    print(f"  Always DeepSeek-R1  : Win-Rate = {score_strong_avg:.2f}% | Total Cost = ${cost_strong_total:.4f} (${cost_strong_total/len(questions)*1000:.2f}/1k)")

    thresholds = [0.20, 0.35, 0.50, 0.65, 0.80]
    sweep_results = []

    print("\n--- Krusch Cascade Router Threshold Sweep (v2.0: QwQ-32B -> DeepSeek-R1) ---")
    print(f"{'Threshold τ':<12}{'Win-Rate (%)':<15}{'Total Cost ($)':<16}{'Cost/1K ($)':<14}{'% R1 Calls':<16}{'Cost Reduction (%)':<20}{'Quality Retained (%)':<20}")

    for tau in thresholds:
        routed_scores = []
        routed_costs = []
        r1_calls = 0

        for q in questions:
            uid = q["uid"]
            chosen_model, meta = adapter.route_binary(
                q["prompt"],
                weak_model="qwq-32b",
                strong_model="deepseek-r1",
                threshold=tau,
            )
            if meta["escalated"]:
                r1_calls += 1
                routed_scores.append(scores_strong.get(uid, 0.5))
                pt, ct = tokens_strong.get(uid, (200, 1500))
                routed_costs.append(compute_query_cost(pt, ct, "deepseek-r1"))
            else:
                routed_scores.append(scores_weak.get(uid, 0.5))
                pt, ct = tokens_weak.get(uid, (200, 1500))
                routed_costs.append(compute_query_cost(pt, ct, "qwq-32b"))

        avg_score = float(np.mean(routed_scores)) * 100
        tot_cost = sum(routed_costs)
        cost_per_1k = (tot_cost / len(questions)) * 1000
        r1_pct = (r1_calls / len(questions)) * 100
        cost_reduction = (1.0 - (tot_cost / cost_strong_total)) * 100
        quality_retained = (avg_score / score_strong_avg) * 100

        sweep_results.append({
            "threshold": tau,
            "win_rate": avg_score,
            "total_cost": tot_cost,
            "cost_per_1k": cost_per_1k,
            "strong_calls_pct": r1_pct,
            "cost_reduction_pct": cost_reduction,
            "quality_retained_pct": quality_retained,
        })

        print(f"{tau:<12.2f}{avg_score:<15.2f}${tot_cost:<15.4f}${cost_per_1k:<13.2f}{r1_pct:<16.1f}%{cost_reduction:<19.2f}%{quality_retained:<19.2f}%")

    costs_norm = [(r["total_cost"] - cost_weak_total) / (cost_strong_total - cost_weak_total) for r in sweep_results]
    scores_norm = [(r["win_rate"] - score_weak_avg) / (score_strong_avg - score_weak_avg) for r in sweep_results]
    sorted_pairs = sorted(zip(costs_norm, scores_norm))
    c_vals = [0.0] + [p[0] for p in sorted_pairs] + [1.0]
    s_vals = [0.0] + [p[1] for p in sorted_pairs] + [1.0]
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    apgr = float(trapezoid(s_vals, c_vals))
    print(f"\nKrusch Cascade Router Arena-Hard v2.0 APGR: {apgr:.4f}")

    return {
        "benchmark": "arena-hard-v2.0",
        "num_queries": len(questions),
        "score_weak": score_weak_avg,
        "score_strong": score_strong_avg,
        "cost_weak_total": cost_weak_total,
        "cost_strong_total": cost_strong_total,
        "apgr": apgr,
        "sweep_results": sweep_results,
    }


def main():
    print("=" * 80)
    print("Running Arena-Hard-Auto Multi-Benchmark Suite")
    print(f"OpenRouter Config: API Key Loaded = {'OPENROUTER_API_KEY' in os.environ}")
    print("=" * 80)

    res_v01 = evaluate_arena_hard_v01()
    res_v20 = evaluate_arena_hard_v20()

    summary = {
        "timestamp": time.time(),
        "v01": res_v01,
        "v20": res_v20,
    }

    out_file = RESULTS_DIR / "arena_hard_eval_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Arena-Hard evaluation complete! Summary saved to {out_file}")


if __name__ == "__main__":
    main()
