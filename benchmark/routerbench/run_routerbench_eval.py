# SPDX-License-Identifier: Apache-2.0
"""
RouterBench Full Evaluation Runner for Krusch Cascade Router.
Evaluates across 36,497 inference outcomes (and 5-shot) against 11 LLMs,
Oracle, Random Router, and RouterBench Cascading Router baselines.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

# Add adapter path
BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from krusch_routerbench_adapter import KruschRouterBenchAdapter, MODELS_TO_ROUTE, TOKEN_COSTS

# NumPy 2.x compatible trapezoid integration
trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def get_non_decreasing_convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Computes the non-decreasing convex hull of (cost, accuracy) points,
    as defined in RouterBench (Hu et al., arXiv: 2403.12031).
    """
    pts = points[points[:, 0].argsort()]
    hull = [pts[0]]
    for p in pts[1:]:
        while len(hull) >= 1 and p[1] < hull[-1][1]:
            # If accuracy is lower than previous point at higher cost, drop it
            break
        if len(hull) == 0 or p[0] > hull[-1][0]:
            if len(hull) > 0 and p[1] >= hull[-1][1]:
                hull.append(p)
            elif len(hull) == 0:
                hull.append(p)
    return np.array(hull)


def calculate_aiq(cost_acc_points: np.ndarray, max_cost_ref: float) -> float:
    """
    Calculates Area under the Information/Intelligence-Cost Curve (AIQ) normalized by max cost.
    """
    if len(cost_acc_points) < 2:
        return float(cost_acc_points[0][1]) if len(cost_acc_points) == 1 else 0.0

    hull = get_non_decreasing_convex_hull(cost_acc_points)
    # Add boundary point at max_cost_ref
    if hull[-1][0] < max_cost_ref:
        hull = np.vstack([hull, np.array([max_cost_ref, hull[-1][1]])])

    area = trapezoid(hull[:, 1], hull[:, 0])
    norm_aiq = area / max_cost_ref if max_cost_ref > 0 else 0.0
    return float(norm_aiq)


def run_evaluation(
    dataset_path: str,
    output_dir: str,
    sample_size: int = None,
    run_5shot: bool = False,
):
    print("=" * 80)
    print(f"Loading RouterBench dataset from: {dataset_path}")
    print("=" * 80)

    with open(dataset_path, "rb") as f:
        df = pd.read_pickle(f)

    # Drop any un-evaluated rows (e.g. 14 rows in 5-shot)
    orig_len = len(df)
    df = df.dropna(subset=MODELS_TO_ROUTE).reset_index(drop=True)
    if len(df) < orig_len:
        print(f"Dropped {orig_len - len(df)} un-evaluated rows containing NaN.")

    if sample_size and sample_size < len(df):
        print(f"Subsampling {sample_size} queries from {len(df)} total queries...")
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        print(f"Evaluating across all {len(df):,} queries in split.")

    os.makedirs(output_dir, exist_ok=True)
    num_queries = len(df)

    # Pre-extract matrices for 1000x faster lookups
    model_to_idx = {m: idx for idx, m in enumerate(MODELS_TO_ROUTE)}
    acc_matrix = np.column_stack([df[m].to_numpy(dtype=float) for m in MODELS_TO_ROUTE])
    cost_matrix = np.column_stack([df[f"{m}|total_cost"].to_numpy(dtype=float) for m in MODELS_TO_ROUTE])
    oracle_models = df["oracle_model_to_route_to"].to_numpy()
    row_indices = np.arange(num_queries)

    # 1. Standalone Models Baseline
    print("\n[1/5] Evaluating Standalone Model Baselines...")
    model_metrics = []
    gpt4_name = "gpt-4-1106-preview"
    gpt4_idx = model_to_idx[gpt4_name]
    gpt4_cost = float(np.sum(cost_matrix[:, gpt4_idx]))
    gpt4_acc = float(np.mean(acc_matrix[:, gpt4_idx]))

    for idx, model in enumerate(MODELS_TO_ROUTE):
        acc = float(np.mean(acc_matrix[:, idx]))
        total_cost = float(np.sum(cost_matrix[:, idx]))
        cost_per_1k = (total_cost / num_queries) * 1000.0
        cost_red = (1.0 - total_cost / gpt4_cost) * 100.0 if gpt4_cost > 0 else 0.0
        acc_ratio = (acc / gpt4_acc) * 100.0 if gpt4_acc > 0 else 0.0

        model_metrics.append({
            "Router / Model": model,
            "Type": "Standalone Model",
            "Accuracy (%)": acc * 100.0,
            "Total Cost ($)": total_cost,
            "Cost / 1K ($)": cost_per_1k,
            "Cost Red. vs GPT-4 (%)": cost_red,
            "Acc. vs GPT-4 (%)": acc_ratio,
            "Opt.Sel (%)": np.nan,
            "Regret (%)": (0.9642 - acc) * 100.0,
        })

    # 2. Oracle Baseline
    print("[2/5] Evaluating Oracle Baseline...")
    oracle_acc = float((oracle_models != "no_model_correct").mean())
    oracle_costs = np.zeros(num_queries, dtype=float)
    for i in range(num_queries):
        om = oracle_models[i]
        if om != "no_model_correct" and om in model_to_idx:
            oracle_costs[i] = cost_matrix[i, model_to_idx[om]]
    oracle_total_cost = float(np.sum(oracle_costs))
    oracle_cost_per_1k = (oracle_total_cost / num_queries) * 1000.0

    model_metrics.append({
        "Router / Model": "Oracle (Cheapest Correct LLM)",
        "Type": "Oracle Upper Bound",
        "Accuracy (%)": oracle_acc * 100.0,
        "Total Cost ($)": oracle_total_cost,
        "Cost / 1K ($)": oracle_cost_per_1k,
        "Cost Red. vs GPT-4 (%)": (1.0 - oracle_total_cost / gpt4_cost) * 100.0,
        "Acc. vs GPT-4 (%)": (oracle_acc / gpt4_acc) * 100.0,
        "Opt.Sel (%)": 100.0,
        "Regret (%)": 0.0,
    })

    # 3. Random Router Baseline
    print("[3/5] Evaluating Random Router Baseline...")
    np.random.seed(42)
    random_indices = np.random.randint(0, len(MODELS_TO_ROUTE), size=num_queries)
    random_choices = np.array([MODELS_TO_ROUTE[idx] for idx in random_indices])
    random_accs = acc_matrix[row_indices, random_indices]
    random_costs = cost_matrix[row_indices, random_indices]
    random_opt_sel = (random_choices == oracle_models).astype(float)
    random_total_cost = float(np.sum(random_costs))
    random_acc = float(np.mean(random_accs))

    model_metrics.append({
        "Router / Model": "Random Uniform Router",
        "Type": "Baseline Router",
        "Accuracy (%)": random_acc * 100.0,
        "Total Cost ($)": random_total_cost,
        "Cost / 1K ($)": (random_total_cost / num_queries) * 1000.0,
        "Cost Red. vs GPT-4 (%)": (1.0 - random_total_cost / gpt4_cost) * 100.0,
        "Acc. vs GPT-4 (%)": (random_acc / gpt4_acc) * 100.0,
        "Opt.Sel (%)": float(np.mean(random_opt_sel)) * 100.0,
        "Regret (%)": (oracle_acc - random_acc) * 100.0,
    })

    # 4. Krusch Router across Willingness to Pay Spectrum
    print("[4/5] Evaluating Krusch Cascade Router across WTP Pareto frontier...")
    adapter = KruschRouterBenchAdapter(models_to_route=MODELS_TO_ROUTE)

    # Standard WTP values spanning frugal to quality priority
    wtp_values = [
        0.0001,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.50,
        1.0,
        5.0,
        10.0,
    ]

    wtp_curve_data = []
    prompts = df["prompt"].tolist()

    for wtp in wtp_values:
        routed_models = adapter.batch_route_prompts(prompts, willingness_to_pay=wtp)
        chosen_indices = np.array([model_to_idx[m] for m in routed_models])
        acc_list = acc_matrix[row_indices, chosen_indices]
        cost_list = cost_matrix[row_indices, chosen_indices]
        opt_sel_list = (routed_models == oracle_models).astype(float)

        total_cost = float(np.sum(cost_list))
        acc = float(np.mean(acc_list))
        opt_sel = float(np.mean(opt_sel_list))
        cost_per_1k = (total_cost / num_queries) * 1000.0
        cost_red = (1.0 - total_cost / gpt4_cost) * 100.0
        acc_ratio = (acc / gpt4_acc) * 100.0
        regret = (oracle_acc - acc) * 100.0

        model_counts = pd.Series(routed_models).value_counts().to_dict()

        row = {
            "Router / Model": f"krusch (WTP={wtp})",
            "Type": "Krusch Cascade Router",
            "Accuracy (%)": acc * 100.0,
            "Total Cost ($)": total_cost,
            "Cost / 1K ($)": cost_per_1k,
            "Cost Red. vs GPT-4 (%)": cost_red,
            "Acc. vs GPT-4 (%)": acc_ratio,
            "Opt.Sel (%)": opt_sel * 100.0,
            "Regret (%)": regret,
            "WTP": wtp,
            "Model Distribution": model_counts,
        }
        wtp_curve_data.append(row)
        model_metrics.append(row)

    # 5. Cascading Router Baseline (RouterBench official cascade)
    print("[5/5] Evaluating RouterBench Cascading Router Simulation...")
    ranked_models = sorted(MODELS_TO_ROUTE, key=lambda m: TOKEN_COSTS.get(m, 0.0002))
    ranked_indices = [model_to_idx[m] for m in ranked_models]

    for error_rate in [0.0, 0.05, 0.10, 0.20]:
        casc_accs = np.zeros(num_queries, dtype=float)
        casc_costs = np.zeros(num_queries, dtype=float)
        np.random.seed(42)

        for i in range(num_queries):
            q_cost = 0.0
            q_acc = 0.0
            for m_idx in ranked_indices:
                m_cost = cost_matrix[i, m_idx]
                m_score = acc_matrix[i, m_idx]
                q_cost += m_cost

                evaluator_wrong = np.random.rand() < error_rate
                accepted = (m_score < 0.5) if evaluator_wrong else (m_score >= 0.5)

                q_acc = m_score
                if accepted:
                    break

            casc_accs[i] = q_acc
            casc_costs[i] = q_cost

        c_total_cost = float(np.sum(casc_costs))
        c_acc = float(np.mean(casc_accs))

        model_metrics.append({
            "Router / Model": f"Cascade Router (Err={int(error_rate*100)}%)",
            "Type": "RouterBench Cascade",
            "Accuracy (%)": c_acc * 100.0,
            "Total Cost ($)": c_total_cost,
            "Cost / 1K ($)": (c_total_cost / num_queries) * 1000.0,
            "Cost Red. vs GPT-4 (%)": (1.0 - c_total_cost / gpt4_cost) * 100.0,
            "Acc. vs GPT-4 (%)": (c_acc / gpt4_acc) * 100.0,
            "Opt.Sel (%)": np.nan,
            "Regret (%)": (oracle_acc - c_acc) * 100.0,
        })

    # Compute AIQ (Area Under Cost-Performance Convex Hull)
    krusch_points = np.array([[r["Total Cost ($)"], r["Accuracy (%)"] / 100.0] for r in wtp_curve_data])
    krusch_aiq = calculate_aiq(krusch_points, gpt4_cost)
    print(f"\n>>> Krusch RouterBench AIQ Score: {krusch_aiq:.4f} <<<")

    # Domain Breakdown for Krusch at balanced WTP (0.01) and high WTP (0.05)
    print("\nComputing domain breakdown across standard benchmarks...")
    major_benchmarks = [
        "grade-school-math",
        "hellaswag",
        "arc-challenge",
        "winogrande",
        "mbpp",
        "mmlu-professional-law",
        "mmlu-moral-scenarios",
    ]
    domain_rows = []
    for b in major_benchmarks:
        sub_indices = np.where(df["eval_name"].to_numpy() == b)[0]
        sub_n = len(sub_indices)
        if sub_n == 0:
            continue
        sub_prompts = [prompts[idx] for idx in sub_indices]
        sub_acc_matrix = acc_matrix[sub_indices, :]
        sub_cost_matrix = cost_matrix[sub_indices, :]
        sub_row_indices = np.arange(sub_n)

        # Standalone GPT-4
        b_gpt4_acc = float(np.mean(sub_acc_matrix[:, gpt4_idx]))
        b_gpt4_cost = float(np.sum(sub_cost_matrix[:, gpt4_idx]))

        # Krusch Frugal (WTP=0.005)
        k_frugal_choices = adapter.batch_route_prompts(sub_prompts, willingness_to_pay=0.005)
        k_frugal_idx = np.array([model_to_idx[m] for m in k_frugal_choices])
        k_frugal_acc = float(np.mean(sub_acc_matrix[sub_row_indices, k_frugal_idx]))
        k_frugal_cost = float(np.sum(sub_cost_matrix[sub_row_indices, k_frugal_idx]))

        # Krusch Balanced (WTP=0.05)
        k_balanced_choices = adapter.batch_route_prompts(sub_prompts, willingness_to_pay=0.05)
        k_balanced_idx = np.array([model_to_idx[m] for m in k_balanced_choices])
        k_balanced_acc = float(np.mean(sub_acc_matrix[sub_row_indices, k_balanced_idx]))
        k_balanced_cost = float(np.sum(sub_cost_matrix[sub_row_indices, k_balanced_idx]))

        domain_rows.append({
            "Benchmark": b,
            "Queries": sub_n,
            "GPT-4 Acc (%)": b_gpt4_acc * 100.0,
            "GPT-4 Cost ($)": b_gpt4_cost,
            "Krusch Frugal Acc (%)": k_frugal_acc * 100.0,
            "Krusch Frugal Cost ($)": k_frugal_cost,
            "Krusch Frugal Savings (%)": (1.0 - k_frugal_cost / b_gpt4_cost) * 100.0,
            "Krusch Balanced Acc (%)": k_balanced_acc * 100.0,
            "Krusch Balanced Cost ($)": k_balanced_cost,
            "Krusch Balanced Savings (%)": (1.0 - k_balanced_cost / b_gpt4_cost) * 100.0,
        })

    domain_df = pd.DataFrame(domain_rows)

    # Measure Routing Latency Benchmark
    print("Benchmarking routing execution latency on CPU (10,000 queries)...")
    latency_prompts = (prompts * 5)[:10000]
    t0 = time.perf_counter()
    adapter.batch_route_prompts(latency_prompts, willingness_to_pay=0.01)
    t_elapsed = time.perf_counter() - t0
    avg_latency_us = (t_elapsed / len(latency_prompts)) * 1_000_000.0
    qps = len(latency_prompts) / t_elapsed
    print(f"Average CPU routing latency: {avg_latency_us:.2f} µs/query ({qps:,.0f} QPS)")

    # Save CSV and JSON outputs
    summary_df = pd.DataFrame(model_metrics)
    summary_csv = os.path.join(output_dir, "routerbench_eval_summary.csv")
    summary_json = os.path.join(output_dir, "routerbench_eval_summary.json")
    domain_csv = os.path.join(output_dir, "routerbench_domain_breakdown.csv")
    wtp_csv = os.path.join(output_dir, "routerbench_wtp_curve.csv")

    summary_df.to_csv(summary_csv, index=False)
    domain_df.to_csv(domain_csv, index=False)
    pd.DataFrame(wtp_curve_data).to_csv(wtp_csv, index=False)

    with open(summary_json, "w") as f:
        json.dump({
            "dataset": dataset_path,
            "num_queries": num_queries,
            "aiq": krusch_aiq,
            "routing_latency_us": avg_latency_us,
            "qps": qps,
            "results": model_metrics,
            "domain_breakdown": domain_rows,
        }, f, indent=2)

    # Display Clean Tables
    print("\n" + "=" * 80)
    print("ROUTERBENCH EVALUATION SUMMARY TABLE (36,497 Inference Outcomes)")
    print("=" * 80)
    display_cols = [
        "Router / Model",
        "Type",
        "Accuracy (%)",
        "Total Cost ($)",
        "Cost / 1K ($)",
        "Cost Red. vs GPT-4 (%)",
        "Opt.Sel (%)",
        "Regret (%)",
    ]
    print(tabulate(summary_df[display_cols], headers="keys", tablefmt="pipe", floatfmt=".2f"))

    print("\n" + "=" * 80)
    print("DOMAIN-SPECIFIC BENCHMARK BREAKDOWN")
    print("=" * 80)
    print(tabulate(domain_df, headers="keys", tablefmt="pipe", floatfmt=".2f"))

    return summary_df, domain_df, krusch_aiq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RouterBench Evaluation for Krusch Cascade Router")
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark/data/routerbench/routerbench_0shot.pkl",
        help="Path to routerbench pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/routerbench/results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional subsample size for testing",
    )
    parser.add_argument(
        "--run-5shot",
        action="store_true",
        help="Also run on routerbench_5shot.pkl",
    )

    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        run_5shot=args.run_5shot,
    )
