# SPDX-License-Identifier: Apache-2.0
"""
AutoMix Benchmark Evaluation Runner for Krusch Cascade Router.
Evaluates early-exit cascade routing across CoQA, CNLI, NarrativeQA, Quality, and QASPER
benchmarked against Google Research's AutoMix POMDP and Threshold baselines.
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

from krusch_automix_adapter import KruschAutomixAdapter

# Published reference baselines from AutoMix paper (Aggarwal et al., NeurIPS 2024; paper_eval_seed_final.json)
PAPER_BASELINES = {
    "coqa": {
        "pomdp_lift": 0.4368,
        "thresh_lift": 0.4316,
        "pomdp_cost": 6.93,
        "pomdp_perf": 0.5054,
    },
    "cnli": {
        "pomdp_lift": 0.8872,
        "thresh_lift": -0.0355,
        "pomdp_cost": 6.66,
        "pomdp_perf": 0.4352,
    },
    "narrative_qa": {
        "pomdp_lift": 0.0644,
        "thresh_lift": 0.1215,
        "pomdp_cost": 9.95,
        "pomdp_perf": 0.2143,
    },
    "quality": {
        "pomdp_lift": -0.1184,
        "thresh_lift": -0.0435,
        "pomdp_cost": 15.80,
        "pomdp_perf": 0.5288,
    },
    "qasper": {
        "pomdp_lift": 0.0693,
        "thresh_lift": 0.0367,
        "pomdp_cost": 45.19,
        "pomdp_perf": 0.2760,
    },
}


def compute_metrics(
    data: pd.DataFrame,
    to_retry: np.ndarray,
    slm_col: str = "llama13b_f1",
    llm_col: str = "llama70b_f1",
    slm_cost: float = 1.0,
    llm_cost: float = 50.0,
    verifier_cost: float = 1.0,
) -> Dict[str, float]:
    """
    Computes performance, cost, and IBC Lift matching AutoMix formulation.
    """
    n = len(data)
    total_cost = (~to_retry).sum() * slm_cost + to_retry.sum() * (llm_cost + slm_cost)
    avg_cost = (total_cost / n) + verifier_cost

    slm_perf = float(data[slm_col].mean())
    llm_perf = float(data[llm_col].mean())
    performances = np.where(to_retry, data[llm_col].to_numpy(), data[slm_col].to_numpy())
    avg_perf = float(np.mean(performances))

    # Slopes
    slm_llm_slope = (llm_perf - slm_perf) / (llm_cost - slm_cost)
    if avg_cost - slm_cost > 0:
        cascade_slope = (avg_perf - slm_perf) / (avg_cost - slm_cost)
        ibc_lift = (cascade_slope - slm_llm_slope) / slm_llm_slope if slm_llm_slope > 0 else 0.0
    else:
        cascade_slope = 0.0
        ibc_lift = 0.0

    return {
        "avg_perf": avg_perf,
        "avg_cost": avg_cost,
        "slm_perf": slm_perf,
        "llm_perf": llm_perf,
        "retry_rate": float(to_retry.mean()),
        "ibc_lift": ibc_lift,
        "cost_red_vs_llm": (1.0 - avg_cost / (llm_cost + verifier_cost)) * 100.0,
        "perf_ratio_vs_llm": (avg_perf / llm_perf) * 100.0 if llm_perf > 0 else 0.0,
    }


def run_evaluation(
    dataset_path: str = "benchmark/data/automix/automix.parquet",
    output_dir: str = "benchmark/automix/results",
    num_seeds: int = 10,
    val_size: int = 1000,
):
    print("=" * 80)
    print(f"Loading AutoMix benchmark dataset from: {dataset_path}")
    print("=" * 80)

    df = pd.read_parquet(dataset_path)
    os.makedirs(output_dir, exist_ok=True)

    datasets = ["coqa", "cnli", "narrative_qa", "quality", "qasper"]
    all_summary_rows = []
    dataset_details = {}

    adapter = KruschAutomixAdapter()
    candidate_thresholds = [0.15, 0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

    for dset in datasets:
        print(f"\n[{dset.upper()}] Evaluating across {num_seeds} cross-validation seeds...")
        dset_df = df[df["dataset"] == dset].copy()
        train_df = dset_df[dset_df["split"] == "train"]
        val_df = dset_df[dset_df["split"] == "val"].reset_index(drop=True)

        if len(val_df) == 0:
            print(f"Skipping {dset}: no validation split found.")
            continue

        val_contexts = val_df["base_ctx"].tolist()
        val_questions = val_df["question"].tolist()
        val_slm_ans = val_df["llama13b_pred_ans"].tolist()
        val_p_ver = val_df["p_ver_13b"].to_numpy()

        # Baselines
        slm_perf = float(val_df["llama13b_f1"].mean())
        llm_perf = float(val_df["llama70b_f1"].mean())
        base_ref = PAPER_BASELINES.get(dset, {})

        # Standard Verifier Mode (verifier_cost = 1)
        seed_lifts_std = []
        seed_costs_std = []
        seed_perfs_std = []
        seed_retries_std = []

        # Zero-Overhead Heuristic Mode (verifier_cost = 0)
        seed_lifts_zero = []
        seed_costs_zero = []
        seed_perfs_zero = []

        for seed in range(num_seeds):
            # Tune threshold on training sample
            sample_size = min(val_size, len(train_df))
            sample_train = train_df.sample(sample_size, random_state=seed)

            train_ctx = sample_train["base_ctx"].tolist()
            train_q = sample_train["question"].tolist()
            train_ans = sample_train["llama13b_pred_ans"].tolist()
            train_p = sample_train["p_ver_13b"].to_numpy()

            best_tau_std = 0.45
            best_lift_std = -1e9

            best_tau_zero = 0.45
            best_lift_zero = -1e9

            for tau in candidate_thresholds:
                # 1. Standard mode
                train_retry_std = adapter.batch_should_cascade(
                    train_ctx, train_q, train_ans, train_p, threshold=tau, dataset=dset
                )
                m_train_std = compute_metrics(sample_train, train_retry_std, verifier_cost=1.0)
                if m_train_std["ibc_lift"] > best_lift_std:
                    best_lift_std = m_train_std["ibc_lift"]
                    best_tau_std = tau

                # 2. Zero-token mode
                train_retry_zero = adapter.batch_should_cascade(
                    train_ctx, train_q, train_ans, None, threshold=tau, dataset=dset
                )
                m_train_zero = compute_metrics(sample_train, train_retry_zero, verifier_cost=0.0)
                if m_train_zero["ibc_lift"] > best_lift_zero:
                    best_lift_zero = m_train_zero["ibc_lift"]
                    best_tau_zero = tau

            # Evaluate on full validation set
            val_retry_std = adapter.batch_should_cascade(
                val_contexts, val_questions, val_slm_ans, val_p_ver, threshold=best_tau_std, dataset=dset
            )
            m_val_std = compute_metrics(val_df, val_retry_std, verifier_cost=1.0)
            seed_lifts_std.append(m_val_std["ibc_lift"])
            seed_costs_std.append(m_val_std["avg_cost"])
            seed_perfs_std.append(m_val_std["avg_perf"])
            seed_retries_std.append(m_val_std["retry_rate"])

            val_retry_zero = adapter.batch_should_cascade(
                val_contexts, val_questions, val_slm_ans, None, threshold=best_tau_zero, dataset=dset
            )
            m_val_zero = compute_metrics(val_df, val_retry_zero, verifier_cost=0.0)
            seed_lifts_zero.append(m_val_zero["ibc_lift"])
            seed_costs_zero.append(m_val_zero["avg_cost"])
            seed_perfs_zero.append(m_val_zero["avg_perf"])

        # Aggregate metrics
        mean_lift_std = float(np.mean(seed_lifts_std))
        mean_cost_std = float(np.mean(seed_costs_std))
        mean_perf_std = float(np.mean(seed_perfs_std))
        mean_retry_std = float(np.mean(seed_retries_std))

        mean_lift_zero = float(np.mean(seed_lifts_zero))
        mean_cost_zero = float(np.mean(seed_costs_zero))
        mean_perf_zero = float(np.mean(seed_perfs_zero))

        dataset_details[dset] = {
            "val_queries": len(val_df),
            "slm_perf": slm_perf,
            "llm_perf": llm_perf,
            "krusch_std": {
                "ibc_lift": mean_lift_std,
                "cost": mean_cost_std,
                "perf": mean_perf_std,
                "retry_rate": mean_retry_std,
            },
            "krusch_zero": {
                "ibc_lift": mean_lift_zero,
                "cost": mean_cost_zero,
                "perf": mean_perf_zero,
            },
            "paper_baselines": base_ref,
        }

        all_summary_rows.append({
            "Dataset": dset.upper(),
            "Val Queries": len(val_df),
            "SLM (13B) F1": slm_perf * 100.0,
            "LLM (70B) F1": llm_perf * 100.0,
            "AutoMix POMDP Lift (%)": base_ref.get("pomdp_lift", np.nan) * 100.0,
            "AutoMix Thresh Lift (%)": base_ref.get("thresh_lift", np.nan) * 100.0,
            "Krusch Cascade F1": mean_perf_std * 100.0,
            "Krusch Avg Cost": mean_cost_std,
            "Krusch Lift (%)": mean_lift_std * 100.0,
            "Krusch Cost Red. (%)": (1.0 - mean_cost_std / 51.0) * 100.0,
            "Krusch Zero-Overhead Lift (%)": mean_lift_zero * 100.0,
        })

    summary_df = pd.DataFrame(all_summary_rows)

    # Benchmark Latency on CPU
    print("\nBenchmarking AutoMix cascade routing latency (10,000 checks)...")
    sample_q = df["question"].head(10000).tolist()
    sample_ctx = df["base_ctx"].head(10000).tolist()
    sample_ans = df["llama13b_pred_ans"].head(10000).tolist()
    t0 = time.perf_counter()
    adapter.batch_should_cascade(sample_ctx, sample_q, sample_ans)
    t_elapsed = time.perf_counter() - t0
    avg_latency_us = (t_elapsed / len(sample_q)) * 1_000_000.0
    qps = len(sample_q) / t_elapsed
    print(f"Average cascade decision latency: {avg_latency_us:.2f} µs/query ({qps:,.0f} QPS)")

    # Save outputs
    summary_csv = os.path.join(output_dir, "automix_eval_summary.csv")
    summary_json = os.path.join(output_dir, "automix_eval_summary.json")

    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump({
            "datasets": dataset_details,
            "summary": all_summary_rows,
            "latency_us": avg_latency_us,
            "qps": qps,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("AUTOMIX BENCHMARK EVALUATION SUMMARY TABLE (14,571 Validation Queries)")
    print("=" * 80)
    display_cols = [
        "Dataset",
        "Val Queries",
        "SLM (13B) F1",
        "LLM (70B) F1",
        "AutoMix POMDP Lift (%)",
        "Krusch Cascade F1",
        "Krusch Avg Cost",
        "Krusch Lift (%)",
        "Krusch Cost Red. (%)",
    ]
    print(tabulate(summary_df[display_cols], headers="keys", tablefmt="pipe", floatfmt=".2f"))

    return summary_df, dataset_details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoMix Evaluation for Krusch Cascade Router")
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmark/data/automix/automix.parquet",
        help="Path to automix parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/automix/results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of random seeds for cross-validation",
    )

    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_seeds=args.num_seeds,
    )
