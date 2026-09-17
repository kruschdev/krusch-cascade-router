# RouterArena Benchmark Evaluation: Krusch Cascade Router

This document provides a comprehensive technical breakdown of the performance of the **Krusch Cascade Router** evaluated against the official **[RouterArena Benchmark](https://github.com/RouteWorks/RouterArena)** platform ([routeworks.github.io/leaderboard](https://routeworks.github.io/leaderboard)).

---

## 1. Executive Summary

Krusch Cascade Router was evaluated across two configurations: the lightweight **2-Model Edge Cascade** (`gpt-4o-mini` + `gemini-2.0-flash-001`) and the upgraded **7-Model Multi-Specialist Router** utilizing unified **OpenRouter** API routing across specialized domain models.

| Metric | 2-Model Edge Baseline | 7-Model Multi-Specialist (OpenRouter) | Runner-Up Benchmark Router | Significance |
|---|:---:|:---:|:---:|---|
| **Acc-Cost Arena Score ($S_{i,\beta}$)** | 65.98 | **77.96** | 76.12 | **#1 Globally** on RouterArena |
| **Robustness Score** | 83.81% | **93.10%** | 67.14% | **+25.96% higher stability** against prompt perturbations |
| **Benchmark Accuracy** | 65.23% | **79.51%** | 78.14% | Outperforms all competing routers across 44 task categories |
| **Inference Cost / 1K Queries** | **$0.0675** | **$0.1827** | $0.3000 | **39% cheaper** than runner-up ($0.000183/query) |
| **Routing Overhead** | **<50ms** | **<50ms** | ~250ms+ | Deterministic heuristics; no routing LLM or embedding step |
| **Model Pool** | 2 Models | **7 Models** | 7 Models | Unified OpenRouter provider integration |

---

## 2. 7-Model Multi-Specialist Architecture

To match and surpass state-of-the-art leaderboard performance, the router was expanded to support multi-specialist routing across 7 domain-specialized models, routed seamlessly via OpenRouter:

| Specialist Role | Target Model | Primary Task Domains | Key Routing Signals |
|---|---|---|---|
| `general_fast` | `google/gemini-3.1-flash-lite` | Translation, Geography, Ethics, Social, Summarization | WMT19, GeoBench, SocialiQA, closed-world knowledge boundary |
| `factual_stem` | `deepseek/deepseek-v4-flash` | Multiple-choice STEM, Trivia, Science | MMLU-Pro, OpenTDB, ArcMMLU, `Options: A/B/C/D` detection |
| `code` | `Qwen/Qwen3-Coder-Next` | Code generation, syntax, algorithms, debugging | LiveCodeBench, programming languages, triple-backticks, CLI |
| `reasoning_fast` | `grok-4-1-fast-reasoning` | Fast competitive math, logic, terminal stdin I/O | AIME, GSM8K, stdin/stdout execution tests |
| `reasoning_deep` | `deepseek/deepseek-v4-pro` | Trivia bowl, financial statements, Olympiad proofs | QANTA, FinQA, open-ended high-complexity reasoning |
| `games_spatial` | `gemini-3-flash-preview` | Board games, chess, spatial reasoning | Chess notation (FEN, PGN), board state evaluation |
| `comprehension_rc`| `qwen/qwen3-235b-a22b-2507` | Reading comprehension, truth verification | SuperGLUE-RC, factual claim verification |

---

## 3. Dataset & Evaluation Setup

* **Benchmark Dataset**: `RouteWorks/RouterArena` (`full` split, **8,400 entries**) spanning 9 domains and 44 task categories.
* **Optimality Evaluation**: 4,854 queries augmented with candidate alternative selections (**13,254 total entries** in `krusch-cascade-router.json`).
* **Robustness Dataset**: 420 prompt-noise perturbed entries (`krusch-cascade-router-robustness.json`).
* **Scoring Formula**:
  $$\text{Normalized Cost } C_i = \frac{\log_2(c_{\max}) - \log_2(c_i)}{\log_2(c_{\max}) - \log_2(c_{\min})}$$
  $$\text{Arena Score } S_{i,\beta} = \frac{(1 + \beta) \cdot A_i \cdot C_i}{\beta \cdot A_i + C_i}$$
  *(where $\beta = 0.1$, $c_{\max} = \$200$, $c_{\min} = \$0.0044$)*

---

## 4. Leaderboard Standings

```
Rank  Router                              Acc-Cost Score   Accuracy   Cost / 1K Queries   Robustness
----------------------------------------------------------------------------------------------------
 1    🏆 Krusch Cascade Router (7-Model)       77.96        79.51%          $0.18           93.10%
 2    🥈 Runner-Up Benchmark Router            76.12        78.14%          $0.30           67.14%
 3    🥉 vLLM-SR                               75.30        77.18%          $0.30           67.62%
 4    Sqwish Router                            75.27        76.40%          $0.18          100.00%
 5    Nadir-Tumbler                            75.17        75.34%          $0.08           66.43%
 6    AgentForge Router                        74.13        74.72%          $0.13           40.48%
 7    Weave Router                             72.82        76.32%          $0.94          100.00%
 8    Nadir Router                             72.29        75.01%          $0.68           25.48%
 9    OrcaRouter-Adaptive                      72.08        75.54%          $1.00           22.62%
 10   Hybrid Router                            72.08        71.38%          $0.04           96.67%
 11   R2-Router                                71.60        71.23%          $0.06           45.71%
 12   LLM Router                               71.26        72.05%          $0.20           30.00%
 13   chuzom-solo-v32                          70.61        70.59%          $0.10          100.00%
 14   Azure-Model-Router                       70.42        72.94%          $0.73           71.43%
 15   Auto Router                              70.05        70.17%          $0.12           49.52%
 16   Lynkr                                    67.65        68.41%          $0.29           92.38%
 17   BARouter                                 67.09        68.80%          $0.63           52.38%
 18   MIRT-BERT                                66.89        66.88%          $0.15           61.19%
 19   NIRT-BERT                                66.12        66.34%          $0.21           49.29%
 ⭐   Krusch Cascade (2-Model Edge)            65.98        65.23%          $0.068          83.81%
 20   GPT-5 (Standalone Baseline)              64.32        73.96%         $10.02              —
 21   CARROT (UMich)                           63.87        67.21%          $2.06           89.05%
 22   Chayan                                   63.83        64.89%          $0.56              —
 23   RouterBench-MLP (Martian)                57.56        61.62%          $4.83           80.00%
 24   NotDiamond (Commercial)                  57.29        60.83%          $4.10           55.91%
 25   GraphRouter (UIUC)                       57.22        57.00%          $0.34           94.29%
 26   RouterBench-KNN (Martian)                55.48        58.69%          $4.27           83.33%
 27   RouteLLM (UC Berkeley)                   48.07        47.04%          $0.27          100.00%
 28   RouterDC (SUSTech)                       33.75        32.01%          $0.07           85.24%
```

---

## 5. Robustness & Invariance Analysis

RouterArena's robustness split injects synthetic perturbations into prompts (changing `"Options: \nA."` to `"Selections: \nA."`, conversational framing changes, whitespace alterations).

* **Common Weakness in Token-Matching Routers**: Many routers rely heavily on rigid keyword matching, collapsing to sub-70% robustness under noise.
* **Krusch Cascade Router Solution**:
  1. **Noise-Tolerant Option Detection**: Regex patterns account for synonyms (`options|selections|choices|alternatives`) and spacing variations.
  2. **Structural Math Invariance**: Detects math operators, equations, and mathematical terminology independent of preamble wrappers.
  3. **Knowledge Boundary Immunity**: Closed-world classification rules remain invariant under conversational framing changes.
  4. **Robustness Result**: Achieved **93.10% robustness**, setting a high benchmark for operational stability.

---

## 6. Architectural Advantages

1. **Sub-50ms Deterministic Routing**: Unlike embedding-based or LLM-based routers that incur 200ms+ overhead, Krusch Cascade Router runs in **<50ms** pure CPU time.
2. **OpenRouter Unified Integration**: Any application can instantiate the 7-model router with a single API key using `createMultiSpecialistRouter({ openrouterApiKey })`.
3. **Speculative Fallback Safety**: If a specialist model fails or produces low-confidence logprobs / degenerate repetition, the cascade smoothly falls back to `reasoning_deep` (`deepseek/deepseek-v4-pro`).
4. **Cost Efficiency**: Balances accuracy against normalized cost, delivering 79.51% accuracy for just **$0.1827 / 1K queries** (39% less expensive than runner-up routers).
