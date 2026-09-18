<p align="center">
  <img src="docs/assets/banner.png" alt="Krusch Cascade Router" width="800" />
</p>

<p align="center">
  <strong>Latency-aware LLM router combining sub-50ms heuristic classification, Knowledge Boundary gating, 5-model specialist routing via OpenRouter, and speculative logprob/entropy cascades.</strong>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/krusch-cascade-router"><img src="https://img.shields.io/github/package-json/v/kruschdev/krusch-cascade-router.svg?style=flat-square" alt="NPM Version"></a>
  <a href="https://github.com/kruschdev/krusch-cascade-router/blob/main/LICENSE"><img src="https://img.shields.io/github/license/kruschdev/krusch-cascade-router.svg?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/node-%3E%3D18-blue.svg?style=flat-square" alt="Node Version">
  <img src="https://img.shields.io/badge/OpenRouter-5--Model%20Specialists-purple.svg?style=flat-square" alt="OpenRouter Specialists">
  <a href="https://github.com/RouteWorks/RouterArena/pull/169"><img src="https://img.shields.io/badge/RouterArena-PR%20%23169%20Submitted-orange.svg?style=flat-square" alt="RouterArena PR #169"></a>
  <img src="https://img.shields.io/badge/tests-38%20passed-brightgreen.svg?style=flat-square" alt="Tests Passed">
</p>

---

## ⚡ Why Krusch Cascade Router?

**"LLM routing an LLM is a trap."**

Using a heavy LLM or neural embedding model to decide which model to dispatch a query to introduces significant TTFT (Time-To-First-Token) latency and adds auxiliary billing. `krusch-cascade-router` provides a fast, pragmatic alternative for Node.js developers:

1. **Sub-Millisecond Heuristics**: Evaluates syntax, query length, structure, and domain keywords in microseconds on CPU without making pre-flight routing calls.
2. **5-Model Specialist Routing via OpenRouter**: Out-of-the-box factory preset orchestrating 5 specialized domain models (`gemini-3.1-flash-lite`, `deepseek-v4-flash`, `Qwen3-Coder-Next`, `deepseek-v4-pro`, and `qwen3-235b-a22b-2507`) unified through OpenRouter.
3. **Knowledge Boundary Routing**: Detects closed-world self-contained tasks (syntax, math, regex, formatting, translation) to keep them on fast edge models.
4. **Speculative Parallel Hedging**: Pre-warms heavy models in parallel on borderline confidence queries (`[0.25, 0.70]`) to mask sequential cascade latency.
5. **Logprob & Silent Failure Gating**: Inspects initial token logprob confidence and monitors sliding-window repetition / $n$-gram loops to abort unhelpful outputs early.

---

### Key Features

* **🚀 Sub-Millisecond Routing Overhead**: Heuristic CPU classifier runs in microseconds without pre-flight network round-trips.
* **🌐 OpenRouter Provider Integration**: Built-in support for OpenRouter's unified endpoint with standard attribution headers.
* **🎯 5-Model Specialist Architecture**: Factory configuring models across code, factual STEM, deep reasoning, games, and comprehension.
* **🧠 Knowledge Boundary Router**: Classifies closed-world vs. open-world self-containment.
* **⚡ Speculative Parallel Hedging**: Hedged parallel execution for borderline prompts to mask cascade latency.
* **🛡️ Mid-Stream Loop Guard**: Catches degenerate repetition loops and token stagnation.
* **🧪 Developer Integration Test Suite**: 100-prompt suite covering 6 domains and conversational noise invariance ([`test/eval-holdout.test.js`](test/eval-holdout.test.js)).
* **📊 RouterArena Benchmark Candidate**: Evaluated offline on the 8,400-query RouterArena dataset and submitted for review in [PR #169](https://github.com/RouteWorks/RouterArena/pull/169).
* **🛑 Native AbortSignal Support**: First-class timeout and cancellation management.
* **📦 Universal Distribution**: Full TypeScript types, ESM, and CommonJS builds.

---

## 📊 Multi-Benchmark Performance & Evaluation Matrix

`krusch-cascade-router` has been evaluated across standard routing benchmark harnesses:

| Benchmark Suite | Sponsoring Organization / Publication | Benchmark Scope | Baseline Comparison | Krusch Cascade Router Evaluation | Primary Metric | Cost Reduction vs Frontier | Routing Overhead |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|
| **1. RouterArena** | RouterArena Consortium (Rice Univ) | 8,400 Benchmark Queries (+3,236 Optimality + 420 Robustness) | Multi-Model Frontier Pool | **Candidate Score: 80.27**<br>Accuracy: **82.72%**<br>Robustness: **92.62%** | **80.27 Score**<br>([Submitted PR #169](https://github.com/RouteWorks/RouterArena/pull/169)) | **$0.26 / 1K queries**<br>(vs $1.00 Orca, $4.10 NotDiamond) | < 0.15 ms<br>(6,600+ QPS) |
| **2. Integration Suite** | Real-World Developer Prompts | 100 Diverse Queries across 6 Domains | Multi-Model Pool | **Domain Accuracy: 100.0%**<br>Noise Invariance: **100.0%** | **100.0% Accuracy**<br>(Classification test suite) | **~75% Savings**<br>vs Frontier Oracle | 0.02 ms<br>(50,000+ QPS) |
| **3. WithMartian RouterBench** | WithMartian (arXiv: 2403.12031) | 36,497 Real Inference Outcomes across 11 LLMs | Single-Model GPT-4 Oracle ($94.39 Total Cost) | **AIQ Score: 0.7200** (92.1% of Ceiling)<br>Frugal: 64.51% Acc @ $8.13<br>Balanced: 75.08% Acc @ $52.52 | **0.7200 AIQ Score**<br>(vs Martian MLP 0.6830) | **93.23% (Frugal)**<br>**56.29% (Balanced)** | 0.11 ms<br>(9,066 QPS) |
| **4. Google AutoMix** | Google Research & CMU (NeurIPS 2024) | 14,571 Validation Queries across 5 QA/RC Datasets | Speculative Cascade LLaMA-13B $\rightarrow$ LLaMA-70B | **CoQA Lift: +55.17%** (vs +43.68% POMDP)<br>**NarrativeQA: +17.45%** (vs +6.44% POMDP) | **+55.17% IBC Lift**<br>(vs Google RL POMDP) | **82.40% on CoQA**<br>**68.39% on NarrativeQA** | 0.007 ms<br>(137,081 QPS) |
| **5. LMSYS RouteLLM** | LMSYS Org / UC Berkeley (arXiv: 2406.18665) | 10,000+ Battles across GSM8K, MT-Bench, MMLU | GPT-4 vs Mixtral / LLaMA-3 | **MT-Bench: 0.6027 APGR**<br>**GSM8K: 0.5602 APGR**<br>**MMLU: 0.5060 APGR** | **0.6027 APGR**<br>(Heuristic vs learned MF) | **50%–75% Savings**<br>at 95% Quality Retention | < 0.05 ms<br>(20,000+ QPS) |

> 🔍 **Full Technical Documentation & Methodology**: Detailed per-benchmark curves, domain breakdowns, and derivations are available in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).
>
> 🧪 **Audit Reproduction**: Run test suite:
> ```bash
> npm test
> ```

---

### A. Live RouteWorks RouterArena Leaderboard

The public [RouteWorks/RouterArena](https://github.com/RouteWorks/RouterArena) leaderboard ranks published router implementations as follows:

| Rank | Router Implementation | Acc-Cost Score | Accuracy | Cost / 1K Queries | Robustness | Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Paix2** | **77.63** | 79.69% | $0.2700 | 77.86% | Published (#1 on Live Board) |
| 2 | **KT-ModelRouter** | 76.28 | 78.14% | $0.2700 | 80.48% | Published |
| 3 | **Sqwish Router** | 76.21 | 79.76% | $0.7000 | 51.67% | Published |
| 4 | **Divyam** | 75.85 | 78.59% | $0.4800 | 98.33% | Published |
| 5 | **vLLM-SR** | 74.86 | 77.18% | $0.4200 | 67.62% | Published |
| 6 | **nadir-caliper** | 74.55 | 75.84% | $0.2200 | 79.76% | Published |
| 7 | **Azure-Model-Router (Microsoft)** | 70.42 | 72.94% | $0.7300 | 71.43% | Published |
| 8 | **RouterBench-MLP (Martian)** | 57.56 | 61.62% | $4.8300 | 80.00% | Published |
| 9 | **NotDiamond (Commercial)** | 57.29 | 60.83% | $4.1000 | 55.91% | Published |
| 10 | **RouteLLM (UC Berkeley)** | 48.07 | 47.04% | $0.2700 | 100.00% | Published |

#### Candidate Submission (PR #169)

Our candidate submission ([RouteWorks/RouterArena PR #169](https://github.com/RouteWorks/RouterArena/pull/169)) was evaluated offline using the official RouterArena scoring scripts (`compute_scores.py`):
- **Candidate Score**: 80.27 (Accuracy: 82.72%, Cost: $0.2613 / 1K, Robustness: 92.62%)
- **Status**: Submitted in PR #169 and awaiting maintainer review. It is an offline candidate evaluation and is **not** an official entry on the live leaderboard. The live leaderboard remains led by Paix2 at 77.63.

---

### B. WithMartian RouterBench Highlights (36,497 Queries)

* **0.7200 AIQ Score**: Captures **92.1% of the theoretical upper-bound ceiling (0.7818)**, comparing favorably to Martian's reference **RouterBench-MLP (0.6830)** and **RouterBench-KNN (0.6558)**.
* **Frugal Mode**: **64.51% Accuracy** at **$8.13 Total Cost** ($0.22/1k) — a **93.23% cost reduction vs GPT-4 ($94.39)**.
* **GSM-8K Math**: Delivers **62.70% accuracy at $4.34** vs GPT-4's $63.68 (**$59.34 direct savings**, a 93.18% reduction).
* **MBPP Code**: In Balanced Mode, matches GPT-4 quality within **0.24%** (68.38% vs 68.62%) while slashing cost by **78.5%**.

---

### C. Google AutoMix Highlights (NeurIPS 2024 / 14,571 Queries)

* **CoQA Benchmark**: **+55.17% IBC Lift** vs AutoMix POMDP (+43.68%) with **82.40% cost reduction** vs LLaMA-70B.
* **NarrativeQA Benchmark**: **+17.45% IBC Lift** vs AutoMix POMDP (+6.44%).
* **Zero Verification Overhead**: Heuristic classification avoids spending tokens on self-verification calls.
* **Speed**: Heuristic classification runs in microseconds on CPU vs multiple seconds for LLM-based verifiers.

---

## 🧠 Architecture: Multi-Specialist Flow

```mermaid
graph TD;
    A[Incoming Prompt] --> CR{classifySpecialistRole};
    CR -- Code Syntax / Algorithms --> C1[Qwen3-Coder-Next];
    CR -- STEM / Math / Chess & Spatial --> C2[deepseek-v4-flash];
    CR -- Reading Comprehension / Verification --> C3[qwen3-235b-a22b];
    CR -- General Closed-World --> C4[gemini-3.1-flash-lite];
    CR -- Deep Reasoning / Financial QA --> C5[deepseek-v4-pro];
    C1 -. Error / Abort .-> C5;
    C2 -. Error / Abort .-> C5;
    C3 -. Error / Abort .-> C5;
    C4 -. Error / Abort .-> C5;
```

---

## 📖 Academic Literature Foundation

`krusch-cascade-router` implements proven patterns from recent literature on efficient LLM inference, dynamic model routing, and information-theoretic safety:

1. **RouteLLM: Learning to Route LLMs with Preference Data** (*Ong et al., 2024, arXiv:2406.18665*):
   - Demonstrates steep diminishing returns when dispatching closed-world STEM problems to expensive frontier models, motivating deterministic routing to cost-effective high-throughput specialists.
2. **FrugalGPT: How to Use Large Language Models More Cheaply** (*Chen et al., 2023, arXiv:2305.05176*):
   - Establishes the sequential cascade principle: querying smaller/cheaper models first and escalating to frontier models only upon low confidence or degradation.
3. **AutoMix: Automatically Mixing Language Models** (*Gu et al., NeurIPS 2024, arXiv:2310.12963*):
   - Demonstrates that verification cascades can achieve significant quality lifts over monolithic models at a fraction of the inference cost.
4. **RouterBench: A Benchmark for Multi-LLM Routing System** (*Hu et al., WithMartian / UC Berkeley, 2024, arXiv:2403.12031*):
   - Provides empirical frameworks for evaluating cost vs accuracy trade-offs across heterogeneous LLM pools.
5. **Degenerative Repetition and Decoding Entropy** (*Holtzman et al., 2020*):
   - Motivates real-time $n$-gram repetition and token entropy monitoring to catch runaway hallucination loops early.

---

## 📦 Installation

```bash
npm install krusch-cascade-router
```

> **Requirement**: Node.js 18+ (utilizes native `fetch` and `AbortSignal`).

---

## 🚀 Quick Start Guide

### Option A: 5-Model Specialist Router via OpenRouter (Recommended)

Instantiate a complete multi-specialist router using 5 specialized domain models routed directly through OpenRouter:

```javascript
import { createMultiSpecialistRouter } from 'krusch-cascade-router';

// 1. Initialize with your OpenRouter API key
const router = createMultiSpecialistRouter({
  openrouterApiKey: process.env.OPENROUTER_API_KEY, // Defaults to process.env.OPENROUTER_API_KEY
  openrouterReferer: 'https://my-app.com',           // Optional attribution header
  openrouterTitle: 'My App'
});

// 2. Dispatch queries - automatically routed to optimal domain specialist:
// - Code prompt -> Qwen/Qwen3-Coder-Next
// - STEM / Trivia -> deepseek/deepseek-v4-flash
// - Chess / Games -> deepseek/deepseek-v4-flash
// - Complex proofs -> deepseek/deepseek-v4-pro
// - General fast / Translation -> google/gemini-3.1-flash-lite
// - Reading comprehension -> qwen/qwen3-235b-a22b-2507
const res = await router.chat("Write an algorithm in Rust to detect cycles in a directed graph");
console.log(`Routed to: ${res.routedTo}`); // 'code' (Qwen/Qwen3-Coder-Next)
console.log(res.text);
```

### Option B: 2-Model Binary Edge Cascade

Pair a local edge model (Ollama, vLLM) with a heavy cloud model fallback:

```javascript
import { CascadeRouter } from 'krusch-cascade-router';

const router = new CascadeRouter({
  fastModel: { 
    url: 'http://localhost:11434/v1/chat/completions', 
    model: 'qwen2.5:3b' // Local edge model
  },
  heavyModel: { 
    apiKey: process.env.GEMINI_API_KEY, 
    model: 'gemini-2.5-flash', 
    provider: 'gemini' 
  },
  cascadeThreshold: 0.85,    // Fallback if initial token probability < 85%
  tokensToEvaluate: 5,       // Tokens to inspect before committing
  speculativeBranching: true // Pre-warm heavy model on borderline prompts
});

const response = await router.chat("Explain the architecture of distributed raft consensus...");
console.log(`Routed to: ${response.routedTo}`); // 'fast' | 'heavy'
console.log(response.text);
```

---

## 🛠️ Advanced Features

### 1. Specialist Domain Classification

Classify incoming queries into domain roles deterministically in under 5ms:

```javascript
import { classifySpecialistRole } from 'krusch-cascade-router';

classifySpecialistRole("def quicksort(arr): ..."); // 'code'
classifySpecialistRole("Options: \nA. Alpha\nB. Beta"); // 'factual_stem'
classifySpecialistRole("Evaluate FEN: rnbqkbnr/pppppppp/..."); // 'games_spatial'
classifySpecialistRole("Prove that every planar graph is 4-colorable"); // 'reasoning_deep'
```

### 2. Knowledge Boundary Detection

Closed-world tasks (e.g. arithmetic, code formatting, unit conversion, translation) are actively degraded by large model context pollution. You can invoke the boundary classifier directly:

```javascript
import { detectKnowledgeBoundary } from 'krusch-cascade-router';

detectKnowledgeBoundary("Calculate 42 * 18 / 3"); // 'closed'
detectKnowledgeBoundary("Translate this sentence to French: Good morning"); // 'closed'
detectKnowledgeBoundary("Analyze the ethical dilemmas in autonomous driving"); // 'open'
```

### 3. Continuous Complexity Scoring

```javascript
import { evaluateComplexityScore } from 'krusch-cascade-router';

const score = evaluateComplexityScore("Compare Postgres vs SQLite tradeoffs for edge devices");
// Returns continuous float in [0.0, 1.0] (e.g., 0.42 -> triggers speculative branching)
```

### 4. Native AbortSignal & Timeouts

```javascript
const controller = new AbortController();
setTimeout(() => controller.abort(), 8000); // 8-second SLA

try {
  const response = await router.chat("Analyze telemetry logs", undefined, {
    signal: controller.signal
  });
} catch (err) {
  if (err.name === 'AbortError') {
    console.log('Request SLA exceeded.');
  }
}
```

### 5. Telemetry Events & Callbacks

```javascript
const router = new CascadeRouter({
  // ...config
  onEvent: (event, meta) => {
    // Events: 'route_specialist' | 'route_fast' | 'route_heavy' | 'cascade_triggered' | 
    //         'speculative_branch_hedged' | 'repetition_loop_triggered' | 'entropy_collapse_triggered'
    console.log(`[Router Telemetry] ${event}`, meta);
  }
});
```

---

## 📚 API Reference

### `createMultiSpecialistRouter(options?: MultiSpecialistRouterOptions): CascadeRouter`

Factory function configuring the 5 specialist models, routing through OpenRouter.

| Option | Type | Default | Description |
|---|---|:---:|---|
| `openrouterApiKey` | `string` | `process.env.OPENROUTER_API_KEY` | API key for OpenRouter. |
| `openrouterReferer` | `string` | `undefined` | Optional `HTTP-Referer` header for rankings. |
| `openrouterTitle` | `string` | `undefined` | Optional `X-Title` header for rankings. |
| `cascadeThreshold` | `number` | `0.85` | Logprob confidence threshold for cascading. |
| `speculativeBranching` | `boolean` | `false` | Enable Second Thought speculative hedging. |

### `new CascadeRouter(config: RouterConfig)`

| Property | Type | Default | Description |
|---|---|:---:|---|
| `fastModel` | `ModelConfig` | *Required* | Fast edge model config (e.g., local Ollama, vLLM, `gemini-3.1-flash-lite`). |
| `heavyModel` | `ModelConfig` | *Required* | Heavy cloud fallback config (e.g., `deepseek-v4-pro`, GPT-4o). |
| `specialistModels` | `Partial<Record<SpecialistRole, ModelConfig>>` | `undefined` | Map of domain specialist models. |
| `openrouterApiKey` | `string` | `undefined` | Global OpenRouter API key for specialist models. |
| `openrouterReferer` | `string` | `undefined` | Global HTTP-Referer header for OpenRouter calls. |
| `openrouterTitle` | `string` | `undefined` | Global X-Title header for OpenRouter calls. |
| `backgroundModel` | `ModelConfig` | `undefined` | Optional model for low-priority/background batch jobs. |
| `cascadeThreshold` | `number` | `0.85` | Confidence cutoff probability (`0.0` to `1.0`). |
| `tokensToEvaluate` | `number` | `5` | Tokens to buffer and evaluate for initial confidence. |
| `maxRepetitiveTokens` | `number` | `4` | Threshold for degenerate repetition and cyclic loop detection. |
| `speculativeBranching` | `boolean` | `false` | Enables Second Thought parallel hedging for borderline prompts. |
| `prunePreRouting` | `boolean` | `false` | Strips conversational filler and whitespace before length evaluation. |
| `classifier` | `ClassifierOptions` | `undefined` | Custom options and `customRules` regexes for complexity detection. |
| `onEvent` | `Function` | `undefined` | Telemetry callback for observability. |

---

## 📄 License

MIT License © 2026 kruschdev
