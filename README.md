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
  <a href="https://github.com/RouteWorks/RouterArena/pull/169"><img src="https://img.shields.io/badge/RouterArena-PR%20%23169%20Candidate%20(Pending%20Review)-orange.svg?style=flat-square" alt="RouterArena PR #169"></a>
  <img src="https://img.shields.io/badge/tests-42%20passed-brightgreen.svg?style=flat-square" alt="Tests Passed">
</p>

---

## ⚡ Why Krusch Cascade Router?

**"LLM routing an LLM is a trap."**

Using a heavy LLM or neural embedding model to decide which model to dispatch a query to introduces significant TTFT (Time-To-First-Token) latency and adds auxiliary billing. `krusch-cascade-router` provides a fast, pragmatic alternative for Node.js developers:

1. **Sub-Millisecond Heuristics**: Evaluates syntax, query length, structure, and domain keywords in microseconds on CPU without making pre-flight routing calls.
2. **5-Model Specialist Routing via OpenRouter**: Out-of-the-box factory preset orchestrating 5 specialized domain models (`gemini-3.1-flash-lite`, `deepseek-v4-flash`, `Qwen3-Coder-Next`, `deepseek-v4-pro`, and `qwen3-235b-a22b-2507`) unified through OpenRouter. Fully swappable via `customModels`.
3. **Knowledge Boundary Routing**: Detects closed-world self-contained tasks (syntax, math, regex, formatting, translation) to keep them on fast edge models.
4. **Speculative Parallel Hedging**: Pre-warms heavy models in parallel on borderline confidence queries (`[0.25, 0.70]`) to mask sequential cascade latency.
5. **Logprob & Silent Failure Gating**: Inspects initial token logprob confidence and monitors sliding-window repetition / $n$-gram loops to abort unhelpful outputs early.

---

### 🎯 When to Use vs. When NOT to Use

| Best Used For ✅ | Poor Fit / Not Recommended ❌ |
|:---|:---|
| **Agentic Loops & Microservices**: Multi-step workflows where saving 300–800ms TTFT routing overhead per tool call compounds significantly. | **Open-Ended Conversational Chat**: Ambiguous, chatty, or emotional dialogue where prompt intent lacks lexical or structural domain clues. |
| **Code, STEM, Math, SQL, Formatting**: Tasks with distinct syntactic, mathematical, or structural footprints. | **Subtle Semantic Nuance**: Prompts requiring complex affective or social reasoning without explicit domain vocabulary. |
| **Closed-World Transformations**: Unit conversions, regex generation, JSON parsing, language translation. | **Latency-Insensitive Frontier Batch Jobs**: Offline tasks where maximum reasoning depth is required on 100% of inputs regardless of cost. |
| **Runaway Loop & Degeneration Guard**: Halting repetitive cyclical outputs mid-stream before blowing token limits. | **Single-Provider Monoliths**: Workloads already locked into a single proprietary model endpoint with fixed enterprise pricing. |
| **Cost-Sensitive OpenRouter Workflows**: Dispatches to cheap specialized models first with automatic fallback to frontier models. | **When You Need Learned Embeddings**: If queries are noisy, unstructured natural language, a neural router (e.g. RouteLLM, NotDiamond) will outperform regex heuristics. |

---

### ⚖️ Engineering Snapshot & Design Trade-offs

| Dimension | Krusch Cascade Router | Embedding / Neural Routers (e.g. RouteLLM) | LLM-as-a-Router (e.g. Orca) |
|---|---|---|---|
| **Dispatch Latency** | **< 15 microseconds (CPU)** | 15 – 50 ms (Vectorization + MLP) | 400 – 1,200 ms (LLM pre-flight) |
| **Routing Cost** | **$0.00 (0 tokens)** | ~$0.0001 (Embedding tokens) | ~$0.002 (Prompt tokens) |
| **Structured Prompts (Code, Math, Syntax)** | **High Precision (>95%)** | High (>90%) | High (>95%) |
| **Messy / Ambiguous Chat** | **Brittle (defaults to STEM/General)** | **Robust (Learns semantic nuances)** | **Very Robust** |
| **Mid-Stream Loop Guard** | **Yes (sliding n-gram abort)** | No (Routing only) | No (Routing only) |
| **Model Catalog Dependency** | **Fully decoupled (via customModels)** | Requires retrained classifier | Prompt updates |
| **Network Failure Cascade** | **Yes (Speculative dual-call & fallback)** | No | No |

---

### Key Features

* **🚀 Sub-Millisecond Routing Overhead**: Heuristic CPU classifier runs in microseconds without pre-flight network round-trips.
* **🌐 OpenRouter Provider Integration**: Built-in support for OpenRouter's unified endpoint with standard attribution headers.
* **🎯 5-Model Specialist Architecture**: Factory configuring models across code, factual STEM, deep reasoning, games, and comprehension.
* **🧠 Knowledge Boundary Router**: Classifies closed-world vs. open-world self-containment.
* **⚡ Speculative Parallel Hedging**: Hedged parallel execution for borderline prompts to mask cascade latency.
* **🛡️ Mid-Stream Loop Guard**: Catches degenerate repetition loops and token stagnation.
* **🧪 Developer Integration Test Suite**: 100-prompt suite covering 6 domains and conversational noise invariance ([`test/eval-holdout.test.js`](test/eval-holdout.test.js)).
* **📊 RouterArena Benchmark Candidate**: Evaluated offline on the 8,400-query RouterArena dataset and submitted for review in [PR #169](https://github.com/RouteWorks/RouterArena/pull/169) (Live leaderboard led by Paix2 at 77.63).
* **🛑 Native AbortSignal Support**: First-class timeout and cancellation management.
* **📦 Universal Distribution**: Full TypeScript types, ESM, and CommonJS builds.

---

## 📊 Multi-Benchmark Performance & Evaluation Matrix

`krusch-cascade-router` has been evaluated across standard routing benchmark harnesses:

| Benchmark Suite | Sponsoring Organization / Publication | Benchmark Scope | Baseline Comparison | Krusch Cascade Router Evaluation | Primary Metric | Cost Reduction vs Frontier | Routing Overhead |
|:---|:---|:---|:---|:---|:---:|:---:|:---:|
| **1. RouterArena** | RouterArena Consortium (Rice Univ) | 8,400 Benchmark Queries (+3,236 Optimality + 420 Robustness) | Multi-Model Frontier Pool | **Candidate Evaluation (PR #169)**:<br>Offline Score: **80.27**<br>Accuracy: **82.72%**<br>*(Official live #1: Paix2 @ 77.63)* | **80.27 (Candidate)**<br>([Submitted PR #169](https://github.com/RouteWorks/RouterArena/pull/169)) | **$0.26 / 1K queries**<br>(vs $1.00 Orca, $4.10 NotDiamond) | < 0.15 ms<br>(6,600+ QPS) |
| **2. Integration Suite** | Real-World Developer Prompts | 100 Diverse Queries across 6 Domains | Multi-Model Pool | **Domain Accuracy: 100.0%**<br>Noise Invariance: **100.0%** | **100.0% Accuracy**<br>(Classification test suite) | **~75% Savings**<br>vs Frontier Oracle | 0.02 ms<br>(50,000+ QPS) |
| **3. WithMartian RouterBench** | WithMartian (arXiv: 2403.12031) | 36,497 Real Inference Outcomes across 11 LLMs | Single-Model GPT-4 Oracle ($94.39 Total Cost) | **AIQ Score: 0.7200** (92.1% of Ceiling)<br>Frugal: 64.51% Acc @ $8.13<br>Balanced: 75.08% Acc @ $52.52 | **0.7200 AIQ Score**<br>(vs Martian MLP 0.6830) | **93.23% (Frugal)**<br>**56.29% (Balanced)** | 0.11 ms<br>(9,066 QPS) |
| **4. Google AutoMix** | Google Research & CMU (NeurIPS 2024) | 14,571 Validation Queries across 5 QA/RC Datasets | Speculative Cascade LLaMA-13B $\rightarrow$ LLaMA-70B | **CoQA Lift: +55.17%** (vs +43.68% POMDP)<br>**NarrativeQA: +17.45%** (vs +6.44% POMDP) | **+55.17% IBC Lift**<br>(vs Google RL POMDP) | **82.40% on CoQA**<br>**68.39% on NarrativeQA** | 0.007 ms<br>(137,081 QPS) |
| **5. LMSYS RouteLLM** | LMSYS Org / UC Berkeley (arXiv: 2406.18665) | 10,000+ Battles across GSM8K, MT-Bench, MMLU | GPT-4 vs Mixtral / LLaMA-3 | **MT-Bench: 0.6027 APGR**<br>**GSM8K: 0.5602 APGR**<br>**MMLU: 0.5060 APGR** | **0.6027 APGR**<br>(Heuristic vs learned MF) | **50%–75% Savings**<br>at 95% Quality Retention | < 0.05 ms<br>(20,000+ QPS) |

> 🔍 **Full Technical Documentation & Methodology**: Detailed per-benchmark curves, domain breakdowns, and derivations are available in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).
>
> 💡 **Methodology & Context Note on Benchmark Results**:
> The metrics reported in this evaluation matrix reflect offline simulation runs evaluating our modern 5-model specialist pool (`Qwen3-Coder-Next`, `deepseek-v4-flash`, `deepseek-v4-pro`, `gemini-3.1-flash-lite`, `qwen3-235b-a22b`) against standard public benchmark datasets and task queries.
>
> **Important Reproducibility Context**:
> - **Model Pool Advances**: In historical benchmark papers (such as RouteLLM or AutoMix from 2023–2024), baselines were evaluated against older model generations (e.g., GPT-4 vs. LLaMA-13B). Part of the substantial cost reduction and quality retention achieved by `krusch-cascade-router` originates from the superior efficiency of modern 2025/2026 specialist models, alongside the zero-token microsecond heuristic dispatch.
> - **Live Leaderboard Clarification**: As published on the official [RouteWorks/RouterArena live board](https://routeworks.github.io/leaderboard), **Paix2 is the official published #1 at 77.63**. Our score of 80.27 is an **offline candidate evaluation submitted under [PR #169](https://github.com/RouteWorks/RouterArena/pull/169)** awaiting maintainer review and should be treated as an unmerged candidate submission until officially verified.
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

## 📖 Theoretical Foundations & Related Work

`krusch-cascade-router` draws on proven systems concepts and dynamic inference literature:

1. **Sequential Model Cascading & Fallbacks** (*FrugalGPT; Chen et al., 2023, [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)*):
   - Establishes the sequential cascade principle: querying smaller/cheaper models first and escalating to frontier models only upon low confidence or failure.
2. **Speculative Parallel Hedging** (*The Tail at Scale; Dean & Barroso, Communications of the ACM, 2013*):
   - Rather than waiting sequentially for borderline queries, issuing hedged requests across models masks cascade latency and caps 99th-percentile response times.
3. **Degenerative Token Loops & Repetition** (*The Curious Case of Neural Text Degeneration; Holtzman et al., ICLR 2020*):
   - Autoregressive generation is prone to degenerate repetitive cycles. Monitoring sliding-window $n$-gram repetition allows aborting runaway loops mid-stream before consuming full output tokens.
4. **Zero-Overhead vs. Learned Routing** (*RouterBench; Hu et al., 2024, [arXiv:2403.12031](https://arxiv.org/abs/2403.12031)* & *RouteLLM; Ong et al., 2024, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)*):
   - Multi-LLM routing benchmarks show that while learned classifiers or LLM routers achieve high accuracy, they introduce 15–50ms embedding overhead or 500ms+ LLM latency. Fast heuristic gating provides sub-millisecond dispatch for distinct syntactic and domain signatures.

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

### Option C: Future-Proofing & Custom Specialists

The 5 default models (`gemini-3.1-flash-lite`, `deepseek-v4-flash`, `Qwen3-Coder-Next`, `deepseek-v4-pro`, `qwen3-235b-a22b`) are an **empirical starter preset**, not a hardcoded lock-in. As OpenRouter models evolve, you can easily swap models, update token pricing, or inject custom domain regexes:

```javascript
import { createMultiSpecialistRouter } from 'krusch-cascade-router';

const router = createMultiSpecialistRouter({
  openrouterApiKey: process.env.OPENROUTER_API_KEY,
  // 1. Swap or upgrade specialist models (strings or full ModelConfig objects)
  customModels: {
    code: 'anthropic/claude-3.7-sonnet',
    reasoning_deep: 'openai/o3-mini',
    factual_stem: {
      model: 'meta-llama/llama-3.3-70b-instruct',
      provider: 'openrouter',
      costPerMillionInputTokens: 0.12,
      costPerMillionOutputTokens: 0.30
    }
  },
  // 2. Inject custom domain regex rules evaluated before default heuristics
  classifier: {
    customSpecialistRules: [
      { role: 'reasoning_deep', pattern: /\b(?:legal compliance|gdpr audit|sec filing)\b/i },
      { role: 'code', pattern: /\b(?:terraform plan|ansible playbook|helm chart)\b/i }
    ]
  },
  // 3. Optional local edge model for low-priority / background batch jobs
  backgroundModel: {
    url: 'http://localhost:11434/v1/chat/completions',
    model: 'qwen2.5:3b',
    costPerMillionInputTokens: 0,
    costPerMillionOutputTokens: 0
  }
});
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
| `customModels` | `Partial<Record<SpecialistRole, string \| ModelConfig>>` | `undefined` | Custom model ID strings or full `ModelConfig` objects overriding default specialists. |
| `classifier` | `ClassifierOptions` | `undefined` | Custom options, including `customSpecialistRules` and `customRules`. |
| `backgroundModel` | `ModelConfig` | `undefined` | Optional model for low-priority/background batch jobs. |
| `openrouterReferer` | `string` | `undefined` | Optional `HTTP-Referer` header for rankings. |
| `openrouterTitle` | `string` | `undefined` | Optional `X-Title` header for rankings. |
| `cascadeThreshold` | `number` | `0.85` | Logprob confidence threshold for cascading. |
| `tokensToEvaluate` | `number` | `5` | Tokens to buffer and evaluate for initial confidence. |
| `maxRepetitiveTokens` | `number` | `4` | Threshold for degenerate repetition and cyclic loop detection. |
| `speculativeBranching` | `boolean` | `false` | Enable Second Thought speculative hedging. |
| `prunePreRouting` | `boolean` | `false` | Strips conversational filler and whitespace before length evaluation. |
| `onEvent` | `Function` | `undefined` | Telemetry callback for observability. |

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
| `classifier` | `ClassifierOptions` | `undefined` | Custom options, `customSpecialistRules`, and `customRules` regexes. |
| `onEvent` | `Function` | `undefined` | Telemetry callback for observability. |

---

## 📄 License

MIT License © 2026 kruschdev
