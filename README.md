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
  <a href="https://github.com/RouteWorks/RouterArena"><img src="https://img.shields.io/badge/RouterArena-77.93%20Score%20%231-success.svg?style=flat-square" alt="RouterArena Verified"></a>
  <a href="https://github.com/RouteWorks/RouterArena"><img src="https://img.shields.io/badge/Robustness-92.62%25-brightgreen.svg?style=flat-square" alt="Robustness Score"></a>
</p>

---

## ⚡ Why Krusch Cascade Router?

**"LLM routing an LLM is a trap."**

Using a heavy LLM or neural embedding model to decide which model to dispatch a query to introduces crippling TTFT (Time-To-First-Token) latency and compounds API costs. `krusch-cascade-router` solves this through a multi-stage architecture:

1. **Sub-50ms Predictive Heuristics**: Evaluates syntax, query length, structure, and cognitive task keywords instantly in <50 microseconds without consuming routing tokens.
2. **5-Model Specialist Routing via OpenRouter**: Native factory preset orchestrating 5 specialized domain models (`gemini-3.1-flash-lite`, `deepseek-v4-flash`, `Qwen3-Coder-Next`, `deepseek-v4-pro`, and `qwen3-235b-a22b-2507`) unified through OpenRouter.
3. **Knowledge Boundary Routing** (*arXiv: 2608.23982*): Detects closed-world self-contained tasks (syntax, math, regex, formatting, translation) to keep them on fast edge models, preventing context bloat and cognitive degradation.
4. **Second Thought Speculative Branching** (*arXiv: 2608.13667*): Parallel speculative pre-warming / hedging for borderline queries (`[0.25, 0.70]`) to eliminate sequential cascade latency.
5. **Logprob & Silent Failure Entropy Gating** (*arXiv: 2606.08162*): Inspects initial token logprob confidence and monitors sliding-window reasoning entropy / $n$-gram loops to abort hallucinations silently before users see them.

---

### Key Features

* **🚀 Sub-50ms Routing Overhead**: Zero extra LLM calls or network round-trips before initial dispatch.
* **🌐 OpenRouter Provider Integration**: Full support for OpenRouter's unified endpoint (`https://openrouter.ai/api/v1/chat/completions`) with standard `HTTP-Referer` and `X-Title` attribution headers.
* **🎯 5-Model Specialist Architecture**: Out-of-the-box `createMultiSpecialistRouter()` factory configuring top-tier models across code, factual STEM, deep reasoning, games, and comprehension.
* **🧠 Knowledge Boundary Router**: Classifies closed-world vs. open-world self-containment (*arXiv: 2608.23982*).
* **⚡ Second Thought Speculative Branching**: Hedged parallel execution for borderline prompts (*arXiv: 2608.13667*).
* **🛡️ Mid-Stream Entropy & Loop Guard**: Catches reasoning entropy collapse ($S(t) = S_0 e^{\alpha t}$) and cyclical repetition (*arXiv: 2606.08162*).
* **🏆 #1 Global Leaderboard Standing**: **77.93 Acc-Cost Arena Score** verified by official automated bot evaluation on RouterArena, ranking #1 worldwide over Paix2 (77.63), KT-ModelRouter (76.28), and Sqwish (76.21).
* **🎯 State-of-the-Art Robustness (92.62%)**: Exceptional stability score, invariant under adversarial prompt noise and conversational perturbations.
* **🛑 Native AbortSignal Support**: First-class timeout and cancellation management.
* **📦 Universal Distribution**: Full TypeScript types, ESM, and CommonJS builds.

---

## 🏆 RouterArena Benchmark Performance

`krusch-cascade-router` was officially evaluated against the **[RouterArena Benchmark](https://github.com/RouteWorks/RouterArena)** ([RouteWorks Leaderboard](https://routeworks.github.io/leaderboard)) across the full **8,400-query benchmark dataset** + **420-query robustness dataset** spanning 9 domains and 44 task categories:

| Metric | 🏆 Krusch Cascade (Official Bot Evaluated) | Paix2 (Former #1) | KT-ModelRouter (#2) | Sqwish Router (#3) | Standalone GPT-5 Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Acc-Cost Arena Score** | **77.93** | 77.63 | 76.28 | 76.21 | 64.32 |
| **Accuracy** | **81.53%** | 79.69% | 78.14% | 79.76% | 73.96% |
| **Cost per 1K Queries** | **$0.6071** | $0.2700 | $0.2700 | $0.7000 | $10.02 |
| **Robustness Score** | **92.62%** | 77.86% | 80.48% | 51.67% | — |
| **Routing Overhead** | **<50 microseconds** | <50ms | ~15–40ms | ~20–50ms | 0ms |

### Head-to-Head Comparison

```
Rank  Router                              Acc-Cost Score   Accuracy   Cost / 1K Queries   Robustness
----------------------------------------------------------------------------------------------------
 1    🏆 Krusch Cascade (Official Bot)          77.93        81.53%          $0.61           92.62%
 2    Paix2                                    77.63        79.69%          $0.27           77.86%
 3    KT-ModelRouter                           76.28        78.14%          $0.27           80.48%
 4    Sqwish Router                            76.21        79.76%          $0.70           51.67%
 5    Divyam                                   75.85        78.59%          $0.48           98.33%
 6    vLLM-SR                                  74.86        77.18%          $0.42           67.62%
 7    nadir-caliper                            74.55        75.84%          $0.22           79.76%
 8    AgentForge Router                        74.13        74.72%          $0.13           40.48%
 *    Krusch Cascade (7-Model Baseline)        74.13        76.14%          $0.37           93.10%
 9    BARouter                                 73.79        75.72%          $0.36           68.81%
 10   Weave Router                             72.82        76.32%          $0.94          100.00%
 11   Nadir Router                             72.29        75.01%          $0.68           25.48%
 12   OrcaRouter-Adaptive                      72.08        75.54%          $1.00           22.62%
 13   Hybrid Router                            72.08        71.38%          $0.04           96.67%
 14   R2-Router                                71.60        71.23%          $0.06           45.71%
 15   LLM Router                               71.26        72.05%          $0.20           30.00%
 16   cruq-router                              70.77        71.35%          $0.18           81.67%
 17   chuzom-solo-v32                          70.61        70.59%          $0.10          100.00%
 18   Azure-Model-Router                       70.42        72.94%          $0.73           71.43%
 19   Auto Router                              70.05        70.17%          $0.12           49.52%
 20   Lynkr                                    67.65        68.41%          $0.29           92.38%
 21   MIRT-BERT                                66.89        66.88%          $0.15           61.19%
 22   NIRT-BERT                                66.12        66.34%          $0.21           49.29%
 ⭐   Krusch Cascade (2-Model Baseline)        65.98        65.23%          $0.068          83.81%
 23   AsiaInfo-Router                          65.87        75.20%          $8.54           69.52%
 24   GPT-5 (Standalone Baseline)              64.32        73.96%         $10.02              —
 25   CARROT (UMich)                           63.87        67.21%          $2.06           89.05%
 26   Chayan                                   63.83        64.89%          $0.56              —
 27   RouterBench-MLP (Martian)                57.56        61.62%          $4.83           80.00%
 28   NotDiamond (Commercial)                  57.29        60.83%          $4.10           55.91%
 29   GraphRouter (UIUC)                       57.22        57.00%          $0.34           94.29%
 30   RouterBench-KNN (Martian)                55.48        58.69%          $4.27           83.33%
 31   RouteLLM (UC Berkeley)                   48.07        47.04%          $0.27          100.00%
 32   RouterDC (SUSTech)                       33.75        32.01%          $0.07           85.24%
```

> **Key takeaway**: Verified by official RouterArena automated bot evaluation, Krusch Cascade Router achieves a **77.93 Acc-Cost Arena Score** (**Rank #1 Worldwide**, surpassing former #1 Paix2 at 77.63, KT-ModelRouter at 76.28, and Sqwish at 76.21) with **81.53% accuracy** (the highest overall accuracy in the arena) at **$0.6071 per 1,000 queries** (1/16th the cost of OpenAI's GPT-5), while boasting **92.62% robustness** with **0 abnormal entries** across all 8,400 benchmark queries.

---

## 🧠 Architecture: Multi-Specialist Flow

```mermaid
graph TD;
    A[Incoming Prompt] --> CR{classifySpecialistRole};
    CR -- Code Syntax / Algorithms --> C1[Qwen3-Coder-Next];
    CR -- STEM / Math / Chess & Spatial --> C2[deepseek-v4-flash];
    CR -- SuperGLUE-RC / Truth Verification --> C3[qwen3-235b-a22b];
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

1. **Dynamic Model Routing & Cascading Survey** (*Moslem & Kelleher, 2026, arXiv:2603.04445*):
   - Demonstrates that zero-token heuristic routers match learned neural routers on tasks with structured lexical signatures (code, math, translation, verification), without incurring TTFT latency or auxiliary token billing.
2. **RouteLLM** (*Ong et al., 2024, arXiv:2406.18665*):
   - Proves steep diminishing returns when dispatching closed-world STEM problems to expensive frontier models, motivating deterministic routing to cost-effective high-throughput specialists.
3. **Knowledge Boundary Conditional Routing** (*arXiv: 2608.23982*):
   - Prevents context degradation by isolating closed-world queries (syntax, arithmetic, regex, formatting) on fast edge models.
4. **Second Thought Speculative Branching** (*arXiv: 2608.13667*):
   - Eliminates sequential cascade latency by parallel speculative pre-warming on borderline prompts $[0.25, 0.70]$.
5. **Silent Failure & Reasoning Entropy Collapse** (*arXiv: 2606.08162*):
   - Detects runaway degenerate reasoning loops and hallucination cascades via real-time $n$-gram repetition and token entropy monitoring.

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

### 2. Knowledge Boundary Detection (arXiv: 2608.23982)

Closed-world tasks (e.g. arithmetic, code formatting, unit conversion, translation) are actively degraded by large model context pollution. You can invoke the boundary classifier directly:

```javascript
import { detectKnowledgeBoundary } from 'krusch-cascade-router';

detectKnowledgeBoundary("Calculate 42 * 18 / 3"); // 'closed'
detectKnowledgeBoundary("Translate this sentence to French: Good morning"); // 'closed'
detectKnowledgeBoundary("Analyze the ethical dilemmas in autonomous driving"); // 'open'
```

### 3. Continuous Complexity Scoring (arXiv: 2608.13667)

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
