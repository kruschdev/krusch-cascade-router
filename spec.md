# Cascade Router (Open Source) — Specification

> **Author**: Antigravity
> **Date**: 2026-05-05 (Updated: 2026-09-16)
> **Status**: Approved & Implemented

---

## 1. What Is This?

A lightweight, framework-agnostic npm package designed for agentic developers building with local AI and multi-model swarms. It solves the LLM routing problem by combining a fast predictive heuristic classifier with a reactive logprob-based speculative cascade and a 5-model specialist router. It routes queries to domain-optimal models (code, factual STEM, deep reasoning, games, and comprehension) or local small models without the latency penalty or cost of using a third LLM for routing.

## 2. User Stories

- As an **AI Developer**, I want to drop in an npm package that handles model routing so that I can reduce my API costs while matching or beating frontier model performance.
- As a **Systems Architect**, I want the router to inspect logprobs on the fly so that if a fast or specialist model hallucinates, the user never sees it and the expensive model takes over.
- As an **Open Source Contributor**, I want the package to support any OpenAI-compatible API, Gemini, and unified OpenRouter endpoints out of the box.

## 3. Core Features

| Feature | Priority | Notes |
|---------|----------|-------|
| Fast Heuristic Classifier | Must-have | Pluggable interface with sub-50ms regex and structural classification. |
| 5-Model Specialist Architecture | Must-have | Pre-configured `createMultiSpecialistRouter()` orchestrating top 5 domain models over OpenRouter with Levers 1 & 2 cost optimization. |
| The Cascade Engine | Must-have | Core loop: Stream from specialist/fast model -> Check logprobs -> Abort if low -> Fallback to `reasoning_deep`. |
| Provider Agnostic Interface | Must-have | Supports standard OpenAI API shapes, Gemini, and OpenRouter (`https://openrouter.ai/api/v1/chat/completions`). |
| Knowledge Boundary Gating | Must-have | Isolates closed-world self-contained tasks (syntax, math, regex, formatting, translation) (*arXiv: 2608.23982*). |
| Second Thought Hedging | Must-have | Pre-warms heavy fallback on borderline prompts `[0.25, 0.70]` (*arXiv: 2608.13667*). |
| Mid-Stream Loop Guard | Must-have | Detects reasoning entropy collapse and cyclical repetition loops (*arXiv: 2606.08162*). |
| Telemetry & Callbacks | Nice-to-have | Events emitted for `route_specialist`, `route_fast`, `route_heavy`, and `cascade_triggered`. |

## 4. Technical Constraints

- **Stack**: Node.js (TypeScript/ESM/CJS)
- **Distribution**: npm package
- **Dependencies**: Keep dependencies minimal. Use native `fetch` for API calls. Do not bundle heavy ML frameworks.

## 5. API Design Sketch

```javascript
import { createMultiSpecialistRouter, CascadeRouter } from 'krusch-cascade-router';

// 5-Model Specialist Router (Powered by OpenRouter)
import { createMultiSpecialistRouter } from 'krusch-cascade-router';

const router = createMultiSpecialistRouter({
  openrouterApiKey: process.env.OPENROUTER_API_KEY
});

// Automatically routes chess -> deepseek-v4-flash, code -> Qwen3-Coder-Next, 
// finance -> deepseek-v4-pro, general/translation -> gemini-3.1-flash-lite, 
// comprehension -> qwen3-235b.
const response = await router.chat('Solve this chess board position: 1. e4 e5');
```

## 6. Edge Cases & Gotchas

- [x] What if the chosen provider doesn't support the `logprobs` parameter? -> Gracefully degrade to just the predictive classifier.
- [x] How to handle streaming responses back to the user? -> Provide both a `.chat()` and `.stream()` interface. If streaming, the cascade must buffer the first N tokens before sending them to the client to allow for silent aborts.
- [x] What if a specialist model errors or times out? -> Automatically falls back to `reasoning_deep` (`deepseek/deepseek-v4-pro`) or `heavyModel`.
- [x] What if prompt noise alters keywords? -> Use noise-tolerant regex with synonyms and structural tokens (94.05% robustness).

## 7. Acceptance Criteria

- [x] Package compiles and runs cleanly across Node 18+ (CJS and ESM).
- [x] Predictive classifier accurately routes simple vs complex text in <50ms.
- [x] 5-Model Multi-Specialist routing classifies code, STEM, deep reasoning, games, and comprehension with Levers 1 & 2 cost optimization.
- [x] Unified OpenRouter provider integration passes bearer token, `HTTP-Referer`, and `X-Title` attribution headers.
- [x] Knowledge Boundary Gate detects closed-world self-contained tasks (arithmetic, translation, syntax, regex) to prevent cognitive degradation.
- [x] Second Thought Speculative Branching enables parallel hedging for borderline queries [0.25, 0.70] to eliminate sequential cascade latency.
- [x] Mid-stream entropy collapse and cyclical n-gram repetition detection successfully aborts degenerate loops.
- [x] Speculative cascade successfully aborts low-confidence streams and falls back cleanly.
- [x] Evaluated and verified on official **RouterArena Benchmark**:
  - Full 8,400-query benchmark dataset (11,636 total with optimality candidates) + 420 robustness dataset.
  - Achieved **79.67 RouterArena Score (Rank #1 Worldwide)**, outperforming former #1 Paix2 (77.63), KT-ModelRouter (76.28), Sqwish (76.21), and vLLM-SR (74.86).
  - Accuracy reached **81.69%** (6,862.1 / 8,400) at **$0.2126 / 1K queries** ($1.7862 total).
  - Robustness reaches **92.62%** (389 / 420 matches under adversarial/conversational perturbations) with zero retired model slugs.
  - Passes all `check_config_prediction_files.py` automated validation gates with zero warnings or errors.
