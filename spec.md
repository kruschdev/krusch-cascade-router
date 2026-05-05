# Cascade Router (Open Source) — Specification

> **Author**: Antigravity
> **Date**: 2026-05-05
> **Status**: Approved

---

## 1. What Is This?

A lightweight, framework-agnostic npm package designed for agentic developers building with local AI. It solves the LLM routing problem by combining a fast predictive heuristic classifier with a reactive logprob-based speculative cascade. It routes simple queries to local small models (e.g., 8B) and hard queries to large models (e.g., 70B) without the latency penalty of using a third LLM for routing.

## 2. User Stories

- As an **AI Developer**, I want to drop in an npm package that handles model routing so that I can reduce my API costs.
- As a **Systems Architect**, I want the router to inspect logprobs on the fly so that if the cheap model hallucinates, the user never sees it and the expensive model takes over.
- As an **Open Source Contributor**, I want the package to support any OpenAI-compatible API so that I can use it with Ollama, vLLM, or LM Studio.

## 3. Core Features

| Feature | Priority | Notes |
|---------|----------|-------|
| Fast Heuristic Classifier | Must-have | Pluggable interface with a lightweight default heuristic. |
| The Cascade Engine | Must-have | Core loop: Stream from fast model -> Check logprobs -> Abort if low -> Fallback. |
| Provider Agnostic Interface | Must-have | Supports standard OpenAI API shapes and Gemini. |
| Telemetry & Callbacks | Nice-to-have | Events emitted for `route_fast`, `route_heavy`, and `cascade_triggered`. |

## 4. Technical Constraints

- **Stack**: Node.js (TypeScript/ESM)
- **Distribution**: npm package
- **Dependencies**: Keep dependencies minimal. Use native `fetch` for API calls. Do not bundle heavy ML frameworks.

## 5. API Design Sketch

```javascript
import { CascadeRouter } from 'krusch-cascade-router';

const router = new CascadeRouter({
  fastModel: { url: 'http://localhost:11434/v1', model: 'qwen2.5' },
  heavyModel: { apiKey: process.env.OPENAI_API_KEY, model: 'gpt-4o' },
  cascadeThreshold: 0.85 // Logprob confidence cutoff
});

const response = await router.chat("Write a complex architectural plan...");
```

## 6. Edge Cases & Gotchas

- [ ] What if the chosen provider doesn't support the `logprobs` parameter? -> Gracefully degrade to just the predictive classifier.
- [ ] How to handle streaming responses back to the user? -> Provide both a `.chat()` and `.stream()` interface. If streaming, the cascade must buffer the first N tokens before sending them to the client to allow for silent aborts.

## 7. Acceptance Criteria

- [x] Package compiles and runs cleanly.
- [x] Predictive classifier accurately routes simple vs complex text in <50ms.
- [ ] Speculative cascade successfully aborts a low-confidence stream and returns the fallback model's response.
- [x] Comprehensive README.md explaining the "LLM routing an LLM is a trap" philosophy.
