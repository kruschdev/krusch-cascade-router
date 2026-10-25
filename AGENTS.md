# krusch-cascade-router — Agent Guidelines & Invariants

> **Role**: L1–L3 Microsecond Swarm Cascade Router (`<25µs` CPU L1 classifier)  
> **Stack**: TypeScript / Node 22 ESM, Ollama, Colab G4 (`L2.5_COLAB_BURST`), OpenRouter (`L3_CLOUD`)  
> **Last updated**: 2026-10-20

## 1. Architectural Invariants (`MUST` / `MUST NOT`)

1. **INV-01 (`<25µs` CPU L1 Classifier Budget)**: `classifyComplexity` in `src/classifier.ts` must execute in `<25µs` on CPU with `$0.00` network cost and `100%` holdout domain accuracy across `code`, `reasoning_deep`, `reasoning_fast`, `factual_stem`, `games_spatial`, and `comprehension_rc`.
2. **INV-02 (4-Tier Cascade Hierarchy)**:
   - **L1 (`Fast` / Local)**: `qwen2.5-coder:7b` & `gemma4:e4b` (`127.0.0.1:11434`).
   - **L2 (`Swarm` / Heavy Local)**: `qwen3-coder-30b-moe:latest` & `hermes3:latest` (`127.0.0.1:11434`).
   - **L2.5 (`DEFAULT_COLAB_BURST_MODEL`)**: `gpt-oss:120b` (`127.0.0.1:11435` SSH tunnel to 96GB Blackwell G4, `131K` ctx) — probed automatically before cloud escalation.
   - **L3 (`Cloud` / Frontier)**: `gemini-4-argon` (`65,536` default `maxOutputTokens`, `thinkingLevel`, multi-hop `CONTINUATION`) or OpenRouter 5-specialist pool.
3. **INV-03 (Strict Routing Parameter Validation)**: `validateRoutingOptions` and `CascadeRouter` in `src/cascade.ts` validate `cascadeThreshold` in `[0, 1]`, finite non-negative `tokensToEvaluate`, and finite non-negative `maxRepetitiveTokens` (throwing `TypeError` on invalid inputs and applying defaults `0.85`, `5`, and `4` when omitted).
4. **INV-04 (Hermetic Offline CI)**: All unit and holdout tests (`npm test`) must pass hermetically without live Ollama or OpenRouter network access.

## 2. Key Files & Verification
- `src/classifier.ts`: Zero-dep regex/structural L1 complexity and specialist-domain classifier.
- `src/cascade.ts`: `CascadeRouter`, `createMultiSpecialistRouter`, L2.5 burst probing, and `gemini-4-argon` multi-hop `CONTINUATION`.
- **Run Tests**: `npm test`
