# GEMINI_INFLIGHT — krusch-cascade-router

> Last updated: 2026-09-16

## Active Environment & Nodes
- Primary target: `krusch-cascade-router` npm package & RouterArena benchmark
- Available Edge Worker: `qwen2.5:3b` (kruschgame)
- Upstream PR: `https://github.com/RouteWorks/RouterArena/pull/169`

## Currently Modifying
- Completed task: 7-Model Multi-Specialist Router with unified OpenRouter API routing, full benchmark regeneration, and updated PR #169.

## Fragile / Don't Touch
- N/A

## Active Background Processes
- None

## Non-Obvious Discoveries
- Must maintain `<50ms` latency overhead for predictive classifier.
- OpenRouter endpoint `https://openrouter.ai/api/v1/chat/completions` with bearer token auth and `HTTP-Referer` / `X-Title` attribution.
- Predictions must match RouteWorks/RouterArena schema (13,254 entries for full split, 420 for robustness split).

## Last Session
- Expanded core router architecture to 7 specialist models:
  1. `google/gemini-3.1-flash-lite` (`general_fast`)
  2. `deepseek/deepseek-v4-flash` (`factual_stem`)
  3. `Qwen/Qwen3-Coder-Next` (`code`)
  4. `grok-4-1-fast-reasoning` (`reasoning_fast`)
  5. `deepseek/deepseek-v4-pro` (`reasoning_deep`)
  6. `gemini-3-flash-preview` (`games_spatial`)
  7. `qwen/qwen3-235b-a22b-2507` (`comprehension_rc`)
- Implemented `createMultiSpecialistRouter()` factory and native OpenRouter provider integration in `src/cascade.ts` and `src/classifier.ts`.
- Added 3 new unit test suites in `test-cascade.js` (34/34 passing).
- Clean `tsup` build generates CJS, ESM, and `.d.ts` declaration maps.
- Updated RouterArena adapter and config, regenerating 13,254 full predictions and 420 robustness predictions.
- Verified validation gates (`check_config_prediction_files.py`: `✓ ALL CHECKS PASSED!`).
- Benchmarked robustness: **93.10%** (+15.24% higher than Paix2's 77.86%).
- Benchmarked Arena Score: **77.96** (Accuracy: 79.51%, Cost/1K: $0.1827; outperforms current #1 Paix2's 77.63).
- Pushed clean commits to `submit/krusch-cascade-router` on PR #169.
- Updated `docs/BENCHMARK.md`, `README.md`, and `spec.md`.
- Bumped package version to `1.1.0`, tagged `v1.1.0`, and pushed commits + tags to public GitHub `kruschdev/krusch-cascade-router`.

## Open Questions
- None.

## Discovered Issues
- RouterArena robustness prompts modify preambles and headers (`"Options: \nA."` -> `"Selections: \nA."`). Solved with noise-invariant regex matching, boosting robustness from 60.24% to 93.10%.

## Visual Verification Status
- N/A

## Next Steps
- [ ] Comment `/evaluate` on PR #169 to trigger the official RouterArena bot evaluation.
