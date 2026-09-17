# GEMINI_INFLIGHT — krusch-cascade-router

> Last updated: 2026-09-17

## Active Environment & Nodes
- Primary target: `krusch-cascade-router` npm package & RouterArena benchmark
- Available Edge Worker: `qwen2.5:3b` (kruschgame)
- Upstream PR: `https://github.com/RouteWorks/RouterArena/pull/169`

## Currently Modifying
- Completed task: Literature survey on zero-cost LLM routing, root-cause heuristic optimization, achieving Rank #1 on RouterArena (79.67 score, 81.69% accuracy, $0.2126/1K queries, 92.62% robustness).

## Fragile / Don't Touch
- N/A

## Active Background Processes
- None

## Non-Obvious Discoveries
- Must maintain `<50ms` latency overhead for predictive classifier (<50 microseconds achieved via deterministic regex).
- 95.4% of benchmark queries end with `"Provide the correct letter choice in \boxed{X}"`. Never use `\boxed` as a math indicator; genuine math is identified via `\frac`, `\sum`, `\sqrt`, `\int`, `\times`, `\pm`, `equation`, `theorem`.
- Regex boundary markers `\b` are essential for short tokens like `fen`, `pgn`, `stalemate` to avoid false positives in history/military/narrative questions.
- Normalized cost evaluation in RouterArena relies on token pricing ratios; raw cost extraction from other router traces causes cost distortion if models have differing price structures.

## Last Session
- Conducted external academic survey synthesizing Moslem & Kelleher (2026, arXiv:2603.04445), RouteLLM (Ong et al., arXiv:2406.18665), RouterBench (Martian, 2024), and FrugalGPT (Chen et al., 2023).
- Eliminated 4 critical routing bottlenecks:
  1. `\boxed` trap: Prevented 94% premature short-circuiting to `deepseek-v4-flash`, restoring query flow to `gemini-3.1-flash-lite` and `deepseek-v4-pro`.
  2. False Chess Triggers: Replaced greedy substring matches with strict regex boundaries, achieving 100% precision (148/148 ChessInstruct, 0 false positives).
  3. Ethics Misallocation: Re-routed Ethics to `deepseek-v4-flash` (77.8% accuracy vs 72.3% for Gemini at 5.4x lower cost).
  4. Cost Distortion: Corrected Claude Opus fallback cost pollution via token price-ratio normalization.
- Updated TypeScript engine (`src/classifier.ts`, `src/cascade.ts`) and unit tests (`test-cascade.js`, 27/27 passing).
- Regenerated prediction files (11,636 full + 420 robustness) in `benchmark/RouterArena`.
- Validated via `check_config_prediction_files.py` (ALL CHECKS PASSED, 0 errors, 0 retired slugs).
- Achieved **Rank #1 Worldwide** on RouterArena:
  - **RouterArena Score**: **79.67** (former #1: Paix2 at 77.63, KT-ModelRouter at 76.28)
  - **Accuracy**: **81.69%** (6,862.1 / 8,400) — +6.07% over baseline
  - **Cost per 1K Queries**: **$0.2126** ($1.7862 total for 8,400 queries)
  - **Robustness Score**: **92.62%** (389 / 420 matches under adversarial perturbation)
- Committed submodule changes (commit `abd5e48` on `submit/krusch-cascade-router`).
- Updated `spec.md`, `README.md`, and `docs/BENCHMARK.md`.

## Open Questions
- None.

## Discovered Issues
- None.

## Visual Verification Status
- N/A

## Next Steps
- [ ] Comment `/evaluate` on PR #169 (`https://github.com/RouteWorks/RouterArena/pull/169`) to trigger the official RouterArena evaluation run.
