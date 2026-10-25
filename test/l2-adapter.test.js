import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cosineSimilarity,
  createCentroidSemanticRouter,
  createContextMcpRouter,
  DEFAULT_L2_ARCHETYPES
} from '../dist/index.js';

test('L2 Adapter - cosineSimilarity mathematical invariants', () => {
  assert.equal(cosineSimilarity([1, 0, 0], [1, 0, 0]), 1.0);
  assert.equal(cosineSimilarity([1, 0, 0], [0, 1, 0]), 0.0);
  assert.equal(cosineSimilarity([1, 0, 0], [-1, 0, 0]), -1.0);
  assert.equal(cosineSimilarity([], []), 0);
  assert.equal(cosineSimilarity([1, 2], [1, 2, 3]), 0);

  // Scaled invariance: 2 * v has identical cosine similarity as v
  const simA = cosineSimilarity([1, 2, 3], [4, 5, 6]);
  const simB = cosineSimilarity([2, 4, 6], [4, 5, 6]);
  assert.ok(Math.abs(simA - simB) < 1e-6);
});

test('L2 Adapter - createCentroidSemanticRouter in-process classification', async () => {
  // Pre-configured toy embeddings (4 dimensions)
  const codeCentroid = [1.0, 0.0, 0.0, 0.0];
  const deepReasoningCentroid = [0.0, 1.0, 0.0, 0.0];
  const factualStemCentroid = [0.0, 0.0, 1.0, 0.0];

  const centroids = [
    {
      archetype: 'code_test',
      role: 'code',
      tier: 'specialist',
      label: 'Code Refactoring',
      exemplar: 'refactor code',
      embedding: codeCentroid,
      threshold: 0.70
    },
    {
      archetype: 'deep_test',
      role: 'reasoning_deep',
      tier: 'heavy',
      label: 'System Deadlock',
      exemplar: 'deadlock analysis',
      embedding: deepReasoningCentroid,
      threshold: 0.70
    }
  ];

  const router = createCentroidSemanticRouter({
    centroids,
    embed: async (text) => {
      if (text.includes('function') || text.includes('refactor')) {
        return [0.95, 0.05, 0.0, 0.0];
      }
      if (text.includes('deadlock') || text.includes('concurrency')) {
        return [0.05, 0.95, 0.0, 0.0];
      }
      // Ambiguous / unrelated
      return [0.2, 0.2, 0.2, 0.2];
    }
  });

  // 1. Coding query
  const res1 = await router('Refactor this async function');
  assert.ok(res1);
  assert.equal(res1.recommendedRole, 'code');
  assert.equal(res1.targetTier, 'specialist');
  assert.ok(res1.confidence > 0.9);

  // 2. Deadlock reasoning query
  const res2 = await router('Analyze database deadlock scenario');
  assert.ok(res2);
  assert.equal(res2.recommendedRole, 'reasoning_deep');
  assert.equal(res2.targetTier, 'heavy');

  // 3. Ambiguous query (falls below threshold 0.70)
  const res3 = await router('Tell me about random stuff');
  assert.ok(res3);
  assert.equal(res3.recommendedRole, 'reasoning_deep');
  assert.equal(res3.targetTier, 'heavy');
  assert.ok(res3.reason.includes('fell below threshold'));
});

test('L2 Adapter - createContextMcpRouter wraps MCP tool call seamlessly', async () => {
  let calledTool = '';
  let calledArgs = null;

  const mockToolCaller = async (name, args) => {
    calledTool = name;
    calledArgs = args;
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            recommendedRole: 'reasoning_deep',
            targetTier: 'heavy',
            recommendedModel: 'qwen3-235b-a22b-2507',
            confidence: 0.842,
            reason: "Matched centroid 'Deep Root-Cause Debugging'"
          })
        }
      ]
    };
  };

  const mcpRouter = createContextMcpRouter(mockToolCaller);
  const result = await mcpRouter('Why did the socket disconnect?', { project: 'krusch-cascade-router' });

  assert.equal(calledTool, 'krusch_context_semantic_route');
  assert.equal(calledArgs.prompt, 'Why did the socket disconnect?');
  assert.equal(calledArgs.project, 'krusch-cascade-router');

  assert.ok(result);
  assert.equal(result.recommendedRole, 'reasoning_deep');
  assert.equal(result.targetTier, 'heavy');
  assert.equal(result.recommendedModel, 'qwen3-235b-a22b-2507');
  assert.equal(result.confidence, 0.842);
});

test('BUG-01: src/cascade.ts emits hedge_error and rethrows instead of silent catch', async () => {
  const fs = await import('node:fs');
  const cascadeSource = fs.readFileSync(new URL('../src/cascade.ts', import.meta.url), 'utf8');
  assert.equal(cascadeSource.includes("this.config.onEvent?.('hedge_error', { error: (hedgeErr as Error).message });"), true);
  assert.equal(cascadeSource.includes('throw hedgeErr;'), true);
});

test('ARCH-01: DEFAULT_L2_ARCHETYPES is frozen and rejects runtime mutation', () => {
  assert.equal(Object.isFrozen(DEFAULT_L2_ARCHETYPES), true);
  assert.equal(Object.isFrozen(DEFAULT_L2_ARCHETYPES[0]), true);
  assert.throws(() => {
    DEFAULT_L2_ARCHETYPES.push({
      archetype: 'rogue',
      role: 'code',
      tier: 'specialist',
      label: 'Rogue',
      exemplar: 'rogue'
    });
  }, TypeError);
});

test('ARCH-01, TEST-01 & VALID-01: sreCache TTL/capacity eviction and routing option validation', async () => {
  const { CascadeRouter, validateRoutingOptions } = await import('../dist/index.js');

  // VALID-01: Valid options succeed; invalid options throw TypeError
  validateRoutingOptions({ cascadeThreshold: 0.85, tokensToEvaluate: 0, maxRepetitiveTokens: 4 });
  assert.throws(() => validateRoutingOptions({ cascadeThreshold: 1.5 }), TypeError);
  assert.throws(() => validateRoutingOptions({ cascadeThreshold: -0.1 }), TypeError);
  assert.throws(() => validateRoutingOptions({ tokensToEvaluate: -1 }), TypeError);
  assert.throws(() => validateRoutingOptions({ maxRepetitiveTokens: Number.NaN }), TypeError);

  const mockFetch = async () => ({
    ok: true,
    json: async () => ({ reply: 'SRE nominal' })
  });

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5-coder:7b', url: 'http://127.0.0.1:11434/v1/chat/completions' },
    heavyModel: { model: 'gpt-oss:120b', url: 'http://127.0.0.1:11435/v1/chat/completions' },
    fetch: mockFetch,
    jeanSREGate: { enabled: true, cacheTtlMs: 50 }
  });

  // Populate SRE_CACHE_MAX_ENTRIES + 1 entries and verify capacity cap + oldest FIFO eviction
  for (let i = 0; i <= CascadeRouter.SRE_CACHE_MAX_ENTRIES; i++) {
    await router.fetchJeanSuggestions(`proj_${i}`);
  }
  assert.equal(router.sreCache.size, CascadeRouter.SRE_CACHE_MAX_ENTRIES);
  assert.equal(router.sreCache.has('proj_0'), false, 'Oldest entry proj_0 must be evicted when capacity is exceeded');
  assert.equal(router.sreCache.has(`proj_${CascadeRouter.SRE_CACHE_MAX_ENTRIES}`), true);

  // Advance simulated time beyond TTL and verify evictExpiredCacheEntries purges expired items
  const evictedCount = router.evictExpiredCacheEntries(Date.now() + 1000);
  assert.equal(evictedCount, CascadeRouter.SRE_CACHE_MAX_ENTRIES);
  assert.equal(router.sreCache.size, 0);
});



