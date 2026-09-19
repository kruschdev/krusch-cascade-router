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
