import test from 'node:test';
import assert from 'node:assert/strict';
import { CascadeRouter } from '../dist/index.js';

test('L2 Neural Semantic Router Integration in CascadeRouter', async (t) => {
  const events = [];
  const onEvent = (evt, meta) => events.push({ evt, meta });

  const mockSpecialists = {
    code: { model: 'mock-qwen-coder', url: 'https://mock/v1' },
    reasoning_deep: { model: 'mock-deepseek-r1', url: 'https://mock/v1' },
    factual_stem: { model: 'mock-gemini-flash', url: 'https://mock/v1' }
  };

  const mockFetch = async (url, opts) => {
    const body = JSON.parse(opts.body);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{ message: { content: `Response from ${body.model}` } }],
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 }
      })
    };
  };

  await t.test('L2 router dynamically resolves role when L0 fast-path misses', async () => {
    events.length = 0;

    let l2Called = false;
    let l2ReceivedPrompt = '';

    const router = new CascadeRouter({
      fastModel: { model: 'mock-fast' },
      heavyModel: { model: 'mock-heavy' },
      specialistModels: mockSpecialists,
      fetch: mockFetch,
      onEvent,
      l2Router: async (prompt, context) => {
        l2Called = true;
        l2ReceivedPrompt = prompt;
        return {
          recommendedRole: 'reasoning_deep',
          targetTier: 'heavy',
          confidence: 0.88,
          reason: 'Matched deep diagnostics centroid'
        };
      }
    });

    // Unstructured / conversational query that misses L0 fast-path
    const prompt = 'Can you help me understand why our event loop is lagging under high load?';
    const res = await router.chat(prompt, { project: 'krusch-context-mcp' });

    assert.equal(l2Called, true, 'L2 router must be queried on L0 miss');
    assert.equal(l2ReceivedPrompt, prompt);
    assert.equal(res.routedTo, 'reasoning_deep', 'Should be routed to reasoning_deep per L2 recommendation');
    assert.equal(res.text, 'Response from mock-deepseek-r1');

    // Verify route_l2_semantic telemetry event was emitted
    const l2Event = events.find(e => e.evt === 'route_l2_semantic');
    assert.ok(l2Event, 'route_l2_semantic event should be emitted');
    assert.equal(l2Event.meta.role, 'reasoning_deep');
    assert.equal(l2Event.meta.confidence, 0.88);
  });

  await t.test('L0 fast-path bypasses L2 router when deterministic syntax is present', async () => {
    events.length = 0;
    let l2Called = false;

    const router = new CascadeRouter({
      fastModel: { model: 'mock-fast' },
      heavyModel: { model: 'mock-heavy' },
      specialistModels: mockSpecialists,
      fetch: mockFetch,
      onEvent,
      l2Router: async () => {
        l2Called = true;
        return { recommendedRole: 'factual_stem' };
      }
    });

    // Fast-path prompt containing code fence
    const codePrompt = '```typescript\nconst x: number = 42;\n```\nExplain this line.';
    const res = await router.chat(codePrompt);

    assert.equal(l2Called, false, 'L0 fast-path must bypass L2 router');
    assert.equal(res.routedTo, 'code', 'Code fence must fast-path directly to code specialist');
    assert.equal(res.text, 'Response from mock-qwen-coder');
  });

  await t.test('Gracefully falls back to default role if L2 router throws or rejects', async () => {
    events.length = 0;

    const router = new CascadeRouter({
      fastModel: { model: 'mock-fast' },
      heavyModel: { model: 'mock-heavy' },
      specialistModels: mockSpecialists,
      fetch: mockFetch,
      onEvent,
      l2Router: async () => {
        throw new Error('Connection refused to local MCP');
      }
    });

    const res = await router.chat('Just a casual chat question here');
    assert.ok(res.text.includes('Response from'), 'Should complete without throwing');
    assert.equal(res.routedTo, 'factual_stem', 'Should fall back to default role on L2 failure');
  });
});
