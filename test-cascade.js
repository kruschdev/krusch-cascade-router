import test from 'node:test';
import assert from 'node:assert/strict';
import { CascadeRouter, isComplexPrompt, evaluateComplexityScore, classifySpecialistRole } from './dist/index.js';

// --- Mock Helpers ---

/**
 * Creates a mock fetch that returns a streaming SSE response for the fast model
 * and a standard JSON response for the heavy model.
 */
function createMockFetch({ fastTokens, fastLogprobs, heavyText, fastShouldFail }) {
  return async (url, opts) => {
    // Determine which model is being called based on the body
    const body = JSON.parse(opts.body);

    if (body.stream === true) {
      // Fast model (streaming SSE)
      if (fastShouldFail) {
        return { ok: false, status: 503 };
      }

      const chunks = fastTokens.map((token, i) => {
        const chunk = {
          choices: [{
            delta: { content: token },
            logprobs: {
              content: [{
                logprob: Math.log(fastLogprobs[i])
              }]
            }
          }]
        };
        return `data: ${JSON.stringify(chunk)}\n\n`;
      });
      chunks.push('data: [DONE]\n\n');

      const fullSSE = chunks.join('');
      const encoder = new TextEncoder();
      const encoded = encoder.encode(fullSSE);

      let read = false;
      const body_stream = {
        getReader: () => ({
          read: async () => {
            if (!read) {
              read = true;
              return { done: false, value: encoded };
            }
            return { done: true, value: undefined };
          },
          releaseLock: () => {}
        })
      };

      return { ok: true, status: 200, body: body_stream };
    } else {
      // Heavy model (non-streaming JSON)
      return {
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{ message: { content: heavyText } }]
        })
      };
    }
  };
}

/**
 * Creates a mock fetch for the Gemini provider.
 */
function createGeminiMockFetch(responseText) {
  return async (url, opts) => {
    // If it's the streaming fast model call, return high-confidence to trigger cascade to heavy
    const body = JSON.parse(opts.body);
    if (body.stream) {
      return { ok: false, status: 503 };
    }
    // Gemini response
    return {
      ok: true,
      status: 200,
      json: async () => ({
        candidates: [{ content: { parts: [{ text: responseText }] } }]
      })
    };
  };
}

// --- Tests ---

test('CascadeRouter - Routes complex prompts directly to heavy model', async () => {
  const events = [];
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: [],
      fastLogprobs: [],
      heavyText: 'Heavy model response',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  // Contains 'analyze' — a complex verb trigger
  const result = await router.chat('Please analyze this complex system architecture.');
  assert.equal(result.routedTo, 'heavy');
  assert.equal(result.aborted, false);
  assert.equal(result.text, 'Heavy model response');
  assert.equal(events[0].event, 'route_heavy');
  assert.equal(events[0].meta.reason, 'classifier_heuristic');
});

test('CascadeRouter - Routes simple prompts to fast model with high confidence', async () => {
  const events = [];
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    cascadeThreshold: 0.85,
    tokensToEvaluate: 3,
    fetch: createMockFetch({
      fastTokens: ['Hello', ', ', 'world', '!'],
      fastLogprobs: [0.95, 0.92, 0.90, 0.93], // All above 0.85
      heavyText: 'Should not see this',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const result = await router.chat('Hello there');
  assert.equal(result.routedTo, 'fast');
  assert.equal(result.aborted, false);
  assert.equal(result.text, 'Hello, world!');
  assert.equal(events[0].event, 'route_fast');
});

test('CascadeRouter - Cascade triggers on low logprob confidence', async () => {
  const events = [];
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    cascadeThreshold: 0.85,
    tokensToEvaluate: 3,
    fetch: createMockFetch({
      fastTokens: ['Um', '...', 'maybe'],
      fastLogprobs: [0.4, 0.3, 0.2], // Way below 0.85
      heavyText: 'Heavy model took over',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const result = await router.chat('Hello there');
  assert.equal(result.routedTo, 'heavy');
  assert.equal(result.aborted, true);
  assert.equal(result.text, 'Heavy model took over');
  // Should have cascade_triggered and route_heavy events
  const cascadeEvent = events.find(e => e.event === 'cascade_triggered');
  assert.ok(cascadeEvent, 'cascade_triggered event should have been emitted');
  assert.ok(cascadeEvent.meta.avgProb < 0.85);
});

test('CascadeRouter - Falls back to heavy model on fast model HTTP error', async () => {
  const events = [];
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: [],
      fastLogprobs: [],
      heavyText: 'Fallback response',
      fastShouldFail: true
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const result = await router.chat('Hello there');
  assert.equal(result.routedTo, 'heavy');
  assert.equal(result.aborted, true);
  assert.equal(result.text, 'Fallback response');
  assert.equal(events[0].event, 'route_heavy');
  assert.equal(events[0].meta.reason, 'fast_model_error');
});

test('CascadeRouter - Gemini provider path works', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gemini-2.5-pro', apiKey: 'test-key', provider: 'gemini' },
    fetch: createGeminiMockFetch('Gemini response here')
  });

  // Code block triggers complex → routes to heavy (Gemini)
  const result = await router.chat('Fix this:\n```js\nconst x = 1;\n```');
  assert.equal(result.routedTo, 'heavy');
  assert.equal(result.text, 'Gemini response here');
});

test('CascadeRouter - Respects AbortSignal', async () => {
  const controller = new AbortController();
  // Abort immediately
  controller.abort();

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async () => {
      throw new Error('Request was aborted');
    }
  });

  // Complex prompt → goes to heavy model → fetch throws because aborted
  await assert.rejects(
    () => router.chat('Please analyze this.', undefined, { signal: controller.signal }),
    { message: 'Request was aborted' }
  );
});

test('CascadeRouter - Handles string input with system prompt', async () => {
  let capturedBody;
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async (url, opts) => {
      capturedBody = JSON.parse(opts.body);
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'ok' } }] })
      };
    }
  });

  // 'architect' triggers complex → heavy model
  await router.chat('Architect a system.', 'You are an expert.');
  assert.equal(capturedBody.messages[0].role, 'system');
  assert.equal(capturedBody.messages[0].content, 'You are an expert.');
  assert.equal(capturedBody.messages[1].role, 'user');
  assert.equal(capturedBody.messages[1].content, 'Architect a system.');
});

test('CascadeRouter - Gemini throws without API key', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gemini-2.5-pro', provider: 'gemini' },
    fetch: async () => ({ ok: true, status: 200, json: async () => ({}) })
  });

  await assert.rejects(
    () => router.chat('Please analyze this.'),
    { message: 'Gemini requires an API key' }
  );
});

test('CascadeRouter - AbortSignal listeners are cleanly removed on stream success', async () => {
  let added = 0;
  let removed = 0;
  const controller = new AbortController();
  const signal = controller.signal;
  const originalAdd = signal.addEventListener;
  const originalRemove = signal.removeEventListener;
  
  signal.addEventListener = function(type, ...args) {
    if (type === 'abort') added++;
    return originalAdd.apply(this, [type, ...args]);
  };
  signal.removeEventListener = function(type, ...args) {
    if (type === 'abort') removed++;
    return originalRemove.apply(this, [type, ...args]);
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    cascadeThreshold: 0.85,
    tokensToEvaluate: 3,
    fetch: createMockFetch({
      fastTokens: ['Hello', ', ', 'world', '!'],
      fastLogprobs: [0.95, 0.92, 0.90, 0.93],
      heavyText: 'Should not see this',
      fastShouldFail: false
    })
  });

  const result = await router.chat('Hello there', undefined, { signal });
  assert.equal(result.routedTo, 'fast');
  assert.equal(added, 1, 'Should register exactly 1 abort listener');
  assert.equal(removed, 1, 'Should clean up the abort listener on success');
});

test('CascadeRouter - AbortSignal listeners are cleanly removed on stream() success', async () => {
  let added = 0;
  let removed = 0;
  const controller = new AbortController();
  const signal = controller.signal;
  const originalAdd = signal.addEventListener;
  const originalRemove = signal.removeEventListener;
  
  signal.addEventListener = function(type, ...args) {
    if (type === 'abort') added++;
    return originalAdd.apply(this, [type, ...args]);
  };
  signal.removeEventListener = function(type, ...args) {
    if (type === 'abort') removed++;
    return originalRemove.apply(this, [type, ...args]);
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    cascadeThreshold: 0.85,
    tokensToEvaluate: 3,
    fetch: createMockFetch({
      fastTokens: ['Hello', ', ', 'world', '!'],
      fastLogprobs: [0.95, 0.92, 0.90, 0.93],
      heavyText: 'Should not see this',
      fastShouldFail: false
    })
  });

  const chunks = [];
  for await (const chunk of router.stream('Hello there', undefined, { signal })) {
    chunks.push(chunk);
  }
  assert.equal(chunks.join(''), 'Hello, world!');
  assert.equal(added, 1, 'Should register exactly 1 abort listener during stream()');
  assert.equal(removed, 1, 'Should clean up the abort listener on stream() success');
});

test('CascadeRouter - stream() High Confidence Path works with buffering', async () => {
  const events = [];
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    cascadeThreshold: 0.85,
    tokensToEvaluate: 3,
    fetch: createMockFetch({
      fastTokens: ['Hello', ', ', 'world', '!'],
      fastLogprobs: [0.95, 0.92, 0.90, 0.93],
      heavyText: 'Should not see this',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const chunks = [];
  for await (const chunk of router.stream('Hello there')) {
    chunks.push(chunk);
  }
  assert.equal(chunks.join(''), 'Hello, world!');
  assert.deepEqual(chunks, ['Hello', ', ', 'world', '!']);
  assert.ok(events.some(e => e.event === 'route_fast'));
});

test('CascadeRouter - stream() Low Confidence Cascade Path triggers fallback to heavy streaming', async () => {
  const events = [];
  
  const mockFetch = async (url, opts) => {
    const body = JSON.parse(opts.body);
    const encoder = new TextEncoder();
    
    if (body.model === 'qwen-fast') {
      const chunks = [
        {
          choices: [{
            delta: { content: 'Um' },
            logprobs: { content: [{ logprob: Math.log(0.4) }] }
          }]
        },
        {
          choices: [{
            delta: { content: '...' },
            logprobs: { content: [{ logprob: Math.log(0.3) }] }
          }]
        },
        {
          choices: [{
            delta: { content: 'maybe' },
            logprobs: { content: [{ logprob: Math.log(0.2) }] }
          }]
        }
      ].map(chunk => `data: ${JSON.stringify(chunk)}\n\n`);
      chunks.push('data: [DONE]\n\n');
      
      let read = false;
      const body_stream = {
        getReader: () => ({
          read: async () => {
            if (!read) {
              read = true;
              return { done: false, value: encoder.encode(chunks.join('')) };
            }
            return { done: true, value: undefined };
          },
          releaseLock: () => {}
        })
      };
      return { ok: true, status: 200, body: body_stream };
    } else if (body.model === 'gpt-heavy') {
      const chunks = [
        { choices: [{ delta: { content: 'Heavy ' } }] },
        { choices: [{ delta: { content: 'stream ' } }] },
        { choices: [{ delta: { content: 'output' } }] }
      ].map(chunk => `data: ${JSON.stringify(chunk)}\n\n`);
      chunks.push('data: [DONE]\n\n');
      
      let read = false;
      const body_stream = {
        getReader: () => ({
          read: async () => {
            if (!read) {
              read = true;
              return { done: false, value: encoder.encode(chunks.join('')) };
            }
            return { done: true, value: undefined };
          },
          releaseLock: () => {}
        })
      };
      return { ok: true, status: 200, body: body_stream };
    }
    return { ok: false, status: 400 };
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen-fast' },
    heavyModel: { model: 'gpt-heavy', apiKey: 'test-key' },
    cascadeThreshold: 0.85,
    tokensToEvaluate: 3,
    fetch: mockFetch,
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const chunks = [];
  for await (const chunk of router.stream('Hello there')) {
    chunks.push(chunk);
  }
  
  assert.equal(chunks.join(''), 'Heavy stream output');
  assert.ok(events.some(e => e.event === 'cascade_triggered'));
  assert.ok(events.some(e => e.event === 'route_heavy' && e.meta.reason === 'cascade_fallback'));
});

test('CascadeRouter - stream() Gemini streaming parser works', async () => {
  const mockFetch = async (url, opts) => {
    const encoder = new TextEncoder();
    
    const geminiPayload = `[
      {"candidates":[{"content":{"parts":[{"text":"Gemi"}]}}]},
      {"candidates":[{"content":{"parts":[{"text":"ni "}]}}]},
      {"candidates":[{"content":{"parts":[{"text":"stream!"}]}}]}
    ]`;
    
    let read = false;
    const body_stream = {
      getReader: () => ({
        read: async () => {
          if (!read) {
            read = true;
            return { done: false, value: encoder.encode(geminiPayload) };
          }
          return { done: true, value: undefined };
        },
        releaseLock: () => {}
      })
    };
    return { ok: true, status: 200, body: body_stream };
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gemini-2.5-pro', apiKey: 'test-key', provider: 'gemini' },
    fetch: mockFetch
  });

  const chunks = [];
  for await (const chunk of router.stream('Please analyze this system.')) {
    chunks.push(chunk);
  }
  
  assert.equal(chunks.join(''), 'Gemini stream!');
});

test('CascadeRouter - JeanSREGate - Injects SRE suggestions into chat()', async () => {
  let capturedBody;
  let sreCalled = false;

  const mockFetch = async (url, opts) => {
    if (url.includes('localhost:3005') || url.includes('custom-sre-url')) {
      sreCalled = true;
      return {
        ok: true,
        status: 200,
        json: async () => ({ reply: 'WARNING: CPU at 99C!' })
      };
    }

    capturedBody = JSON.parse(opts.body);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{ message: { content: 'SRE acknowledged' } }]
      })
    };
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://custom-sre-url/chat',
      projectContext: 'pocketlawyer'
    },
    fetch: mockFetch
  });

  const result = await router.chat('Fix this');
  
  assert.equal(sreCalled, true, 'Should have queried Jean SRE');
  assert.equal(result.routedTo, 'heavy');
  assert.equal(result.text, 'SRE acknowledged');

  const lastMessage = capturedBody.messages[capturedBody.messages.length - 1];
  assert.equal(lastMessage.role, 'system');
  assert.ok(lastMessage.content.includes('WARNING: CPU at 99C!'));
  assert.ok(lastMessage.content.includes('INSTRUCTION: Jean SRE has detected these fleet/system anomalies'));
});

test('CascadeRouter - JeanSREGate - Injects SRE suggestions into stream()', async () => {
  let capturedBody;
  let sreCalled = false;

  const mockFetch = async (url, opts) => {
    if (url.includes('localhost:3005') || url.includes('custom-sre-url')) {
      sreCalled = true;
      return {
        ok: true,
        status: 200,
        json: async () => ({ reply: 'WARNING: Docker loop!' })
      };
    }

    capturedBody = JSON.parse(opts.body);
    const encoder = new TextEncoder();
    const chunks = [
      { choices: [{ delta: { content: 'Stream ok' } }] }
    ].map(chunk => `data: ${JSON.stringify(chunk)}\n\n`);
    chunks.push('data: [DONE]\n\n');

    let read = false;
    const body_stream = {
      getReader: () => ({
        read: async () => {
          if (!read) {
            read = true;
            return { done: false, value: encoder.encode(chunks.join('')) };
          }
          return { done: true, value: undefined };
        },
        releaseLock: () => {}
      })
    };
    return { ok: true, status: 200, body: body_stream };
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://custom-sre-url/chat',
    },
    fetch: mockFetch
  });

  const chunks = [];
  for await (const chunk of router.stream('Hello')) {
    chunks.push(chunk);
  }

  assert.equal(sreCalled, true, 'Should have queried Jean SRE');
  assert.equal(chunks.join(''), 'Stream ok');

  const lastMessage = capturedBody.messages[capturedBody.messages.length - 1];
  assert.equal(lastMessage.role, 'system');
  assert.ok(lastMessage.content.includes('WARNING: Docker loop!'));
});

test('CascadeRouter - JeanSREGate - Gracefully degrades if SRE is offline', async () => {
  let capturedBody;
  let sreCalled = false;

  const mockFetch = async (url, opts) => {
    if (url.includes('localhost:3005') || url.includes('custom-sre-url')) {
      sreCalled = true;
      throw new Error('ECONNREFUSED');
    }

    capturedBody = JSON.parse(opts.body);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{ message: { content: 'Succeeded without SRE' } }]
      })
    };
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://custom-sre-url/chat'
    },
    fetch: mockFetch
  });

  const result = await router.chat('Fix this');
  
  assert.equal(sreCalled, true, 'Should have attempted SRE query');
  assert.equal(result.text, 'Succeeded without SRE');
  
  const lastMessage = capturedBody.messages[capturedBody.messages.length - 1];
  assert.equal(lastMessage.role, 'user');
  assert.equal(lastMessage.content, 'Fix this');
});

test('CascadeRouter - speedPriority: low - Routes to backgroundModel', async () => {
  const events = [];
  let fetchedUrl = '';
  
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    backgroundModel: { model: 'qwen2.5-background', url: 'http://background-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async (url, opts) => {
      fetchedUrl = url;
      return createMockFetch({
        fastTokens: ['Hello', ' background'],
        fastLogprobs: [0.95, 0.96],
        heavyText: 'Should not see this',
        fastShouldFail: false
      })(url, opts);
    },
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const result = await router.chat('Hello there', undefined, { speedPriority: 'low' });
  assert.equal(result.routedTo, 'fast');
  assert.equal(fetchedUrl, 'http://background-url/v1/chat/completions');
  assert.equal(result.text, 'Hello background');
  
  const routeEvent = events.find(e => e.event === 'route_fast');
  assert.ok(routeEvent);
  assert.equal(routeEvent.meta.model, 'qwen2.5-background');
  assert.equal(routeEvent.meta.speedPriority, 'low');
});

test('CascadeRouter - urgency: low - stream() routes to backgroundModel', async () => {
  let fetchedUrl = '';
  
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    backgroundModel: { model: 'qwen2.5-background', url: 'http://background-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async (url, opts) => {
      fetchedUrl = url;
      return createMockFetch({
        fastTokens: ['Hello', ' background'],
        fastLogprobs: [0.95, 0.96],
        heavyText: 'Should not see this',
        fastShouldFail: false
      })(url, opts);
    }
  });

  const chunks = [];
  for await (const chunk of router.stream('Hello there', undefined, { urgency: 'low' })) {
    chunks.push(chunk);
  }
  assert.equal(chunks.join(''), 'Hello background');
  assert.equal(fetchedUrl, 'http://background-url/v1/chat/completions');
});

test('CascadeRouter - Token Usage & Router Metrics Telemetry', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions', costPerMillionInputTokens: 0, costPerMillionOutputTokens: 0 },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key', costPerMillionInputTokens: 0.15, costPerMillionOutputTokens: 0.60 },
    fetch: createMockFetch({
      fastTokens: ['Fast', ' response'],
      fastLogprobs: [0.95, 0.96],
      heavyText: 'Heavy cloud response for complex problem',
      fastShouldFail: false
    })
  });

  // Fast request
  const fastRes = await router.chat('Simple hello');
  assert.equal(fastRes.routedTo, 'fast');
  assert.ok(fastRes.usage);
  assert.ok(fastRes.usage.promptTokens > 0);
  assert.ok(fastRes.usage.completionTokens > 0);
  assert.equal(fastRes.usage.estimatedCostUsd, 0);

  // Heavy request (complex verb "architect")
  const heavyRes = await router.chat('Please architect a distributed database.');
  assert.equal(heavyRes.routedTo, 'heavy');
  assert.ok(heavyRes.usage);
  assert.ok(heavyRes.usage.estimatedCostUsd > 0);

  // Check router metrics
  const metrics = router.getMetrics();
  assert.equal(metrics.totalRequests, 2);
  assert.equal(metrics.fastRequests, 1);
  assert.equal(metrics.heavyRequests, 1);
  assert.ok(metrics.totalPromptTokens > 0);
  assert.ok(metrics.totalCompletionTokens > 0);
  assert.ok(metrics.estimatedSavingsUsd > 0);

  router.resetMetrics();
  assert.equal(router.getMetrics().totalRequests, 0);
});

test('CascadeRouter - chatJson() Happy Path and Fallback Cascade on Malformed JSON', async () => {
  const events = [];

  // 1. Happy path: Fast model returns valid JSON
  const happyRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: ['{"status":', ' "ok",', ' "code": 200}'],
      fastLogprobs: [0.95, 0.95, 0.95],
      heavyText: '{"status": "heavy"}',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const happyJson = await happyRouter.chatJson('Get status');
  assert.equal(happyJson.status, 'ok');
  assert.equal(happyJson.code, 200);
  assert.equal(happyJson._routedTo, 'fast');

  // 2. Fallback path: Fast model returns non-JSON text -> triggers json fallback cascade
  const fallbackRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: ['Here', ' is your', ' plain text response without JSON.'],
      fastLogprobs: [0.95, 0.95, 0.95],
      heavyText: '```json\n{"status": "recovered_by_heavy", "valid": true}\n```',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const fallbackJson = await fallbackRouter.chatJson('Get data');
  assert.equal(fallbackJson.status, 'recovered_by_heavy');
  assert.equal(fallbackJson.valid, true);
  assert.equal(fallbackJson._routedTo, 'heavy');

  const fallbackEvent = events.find(e => e.event === 'json_fallback_triggered');
  assert.ok(fallbackEvent, 'json_fallback_triggered event must be emitted');
});

test('CascadeRouter - Speculative Repetition Loop Detection Triggers Cascade', async () => {
  const events = [];

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    maxRepetitiveTokens: 3,
    fetch: createMockFetch({
      // Fast model outputs repeating token loop
      fastTokens: ['loop', 'loop', 'loop', 'loop'],
      fastLogprobs: [0.99, 0.99, 0.99, 0.99],
      heavyText: 'Clean recovery from heavy model',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const result = await router.chat('Tell me something');
  assert.equal(result.routedTo, 'heavy');
  assert.equal(result.text, 'Clean recovery from heavy model');
  assert.equal(result.aborted, true);

  const loopEvent = events.find(e => e.event === 'repetition_loop_triggered');
  assert.ok(loopEvent, 'repetition_loop_triggered event must be emitted');
});

test('CascadeRouter - Cyclic 2-Gram Loop Triggers Cascade', async () => {
  const events = [];

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      // 2-gram cyclic loop: 'apple', 'banana', 'apple', 'banana', 'apple', 'banana'
      fastTokens: ['apple', 'banana', 'apple', 'banana', 'apple', 'banana'],
      fastLogprobs: [0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
      heavyText: 'Recovered from cyclic loop',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const result = await router.chat('Loop prompt');
  assert.equal(result.routedTo, 'heavy');
  assert.equal(result.text, 'Recovered from cyclic loop');
  assert.equal(result.aborted, true);
  const loopEvent = events.find(e => e.event === 'repetition_loop_triggered');
  assert.ok(loopEvent, 'cyclic repetition should trigger repetition_loop_triggered event');
});

test('CascadeRouter - Second Thought Speculative Branching for Borderline Query', async () => {
  const events = [];
  let heavyCalled = false;

  const mockFetch = async (url, opts) => {
    const body = JSON.parse(opts.body);
    if (body.stream) {
      // Fast model stream fails logprob on 2nd token
      const chunks = [
        { choices: [{ delta: { content: 'Intro ' }, logprobs: { content: [{ logprob: -0.01 }] } }] },
        { choices: [{ delta: { content: 'bad' }, logprobs: { content: [{ logprob: -2.5 }] } }] }
      ];
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => {
            let i = 0;
            return {
              read: async () => {
                if (i < chunks.length) {
                  const chunk = chunks[i++];
                  return { value: new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`), done: false };
                }
                return { done: true };
              }
            };
          }
        }
      };
    } else {
      // Heavy model
      heavyCalled = true;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{ message: { content: 'Speculative heavy answer' } }],
          usage: { prompt_tokens: 10, completion_tokens: 15, total_tokens: 25 }
        })
      };
    }
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    speculativeBranching: true,
    tokensToEvaluate: 2,
    cascadeThreshold: 0.85,
    fetch: mockFetch,
    onEvent: (event, meta) => events.push({ event, meta })
  });

  // Prompt with moderate complexity (borderline inquiry, not directly matching isComplexPrompt verbs)
  const borderlinePrompt = 'Could you explain why this happened and clarify the tradeoffs?';
  const res = await router.chat(borderlinePrompt);
  assert.equal(res.routedTo, 'heavy');
  assert.equal(res.text, 'Speculative heavy answer');
  assert.equal(heavyCalled, true);
  const hedgedEvent = events.find(e => e.event === 'speculative_branch_hedged');
  assert.ok(hedgedEvent, 'speculative_branch_hedged event should be emitted');
});

test('Knowledge Boundary and Continuous Complexity Scoring', async () => {
  const { detectKnowledgeBoundary, evaluateComplexityScore } = await import('./dist/index.js');
  
  // Closed-world tasks
  assert.equal(detectKnowledgeBoundary('Translate this paragraph to Spanish: Hello world'), 'closed');
  assert.equal(detectKnowledgeBoundary('Calculate 25 * 40 / 2'), 'closed');
  assert.equal(detectKnowledgeBoundary('Format this JSON string properly'), 'closed');

  // Open-world tasks
  assert.equal(detectKnowledgeBoundary('What are the ethical implications of autonomous AI in judicial systems?'), 'open');

  // Complexity score
  const simpleScore = evaluateComplexityScore('What is the capital of Vermont?');
  assert.ok(simpleScore < 0.35, `Simple score ${simpleScore} should be < 0.35`);

  const complexScore = evaluateComplexityScore('```typescript\nfunction analyze(x: number) { return x; }\n```\nSynthesize an architectural plan');
  assert.ok(complexScore >= 0.65, `Complex score ${complexScore} should be >= 0.65`);
});

test('classifySpecialistRole - Accurate Domain Classification for 5 Specialist Models', async () => {
  const { classifySpecialistRole } = await import('./dist/index.js');

  // 1. Games & Spatial (Chess) -> deepseek-v4-flash
  assert.equal(
    classifySpecialistRole('Given the board position after 1. e4 e5 2. Nf3, evaluate the best chess move and check for stalemate.'),
    'games_spatial'
  );

  // 2. Code Generation & Execution -> Qwen3-Coder-Next
  assert.equal(
    classifySpecialistRole('Generate an executable Python function to calculate the Fibonacci series:\ndef fib(n):'),
    'code'
  );

  // 3. Code Execution with Stdin / Complex Logic -> Qwen3-Coder-Next (code)
  assert.equal(
    classifySpecialistRole('Write a Python function that takes a list of integers and returns all prime numbers.'),
    'code'
  );

  // 4. Reading Comprehension & Text Verification -> qwen3-235b
  assert.equal(
    classifySpecialistRole('Based on the provided passage, what was the primary economic factor that led to the currency crisis?'),
    'comprehension_rc'
  );

  // 5. Financial Statements & Balance Sheets -> deepseek-v4-pro
  assert.equal(
    classifySpecialistRole('Table:\nFiscal year 2025 net income was $12.4B with operating income of $15.1B. Calculate diluted EPS.'),
    'reasoning_deep'
  );

  // 6. Open-ended Quiz Bowl without Options (QANTA) -> gemini-3.1-flash-lite (general_fast)
  assert.equal(
    classifySpecialistRole('Please read the following question and provide the correct answer.\n\nContext: None\n\nQuestion: This author wrote The Sound and the Fury and As I Lay Dying, set in Yoknapatawpha County.'),
    'general_fast'
  );

  // 7. General Fast / Multilingual / Geo / Medicine / Trivia -> gemini-3.1-flash-lite
  assert.equal(classifySpecialistRole('Translate this Gujarati paragraph into English.'), 'general_fast');
  assert.equal(classifySpecialistRole('What is the capital of Kazakhstan, its latitude, and neighboring countries?'), 'general_fast');
  assert.equal(classifySpecialistRole('Patient presents with acute chest pain and dyspnea. Clinical diagnosis indicates myocardial infarction.'), 'general_fast');

  // 8. Factual STEM / MMLU / Ethics / Science with Options -> deepseek-v4-flash
  assert.equal(classifySpecialistRole('From a utilitarian ethics perspective, analyze whether the moral dilemma permits action.'), 'factual_stem');
  assert.equal(
    classifySpecialistRole('Which of the following compounds has the highest boiling point?\nOptions:\nA. Water\nB. Methane\nC. Ethanol'),
    'factual_stem'
  );
});

test('createMultiSpecialistRouter - Preconfigures 5 Specialist Models via OpenRouter', async () => {
  const { createMultiSpecialistRouter } = await import('./dist/index.js');

  const interceptedCalls = [];
  const mockFetch = async (url, opts) => {
    const body = JSON.parse(opts.body);
    interceptedCalls.push({
      url,
      headers: opts.headers,
      model: body.model,
      messages: body.messages
    });

    return {
      ok: true,
      status: 200,
      json: async () => ({
        id: 'gen-test-123',
        choices: [{
          message: { role: 'assistant', content: `Response from ${body.model}` },
          finish_reason: 'stop'
        }],
        usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 }
      })
    };
  };

  const events = [];
  const router = createMultiSpecialistRouter({
    openrouterApiKey: 'sk-or-v1-mock-secret',
    siteUrl: 'https://krusch.homelab.dev',
    appName: 'Krusch Swarm Router',
    fetch: mockFetch,
    onEvent: (event, meta) => events.push({ event, meta })
  });

  // 1. Test Chess query routes to Qwen/Qwen3-Coder-Next
  const chessRes = await router.chat('What is the best chess continuation from this board position: 1. e4 e5?');
  assert.equal(chessRes.routedTo, 'games_spatial');
  assert.equal(chessRes.model, 'Qwen/Qwen3-Coder-Next');
  assert.equal(interceptedCalls[0].model, 'Qwen/Qwen3-Coder-Next');
  assert.equal(interceptedCalls[0].url, 'https://openrouter.ai/api/v1/chat/completions');
  assert.equal(interceptedCalls[0].headers['Authorization'], 'Bearer sk-or-v1-mock-secret');
  assert.equal(interceptedCalls[0].headers['HTTP-Referer'], 'https://krusch.homelab.dev');
  assert.equal(interceptedCalls[0].headers['X-Title'], 'Krusch Swarm Router');

  // 2. Test Code query routes to Qwen3-Coder-Next
  const codeRes = await router.chat('Generate an executable Python function to sort a list:\ndef quicksort(arr):');
  assert.equal(codeRes.routedTo, 'code');
  assert.equal(codeRes.model, 'Qwen/Qwen3-Coder-Next');
  assert.equal(interceptedCalls[1].model, 'Qwen/Qwen3-Coder-Next');

  // 3. Test Translation query routes to gemini-3.1-flash-lite
  const transRes = await router.chat('Translate this text from German to English: Guten Morgen!');
  assert.equal(transRes.routedTo, 'general_fast');
  assert.equal(transRes.model, 'google/gemini-3.1-flash-lite');

  // 4. Test STEM question routes to deepseek-v4-flash
  const stemRes = await router.chat('Calculate the force in Newtons when mass is 10kg and acceleration is 9.8 m/s^2.\nOptions:\nA. 98N\nB. 10N');
  assert.equal(stemRes.routedTo, 'factual_stem');
  assert.equal(stemRes.model, 'deepseek/deepseek-v4-flash');

  // Check telemetry events
  const routeSpecialistEvents = events.filter(e => e.event === 'route_specialist');
  assert.equal(routeSpecialistEvents.length, 4);
  assert.equal(router.config.specialistModels.games_spatial.model, 'Qwen/Qwen3-Coder-Next');
  assert.equal(router.config.specialistModels.comprehension_rc.model, 'qwen/qwen3-235b-a22b-2507');
});

test('createMultiSpecialistRouter - Cascades to reasoning_deep on specialist failure', async () => {
  const { createMultiSpecialistRouter } = await import('./dist/index.js');

  const calledModels = [];
  const mockFetch = async (url, opts) => {
    const body = JSON.parse(opts.body);
    calledModels.push(body.model);

    // Fail if calling chess specialist, succeed if calling heavy deep reasoning
    if (body.model === 'Qwen/Qwen3-Coder-Next') {
      return {
        ok: false,
        status: 502,
        text: async () => 'Bad Gateway'
      };
    }

    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'Fallback answer from deepseek-v4-pro' } }],
        usage: { prompt_tokens: 15, completion_tokens: 10, total_tokens: 25 }
      })
    };
  };

  const events = [];
  const router = createMultiSpecialistRouter({
    openrouterApiKey: 'sk-or-fallback-test',
    fetch: mockFetch,
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const res = await router.chat('What is the best chess continuation from this position: 1. e4 e5?');
  assert.equal(res.routedTo, 'reasoning_deep');
  assert.equal(res.aborted, true);
  assert.equal(calledModels[0], 'Qwen/Qwen3-Coder-Next');
  assert.equal(calledModels[1], 'deepseek/deepseek-v4-pro');

  const errorEvent = events.find(e => e.event === 'route_heavy' && e.meta?.reason === 'specialist_model_error');
  assert.ok(errorEvent, 'route_heavy event with specialist_model_error should be emitted');
});

test('createMultiSpecialistRouter - Preserves code routing even for complex prompts with code blocks and length > 2000', async () => {
  const { createMultiSpecialistRouter } = await import('./dist/index.js');

  const calledModels = [];
  const mockFetch = async (url, opts) => {
    const body = JSON.parse(opts.body);
    calledModels.push(body.model);
    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{ message: { role: 'assistant', content: 'def refactored(): pass' } }],
        usage: { prompt_tokens: 50, completion_tokens: 10, total_tokens: 60 }
      })
    };
  };

  const router = createMultiSpecialistRouter({
    openrouterApiKey: 'sk-or-code-test',
    fetch: mockFetch
  });

  // Long complex code prompt with markdown code block and cognitive verb 'refactor'
  const longCodePrompt = `Please refactor this complex Rust implementation:\n\`\`\`rust\nfn process_data(data: &[u8]) -> Result<Vec<u8>, Error> {\n    // Some code here\n    let mut result = Vec::new();\n    for byte in data {\n        result.push(byte.wrapping_add(1));\n    }\n    Ok(result)\n}\n\`\`\`\n` + 'Additional context: '.repeat(200);

  const res = await router.chat(longCodePrompt);
  assert.equal(res.routedTo, 'code', 'Complex code prompt must route to code specialist, NOT reasoning_deep');
  assert.equal(res.model, 'Qwen/Qwen3-Coder-Next');
  assert.equal(calledModels[0], 'Qwen/Qwen3-Coder-Next');
});

test('Hardening - 2-argument ChatOptions overload propagates speedPriority and signal in chat() and stream()', async () => {
  let fetchedUrls = [];
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    backgroundModel: { model: 'qwen2.5-bg', url: 'http://bg-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async (url, opts) => {
      fetchedUrls.push(url);
      return createMockFetch({
        fastTokens: ['Bg', ' ok'],
        fastLogprobs: [0.95, 0.96],
        heavyText: 'Heavy',
        fastShouldFail: false
      })(url, opts);
    }
  });

  // 2-argument call (no systemPrompt argument)
  const chatRes = await router.chat('Simple hello', { speedPriority: 'low' });
  assert.equal(chatRes.model, 'qwen2.5-bg');
  assert.equal(fetchedUrls[0], 'http://bg-url/v1/chat/completions');

  const streamChunks = [];
  for await (const chunk of router.stream('Simple hello', { urgency: 'low' })) {
    streamChunks.push(chunk);
  }
  assert.equal(streamChunks.join(''), 'Bg ok');
  assert.equal(fetchedUrls[1], 'http://bg-url/v1/chat/completions');
});

test('Hardening - Caller AbortSignal during fast model does NOT fall back to heavy model', async () => {
  let heavyCalled = false;
  const controller = new AbortController();

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async (url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.stream) {
        controller.abort();
        throw new Error('AbortError: caller cancelled');
      }
      heavyCalled = true;
      return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'Heavy' } }] }) };
    }
  });

  await assert.rejects(
    () => router.chat('Simple hello', { signal: controller.signal }),
    /AbortError/
  );
  assert.equal(heavyCalled, false, 'Heavy model must NOT be called when caller explicitly aborted');
});

test('Hardening - Speculative hedging does not trigger UnhandledPromiseRejection when fast model wins', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    speculativeBranching: true,
    tokensToEvaluate: 2,
    cascadeThreshold: 0.85,
    fetch: async (url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.stream) {
        // Fast model succeeds with high logprobs
        return createMockFetch({
          fastTokens: ['Fast', ' wins'],
          fastLogprobs: [0.98, 0.99],
          heavyText: 'Unused',
          fastShouldFail: false
        })(url, opts);
      } else {
        // Heavy model waits for abort signal and rejects when aborted
        return new Promise((_, reject) => {
          if (opts.signal?.aborted) {
            reject(new Error('Hedged request aborted'));
            return;
          }
          opts.signal?.addEventListener('abort', () => {
            reject(new Error('Hedged request aborted'));
          }, { once: true });
        });
      }
    }
  });

  const borderlinePrompt = 'Could you explain why this happened and clarify the tradeoffs?';
  const res = await router.chat(borderlinePrompt);
  assert.equal(res.routedTo, 'fast');
  assert.equal(res.text, 'Fast wins');
});

test('Hardening - streamGemini handles braces inside streamed code/JSON strings without hanging', async () => {
  const mockFetch = async () => {
    const encoder = new TextEncoder();
    const geminiPayload = `[
      {"candidates":[{"content":{"parts":[{"text":"function demo() {\\n  const obj = { a: \\"}\\" };\\n"}]}}]},
      {"candidates":[{"content":{"parts":[{"text":"  return obj;\\n}"}]}}]}
    ]`;
    let read = false;
    return {
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: async () => {
            if (!read) {
              read = true;
              return { done: false, value: encoder.encode(geminiPayload) };
            }
            return { done: true, value: undefined };
          },
          releaseLock: () => {}
        })
      }
    };
  };

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gemini-2.5-pro', apiKey: 'test-key', provider: 'gemini' },
    fetch: mockFetch
  });

  const chunks = [];
  for await (const chunk of router.stream('Please analyze this code.')) {
    chunks.push(chunk);
  }
  assert.equal(chunks.join(''), 'function demo() {\n  const obj = { a: "}" };\n  return obj;\n}');
});

test('Hardening - chatJson cascades to heavy model in Multi-Specialist mode when specialist returns invalid JSON', async () => {
  const { createMultiSpecialistRouter } = await import('./dist/index.js');
  const events = [];

  const router = createMultiSpecialistRouter({
    openrouterApiKey: 'test-key',
    onEvent: (event, meta) => events.push({ event, meta }),
    fetch: async (url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.model === 'deepseek/deepseek-v4-pro') {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            choices: [{ message: { content: '{"recovered": true, "specialist": false}' } }],
            usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 }
          })
        };
      }
      // Fast specialist returns invalid non-JSON text
      return {
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{ message: { content: 'Not valid JSON at all' } }],
          usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
        })
      };
    }
  });

  const parsed = await router.chatJson('Translate "hello" into Spanish as JSON');
  assert.equal(parsed.recovered, true);
  assert.equal(parsed._routedTo, 'heavy');
  assert.ok(events.some(e => e.event === 'json_fallback_triggered'));
});

test('Hardening - ReDoS immunity on pathological unclosed braces and long SQL-keyword prose', async () => {
  const { isComplexPrompt, evaluateComplexityScore, classifyPreRoute } = await import('./dist/index.js');

  const pathologicalBraces = '{ ' + '"key": "value", '.repeat(3000); // No closing '}'
  const pathologicalSelect = 'Please select the best items and update your notes ' + 'word '.repeat(3000);

  const t0 = performance.now();
  const isComp = isComplexPrompt(pathologicalBraces);
  const score = evaluateComplexityScore(pathologicalBraces);
  const route1 = classifyPreRoute(pathologicalBraces);
  const route2 = classifyPreRoute(pathologicalSelect);
  const elapsedMs = performance.now() - t0;

  assert.equal(isComp, true);
  assert.ok(score >= 0);
  assert.ok(route1);
  assert.ok(route2);
  assert.ok(elapsedMs < 25, `Pathological inputs must evaluate in <25ms without catastrophic backtracking (took ${elapsedMs.toFixed(2)}ms)`);
});

test('Hardening - L2 Centroid Router caches repeated prompt embeddings and recovers from transient cold-start failure', async () => {
  const { createCentroidSemanticRouter } = await import('./dist/index.js');

  let embedCalls = 0;
  let shouldFail = true;

  const centroids = [
    {
      archetype: 'code_test',
      role: 'code',
      tier: 'specialist',
      label: 'Code Refactoring',
      exemplar: 'refactor code',
      threshold: 0.70
    }
  ];

  const router = createCentroidSemanticRouter({
    centroids,
    cacheSize: 16,
    embed: async (text) => {
      embedCalls++;
      if (shouldFail) return null;
      return [1.0, 0.0, 0.0, 0.0];
    }
  });

  // 1. First call while embedding service is temporarily offline returns null
  const firstAttempt = await router('How do I structure this module?');
  assert.equal(firstAttempt, null);

  // 2. Service recovers: next call warms centroids and embeds prompt
  shouldFail = false;
  embedCalls = 0;
  const secondAttempt = await router('How do I structure this module?');
  assert.ok(secondAttempt);
  assert.equal(secondAttempt.recommendedRole, 'code');
  assert.equal(embedCalls, 2, 'Should embed 1 centroid exemplar + 1 prompt');

  // 3. Repeated call with same prompt hits LRU cache (0 new embed calls!)
  const thirdAttempt = await router('How do I structure this module?');
  assert.ok(thirdAttempt);
  assert.equal(thirdAttempt.recommendedRole, 'code');
  assert.equal(embedCalls, 2, 'Repeated prompt should be served from LRU cache with 0 new embed calls');
});

test('Pass 2 Hardening - chatJson extracts balanced JSON when surrounded by prose containing braces or brackets', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: [
        'Note {ref-1}: Here is your payload: ',
        '{"status": "ok", "expr": "a } b", "items": [1, 2]}',
        ' (see footnote [2] or {appendix} for details)'
      ],
      fastLogprobs: [0.98, 0.99, 0.97],
      heavyText: '{"status": "heavy"}',
      fastShouldFail: false
    })
  });

  const parsed = await router.chatJson('Get status');
  assert.equal(parsed.status, 'ok');
  assert.equal(parsed.expr, 'a } b');
  assert.deepEqual(parsed.items, [1, 2]);
  assert.equal(parsed._routedTo, 'fast');
});

test('Pass 2 Hardening - Entropy collapse and 3-gram cyclic loops emit telemetry and trigger cascade', async () => {
  const events = [];

  // 1. Entropy collapse (10 tokens alternating irregularly between 2 words so not a simple 2-gram or 4x identical)
  const collapseRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    tokensToEvaluate: 15,
    maxRepetitiveTokens: 4,
    fetch: createMockFetch({
      fastTokens: ['yes', 'no', 'yes', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'yes'],
      fastLogprobs: Array(10).fill(0.99),
      heavyText: 'Recovered from entropy collapse',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const res1 = await collapseRouter.chat('Simple question');
  assert.equal(res1.routedTo, 'heavy');
  assert.equal(res1.text, 'Recovered from entropy collapse');
  assert.ok(events.some(e => e.event === 'entropy_collapse_triggered'), 'Should emit entropy_collapse_triggered');

  // 2. 3-gram cyclic loop (A, B, C, A, B, C, A, B, C)
  const events3Gram = [];
  const triGramRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    tokensToEvaluate: 15,
    fetch: createMockFetch({
      fastTokens: ['one', 'two', 'three', 'one', 'two', 'three', 'one', 'two', 'three'],
      fastLogprobs: Array(9).fill(0.99),
      heavyText: 'Recovered from 3-gram loop',
      fastShouldFail: false
    }),
    onEvent: (event, meta) => events3Gram.push({ event, meta })
  });

  const res2 = await triGramRouter.chat('Simple count');
  assert.equal(res2.routedTo, 'heavy');
  assert.equal(res2.text, 'Recovered from 3-gram loop');
  assert.ok(events3Gram.some(e => e.event === 'repetition_loop_triggered'));
});

test('Pass 2 Hardening - stream() cancels underlying reader on early consumer break', async () => {
  let cancelCalled = false;
  let releaseCalled = false;
  const encoder = new TextEncoder();

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    tokensToEvaluate: 1,
    fetch: async () => ({
      ok: true,
      status: 200,
      body: {
        getReader: () => {
          let step = 0;
          return {
            read: async () => {
              step++;
              if (step <= 5) {
                const chunk = { choices: [{ delta: { content: `tok${step} ` }, logprobs: { content: [{ logprob: -0.01 }] } }] };
                return { done: false, value: encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`) };
              }
              return { done: true, value: undefined };
            },
            cancel: async () => { cancelCalled = true; },
            releaseLock: () => { releaseCalled = true; }
          };
        }
      }
    })
  });

  const received = [];
  for await (const chunk of router.stream('Hello')) {
    received.push(chunk);
    break; // Consumer breaks after first yielded chunk
  }

  assert.equal(received.length, 1);
  assert.equal(cancelCalled, true, 'reader.cancel() must be called when consumer breaks early');
  assert.equal(releaseCalled, true, 'reader.releaseLock() must be called');
});

test('Pass 2 Hardening - JeanSREGate respects timeoutMs and cacheTtlMs', async () => {
  let sreFetchCount = 0;

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://sre-test/chat',
      timeoutMs: 50,
      cacheTtlMs: 5000
    },
    fetch: async (url, opts) => {
      if (url.includes('sre-test')) {
        sreFetchCount++;
        return {
          ok: true,
          status: 200,
          json: async () => ({ reply: 'Cached SRE alert' })
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Heavy ok' } }] })
      };
    }
  });

  await router.chat('Please analyze this architecture');
  await router.chat('Please analyze this second architecture');
  assert.equal(sreFetchCount, 1, 'Second request within cacheTtlMs must use cached SRE reply');

  // Test timeout degradation when SRE endpoint stalls
  const stallingRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://sre-stall/chat',
      timeoutMs: 30
    },
    fetch: async (url, opts) => {
      if (url.includes('sre-stall')) {
        return new Promise((_, reject) => {
          opts.signal?.addEventListener('abort', () => reject(new Error('SRE Timed Out')), { once: true });
        });
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Completed despite SRE stall' } }] })
      };
    }
  });

  const t0 = performance.now();
  const res = await stallingRouter.chat('Please analyze this');
  const elapsed = performance.now() - t0;
  assert.equal(res.text, 'Completed despite SRE stall');
  assert.ok(elapsed < 500, `Stalled SRE must abort quickly via timeoutMs (took ${elapsed.toFixed(1)}ms)`);
});

test('Pass 2 Hardening - L2 Centroid Router self-heals partial centroid warmup failure', async () => {
  const { createCentroidSemanticRouter } = await import('./dist/index.js');

  let failSecondCentroid = true;
  const centroids = [
    {
      archetype: 'c1',
      role: 'general_fast',
      tier: 'specialist',
      label: 'General',
      exemplar: 'ex1',
      threshold: 0.70
    },
    {
      archetype: 'c2',
      role: 'code',
      tier: 'specialist',
      label: 'Code',
      exemplar: 'ex2',
      threshold: 0.70
    }
  ];

  const router = createCentroidSemanticRouter({
    centroids,
    embed: async (text) => {
      if (text === 'ex1') return [1, 0];
      if (text === 'ex2') {
        if (failSecondCentroid) return null; // Transient failure on 2nd centroid only
        return [0, 1];
      }
      return [0, 1]; // Query vector matches c2 ('code')
    }
  });

  // First call: c1 warms ([1,0]), c2 fails (null). Query [0,1] only compares against c1 (dot=0 < 0.70 -> escalates)
  const r1 = await router('my query');
  assert.equal(r1.recommendedRole, 'reasoning_deep');

  // Second call: c2 is retried and warms ([0,1]). Query [0,1] now matches c2 with similarity 1.0!
  failSecondCentroid = false;
  const r2 = await router('my query');
  assert.equal(r2.recommendedRole, 'code');
  assert.ok(r2.confidence > 0.99);
});

test('Pass 3 Hardening - chatJson attaches _routedTo and _usage as non-enumerable properties on arrays and objects', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: ['[{"id": 1}, {"id": 2}]'],
      fastLogprobs: [0.99],
      heavyText: '[]',
      fastShouldFail: false
    })
  });

  const arr = await router.chatJson('List items');
  assert.equal(Array.isArray(arr), true);
  assert.equal(arr._routedTo, 'fast');
  assert.ok(arr._usage);
  // Non-enumerable metadata must not pollute JSON.stringify, Object.keys, or deepEqual
  assert.deepEqual(Object.keys(arr), ['0', '1']);
  assert.equal(JSON.stringify(arr), '[{"id":1},{"id":2}]');
  assert.deepEqual(arr, [{ id: 1 }, { id: 2 }]);
});

test('Pass 3 Hardening - streamGemini incrementally scans multi-chunk JSON payloads without re-scanning state corruption', async () => {
  const encoder = new TextEncoder();
  const rawChunks = [
    '[\n  {"candidates":[{"content":{"parts":[{"text":"Chunk 1 { ',
    'still in string } ok"}]}}]},\n  {"candidates":[{"content":',
    '{"parts":[{"text":" and Chunk 2!"}]}}]}\n]'
  ].map(s => encoder.encode(s));

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gemini-2.5-pro', apiKey: 'test-key', provider: 'gemini' },
    fetch: async () => {
      let idx = 0;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              if (idx < rawChunks.length) {
                return { done: false, value: rawChunks[idx++] };
              }
              return { done: true, value: undefined };
            },
            releaseLock: () => {}
          })
        }
      };
    }
  });

  const yielded = [];
  for await (const chunk of router.stream('Please analyze this multi-chunk payload.')) {
    yielded.push(chunk);
  }
  assert.equal(yielded.join(''), 'Chunk 1 { still in string } ok and Chunk 2!');
});

test('Pass 3 Hardening - CascadeRouter with only fastModel configured defaults heavyModel safely', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5-only', url: 'http://fast-url/v1/chat/completions' },
    fetch: createMockFetch({
      fastTokens: ['Fast'],
      fastLogprobs: [0.99],
      heavyText: 'Fallback to same fastModel in non-stream mode',
      fastShouldFail: false
    })
  });

  const res = await router.chat('Please analyze this complex prompt with only fastModel configured');
  assert.equal(res.model, 'qwen2.5-only');
  assert.equal(res.text, 'Fallback to same fastModel in non-stream mode');
});

test('Pass 4 Hardening - stream() records request, token, and cascade metrics in router.getMetrics()', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://localhost:11434/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    tokensToEvaluate: 2,
    cascadeThreshold: 0.85,
    fetch: createMockFetch({
      fastTokens: ['Fast', ' stream', ' ok'],
      fastLogprobs: [0.99, 0.98, 0.99],
      heavyText: 'Heavy stream fallback',
      fastShouldFail: false
    })
  });

  // 1. Fast stream completion
  const fastChunks = [];
  for await (const chunk of router.stream('Hello world')) {
    fastChunks.push(chunk);
  }
  assert.equal(fastChunks.join(''), 'Fast stream ok');

  let metrics = router.getMetrics();
  assert.equal(metrics.totalRequests, 1);
  assert.equal(metrics.fastRequests, 1);
  assert.equal(metrics.heavyRequests, 0);
  assert.equal(metrics.cascadedRequests, 0);
  assert.ok(metrics.totalCompletionTokens > 0);

  // 2. Cascaded stream fallback (low logprobs on fast model)
  const encoder = new TextEncoder();
  const cascadeStreamRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://localhost:11434/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    tokensToEvaluate: 2,
    cascadeThreshold: 0.85,
    fetch: async (url, opts) => {
      const body = JSON.parse(opts.body);
      const isFast = body.model === 'qwen2.5';
      const sse = isFast
        ? [
            `data: ${JSON.stringify({ choices: [{ delta: { content: 'Low' }, logprobs: { content: [{ logprob: Math.log(0.10) }] } }] })}\n\n`,
            `data: ${JSON.stringify({ choices: [{ delta: { content: ' conf' }, logprobs: { content: [{ logprob: Math.log(0.20) }] } }] })}\n\n`,
            `data: [DONE]\n\n`
          ].join('')
        : [
            `data: ${JSON.stringify({ choices: [{ delta: { content: 'Heavy stream fallback' } }] })}\n\n`,
            `data: [DONE]\n\n`
          ].join('');
      let read = false;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              if (!read) {
                read = true;
                return { done: false, value: encoder.encode(sse) };
              }
              return { done: true, value: undefined };
            },
            releaseLock: () => {}
          })
        }
      };
    }
  });

  const heavyChunks = [];
  for await (const chunk of cascadeStreamRouter.stream('Hello world')) {
    heavyChunks.push(chunk);
  }
  assert.equal(heavyChunks.join(''), 'Heavy stream fallback');

  metrics = cascadeStreamRouter.getMetrics();
  assert.equal(metrics.totalRequests, 1, 'Cascaded stream should count as 1 total request, not 2');
  assert.equal(metrics.fastRequests, 0);
  assert.equal(metrics.heavyRequests, 1);
  assert.equal(metrics.cascadedRequests, 1);
});

test('Pass 4 Hardening - fetchGemini and streamGemini preserve all system messages (including JeanSREGate)', async () => {
  let capturedBody = null;

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gemini-2.5-pro', apiKey: 'test-key', provider: 'gemini' },
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://sre-mock/chat'
    },
    fetch: async (url, opts) => {
      if (url.includes('sre-mock')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ reply: 'GPU0 temp 81C' })
        };
      }
      capturedBody = JSON.parse(opts.body);
      return {
        ok: true,
        status: 200,
        json: async () => ({
          candidates: [{ content: { parts: [{ text: 'Gemini response' }] } }]
        })
      };
    }
  });

  await router.chat('Please analyze this system state', 'You are a principal SRE.');
  assert.ok(capturedBody?.system_instruction?.parts?.[0]?.text);
  const sysText = capturedBody.system_instruction.parts[0].text;
  assert.ok(sysText.includes('You are a principal SRE.'), 'Must include caller system prompt');
  assert.ok(sysText.includes('GPU0 temp 81C'), 'Must also include JeanSREGate system prompt');
});

test('Pass 4 Hardening - SSE parser handles no-space data: lines and vLLM token_logprobs format', async () => {
  const encoder = new TextEncoder();

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    tokensToEvaluate: 2,
    cascadeThreshold: 0.85,
    fetch: async (url) => {
      if (url.includes('localhost:11434')) {
        // Return no-space `data:{...}` and `logprobs.token_logprobs` array of log-probabilities
        const lines = [
          `data:${JSON.stringify({ choices: [{ delta: { content: ' Low' }, logprobs: { token_logprobs: [-2.3] } }] })}\n\n`,
          `data:${JSON.stringify({ choices: [{ delta: { content: ' conf' }, logprobs: { token_logprobs: [-2.3] } }] })}\n\n`,
          `data:[DONE]\n\n`
        ];
        let idx = 0;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (idx < lines.length) {
                  return { done: false, value: encoder.encode(lines[idx++]) };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Cascaded via token_logprobs' } }] })
      };
    }
  });

  const res = await router.chat('Simple question');
  assert.equal(res.routedTo, 'heavy');
  assert.equal(res.text, 'Cascaded via token_logprobs');
});

test('Pass 4 Hardening - createContextMcpRouter accepts direct object and JSON string responses', async () => {
  const { createContextMcpRouter } = await import('./dist/index.js');

  const directRouter = createContextMcpRouter({
    callTool: async () => ({
      recommended_role: 'games_spatial',
      target_tier: 'specialist',
      confidence: 0.91,
      reason: 'direct_object_response'
    })
  });

  const decision = await directRouter('Move knight to f3');
  assert.ok(decision);
  assert.equal(decision.recommendedRole, 'games_spatial');
  assert.equal(decision.targetTier, 'specialist');
  assert.equal(decision.confidence, 0.91);
  assert.equal(decision.reason, 'direct_object_response');
});

test('Pass 5 Hardening - chatJson enforces schema without false complexity escalation and avoids double-counting cascade metrics', async () => {
  let capturedSystemPrompt = '';
  const schemaRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://localhost:11434/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async (url, opts) => {
      const body = JSON.parse(opts.body);
      const sysMsg = body.messages.find(m => m.role === 'system');
      capturedSystemPrompt = sysMsg?.content || '';
      return createMockFetch({
        fastTokens: ['{"name": "Krusch"}'],
        fastLogprobs: [0.99],
        heavyText: '{"name": "Heavy"}',
        fastShouldFail: false
      })(url, opts);
    }
  });

  const schemaResult = await schemaRouter.chatJson('Return user name', {
    schema: { type: 'object', properties: { name: { type: 'string' } }, required: ['name'] }
  });
  assert.equal(schemaResult.name, 'Krusch');
  assert.equal(schemaResult._routedTo, 'fast', 'Schema option must not falsely escalate simple prompt to heavy model');
  assert.ok(capturedSystemPrompt.includes('Required JSON Schema:'), 'System prompt must include schema instruction');

  // Verify chatJson fallback to heavy model does not double-count totalRequests or leave fastRequests=1
  const fallbackMetricsRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://localhost:11434/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: ['Not valid JSON at all'],
      fastLogprobs: [0.99],
      heavyText: '{"recovered": true}',
      fastShouldFail: false
    })
  });

  const recovered = await fallbackMetricsRouter.chatJson('Get status');
  assert.equal(recovered.recovered, true);
  assert.equal(recovered._routedTo, 'heavy');

  const m = fallbackMetricsRouter.getMetrics();
  assert.equal(m.totalRequests, 1, 'Single cascaded chatJson call must record totalRequests = 1');
  assert.equal(m.fastRequests, 0, 'Failed fast JSON attempt must be reclassified to cascaded heavy');
  assert.equal(m.heavyRequests, 1);
  assert.equal(m.cascadedRequests, 1);
  assert.equal(m.estimatedSavingsUsd, 0, 'Failed fast JSON attempt must not retain false cost savings');
});

test('Pass 5 Hardening - chatJson respects maxJsonRetries even when initially routed to heavy model', async () => {
  let heavyAttempts = 0;

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async () => {
      heavyAttempts++;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{
            message: {
              content: heavyAttempts === 1 ? 'Malformed first heavy output' : '{"retrySuccess": true}'
            }
          }]
        })
      };
    }
  });

  const res = await router.chatJson('Please analyze and return JSON', { maxJsonRetries: 2 });
  assert.equal(res.retrySuccess, true);
  assert.equal(heavyAttempts, 2, 'Heavy model should be retried when maxJsonRetries = 2');
  assert.equal(router.getMetrics().totalRequests, 1, 'Retries within a single chatJson call should count as 1 request');
});

test('Pass 5 Hardening - In-band HTTP 200 and SSE error payloads trigger cascade fallback to heavy model', async () => {
  const { createMultiSpecialistRouter } = await import('./dist/index.js');

  // 1. Multi-Specialist HTTP 200 with in-band { error: { message: "..." } }
  const specialistRouter = createMultiSpecialistRouter({
    openrouterApiKey: 'test-key',
    fetch: async (_url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.model !== 'deepseek/deepseek-v4-pro') {
        return {
          ok: true,
          status: 200,
          json: async () => ({ error: { message: 'Upstream provider overloaded', code: 502 } })
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{ message: { content: 'Recovered by reasoning_deep after in-band HTTP 200 error' } }]
        })
      };
    }
  });

  const res1 = await specialistRouter.chat('Write a Python function to reverse a string');
  assert.equal(res1.routedTo, 'reasoning_deep');
  assert.equal(res1.text, 'Recovered by reasoning_deep after in-band HTTP 200 error');

  // 2. Standard 2-Tier SSE stream with in-band data: {"error": ...}
  const encoder = new TextEncoder();
  const sseErrorRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: async (url) => {
      if (url.includes('localhost:11434')) {
        let read = false;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (!read) {
                  read = true;
                  return {
                    done: false,
                    value: encoder.encode(`data: ${JSON.stringify({ error: { message: 'Mid-stream GPU OOM' } })}\n\n`)
                  };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Recovered from mid-stream SSE error' } }] })
      };
    }
  });

  const res2 = await sseErrorRouter.chat('Simple question');
  assert.equal(res2.routedTo, 'heavy');
  assert.equal(res2.text, 'Recovered from mid-stream SSE error');
});

test('Pass 5 Hardening - stream() with tokensToEvaluate: 0 yields immediately and evaluates logprobs on empty-delta chunks', async () => {
  const encoder = new TextEncoder();

  // 1. tokensToEvaluate: 0 yields immediately on first chunk even without logprobs
  const zeroEvalRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    tokensToEvaluate: 0,
    fetch: async () => {
      let step = 0;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              step++;
              if (step === 1) {
                return {
                  done: false,
                  value: encoder.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Immediate' } }] })}\n\n`)
                };
              }
              return { done: true, value: undefined };
            },
            releaseLock: () => {}
          })
        }
      };
    }
  });

  const chunks = [];
  for await (const c of zeroEvalRouter.stream('Hi')) {
    chunks.push(c);
  }
  assert.deepEqual(chunks, ['Immediate']);
});

test('Pass 6 Hardening - Default local fastModel without explicit url accrues estimatedSavingsUsd at 0.0 local cost', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' }, // No url or costPerMillionInputTokens specified (defaults to localhost:11434)
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: ['Local', ' free', ' response'],
      fastLogprobs: [0.99, 0.99, 0.99],
      heavyText: 'Cloud response',
      fastShouldFail: false
    })
  });

  const res = await router.chat('Hello local model');
  assert.equal(res.routedTo, 'fast');
  assert.equal(res.usage.estimatedCostUsd, 0, 'Default local fastModel must have 0.0 estimatedCostUsd');
  const metrics = router.getMetrics();
  assert.equal(metrics.estimatedCostUsd, 0);
  assert.ok(metrics.estimatedSavingsUsd > 0, 'Default local fastModel must record positive estimatedSavingsUsd vs cloud heavy baseline');
});

test('Pass 6 Hardening - L2 targetTier heavy/frontier escalates to reasoning_deep and emits recommendedModel', async () => {
  const { createMultiSpecialistRouter } = await import('./dist/index.js');
  const events = [];

  const router = createMultiSpecialistRouter({
    openrouterApiKey: 'test-key',
    l2Router: async () => ({
      recommendedRole: 'factual_stem',
      targetTier: 'frontier',
      recommendedModel: 'claude-3-5-sonnet',
      confidence: 0.94,
      reason: 'High-stakes safety escalation'
    }),
    fetch: async (_url, opts) => {
      const body = JSON.parse(opts.body);
      return {
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{ message: { content: `Handled by ${body.model}` } }]
        })
      };
    },
    onEvent: (event, meta) => events.push({ event, meta })
  });

  const res = await router.chat('What should we do about the organizational restructuring next quarter?');
  assert.equal(res.routedTo, 'reasoning_deep', 'L2 frontier tier should escalate targetRole to reasoning_deep');
  assert.equal(res.text, 'Handled by deepseek/deepseek-v4-pro');

  const l2Evt = events.find(e => e.event === 'route_l2_semantic');
  assert.ok(l2Evt);
  assert.equal(l2Evt.meta.recommendedModel, 'claude-3-5-sonnet');
});

test('Pass 6 Hardening - createCentroidSemanticRouter normalizes trailing slash and supports embeddings[] array format', async () => {
  const { createCentroidSemanticRouter } = await import('./dist/index.js');
  let requestedUrl = '';

  const router = createCentroidSemanticRouter({
    ollamaUrl: 'http://127.0.0.1:11434///',
    centroids: [
      {
        archetype: 'pre_warmed_code',
        role: 'code',
        tier: 'specialist',
        label: 'Pre-warmed Code',
        exemplar: 'write code',
        embedding: [1, 0, 0], // Static pre-supplied embedding
        threshold: 0.70
      }
    ],
    fetch: async (url) => {
      requestedUrl = url;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          embeddings: [[1, 0, 0]] // Modern Ollama /api/embed response shape
        })
      };
    }
  });

  const res = await router('How do I structure this module?');
  assert.equal(requestedUrl, 'http://127.0.0.1:11434/api/embeddings', 'Trailing slashes on ollamaUrl must be normalized');
  assert.ok(res);
  assert.equal(res.recommendedRole, 'code');
});

test('Pass 7 Hardening - NaN logprobs do not poison avgProb and safely trigger cascade fallback', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    tokensToEvaluate: 2,
    cascadeThreshold: 0.85,
    fetch: createMockFetch({
      fastTokens: ['Uncertain', ' token'],
      // Previously Math.exp(NaN) -> NaN -> (NaN < 0.85 === false) bypassed cascade!
      fastLogprobs: [NaN, NaN],
      heavyText: 'Heavy rescued from NaN logprob',
      fastShouldFail: false
    })
  });

  const res = await router.chat('Simple question');
  assert.equal(res.routedTo, 'heavy', 'NaN logprobs must be clamped to 0 confidence and trigger cascade fallback');
  assert.equal(res.aborted, true);
  assert.equal(router.getMetrics().cascadedRequests, 1);
  assert.equal(res.text, 'Heavy rescued from NaN logprob');
});

test('Pass 7 Hardening - Gemini provider supported on fastModel and in-band Gemini errors trigger fallback', async () => {
  const encoder = new TextEncoder();

  // 1. fastModel with provider: 'gemini' succeeds in both chat() and stream()
  const geminiFastRouter = new CascadeRouter({
    fastModel: { model: 'gemini-2.5-flash', provider: 'gemini', apiKey: 'gem-key' },
    heavyModel: { model: 'gpt-4o', apiKey: 'openai-key' },
    fetch: async (url) => {
      if (url.includes(':generateContent')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            candidates: [{ content: { parts: [{ text: 'Gemini Flash fast chat' }] } }],
            usageMetadata: { promptTokenCount: 8, candidatesTokenCount: 4 }
          })
        };
      }
      if (url.includes(':streamGenerateContent')) {
        let done = false;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (!done) {
                  done = true;
                  return {
                    done: false,
                    value: encoder.encode('[{"candidates":[{"content":{"parts":[{"text":"Gemini Flash stream"}]}}]}]')
                  };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    }
  });

  const chatRes = await geminiFastRouter.chat('Hi');
  assert.equal(chatRes.routedTo, 'fast');
  assert.equal(chatRes.text, 'Gemini Flash fast chat');

  const streamChunks = [];
  for await (const c of geminiFastRouter.stream('Hi')) {
    streamChunks.push(c);
  }
  assert.deepEqual(streamChunks, ['Gemini Flash stream']);

  // 2. In-band Gemini error payload in streamGemini throws and cascades to heavyModel
  const geminiErrRouter = new CascadeRouter({
    fastModel: { model: 'gemini-2.5-flash', provider: 'gemini', apiKey: 'gem-key' },
    heavyModel: { model: 'gpt-4o', apiKey: 'openai-key' },
    fetch: async (url) => {
      if (url.includes('generativelanguage.googleapis.com')) {
        let done = false;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (!done) {
                  done = true;
                  return {
                    done: false,
                    value: encoder.encode('[{"error":{"code":429,"message":"Resource exhausted"}}]')
                  };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => {
            let done = false;
            return {
              read: async () => {
                if (!done) {
                  done = true;
                  return {
                    done: false,
                    value: encoder.encode(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Heavy fallback after Gemini error' } }] })}\n\n`)
                  };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            };
          }
        }
      };
    }
  });

  const recoveredChunks = [];
  for await (const c of geminiErrRouter.stream('Hi')) {
    recoveredChunks.push(c);
  }
  assert.deepEqual(recoveredChunks, ['Heavy fallback after Gemini error']);
});

test('Pass 7 Hardening - JeanSREGate forwards per-request options.project and caches per project without thrashing', async () => {
  const sreBodies = [];

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://sre-multi/chat',
      cacheTtlMs: 10000
    },
    fetch: async (url, opts) => {
      if (url.includes('sre-multi')) {
        const parsed = JSON.parse(opts.body);
        sreBodies.push(parsed.message);
        return {
          ok: true,
          status: 200,
          json: async () => ({ reply: `SRE ok for ${parsed.message.slice(0, 35)}` })
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Done' } }] })
      };
    }
  });

  await router.chat('Please analyze system A', { project: 'alpha' });
  await router.chat('Please analyze system B', { project: 'beta' });
  // Alternating back to alpha and beta within TTL must hit the per-project Map cache (0 extra SRE fetches)
  await router.chat('Please analyze system A again', { project: 'alpha' });
  await router.chat('Please analyze system B again', { project: 'beta' });

  assert.equal(sreBodies.length, 2, 'Per-project SRE cache must not thrash when alternating between projects');
  assert.ok(sreBodies[0].includes('[Project Isolation Context: alpha]'));
  assert.ok(sreBodies[1].includes('[Project Isolation Context: beta]'));
});

test('Pass 7 Hardening - formatMessages safely handles nullish entries and non-string message content', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: ['Safe'],
      fastLogprobs: [0.99],
      heavyText: 'Heavy',
      fastShouldFail: false
    })
  });

  const res = await router.chat([
    null,
    undefined,
    { role: 'user', content: 12345 },
    { role: 'user', content: 'Hello' }
  ]);
  assert.equal(res.text, 'Safe');
});

test('Pass 8 Hardening - parseJsonSafe extracts ```json fence even when preceded by non-JSON code fences', async () => {
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    fetch: createMockFetch({
      fastTokens: [
        'Here is how NOT to write it:\n```ts\nconst bad = {"broken": undefined};\n```\nAnd here is the valid JSON:\n```json\n{"extracted": "from_second_fence", "count": 3}\n```'
      ],
      fastLogprobs: [0.99],
      heavyText: '{"extracted": "heavy"}',
      fastShouldFail: false
    })
  });

  const parsed = await router.chatJson('Give me the payload');
  assert.equal(parsed._routedTo, 'fast');
  assert.equal(parsed.extracted, 'from_second_fence');
  assert.equal(parsed.count, 3);
});

test('Pass 8 Hardening - Speculative hedging emits route_heavy on fast error and retries cleanly if hedge fails on cascade', async () => {
  // 1. Fast model throws HTTP 503 while hedgedHeavyPromise succeeds -> must still emit route_heavy
  const events1 = [];
  const router1 = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    speculativeBranching: true,
    fetch: async (_url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.stream) {
        return { ok: false, status: 503 };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Hedged heavy rescued error' } }] })
      };
    },
    onEvent: (event, meta) => events1.push({ event, meta })
  });

  const borderlinePrompt = 'Could you explain why this happened and clarify the tradeoffs?';
  const res1 = await router1.chat(borderlinePrompt);
  assert.equal(res1.routedTo, 'heavy');
  assert.equal(res1.text, 'Hedged heavy rescued error');
  assert.ok(events1.some(e => e.event === 'route_heavy' && e.meta.reason === 'fast_model_error'), 'route_heavy must be emitted even when hedgedHeavyPromise resolves the fallback');

  // 2. Fast model aborts on low confidence AND hedgedHeavyPromise transiently fails -> retries heavy without duplicate route_heavy
  const events2 = [];
  let heavyAttempt = 0;
  const encoder = new TextEncoder();
  const router2 = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    speculativeBranching: true,
    tokensToEvaluate: 1,
    cascadeThreshold: 0.85,
    fetch: async (_url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.stream) {
        let done = false;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (!done) {
                  done = true;
                  const chunk = { choices: [{ delta: { content: 'um' }, logprobs: { content: [{ logprob: -2.0 }] } }] };
                  return { done: false, value: encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`) };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }
      heavyAttempt++;
      if (heavyAttempt === 1) {
        throw new Error('Transient hedge failure');
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Second heavy attempt succeeded' } }] })
      };
    },
    onEvent: (event, meta) => events2.push({ event, meta })
  });

  const res2 = await router2.chat(borderlinePrompt);
  assert.equal(res2.routedTo, 'heavy');
  assert.equal(res2.text, 'Second heavy attempt succeeded');
  const heavyEvents = events2.filter(e => e.event === 'route_heavy');
  assert.equal(heavyEvents.length, 1, 'Should emit exactly 1 route_heavy event (cascade_fallback) without duplicate fast_model_error');
  assert.equal(heavyEvents[0].meta.reason, 'cascade_fallback');
});

test('Pass 8 Hardening - chatJson fallback to heavyModel preserves JeanSREGate system prompt', async () => {
  let heavyMessages = [];

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'test-key' },
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://sre-json/chat',
      cacheTtlMs: 5000
    },
    fetch: async (url, opts) => {
      if (url.includes('sre-json')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ reply: 'SRE Disk Alert: 92% full' })
        };
      }
      const body = JSON.parse(opts.body);
      if (body.stream) {
        return createMockFetch({
          fastTokens: ['Not valid JSON at all'],
          fastLogprobs: [0.99],
          heavyText: '{}',
          fastShouldFail: false
        })(url, opts);
      }
      heavyMessages = body.messages;
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: '{"srePreserved": true}' } }] })
      };
    }
  });

  const res = await router.chatJson('Get status');
  assert.equal(res.srePreserved, true);
  assert.ok(
    heavyMessages.some(m => m.role === 'system' && m.content.includes('SRE Disk Alert: 92% full')),
    'chatJson fallback to heavyModel must preserve JeanSREGate system message'
  );
});

test('Pass 8 Hardening - L2 adapters skip mismatched-dimension/NaN vectors and sanitize MCP fields', async () => {
  const { createCentroidSemanticRouter, createContextMcpRouter } = await import('./dist/index.js');

  const centroidRouter = createCentroidSemanticRouter({
    centroids: [
      {
        archetype: 'wrong_dim',
        role: 'reasoning_deep',
        tier: 'heavy',
        label: 'Wrong Dimension',
        exemplar: 'ex1',
        embedding: [1, 0], // 2D vector (mismatches 3D query)
        threshold: 0.50
      },
      {
        archetype: 'nan_vec',
        role: 'reasoning_deep',
        tier: 'heavy',
        label: 'NaN Vector',
        exemplar: 'ex2',
        embedding: [NaN, 1, 0], // Corrupted NaN vector
        threshold: 0.50
      },
      {
        archetype: 'valid_3d',
        role: 'code',
        tier: 'specialist',
        label: 'Valid 3D Code',
        exemplar: 'ex3',
        embedding: [0, 0, 1], // Valid 3D vector
        threshold: 0.70
      }
    ],
    embed: async () => [0, 0, 1]
  });

  const cRes = await centroidRouter('Write a helper');
  assert.ok(cRes);
  assert.equal(cRes.recommendedRole, 'code');

  // MCP router rejects blank role and coerces string confidence
  const mcpBlank = createContextMcpRouter(async () => ({ recommendedRole: '   ' }));
  assert.equal(await mcpBlank('test'), null);

  const mcpStringConf = createContextMcpRouter(async () => ({
    recommended_role: ' comprehension_rc ',
    confidence: '0.91'
  }));
  const mRes = await mcpStringConf('test');
  assert.equal(mRes.recommendedRole, 'comprehension_rc');
  assert.equal(mRes.confidence, 0.91);
});

test('Pass 9 Hardening - fastModel with explicit OpenAI apiKey or provider routes to api.openai.com instead of localhost:11434', async () => {
  let fastRequestedUrl = '';
  let fastAuthHeader = '';

  const router = new CascadeRouter({
    fastModel: { model: 'gpt-4o-mini', provider: 'openai', apiKey: 'sk-fast-openai' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-heavy-openai' },
    fetch: async (url, opts) => {
      fastRequestedUrl = url;
      fastAuthHeader = opts.headers?.Authorization || '';
      return createMockFetch({
        fastTokens: ['Cloud', ' fast'],
        fastLogprobs: [0.99, 0.99],
        heavyText: 'Heavy',
        fastShouldFail: false
      })(url, opts);
    }
  });

  const res = await router.chat('Hello');
  assert.equal(res.routedTo, 'fast');
  assert.equal(res.text, 'Cloud fast');
  assert.equal(fastRequestedUrl, 'https://api.openai.com/v1/chat/completions');
  assert.equal(fastAuthHeader, 'Bearer sk-fast-openai');
});

test('Pass 9 Hardening - Gemini multi-part responses skip thought:true parts and OpenAI multi-part content arrays are joined', async () => {
  const encoder = new TextEncoder();

  // 1. Gemini 2.5 thinking parts ({ thought: true, text: '...' }) are filtered out in both fetchGemini and streamGemini
  const geminiThinkingRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gemini-2.5-pro', provider: 'gemini', apiKey: 'gem-key' },
    fetch: async (url) => {
      if (url.includes(':generateContent')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            candidates: [{
              content: {
                parts: [
                  { thought: true, text: 'Internal chain of thought that should not leak...' },
                  { text: 'First visible part ' },
                  { text: 'and second visible part' }
                ]
              }
            }]
          })
        };
      }
      if (url.includes(':streamGenerateContent')) {
        let done = false;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (!done) {
                  done = true;
                  const payload = JSON.stringify([
                    { candidates: [{ content: { parts: [{ thought: true, text: 'Secret thinking chunk' }] } }] },
                    { candidates: [{ content: { parts: [{ text: 'Visible ' }, { text: 'streamed answer' }] } }] }
                  ]);
                  return { done: false, value: encoder.encode(payload) };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }
      throw new Error(`Unexpected URL: ${url}`);
    }
  });

  const chatRes = await geminiThinkingRouter.chat('Please analyze this problem.');
  assert.equal(chatRes.text, 'First visible part and second visible part');

  const streamChunks = [];
  for await (const c of geminiThinkingRouter.stream('Please analyze this problem.')) {
    streamChunks.push(c);
  }
  assert.deepEqual(streamChunks, ['Visible streamed answer']);

  // 2. OpenAI multi-part message.content array is joined cleanly
  const multiPartOpenAiRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-test' },
    fetch: async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{
          message: {
            content: [
              { type: 'text', text: 'Part A ' },
              { type: 'text', text: 'Part B' }
            ]
          }
        }]
      })
    })
  });

  const openAiRes = await multiPartOpenAiRouter.chat('Please analyze this multi-part response.');
  assert.equal(openAiRes.text, 'Part A Part B');
});

test('Pass 10 Hardening - Stateful /g and /y regexes in customRules and customSpecialistRules do not alternate across consecutive calls', () => {
  const statefulCustomRule = /URGENT_ESCALATION/g;
  const statefulSpecialistRule = { pattern: /SPECIAL_DOMAIN_TAG/g, role: 'comprehension_rc' };

  const opts = {
    customRules: [statefulCustomRule],
    customSpecialistRules: [statefulSpecialistRule]
  };

  for (let i = 0; i < 6; i++) {
    assert.equal(
      isComplexPrompt('Hello URGENT_ESCALATION world', opts),
      true,
      `isComplexPrompt failed on iteration ${i} due to stateful /g regex lastIndex`
    );
    assert.ok(
      evaluateComplexityScore('Hello URGENT_ESCALATION world', opts) >= 0.35,
      `evaluateComplexityScore failed on iteration ${i} due to stateful /g regex lastIndex`
    );
    assert.equal(
      classifySpecialistRole('Check SPECIAL_DOMAIN_TAG now', opts),
      'comprehension_rc',
      `classifySpecialistRole failed on iteration ${i} due to stateful /g regex lastIndex`
    );
  }
});

test('Pass 10 Hardening - stream() telemetry parity (route_specialist complexityScore & route_fast model/speedPriority) and multi-part array SSE delta.content', async () => {
  const encoder = new TextEncoder();
  const events = [];

  const router = new CascadeRouter({
    fastModel: { model: 'qwen-fast' },
    backgroundModel: { model: 'qwen-bg' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-test' },
    specialistModels: {
      code: { model: 'qwen-coder' }
    },
    tokensToEvaluate: 1,
    onEvent: (ev, data) => events.push({ ev, data }),
    fetch: async () => {
      let step = 0;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              if (step === 0) {
                step++;
                const chunk = {
                  choices: [{
                    delta: {
                      content: [
                        { type: 'text', text: 'Multi-' },
                        { type: 'text', text: 'part ' }
                      ]
                    },
                    logprobs: { content: [{ token: 'Multi-', logprob: Math.log(0.99) }] }
                  }]
                };
                return { done: false, value: encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`) };
              }
              if (step === 1) {
                step++;
                const chunk = {
                  choices: [{
                    delta: {
                      content: [{ type: 'text', text: 'stream!' }]
                    },
                    logprobs: { content: [{ token: 'stream!', logprob: Math.log(0.99) }] }
                  }]
                };
                return { done: false, value: encoder.encode(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`) };
              }
              return { done: true, value: undefined };
            },
            releaseLock: () => {}
          })
        }
      };
    }
  });

  const out = [];
  for await (const tok of router.stream('Write a Python function to reverse a list', { speedPriority: 'low' })) {
    out.push(tok);
  }
  assert.equal(out.join(''), 'Multi-part stream!');

  const specEv = events.find(e => e.ev === 'route_specialist');
  assert.ok(specEv, 'stream() must emit route_specialist');
  assert.equal(specEv.data.role, 'code');
  assert.equal(specEv.data.model, 'qwen-bg');
  assert.equal(typeof specEv.data.complexityScore, 'number', 'stream() route_specialist must include numeric complexityScore');

  const fastEv = events.find(e => e.ev === 'route_fast');
  assert.ok(fastEv, 'stream() must emit route_fast');
  assert.equal(fastEv.data.reason, 'high_confidence');
  assert.equal(fastEv.data.model, 'qwen-bg');
  assert.equal(fastEv.data.speedPriority, 'low');
});

test('Pass 11 Hardening - Multi-part input Message.content arrays are extracted in classifier and CascadeRouter', async () => {
  const multiPartMessages = [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Write a Python function ' },
        { type: 'text', text: 'to implement binary search.' }
      ]
    }
  ];

  assert.equal(classifySpecialistRole(multiPartMessages), 'code');

  const complexMultiPart = [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Please analyze and architect ' },
        { type: 'text', text: 'this distributed microservice system.' }
      ]
    }
  ];
  assert.equal(isComplexPrompt(complexMultiPart), true);

  let sentMessages = null;
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-test' },
    fetch: async (url, opts) => {
      sentMessages = JSON.parse(opts.body).messages;
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Handled multi-part input' } }] })
      };
    }
  });

  const res = await router.chat(complexMultiPart);
  assert.equal(res.routedTo, 'heavy');
  assert.equal(sentMessages[0].content, 'Please analyze and architect this distributed microservice system.');
});

test('Pass 11 Hardening - chatJson rejects primitive JSON ("null", "true", numbers) and preserves metrics on retry after transient heavy error', async () => {
  let heavyAttempt = 0;

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://localhost:11434/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-test' },
    fetch: async (url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.stream) {
        // Fast model returns literal JSON primitive "null" with high confidence
        const encoder = new TextEncoder();
        let done = false;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (!done) {
                  done = true;
                  const chunk = {
                    choices: [{
                      delta: { content: 'null' },
                      logprobs: { content: [{ token: 'null', logprob: Math.log(0.99) }] }
                    }]
                  };
                  return { done: false, value: encoder.encode(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`) };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }

      // Heavy model: attempt 0 fails with HTTP 503, attempt 1 succeeds with valid JSON object
      heavyAttempt++;
      if (heavyAttempt === 1) {
        return { ok: false, status: 503 };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          choices: [{ message: { content: '{"recoveredAfterPrimitiveAnd503": true}' } }],
          usage: { prompt_tokens: 12, completion_tokens: 8, total_tokens: 20 }
        })
      };
    }
  });

  const parsed = await router.chatJson('Return user profile JSON', { maxJsonRetries: 2 });
  assert.equal(parsed.recoveredAfterPrimitiveAnd503, true);
  assert.equal(parsed._routedTo, 'heavy');
  assert.equal(heavyAttempt, 2);

  const metrics = router.getMetrics();
  assert.equal(metrics.totalRequests, 1, 'Should record 1 total request after retry recovery');
  assert.equal(metrics.fastRequests, 0, 'Fast request should be reclassified to heavy');
  assert.equal(metrics.heavyRequests, 1, 'Should record 1 heavy request');
  assert.equal(metrics.cascadedRequests, 1, 'Should record 1 cascaded request');
});

test('Pass 11 Hardening - vLLM token_logprobs with null entries are clamped to 0 and trigger cascade', async () => {
  const encoder = new TextEncoder();

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-test' },
    tokensToEvaluate: 2,
    cascadeThreshold: 0.85,
    fetch: async (url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.stream) {
        let done = false;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (!done) {
                  done = true;
                  const chunk = {
                    choices: [{
                      delta: { content: 'Uncertain answer' },
                      logprobs: { token_logprobs: [null, null] }
                    }]
                  };
                  return { done: false, value: encoder.encode(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`) };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: 'Heavy fallback from null token_logprobs' } }] })
      };
    }
  });

  const res = await router.chat('Hello');
  assert.equal(res.routedTo, 'heavy');
  assert.equal(res.aborted, true);
  assert.equal(res.text, 'Heavy fallback from null token_logprobs');
});

test('Pass 12 Hardening - Mid-stream caller AbortSignal during reader.read() aborts immediately without falling back to heavyModel', async () => {
  const encoder = new TextEncoder();
  let heavyCalled = false;
  const controller = new AbortController();

  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-test' },
    fetch: async (url, opts) => {
      const body = JSON.parse(opts.body);
      if (body.stream) {
        let step = 0;
        return {
          ok: true,
          status: 200,
          body: {
            getReader: () => ({
              read: async () => {
                if (step === 0) {
                  step++;
                  // Caller aborts while reader.read() is in flight on custom stream that does not auto-reject
                  controller.abort(new Error('MidStreamCallerAbort'));
                  const chunk = {
                    choices: [{
                      delta: { content: 'First' },
                      logprobs: { content: [{ token: 'First', logprob: Math.log(0.10) }] }
                    }]
                  };
                  return { done: false, value: encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`) };
                }
                return { done: true, value: undefined };
              },
              releaseLock: () => {}
            })
          }
        };
      }
      heavyCalled = true;
      return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'Should not run' } }] }) };
    }
  });

  await assert.rejects(
    () => router.chat('Hello', { signal: controller.signal }),
    /MidStreamCallerAbort/
  );
  assert.equal(heavyCalled, false, 'heavyModel must not be called when caller aborts mid-stream');
});

test('Pass 12 Hardening - Gemini provider honors custom url and custom headers in fetchGemini and streamGemini', async () => {
  const { createCentroidSemanticRouter, createContextMcpRouter } = await import('./dist/index.js');
  const encoder = new TextEncoder();
  const captured = [];

  const geminiGatewayRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: {
      model: 'gemini-2.5-pro',
      provider: 'gemini',
      apiKey: 'gem-proxy-key',
      url: 'https://gateway.krusch.dev/v1beta/models/',
      headers: { 'X-Custom-Gateway': 'krusch-edge' }
    },
    fetch: async (url, opts) => {
      captured.push({ url, headers: opts.headers });
      if (url.includes(':generateContent')) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            candidates: [{ content: { parts: [{ text: 'Gateway Gemini chat' }] } }]
          })
        };
      }
      let done = false;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              if (!done) {
                done = true;
                return {
                  done: false,
                  value: encoder.encode('[{"candidates":[{"content":{"parts":[{"text":"Gateway Gemini stream"}]}}]}]')
                };
              }
              return { done: true, value: undefined };
            },
            releaseLock: () => {}
          })
        }
      };
    }
  });

  const chatRes = await geminiGatewayRouter.chat('Please analyze this architecture.');
  assert.equal(chatRes.text, 'Gateway Gemini chat');
  assert.equal(captured[0].url, 'https://gateway.krusch.dev/v1beta/models/gemini-2.5-pro:generateContent?key=gem-proxy-key');
  assert.equal(captured[0].headers['X-Custom-Gateway'], 'krusch-edge');

  const streamChunks = [];
  for await (const c of geminiGatewayRouter.stream('Please analyze this architecture.')) {
    streamChunks.push(c);
  }
  assert.deepEqual(streamChunks, ['Gateway Gemini stream']);
  assert.equal(captured[1].url, 'https://gateway.krusch.dev/v1beta/models/gemini-2.5-pro:streamGenerateContent?key=gem-proxy-key');
  assert.equal(captured[1].headers['X-Custom-Gateway'], 'krusch-edge');

  // Verify L2 OpenRouter custom embedModel and MCP isError / uppercase targetTier handling
  let openrouterModelUsed = null;
  const l2OpenRouter = createCentroidSemanticRouter({
    openrouterApiKey: 'or-key',
    embedModel: 'openai/text-embedding-3-small',
    centroids: [{
      archetype: 'custom_code',
      role: 'code',
      tier: 'specialist',
      label: 'Code',
      exemplar: 'ex',
      embedding: [1, 0],
      threshold: 0.5
    }],
    fetch: async (url, opts) => {
      openrouterModelUsed = JSON.parse(opts.body).model;
      return {
        ok: true,
        status: 200,
        json: async () => ({ data: [{ embedding: [1, 0] }] })
      };
    }
  });

  const l2Res = await l2OpenRouter('test prompt');
  assert.equal(openrouterModelUsed, 'openai/text-embedding-3-small');
  assert.equal(l2Res.recommendedRole, 'code');

  const mcpErrorRouter = createContextMcpRouter(async () => ({
    isError: true,
    content: [{ type: 'text', text: JSON.stringify({ recommendedRole: 'code' }) }]
  }));
  assert.equal(await mcpErrorRouter('test'), null, 'MCP isError: true must return null');

  const mcpUpperTierRouter = createContextMcpRouter(async () => ({
    recommended_role: 'factual_stem',
    target_tier: 'FRONTIER',
    confidence: 0.92
  }));
  const upperTierRes = await mcpUpperTierRouter('test');
  assert.equal(upperTierRes.targetTier, 'frontier');
});

test('Pass 13 Hardening - chatJson isolates schema keywords from routing, avoids duplicate SRE fetch on fallback, and supports jsonc/json5 fences', async () => {
  const { createMultiSpecialistRouter } = await import('./dist/index.js');

  // 1. Schema containing "analyze", "evaluate", "architect", "function" must NOT falsely escalate a simple prompt to heavyModel
  const router = new CascadeRouter({
    fastModel: { model: 'qwen2.5', url: 'http://fast-url/v1/chat/completions' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-test' },
    fetch: createMockFetch({
      fastTokens: ['```ts\nconst x = 1;\n```\n```jsonc\n{"answer": 4}\n```'],
      fastLogprobs: [0.99],
      heavyText: '{"answer": 99}',
      fastShouldFail: false
    })
  });

  const res = await router.chatJson('What is 2 + 2?', {
    schema: {
      type: 'object',
      description: 'Analyze, evaluate, and architect the function class output',
      properties: { answer: { type: 'number' } }
    }
  });
  assert.equal(res._routedTo, 'fast', 'Complex verbs in chatJson schema must not escalate simple prompt to heavy');
  assert.equal(res.answer, 4, 'parseJsonSafe must extract ```jsonc fence over preceding ```ts fence');

  // 2. When chatJson falls back from code specialist to heavyModel with cacheTtlMs: 0, JeanSREGate must NOT be fetched a second time
  let fallbackSreFetches = 0;
  let heavyMessages = [];
  const fallbackRouter = createMultiSpecialistRouter({
    openrouterApiKey: 'sk-test',
    jeanSREGate: {
      enabled: true,
      sreUrl: 'http://sre-nocache/chat',
      cacheTtlMs: 0
    },
    fetch: async (url, opts) => {
      if (url.includes('sre-nocache')) {
        fallbackSreFetches++;
        return {
          ok: true,
          status: 200,
          json: async () => ({ reply: 'SRE Warning' })
        };
      }
      const body = JSON.parse(opts.body);
      if (body.model === 'Qwen/Qwen3-Coder-Next') {
        return {
          ok: true,
          status: 200,
          json: async () => ({ choices: [{ message: { content: 'Not valid JSON' } }] })
        };
      }
      heavyMessages = body.messages;
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: '```json5\n{"ok": true}\n```' } }] })
      };
    }
  });

  const fallbackRes = await fallbackRouter.chatJson('Write a Python function to return status JSON');
  assert.equal(fallbackRes.ok, true);
  assert.equal(fallbackRes._routedTo, 'heavy');
  assert.equal(fallbackSreFetches, 1, 'chatJson fallback must reuse prepared messages without duplicate SRE HTTP call');
  assert.ok(heavyMessages.some(m => m.role === 'system' && m.content.includes('SRE Warning')));
});

test('Pass 13 Hardening - Fast model SSE usage and Gemini fastModel usageMetadata are preserved, and L2 duplicate archetypes/roles are validated', async () => {
  const { createCentroidSemanticRouter, createContextMcpRouter, createMultiSpecialistRouter } = await import('./dist/index.js');
  const encoder = new TextEncoder();

  // 1. Fast model SSE stream with final usage chunk preserves exact token counts in res.usage and getMetrics()
  const sseUsageRouter = new CascadeRouter({
    fastModel: { model: 'qwen2.5' },
    heavyModel: { model: 'gpt-4o', apiKey: 'sk-test' },
    fetch: async () => {
      let step = 0;
      return {
        ok: true,
        status: 200,
        body: {
          getReader: () => ({
            read: async () => {
              if (step === 0) {
                step++;
                const chunk = {
                  choices: [{ delta: { content: 'Exact usage' }, logprobs: { content: [{ token: 'Exact', logprob: Math.log(0.99) }] } }],
                  usage: { prompt_tokens: 42, completion_tokens: 17, total_tokens: 59 }
                };
                return { done: false, value: encoder.encode(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`) };
              }
              return { done: true, value: undefined };
            },
            releaseLock: () => {}
          })
        }
      };
    }
  });

  const sseRes = await sseUsageRouter.chat('Hello');
  assert.equal(sseRes.usage.promptTokens, 42);
  assert.equal(sseRes.usage.completionTokens, 17);
  assert.equal(sseUsageRouter.getMetrics().totalPromptTokens, 42);
  assert.equal(sseUsageRouter.getMetrics().totalCompletionTokens, 17);

  // 2. Duplicate archetype names in custom centroids warm once without infinite re-warming
  let embedCalls = 0;
  const dupCentroidRouter = createCentroidSemanticRouter({
    centroids: [
      { archetype: 'dup_arch', role: 'code', tier: 'specialist', label: 'Code 1', exemplar: 'ex1', threshold: 0.5 },
      { archetype: 'dup_arch', role: 'factual_stem', tier: 'specialist', label: 'Stem 2', exemplar: 'ex2', threshold: 0.5 }
    ],
    embed: async (text) => {
      embedCalls++;
      return text === 'ex1' ? [1, 0] : [0, 1];
    }
  });

  await dupCentroidRouter('prompt1');
  assert.equal(embedCalls, 3, 'First call embeds 2 centroids + 1 prompt');
  await dupCentroidRouter('prompt1');
  assert.equal(embedCalls, 3, 'Second cached call must not re-run warmCentroids when duplicate archetype names exist');

  // 3. Invalid L2 role is ignored while uppercase valid role is normalized
  const mcpInvalidRole = createContextMcpRouter(async () => ({
    recommendedRole: 'hallucinated_role',
    confidence: 0.99
  }));
  assert.equal(await mcpInvalidRole('test'), null, 'Invalid SpecialistRole from MCP must return null');

  const mcpUpperRole = createContextMcpRouter(async () => ({
    recommendedRole: 'CODE',
    confidence: 0.91
  }));
  const normalizedRoleRes = await mcpUpperRole('test');
  assert.equal(normalizedRoleRes.recommendedRole, 'code');

  const multiRouter = createMultiSpecialistRouter({
    openrouterApiKey: 'sk-test',
    l2Router: async () => ({ recommendedRole: 'not_a_valid_role' }),
    fetch: async () => ({
      ok: true,
      status: 200,
      json: async () => ({ choices: [{ message: { content: 'Fallback to default factual_stem' } }] })
    })
  });
  const multiRes = await multiRouter.chat('What should we do next quarter?');
  assert.equal(multiRes.routedTo, 'factual_stem', 'Invalid L2 role must safely fall back to default role');
});

