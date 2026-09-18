import test from 'node:test';
import assert from 'node:assert/strict';
import { CascadeRouter } from './dist/index.js';

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
    classifySpecialistRole('Generate an executable Python function that takes stdin as input and prints the result.'),
    'code'
  );

  // 4. Reading Comprehension & Truth Verification (SuperGLUE-RC) -> qwen3-235b
  assert.equal(
    classifySpecialistRole('Your task is to evaluate if the "Provided Answer" is a correct response to the "Question" based on the "Paragraph".\nQuestion: ...\nProvided Answer: ...'),
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

