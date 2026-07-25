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


