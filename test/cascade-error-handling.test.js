import test from 'node:test';
import assert from 'node:assert/strict';
import { CascadeRouter, parseJsonFragment } from '../dist/index.js';

test('BUG-02 & TEST-01: parseJsonFragment logs warning on malformed balanced JSON fragment and recovers on next valid fragment', () => {
  const warnings = [];
  const origWarn = console.warn;
  console.warn = (...args) => warnings.push(args);
  try {
    const res = parseJsonFragment('prefix {invalid json} middle {"ok":true}', '{', '}');
    assert.deepEqual(res, { ok: true, value: { ok: true } });
    assert.equal(warnings.length, 1);
    assert.equal(warnings[0][0], 'Failed JSON fragment parse:');
  } finally {
    console.warn = origWarn;
  }
});

test('BUG-01 & TEST-01: stream cancellation errors during reader.cancel() are logged via console.error', async () => {
  const errors = [];
  const origErr = console.error;
  console.error = (...args) => errors.push(args);

  const mockFetch = async () => ({
    ok: true,
    status: 200,
    body: {
      getReader() {
        let step = 0;
        return {
          async read() {
            step++;
            if (step === 1) {
              const chunk = 'data: {"choices":[{"delta":{"content":"hello"},"logprobs":{"content":[{"token":"hello","logprob":-0.01}]}}]}\n\n';
              return { done: false, value: new TextEncoder().encode(chunk) };
            }
            return { done: true, value: undefined };
          },
          async cancel() {
            throw new Error('Synthetic stream cancel failure');
          },
          releaseLock() {}
        };
      }
    }
  });

  try {
    const router = new CascadeRouter({
      fastModel: { model: 'qwen2.5-coder:7b', url: 'http://127.0.0.1:11434/v1/chat/completions' },
      heavyModel: { model: 'gpt-oss:120b', url: 'http://127.0.0.1:11435/v1/chat/completions' },
      fetch: mockFetch
    });

    const out = await router.chat([{ role: 'user', content: 'hi' }]);
    assert.equal(out.text, 'hello');
    assert.equal(errors.some(e => e[0] === 'CascadeRouter error:' && e[1]?.message === 'Synthetic stream cancel failure'), true);

    errors.length = 0;
    let streamed = '';
    for await (const chunk of router.stream([{ role: 'user', content: 'hi' }])) {
      streamed += chunk;
    }
    assert.equal(streamed, 'hello');
    assert.equal(errors.some(e => e[0] === 'CascadeRouter error:' && e[1]?.message === 'Synthetic stream cancel failure'), true);
  } finally {
    console.error = origErr;
  }
});
