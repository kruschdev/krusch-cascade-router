import test from 'node:test';
import assert from 'node:assert/strict';
import { isComplexPrompt } from './dist/index.js';

test('isComplexPrompt - Simple Prompts', () => {
    assert.equal(isComplexPrompt('Hello, how are you?'), false);
    assert.equal(isComplexPrompt('What is the capital of France?'), false);
});

test('isComplexPrompt - Complex Verbs', () => {
    assert.equal(isComplexPrompt('Please analyze this text and synthesize a response.'), true);
    assert.equal(isComplexPrompt('I need you to architect a solution.'), true);
});

test('isComplexPrompt - Code Blocks', () => {
    assert.equal(isComplexPrompt('Here is some code:\n```javascript\nconst x = 1;\n```'), true);
});

test('isComplexPrompt - JSON / XML', () => {
    assert.equal(isComplexPrompt('Parse this <xml>data</xml>'), true);
    assert.equal(isComplexPrompt('Here is the {"key": "value"} structure'), true);
});

test('isComplexPrompt - Length', () => {
    const longString = 'A'.repeat(2500);
    assert.equal(isComplexPrompt(longString), true);
});

test('isComplexPrompt - Custom Rules', () => {
    assert.equal(isComplexPrompt('Please do something MAGIC.', { customRules: [/MAGIC/] }), true);
    assert.equal(isComplexPrompt('Please do something NORMAL.', { customRules: [/MAGIC/] }), false);
});
