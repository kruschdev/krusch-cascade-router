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

test('pruneText and isComplexPrompt - Pre-Routing Pruning', async () => {
    const { pruneText } = await import('./dist/index.js');
    const bloated = "Hey, could you please tell me what the status is? Thanks in advance!";
    const cleaned = pruneText(bloated);
    assert.equal(cleaned, "what the status is?");

    // borderText length is > 2000 chars with filler (2021 chars), but < 2000 once pruned (1980 chars)
    const borderText = "Hey, could you please " + "word ".repeat(396) + " Thanks in advance!";
    assert.equal(isComplexPrompt(borderText, { lengthThreshold: 2000, prunePreRouting: false }), true);
    assert.equal(isComplexPrompt(borderText, { lengthThreshold: 2000, prunePreRouting: true }), false);
});

test('isComplexPrompt - casual "test" queries do not trigger complexity', () => {
    assert.equal(isComplexPrompt('Can you test if this works?'), false);
    assert.equal(isComplexPrompt('Please run a quick test on the server.'), false);
});

test('classifySpecialistRole - language name disambiguation', async () => {
    const { classifySpecialistRole } = await import('./dist/index.js');
    // Trivia asking for capital
    assert.equal(classifySpecialistRole('What is the German capital?'), 'general_fast');
    // Real translation query
    assert.equal(classifySpecialistRole('Translate this text into German: Good morning'), 'general_fast');
    // Factual history query containing language name - must NOT be hijacked into translation
    assert.equal(classifySpecialistRole('Explain the economic impact of the German reunification in 1990.'), 'factual_stem');
    assert.equal(classifySpecialistRole('Analyze the themes of Russian literature in the 19th century.'), 'factual_stem');
});

test('classifySpecialistRole - supports customSpecialistRules', async () => {
    const { classifySpecialistRole } = await import('./dist/index.js');
    const customRules = [
        { role: 'reasoning_deep', pattern: /\b(?:legal compliance|gdpr audit)\b/i },
        { role: 'code', pattern: /\b(?:terraform plan|ansible playbook)\b/i }
    ];

    // Matches custom rule 1
    assert.equal(classifySpecialistRole('Conduct a legal compliance review for our policy', { customSpecialistRules: customRules }), 'reasoning_deep');
    // Matches custom rule 2
    assert.equal(classifySpecialistRole('Review this terraform plan for VPC peering', { customSpecialistRules: customRules }), 'code');
    // Without custom rules, "legal compliance" defaults to general_fast or factual_stem
    assert.notEqual(classifySpecialistRole('Conduct a legal compliance review for our policy'), 'reasoning_deep');
});

test('createMultiSpecialistRouter - supports customModels (string & ModelConfig), classifier, and backgroundModel', async () => {
    const { createMultiSpecialistRouter } = await import('./dist/index.js');
    const router = createMultiSpecialistRouter({
        openrouterApiKey: 'test-key',
        customModels: {
            code: 'anthropic/claude-3.7-sonnet',
            factual_stem: {
                model: 'meta-llama/llama-3.3-70b-instruct',
                provider: 'openrouter',
                costPerMillionInputTokens: 0.12,
                costPerMillionOutputTokens: 0.30
            }
        },
        backgroundModel: {
            model: 'qwen2.5:3b',
            url: 'http://localhost:11434/v1/chat/completions',
            costPerMillionInputTokens: 0,
            costPerMillionOutputTokens: 0
        },
        classifier: {
            customSpecialistRules: [
                { role: 'reasoning_deep', pattern: /\bquantum mechanics\b/i }
            ]
        }
    });

    assert.equal(router.config.specialistModels.code.model, 'anthropic/claude-3.7-sonnet');
    assert.equal(router.config.specialistModels.factual_stem.model, 'meta-llama/llama-3.3-70b-instruct');
    assert.equal(router.config.specialistModels.factual_stem.costPerMillionInputTokens, 0.12);
    assert.equal(router.config.backgroundModel.model, 'qwen2.5:3b');
    assert.equal(router.config.classifier.customSpecialistRules.length, 1);
});

