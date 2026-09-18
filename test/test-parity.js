import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { classifySpecialistRole } from '../dist/index.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const fixturesPath = path.join(__dirname, 'fixtures', 'routing-spec.json');
const fixtures = JSON.parse(fs.readFileSync(fixturesPath, 'utf8'));

// Default model mapping for specialist roles
const roleToModel = {
  comprehension_rc: 'qwen/qwen3-235b-a22b-2507',
  reasoning_deep: 'deepseek/deepseek-v4-pro',
  games_spatial: 'Qwen/Qwen3-Coder-Next',
  code: 'Qwen/Qwen3-Coder-Next',
  general_fast: 'google/gemini-3.1-flash-lite',
  factual_stem: 'deepseek/deepseek-v4-flash'
};

test('Routing Spec Parity - TypeScript Classifier against Shared Fixtures', () => {
  let passed = 0;
  for (const item of fixtures) {
    const role = classifySpecialistRole(item.query);
    const predictedModel = roleToModel[role];

    assert.equal(
      role, 
      item.domain, 
      `Prompt [${item.query}] expected role [${item.domain}] but got [${role}]`
    );
    assert.equal(
      predictedModel, 
      item.expected_model, 
      `Prompt [${item.query}] expected model [${item.expected_model}] but got [${predictedModel}]`
    );
    passed++;
  }
  assert.equal(passed, fixtures.length);
});
