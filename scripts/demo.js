#!/usr/bin/env node

/**
 * @file scripts/demo.js
 * @description Interactive routing inspector demonstrating dual-stage L1/L2 classification in CPU microseconds.
 */

import { classifyPreRoute, classifySpecialistRole } from '../dist/index.js';

const prompt = process.argv.slice(2).join(' ') || 'Write an async Rust parser for PCAP ethernet frames';

console.log(`\n======================================================`);
console.log(`🧭 Krusch Cascade Router: Dual-Stage Routing Inspector`);
console.log(`======================================================\n`);
console.log(`Input Prompt: "${prompt}"\n`);

const t0 = performance.now();
const preRoute = classifyPreRoute(prompt);
const t1 = performance.now();
const elapsedUs = ((t1 - t0) * 1000).toFixed(2);

console.log(`⚡ Stage 1: L1 In-Memory Pre-Router (<15µs CPU / $0.00 Cost)`);
console.log(`   • Fast-Path Match:    ${preRoute.isFastPath ? '🚀 FAST-PATH HIT (Zero-Token Bypass)' : '🔄 L1 MISS -> DELEGATE TO L2 NEURAL ROUTER'}`);
console.log(`   • Suggested Action:   ${preRoute.suggestedAction}`);
console.log(`   • Specialist Role:    ${preRoute.role || '(None - Open-world unstructured query)'}`);
console.log(`   • Confidence:         ${preRoute.confidence}`);
console.log(`   • Complexity Score:   ${preRoute.complexityScore}`);
console.log(`   • Evaluation Latency: ${elapsedUs} µs`);

if (!preRoute.isFastPath) {
    console.log(`\n🧠 Stage 2: L2 Neural Semantic Escalation`);
    console.log(`   • Delegating open-world unstructured prompt to neural semantic centroids or secondary router.`);
    console.log(`   • Fallback specialist role: ${classifySpecialistRole(prompt)}`);
}

console.log(`\n======================================================\n`);
