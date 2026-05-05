# krusch-cascade-router

A lightweight, framework-agnostic npm package designed for agentic developers building with local AI. 

**"LLM routing an LLM is a trap."**
Using a massive third LLM to decide which LLM to route a query to adds severe TTFT (Time To First Token) latency and API costs. `krusch-cascade-router` solves this by combining a fast predictive heuristic classifier (<50ms latency) with a reactive logprob-based speculative cascade.

## How it works

1. **Predictive Classifier**: Instantly evaluates the prompt's complexity via string heuristics (length, code blocks, complex cognitive verbs). If complex, routes directly to the heavy cloud model.
2. **Speculative Cascade**: If simple, streams the fast local edge model. It buffers and inspects the logprobs of the first N tokens. If the confidence (probability) dips below the threshold, it silently aborts the stream and falls back to the heavy cloud model.

## Installation

```bash
npm install krusch-cascade-router
```

## Usage

```javascript
import { CascadeRouter } from 'krusch-cascade-router';

const router = new CascadeRouter({
  fastModel: { url: 'http://localhost:11434/v1/chat/completions', model: 'qwen2.5' },
  heavyModel: { apiKey: process.env.GEMINI_API_KEY, model: 'gemini-2.5-pro', provider: 'gemini' },
  cascadeThreshold: 0.85, // Abort if average probability of first 5 tokens is < 85%
  tokensToEvaluate: 5
});

const response = await router.chat("Write a complex architectural plan...");
console.log(`Routed to: ${response.routedTo}`);
console.log(response.text);
```
