/**
 * scratch/demo-streaming.js
 * 
 * Interactive demonstration showcasing the new speculative streaming cascade
 * and real-time logprob confidence metrics.
 * Runs simulated edge and cloud model streams to visually demonstrate the routing mechanics.
 */

import { CascadeRouter } from '../dist/index.js';

// ANSI Colors for premium visual presentation
const colors = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  red: '\x1b[31m',
  gray: '\x1b[90m'
};

// Sleep helper for visual pacing
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Creates a mock fetch engine that simulates chunked SSE streams for both edge and cloud models.
 */
function createDemoMockFetch({ scenario }) {
  return async (url, opts) => {
    const body = JSON.parse(opts.body);
    const encoder = new TextEncoder();

    if (body.model === 'qwen-local-edge') {
      let tokens, logprobs;
      
      if (scenario === 'high_confidence') {
        tokens = ['Analyzing', ' system', ' performance', ' metrics', '...', ' All', ' operations', ' nominal.'];
        logprobs = [0.96, 0.94, 0.93, 0.95, 0.91, 0.92, 0.95, 0.97];
      } else {
        // Low confidence sequence
        tokens = ['Um', '...', 'perhaps', ' you', ' should', ' ask', ' someone', ' else'];
        logprobs = [0.45, 0.35, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05];
      }

      // Build chunked SSE payload
      const chunks = tokens.map((token, i) => {
        const chunk = {
          choices: [{
            delta: { content: token },
            logprobs: {
              content: [{
                logprob: Math.log(logprobs[i])
              }]
            }
          }]
        };
        return `data: ${JSON.stringify(chunk)}\n\n`;
      });
      chunks.push('data: [DONE]\n\n');

      const fullSSE = chunks.join('');
      const encoded = encoder.encode(fullSSE);

      let read = false;
      const body_stream = {
        getReader: () => ({
          read: async () => {
            if (!read) {
              read = true;
              await sleep(100); // Simulate network latency
              return { done: false, value: encoded };
            }
            return { done: true, value: undefined };
          },
          releaseLock: () => {}
        })
      };

      return { ok: true, status: 200, body: body_stream };

    } else if (body.model === 'gpt-cloud-heavy') {
      const tokens = ['[Cloud Heavy]', ' Executive', ' Summary:', ' The', ' requested', ' task', ' requires', ' comprehensive', ' multi-layer', ' analysis', ' which', ' has', ' been', ' successfully', ' compiled.'];
      const chunks = tokens.map(token => {
        const chunk = {
          choices: [{ delta: { content: token } }]
        };
        return `data: ${JSON.stringify(chunk)}\n\n`;
      });
      chunks.push('data: [DONE]\n\n');

      const fullSSE = chunks.join('');
      const encoded = encoder.encode(fullSSE);

      let read = false;
      const body_stream = {
        getReader: () => ({
          read: async () => {
            if (!read) {
              read = true;
              await sleep(250); // Simulate heavy cloud latency
              return { done: false, value: encoded };
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
}

async function runDemo(scenarioName, scenarioType) {
  console.log(`\n${colors.bold}${colors.cyan}======================================================================${colors.reset}`);
  console.log(`${colors.bold}${colors.yellow}🎬 SCENARIO: ${scenarioName}${colors.reset}`);
  console.log(`${colors.bold}${colors.cyan}======================================================================${colors.reset}\n`);

  const router = new CascadeRouter({
    fastModel: { model: 'qwen-local-edge' },
    heavyModel: { model: 'gpt-cloud-heavy', apiKey: 'demo-key' },
    cascadeThreshold: 0.85,
    tokensToEvaluate: 4,
    fetch: createDemoMockFetch({ scenario: scenarioType }),
    onEvent: (event, meta) => {
      if (event === 'cascade_triggered') {
        console.log(`\n${colors.bold}${colors.red}⚠️  [Telemetry] CASCADE TRIGGERED!${colors.reset}`);
        console.log(`${colors.gray}   Reason: Avg Logprob (${(meta.avgProb * 100).toFixed(1)}%) dipped below Threshold (${(meta.threshold * 100).toFixed(0)}%) at token #${meta.tokenCount}.${colors.reset}\n`);
      } else if (event === 'route_fast') {
        console.log(`\n\n${colors.bold}${colors.green}✅ [Telemetry] ROUTED TO FAST MODEL (Confidence satisfied!)${colors.reset}\n`);
      } else if (event === 'route_heavy') {
        console.log(`${colors.bold}${colors.magenta}🔄 [Telemetry] FALLBACK ROUTING TO HEAVY MODEL (${meta.reason})${colors.reset}\n`);
      }
    }
  });

  console.log(`${colors.bold}Prompt:${colors.reset} "Run system check..."`);
  console.log(`${colors.bold}Config:${colors.reset} cascadeThreshold = ${colors.bold}0.85${colors.reset}, tokensToEvaluate = ${colors.bold}4${colors.reset}\n`);
  
  process.stdout.write(`${colors.bold}Streaming Response:${colors.reset} `);

  // Buffer state tracker to print visual cues
  let tokenCount = 0;
  for await (const chunk of router.stream('Run system check...')) {
    tokenCount++;
    if (tokenCount <= 4 && scenarioType === 'high_confidence') {
      // Show buffering visualization in real-time
      process.stdout.write(`${colors.blue}[Buf:${chunk}]${colors.reset} `);
      await sleep(150);
    } else {
      process.stdout.write(`${colors.green}${chunk}${colors.reset}`);
      await sleep(80);
    }
  }
  console.log('\n');
}

async function main() {
  console.log(`\n${colors.bold}${colors.magenta}✨ KRUSCH CASCADE ROUTER SPECULATIVE STREAMING DEMO ✨${colors.reset}`);
  
  await runDemo('HIGH CONFIDENCE FAST-PATH (No Cascade)', 'high_confidence');
  await sleep(1000);
  await runDemo('LOW CONFIDENCE SPECULATIVE CASCADE (Silent Abort & Fallback)', 'low_confidence');
  
  console.log(`${colors.bold}${colors.cyan}======================================================================${colors.reset}`);
  console.log(`${colors.bold}${colors.green}🎉 Demonstration Completed Successfully!${colors.reset}`);
  console.log(`${colors.bold}${colors.cyan}======================================================================${colors.reset}\n`);
}

main().catch(console.error);
