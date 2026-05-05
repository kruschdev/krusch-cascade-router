import { isComplexPrompt, Message, ClassifierOptions } from './classifier.js';

export interface ModelConfig {
  url?: string;
  apiKey?: string;
  model: string;
  provider?: 'openai' | 'gemini';
}

export type TelemetryEvent = 'route_fast' | 'route_heavy' | 'cascade_triggered';

export interface RouterConfig {
  fastModel: ModelConfig;
  heavyModel: ModelConfig;
  cascadeThreshold?: number; // Default 0.85 (Linear probability)
  tokensToEvaluate?: number; // Default 5
  classifier?: ClassifierOptions;
  fetch?: typeof fetch;
  onEvent?: (event: TelemetryEvent, metadata?: Record<string, any>) => void;
}

export interface CascadeResponse {
  text: string;
  routedTo: 'fast' | 'heavy';
  aborted: boolean;
}

export interface ChatOptions {
  signal?: AbortSignal;
}

export class CascadeRouter {
  private config: RouterConfig;
  private fetchFn: typeof fetch;

  constructor(config: RouterConfig) {
    this.config = {
      ...config,
      cascadeThreshold: config.cascadeThreshold ?? 0.85,
      tokensToEvaluate: config.tokensToEvaluate ?? 5
    };
    this.fetchFn = config.fetch ?? (typeof globalThis !== 'undefined' ? globalThis.fetch : fetch);
    if (!this.fetchFn) {
      throw new Error('A global fetch API is required, or a custom fetch implementation must be provided in RouterConfig.');
    }
  }

  /**
   * Complete a chat request, routing automatically.
   */
  async chat(messages: Message[] | string, systemPrompt?: string, options?: ChatOptions): Promise<CascadeResponse> {
    const formattedMessages = this.formatMessages(messages, systemPrompt);

    // 1. Predictive Classifier
    const isComplex = isComplexPrompt(formattedMessages, this.config.classifier);

    if (isComplex) {
      // Bypass fast model entirely
      this.config.onEvent?.('route_heavy', { reason: 'classifier_heuristic' });
      const text = await this.fetchHeavyModel(formattedMessages, options);
      return { text, routedTo: 'heavy', aborted: false };
    }

    // 2. Try fast model with Speculative Cascade
    try {
      const fastResult = await this.streamAndEvaluateFastModel(formattedMessages, options);
      if (fastResult.aborted) {
        this.config.onEvent?.('route_heavy', { reason: 'cascade_fallback' });
        const heavyText = await this.fetchHeavyModel(formattedMessages, options);
        return { text: heavyText, routedTo: 'heavy', aborted: true };
      }
      this.config.onEvent?.('route_fast', { reason: 'high_confidence' });
      return { text: fastResult.text, routedTo: 'fast', aborted: false };
    } catch (err) {
      this.config.onEvent?.('route_heavy', { reason: 'fast_model_error', error: (err as Error).message });
      const heavyText = await this.fetchHeavyModel(formattedMessages, options);
      return { text: heavyText, routedTo: 'heavy', aborted: true };
    }
  }

  private formatMessages(messages: Message[] | string, systemPrompt?: string): Message[] {
    const msgs: Message[] = [];
    if (systemPrompt) {
      msgs.push({ role: 'system', content: systemPrompt });
    }
    if (typeof messages === 'string') {
      msgs.push({ role: 'user', content: messages });
    } else {
      msgs.push(...messages);
    }
    return msgs;
  }

  /**
   * Streams the fast model, buffering the first N tokens to check logprobs.
   * If confidence is lower than threshold, aborts and returns { aborted: true }.
   */
  private async streamAndEvaluateFastModel(messages: Message[], options?: ChatOptions): Promise<{ text: string, aborted: boolean }> {
    const { fastModel } = this.config;
    const url = fastModel.url || 'http://localhost:11434/v1/chat/completions';
    
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (fastModel.apiKey) headers['Authorization'] = `Bearer ${fastModel.apiKey}`;

    const controller = new AbortController();
    
    // Link external abort signal to our internal controller
    if (options?.signal) {
      options.signal.addEventListener('abort', () => controller.abort(), { once: true });
      if (options.signal.aborted) controller.abort();
    }

    const response = await this.fetchFn(url, {
      method: 'POST',
      headers,
      signal: controller.signal,
      body: JSON.stringify({
        model: fastModel.model,
        messages,
        stream: true,
        logprobs: true // Request logprobs (OpenAI format)
      })
    });

    if (!response.ok) {
      throw new Error(`Fast model HTTP ${response.status}`);
    }

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    let fullText = '';
    let tokenCount = 0;
    let accumulatedProb = 0;

    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') continue;
        if (!trimmed.startsWith('data: ')) continue;

        try {
          const data = JSON.parse(trimmed.slice(6));
          const choice = data.choices?.[0];
          const delta = choice?.delta?.content || '';
          
          if (delta) fullText += delta;

          // Evaluate logprobs if present
          const logprobsObj = choice?.logprobs?.content;
          if (logprobsObj && Array.isArray(logprobsObj) && logprobsObj.length > 0) {
            for (const lp of logprobsObj) {
              const logprob = lp.logprob;
              if (logprob !== undefined) {
                const linearProb = Math.exp(logprob);
                accumulatedProb += linearProb;
                tokenCount++;

                if (tokenCount === this.config.tokensToEvaluate) {
                  const avgProb = accumulatedProb / tokenCount;
                  if (avgProb < (this.config.cascadeThreshold || 0.85)) {
                    // Abort! Confidence too low.
                    this.config.onEvent?.('cascade_triggered', { tokenCount, avgProb, threshold: this.config.cascadeThreshold || 0.85 });
                    controller.abort();
                    return { text: '', aborted: true };
                  }
                }
              }
            }
          }
        } catch (e) {
          if (!(e instanceof SyntaxError)) throw e;
        }
      }
    }

    // Flush remaining buffer
    if (buffer.trim().startsWith('data: ')) {
       try {
           const data = JSON.parse(buffer.trim().slice(6));
           const delta = data.choices?.[0]?.delta?.content || '';
           if (delta) fullText += delta;
       } catch (e) { if (!(e instanceof SyntaxError)) throw e; }
    }

    return { text: fullText, aborted: false };
  }

  /**
   * Fallback to heavy model. Only returns the full string for now.
   */
  private async fetchHeavyModel(messages: Message[], options?: ChatOptions): Promise<string> {
    const { heavyModel } = this.config;
    const provider = heavyModel.provider || 'openai';

    if (provider === 'gemini') {
      return this.fetchGemini(messages, options);
    }

    // Default OpenAI format
    const url = heavyModel.url || 'https://api.openai.com/v1/chat/completions';
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (heavyModel.apiKey) headers['Authorization'] = `Bearer ${heavyModel.apiKey}`;

    const response = await this.fetchFn(url, {
      method: 'POST',
      headers,
      signal: options?.signal,
      body: JSON.stringify({
        model: heavyModel.model,
        messages,
        stream: false
      })
    });

    if (!response.ok) {
      throw new Error(`Heavy model HTTP ${response.status}`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content || '';
  }

  private async fetchGemini(messages: Message[], options?: ChatOptions): Promise<string> {
    const { heavyModel } = this.config;
    const apiKey = heavyModel.apiKey;
    if (!apiKey) throw new Error('Gemini requires an API key');

    // NOTE: Google's REST API uses the key as a query parameter. This means the API key
    // may appear in server logs, proxy logs, and error reporting. For higher security,
    // consider using the Google Cloud client libraries with service account auth instead.
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${heavyModel.model}:generateContent?key=${apiKey}`;

    const systemPrompt = messages.find(m => m.role === 'system')?.content;
    const contents = messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }]
      }));

    const body: any = { contents };
    if (systemPrompt) {
      body.system_instruction = { parts: [{ text: systemPrompt }] };
    }

    const response = await this.fetchFn(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: options?.signal,
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      throw new Error(`Gemini HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
  }
}
