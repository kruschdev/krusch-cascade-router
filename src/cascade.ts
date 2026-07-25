import { isComplexPrompt, Message, ClassifierOptions } from './classifier.js';

export interface ModelConfig {
  url?: string;
  apiKey?: string;
  model: string;
  provider?: 'openai' | 'gemini';
}

export type TelemetryEvent = 'route_fast' | 'route_heavy' | 'cascade_triggered';

export interface JeanSREGateConfig {
  enabled: boolean;
  sreUrl?: string;
  conversationId?: string;
  projectContext?: string;
}

export interface RouterConfig {
  fastModel: ModelConfig;
  heavyModel: ModelConfig;
  backgroundModel?: ModelConfig;
  cascadeThreshold?: number; // Default 0.85 (Linear probability)
  tokensToEvaluate?: number; // Default 5
  classifier?: ClassifierOptions;
  fetch?: typeof fetch;
  onEvent?: (event: TelemetryEvent, metadata?: Record<string, any>) => void;
  jeanSREGate?: JeanSREGateConfig;
}

export interface CascadeResponse {
  text: string;
  routedTo: 'fast' | 'heavy';
  aborted: boolean;
}

export interface ChatOptions {
  signal?: AbortSignal;
  speedPriority?: 'high' | 'medium' | 'low';
  urgency?: 'high' | 'medium' | 'low';
}

export class CascadeTriggeredError extends Error {
  constructor(message?: string) {
    super(message);
    this.name = 'CascadeTriggeredError';
  }
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

    // Fetch SRE suggestions if gate is enabled
    if (this.config.jeanSREGate?.enabled) {
      const sreSuggestions = await this.fetchJeanSuggestions(this.config.jeanSREGate.projectContext);
      if (sreSuggestions) {
        formattedMessages.push({
          role: 'system',
          content: `[JEAN SRE TELEMETRY & RECOMMENDATIONS]\n${sreSuggestions}\n\nINSTRUCTION: Jean SRE has detected these fleet/system anomalies or suggestions. As the agent-in-the-loop, you MUST present these to the human user for review, analyze their risks/benefits, and wait for the user's explicit approval before proposing or executing any commands or modifications.`
        });
      }
    }

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
      const modelToUse = (options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : this.config.fastModel;
      this.config.onEvent?.('route_fast', { reason: 'high_confidence', model: modelToUse.model, speedPriority: options?.speedPriority || 'normal' });
      return { text: fastResult.text, routedTo: 'fast', aborted: false };
    } catch (err) {
      this.config.onEvent?.('route_heavy', { reason: 'fast_model_error', error: (err as Error).message });
      const heavyText = await this.fetchHeavyModel(formattedMessages, options);
      return { text: heavyText, routedTo: 'heavy', aborted: true };
    }
  }

  /**
   * Stream a chat request, yielding text chunks in real-time.
   * Speculatively buffers the first N tokens from the fast model.
   * If confidence dips below threshold, silent abort occurs and fallbacks to heavy.
   */
  async *stream(messages: Message[] | string, systemPrompt?: string, options?: ChatOptions): AsyncGenerator<string, void, unknown> {
    const formattedMessages = this.formatMessages(messages, systemPrompt);

    // Fetch SRE suggestions if gate is enabled
    if (this.config.jeanSREGate?.enabled) {
      const sreSuggestions = await this.fetchJeanSuggestions(this.config.jeanSREGate.projectContext);
      if (sreSuggestions) {
        formattedMessages.push({
          role: 'system',
          content: `[JEAN SRE TELEMETRY & RECOMMENDATIONS]\n${sreSuggestions}\n\nINSTRUCTION: Jean SRE has detected these fleet/system anomalies or suggestions. As the agent-in-the-loop, you MUST present these to the human user for review, analyze their risks/benefits, and wait for the user's explicit approval before proposing or executing any commands or modifications.`
        });
      }
    }

    // 1. Predictive Classifier
    const isComplex = isComplexPrompt(formattedMessages, this.config.classifier);

    if (isComplex) {
      this.config.onEvent?.('route_heavy', { reason: 'classifier_heuristic' });
      yield* this.streamHeavyModel(formattedMessages, options);
      return;
    }

    // 2. Try fast model streaming with Speculative Cascade fallback
    try {
      yield* this.streamAndEvaluateFastModelGen(formattedMessages, options);
    } catch (err) {
      if (err instanceof CascadeTriggeredError) {
        this.config.onEvent?.('route_heavy', { reason: 'cascade_fallback' });
      } else {
        this.config.onEvent?.('route_heavy', { reason: 'fast_model_error', error: (err as Error).message });
      }
      yield* this.streamHeavyModel(formattedMessages, options);
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
    const modelToUse = (options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
      ? this.config.backgroundModel
      : this.config.fastModel;
    const url = modelToUse.url || 'http://localhost:11434/v1/chat/completions';
    
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (modelToUse.apiKey) headers['Authorization'] = `Bearer ${modelToUse.apiKey}`;

    const controller = new AbortController();
    
    // Link external abort signal to our internal controller
    const onAbort = () => controller.abort();
    if (options?.signal) {
      options.signal.addEventListener('abort', onAbort, { once: true });
      if (options.signal.aborted) controller.abort();
    }

    try {
      const response = await this.fetchFn(url, {
        method: 'POST',
        headers,
        signal: controller.signal,
        body: JSON.stringify({
          model: modelToUse.model,
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

      try {
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
      } finally {
        reader.releaseLock();
      }
    } finally {
      if (options?.signal) {
        options.signal.removeEventListener('abort', onAbort);
      }
    }
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

  private async *streamAndEvaluateFastModelGen(messages: Message[], options?: ChatOptions): AsyncGenerator<string, void, unknown> {
    const modelToUse = (options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
      ? this.config.backgroundModel
      : this.config.fastModel;
    const url = modelToUse.url || 'http://localhost:11434/v1/chat/completions';
    
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (modelToUse.apiKey) headers['Authorization'] = `Bearer ${modelToUse.apiKey}`;

    const controller = new AbortController();
    const onAbort = () => controller.abort();
    if (options?.signal) {
      options.signal.addEventListener('abort', onAbort, { once: true });
      if (options.signal.aborted) controller.abort();
    }

    let response: Response;
    try {
      response = await this.fetchFn(url, {
        method: 'POST',
        headers,
        signal: controller.signal,
        body: JSON.stringify({
          model: modelToUse.model,
          messages,
          stream: true,
          logprobs: true
        })
      });
    } catch (err) {
      throw err;
    }

    if (!response.ok) {
      if (options?.signal) options.signal.removeEventListener('abort', onAbort);
      throw new Error(`Fast model HTTP ${response.status}`);
    }

    if (!response.body) {
      if (options?.signal) options.signal.removeEventListener('abort', onAbort);
      throw new Error('No response body');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    const bufferedTokens: string[] = [];
    let tokenCount = 0;
    let accumulatedProb = 0;
    let evaluationComplete = false;

    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]') continue;
          if (!trimmed.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(trimmed.slice(6));
            const choice = data.choices?.[0];
            const delta = choice?.delta?.content || '';

            // Handle logprobs
            const logprobsObj = choice?.logprobs?.content;
            let currentTokenProbs: number[] = [];

            if (logprobsObj && Array.isArray(logprobsObj) && logprobsObj.length > 0) {
              for (const lp of logprobsObj) {
                const logprob = lp.logprob;
                if (logprob !== undefined) {
                  currentTokenProbs.push(Math.exp(logprob));
                }
              }
            }

            if (delta) {
              if (evaluationComplete) {
                yield delta;
              } else {
                bufferedTokens.push(delta);
                for (const prob of currentTokenProbs) {
                  accumulatedProb += prob;
                  tokenCount++;

                  if (tokenCount === this.config.tokensToEvaluate) {
                    evaluationComplete = true;
                    const avgProb = accumulatedProb / tokenCount;
                    if (avgProb < (this.config.cascadeThreshold || 0.85)) {
                      this.config.onEvent?.('cascade_triggered', { tokenCount, avgProb, threshold: this.config.cascadeThreshold || 0.85 });
                      controller.abort();
                      throw new CascadeTriggeredError('Speculative cascade triggered');
                    } else {
                      // Flush buffer
                      for (const t of bufferedTokens) {
                        yield t;
                      }
                      bufferedTokens.length = 0;
                    }
                  }
                }
              }
            }
          } catch (e) {
            if (e instanceof CascadeTriggeredError) throw e;
            if (!(e instanceof SyntaxError)) throw e;
          }
        }
      }

      // Flush remaining line buffer if any
      if (buffer.trim().startsWith('data: ')) {
        try {
          const data = JSON.parse(buffer.trim().slice(6));
          const delta = data.choices?.[0]?.delta?.content || '';
          if (delta) {
            if (evaluationComplete) {
              yield delta;
            } else {
              bufferedTokens.push(delta);
            }
          }
        } catch (e) { if (!(e instanceof SyntaxError)) throw e; }
      }

      if (!evaluationComplete) {
        evaluationComplete = true;
        const avgProb = tokenCount > 0 ? accumulatedProb / tokenCount : 1.0;
        if (tokenCount > 0 && avgProb < (this.config.cascadeThreshold || 0.85)) {
          this.config.onEvent?.('cascade_triggered', { tokenCount, avgProb, threshold: this.config.cascadeThreshold || 0.85 });
          throw new CascadeTriggeredError('Speculative cascade triggered on short stream');
        } else {
          for (const t of bufferedTokens) {
            yield t;
          }
        }
      }
      
      this.config.onEvent?.('route_fast', { reason: 'high_confidence' });

    } finally {
      reader.releaseLock();
      if (options?.signal) {
        options.signal.removeEventListener('abort', onAbort);
      }
    }
  }

  private async *streamHeavyModel(messages: Message[], options?: ChatOptions): AsyncGenerator<string, void, unknown> {
    const { heavyModel } = this.config;
    const provider = heavyModel.provider || 'openai';

    if (provider === 'gemini') {
      yield* this.streamGemini(messages, options);
      return;
    }

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
        stream: true
      })
    });

    if (!response.ok) {
      throw new Error(`Heavy model HTTP ${response.status}`);
    }

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]') continue;
          if (!trimmed.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(trimmed.slice(6));
            const delta = data.choices?.[0]?.delta?.content || '';
            if (delta) yield delta;
          } catch (e) {
            if (!(e instanceof SyntaxError)) throw e;
          }
        }
      }

      if (buffer.trim().startsWith('data: ')) {
        try {
          const data = JSON.parse(buffer.trim().slice(6));
          const delta = data.choices?.[0]?.delta?.content || '';
          if (delta) yield delta;
        } catch (e) { if (!(e instanceof SyntaxError)) throw e; }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private async *streamGemini(messages: Message[], options?: ChatOptions): AsyncGenerator<string, void, unknown> {
    const { heavyModel } = this.config;
    const apiKey = heavyModel.apiKey;
    if (!apiKey) throw new Error('Gemini requires an API key');

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${heavyModel.model}:streamGenerateContent?key=${apiKey}`;

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

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        let braceCount = 0;
        let startIdx = -1;
        
        for (let i = 0; i < buffer.length; i++) {
          const char = buffer[i];
          if (char === '{') {
            if (braceCount === 0) startIdx = i;
            braceCount++;
          } else if (char === '}') {
            braceCount--;
            if (braceCount === 0 && startIdx !== -1) {
              const jsonStr = buffer.substring(startIdx, i + 1);
              try {
                const data = JSON.parse(jsonStr);
                const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
                if (text) yield text;
              } catch (e) {
                // Ignore incomplete JSON chunks
              }
              buffer = buffer.substring(i + 1);
              i = -1;
              startIdx = -1;
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private async fetchJeanSuggestions(project?: string): Promise<string | null> {
    const gate = this.config.jeanSREGate;
    if (!gate || !gate.enabled) return null;

    const url = gate.sreUrl || 'http://localhost:3005/chat';
    const conversationId = gate.conversationId || 'a123e456-789b-12d3-a456-426614174000';
    const query = "Provide a concise list of any current fleet warnings, database anomalies, CPU thermal alerts, or Docker container issues that require human-agent review.";
    const message = project ? `[Project Isolation Context: ${project}] ${query}` : query;

    try {
      const response = await this.fetchFn(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversationId, message, noTools: false })
      });
      if (!response.ok) return null;
      const data = await response.json();
      return data.reply || null;
    } catch (err) {
      // Gracefully degrade if SRE endpoint is down or unreachable
      return null;
    }
  }
}
