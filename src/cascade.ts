import { 
  isComplexPrompt, 
  pruneText, 
  evaluateComplexityScore, 
  classifySpecialistRole, 
  SpecialistRole, 
  Message, 
  ClassifierOptions 
} from './classifier.js';

export interface ModelConfig {
  url?: string;
  apiKey?: string;
  model: string;
  provider?: 'openai' | 'gemini' | 'openrouter';
  costPerMillionInputTokens?: number;  // Default: 0 for local/fast, 0.15 for cloud
  costPerMillionOutputTokens?: number; // Default: 0 for local/fast, 0.60 for cloud
  headers?: Record<string, string>;
}

export type TelemetryEvent = 
  | 'route_fast' 
  | 'route_heavy' 
  | 'route_specialist'
  | 'cascade_triggered' 
  | 'json_fallback_triggered' 
  | 'repetition_loop_triggered'
  | 'speculative_branch_hedged'
  | 'entropy_collapse_triggered';

export interface JeanSREGateConfig {
  enabled: boolean;
  sreUrl?: string;
  conversationId?: string;
  projectContext?: string;
}

export interface RouterConfig {
  fastModel?: ModelConfig;
  heavyModel?: ModelConfig;
  backgroundModel?: ModelConfig;
  specialistModels?: Partial<Record<SpecialistRole, ModelConfig>>;
  openrouterApiKey?: string;
  openrouterReferer?: string;
  openrouterTitle?: string;
  cascadeThreshold?: number; // Default 0.85 (Linear probability)
  tokensToEvaluate?: number; // Default 5
  maxRepetitiveTokens?: number; // Default 4 (catches degenerate repetitive loops)
  classifier?: ClassifierOptions;
  prunePreRouting?: boolean; // If true, automatically prunes whitespace/filler before routing
  speculativeBranching?: boolean; // If true, enables Second Thought parallel hedging for borderline queries
  fetch?: typeof fetch;
  onEvent?: (event: TelemetryEvent, metadata?: Record<string, any>) => void;
  jeanSREGate?: JeanSREGateConfig;
}

export interface UsageMetrics {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  estimatedCostUsd: number;
}

export interface RouterMetrics {
  totalRequests: number;
  fastRequests: number;
  heavyRequests: number;
  cascadedRequests: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  estimatedCostUsd: number;
  estimatedSavingsUsd: number;
}

export interface CascadeResponse {
  text: string;
  routedTo: 'fast' | 'heavy' | SpecialistRole;
  aborted: boolean;
  usage?: UsageMetrics;
  model?: string;
}

export interface ChatOptions {
  signal?: AbortSignal;
  speedPriority?: 'high' | 'medium' | 'low';
  urgency?: 'high' | 'medium' | 'low';
  prunePreRouting?: boolean;
  speculativeBranching?: boolean;
}

export interface ChatJsonOptions extends ChatOptions {
  schema?: Record<string, any>;
  maxJsonRetries?: number;
}

export interface CrossRouterOptions {
  openrouterApiKey?: string;
  siteUrl?: string;
  appName?: string;
  customModels?: Partial<Record<SpecialistRole, string>>;
  cascadeThreshold?: number;
  tokensToEvaluate?: number;
  maxRepetitiveTokens?: number;
  speculativeBranching?: boolean;
  prunePreRouting?: boolean;
  fetch?: typeof fetch;
  onEvent?: (event: TelemetryEvent, metadata?: Record<string, any>) => void;
  jeanSREGate?: JeanSREGateConfig;
}

/**
 * Creates a CascadeRouter pre-configured with the 7 specialist models from
 * the #1 ranked Cross-Router, routing via OpenRouter's unified API layer.
 */
export function createCrossRouter(options?: CrossRouterOptions): CascadeRouter {
  const globalProcess = typeof globalThis !== 'undefined' ? (globalThis as any).process : undefined;
  const apiKey = options?.openrouterApiKey || globalProcess?.env?.OPENROUTER_API_KEY;
  const referer = options?.siteUrl || 'https://github.com/kruschdev/krusch-cascade-router';
  const title = options?.appName || 'krusch-cascade-router';

  // Cross-Router top 7 empirical model pool with published OpenRouter pricing ($/1M tokens)
  const defaultModels: Record<SpecialistRole, { model: string; inputCost: number; outputCost: number }> = {
    general_fast: { model: 'google/gemini-3.1-flash-lite', inputCost: 0.25, outputCost: 1.50 },
    factual_stem: { model: 'deepseek/deepseek-v4-flash', inputCost: 0.14, outputCost: 0.28 },
    code: { model: 'Qwen/Qwen3-Coder-Next', inputCost: 0.12, outputCost: 0.48 },
    reasoning_fast: { model: 'grok-4-1-fast-reasoning', inputCost: 0.60, outputCost: 2.40 },
    reasoning_deep: { model: 'deepseek/deepseek-v4-pro', inputCost: 0.55, outputCost: 2.19 },
    games_spatial: { model: 'gemini-3-flash-preview', inputCost: 0.35, outputCost: 1.50 },
    comprehension_rc: { model: 'qwen/qwen3-235b-a22b-2507', inputCost: 0.05, outputCost: 0.20 }
  };

  const specialistModels: Record<SpecialistRole, ModelConfig> = {} as any;

  for (const [roleKey, def] of Object.entries(defaultModels)) {
    const role = roleKey as SpecialistRole;
    const modelName = options?.customModels?.[role] || def.model;
    specialistModels[role] = {
      model: modelName,
      provider: 'openrouter',
      apiKey,
      costPerMillionInputTokens: def.inputCost,
      costPerMillionOutputTokens: def.outputCost
    };
  }

  return new CascadeRouter({
    fastModel: specialistModels.factual_stem,
    heavyModel: specialistModels.reasoning_deep,
    specialistModels,
    openrouterApiKey: apiKey,
    openrouterReferer: referer,
    openrouterTitle: title,
    cascadeThreshold: options?.cascadeThreshold,
    tokensToEvaluate: options?.tokensToEvaluate,
    maxRepetitiveTokens: options?.maxRepetitiveTokens,
    speculativeBranching: options?.speculativeBranching,
    prunePreRouting: options?.prunePreRouting,
    fetch: options?.fetch,
    onEvent: options?.onEvent,
    jeanSREGate: options?.jeanSREGate
  });
}

export class CascadeTriggeredError extends Error {
  constructor(message?: string) {
    super(message);
    this.name = 'CascadeTriggeredError';
  }
}

/**
 * Robust JSON extraction helper that parses raw AI JSON output or repairs markdown-wrapped json.
 */
function parseJsonSafe<T = any>(text: string): T {
  if (!text || typeof text !== 'string') {
    throw new Error('Cannot parse empty or non-string AI response as JSON.');
  }

  const trimmed = text.trim();

  // 1. Direct parse attempt
  try {
    return JSON.parse(trimmed) as T;
  } catch (_) {
    // Fall through to markdown regex extractor
  }

  // 2. Extract code fence ```json ... ```
  const codeBlockMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (codeBlockMatch && codeBlockMatch[1]) {
    try {
      return JSON.parse(codeBlockMatch[1].trim()) as T;
    } catch (_) {
      // Fall through
    }
  }

  // 3. Extract outermost object or array brackets
  const firstBrace = trimmed.indexOf('{');
  const lastBrace = trimmed.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    try {
      return JSON.parse(trimmed.slice(firstBrace, lastBrace + 1)) as T;
    } catch (_) {
      // Fall through
    }
  }

  const firstBracket = trimmed.indexOf('[');
  const lastBracket = trimmed.lastIndexOf(']');
  if (firstBracket !== -1 && lastBracket > firstBracket) {
    try {
      return JSON.parse(trimmed.slice(firstBracket, lastBracket + 1)) as T;
    } catch (_) {
      // Fall through
    }
  }

  throw new Error(`Failed to parse AI JSON response: ${trimmed.slice(0, 100)}...`);
}

/**
 * Approximate token estimator: ~4 characters per token in English.
 */
function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

export class CascadeRouter {
  private config: RouterConfig;
  private fetchFn: typeof fetch;
  private metrics: RouterMetrics;

  constructor(config: RouterConfig) {
    if (!config.fastModel && !config.specialistModels) {
      throw new Error('RouterConfig requires either fastModel or specialistModels.');
    }

    const defaultFast = config.fastModel ?? (config.specialistModels?.factual_stem || config.specialistModels?.general_fast || Object.values(config.specialistModels || {})[0]);
    const defaultHeavy = config.heavyModel ?? (config.specialistModels?.reasoning_deep || config.specialistModels?.reasoning_fast || Object.values(config.specialistModels || {})[0]);

    this.config = {
      ...config,
      fastModel: defaultFast!,
      heavyModel: defaultHeavy!,
      cascadeThreshold: config.cascadeThreshold ?? 0.85,
      tokensToEvaluate: config.tokensToEvaluate ?? 5,
      maxRepetitiveTokens: config.maxRepetitiveTokens ?? 4
    };
    this.fetchFn = config.fetch ?? (typeof globalThis !== 'undefined' ? globalThis.fetch : fetch);
    if (!this.fetchFn) {
      throw new Error('A global fetch API is required, or a custom fetch implementation must be provided in RouterConfig.');
    }

    this.metrics = {
      totalRequests: 0,
      fastRequests: 0,
      heavyRequests: 0,
      cascadedRequests: 0,
      totalPromptTokens: 0,
      totalCompletionTokens: 0,
      estimatedCostUsd: 0,
      estimatedSavingsUsd: 0
    };
  }

  /**
   * Retrieves aggregate routing and token telemetry metrics.
   */
  getMetrics(): RouterMetrics {
    return { ...this.metrics };
  }

  /**
   * Resets internal telemetry counters.
   */
  resetMetrics(): void {
    this.metrics = {
      totalRequests: 0,
      fastRequests: 0,
      heavyRequests: 0,
      cascadedRequests: 0,
      totalPromptTokens: 0,
      totalCompletionTokens: 0,
      estimatedCostUsd: 0,
      estimatedSavingsUsd: 0
    };
  }

  /**
   * Complete a chat request, routing automatically between fast edge and heavy cloud models,
   * or across the 7-model specialist pool if configured.
   */
  async chat(messages: Message[] | string, systemPrompt?: string, options?: ChatOptions): Promise<CascadeResponse> {
    const formattedMessages = this.formatMessages(messages, systemPrompt, options);

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

    const promptText = formattedMessages.map(m => m.content).join('\n');
    const promptTokens = estimateTokens(promptText);

    // 1. Predictive Classifier Options
    const classifierOpts = {
      ...this.config.classifier,
      prunePreRouting: options?.prunePreRouting ?? this.config.prunePreRouting ?? this.config.classifier?.prunePreRouting
    };
    const isComplex = isComplexPrompt(formattedMessages, classifierOpts);
    const complexityScore = evaluateComplexityScore(formattedMessages, classifierOpts);
    const isBorderline = complexityScore >= 0.25 && complexityScore <= 0.70;
    const shouldHedge = (options?.speculativeBranching ?? this.config.speculativeBranching) && isBorderline;

    // --- Path A: Multi-Specialist Pool Routing ---
    if (this.config.specialistModels && Object.keys(this.config.specialistModels).length > 0) {
      const specialistRole = classifySpecialistRole(formattedMessages, classifierOpts);
      let targetRole: SpecialistRole = specialistRole;
      let targetModel = this.config.specialistModels[specialistRole] || this.config.fastModel!;
      const heavyModel = this.config.specialistModels.reasoning_deep || this.config.heavyModel!;

      if (isComplex && targetRole !== 'reasoning_deep' && targetRole !== 'reasoning_fast') {
        targetRole = 'reasoning_deep';
        targetModel = heavyModel;
      }

      const modelToUse = (options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : targetModel;

      this.config.onEvent?.('route_specialist', {
        role: targetRole,
        model: modelToUse.model,
        complexityScore,
        isComplex
      });

      let hedgedController: AbortController | null = null;
      let hedgedHeavyPromise: Promise<{ text: string; usage: UsageMetrics }> | null = null;

      if (shouldHedge && targetRole !== 'reasoning_deep') {
        hedgedController = new AbortController();
        this.config.onEvent?.('speculative_branch_hedged', { complexityScore });
        hedgedHeavyPromise = this.fetchModel(heavyModel, formattedMessages, { ...options, signal: hedgedController.signal });
      }

      try {
        const { text, usage } = await this.fetchModel(modelToUse, formattedMessages, options);
        if (hedgedController) hedgedController.abort();

        const isHeavyRole = targetRole === 'reasoning_deep';
        this.recordRequestMetrics(isHeavyRole ? 'heavy' : 'fast', false, promptTokens, usage.completionTokens, usage.estimatedCostUsd);
        return { text, routedTo: targetRole, aborted: false, usage, model: modelToUse.model };
      } catch (err) {
        if (hedgedHeavyPromise) {
          try {
            const { text, usage } = await hedgedHeavyPromise;
            this.recordRequestMetrics('heavy', true, promptTokens, usage.completionTokens, usage.estimatedCostUsd);
            return { text, routedTo: 'reasoning_deep', aborted: true, usage, model: heavyModel.model };
          } catch (_) {}
        }
        this.config.onEvent?.('route_heavy', { reason: 'specialist_model_error', role: targetRole, error: (err as Error).message });
        const { text, usage } = await this.fetchModel(heavyModel, formattedMessages, options);
        this.recordRequestMetrics('heavy', true, promptTokens, usage.completionTokens, usage.estimatedCostUsd);
        return { text, routedTo: 'reasoning_deep', aborted: true, usage, model: heavyModel.model };
      }
    }

    // --- Path B: Standard 2-Tier Cascade ---
    if (isComplex) {
      // Bypass fast model entirely
      this.config.onEvent?.('route_heavy', { reason: 'classifier_heuristic' });
      const { text, usage } = await this.fetchHeavyModel(formattedMessages, options);
      this.recordRequestMetrics('heavy', false, promptTokens, usage.completionTokens, usage.estimatedCostUsd);
      return { text, routedTo: 'heavy', aborted: false, usage, model: this.config.heavyModel!.model };
    }

    // Speculative Branching ("Second Thought" arXiv: 2608.13667)
    // For borderline queries [0.25, 0.70], hedge heavy model in parallel to mask cascade latency
    let hedgedController: AbortController | null = null;
    let hedgedHeavyPromise: Promise<{ text: string; usage: UsageMetrics }> | null = null;

    if (shouldHedge) {
      hedgedController = new AbortController();
      this.config.onEvent?.('speculative_branch_hedged', { complexityScore });
      hedgedHeavyPromise = this.fetchHeavyModel(formattedMessages, { ...options, signal: hedgedController.signal });
    }

    // 2. Try fast model with Speculative Cascade
    try {
      const fastResult = await this.streamAndEvaluateFastModel(formattedMessages, options);
      if (fastResult.aborted) {
        this.config.onEvent?.('route_heavy', { reason: 'cascade_fallback' });
        const { text, usage } = hedgedHeavyPromise 
          ? await hedgedHeavyPromise 
          : await this.fetchHeavyModel(formattedMessages, options);
        this.recordRequestMetrics('heavy', true, promptTokens, usage.completionTokens, usage.estimatedCostUsd);
        return { text, routedTo: 'heavy', aborted: true, usage, model: this.config.heavyModel!.model };
      }

      // Fast model succeeded: cleanly abort the speculative heavy hedge if active
      if (hedgedController) {
        hedgedController.abort();
      }

      const modelToUse = (options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : this.config.fastModel!;

      const completionTokens = estimateTokens(fastResult.text);
      const usage = this.calculateUsage(modelToUse, promptTokens, completionTokens);

      this.config.onEvent?.('route_fast', { reason: 'high_confidence', model: modelToUse.model, speedPriority: options?.speedPriority || 'normal' });
      this.recordRequestMetrics('fast', false, promptTokens, completionTokens, usage.estimatedCostUsd);

      return { text: fastResult.text, routedTo: 'fast', aborted: false, usage, model: modelToUse.model };
    } catch (err) {
      if (hedgedHeavyPromise) {
        try {
          const { text, usage } = await hedgedHeavyPromise;
          this.recordRequestMetrics('heavy', true, promptTokens, usage.completionTokens, usage.estimatedCostUsd);
          return { text, routedTo: 'heavy', aborted: true, usage, model: this.config.heavyModel!.model };
        } catch (_) {}
      }
      this.config.onEvent?.('route_heavy', { reason: 'fast_model_error', error: (err as Error).message });
      const { text, usage } = await this.fetchHeavyModel(formattedMessages, options);
      this.recordRequestMetrics('heavy', true, promptTokens, usage.completionTokens, usage.estimatedCostUsd);
      return { text, routedTo: 'heavy', aborted: true, usage, model: this.config.heavyModel!.model };
    }
  }

  /**
   * Complete a chat request and parse the output as structured JSON.
   * If the fast model produces malformed JSON, automatically cascades to heavy model.
   */
  async chatJson<T = any>(messages: Message[] | string, systemPrompt?: string, options?: ChatJsonOptions): Promise<T & { _routedTo?: 'fast' | 'heavy'; _usage?: UsageMetrics }> {
    const jsonInstruction = 'IMPORTANT: You must respond ONLY with a valid JSON object or array. Do not include markdown preamble, commentary, or backticks.';
    const combinedSystemPrompt = systemPrompt ? `${systemPrompt}\n\n${jsonInstruction}` : jsonInstruction;

    const response = await this.chat(messages, combinedSystemPrompt, options);

    try {
      const parsed = parseJsonSafe<T>(response.text);
      if (typeof parsed === 'object' && parsed !== null) {
        (parsed as any)._routedTo = response.routedTo;
        (parsed as any)._usage = response.usage;
      }
      return parsed as any;
    } catch (parseErr) {
      // If fast model produced unparseable JSON, cascade to heavy model
      if (response.routedTo === 'fast') {
        this.config.onEvent?.('json_fallback_triggered', { error: (parseErr as Error).message });
        const formatted = this.formatMessages(messages, combinedSystemPrompt, options);
        const { text, usage } = await this.fetchHeavyModel(formatted, options);
        const parsedHeavy = parseJsonSafe<T>(text);
        if (typeof parsedHeavy === 'object' && parsedHeavy !== null) {
          (parsedHeavy as any)._routedTo = 'heavy';
          (parsedHeavy as any)._usage = usage;
        }
        return parsedHeavy as any;
      }
      throw parseErr;
    }
  }

  /**
   * Stream a chat request, yielding text chunks in real-time.
   * Speculatively buffers the first N tokens from the fast model.
   * If confidence dips below threshold or a repetitive loop is detected, silent abort occurs and falls back to heavy.
   */
  async *stream(messages: Message[] | string, systemPrompt?: string, options?: ChatOptions): AsyncGenerator<string, void, unknown> {
    const formattedMessages = this.formatMessages(messages, systemPrompt, options);

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
    const classifierOpts = {
      ...this.config.classifier,
      prunePreRouting: options?.prunePreRouting ?? this.config.prunePreRouting ?? this.config.classifier?.prunePreRouting
    };
    const isComplex = isComplexPrompt(formattedMessages, classifierOpts);

    // --- Path A: Multi-Specialist Pool Streaming ---
    if (this.config.specialistModels && Object.keys(this.config.specialistModels).length > 0) {
      const specialistRole = classifySpecialistRole(formattedMessages, classifierOpts);
      let targetRole: SpecialistRole = specialistRole;
      let targetModel = this.config.specialistModels[specialistRole] || this.config.fastModel!;
      const heavyModel = this.config.specialistModels.reasoning_deep || this.config.heavyModel!;

      if (isComplex && targetRole !== 'reasoning_deep' && targetRole !== 'reasoning_fast') {
        targetRole = 'reasoning_deep';
        targetModel = heavyModel;
      }

      const modelToUse = (options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : targetModel;

      this.config.onEvent?.('route_specialist', {
        role: targetRole,
        model: modelToUse.model,
        isComplex
      });

      if (targetRole === 'reasoning_deep') {
        yield* this.streamHeavyModel(formattedMessages, options, heavyModel);
        return;
      }

      try {
        yield* this.streamAndEvaluateFastModelGen(formattedMessages, options, modelToUse);
      } catch (err) {
        if (err instanceof CascadeTriggeredError) {
          this.config.onEvent?.('route_heavy', { reason: 'cascade_fallback', fallbackFrom: targetRole });
        } else {
          this.config.onEvent?.('route_heavy', { reason: 'fast_model_error', fallbackFrom: targetRole, error: (err as Error).message });
        }
        yield* this.streamHeavyModel(formattedMessages, options, heavyModel);
      }
      return;
    }

    // --- Path B: Standard 2-Tier Cascade Streaming ---
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

  private formatMessages(messages: Message[] | string, systemPrompt?: string, options?: ChatOptions): Message[] {
    const msgs: Message[] = [];
    const shouldPrune = options?.prunePreRouting ?? this.config.prunePreRouting;

    if (systemPrompt) {
      msgs.push({ role: 'system', content: shouldPrune ? pruneText(systemPrompt) : systemPrompt });
    }

    if (typeof messages === 'string') {
      msgs.push({ role: 'user', content: shouldPrune ? pruneText(messages) : messages });
    } else {
      for (const m of messages) {
        msgs.push({
          role: m.role,
          content: shouldPrune ? pruneText(m.content) : m.content
        });
      }
    }
    return msgs;
  }

  /**
   * Helper: Detects repetitive token degenerate loops and reasoning entropy collapse
   * during speculative buffer evaluation (PIG Engine / Trajectory Guard arXiv: 2606.08162).
   */
  private detectRepetitiveLoop(tokens: string[]): boolean {
    const maxRep = this.config.maxRepetitiveTokens || 4;
    if (tokens.length < maxRep) return false;

    // 1. Single token repetition check
    const lastToken = tokens[tokens.length - 1].trim();
    if (lastToken) {
      let identicalCount = 0;
      for (let i = tokens.length - 1; i >= 0; i--) {
        if (tokens[i].trim() === lastToken) {
          identicalCount++;
        } else {
          break;
        }
      }
      if (identicalCount >= maxRep) return true;
    }

    // 2. 2-gram cyclic repetition check (e.g. A, B, A, B, A, B)
    if (tokens.length >= 6) {
      const t1 = tokens[tokens.length - 2].trim();
      const t2 = tokens[tokens.length - 1].trim();
      if (t1 && t2 && t1 !== t2) {
        if (
          tokens[tokens.length - 4].trim() === t1 &&
          tokens[tokens.length - 3].trim() === t2 &&
          tokens[tokens.length - 6].trim() === t1 &&
          tokens[tokens.length - 5].trim() === t2
        ) {
          return true;
        }
      }
    }

    // 3. Sliding-window unique token entropy collapse (last 10 tokens)
    if (tokens.length >= 10) {
      const windowTokens = tokens.slice(-10).map(t => t.trim().toLowerCase()).filter(Boolean);
      const unique = new Set(windowTokens);
      // If 8+ tokens contain 2 or fewer distinct words, entropy has collapsed
      if (windowTokens.length >= 8 && unique.size <= 2) {
        return true;
      }
    }

    return false;
  }

  /**
   * Streams the fast model, buffering the first N tokens to check logprobs and loops.
   * If confidence is lower than threshold or loop is detected, aborts and returns { aborted: true }.
   */
  private async streamAndEvaluateFastModel(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): Promise<{ text: string, aborted: boolean }> {
    const modelToUse = customModel
      || ((options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : this.config.fastModel!);
    const { url, headers } = this.getModelEndpoint(modelToUse, true);

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
      const seenDeltas: string[] = [];

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
              
              if (delta) {
                fullText += delta;
                seenDeltas.push(delta);

                // Repetition loop check
                if (this.detectRepetitiveLoop(seenDeltas)) {
                  this.config.onEvent?.('repetition_loop_triggered', { tokenCount, delta });
                  controller.abort();
                  return { text: '', aborted: true };
                }
              }

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

  private getModelEndpoint(modelConfig: ModelConfig, isLocalFastDefault: boolean = false): { url: string; headers: Record<string, string> } {
    const provider = modelConfig.provider || 'openai';
    const isOpener = provider === 'openrouter';
    let defaultUrl = 'https://api.openai.com/v1/chat/completions';
    if (isOpener) {
      defaultUrl = 'https://openrouter.ai/api/v1/chat/completions';
    } else if (isLocalFastDefault) {
      defaultUrl = 'http://localhost:11434/v1/chat/completions';
    }
    const url = modelConfig.url || defaultUrl;

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const apiKey = modelConfig.apiKey || (isOpener ? this.config.openrouterApiKey : undefined);
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }

    if (isOpener) {
      headers['HTTP-Referer'] = modelConfig.headers?.['HTTP-Referer'] || this.config.openrouterReferer || 'https://github.com/kruschdev/krusch-cascade-router';
      headers['X-Title'] = modelConfig.headers?.['X-Title'] || this.config.openrouterTitle || 'krusch-cascade-router';
    }

    if (modelConfig.headers) {
      Object.assign(headers, modelConfig.headers);
    }

    return { url, headers };
  }

  private async fetchModel(modelConfig: ModelConfig, messages: Message[], options?: ChatOptions): Promise<{ text: string, usage: UsageMetrics }> {
    const provider = modelConfig.provider || 'openai';

    if (provider === 'gemini') {
      return this.fetchGemini(messages, options, modelConfig);
    }

    const { url, headers } = this.getModelEndpoint(modelConfig, false);

    const response = await this.fetchFn(url, {
      method: 'POST',
      headers,
      signal: options?.signal,
      body: JSON.stringify({
        model: modelConfig.model,
        messages,
        stream: false
      })
    });

    if (!response.ok) {
      throw new Error(`${modelConfig.model} HTTP ${response.status}`);
    }

    const data = await response.json();
    const text = data.choices?.[0]?.message?.content || '';
    
    // Extract usage
    const promptTokens = data.usage?.prompt_tokens ?? estimateTokens(messages.map(m => m.content).join('\n'));
    const completionTokens = data.usage?.completion_tokens ?? estimateTokens(text);
    const usage = this.calculateUsage(modelConfig, promptTokens, completionTokens);

    return { text, usage };
  }

  /**
   * Fallback to heavy model.
   */
  private async fetchHeavyModel(messages: Message[], options?: ChatOptions): Promise<{ text: string, usage: UsageMetrics }> {
    return this.fetchModel(this.config.heavyModel!, messages, options);
  }

  private async fetchGemini(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): Promise<{ text: string, usage: UsageMetrics }> {
    const heavyModel = customModel || this.config.heavyModel!;
    const apiKey = heavyModel.apiKey;
    if (!apiKey) throw new Error('Gemini requires an API key');

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
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    
    const promptTokens = data.usageMetadata?.promptTokenCount ?? estimateTokens(messages.map(m => m.content).join('\n'));
    const completionTokens = data.usageMetadata?.candidatesTokenCount ?? estimateTokens(text);
    const usage = this.calculateUsage(heavyModel, promptTokens, completionTokens);

    return { text, usage };
  }

  private async *streamAndEvaluateFastModelGen(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): AsyncGenerator<string, void, unknown> {
    const modelToUse = customModel
      || ((options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : this.config.fastModel!);
    const { url, headers } = this.getModelEndpoint(modelToUse, true);

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
    const seenDeltas: string[] = [];
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
              seenDeltas.push(delta);

              // Check for degenerate repetitive loop
              if (this.detectRepetitiveLoop(seenDeltas)) {
                this.config.onEvent?.('repetition_loop_triggered', { tokenCount, delta });
                controller.abort();
                throw new CascadeTriggeredError('Repetitive token loop detected');
              }

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

  private async *streamHeavyModel(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): AsyncGenerator<string, void, unknown> {
    const heavyModel = customModel || this.config.heavyModel!;
    const provider = heavyModel.provider || 'openai';

    if (provider === 'gemini') {
      yield* this.streamGemini(messages, options, heavyModel);
      return;
    }

    const { url, headers } = this.getModelEndpoint(heavyModel, false);

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

  private async *streamGemini(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): AsyncGenerator<string, void, unknown> {
    const heavyModel = customModel || this.config.heavyModel!;
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

  private calculateUsage(modelConfig: ModelConfig, promptTokens: number, completionTokens: number): UsageMetrics {
    const inRate = modelConfig.costPerMillionInputTokens ?? (modelConfig.provider === 'gemini' ? 0.15 : (modelConfig.url?.includes('localhost') ? 0.0 : 0.15));
    const outRate = modelConfig.costPerMillionOutputTokens ?? (modelConfig.provider === 'gemini' ? 0.60 : (modelConfig.url?.includes('localhost') ? 0.0 : 0.60));

    const promptCost = (promptTokens / 1_000_000) * inRate;
    const completionCost = (completionTokens / 1_000_000) * outRate;
    const estimatedCostUsd = Math.round((promptCost + completionCost) * 1_000_000) / 1_000_000;

    return {
      promptTokens,
      completionTokens,
      totalTokens: promptTokens + completionTokens,
      estimatedCostUsd
    };
  }

  private recordRequestMetrics(routedTo: 'fast' | 'heavy', cascaded: boolean, promptTokens: number, completionTokens: number, actualCostUsd: number) {
    this.metrics.totalRequests++;
    this.metrics.totalPromptTokens += promptTokens;
    this.metrics.totalCompletionTokens += completionTokens;
    this.metrics.estimatedCostUsd += actualCostUsd;

    if (routedTo === 'fast') {
      this.metrics.fastRequests++;
      // Calculate savings vs heavy cloud baseline ($0.15/$0.60 per 1M tokens)
      const cloudHypotheticalCost = ((promptTokens / 1_000_000) * 0.15) + ((completionTokens / 1_000_000) * 0.60);
      this.metrics.estimatedSavingsUsd += Math.max(0, cloudHypotheticalCost - actualCostUsd);
    } else {
      this.metrics.heavyRequests++;
      if (cascaded) {
        this.metrics.cascadedRequests++;
      }
    }
  }
}
