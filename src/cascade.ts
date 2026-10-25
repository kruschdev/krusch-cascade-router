import { 
  isComplexPrompt, 
  pruneText, 
  evaluateComplexityScore, 
  classifySpecialistRole, 
  classifyPreRoute,
  SpecialistRole, 
  Message, 
  ClassifierOptions 
} from './classifier.js';

export interface ModelConfig {
  url?: string;
  apiKey?: string;
  model: string;
  provider?: 'openai' | 'gemini' | 'openrouter';
  costPerMillionInputTokens?: number;  // Default: 0 for local/fast, 2.00 for gemini-4-argon, 0.15 for other cloud
  costPerMillionOutputTokens?: number; // Default: 0 for local/fast, 10.00 for gemini-4-argon, 0.60 for other cloud
  cachedInputDiscountRate?: number;    // Default: 0.95 for Gemini implicit caching (95% discount)
  thinkingLevel?: 'low' | 'medium' | 'high' | 'minimal';
  maxOutputTokens?: number;
  automaticContinuation?: boolean;
  maxContinuationHops?: number;
  headers?: Record<string, string>;
}

export type TelemetryEvent = 
  | 'route_fast' 
  | 'route_burst'
  | 'route_heavy' 
  | 'route_specialist'
  | 'route_l2_semantic'
  | 'cascade_triggered' 
  | 'json_fallback_triggered' 
  | 'repetition_loop_triggered'
  | 'speculative_branch_hedged'
  | 'hedged_branch_error'
  | 'hedge_error'
  | 'entropy_collapse_triggered';

export interface ColabBurstEnvConfig {
  defaultModelUrl?: string;
}

export function resolveColabBurstUrl(envOverride?: Record<string, string | undefined>): string {
  const envSource = envOverride ?? (typeof process !== 'undefined' && typeof process.env !== 'undefined' ? process.env : undefined);
  const candidate = envSource?.COLAB_BURST_URL?.trim();
  return candidate && candidate.length > 0
    ? candidate
    : 'http://127.0.0.1:11435/v1/chat/completions';
}

export const DEFAULT_COLAB_BURST_MODEL: ModelConfig = {
  get url(): string {
    return resolveColabBurstUrl();
  },
  model: 'gpt-oss:120b',
  costPerMillionInputTokens: 0,
  costPerMillionOutputTokens: 0
};


export interface JeanSREGateConfig {
  enabled: boolean;
  sreUrl?: string;
  conversationId?: string;
  projectContext?: string;
  timeoutMs?: number;   // Request timeout in ms (default: 2000)
  cacheTtlMs?: number;  // Optional TTL in ms to cache SRE telemetry across consecutive requests
}

export interface SemanticRouteResult {
  recommendedRole: SpecialistRole;
  targetTier?: 'specialist' | 'heavy' | 'frontier';
  recommendedModel?: string;
  confidence?: number;
  reason?: string;
}

export type SemanticRouterL2 = (
  prompt: string,
  context?: { project?: string; metadata?: Record<string, any> }
) => Promise<SemanticRouteResult | null>;

export interface RouterConfig {
  fastModel?: ModelConfig;
  burstModel?: ModelConfig; // Optional intermediate high-VRAM burst tier (e.g. 96GB G4 Colab tunnel) before heavyModel/OpenRouter
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
  l2Router?: SemanticRouterL2; // Optional L2 Neural Semantic Router
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
  initialPromptTokens?: number;
  cachedTokens?: number;
  thoughtTokens?: number;
  hops?: number;
}

export interface RouterMetrics {
  totalRequests: number;
  fastRequests: number;
  burstRequests?: number;
  heavyRequests: number;
  cascadedRequests: number;
  totalPromptTokens: number;
  totalCompletionTokens: number;
  estimatedCostUsd: number;
  estimatedSavingsUsd: number;
}

export interface CascadeResponse {
  text: string;
  routedTo: 'fast' | 'burst' | 'heavy' | SpecialistRole;
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
  project?: string;
  metadata?: Record<string, any>;
  thinkingLevel?: 'low' | 'medium' | 'high' | 'minimal';
  maxOutputTokens?: number;
  continuationToken?: string;
  automaticContinuation?: boolean;
  maxContinuationHops?: number;
}

export interface ChatJsonOptions extends ChatOptions {
  schema?: Record<string, any>;
  maxJsonRetries?: number;
}

export interface MultiSpecialistRouterOptions {
  openrouterApiKey?: string;
  siteUrl?: string;
  appName?: string;
  customModels?: Partial<Record<SpecialistRole, string | ModelConfig>>;
  burstModel?: ModelConfig | boolean;
  backgroundModel?: ModelConfig;
  cascadeThreshold?: number;
  tokensToEvaluate?: number;
  maxRepetitiveTokens?: number;
  speculativeBranching?: boolean;
  prunePreRouting?: boolean;
  classifier?: ClassifierOptions;
  l2Router?: SemanticRouterL2;
  fetch?: typeof fetch;
  onEvent?: (event: TelemetryEvent, metadata?: Record<string, any>) => void;
  jeanSREGate?: JeanSREGateConfig;
  requireApiKey?: boolean;
}

declare const process: { env?: Record<string, string | undefined> } | undefined;

function getProcessEnv(): Record<string, string | undefined> | undefined {
  const isNode = typeof process !== 'undefined' && typeof process.env !== 'undefined';
  return isNode ? process.env : undefined;
}

/**
 * Validates numeric routing option parameters (VALID-01).
 */
export function validateRoutingOptions(options?: {
  cascadeThreshold?: number;
  tokensToEvaluate?: number;
  maxRepetitiveTokens?: number;
}): void {
  if (!options) return;
  if (options.cascadeThreshold !== undefined) {
    if (
      typeof options.cascadeThreshold !== 'number' ||
      !Number.isFinite(options.cascadeThreshold) ||
      options.cascadeThreshold < 0 ||
      options.cascadeThreshold > 1
    ) {
      throw new TypeError(
        `Invalid cascadeThreshold: expected a finite number in [0, 1], got ${String(options.cascadeThreshold)}`
      );
    }
  }
  if (options.tokensToEvaluate !== undefined) {
    if (
      typeof options.tokensToEvaluate !== 'number' ||
      !Number.isFinite(options.tokensToEvaluate) ||
      options.tokensToEvaluate < 0
    ) {
      throw new TypeError(
        `Invalid tokensToEvaluate: expected a finite non-negative number, got ${String(options.tokensToEvaluate)}`
      );
    }
  }
  if (options.maxRepetitiveTokens !== undefined) {
    if (
      typeof options.maxRepetitiveTokens !== 'number' ||
      !Number.isFinite(options.maxRepetitiveTokens) ||
      options.maxRepetitiveTokens < 0
    ) {
      throw new TypeError(
        `Invalid maxRepetitiveTokens: expected a finite non-negative number, got ${String(options.maxRepetitiveTokens)}`
      );
    }
  }
}

/**
 * Creates a CascadeRouter pre-configured with 5 domain specialist models
 * routing via OpenRouter's unified API layer.
 */
export function createMultiSpecialistRouter(options?: MultiSpecialistRouterOptions): CascadeRouter {
  validateRoutingOptions(options);
  const envVars = getProcessEnv();
  const envApiKey = envVars ? envVars.OPENROUTER_API_KEY : undefined;
  const rawApiKey = options?.openrouterApiKey ?? envApiKey;
  const apiKey = typeof rawApiKey === 'string' && rawApiKey.trim().length > 0 ? rawApiKey.trim() : undefined;
  if (!apiKey && options?.requireApiKey !== false) {
    throw new Error('OPENROUTER_API_KEY is required but not set');
  }
  const referer = options?.siteUrl || 'https://github.com/kruschdev/krusch-cascade-router';
  const title = options?.appName || 'krusch-cascade-router';

  // 5-model specialist empirical pool with published OpenRouter pricing ($/1M tokens)
  const defaultModels: Record<SpecialistRole, { model: string; inputCost: number; outputCost: number }> = {
    general_fast: { model: 'google/gemini-3.1-flash-lite', inputCost: 0.25, outputCost: 1.50 },
    factual_stem: { model: 'deepseek/deepseek-v4-flash', inputCost: 0.14, outputCost: 0.28 },
    code: { model: 'Qwen/Qwen3-Coder-Next', inputCost: 0.07, outputCost: 0.30 },
    reasoning_fast: { model: 'Qwen/Qwen3-Coder-Next', inputCost: 0.07, outputCost: 0.30 },
    reasoning_deep: { model: 'deepseek/deepseek-v4-pro', inputCost: 0.435, outputCost: 0.87 },
    games_spatial: { model: 'Qwen/Qwen3-Coder-Next', inputCost: 0.07, outputCost: 0.30 },
    comprehension_rc: { model: 'qwen/qwen3-235b-a22b-2507', inputCost: 0.071, outputCost: 0.10 }
  };

  const buildRoleConfig = (role: SpecialistRole): ModelConfig => {
    const def = defaultModels[role];
    const userOverride = options?.customModels?.[role];
    if (typeof userOverride === 'string') {
      return {
        model: userOverride,
        provider: 'openrouter',
        apiKey,
        costPerMillionInputTokens: def.inputCost,
        costPerMillionOutputTokens: def.outputCost
      };
    }
    if (userOverride && typeof userOverride === 'object') {
      return {
        provider: 'openrouter',
        apiKey,
        costPerMillionInputTokens: def.inputCost,
        costPerMillionOutputTokens: def.outputCost,
        ...userOverride
      };
    }
    return {
      model: def.model,
      provider: 'openrouter',
      apiKey,
      costPerMillionInputTokens: def.inputCost,
      costPerMillionOutputTokens: def.outputCost
    };
  };

  const specialistModels: Record<SpecialistRole, ModelConfig> = {
    general_fast: buildRoleConfig('general_fast'),
    factual_stem: buildRoleConfig('factual_stem'),
    code: buildRoleConfig('code'),
    reasoning_fast: buildRoleConfig('reasoning_fast'),
    reasoning_deep: buildRoleConfig('reasoning_deep'),
    games_spatial: buildRoleConfig('games_spatial'),
    comprehension_rc: buildRoleConfig('comprehension_rc')
  };

  const resolvedBurstModel: ModelConfig | undefined =
    options?.burstModel === true
      ? { ...DEFAULT_COLAB_BURST_MODEL }
      : options?.burstModel && typeof options.burstModel === 'object'
        ? { ...DEFAULT_COLAB_BURST_MODEL, ...options.burstModel }
        : undefined;

  return new CascadeRouter({
    fastModel: specialistModels.factual_stem,
    burstModel: resolvedBurstModel,
    heavyModel: specialistModels.reasoning_deep,
    backgroundModel: options?.backgroundModel,
    specialistModels,
    openrouterApiKey: apiKey,
    openrouterReferer: referer,
    openrouterTitle: title,
    cascadeThreshold: options?.cascadeThreshold,
    tokensToEvaluate: options?.tokensToEvaluate,
    maxRepetitiveTokens: options?.maxRepetitiveTokens,
    speculativeBranching: options?.speculativeBranching,
    prunePreRouting: options?.prunePreRouting,
    classifier: options?.classifier,
    l2Router: options?.l2Router,
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
 * Scans text for balanced `{...}` or `[...]` candidates, ignoring delimiters inside string literals.
 */
export function tryParseBalancedDelimited<T>(text: string, openChar: '{' | '[', closeChar: '}' | ']'): { ok: true; value: T } | { ok: false } {
  let searchFrom = 0;
  while (searchFrom < text.length) {
    const start = text.indexOf(openChar, searchFrom);
    if (start === -1) break;

    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = -1;

    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (ch === '\\') {
          escaped = true;
        } else if (ch === '"') {
          inString = false;
        }
        continue;
      }

      if (ch === '"') {
        inString = true;
      } else if (ch === openChar) {
        depth++;
      } else if (ch === closeChar) {
        depth--;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }

    if (end !== -1) {
      try {
        return { ok: true, value: JSON.parse(text.slice(start, end + 1)) as T };
      } catch (err) {
        console.warn('Failed JSON fragment parse:', err);
        searchFrom = start + 1;
      }
    } else {
      break;
    }
  }
  return { ok: false };
}

export const parseJsonFragment = tryParseBalancedDelimited;

/**
 * Robust JSON extraction helper that parses raw AI JSON output or repairs markdown/prose-wrapped JSON.
 */
function parseJsonSafe<T = any>(text: string): T {
  if (!text || typeof text !== 'string') {
    throw new Error('Cannot parse empty or non-string AI response as JSON.');
  }

  const trimmed = text.trim();
  let lastParseError: unknown = null;

  // 1. Direct parse attempt (must be a non-null JSON object or array)
  try {
    const direct = JSON.parse(trimmed);
    if (typeof direct === 'object' && direct !== null) {
      return direct as T;
    }
  } catch (err) {
    lastParseError = err;
  }

  // 2. Extract code fences (prioritize explicit ```json/jsonc/json5 fences, then any ``` fence)
  if (trimmed.includes('```')) {
    const jsonFenceRe = /```json(?:c|5)?\s*([\s\S]*?)```/gi;
    let match: RegExpExecArray | null;
    while ((match = jsonFenceRe.exec(trimmed)) !== null) {
      if (match[1]) {
        try {
          const parsed = JSON.parse(match[1].trim());
          if (typeof parsed === 'object' && parsed !== null) {
            return parsed as T;
          }
        } catch (err) {
          lastParseError = err;
        }
      }
    }
    const anyFenceRe = /```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)```/g;
    while ((match = anyFenceRe.exec(trimmed)) !== null) {
      if (match[1]) {
        try {
          const parsed = JSON.parse(match[1].trim());
          if (typeof parsed === 'object' && parsed !== null) {
            return parsed as T;
          }
        } catch (err) {
          lastParseError = err;
        }
      }
    }
  }

  // 3. String-aware balanced object or array extraction in order of first appearance
  const firstBrace = trimmed.indexOf('{');
  const firstBracket = trimmed.indexOf('[');
  const tryArrayFirst = firstBracket !== -1 && (firstBrace === -1 || firstBracket < firstBrace);

  const primary = tryArrayFirst
    ? tryParseBalancedDelimited<T>(trimmed, '[', ']')
    : tryParseBalancedDelimited<T>(trimmed, '{', '}');
  if (primary.ok) return primary.value;

  const secondary = tryArrayFirst
    ? tryParseBalancedDelimited<T>(trimmed, '{', '}')
    : tryParseBalancedDelimited<T>(trimmed, '[', ']');
  if (secondary.ok) return secondary.value;

  const errDetail = lastParseError instanceof Error ? ` (${lastParseError.message})` : '';
  throw new Error(`Failed to parse AI JSON response${errDetail}: ${trimmed.slice(0, 100)}...`);
}

/**
 * Approximate token estimator: ~4 characters per token in English.
 */
function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

function extractMessageContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    let out = '';
    for (let i = 0; i < content.length; i++) {
      const part = content[i];
      if (typeof part === 'string') {
        out += part;
      } else if (part && typeof part.text === 'string') {
        out += part.text;
      }
    }
    return out;
  }
  return content != null ? String(content) : '';
}

/**
 * Joins message contents with newlines without intermediate .filter()/.map() array allocations.
 */
function joinMessagesContent(messages: Message[], roleFilter?: Message['role']): string {
  if (!Array.isArray(messages)) return '';
  const len = messages.length;
  if (len === 0) return '';
  let acc = '';
  let first = true;
  for (let i = 0; i < len; i++) {
    const m = messages[i];
    if (!m || (roleFilter && m.role !== roleFilter)) continue;
    const content = extractMessageContent(m.content);
    if (first) {
      acc = content;
      first = false;
    } else {
      acc += `\n${content}`;
    }
  }
  return acc;
}

function attachJsonMetadata(target: unknown, routedTo: 'fast' | 'burst' | 'heavy' | SpecialistRole, usage?: UsageMetrics): void {
  if (typeof target === 'object' && target !== null) {
    Object.defineProperty(target, '_routedTo', {
      value: routedTo,
      enumerable: false,
      configurable: true,
      writable: true
    });
    Object.defineProperty(target, '_usage', {
      value: usage,
      enumerable: false,
      configurable: true,
      writable: true
    });
  }
}

const VALID_SPECIALIST_ROLES_SET = new Set<SpecialistRole>([
  'general_fast',
  'factual_stem',
  'code',
  'reasoning_fast',
  'reasoning_deep',
  'games_spatial',
  'comprehension_rc'
]);

export class CascadeRouter {
  public static readonly SRE_CACHE_MAX_ENTRIES = 32;
  public static readonly DEFAULT_SRE_CACHE_TTL_MS = 5 * 60_000;
  private config: RouterConfig;
  private fetchFn: typeof fetch;
  private metrics: RouterMetrics;
  /**
   * In-process Jean SRE suggestion cache (ARCH-01 / DX-01):
   * - Entries expire after `cacheTtlMs` (default documented policy: `DEFAULT_SRE_CACHE_TTL_MS = 5 * 60_000` ms / 5 min when enabled).
   * - Bounded to `CascadeRouter.SRE_CACHE_MAX_ENTRIES` (32); expired entries are purged via `evictExpiredCacheEntries()`
   *   before every cache read/write, and the oldest FIFO key is evicted when capacity is reached.
   */
  private sreCache = new Map<string, { reply: string | null; expiresAt: number }>();
  private chatJsonContextMap = new WeakMap<object, { jsonInstruction: string; formattedMessages?: Message[]; promptTokens?: number }>();

  constructor(config: RouterConfig) {
    if (!config.fastModel && !config.specialistModels) {
      throw new Error('RouterConfig requires either fastModel or specialistModels.');
    }
    validateRoutingOptions(config);

    const defaultFast = config.fastModel ?? (config.specialistModels?.factual_stem || config.specialistModels?.general_fast || Object.values(config.specialistModels || {})[0]);
    const defaultHeavy = config.heavyModel ?? (config.specialistModels?.reasoning_deep || config.specialistModels?.reasoning_fast || Object.values(config.specialistModels || {})[0] || defaultFast);

    const validThreshold = typeof config.cascadeThreshold === 'number' && Number.isFinite(config.cascadeThreshold)
      ? Math.min(1, Math.max(0, config.cascadeThreshold))
      : 0.85;
    const validTokensToEvaluate = typeof config.tokensToEvaluate === 'number' && Number.isFinite(config.tokensToEvaluate)
      ? Math.max(1, Math.floor(config.tokensToEvaluate))
      : 5;
    const validMaxRepetitive = typeof config.maxRepetitiveTokens === 'number' && Number.isFinite(config.maxRepetitiveTokens)
      ? Math.max(1, Math.floor(config.maxRepetitiveTokens))
      : 4;

    this.config = {
      ...config,
      fastModel: defaultFast!,
      heavyModel: defaultHeavy!,
      cascadeThreshold: validThreshold,
      tokensToEvaluate: validTokensToEvaluate,
      maxRepetitiveTokens: validMaxRepetitive
    };
    this.fetchFn = config.fetch ?? (typeof globalThis !== 'undefined' ? globalThis.fetch : fetch);
    if (!this.fetchFn) {
      throw new Error('A global fetch API is required, or a custom fetch implementation must be provided in RouterConfig.');
    }

    this.metrics = {
      totalRequests: 0,
      fastRequests: 0,
      burstRequests: 0,
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
      burstRequests: 0,
      heavyRequests: 0,
      cascadedRequests: 0,
      totalPromptTokens: 0,
      totalCompletionTokens: 0,
      estimatedCostUsd: 0,
      estimatedSavingsUsd: 0
    };
  }

  /**
   * Complete a chat request, routing automatically between fast edge, optional burst tier (Colab G4), and heavy cloud models,
   * or across the 5-model specialist pool if configured.
   */
  async chat(
    messages: Message[] | string, 
    systemPromptOrOptions?: string | ChatOptions, 
    options?: ChatOptions
  ): Promise<CascadeResponse> {
    const actualSystemPrompt = typeof systemPromptOrOptions === 'string' ? systemPromptOrOptions : undefined;
    const actualOptions = typeof systemPromptOrOptions === 'object' ? systemPromptOrOptions : options;
    const formattedMessages = this.formatMessages(messages, actualSystemPrompt, actualOptions);

    // Fetch SRE suggestions if gate is enabled
    if (this.config.jeanSREGate?.enabled) {
      const sreProject = actualOptions?.project || this.config.jeanSREGate.projectContext;
      const sreSuggestions = await this.fetchJeanSuggestions(sreProject);
      if (sreSuggestions) {
        formattedMessages.push({
          role: 'system',
          content: `[JEAN SRE TELEMETRY & RECOMMENDATIONS]\n${sreSuggestions}\n\nINSTRUCTION: Jean SRE has detected these fleet/system anomalies or suggestions. As the agent-in-the-loop, you MUST present these to the human user for review, analyze their risks/benefits, and wait for the user's explicit approval before proposing or executing any commands or modifications.`
        });
      }
    }

    // Evaluate routing heuristics on user/caller messages before injecting synthetic chatJson instructions
    const promptText = joinMessagesContent(formattedMessages);
    const jsonCtx = actualOptions ? this.chatJsonContextMap.get(actualOptions) : undefined;
    if (jsonCtx) {
      if (actualSystemPrompt && formattedMessages[0]?.role === 'system') {
        formattedMessages[0] = {
          role: 'system',
          content: `${formattedMessages[0].content}\n\n${jsonCtx.jsonInstruction}`
        };
      } else {
        formattedMessages.unshift({
          role: 'system',
          content: jsonCtx.jsonInstruction
        });
      }
    }
    const promptTokens = estimateTokens(jsonCtx ? joinMessagesContent(formattedMessages) : promptText);
    if (jsonCtx) {
      jsonCtx.formattedMessages = formattedMessages;
      jsonCtx.promptTokens = promptTokens;
    }

    // 1. Predictive Classifier Options (formattedMessages already pruned if prunePreRouting was enabled)
    const classifierOpts: ClassifierOptions = {
      ...this.config.classifier,
      prunePreRouting: false
    };

    // --- Path A: Multi-Specialist Pool Routing ---
    if (this.config.specialistModels && Object.keys(this.config.specialistModels).length > 0) {
      const preRoute = classifyPreRoute(promptText, classifierOpts);
      const complexityScore = preRoute.complexityScore;
      const isBorderline = complexityScore >= 0.25 && complexityScore <= 0.70;
      const shouldHedge = (actualOptions?.speculativeBranching ?? this.config.speculativeBranching) && isBorderline;
      let targetRole: SpecialistRole = preRoute.role ?? 'factual_stem';

      // If L0 gate missed (delegate_to_l2) and L2 semantic router is configured, query L2
      let l2EscalatedToHeavy = false;
      if (!preRoute.isFastPath && this.config.l2Router) {
        try {
          const l2Prompt = joinMessagesContent(formattedMessages, 'user') || promptText;
          const l2Result = await this.config.l2Router(l2Prompt, {
            project: actualOptions?.project,
            metadata: actualOptions?.metadata
          });
          const candidateRole = typeof l2Result?.recommendedRole === 'string'
            ? (l2Result.recommendedRole.trim().toLowerCase() as SpecialistRole)
            : undefined;
          if (candidateRole && VALID_SPECIALIST_ROLES_SET.has(candidateRole)) {
            targetRole = candidateRole;
            if (l2Result!.targetTier === 'heavy' || l2Result!.targetTier === 'frontier') {
              l2EscalatedToHeavy = true;
            }
            this.config.onEvent?.('route_l2_semantic', {
              role: targetRole,
              tier: l2Result!.targetTier,
              recommendedModel: l2Result!.recommendedModel,
              confidence: l2Result!.confidence,
              reason: l2Result!.reason
            });
          }
        } catch {
          // Graceful fallback to default role
        }
      }

      let targetModel = this.config.specialistModels[targetRole] || this.config.fastModel!;
      const heavyModel = this.config.specialistModels.reasoning_deep || this.config.heavyModel!;

      // Domain specialists (code, games_spatial, comprehension_rc) retain their specialization;
      // only evaluate isComplexPrompt when needed for escalation or telemetry
      const isDomainSpecialist = targetRole === 'code' || targetRole === 'games_spatial' || targetRole === 'comprehension_rc';
      const needsComplexCheck = !isDomainSpecialist && targetRole !== 'reasoning_deep' && targetRole !== 'reasoning_fast';
      const isComplex = (needsComplexCheck || this.config.onEvent) ? isComplexPrompt(promptText, classifierOpts) : false;

      if (l2EscalatedToHeavy || (needsComplexCheck && isComplex)) {
        targetRole = 'reasoning_deep';
        targetModel = heavyModel;
      }

      const isLowPriority = actualOptions?.speedPriority === 'low' || actualOptions?.urgency === 'low';
      if (targetRole === 'reasoning_deep' && this.config.burstModel && !isLowPriority) {
        const res = await this.fetchEscalatedModel(
          formattedMessages,
          actualOptions,
          l2EscalatedToHeavy ? 'l2_semantic_escalation' : 'classifier_heuristic',
          { role: targetRole, complexityScore, isComplex },
          heavyModel
        );
        this.recordRequestMetrics(res.routedTo, false, res.usage.promptTokens, res.usage.completionTokens, res.usage.estimatedCostUsd);
        return { text: res.text, routedTo: res.routedTo === 'burst' ? 'burst' : 'reasoning_deep', aborted: false, usage: res.usage, model: res.model };
      }

      const modelToUse = isLowPriority && this.config.backgroundModel
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
      let onExternalAbortHedge: (() => void) | null = null;
      const hedgeTargetModel = this.config.burstModel || heavyModel;

      if (shouldHedge && targetRole !== 'reasoning_deep') {
        hedgedController = new AbortController();
        onExternalAbortHedge = () => hedgedController?.abort();
        if (actualOptions?.signal) {
          actualOptions.signal.addEventListener('abort', onExternalAbortHedge, { once: true });
          if (actualOptions.signal.aborted) hedgedController.abort();
        }
        this.config.onEvent?.('speculative_branch_hedged', { complexityScore });
        hedgedHeavyPromise = this.fetchModel(hedgeTargetModel, formattedMessages, { ...actualOptions, signal: hedgedController.signal });
        // Suppress unhandled rejection if fast specialist wins and hedgedController is aborted
        hedgedHeavyPromise.catch(() => {});
      }

      try {
        const { text, usage } = await this.fetchModel(modelToUse, formattedMessages, actualOptions);
        if (hedgedController) hedgedController.abort();

        const isHeavyRole = targetRole === 'reasoning_deep';
        this.recordRequestMetrics(isHeavyRole ? 'heavy' : 'fast', false, usage.promptTokens, usage.completionTokens, usage.estimatedCostUsd);
        return { text, routedTo: targetRole, aborted: false, usage, model: modelToUse.model };
      } catch (err) {
        if (actualOptions?.signal?.aborted) {
          if (hedgedController) hedgedController.abort();
          throw err;
        }
        if (hedgedHeavyPromise) {
          try {
            const { text, usage } = await hedgedHeavyPromise;
            if (this.config.burstModel) {
              this.config.onEvent?.('route_burst', { reason: 'specialist_model_error', role: targetRole, model: this.config.burstModel.model, error: (err as Error).message });
              this.recordRequestMetrics('burst', true, usage.promptTokens, usage.completionTokens, usage.estimatedCostUsd);
              return { text, routedTo: 'burst', aborted: true, usage, model: this.config.burstModel.model };
            }
            this.config.onEvent?.('route_heavy', { reason: 'specialist_model_error', role: targetRole, error: (err as Error).message });
            this.recordRequestMetrics('heavy', true, usage.promptTokens, usage.completionTokens, usage.estimatedCostUsd);
            return { text, routedTo: 'reasoning_deep', aborted: true, usage, model: heavyModel.model };
          } catch (hedgeErr) {
            this.config.onEvent?.('hedged_branch_error', {
              role: targetRole,
              error: (hedgeErr as Error)?.message || String(hedgeErr)
            });
          }
        }
        const res = await this.fetchEscalatedModel(
          formattedMessages,
          actualOptions,
          'specialist_model_error',
          { role: targetRole, error: (err as Error).message },
          heavyModel
        );
        this.recordRequestMetrics(res.routedTo, true, res.usage.promptTokens, res.usage.completionTokens, res.usage.estimatedCostUsd);
        return { text: res.text, routedTo: res.routedTo === 'burst' ? 'burst' : 'reasoning_deep', aborted: true, usage: res.usage, model: res.model };
      } finally {
        if (actualOptions?.signal && onExternalAbortHedge) {
          actualOptions.signal.removeEventListener('abort', onExternalAbortHedge);
        }
      }
    }

    // --- Path B: Standard 2-Tier (+ Optional Burst Tier) Cascade ---
    const isComplex = isComplexPrompt(promptText, classifierOpts);
    if (isComplex) {
      // Bypass fast model entirely -> escalate to burstModel first (if configured), then heavyModel
      const res = await this.fetchEscalatedModel(formattedMessages, actualOptions, 'classifier_heuristic');
      this.recordRequestMetrics(res.routedTo, false, res.usage.promptTokens, res.usage.completionTokens, res.usage.estimatedCostUsd);
      return { text: res.text, routedTo: res.routedTo, aborted: false, usage: res.usage, model: res.model };
    }

    const speculativeEnabled = Boolean(actualOptions?.speculativeBranching ?? this.config.speculativeBranching);
    const complexityScore = speculativeEnabled ? evaluateComplexityScore(promptText, classifierOpts) : 0;
    const isBorderline = speculativeEnabled && complexityScore >= 0.25 && complexityScore <= 0.70;
    const shouldHedge = isBorderline;

    // Speculative Parallel Hedging
    // For borderline queries [0.25, 0.70], hedge burst/heavy model in parallel to mask cascade latency
    let hedgedController: AbortController | null = null;
    let hedgedHeavyPromise: Promise<{ text: string; usage: UsageMetrics }> | null = null;
    let onExternalAbortHedge: (() => void) | null = null;
    const hedgeTargetModel = this.config.burstModel || this.config.heavyModel!;

    if (shouldHedge) {
      hedgedController = new AbortController();
      onExternalAbortHedge = () => hedgedController?.abort();
      if (actualOptions?.signal) {
        actualOptions.signal.addEventListener('abort', onExternalAbortHedge, { once: true });
        if (actualOptions.signal.aborted) hedgedController.abort();
      }
      this.config.onEvent?.('speculative_branch_hedged', { complexityScore });
      hedgedHeavyPromise = this.fetchModel(hedgeTargetModel, formattedMessages, { ...actualOptions, signal: hedgedController.signal });
      // Suppress unhandled rejection if fast model succeeds and hedgedController is aborted
      hedgedHeavyPromise.catch(() => {});
    }

    // 2. Try fast model with Speculative Cascade
    try {
      const fastResult = await this.streamAndEvaluateFastModel(formattedMessages, actualOptions);
      if (actualOptions?.signal?.aborted) {
        if (hedgedController) hedgedController.abort();
        throw actualOptions.signal.reason ?? new Error('AbortError: The operation was aborted');
      }
      if (fastResult.aborted) {
        if (hedgedHeavyPromise) {
          try {
            const heavyRes = await hedgedHeavyPromise;
            if (this.config.burstModel) {
              this.config.onEvent?.('route_burst', { reason: 'cascade_fallback', model: this.config.burstModel.model });
              this.recordRequestMetrics('burst', true, heavyRes.usage.promptTokens, heavyRes.usage.completionTokens, heavyRes.usage.estimatedCostUsd);
              return { text: heavyRes.text, routedTo: 'burst', aborted: true, usage: heavyRes.usage, model: this.config.burstModel.model };
            }
            this.config.onEvent?.('route_heavy', { reason: 'cascade_fallback' });
            this.recordRequestMetrics('heavy', true, heavyRes.usage.promptTokens, heavyRes.usage.completionTokens, heavyRes.usage.estimatedCostUsd);
            return { text: heavyRes.text, routedTo: 'heavy', aborted: true, usage: heavyRes.usage, model: this.config.heavyModel!.model };
          } catch (_) {
            if (actualOptions?.signal?.aborted) throw _;
          }
        }
        const res = await this.fetchEscalatedModel(formattedMessages, actualOptions, 'cascade_fallback');
        this.recordRequestMetrics(res.routedTo, true, res.usage.promptTokens, res.usage.completionTokens, res.usage.estimatedCostUsd);
        return { text: res.text, routedTo: res.routedTo, aborted: true, usage: res.usage, model: res.model };
      }

      // Fast model succeeded: cleanly abort the speculative heavy hedge if active
      if (hedgedController) {
        hedgedController.abort();
      }

      const modelToUse = (actualOptions?.speedPriority === 'low' || actualOptions?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : this.config.fastModel!;

      const usage = fastResult.usage ?? this.calculateUsage(modelToUse, promptTokens, estimateTokens(fastResult.text));

      this.config.onEvent?.('route_fast', { reason: 'high_confidence', model: modelToUse.model, speedPriority: actualOptions?.speedPriority || 'normal' });
      this.recordRequestMetrics('fast', false, usage.promptTokens, usage.completionTokens, usage.estimatedCostUsd);

      return { text: fastResult.text, routedTo: 'fast', aborted: false, usage, model: modelToUse.model };
    } catch (err) {
      if (actualOptions?.signal?.aborted) {
        if (hedgedController) hedgedController.abort();
        throw err;
      }
      if (hedgedHeavyPromise) {
        try {
          const { text, usage } = await hedgedHeavyPromise;
          if (this.config.burstModel) {
            this.config.onEvent?.('route_burst', { reason: 'fast_model_error', model: this.config.burstModel.model, error: (err as Error).message });
            this.recordRequestMetrics('burst', true, usage.promptTokens, usage.completionTokens, usage.estimatedCostUsd);
            return { text, routedTo: 'burst', aborted: true, usage, model: this.config.burstModel.model };
          }
          this.config.onEvent?.('route_heavy', { reason: 'fast_model_error', error: (err as Error).message });
          this.recordRequestMetrics('heavy', true, usage.promptTokens, usage.completionTokens, usage.estimatedCostUsd);
          return { text, routedTo: 'heavy', aborted: true, usage, model: this.config.heavyModel!.model };
        } catch (hedgeErr) {
          this.config.onEvent?.('hedge_error', { error: (hedgeErr as Error).message });
          throw hedgeErr;
        }
      }
      const res = await this.fetchEscalatedModel(formattedMessages, actualOptions, 'fast_model_error', { error: (err as Error).message });
      this.recordRequestMetrics(res.routedTo, true, res.usage.promptTokens, res.usage.completionTokens, res.usage.estimatedCostUsd);
      return { text: res.text, routedTo: res.routedTo, aborted: true, usage: res.usage, model: res.model };
    } finally {
      if (actualOptions?.signal && onExternalAbortHedge) {
        actualOptions.signal.removeEventListener('abort', onExternalAbortHedge);
      }
    }
  }

  /**
   * Complete a chat request and parse the output as structured JSON.
   * If the fast/specialist model produces malformed JSON, automatically cascades to burst/heavy model.
   */
  async chatJson<T = any>(
    messages: Message[] | string,
    systemPromptOrOptions?: string | ChatJsonOptions,
    options?: ChatJsonOptions
  ): Promise<T & { _routedTo?: 'fast' | 'burst' | 'heavy' | SpecialistRole; _usage?: UsageMetrics }> {
    const actualSystemPrompt = typeof systemPromptOrOptions === 'string' ? systemPromptOrOptions : undefined;
    const actualOptions = typeof systemPromptOrOptions === 'object' ? systemPromptOrOptions : options;

    let jsonInstruction = 'IMPORTANT: You must respond ONLY with a valid JSON object or array. Do not include markdown preamble, commentary, or backticks.';
    if (actualOptions?.schema) {
      const schemaSpec = JSON.stringify(actualOptions.schema).replace(/"/g, "'");
      jsonInstruction += `\nRequired JSON Schema: ${schemaSpec}`;
    }

    const internalOpts: ChatJsonOptions = { ...actualOptions };
    this.chatJsonContextMap.set(internalOpts, { jsonInstruction });

    const response = await this.chat(messages, actualSystemPrompt, internalOpts);

    try {
      const parsed = parseJsonSafe<T>(response.text);
      attachJsonMetadata(parsed, response.routedTo, response.usage);
      return parsed as any;
    } catch (parseErr) {
      const wasHeavy = response.routedTo === 'heavy' || response.routedTo === 'reasoning_deep';
      const maxRetries = Math.max(1, actualOptions?.maxJsonRetries ?? 1);
      const remainingAttempts = wasHeavy ? (maxRetries - 1) : maxRetries;

      if (remainingAttempts <= 0) {
        throw parseErr;
      }

      if (!wasHeavy) {
        this.config.onEvent?.('json_fallback_triggered', { error: (parseErr as Error).message, fromRole: response.routedTo });
      }

      const jsonCtx = this.chatJsonContextMap.get(internalOpts);
      const formatted = jsonCtx?.formattedMessages ?? this.formatMessages(
        messages,
        actualSystemPrompt ? `${actualSystemPrompt}\n\n${jsonInstruction}` : jsonInstruction,
        actualOptions
      );
      const heavyModel = this.config.specialistModels?.reasoning_deep || this.config.heavyModel!;
      let lastErr: unknown = parseErr;
      let recordedHeavyCascade = false;

      // Try burstModel first if configured and the initial attempt wasn't already burst/heavy
      if (!wasHeavy && response.routedTo !== 'burst' && this.config.burstModel) {
        if (actualOptions?.signal?.aborted) {
          throw lastErr;
        }
        try {
          this.config.onEvent?.('route_burst', { reason: 'json_fallback', fromRole: response.routedTo, model: this.config.burstModel.model });
          const { text: burstText, usage: burstUsage } = await this.fetchModel(this.config.burstModel, formatted, actualOptions);
          recordedHeavyCascade = true;
          if (this.metrics.fastRequests > 0) this.metrics.fastRequests--;
          if (this.metrics.totalRequests > 0) this.metrics.totalRequests--;
          if (response.usage) {
            const cloudHypotheticalCost = ((response.usage.promptTokens / 1_000_000) * 0.15) + ((response.usage.completionTokens / 1_000_000) * 0.60);
            const falseSavings = Math.max(0, cloudHypotheticalCost - response.usage.estimatedCostUsd);
            this.metrics.estimatedSavingsUsd = Math.max(0, this.metrics.estimatedSavingsUsd - falseSavings);
          }
          this.recordRequestMetrics('burst', true, burstUsage.promptTokens, burstUsage.completionTokens, burstUsage.estimatedCostUsd);
          const parsedBurst = parseJsonSafe<T>(burstText);
          attachJsonMetadata(parsedBurst, 'burst', burstUsage);
          return parsedBurst as any;
        } catch (burstErr) {
          lastErr = burstErr;
          if (actualOptions?.signal?.aborted) {
            throw burstErr;
          }
          this.config.onEvent?.('route_heavy', {
            reason: 'burst_model_error',
            fallbackFrom: this.config.burstModel.model,
            error: (burstErr as Error).message
          });
        }
      }

      for (let attempt = 0; attempt < remainingAttempts; attempt++) {
        if (actualOptions?.signal?.aborted) {
          throw lastErr;
        }
        try {
          const { text, usage } = await this.fetchModel(heavyModel, formatted, actualOptions);
          if (!wasHeavy && !recordedHeavyCascade) {
            recordedHeavyCascade = true;
            // Reclassify the initial fast/burst attempt so metrics reflect 1 cascaded heavy request
            if (response.routedTo === 'burst') {
              if ((this.metrics.burstRequests ?? 0) > 0) this.metrics.burstRequests!--;
            } else if (this.metrics.fastRequests > 0) {
              this.metrics.fastRequests--;
            }
            if (this.metrics.totalRequests > 0) this.metrics.totalRequests--;
            if (response.usage) {
              const cloudHypotheticalCost = ((response.usage.promptTokens / 1_000_000) * 0.15) + ((response.usage.completionTokens / 1_000_000) * 0.60);
              const falseSavings = Math.max(0, cloudHypotheticalCost - response.usage.estimatedCostUsd);
              this.metrics.estimatedSavingsUsd = Math.max(0, this.metrics.estimatedSavingsUsd - falseSavings);
            }
            this.recordRequestMetrics('heavy', true, usage.promptTokens, usage.completionTokens, usage.estimatedCostUsd);
          } else {
            this.metrics.totalPromptTokens += usage.promptTokens;
            this.metrics.totalCompletionTokens += usage.completionTokens;
            this.metrics.estimatedCostUsd += usage.estimatedCostUsd;
          }
          const parsedHeavy = parseJsonSafe<T>(text);
          attachJsonMetadata(parsedHeavy, 'heavy', usage);
          return parsedHeavy as any;
        } catch (retryErr) {
          lastErr = retryErr;
          if (actualOptions?.signal?.aborted) {
            throw retryErr;
          }
        }
      }
      throw lastErr;
    }
  }

  /**
   * Stream a chat request, yielding text chunks in real-time.
   * Speculatively buffers the first N tokens from the fast model.
   * If confidence dips below threshold or a repetitive loop is detected, silent abort occurs and falls back to burst/heavy.
   */
  async *stream(
    messages: Message[] | string, 
    systemPromptOrOptions?: string | ChatOptions, 
    options?: ChatOptions
  ): AsyncGenerator<string, void, unknown> {
    const actualSystemPrompt = typeof systemPromptOrOptions === 'string' ? systemPromptOrOptions : undefined;
    const actualOptions = typeof systemPromptOrOptions === 'object' ? systemPromptOrOptions : options;
    const formattedMessages = this.formatMessages(messages, actualSystemPrompt, actualOptions);

    // Fetch SRE suggestions if gate is enabled
    if (this.config.jeanSREGate?.enabled) {
      const sreProject = actualOptions?.project || this.config.jeanSREGate.projectContext;
      const sreSuggestions = await this.fetchJeanSuggestions(sreProject);
      if (sreSuggestions) {
        formattedMessages.push({
          role: 'system',
          content: `[JEAN SRE TELEMETRY & RECOMMENDATIONS]\n${sreSuggestions}\n\nINSTRUCTION: Jean SRE has detected these fleet/system anomalies or suggestions. As the agent-in-the-loop, you MUST present these to the human user for review, analyze their risks/benefits, and wait for the user's explicit approval before proposing or executing any commands or modifications.`
        });
      }
    }

    const promptText = joinMessagesContent(formattedMessages);
    const promptTokens = estimateTokens(promptText);

    // 1. Predictive Classifier (formattedMessages already pruned if prunePreRouting was enabled)
    const classifierOpts: ClassifierOptions = {
      ...this.config.classifier,
      prunePreRouting: false
    };

    // --- Path A: Multi-Specialist Pool Streaming ---
    if (this.config.specialistModels && Object.keys(this.config.specialistModels).length > 0) {
      const preRoute = classifyPreRoute(promptText, classifierOpts);
      let targetRole: SpecialistRole = preRoute.role ?? 'factual_stem';
      let l2EscalatedToHeavy = false;

      // If L0 gate missed (delegate_to_l2) and L2 semantic router is configured, query L2
      if (!preRoute.isFastPath && this.config.l2Router) {
        try {
          const l2Prompt = joinMessagesContent(formattedMessages, 'user') || promptText;
          const l2Result = await this.config.l2Router(l2Prompt, {
            project: actualOptions?.project,
            metadata: actualOptions?.metadata
          });
          const candidateRole = typeof l2Result?.recommendedRole === 'string'
            ? (l2Result.recommendedRole.trim().toLowerCase() as SpecialistRole)
            : undefined;
          if (candidateRole && VALID_SPECIALIST_ROLES_SET.has(candidateRole)) {
            targetRole = candidateRole;
            if (l2Result!.targetTier === 'heavy' || l2Result!.targetTier === 'frontier') {
              l2EscalatedToHeavy = true;
            }
            this.config.onEvent?.('route_l2_semantic', {
              role: targetRole,
              tier: l2Result!.targetTier,
              recommendedModel: l2Result!.recommendedModel,
              confidence: l2Result!.confidence,
              reason: l2Result!.reason
            });
          }
        } catch {
          // Graceful fallback to default role
        }
      }

      let targetModel = this.config.specialistModels[targetRole] || this.config.fastModel!;
      const heavyModel = this.config.specialistModels.reasoning_deep || this.config.heavyModel!;

      // Domain specialists (code, games_spatial, comprehension_rc) should retain their specialization
      const isDomainSpecialist = targetRole === 'code' || targetRole === 'games_spatial' || targetRole === 'comprehension_rc';
      const needsComplexCheck = !isDomainSpecialist && targetRole !== 'reasoning_deep' && targetRole !== 'reasoning_fast';
      const isComplex = (needsComplexCheck || this.config.onEvent) ? isComplexPrompt(promptText, classifierOpts) : false;

      if (l2EscalatedToHeavy || (needsComplexCheck && isComplex)) {
        targetRole = 'reasoning_deep';
        targetModel = heavyModel;
      }

      const modelToUse = (actualOptions?.speedPriority === 'low' || actualOptions?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : targetModel;

      this.config.onEvent?.('route_specialist', {
        role: targetRole,
        model: modelToUse.model,
        complexityScore: preRoute.complexityScore,
        isComplex
      });

      if (targetRole === 'reasoning_deep') {
        yield* this.streamEscalatedModel(
          formattedMessages,
          actualOptions,
          heavyModel,
          false,
          promptTokens,
          l2EscalatedToHeavy ? 'l2_semantic_escalation' : 'classifier_heuristic',
          false
        );
        return;
      }

      try {
        yield* this.streamWithMetrics(
          this.streamAndEvaluateFastModelGen(formattedMessages, actualOptions, modelToUse),
          modelToUse,
          'fast',
          false,
          promptTokens
        );
      } catch (err) {
        if (actualOptions?.signal?.aborted) {
          throw err;
        }
        const reason = err instanceof CascadeTriggeredError ? 'cascade_fallback' : 'fast_model_error';
        const extraMeta = err instanceof CascadeTriggeredError
          ? { fallbackFrom: targetRole }
          : { fallbackFrom: targetRole, error: (err as Error).message };
        yield* this.streamEscalatedModel(
          formattedMessages,
          actualOptions,
          heavyModel,
          true,
          promptTokens,
          reason,
          true,
          extraMeta
        );
      }
      return;
    }

    // --- Path B: Standard 2-Tier (+ Optional Burst Tier) Cascade Streaming ---
    const isComplex = isComplexPrompt(promptText, classifierOpts);
    const heavyModel = this.config.heavyModel!;
    if (isComplex) {
      yield* this.streamEscalatedModel(
        formattedMessages,
        actualOptions,
        heavyModel,
        false,
        promptTokens,
        'classifier_heuristic',
        true
      );
      return;
    }

    const fastModelToUse = (actualOptions?.speedPriority === 'low' || actualOptions?.urgency === 'low') && this.config.backgroundModel
      ? this.config.backgroundModel
      : this.config.fastModel!;

    // 2. Try fast model streaming with Speculative Cascade fallback
    try {
      yield* this.streamWithMetrics(
        this.streamAndEvaluateFastModelGen(formattedMessages, actualOptions, fastModelToUse),
        fastModelToUse,
        'fast',
        false,
        promptTokens
      );
    } catch (err) {
      if (actualOptions?.signal?.aborted) {
        throw err;
      }
      const reason = err instanceof CascadeTriggeredError ? 'cascade_fallback' : 'fast_model_error';
      const extraMeta = err instanceof CascadeTriggeredError ? undefined : { error: (err as Error).message };
      yield* this.streamEscalatedModel(
        formattedMessages,
        actualOptions,
        heavyModel,
        true,
        promptTokens,
        reason,
        true,
        extraMeta
      );
    }
  }

  private async *streamEscalatedModel(
    messages: Message[],
    options: ChatOptions | undefined,
    heavyModel: ModelConfig,
    cascaded: boolean,
    promptTokens: number,
    reason: string,
    emitHeavyEvent: boolean,
    extraMeta?: Record<string, any>
  ): AsyncGenerator<string, void, unknown> {
    if (this.config.burstModel) {
      this.config.onEvent?.('route_burst', { reason, model: this.config.burstModel.model, ...extraMeta });
      try {
        yield* this.streamWithMetrics(
          this.streamHeavyModel(messages, options, this.config.burstModel),
          this.config.burstModel,
          'burst',
          cascaded,
          promptTokens
        );
        return;
      } catch (burstErr) {
        if (options?.signal?.aborted) {
          throw burstErr;
        }
        this.config.onEvent?.('route_heavy', {
          reason: 'burst_model_error',
          fallbackFrom: this.config.burstModel.model,
          originalReason: reason,
          error: (burstErr as Error).message,
          ...extraMeta
        });
        yield* this.streamWithMetrics(
          this.streamHeavyModel(messages, options, heavyModel),
          heavyModel,
          'heavy',
          true,
          promptTokens
        );
        return;
      }
    }

    if (emitHeavyEvent) {
      this.config.onEvent?.('route_heavy', { reason, ...extraMeta });
    }
    yield* this.streamWithMetrics(
      this.streamHeavyModel(messages, options, heavyModel),
      heavyModel,
      'heavy',
      cascaded,
      promptTokens
    );
  }

  private async *streamWithMetrics(
    gen: AsyncGenerator<string, void, unknown>,
    modelConfig: ModelConfig,
    routedTo: 'fast' | 'burst' | 'heavy',
    cascaded: boolean,
    promptTokens: number
  ): AsyncGenerator<string, void, unknown> {
    let streamedChars = 0;
    let failed = false;
    try {
      for await (const chunk of gen) {
        streamedChars += chunk.length;
        yield chunk;
      }
    } catch (err) {
      failed = true;
      throw err;
    } finally {
      if (!failed) {
        const completionTokens = Math.ceil(streamedChars / 4);
        const usage = this.calculateUsage(modelConfig, promptTokens, completionTokens);
        this.recordRequestMetrics(routedTo, cascaded, promptTokens, completionTokens, usage.estimatedCostUsd);
      }
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
    } else if (Array.isArray(messages)) {
      for (const m of messages) {
        if (!m) continue;
        const rawContent = extractMessageContent(m.content);
        msgs.push({
          role: m.role || 'user',
          content: shouldPrune ? pruneText(rawContent) : rawContent
        });
      }
    }
    return msgs;
  }

  /**
   * Helper: Detects repetitive token degenerate loops and reasoning entropy collapse
   * during speculative buffer evaluation without per-token array/Set allocations.
   */
  private detectRepetitiveLoop(tokens: string[]): 'repetition' | 'entropy_collapse' | null {
    const len = tokens.length;
    const maxRep = this.config.maxRepetitiveTokens || 4;
    if (len < maxRep) return null;

    // 1. Single token repetition check
    const lastToken = tokens[len - 1].trim();
    if (lastToken) {
      let identicalCount = 0;
      for (let i = len - 1; i >= 0; i--) {
        if (tokens[i].trim() === lastToken) {
          identicalCount++;
          if (identicalCount >= maxRep) return 'repetition';
        } else {
          break;
        }
      }
    }

    // 2. 2-gram cyclic repetition check (e.g. A, B, A, B, A, B)
    if (len >= 6) {
      const t1 = tokens[len - 2].trim();
      const t2 = tokens[len - 1].trim();
      if (t1 && t2 && t1 !== t2) {
        if (
          tokens[len - 4].trim() === t1 &&
          tokens[len - 3].trim() === t2 &&
          tokens[len - 6].trim() === t1 &&
          tokens[len - 5].trim() === t2
        ) {
          return 'repetition';
        }
      }
    }

    // 3. 3-gram cyclic repetition check (e.g. A, B, C, A, B, C, A, B, C)
    if (len >= 9) {
      const t1 = tokens[len - 3].trim();
      const t2 = tokens[len - 2].trim();
      const t3 = tokens[len - 1].trim();
      if (t1 && t2 && t3) {
        if (
          tokens[len - 6].trim() === t1 &&
          tokens[len - 5].trim() === t2 &&
          tokens[len - 4].trim() === t3 &&
          tokens[len - 9].trim() === t1 &&
          tokens[len - 8].trim() === t2 &&
          tokens[len - 7].trim() === t3
        ) {
          return 'repetition';
        }
      }
    }

    // 4. Sliding-window unique token entropy collapse (last 10 tokens, evaluated in-place)
    if (len >= 10) {
      let nonEmptyCount = 0;
      let word1 = '';
      let word2 = '';
      let distinctCount = 0;

      for (let i = len - 10; i < len; i++) {
        const w = tokens[i].trim().toLowerCase();
        if (!w) continue;
        nonEmptyCount++;
        if (distinctCount === 0) {
          word1 = w;
          distinctCount = 1;
        } else if (w !== word1) {
          if (distinctCount === 1) {
            word2 = w;
            distinctCount = 2;
          } else if (w !== word2) {
            distinctCount = 3;
            break; // Fast exit: more than 2 distinct words
          }
        }
      }

      if (nonEmptyCount >= 8 && distinctCount <= 2) {
        return 'entropy_collapse';
      }
    }

    return null;
  }

  private extractSseDataPayload(line: string): string | null {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return null;
    const payload = trimmed.slice(5).trim();
    if (!payload || payload === '[DONE]') return null;
    return payload;
  }

  private extractLinearProbsFromChoice(choice: any): number[] {
    const out: number[] = [];
    const content = choice?.logprobs?.content;
    if (Array.isArray(content) && content.length > 0) {
      for (const lp of content) {
        if (lp && typeof lp === 'object' && 'logprob' in lp) {
          const raw = lp.logprob;
          if (typeof raw === 'number') {
            const prob = Math.exp(raw);
            out.push(Number.isFinite(prob) ? Math.min(1, prob) : 0);
          } else if (typeof raw === 'string') {
            const prob = Math.exp(Number(raw));
            out.push(Number.isFinite(prob) ? Math.min(1, prob) : 0);
          } else if (raw === null) {
            // JSON.stringify serializes NaN and -Infinity (Math.log(0)) as null
            out.push(0);
          }
        }
      }
      return out;
    }
    const tokenLogprobs = choice?.logprobs?.token_logprobs;
    if (Array.isArray(tokenLogprobs) && tokenLogprobs.length > 0) {
      for (const lp of tokenLogprobs) {
        if (typeof lp === 'number') {
          const prob = Math.exp(lp);
          out.push(Number.isFinite(prob) ? Math.min(1, prob) : 0);
        } else if (typeof lp === 'string') {
          const prob = Math.exp(Number(lp));
          out.push(Number.isFinite(prob) ? Math.min(1, prob) : 0);
        } else if (lp === null) {
          out.push(0);
        }
      }
      return out;
    }
    return out;
  }

  private extractDeltaText(choice: any): string {
    const content = choice?.delta?.content;
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) {
      let out = '';
      for (const part of content) {
        if (typeof part?.text === 'string') out += part.text;
      }
      return out;
    }
    return '';
  }

  private isGemini4Model(modelName?: string): boolean {
    return typeof modelName === 'string' && /^(?:google\/)?gemini-4/i.test(modelName.trim());
  }

  private normalizeGeminiThinkingLevel(
    level: string | undefined,
    modelName?: string
  ): 'low' | 'medium' | 'high' | 'minimal' | undefined {
    const gemini4 = this.isGemini4Model(modelName);
    if (!level) {
      return gemini4 ? 'medium' : undefined;
    }
    const normalized = String(level).trim().toLowerCase();
    if (normalized === 'minimal') {
      return gemini4 ? 'low' : 'minimal';
    }
    if (normalized === 'low' || normalized === 'medium' || normalized === 'high') {
      return normalized;
    }
    return gemini4 ? 'medium' : undefined;
  }

  private buildGeminiPayload(
    messages: Message[],
    modelConfig?: ModelConfig,
    options?: ChatOptions,
    continuationToken?: string
  ): Record<string, any> {
    const systemParts: string[] = [];
    const contents: Array<{ role: string; parts: Array<{ text: string }> }> = [];
    for (const m of messages) {
      if (m.role === 'system') {
        if (m.content) systemParts.push(m.content);
      } else {
        contents.push({
          role: m.role === 'assistant' ? 'model' : 'user',
          parts: [{ text: m.content }]
        });
      }
    }
    const body: Record<string, any> = { contents };
    if (systemParts.length > 0) {
      body.system_instruction = { parts: [{ text: systemParts.join('\n\n') }] };
    }

    const gemini4 = this.isGemini4Model(modelConfig?.model);
    const thinkingLevel = this.normalizeGeminiThinkingLevel(
      options?.thinkingLevel ?? modelConfig?.thinkingLevel,
      modelConfig?.model
    );
    const rawMaxTokens = options?.maxOutputTokens ?? modelConfig?.maxOutputTokens;
    const maxOutputTokens = rawMaxTokens ?? (gemini4 ? 65536 : undefined);

    if (maxOutputTokens !== undefined || thinkingLevel !== undefined) {
      body.generationConfig = {
        ...(maxOutputTokens !== undefined ? { maxOutputTokens } : {}),
        ...(thinkingLevel !== undefined ? { thinkingConfig: { thinkingLevel } } : {})
      };
    }

    if (continuationToken) {
      body.continuationToken = continuationToken;
    }

    return body;
  }

  /**
   * Streams the fast model, buffering the first N tokens to check logprobs and loops.
   * If confidence is lower than threshold or loop is detected, aborts and returns { aborted: true }.
   */
  private async streamAndEvaluateFastModel(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): Promise<{ text: string; aborted: boolean; usage?: UsageMetrics }> {
    const modelToUse = customModel
      || ((options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : this.config.fastModel!);
    if (modelToUse.provider === 'gemini') {
      const res = await this.fetchGemini(messages, options, modelToUse);
      return { text: res.text, aborted: false, usage: res.usage };
    }
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
      const tokensToEval = this.config.tokensToEvaluate ?? 5;
      let evaluationComplete = tokensToEval <= 0;
      const threshold = this.config.cascadeThreshold ?? 0.85;
      const seenDeltas: string[] = [];
      const maxLoopWindow = Math.max(10, this.config.maxRepetitiveTokens || 4);
      let streamPromptTokens: number | undefined;
      let streamCompletionTokens: number | undefined;

      let buffer = '';

      try {
        while (true) {
          if (options?.signal?.aborted) {
            throw options.signal.reason ?? new Error('AbortError: The operation was aborted');
          }
          const { done, value } = await reader.read();
          if (options?.signal?.aborted) {
            throw options.signal.reason ?? new Error('AbortError: The operation was aborted');
          }
          if (done) {
            buffer += decoder.decode();
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

          for (const line of lines) {
            const payload = this.extractSseDataPayload(line);
            if (!payload) continue;

            try {
              const data = JSON.parse(payload);
              if (data?.error) {
                const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
                throw new Error(`Fast model SSE error: ${msg}`);
              }
              if (data?.usage) {
                if (typeof data.usage.prompt_tokens === 'number') streamPromptTokens = data.usage.prompt_tokens;
                if (typeof data.usage.completion_tokens === 'number') streamCompletionTokens = data.usage.completion_tokens;
              }
              const choice = data.choices?.[0];
              const delta = this.extractDeltaText(choice);
              
              if (delta) {
                fullText += delta;
                seenDeltas.push(delta);
                if (seenDeltas.length > maxLoopWindow * 2) {
                  seenDeltas.splice(0, seenDeltas.length - maxLoopWindow);
                }

                // Repetition loop / entropy collapse check
                const loopKind = this.detectRepetitiveLoop(seenDeltas);
                if (loopKind) {
                  if (loopKind === 'entropy_collapse') {
                    this.config.onEvent?.('entropy_collapse_triggered', { tokenCount, delta });
                  }
                  this.config.onEvent?.('repetition_loop_triggered', { tokenCount, delta, reason: loopKind });
                  controller.abort();
                  return { text: '', aborted: true };
                }
              }

              // Evaluate logprobs if present
              if (!evaluationComplete) {
                const linearProbs = this.extractLinearProbsFromChoice(choice);
                for (const linearProb of linearProbs) {
                  accumulatedProb += linearProb;
                  tokenCount++;

                  if (tokenCount >= tokensToEval) {
                    evaluationComplete = true;
                    const avgProb = accumulatedProb / tokenCount;
                    if (avgProb < threshold) {
                      // Abort! Confidence too low.
                      this.config.onEvent?.('cascade_triggered', { tokenCount, avgProb, threshold });
                      controller.abort();
                      return { text: '', aborted: true };
                    }
                    break;
                  }
                }
              }
            } catch (e) {
              if (!(e instanceof SyntaxError)) throw e;
            }
          }
        }

        // Flush remaining buffer
        const remPayload = this.extractSseDataPayload(buffer);
        if (remPayload) {
          try {
            const data = JSON.parse(remPayload);
            if (data?.error) {
              const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
              throw new Error(`Fast model SSE error: ${msg}`);
            }
            if (data?.usage) {
              if (typeof data.usage.prompt_tokens === 'number') streamPromptTokens = data.usage.prompt_tokens;
              if (typeof data.usage.completion_tokens === 'number') streamCompletionTokens = data.usage.completion_tokens;
            }
            const choice = data.choices?.[0];
            const delta = this.extractDeltaText(choice);
            if (delta) {
              fullText += delta;
              seenDeltas.push(delta);
              const loopKind = this.detectRepetitiveLoop(seenDeltas);
              if (loopKind) {
                if (loopKind === 'entropy_collapse') {
                  this.config.onEvent?.('entropy_collapse_triggered', { tokenCount, delta });
                }
                this.config.onEvent?.('repetition_loop_triggered', { tokenCount, delta, reason: loopKind });
                return { text: '', aborted: true };
              }
            }
            if (!evaluationComplete) {
              const linearProbs = this.extractLinearProbsFromChoice(choice);
              for (const linearProb of linearProbs) {
                accumulatedProb += linearProb;
                tokenCount++;
              }
            }
          } catch (e) { if (!(e instanceof SyntaxError)) throw e; }
        }

        if (!evaluationComplete && tokenCount > 0) {
          const avgProb = accumulatedProb / tokenCount;
          if (avgProb < threshold) {
            this.config.onEvent?.('cascade_triggered', { tokenCount, avgProb, threshold });
            return { text: '', aborted: true };
          }
        }

        const usage = (streamPromptTokens !== undefined || streamCompletionTokens !== undefined)
          ? this.calculateUsage(
              modelToUse,
              streamPromptTokens ?? estimateTokens(joinMessagesContent(messages)),
              streamCompletionTokens ?? estimateTokens(fullText)
            )
          : undefined;

        return { text: fullText, aborted: false, usage };
      } finally {
        if (typeof (reader as any).cancel === 'function') {
          // Error-handling policy: log stream cancellation failures without masking primary stream outcome (BUG-01)
          try { await reader.cancel(); } catch (err) { console.error('CascadeRouter error:', err); }
        }
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
    } else if (isLocalFastDefault && !modelConfig.provider && !modelConfig.apiKey) {
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

  private extractGeminiText(data: any): string {
    const parts = data?.candidates?.[0]?.content?.parts;
    if (!Array.isArray(parts) || parts.length === 0) return '';
    if (parts.length === 1) {
      const p0 = parts[0];
      return (p0 && !p0.thought && typeof p0.text === 'string') ? p0.text : '';
    }
    let out = '';
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      if (p && !p.thought && typeof p.text === 'string') {
        out += p.text;
      }
    }
    return out;
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
    if (data?.error) {
      const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
      throw new Error(`${modelConfig.model} error: ${msg}`);
    }
    const rawContent = data.choices?.[0]?.message?.content;
    let text = '';
    if (typeof rawContent === 'string') {
      text = rawContent;
    } else if (Array.isArray(rawContent)) {
      for (const part of rawContent) {
        if (typeof part?.text === 'string') text += part.text;
      }
    }
    
    // Extract usage
    const promptTokens = data.usage?.prompt_tokens ?? estimateTokens(joinMessagesContent(messages));
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

  /**
   * Escalate to intermediate burstModel (Colab G4 gpt-oss:120b) if configured, falling back to heavyModel (OpenRouter/cloud) on error.
   */
  private async fetchEscalatedModel(
    messages: Message[],
    options: ChatOptions | undefined,
    reason: string,
    extraMeta?: Record<string, any>,
    heavyOverride?: ModelConfig
  ): Promise<{ text: string; usage: UsageMetrics; routedTo: 'burst' | 'heavy'; model: string }> {
    const heavyModel = heavyOverride || this.config.heavyModel!;
    if (this.config.burstModel) {
      this.config.onEvent?.('route_burst', { reason, model: this.config.burstModel.model, ...extraMeta });
      try {
        const { text, usage } = await this.fetchModel(this.config.burstModel, messages, options);
        return { text, usage, routedTo: 'burst', model: this.config.burstModel.model };
      } catch (burstErr) {
        if (options?.signal?.aborted) {
          throw burstErr;
        }
        this.config.onEvent?.('route_heavy', {
          reason: 'burst_model_error',
          fallbackFrom: this.config.burstModel.model,
          originalReason: reason,
          error: (burstErr as Error).message,
          ...extraMeta
        });
        const { text, usage } = await this.fetchModel(heavyModel, messages, options);
        return { text, usage, routedTo: 'heavy', model: heavyModel.model };
      }
    }

    this.config.onEvent?.('route_heavy', { reason, ...extraMeta });
    const { text, usage } = await this.fetchModel(heavyModel, messages, options);
    return { text, usage, routedTo: 'heavy', model: heavyModel.model };
  }

  private getGeminiEndpoint(modelConfig: ModelConfig, stream: boolean): { url: string; headers: Record<string, string> } {
    const apiKey = modelConfig.apiKey;
    if (!apiKey && !modelConfig.url) throw new Error('Gemini requires an API key');
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    let url: string;
    if (modelConfig.url) {
      if (modelConfig.url.includes(':generateContent') || modelConfig.url.includes(':streamGenerateContent')) {
        url = modelConfig.url;
      } else {
        const base = modelConfig.url.replace(/\/+$/, '');
        url = `${base}/${modelConfig.model}:${action}`;
      }
      if (apiKey && !url.includes('key=')) {
        url += `${url.includes('?') ? '&' : '?'}key=${apiKey}`;
      }
    } else {
      url = `https://generativelanguage.googleapis.com/v1beta/models/${modelConfig.model}:${action}?key=${apiKey}`;
    }
    const headers: Record<string, string> = { 'Content-Type': 'application/json', ...modelConfig.headers };
    return { url, headers };
  }

  private async fetchGemini(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): Promise<{ text: string, usage: UsageMetrics }> {
    const heavyModel = customModel || this.config.heavyModel!;
    const { url, headers } = this.getGeminiEndpoint(heavyModel, false);
    const autoContinue = options?.automaticContinuation ?? heavyModel.automaticContinuation ?? true;
    const maxContinuationHops = options?.maxContinuationHops ?? 8;

    let currentContinuationToken = options?.continuationToken;
    let accumulatedText = '';
    let totalInputTokens = 0;
    let initialPromptTokens = 0;
    let totalOutputTokens = 0;
    let totalThoughtTokens = 0;
    let totalCachedTokens = 0;
    let hops = 0;
    let hasUsageMetadata = false;

    while (true) {
      hops += 1;
      const body = this.buildGeminiPayload(messages, heavyModel, options, currentContinuationToken);

      const response = await this.fetchFn(url, {
        method: 'POST',
        headers,
        signal: options?.signal,
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        throw new Error(`Gemini HTTP ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      if (data?.error) {
        const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
        throw new Error(`Gemini error: ${msg}`);
      }
      accumulatedText += this.extractGeminiText(data);

      if (data?.usageMetadata) {
        hasUsageMetadata = true;
        const hopPromptTokens = data.usageMetadata.promptTokenCount ?? 0;
        if (hops === 1) {
          initialPromptTokens = hopPromptTokens;
        }
        totalInputTokens += hopPromptTokens;
        totalOutputTokens += data.usageMetadata.candidatesTokenCount ?? 0;
        totalThoughtTokens += data.usageMetadata.thoughtsTokenCount ?? 0;
        totalCachedTokens += data.usageMetadata.cachedContentTokenCount ?? 0;
      }

      const candidate = data?.candidates?.[0];
      const finishReason = candidate?.finishReason;
      const nextToken = candidate?.continuationToken || data?.continuationToken;

      if (autoContinue && finishReason === 'CONTINUATION' && nextToken && hops <= maxContinuationHops) {
        currentContinuationToken = nextToken;
        continue;
      }
      break;
    }

    const promptTokens = hasUsageMetadata ? totalInputTokens : estimateTokens(joinMessagesContent(messages));
    const completionTokens = hasUsageMetadata ? totalOutputTokens : estimateTokens(accumulatedText);
    const usage = this.calculateUsage(
      heavyModel,
      promptTokens,
      completionTokens,
      hasUsageMetadata
        ? {
            initialPromptTokens,
            cachedTokens: totalCachedTokens,
            thoughtTokens: totalThoughtTokens,
            hops
          }
        : undefined
    );

    return { text: accumulatedText, usage };
  }

  private async *streamAndEvaluateFastModelGen(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): AsyncGenerator<string, void, unknown> {
    const modelToUse = customModel
      || ((options?.speedPriority === 'low' || options?.urgency === 'low') && this.config.backgroundModel
        ? this.config.backgroundModel
        : this.config.fastModel!);
    if (modelToUse.provider === 'gemini') {
      yield* this.streamGemini(messages, options, modelToUse);
      this.config.onEvent?.('route_fast', {
        reason: 'high_confidence',
        model: modelToUse.model,
        speedPriority: options?.speedPriority || 'normal'
      });
      return;
    }
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
      if (options?.signal) options.signal.removeEventListener('abort', onAbort);
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
    const tokensToEval = this.config.tokensToEvaluate ?? 5;
    let evaluationComplete = tokensToEval <= 0;
    const threshold = this.config.cascadeThreshold ?? 0.85;

    let buffer = '';

    try {
      while (true) {
        if (options?.signal?.aborted) {
          throw options.signal.reason ?? new Error('AbortError: The operation was aborted');
        }
        const { done, value } = await reader.read();
        if (options?.signal?.aborted) {
          throw options.signal.reason ?? new Error('AbortError: The operation was aborted');
        }
        if (done) {
          buffer += decoder.decode();
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const payload = this.extractSseDataPayload(line);
          if (!payload) continue;

          try {
            const data = JSON.parse(payload);
            if (data?.error) {
              const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
              throw new Error(`Fast model SSE error: ${msg}`);
            }
            const choice = data.choices?.[0];
            const delta = this.extractDeltaText(choice);
            const currentTokenProbs = !evaluationComplete ? this.extractLinearProbsFromChoice(choice) : [];

            if (delta) {
              if (evaluationComplete) {
                yield delta;
              } else {
                seenDeltas.push(delta);
                // Check for degenerate repetitive loop while still in speculative buffer window
                const loopKind = this.detectRepetitiveLoop(seenDeltas);
                if (loopKind) {
                  if (loopKind === 'entropy_collapse') {
                    this.config.onEvent?.('entropy_collapse_triggered', { tokenCount, delta });
                  }
                  this.config.onEvent?.('repetition_loop_triggered', { tokenCount, delta, reason: loopKind });
                  controller.abort();
                  throw new CascadeTriggeredError('Repetitive token loop detected');
                }

                bufferedTokens.push(delta);
              }
            }

            if (!evaluationComplete) {
              if (currentTokenProbs.length > 0) {
                for (const prob of currentTokenProbs) {
                  accumulatedProb += prob;
                  tokenCount++;

                  if (!evaluationComplete && tokenCount >= tokensToEval) {
                    evaluationComplete = true;
                    const avgProb = accumulatedProb / tokenCount;
                    if (avgProb < threshold) {
                      this.config.onEvent?.('cascade_triggered', { tokenCount, avgProb, threshold });
                      controller.abort();
                      throw new CascadeTriggeredError('Speculative cascade triggered');
                    } else {
                      for (const t of bufferedTokens) {
                        yield t;
                      }
                      bufferedTokens.length = 0;
                    }
                  }
                }
              } else if (tokenCount === 0 && bufferedTokens.length >= tokensToEval) {
                // Provider does not return logprobs: flush buffer after initial loop-guard window
                evaluationComplete = true;
                for (const t of bufferedTokens) {
                  yield t;
                }
                bufferedTokens.length = 0;
              }
            }
          } catch (e) {
            if (e instanceof CascadeTriggeredError) throw e;
            if (!(e instanceof SyntaxError)) throw e;
          }
        }
      }

      // Flush remaining line buffer if any
      const remPayload = this.extractSseDataPayload(buffer);
      if (remPayload) {
        try {
          const data = JSON.parse(remPayload);
          if (data?.error) {
            const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
            throw new Error(`Fast model SSE error: ${msg}`);
          }
          const choice = data.choices?.[0];
          const delta = this.extractDeltaText(choice);
          if (delta) {
            if (evaluationComplete) {
              yield delta;
            } else {
              seenDeltas.push(delta);
              const loopKind = this.detectRepetitiveLoop(seenDeltas);
              if (loopKind) {
                if (loopKind === 'entropy_collapse') {
                  this.config.onEvent?.('entropy_collapse_triggered', { tokenCount, delta });
                }
                this.config.onEvent?.('repetition_loop_triggered', { tokenCount, delta, reason: loopKind });
                throw new CascadeTriggeredError('Repetitive token loop detected');
              }
              bufferedTokens.push(delta);
            }
          }
          if (!evaluationComplete) {
            const linearProbs = this.extractLinearProbsFromChoice(choice);
            for (const linearProb of linearProbs) {
              accumulatedProb += linearProb;
              tokenCount++;
            }
          }
        } catch (e) {
          if (e instanceof CascadeTriggeredError) throw e;
          if (!(e instanceof SyntaxError)) throw e;
        }
      }

      if (!evaluationComplete) {
        evaluationComplete = true;
        const avgProb = tokenCount > 0 ? accumulatedProb / tokenCount : 1.0;
        if (tokenCount > 0 && avgProb < threshold) {
          this.config.onEvent?.('cascade_triggered', { tokenCount, avgProb, threshold });
          throw new CascadeTriggeredError('Speculative cascade triggered on short stream');
        } else {
          for (const t of bufferedTokens) {
            yield t;
          }
        }
      }
      
      this.config.onEvent?.('route_fast', {
        reason: 'high_confidence',
        model: modelToUse.model,
        speedPriority: options?.speedPriority || 'normal'
      });

    } finally {
      if (typeof (reader as any).cancel === 'function') {
        try { await reader.cancel(); } catch (err) { console.error('CascadeRouter error:', err); }
      }
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
        if (options?.signal?.aborted) {
          throw options.signal.reason ?? new Error('AbortError: The operation was aborted');
        }
        const { done, value } = await reader.read();
        if (options?.signal?.aborted) {
          throw options.signal.reason ?? new Error('AbortError: The operation was aborted');
        }
        if (done) {
          buffer += decoder.decode();
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const payload = this.extractSseDataPayload(line);
          if (!payload) continue;

          try {
            const data = JSON.parse(payload);
            if (data?.error) {
              const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
              throw new Error(`Heavy model SSE error: ${msg}`);
            }
            const delta = this.extractDeltaText(data.choices?.[0]);
            if (delta) yield delta;
          } catch (e) {
            if (!(e instanceof SyntaxError)) throw e;
          }
        }
      }

      const remPayload = this.extractSseDataPayload(buffer);
      if (remPayload) {
        try {
          const data = JSON.parse(remPayload);
          if (data?.error) {
            const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
            throw new Error(`Heavy model SSE error: ${msg}`);
          }
          const delta = this.extractDeltaText(data.choices?.[0]);
          if (delta) yield delta;
        } catch (e) { if (!(e instanceof SyntaxError)) throw e; }
      }
    } finally {
      if (typeof (reader as any).cancel === 'function') {
        try { await reader.cancel(); } catch (err) { console.error('CascadeRouter error:', err); }
      }
      reader.releaseLock();
    }
  }

  private async *streamGemini(messages: Message[], options?: ChatOptions, customModel?: ModelConfig): AsyncGenerator<string, void, unknown> {
    const heavyModel = customModel || this.config.heavyModel!;
    const { url, headers } = this.getGeminiEndpoint(heavyModel, true);
    const body = this.buildGeminiPayload(messages, heavyModel, options);

    const response = await this.fetchFn(url, {
      method: 'POST',
      headers,
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
    let braceCount = 0;
    let startIdx = -1;
    let inString = false;
    let escaped = false;
    let scanPos = 0;

    try {
      while (true) {
        if (options?.signal?.aborted) {
          throw options.signal.reason ?? new Error('AbortError: The operation was aborted');
        }
        const { done, value } = await reader.read();
        if (options?.signal?.aborted) {
          throw options.signal.reason ?? new Error('AbortError: The operation was aborted');
        }
        if (done) {
          buffer += decoder.decode();
        } else {
          buffer += decoder.decode(value, { stream: true });
        }
        
        for (let i = scanPos; i < buffer.length; i++) {
          const char = buffer[i];
          if (inString) {
            if (escaped) {
              escaped = false;
            } else if (char === '\\') {
              escaped = true;
            } else if (char === '"') {
              inString = false;
            }
            continue;
          }

          if (char === '"') {
            inString = true;
          } else if (char === '{') {
            if (braceCount === 0) startIdx = i;
            braceCount++;
          } else if (char === '}') {
            if (braceCount > 0) {
              braceCount--;
              if (braceCount === 0 && startIdx !== -1) {
                const jsonStr = buffer.substring(startIdx, i + 1);
                try {
                  const data = JSON.parse(jsonStr);
                  if (data?.error) {
                    const msg = typeof data.error === 'string' ? data.error : (data.error.message || JSON.stringify(data.error));
                    throw new Error(`Gemini SSE error: ${msg}`);
                  }
                  const text = this.extractGeminiText(data);
                  if (text) yield text;
                } catch (e) {
                  if (!(e instanceof SyntaxError)) throw e;
                }
                buffer = buffer.substring(i + 1);
                i = -1;
                startIdx = -1;
                inString = false;
                escaped = false;
              }
            }
          }
        }
        scanPos = buffer.length;
        if (done) break;
      }
    } finally {
      if (typeof (reader as any).cancel === 'function') {
        try { await reader.cancel(); } catch (err) { console.error('CascadeRouter error:', err); }
      }
      reader.releaseLock();
    }
  }

  public evictExpiredCacheEntries(now: number = Date.now()): number {
    let evicted = 0;
    for (const [key, entry] of this.sreCache.entries()) {
      if (entry.expiresAt <= now) {
        this.sreCache.delete(key);
        evicted++;
      }
    }
    return evicted;
  }

  private async fetchJeanSuggestions(project?: string): Promise<string | null> {
    const gate = this.config.jeanSREGate;
    if (!gate || !gate.enabled) return null;

    this.evictExpiredCacheEntries();
    const cacheKey = project || '';
    const cacheTtlMs = gate.cacheTtlMs ?? 0;
    if (cacheTtlMs > 0) {
      const cached = this.sreCache.get(cacheKey);
      if (cached) {
        if (Date.now() < cached.expiresAt) {
          return cached.reply;
        }
        this.sreCache.delete(cacheKey);
      }
    }

    const url = gate.sreUrl || 'http://localhost:3005/chat';
    const conversationId = gate.conversationId || 'a123e456-789b-12d3-a456-426614174000';
    const query = "Provide a concise list of any current fleet warnings, database anomalies, CPU thermal alerts, or Docker container issues that require human-agent review.";
    const message = project ? `[Project Isolation Context: ${project}] ${query}` : query;
    const timeoutMs = gate.timeoutMs ?? 2000;

    const controller = new AbortController();
    const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;

    try {
      const response = await this.fetchFn(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({ conversationId, message, noTools: false })
      });
      if (!response.ok) return null;
      const data = await response.json();
      const reply = data.reply || null;
      if (cacheTtlMs > 0) {
        this.evictExpiredCacheEntries();
        while (this.sreCache.size >= CascadeRouter.SRE_CACHE_MAX_ENTRIES && !this.sreCache.has(cacheKey)) {
          const oldestKey = this.sreCache.keys().next().value;
          if (oldestKey === undefined) break;
          this.sreCache.delete(oldestKey);
        }
        this.sreCache.set(cacheKey, { reply, expiresAt: Date.now() + cacheTtlMs });
      }
      return reply;
    } catch (err) {
      // Gracefully degrade if SRE endpoint is down, unreachable, or times out
      return null;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  private calculateUsage(
    modelConfig: ModelConfig,
    promptTokens: number,
    completionTokens: number,
    extra?: {
      initialPromptTokens?: number;
      cachedTokens?: number;
      thoughtTokens?: number;
      hops?: number;
    }
  ): UsageMetrics {
    const isLocalEndpoint = Boolean(
      modelConfig.url?.includes('localhost') ||
      modelConfig.url?.includes('127.0.0.1') ||
      ((modelConfig === this.config.fastModel || modelConfig === this.config.backgroundModel) &&
        !modelConfig.url &&
        !modelConfig.apiKey &&
        !modelConfig.provider)
    );
    const gemini4 = this.isGemini4Model(modelConfig.model);
    const defaultInRate = gemini4 ? 2.0 : (modelConfig.provider === 'gemini' ? 0.15 : (isLocalEndpoint ? 0.0 : 0.15));
    const defaultOutRate = gemini4 ? 10.0 : (modelConfig.provider === 'gemini' ? 0.60 : (isLocalEndpoint ? 0.0 : 0.60));
    const inRate = modelConfig.costPerMillionInputTokens ?? defaultInRate;
    const outRate = modelConfig.costPerMillionOutputTokens ?? defaultOutRate;

    const cachedTokens = Math.max(0, Math.min(promptTokens, extra?.cachedTokens ?? 0));
    const uncachedPromptTokens = Math.max(0, promptTokens - cachedTokens);
    const cachedInRate = gemini4 ? inRate * 0.05 : inRate;
    const thoughtTokens = Math.max(0, extra?.thoughtTokens ?? 0);

    const promptCost = ((uncachedPromptTokens / 1_000_000) * inRate) + ((cachedTokens / 1_000_000) * cachedInRate);
    const completionCost = ((completionTokens + thoughtTokens) / 1_000_000) * outRate;
    const estimatedCostUsd = Math.round((promptCost + completionCost) * 1_000_000) / 1_000_000;

    return {
      promptTokens,
      ...((gemini4 || (extra?.hops !== undefined && extra.hops > 1)) && extra?.initialPromptTokens !== undefined
        ? { initialPromptTokens: extra.initialPromptTokens }
        : {}),
      ...(extra?.cachedTokens !== undefined && extra.cachedTokens > 0 ? { cachedTokens: extra.cachedTokens } : {}),
      completionTokens,
      ...(extra?.thoughtTokens !== undefined && extra.thoughtTokens > 0 ? { thoughtTokens: extra.thoughtTokens } : {}),
      totalTokens: promptTokens + completionTokens + thoughtTokens,
      estimatedCostUsd,
      ...(extra?.hops !== undefined && extra.hops > 1 ? { hops: extra.hops } : {})
    };
  }

  private recordRequestMetrics(routedTo: 'fast' | 'burst' | 'heavy', cascaded: boolean, promptTokens: number, completionTokens: number, actualCostUsd: number) {
    this.metrics.totalRequests++;
    this.metrics.totalPromptTokens += promptTokens;
    this.metrics.totalCompletionTokens += completionTokens;
    this.metrics.estimatedCostUsd += actualCostUsd;

    if (routedTo === 'fast') {
      this.metrics.fastRequests++;
      // Calculate savings vs heavy cloud baseline ($0.15/$0.60 per 1M tokens)
      const cloudHypotheticalCost = ((promptTokens / 1_000_000) * 0.15) + ((completionTokens / 1_000_000) * 0.60);
      this.metrics.estimatedSavingsUsd += Math.max(0, cloudHypotheticalCost - actualCostUsd);
    } else if (routedTo === 'burst') {
      this.metrics.burstRequests = (this.metrics.burstRequests ?? 0) + 1;
      if (cascaded) {
        this.metrics.cascadedRequests++;
      }
      // Colab G4 burst tier avoids paid OpenRouter cloud token costs
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
