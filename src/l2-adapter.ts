/**
 * @module l2-adapter
 * Level 2 (L2) Neural Semantic Router Adapter for krusch-cascade-router.
 * Enables in-process vector centroid classification against domain archetypes
 * or seamless connection to krusch-context-mcp / local embedding engines.
 */

import { SpecialistRole } from './classifier.js';
import { SemanticRouteResult, SemanticRouterL2 } from './cascade.js';

export interface ArchetypeCentroid {
  archetype: string;
  role: SpecialistRole;
  tier: 'specialist' | 'heavy' | 'frontier';
  label: string;
  exemplar: string;
  embedding?: number[];
  threshold?: number;
  metadata?: Record<string, any>;
}

export const DEFAULT_L2_ARCHETYPES: readonly ArchetypeCentroid[] = Object.freeze(([
  {
    archetype: 'code_implementation',
    role: 'code',
    tier: 'specialist',
    label: 'Code Refactoring & Implementation',
    exemplar: 'Refactor this TypeScript function to use async/await instead of nested promises, add error handling and type annotations.',
    threshold: 0.65,
    metadata: { domain: 'software_engineering', preferredModel: 'Qwen3-Coder-Next' }
  },
  {
    archetype: 'code_api_backend',
    role: 'code',
    tier: 'specialist',
    label: 'API & Backend Engineering',
    exemplar: 'Implement an Express.js router endpoint with schema validation, rate limiting, and PostgreSQL connection pooling.',
    threshold: 0.65,
    metadata: { domain: 'backend', preferredModel: 'Qwen3-Coder-Next' }
  },
  {
    archetype: 'code_testing',
    role: 'code',
    tier: 'specialist',
    label: 'Unit & Integration Testing',
    exemplar: 'Write a comprehensive unit test suite using vitest covering edge cases, mock dependencies, and error branches.',
    threshold: 0.65,
    metadata: { domain: 'testing', preferredModel: 'Qwen3-Coder-Next' }
  },
  {
    archetype: 'reasoning_architecture',
    role: 'reasoning_deep',
    tier: 'heavy',
    label: 'System Architecture & Concurrency',
    exemplar: 'Analyze this distributed system architecture and identify single points of failure, network partition risks, and transaction isolation anomalies.',
    threshold: 0.65,
    metadata: { domain: 'architecture', preferredModel: 'qwen3-235b-a22b-2507' }
  },
  {
    archetype: 'reasoning_root_cause',
    role: 'reasoning_deep',
    tier: 'heavy',
    label: 'Deep Root-Cause Debugging',
    exemplar: 'Why is our Node.js event loop lagging by 400ms under high WebSocket throughput? Walk through V8 heap dumps, GC pauses, and microtask starvation.',
    threshold: 0.65,
    metadata: { domain: 'diagnostics', preferredModel: 'deepseek-v4-pro' }
  },
  {
    archetype: 'reasoning_formal_logic',
    role: 'reasoning_deep',
    tier: 'heavy',
    label: 'Formal Logic & Mathematical Proof',
    exemplar: 'Provide a formal mathematical proof for the convergence and stability bounds of this gradient descent optimization algorithm.',
    threshold: 0.68,
    metadata: { domain: 'mathematics', preferredModel: 'deepseek-v4-pro' }
  },
  {
    archetype: 'factual_documentation',
    role: 'factual_stem',
    tier: 'specialist',
    label: 'Factual Technical Documentation',
    exemplar: 'What is the exact specification of HTTP 429 Too Many Requests and how does the Retry-After header behave in RFC 6585?',
    threshold: 0.62,
    metadata: { domain: 'standards', preferredModel: 'deepseek-v4-flash' }
  },
  {
    archetype: 'factual_scientific',
    role: 'factual_stem',
    tier: 'specialist',
    label: 'Scientific & Domain Facts',
    exemplar: 'Explain the mechanism of action for mRNA vaccines and how antigen presentation triggers memory B cell differentiation.',
    threshold: 0.62,
    metadata: { domain: 'science', preferredModel: 'gemini-3.1-flash-lite' }
  },
  {
    archetype: 'comprehension_synthesis',
    role: 'comprehension_rc',
    tier: 'specialist',
    label: 'Long-Form Reading & Document Synthesis',
    exemplar: 'Summarize the primary risk factors, indemnification obligations, and renewal terms from this 30-page vendor contract.',
    threshold: 0.63,
    metadata: { domain: 'document_analysis', preferredModel: 'deepseek-v4-flash' }
  },
  {
    archetype: 'creative_copy',
    role: 'general_fast',
    tier: 'specialist',
    label: 'Creative Writing & Prose',
    exemplar: 'Write an engaging, humorous developer blog post introducing our new zero-dependency CLI tool with an analogy to mechanical watches.',
    threshold: 0.62,
    metadata: { domain: 'creative', preferredModel: 'gemini-3.1-flash-lite' }
  },
  {
    archetype: 'high_risk_guardrail',
    role: 'reasoning_deep',
    tier: 'frontier',
    label: 'High-Stakes Safety / Clinical / Legal Exclusion',
    exemplar: 'Patient presenting with acute chest pain, diaphoresis, and left arm numbness; provide emergency clinical triage protocols.',
    threshold: 0.65,
    metadata: { domain: 'safety_exclusion', preferredModel: 'claude-3-5-sonnet' }
  }
] as ArchetypeCentroid[]).map((entry): ArchetypeCentroid => Object.freeze({ ...entry, metadata: entry.metadata ? Object.freeze({ ...entry.metadata }) : undefined })));

/**
 * Computes cosine similarity between two numeric vectors in single-digit microseconds.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0 || !Number.isFinite(dot) || !Number.isFinite(normA) || !Number.isFinite(normB)) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function normalizeVector(vec: number[]): number[] | null {
  if (!Array.isArray(vec) || vec.length === 0) return null;
  let normSq = 0;
  for (let i = 0; i < vec.length; i++) {
    const v = vec[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) return null;
    normSq += v * v;
  }
  if (normSq === 0 || !Number.isFinite(normSq)) return null;
  const invNorm = 1 / Math.sqrt(normSq);
  const unit = new Array<number>(vec.length);
  for (let i = 0; i < vec.length; i++) {
    unit[i] = vec[i] * invNorm;
  }
  return unit;
}

function dotProduct(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

export interface CentroidRouterOptions {
  /**
   * Custom embedding generation function. If omitted, uses Ollama or OpenRouter configuration.
   */
  embed?: (text: string) => Promise<number[] | null>;
  /**
   * Local Ollama server endpoint (default: http://localhost:11434).
   */
  ollamaUrl?: string;
  /**
   * Embedding model for Ollama (default: 'bge-large').
   */
  embedModel?: string;
  /**
   * OpenRouter API key for cloud embeddings (baai/bge-large-en-v1.5).
   */
  openrouterApiKey?: string;
  /**
   * Archetype centroids to classify against (defaults to DEFAULT_L2_ARCHETYPES).
   */
  centroids?: ArchetypeCentroid[];
  /**
   * Global fallback confidence threshold (default: 0.65).
   */
  defaultThreshold?: number;
  /**
   * Maximum number of prompt embeddings to cache in memory (default: 256, 0 to disable).
   */
  cacheSize?: number;
  /**
   * Request timeout in milliseconds for default HTTP embedding calls (default: 3000).
   */
  timeoutMs?: number;
  /**
   * Fetch implementation (default: globalThis.fetch).
   */
  fetch?: typeof fetch;
}

/**
 * Factory function creating a high-performance in-process L2 Neural Semantic Router.
 * Computes cosine similarity against calibrated archetype centroids in <1ms once embedded.
 */
export function createCentroidSemanticRouter(options: CentroidRouterOptions = {}): SemanticRouterL2 {
  const fetchFn = options.fetch || globalThis.fetch;
  const centroids = options.centroids ? [...options.centroids] : [...DEFAULT_L2_ARCHETYPES];
  const defaultThreshold = options.defaultThreshold ?? 0.65;
  const ollamaEmbedModel = options.embedModel || 'snowflake-arctic-embed2';
  const openrouterEmbedModel = options.embedModel || 'baai/bge-large-en-v1.5';
  const baseOllamaUrl = (options.ollamaUrl || 'http://localhost:11434').replace(/\/+$/, '');
  const maxCacheSize = options.cacheSize ?? 256;
  const timeoutMs = options.timeoutMs ?? 3000;

  // Embedder implementation
  const rawEmbedder: (text: string) => Promise<number[] | null> = options.embed || (async (text: string) => {
    const controller = new AbortController();
    const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;
    try {
      if (options.openrouterApiKey) {
        const res = await fetchFn('https://openrouter.ai/api/v1/embeddings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${options.openrouterApiKey}`
          },
          signal: controller.signal,
          body: JSON.stringify({
            model: openrouterEmbedModel,
            input: text
          })
        });
        if (!res.ok) return null;
        const data = await res.json() as any;
        return data?.data?.[0]?.embedding || data?.embedding || null;
      }

      // Default: Local Ollama (supports /api/embeddings, /api/embed, and OpenAI /v1/embeddings response formats)
      const res = await fetchFn(`${baseOllamaUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          model: ollamaEmbedModel,
          prompt: text
        })
      });
      if (!res.ok) return null;
      const data = await res.json() as any;
      return data?.embedding || data?.embeddings?.[0] || data?.data?.[0]?.embedding || null;
    } catch {
      return null;
    } finally {
      if (timer) clearTimeout(timer);
    }
  });

  // Bounded LRU cache for unit-normalized prompt embeddings
  const promptVecCache = new Map<string, number[]>();

  async function getPromptUnitEmbedding(cleanPrompt: string): Promise<number[] | null> {
    if (maxCacheSize > 0) {
      const cached = promptVecCache.get(cleanPrompt);
      if (cached) {
        // Refresh LRU order
        promptVecCache.delete(cleanPrompt);
        promptVecCache.set(cleanPrompt, cached);
        return cached;
      }
    }
    const rawVec = await rawEmbedder(cleanPrompt);
    const unitVec = rawVec ? normalizeVector(rawVec) : null;
    if (unitVec && maxCacheSize > 0) {
      if (promptVecCache.size >= maxCacheSize) {
        const oldestKey = promptVecCache.keys().next().value;
        if (oldestKey !== undefined) promptVecCache.delete(oldestKey);
      }
      promptVecCache.set(cleanPrompt, unitVec);
    }
    return unitVec;
  }

  // Cached unit-normalized exemplar embeddings (index-aligned for O(1) lookup and duplicate archetype safety)
  const centroidEmbeddings: Array<number[] | null> = new Array(centroids.length).fill(null);
  let warmedCount = 0;
  for (let i = 0; i < centroids.length; i++) {
    const c = centroids[i];
    if (c.embedding && c.embedding.length > 0) {
      const unit = normalizeVector(c.embedding);
      if (unit) {
        centroidEmbeddings[i] = unit;
        warmedCount++;
      }
    }
  }

  let warmPromise: Promise<void> | null = null;

  async function warmCentroids() {
    if (warmedCount === centroids.length) return;
    if (warmPromise) return warmPromise;
    warmPromise = (async () => {
      await Promise.all(
        centroids.map(async (c, idx) => {
          if (centroidEmbeddings[idx]) return;
          if (c.embedding && c.embedding.length > 0) {
            const unit = normalizeVector(c.embedding);
            if (unit) {
              centroidEmbeddings[idx] = unit;
              warmedCount++;
            }
          } else {
            try {
              const vec = await rawEmbedder(c.exemplar);
              const unit = vec ? normalizeVector(vec) : null;
              if (unit) {
                centroidEmbeddings[idx] = unit;
                warmedCount++;
              }
            } catch {
              // Non-fatal if single centroid fails
            }
          }
        })
      );
      // If any centroids failed to warm (e.g. transient rate-limit or offline service),
      // clear warmPromise so future calls retry only the missing centroids.
      if (warmedCount < centroids.length) {
        warmPromise = null;
      }
    })();
    return warmPromise;
  }

  return async (prompt: string, context?: { project?: string; metadata?: Record<string, any> }): Promise<SemanticRouteResult | null> => {
    if (!prompt || typeof prompt !== 'string') {
      return null;
    }
    const cleanPrompt = prompt.trim();
    if (cleanPrompt.length === 0) {
      return null;
    }

    if (warmedCount < centroids.length) {
      await warmCentroids();
    }

    if (warmedCount === 0) {
      return null;
    }

    const promptUnitVec = await getPromptUnitEmbedding(cleanPrompt);
    if (!promptUnitVec) {
      return null;
    }

    let bestScore = -1;
    let bestMatch: ArchetypeCentroid | null = null;

    for (let i = 0; i < centroids.length; i++) {
      const cUnitVec = centroidEmbeddings[i];
      if (!cUnitVec || cUnitVec.length !== promptUnitVec.length) continue;
      const c = centroids[i];

      let score = dotProduct(promptUnitVec, cUnitVec);

      // Contextual project biasing
      if (context?.project && c.role === 'code') {
        score = Math.min(1.0, score + 0.05);
      }

      if (score > bestScore) {
        bestScore = score;
        bestMatch = c;
      }
    }

    if (!bestMatch) {
      return null;
    }

    const threshold = bestMatch.threshold ?? defaultThreshold;
    const passed = bestScore >= threshold;

    if (passed) {
      return {
        recommendedRole: bestMatch.role,
        targetTier: bestMatch.tier,
        recommendedModel: bestMatch.metadata?.preferredModel,
        confidence: bestScore,
        reason: `L2 Matched '${bestMatch.label}' with cosine similarity ${bestScore.toFixed(3)} (threshold: ${threshold})`
      };
    }

    // Borderline fallback
    return {
      recommendedRole: 'reasoning_deep',
      targetTier: 'heavy',
      confidence: bestScore,
      reason: `L2 Nearest centroid '${bestMatch.label}' similarity (${bestScore.toFixed(3)}) fell below threshold (${threshold}); escalating to heavy reasoning`
    };
  };
}

const VALID_SPECIALIST_ROLES = new Set<SpecialistRole>([
  'general_fast',
  'factual_stem',
  'code',
  'reasoning_fast',
  'reasoning_deep',
  'games_spatial',
  'comprehension_rc'
]);

/**
 * Adapter that connects CascadeRouter directly to a running krusch-context-mcp server tool.
 */
export function createContextMcpRouter(
  mcpToolCaller:
    | ((toolName: string, args: Record<string, any>) => Promise<any>)
    | { callTool: (toolName: string, args: Record<string, any>) => Promise<any> }
): SemanticRouterL2 {
  const caller = typeof mcpToolCaller === 'function'
    ? mcpToolCaller
    : mcpToolCaller.callTool.bind(mcpToolCaller);

  return async (prompt: string, context?: { project?: string; metadata?: Record<string, any> }) => {
    try {
      const res = await caller('krusch_context_semantic_route', {
        prompt,
        project: context?.project,
        metadata: context?.metadata
      });

      if (!res || res.isError) return null;
      const text = res?.content?.[0]?.text;
      const parsed = typeof text === 'string'
        ? JSON.parse(text)
        : (typeof res === 'string' ? JSON.parse(res) : res);

      if (!parsed || parsed.isError) return null;

      const rawRole = parsed?.recommendedRole ?? parsed?.recommended_role;
      const recommendedRole = typeof rawRole === 'string' ? rawRole.trim().toLowerCase() : '';
      if (!recommendedRole || !VALID_SPECIALIST_ROLES.has(recommendedRole as SpecialistRole)) {
        return null;
      }

      const rawConf = parsed.confidence;
      const numConf = rawConf !== undefined && rawConf !== null ? Number(rawConf) : NaN;

      const rawTierVal = parsed.targetTier ?? parsed.target_tier;
      const normalizedTier = typeof rawTierVal === 'string' ? rawTierVal.trim().toLowerCase() : '';
      const targetTier: 'specialist' | 'heavy' | 'frontier' =
        normalizedTier === 'heavy' || normalizedTier === 'frontier' || normalizedTier === 'specialist'
          ? normalizedTier
          : 'specialist';

      return {
        recommendedRole: recommendedRole as SpecialistRole,
        targetTier,
        recommendedModel: parsed.recommendedModel ?? parsed.recommended_model,
        confidence: Number.isFinite(numConf) ? numConf : 0.5,
        reason: parsed.reason
      };
    } catch {
      return null;
    }
  };
}

