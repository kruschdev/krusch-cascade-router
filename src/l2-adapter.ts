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

export const DEFAULT_L2_ARCHETYPES: ArchetypeCentroid[] = [
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
];

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
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
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
  const defaultThreshold = options.defaultThreshold || 0.65;
  const embedModel = options.embedModel || 'bge-large';
  const ollamaUrl = options.ollamaUrl || 'http://localhost:11434';

  // Embedder implementation
  const embedder: (text: string) => Promise<number[] | null> = options.embed || (async (text: string) => {
    if (options.openrouterApiKey) {
      try {
        const res = await fetchFn('https://openrouter.ai/api/v1/embeddings', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${options.openrouterApiKey}`
          },
          body: JSON.stringify({
            model: 'baai/bge-large-en-v1.5',
            input: text
          })
        });
        if (!res.ok) return null;
        const data = await res.json() as any;
        return data?.data?.[0]?.embedding || null;
      } catch {
        return null;
      }
    }

    // Default: Local Ollama
    try {
      const res = await fetchFn(`${ollamaUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: embedModel,
          prompt: text
        })
      });
      if (!res.ok) return null;
      const data = await res.json() as any;
      return data?.embedding || null;
    } catch {
      return null;
    }
  });

  // Cached exemplar embeddings (lazily initialized on first route call)
  const centroidEmbeddings = new Map<string, number[]>();
  let isWarming = false;
  let warmPromise: Promise<void> | null = null;

  async function warmCentroids() {
    if (warmPromise) return warmPromise;
    warmPromise = (async () => {
      for (const c of centroids) {
        if (c.embedding && c.embedding.length > 0) {
          centroidEmbeddings.set(c.archetype, c.embedding);
        } else {
          try {
            const vec = await embedder(c.exemplar);
            if (vec) centroidEmbeddings.set(c.archetype, vec);
          } catch {
            // Non-fatal if single centroid fails
          }
        }
      }
    })();
    return warmPromise;
  }

  return async (prompt: string, context?: { project?: string; metadata?: Record<string, any> }): Promise<SemanticRouteResult | null> => {
    if (!prompt || typeof prompt !== 'string' || prompt.trim().length === 0) {
      return null;
    }

    await warmCentroids();

    if (centroidEmbeddings.size === 0) {
      return null;
    }

    const promptVec = await embedder(prompt.trim());
    if (!promptVec) {
      return null;
    }

    let bestScore = -1;
    let bestMatch: ArchetypeCentroid | null = null;

    for (const c of centroids) {
      const cVec = centroidEmbeddings.get(c.archetype);
      if (!cVec) continue;

      let score = cosineSimilarity(promptVec, cVec);

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

    const threshold = bestMatch.threshold || defaultThreshold;
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

/**
 * Adapter that connects CascadeRouter directly to a running krusch-context-mcp server tool.
 */
export function createContextMcpRouter(mcpToolCaller: (toolName: string, args: Record<string, any>) => Promise<any>): SemanticRouterL2 {
  return async (prompt: string, context?: { project?: string; metadata?: Record<string, any> }) => {
    try {
      const res = await mcpToolCaller('krusch_context_semantic_route', {
        prompt,
        project: context?.project,
        metadata: context?.metadata
      });

      const text = res?.content?.[0]?.text;
      if (!text) return null;
      const parsed = JSON.parse(text);

      return {
        recommendedRole: parsed.recommendedRole,
        targetTier: parsed.targetTier,
        recommendedModel: parsed.recommendedModel,
        confidence: parsed.confidence,
        reason: parsed.reason
      };
    } catch {
      return null;
    }
  };
}
