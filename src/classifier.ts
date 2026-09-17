export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export type SpecialistRole = 
  | 'general_fast'     // google/gemini-3.1-flash-lite (translation, geo, social, ethics, medicine, narrative)
  | 'factual_stem'     // deepseek/deepseek-v4-flash (MMLU-Pro, OpenTDB, science, arithmetic, STEM)
  | 'code'             // Qwen/Qwen3-Coder-Next (code generation, syntax, functions)
  | 'reasoning_fast'   // Qwen/Qwen3-Coder-Next (logic puzzles, execution, algorithmic reasoning)
  | 'reasoning_deep'   // deepseek/deepseek-v4-pro (open-ended quiz bowl/trivia, SEC/financial statements)
  | 'games_spatial'    // deepseek/deepseek-v4-flash (chess, FEN/PGN, board positions)
  | 'comprehension_rc';// qwen/qwen3-235b-a22b-2507 (paragraph answer evaluation, SuperGLUE-RC)

export interface ClassifierOptions {
  lengthThreshold?: number; // String length, not tokens, for speed. Default 2000.
  customRules?: RegExp[];   // Custom Regex patterns to mark a prompt as complex
  prunePreRouting?: boolean; // If true, clean conversational filler and whitespace before length evaluation
  knowledgeBoundaryGating?: boolean; // If true, prioritize closed-world self-contained routing (default true)
}

/**
 * Knowledge Boundary Router (arXiv: 2608.23982).
 * Determines if a query represents a self-contained "closed-world" task
 * (e.g. arithmetic, unit conversion, code syntax translation, regex, dictionary lookup)
 * that does not require open-world reasoning and is degraded by cognitive context bloat.
 */
export function detectKnowledgeBoundary(text: string): 'closed' | 'open' {
  if (!text) return 'closed';
  const clean = text.trim().toLowerCase();

  // Closed-world signals: self-contained transformations and lookup queries
  const closedWorldPatterns = [
    /^(?:translate|convert|calculate|format|prettify|lint|capitalize|lowercase|reverse)\b/i,
    /\b(?:regex|regular expression|json format|csv format|unit conversion|celsius to fahrenheit|miles to km)\b/i,
    /^(?:what is|solve)\s+[\d\s+\-*/^().=]+$/i, // Direct arithmetic expressions
    /\b(?:dictionary definition|synonym for|antonym for|spelling of)\b/i
  ];

  for (const pattern of closedWorldPatterns) {
    if (pattern.test(clean)) {
      return 'closed';
    }
  }

  return 'open';
}

/**
 * Lightweight, zero-dependency text cleaner for pre-routing prompt compaction.
 */
export function pruneText(text: string): string {
  if (!text) return '';
  let cleaned = text;
  
  // Strip common conversational filler patterns (iteratively)
  let changed = true;
  while (changed) {
    const before = cleaned;
    cleaned = cleaned
      .replace(/^(?:hey|hello|hi|greetings|dear|please)[,.\s]+/i, '')
      .replace(/^(?:could you please|can you please|would you kindly|would you please|i want you to|i need you to|tell me|show me)[,.\s]+/i, '')
      .replace(/\b(?:as we discussed earlier|like i mentioned before|as you know)\b/gi, '')
      .replace(/\b(?:thanks in advance|thank you very much|thank you|thanks|let me know what you think)[.!?\s]*$/gi, '')
      .trim();
    changed = before !== cleaned;
  }

  // Deduplicate consecutive whitespace and punctuation
  return cleaned.replace(/\s+/g, ' ').replace(/([?!.,;])\1+/g, '$1').trim();
}

/**
 * Continuous complexity scorer [0.0, 1.0].
 * Provides fine-grained probability for speculative hedging ("Second Thought" arXiv: 2608.13667).
 * Scores between 0.35 and 0.65 represent the uncertainty boundary suitable for speculative pre-warming.
 */
export function evaluateComplexityScore(messages: Message[] | string, options?: ClassifierOptions): number {
  const lengthThreshold = options?.lengthThreshold || 2000;
  
  let fullText = Array.isArray(messages) 
    ? messages.map(m => m.content).join('\n') 
    : messages;

  if (options?.prunePreRouting) {
    fullText = pruneText(fullText);
  }

  // Knowledge boundary check (closed-world task deduction)
  if (options?.knowledgeBoundaryGating !== false && detectKnowledgeBoundary(fullText) === 'closed' && fullText.length < 500) {
    return 0.15;
  }

  let score = 0.0;

  // Length scoring (scaled up to 0.60)
  const lengthRatio = Math.min(1.0, fullText.length / lengthThreshold);
  score += lengthRatio * 0.60;

  // Structural markers
  if (/```[a-z]*/i.test(fullText)) score += 0.30;
  if (/<\/?([a-z][a-z0-9]*)\b[^>]*>/i.test(fullText)) score += 0.20;
  if (/\{[\s\S]*"[\s\S]*\}/.test(fullText)) score += 0.20;

  // Moderate cognitive / comparative inquiry (borderline indicators)
  if (/\b(compare|contrast|explain why|how does|tradeoffs|pros and cons|difference between)\b/i.test(fullText)) {
    score += 0.25;
  }

  // High cognitive complexity verbs
  if (/\b(analyze|evaluate|architect|synthesize|speculate|refactor|debug|test|benchmark)\b/i.test(fullText)) {
    score += 0.35;
  }

  // Custom regex rules
  if (options?.customRules) {
    for (const rule of options.customRules) {
      if (rule.test(fullText)) {
        score += 0.35;
        break;
      }
    }
  }

  return Math.min(1.0, Math.max(0.0, score));
}

/**
 * A fast, <50ms heuristic classifier to predict if a prompt is "simple" or "complex".
 * Evaluates message length and structural markers (code blocks, XML, JSON).
 */
export function isComplexPrompt(messages: Message[] | string, options?: ClassifierOptions): boolean {
  const lengthThreshold = options?.lengthThreshold || 2000;
  
  let fullText = Array.isArray(messages) 
    ? messages.map(m => m.content).join('\n') 
    : messages;

  if (options?.prunePreRouting) {
    fullText = pruneText(fullText);
  }

  if (fullText.length > lengthThreshold) {
    return true;
  }

  // Fast regex markers for complexity
  const complexMarkers = [
    /```[a-z]*/i,             // Contains code blocks
    /<\/?([a-z][a-z0-9]*)\b[^>]*>/i, // Contains XML/HTML tags
    /\{[\s\S]*"[\s\S]*\}/,    // Contains JSON-like structures
    /\b(analyze|evaluate|architect|synthesize|speculate|refactor|debug|test|benchmark)\b/i // Complex cognitive verbs
  ];

  if (options?.customRules) {
    complexMarkers.push(...options.customRules);
  }

  for (const marker of complexMarkers) {
    if (marker.test(fullText)) {
      return true;
    }
  }

  return false;
}

/**
 * Classifies a prompt into one of 7 domain specialist roles optimized
 * for sub-50ms multi-model swarm routing.
 */
export function classifySpecialistRole(messages: Message[] | string, options?: ClassifierOptions): SpecialistRole {
  let fullText = Array.isArray(messages) 
    ? messages.map(m => m.content).join('\n') 
    : messages;

  if (options?.prunePreRouting) {
    fullText = pruneText(fullText);
  }

  const p = fullText.toLowerCase();

  // 1. SuperGLUE-RC / Paragraph Reading Comprehension & Verification (qwen3-235b)
  if (
    /based on the "paragraph"/i.test(p) ||
    /provided answer" is a correct response/i.test(p) ||
    /evaluate if the "provided answer"/i.test(p) ||
    /assess the provided/i.test(p)
  ) {
    return 'comprehension_rc';
  }

  // 2. Chess & Spatial Board Games (Qwen3-Coder-Next via games_spatial)
  const isChess = 
    /chess move/i.test(p) ||
    /chess game/i.test(p) ||
    /chess position/i.test(p) ||
    /board position/i.test(p) ||
    /\b(?:fen|pgn|checkmate|castling)\b/i.test(p) ||
    /\b[a-h][1-8]-[a-h][1-8]\b/.test(p) ||
    /(?:1\.|\b(?:e4|d4|nf3|c4))\s+[a-z0-9+#=-]+/i.test(p);

  if (isChess) {
    return 'games_spatial';
  }

  // 3. Code Generation & Algorithm Synthesis (Qwen3-Coder-Next)
  const isCode = 
    /generate an executable python function/i.test(p) ||
    /craft a python/i.test(p) ||
    /py[th]{2}[on]{1,2}/i.test(p) ||
    /```(?:python|javascript|typescript|c\+\+|cpp|java|go|rust|sql|html|css|bash|sh)\b/i.test(p) ||
    /\b(?:def\s+[a-zA-Z_]\w*\s*\(|function\s+[a-zA-Z_]\w*\s*\(|class\s+[a-zA-Z_]\w*[:\{])/i.test(p) ||
    /subroutine/i.test(p) ||
    /runnable python/i.test(p) ||
    /source code/i.test(p);

  if (isCode) {
    return 'code';
  }

  // 4. Financial Statements & Balance Sheets (deepseek-v4-pro)
  if (
    /\b(net income|operating income|fiscal year|cash flows|diluted eps|balance sheet|sec filing|ebitda)\b/i.test(p)
  ) {
    return 'reasoning_deep';
  }

  // 5. Linguistics, Translation, Geography, Medicine, Trivia QANTA, Entailment
  // (Empirically superior on google/gemini-3.1-flash-lite)
  const generalFastPatterns = [
    /\b(?:translat|translation|gujarati|german|chinese|czech|finnish|lithuanian|kazakh|russian)\b/i,
    /\b(?:geograph|latitude|longitude|elevation|continent|bordering countries|capital of)\b/i,
    /\b(?:patient|symptom|clinic|diagnos|syndrome|treatment|pubmed|disease|prescribe|medmcqa)\b/i,
    /\b(?:narrative|protagonist|author's intent|storyline|allegory)\b/i,
    /\b(?:does sentence a imply|same sense of the word|entailment)\b/i
  ];

  for (const pattern of generalFastPatterns) {
    if (pattern.test(p)) {
      return 'general_fast';
    }
  }

  // Open-ended trivia without multiple choice options (Quiz Bowl / QANTA)
  const hasOptions = /\b(?:options|selections|choices|alternatives):\s*\n?\s*[a-d]\./i.test(p) || /\n\s*[a-d]\.\s+\S+/i.test(p);
  if (!hasOptions && /\b(this author|this poet|this battle|name this|identify this|for 10 points|this composer|this novel|this leader|this president|who was|which country|what city|identify the nation)\b/i.test(p)) {
    return 'general_fast';
  }

  // 6. Default STEM / Science / Math / Logic / MMLU-Pro / Ethics
  // (Empirically highest accuracy & throughput on deepseek/deepseek-v4-flash)
  return 'factual_stem';
}
