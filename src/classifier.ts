export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export type SpecialistRole = 
  | 'general_fast'     // google/gemini-3.1-flash-lite (translation, geo, social, ethics, medicine, narrative)
  | 'factual_stem'     // deepseek/deepseek-v4-flash (factual knowledge, science, arithmetic, STEM)
  | 'code'             // Qwen/Qwen3-Coder-Next (code generation, syntax, functions)
  | 'reasoning_fast'   // Qwen/Qwen3-Coder-Next (logic puzzles, execution, algorithmic reasoning)
  | 'reasoning_deep'   // deepseek/deepseek-v4-pro (financial statements, balance sheets, deep reasoning)
  | 'games_spatial'    // deepseek/deepseek-v4-flash (chess, FEN/PGN, board positions)
  | 'comprehension_rc';// qwen/qwen3-235b-a22b-2507 (paragraph answer evaluation, reading comprehension)

export interface CustomSpecialistRule {
  role: SpecialistRole;
  pattern: RegExp;
}

export interface PreRouteResult {
  isFastPath: boolean;
  role?: SpecialistRole;
  confidence: 'high' | 'borderline' | 'unstructured';
  complexityScore: number;
  suggestedAction: 'dispatch_specialist' | 'delegate_to_l2';
}

export interface ClassifierOptions {
  lengthThreshold?: number; // String length, not tokens, for speed. Default 2000.
  customRules?: RegExp[];   // Custom Regex patterns to mark a prompt as complex
  customSpecialistRules?: CustomSpecialistRule[]; // Custom regex overrides for specialist routing
  prunePreRouting?: boolean; // If true, clean conversational filler and whitespace before length evaluation
  knowledgeBoundaryGating?: boolean; // If true, prioritize closed-world self-contained routing (default true)
}

// --- Pre-Compiled Module-Scope Regex Patterns (Zero Per-Call Allocation & ReDoS Hardened) ---

const CLOSED_WORLD_RE = new RegExp(
  [
    /^(?:translate|convert|calculate|format|prettify|lint|capitalize|lowercase|reverse)\b/.source,
    /\b(?:regex|regular expression|json format|csv format|unit conversion|celsius to fahrenheit|miles to km)\b/.source,
    /^(?:what is|solve|calculate)\s+[\d\s+\-*/^().=]+[?]?$/.source,
    /\b(?:dictionary definition|synonym for|antonym for|spelling of)\b/.source
  ].join('|'),
  'i'
);

const PRUNE_GREETING_RE = /^(?:hey|hello|hi|greetings|dear|please)[,!.\s]+/i;
const PRUNE_POLITE_PREFIX_RE = /^(?:could you please|can you please|would you kindly|would you please|i want you to|i need you to|tell me|show me)[,.\s]+/i;
const PRUNE_CALLBACK_RE = /\b(?:as we discussed earlier|like i mentioned before|as you know)\b/gi;
const PRUNE_SIGN_OFF_RE = /\b(?:thanks in advance|thank you very much|thank you|thanks|let me know what you think)[.!?\s]*$/i;
const PRUNE_WHITESPACE_RE = /\s+/g;
const PRUNE_PUNCT_RE = /([?!.,;])\1+/g;

const XML_TAG_MARKER_RE = /<\/?(?:[a-z][a-z0-9]*)\b[^>]*>/i;
const COMPARATIVE_INQUIRY_RE = /\b(?:compare|contrast|explain why|how does|tradeoffs|pros and cons|difference between)\b/i;
const HIGH_COMPLEXITY_SCORE_VERBS_RE = /\b(?:analyze|evaluate|architect|synthesize|speculate|refactor|debug|test|benchmark)\b/i;
const HIGH_COMPLEXITY_PROMPT_VERBS_RE = /\b(?:analyze|evaluate|architect|synthesize|speculate|refactor|debug|benchmark)\b/i;

const RC_READING_COMPREHENSION_RE = new RegExp(
  [
    /\b(?:based on (?:the|this|that)?\s*(?:provided|following|above|below)?\s*["']?(?:text|passage|article|excerpt|document|context|paragraph|historical account|case study)["']?)/.source,
    /\b(?:according to (?:the|this|that)?\s*(?:provided|following|above|below)?\s*["']?(?:text|passage|article|excerpt|document|context|historical account|case study)["']?)/.source,
    /\b(?:in (?:the|this)\s+(?:provided|following)?\s*["']?(?:text|passage|article|excerpt|document|paragraph|case study)["']?\s+(?:above|below|provided)?)/.source,
    /\b(?:in paragraph \d+)\b/.source,
    /\b(?:summarize (?:the|this)\s+["']?(?:text|passage|article|excerpt|document|chapter|section)["']?)/.source,
    /\b(?:what does the author (?:mean|state|imply|claim|conclude|suggest|argue))\b/.source,
    /\b(?:main thesis of the author|author's main argument)\b/.source,
    /\b(?:from the\s+["']?(?:text|passage|excerpt|article|document)["']?\s+(?:above|below)?)/.source,
    /\b(?:reading comprehension|comprehension question|evaluate (?:whether|if) (?:the|this) (?:statement|claim|assertion) is (?:true|false|accurate|supported))\b/.source,
    /\b(?:information provided in (?:the|this)\s+["']?(?:preceding|provided|following)?\s*(?:text|case study|article|passage)["']?)/.source
  ].join('|'),
  'i'
);

const CHESS_KEYWORDS_RE = /\b(?:chess|checkmate|stalemate|castling|fen|pgn|en passant|zugzwang|board position|legal moves|pawn move|knight move|bishop move|rook move|queen move|king move|sudoku grid|tic-tac-toe|connect four|gomoku)\b/i;
const CHESS_COORD_RE = /\b[a-h][1-8]-[a-h][1-8]\b/;
const CHESS_ALGEBRAIC_RE = /(?:1\.|\b(?:e4|d4|nf3|c4))\s+[a-z0-9+#=-]+/i;

const CODE_INTENT_RE = /\b(?:write|create|implement|build|refactor|debug|fix|optimize|convert)\b[\s\S]{0,60}\b(?:code|script|function|class|method|algorithm|program|solution|api|endpoint|query|component|hook|test|handler|decorator|type|interface|schema|middleware|resolver|generator|workflow|pipeline|dockerfile|regex|callback|promise|async\/await|binary search|quicksort|sorting|bfs|dfs)\b/i;
const CODE_LANG_FRAMEWORK_RE = /\b(?:typescript|javascript|python|rust|golang|c\+\+|cpp|c#|java|scala|kotlin|swift|ruby|php|react|vue|angular|svelte|next\.js|node\.js|express|fastapi|django|flask|graphql|dockerfile|github actions|kubernetes|k8s|css flexbox|css grid|tailwind|sql query|postgresql|sqlite|redis|mongodb)\b/i;
const CODE_TECH_TERMS_RE = /\b(?:error|bug|issue|exception|function|class|component|hook|query|schema|type|import|export|install|build|compile|syntax|loop|re-render|memory leak|thread|mutex|deadlock|concurrency|async|await|promise|callback|iterator|package|module|resolver|endpoint|route|layout|generic|workflow|search|sort|algorithm)\b/i;

const CODE_COMBINED_CI_RE = new RegExp(
  [
    /\b(?:how (?:do|can) I (?:implement|code|write|program|fix|debug|test|optimize|refactor)|fix this (?:code|bug|error|issue|exception|stack trace|syntax|crash|warning)|unit test|test suite|test case|pytest|jest|vitest|mocha|cargo test)\b/.source,
    /(?:Traceback \(most recent call last\)|TypeError:|SyntaxError:|ReferenceError:|NullPointerException|IndexOutOfBoundsException|ModuleNotFoundError:|panic:|Segmentation fault|SIGSEGV|Uncaught Error:)/.source,
    /\b(?:SELECT\s+[\s\S]{1,240}?\s+FROM|INSERT\s+INTO|UPDATE\s+[\s\S]{1,240}?\s+SET|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE)\b/.source,
    /\b(?:generic type|type alias|interface\s+[a-zA-Z_]|struct\s+[a-zA-Z_]|impl\s+[a-zA-Z_]|Arc<Mutex<|RwLock<|flexbox layout|token bucket|lru cache|event emitter|pull request|git commit|git diff)\b/.source
  ].join('|'),
  'i'
);

const CODE_COMBINED_CS_RE = /\b(?:def\s+[a-zA-Z_]\w*|function\s+[a-zA-Z_]\w*|const\s+[a-zA-Z_]\w*\s*=|let\s+[a-zA-Z_]\w*\s*=|var\s+[a-zA-Z_]\w*\s*=|fn\s+[a-zA-Z_]\w*|func\s+[a-zA-Z_]\w*|class\s+[a-zA-Z_]\w*|public\s+(?:static\s+)?void|import\s+[^\n]{1,120}\s+from|from\s+[^\n]{1,120}\s+import|#include\s+<|require\(['"][^'"]{1,120}['"]\)|package\s+main|console\.log\(|println!|std::|fmt\.Println|use[A-Z][a-zA-Z0-9_]+)\b/;

const DEEP_REASONING_RE = new RegExp(
  [
    /\b(?:net income|operating income|operating margin|gross margin|fiscal year|cash flow[s]?|diluted eps|earnings per share|balance sheet|sec filing|10-k|10-q|ebitda|ebit|cagr|amortization|depreciation|discounted cash flow|dcf model|valuation model|p\/e ratio|return on equity|roe|roic|capital expenditure|capex|free cash flow|wacc|working capital|covenant breach)\b/.source,
    /\b(?:formal (?:deductive )?logic proof|formal mathematical proof|deductive reasoning|proof by contradiction|mathematical proof|game theory|nash equilibrium|prisoner's dilemma|pareto optimal(?:ity|)?|counterfactual analysis|formal logic proof|first-order logic|syllogism proof|grim trigger|tit-for-tat|first fundamental theorem)\b/.source
  ].join('|'),
  'i'
);

const GENERAL_FAST_COMBINED_RE = new RegExp(
  [
    /\b(?:translate|translation|translated|translating)\b/.source,
    /\b(?:how do you say\b[\s\S]{1,160}?\bin (?:spanish|french|german|chinese|japanese|russian|italian|portuguese|hindi|arabic|korean|dutch|swedish|latin))\b/.source,
    /\b(?:in (?:spanish|french|german|chinese|japanese|russian|italian|portuguese|hindi|arabic|korean|dutch|swedish|latin):)\b/.source,
    /\b(?:from\s+\w+\s+(?:to|into)\s+(?:spanish|french|german|chinese|japanese|russian|italian|portuguese|hindi|arabic|korean|dutch|swedish|latin|english))\b/.source,
    /\b(?:(?:to|into)\s+(?:spanish|french|german|chinese|japanese|russian|italian|portuguese|hindi|arabic|korean|dutch|swedish|latin))\b/.source,
    /\b(?:geograph|latitude|longitude|elevation|continent|bordering countries|countries that border|capital of|mountain range|peninsula)\b/.source,
    /\b(?:patient|symptom|clinic|diagnos|syndrome|treatment|disease|prescribe|prognosis|pharmacolog(?:y|ical)|lyme disease)\b/.source,
    /\b(?:write (?:a|an)?(?:\s+\w+)?\s*(?:poem|story|haiku|essay|song|dialogue|letter|email))\b/.source,
    /\b(?:grammar|proofread|correct the grammar|spelling|rephrase|paraphrase)\b/.source,
    /\b(?:narrative|protagonist|storyline|allegory|metaphor)\b/.source,
    /\b(?:author|poet|novelist|playwright)\s+(?:wrote|penned|composed|published|authored)\b/.source,
    /\b(?:literary|novel|poem|playwright|poetry|biography|novelist)\b/.source,
    /\b(?:does sentence a imply|same sense of the word|entailment)\b/.source
  ].join('|'),
  'i'
);

const MULTIPLE_CHOICE_OPTIONS_RE = /\b(?:options|selections|choices|alternatives):\s*\n?\s*[a-d]\.|\n\s*[a-d]\.\s+\S+/i;
const OPEN_ENDED_TRIVIA_RE = /\b(?:who (?:was|wrote|directed|composed|invented|discovered)|what is the (?:capital of|[\w-]+\s+capital)|which country|what city)\b/i;

const EXPLICIT_STEM_RE = new RegExp(
  [
    /(?:\\frac|\\sum|\\sqrt|\\int|\\times|\\pm|equation|theorem|polynomial|integral|derivative|matrix|vector|logarithm|physics|chemistry|biology|astronomy|thermodynamics|quantum|velocity|acceleration|voltage|current|resistance|molecule|atom|gravit|calculus|algebra|geometry|trigonometry|logarithmic|exponential|mitochondria|photosynthesis|eukaryot|orbital|fluid flow|navier-stokes|stefan-boltzmann|heisenberg|half-life|carbon-14|kinetic energy|entropy|eigenvalue|eigenvector)\b/.source,
    /\b(?:utilitarianism|deontolog|epistemolog|syllogism|deductive logic|inductive logic|probability|calculate|derivative|integral|force|newtons|mass|acceleration|speed of sound|blackbody|dark energy|cosmological constant|mitosis|meiosis|dna|crispr)\b/.source
  ].join('|'),
  'i'
);
const ARITHMETIC_EXPRESSION_RE = /\b\d+\s*[+\-*/^=]\s*\d+\b/;

/**
 * O(N) linear-time check for JSON-like object structures `{"..."...}`.
 * Replaces `/\{[\s\S]*"[\s\S]*\}/` to eliminate O(N^2) catastrophic backtracking on unclosed braces.
 */
function hasJsonLikeStructure(text: string): boolean {
  const firstBrace = text.indexOf('{');
  if (firstBrace === -1) return false;
  const lastBrace = text.lastIndexOf('}');
  if (lastBrace <= firstBrace) return false;
  const quoteIdx = text.indexOf('"', firstBrace + 1);
  return quoteIdx !== -1 && quoteIdx < lastBrace;
}

/**
 * Extracts text from a string, multi-part content array, or primitive value.
 */
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
 * Safely extracts text from Message[] or string input without throwing on malformed items.
 */
function extractFullText(messages: Message[] | string, shouldPrune?: boolean): string {
  if (!messages) return '';
  let fullText: string;
  if (typeof messages === 'string') {
    fullText = messages;
  } else if (Array.isArray(messages)) {
    const len = messages.length;
    if (len === 0) {
      fullText = '';
    } else if (len === 1) {
      fullText = extractMessageContent(messages[0]?.content);
    } else {
      let acc = '';
      for (let i = 0; i < len; i++) {
        const part = extractMessageContent(messages[i]?.content);
        acc = i === 0 ? part : `${acc}\n${part}`;
      }
      fullText = acc;
    }
  } else {
    fullText = String(messages);
  }

  if (shouldPrune) {
    fullText = pruneText(fullText);
  }
  return fullText;
}

/**
 * Knowledge Boundary Router.
 * Determines if a query represents a self-contained "closed-world" task
 * (e.g. arithmetic, unit conversion, code syntax translation, regex, dictionary lookup)
 * that does not require open-world reasoning and is degraded by cognitive context bloat.
 */
export function detectKnowledgeBoundary(text: string): 'closed' | 'open' {
  if (!text || typeof text !== 'string') return 'open';
  const clean = text.trim();
  if (!clean) return 'open';
  return CLOSED_WORLD_RE.test(clean) ? 'closed' : 'open';
}

/**
 * Lightweight, zero-dependency text cleaner for pre-routing prompt compaction.
 */
export function pruneText(text: string): string {
  if (!text || typeof text !== 'string') return '';
  let cleaned = text.replace(PRUNE_CALLBACK_RE, '').trim();
  
  // Strip common conversational prefix/suffix filler patterns (bounded iterative pass)
  let iterations = 0;
  while (iterations < 8) {
    iterations++;
    const before = cleaned;
    cleaned = cleaned
      .replace(PRUNE_GREETING_RE, '')
      .replace(PRUNE_POLITE_PREFIX_RE, '')
      .replace(PRUNE_SIGN_OFF_RE, '')
      .trim();
    if (before === cleaned) break;
  }

  // Deduplicate consecutive whitespace and punctuation
  return cleaned.replace(PRUNE_WHITESPACE_RE, ' ').replace(PRUNE_PUNCT_RE, '$1').trim();
}

function computeComplexityScoreFromText(fullText: string, options?: ClassifierOptions, precomputedBoundary?: 'closed' | 'open'): number {
  const lengthThreshold = options?.lengthThreshold || 2000;

  // Knowledge boundary check (closed-world task deduction)
  if (options?.knowledgeBoundaryGating !== false && fullText.length < 500) {
    const boundary = precomputedBoundary ?? detectKnowledgeBoundary(fullText);
    if (boundary === 'closed') {
      return 0.15;
    }
  }

  let score = 0.0;

  // Length scoring (scaled up to 0.60)
  const lengthRatio = Math.min(1.0, fullText.length / lengthThreshold);
  score += lengthRatio * 0.60;

  // Structural markers (with O(1) character presence guards)
  if (fullText.includes('```')) score += 0.30;
  if (score < 1.0 && fullText.includes('<') && XML_TAG_MARKER_RE.test(fullText)) score += 0.20;
  if (score < 1.0 && hasJsonLikeStructure(fullText)) score += 0.20;

  // Moderate cognitive / comparative inquiry (borderline indicators)
  if (score < 1.0 && COMPARATIVE_INQUIRY_RE.test(fullText)) {
    score += 0.25;
  }

  // High cognitive complexity verbs
  if (score < 1.0 && HIGH_COMPLEXITY_SCORE_VERBS_RE.test(fullText)) {
    score += 0.35;
  }

  // Custom regex rules
  if (score < 1.0 && options?.customRules) {
    for (let i = 0; i < options.customRules.length; i++) {
      const rule = options.customRules[i];
      if (!(rule instanceof RegExp)) continue;
      rule.lastIndex = 0;
      if (rule.test(fullText)) {
        score += 0.35;
        break;
      }
    }
  }

  return Math.min(1.0, Math.max(0.0, score));
}

/**
 * Continuous complexity scorer [0.0, 1.0].
 * Provides fine-grained probability for speculative hedging or routing gates.
 */
export function evaluateComplexityScore(messages: Message[] | string, options?: ClassifierOptions): number {
  const fullText = extractFullText(messages, options?.prunePreRouting);
  return computeComplexityScoreFromText(fullText, options);
}

/**
 * A fast, <50ms heuristic classifier to predict if a prompt is "simple" or "complex".
 * Evaluates message length and structural markers (code blocks, XML, JSON).
 */
export function isComplexPrompt(messages: Message[] | string, options?: ClassifierOptions): boolean {
  const lengthThreshold = options?.lengthThreshold || 2000;
  const fullText = extractFullText(messages, options?.prunePreRouting);

  if (fullText.length > lengthThreshold) {
    return true;
  }

  if (
    fullText.includes('```') ||
    (fullText.includes('<') && XML_TAG_MARKER_RE.test(fullText)) ||
    hasJsonLikeStructure(fullText) ||
    HIGH_COMPLEXITY_PROMPT_VERBS_RE.test(fullText)
  ) {
    return true;
  }

  if (options?.customRules) {
    for (let i = 0; i < options.customRules.length; i++) {
      const rule = options.customRules[i];
      if (!(rule instanceof RegExp)) continue;
      rule.lastIndex = 0;
      if (rule.test(fullText)) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Internal fast-path domain matcher returning the matched SpecialistRole or undefined on L1 miss.
 */
function matchSpecialistRoleFromText(fullText: string, options?: ClassifierOptions, precomputedBoundary?: 'closed' | 'open'): SpecialistRole | undefined {
  // 0. Custom Specialist Overrides (User-defined domain rules)
  if (options?.customSpecialistRules && options.customSpecialistRules.length > 0) {
    for (let i = 0; i < options.customSpecialistRules.length; i++) {
      const rule = options.customSpecialistRules[i];
      if (!rule || !(rule.pattern instanceof RegExp)) continue;
      rule.pattern.lastIndex = 0;
      if (rule.pattern.test(fullText)) {
        return rule.role;
      }
    }
  }

  // 1. Paragraph Reading Comprehension & Verification (qwen3-235b)
  if (RC_READING_COMPREHENSION_RE.test(fullText)) {
    return 'comprehension_rc';
  }

  // 2. Chess & Spatial Board Games (Qwen3-Coder-Next via games_spatial)
  if (
    CHESS_KEYWORDS_RE.test(fullText) ||
    (fullText.includes('-') && CHESS_COORD_RE.test(fullText)) ||
    CHESS_ALGEBRAIC_RE.test(fullText)
  ) {
    return 'games_spatial';
  }

  // 3. Code Generation, Refactoring & Algorithm Synthesis (Qwen3-Coder-Next)
  if (
    fullText.includes('```') ||
    CODE_INTENT_RE.test(fullText) ||
    CODE_COMBINED_CI_RE.test(fullText) ||
    CODE_COMBINED_CS_RE.test(fullText) ||
    (CODE_LANG_FRAMEWORK_RE.test(fullText) && CODE_TECH_TERMS_RE.test(fullText))
  ) {
    return 'code';
  }

  // 4. Financial Statements, Balance Sheets & Formal Proofs (deepseek-v4-pro)
  if (DEEP_REASONING_RE.test(fullText)) {
    return 'reasoning_deep';
  }

  // 5. Linguistics, Translation, Geography, Medicine, Open-ended Trivia, Entailment
  // (Empirically superior on google/gemini-3.1-flash-lite)
  if (GENERAL_FAST_COMBINED_RE.test(fullText)) {
    return 'general_fast';
  }

  // Open-ended trivia without multiple choice options
  const hasOptions = fullText.includes('.') && MULTIPLE_CHOICE_OPTIONS_RE.test(fullText);
  if (!hasOptions && OPEN_ENDED_TRIVIA_RE.test(fullText)) {
    return 'general_fast';
  }

  // 6. Explicit STEM / Math / Logic / Science
  if (
    EXPLICIT_STEM_RE.test(fullText) ||
    ARITHMETIC_EXPRESSION_RE.test(fullText) ||
    hasOptions
  ) {
    return 'factual_stem';
  }

  // 7. Knowledge Boundary: Closed-World Transformations (unit conversion, regex, translation, formatting)
  const boundary = precomputedBoundary ?? detectKnowledgeBoundary(fullText);
  if (boundary === 'closed') {
    return 'general_fast';
  }

  return undefined;
}

/**
 * L1 Pre-Router Gate.
 * Evaluates in <15 microseconds whether an incoming prompt has a deterministic
 * structural or domain footprint (code, SQL, math, chess, closed-world transform)
 * suitable for immediate fast-path dispatch, or whether it should be delegated
 * to an L2 neural/embedding router or frontier model.
 */
export function classifyPreRoute(messages: Message[] | string, options?: ClassifierOptions): PreRouteResult {
  const fullText = extractFullText(messages, options?.prunePreRouting);
  const boundary = fullText.length < 500 ? detectKnowledgeBoundary(fullText) : undefined;
  const complexityScore = computeComplexityScoreFromText(fullText, options, boundary);
  const role = matchSpecialistRoleFromText(fullText, options, boundary);

  if (role !== undefined) {
    return {
      isFastPath: true,
      role,
      confidence: 'high',
      complexityScore,
      suggestedAction: 'dispatch_specialist'
    };
  }

  // 8. Default Unstructured / Ambiguous Chat
  // When queries lack clear structural/syntactic domain signatures,
  // pass through to L2 Neural / Embedding Router.
  return {
    isFastPath: false,
    role: undefined,
    confidence: complexityScore >= 0.35 && complexityScore <= 0.65 ? 'borderline' : 'unstructured',
    complexityScore,
    suggestedAction: 'delegate_to_l2'
  };
}

/**
 * Classifies a prompt into one of 7 domain specialist roles optimized
 * for sub-50ms multi-model swarm routing.
 */
export function classifySpecialistRole(messages: Message[] | string, options?: ClassifierOptions): SpecialistRole {
  const fullText = extractFullText(messages, options?.prunePreRouting);
  return matchSpecialistRoleFromText(fullText, options) ?? 'factual_stem';
}
