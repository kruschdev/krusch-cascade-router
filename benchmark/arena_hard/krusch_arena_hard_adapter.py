# SPDX-FileCopyrightText: Copyright contributors to the Krusch Cascade Router project
# SPDX-License-Identifier: MIT

"""
Krusch Cascade Router Adapter for LMSYS Arena-Hard-Auto Benchmark.
Evaluates deterministic zero-latency multi-specialist routing and threshold-gated cascades
across Arena-Hard-Auto v0.1 (500 prompts) and v2.0 (750 prompts).
"""

import os
import re
from typing import Any, Dict, Optional, Tuple


class KruschArenaHardAdapter:
    """
    Adapter implementing Krusch Cascade Router's multi-specialist and complexity-gated
    routing logic for the LMSYS Arena-Hard benchmark suite.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate_complexity(self, prompt: str) -> float:
        """
        Sub-millisecond heuristic continuous complexity scoring in [0.0, 1.0].
        Computes composite complexity from length, code syntax, mathematical notation,
        multi-step constraints, and structural formatting requirements.
        """
        p = prompt.lower()
        score = 0.15  # baseline complexity

        # Length factor (Arena-Hard prompts range from short puzzles to long technical specs)
        char_len = len(prompt)
        if char_len > 2500:
            score += 0.35
        elif char_len > 1200:
            score += 0.25
        elif char_len > 500:
            score += 0.15
        elif char_len > 200:
            score += 0.08

        # Code syntax & algorithmic requirements
        if any(tok in prompt for tok in ["```", "def ", "class ", "function(", "public class", "fn ", "impl "]):
            score += 0.25
        if any(tok in p for tok in ["algorithm", "time complexity", "space complexity", "dynamic programming", "recursion", "leetc"]):
            score += 0.20

        # Mathematical & formal logic formulation
        if any(tok in prompt for tok in ["\\frac", "\\sqrt", "\\sum", "\\int", "\\mathbf", "\\in", "\\forall", "\\exists"]):
            score += 0.25
        if any(tok in p for tok in ["prove that", "theorem", "lemma", "differential equation", "eigenvalue", "matrix multiplication", "combinatorics"]):
            score += 0.20

        # Multi-step instructions & high-constraint tasks
        instruction_markers = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|\-|\*)\s+", prompt))
        if instruction_markers >= 5:
            score += 0.25
        elif instruction_markers >= 3:
            score += 0.15

        # Strict formatting / JSON / schema output
        if any(tok in p for tok in ["json format", "valid json", "schema", "output format:", "xml tags", "csv format"]):
            score += 0.15

        # Puzzles & games with strict rule state
        if any(tok in p for tok in ["chess", "fen", "pgn", "sudoku", "grid", "rubik", "cipher", "decrypt"]):
            score += 0.20

        return min(max(score, 0.0), 1.0)

    def classify_domain(self, prompt: str) -> str:
        """
        Classifies prompt into one of 6 specialized capability domains:
        - code: Software engineering, programming, algorithm synthesis.
        - games_spatial: Chess, spatial puzzles, board states.
        - comprehension_rc: Long reading comprehension, document analysis, verification.
        - reasoning_deep: Financial analysis, multi-step math/proofs, logic chains.
        - general_fast: Creative writing, translation, geography, medical, trivia.
        - factual_stem: STEM sciences, physics, biology, general academic QA.
        """
        p = prompt.lower()

        # 1. Code generation & algorithms
        if (
            "```" in prompt
            or any(tok in prompt for tok in ["def ", "class ", "import ", "function(", "SELECT ", "CREATE TABLE"])
            or any(tok in p for tok in ["python", "javascript", "typescript", "c++", "rust", "golang", "sql query", "write a function", "write a script", "regex", "regular expression"])
        ):
            return "code"

        # 2. Chess & Spatial Puzzles
        if (
            any(tok in p for tok in ["chess move", "chess game", "fen", "pgn", "checkmate", "castling", "board position"])
            or re.search(r"\b[a-h][1-8]-[a-h][1-8]\b", p)
        ):
            return "games_spatial"

        # 3. Financial Statements / Accounting
        if any(tok in p for tok in ["net income", "balance sheet", "operating income", "cash flows", "diluted eps", "ebitda", "sec filing", "earnings per share"]):
            return "reasoning_deep"

        # 4. Long Reading Comprehension & Text Verification
        if (
            len(prompt) > 1500
            and any(tok in p for tok in ["based on the text", "read the passage", "according to the excerpt", "summarize the article", "evaluate the answer", "comprehension"])
        ):
            return "comprehension_rc"

        # 5. Linguistics, Translation, Medical, Geography, Creative Writing
        if (
            any(tok in p for tok in ["translate", "translation", "german", "french", "spanish", "chinese", "japanese", "russian"])
            or any(tok in p for tok in ["symptom", "patient", "clinical", "diagnosis", "therapy", "dosage", "medical"])
            or any(tok in p for tok in ["capital of", "latitude", "longitude", "continent", "geography", "country borders"])
            or any(tok in p for tok in ["write a story", "write a poem", "creative writing", "dialogue between", "screenplay"])
        ):
            return "general_fast"

        # 6. Deep Multi-Step Reasoning / Math
        if any(tok in p for tok in ["prove that", "step by step proof", "derivation", "integrate", "solve the equation"]):
            return "reasoning_deep"

        # 7. Default Factual STEM
        return "factual_stem"

    def route_binary(
        self,
        prompt: str,
        weak_model: str,
        strong_model: str,
        threshold: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Binary routing between weak/cheap model and strong/expensive frontier model.
        """
        thresh = self.threshold if threshold is None else threshold
        complexity = self.evaluate_complexity(prompt)
        domain = self.classify_domain(prompt)

        # Complex domains (code, complex reasoning) or complexity exceeding threshold
        # escalate to the strong model; simple, low-complexity queries stay with weak model.
        is_hard_domain = domain in ("code", "reasoning_deep")
        should_escalate = (complexity >= thresh) or (is_hard_domain and complexity >= (thresh * 0.75))

        chosen_model = strong_model if should_escalate else weak_model

        meta = {
            "complexity": complexity,
            "domain": domain,
            "threshold": thresh,
            "escalated": should_escalate,
            "model": chosen_model,
        }
        return chosen_model, meta
