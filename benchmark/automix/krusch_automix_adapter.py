# SPDX-License-Identifier: Apache-2.0
"""
Krusch AutoMix Adapter (Early-Exit Cascade & Self-Verification Routing).
Implements context-complexity scoring, knowledge-boundary gating, and
calibrated verification thresholding for cascading between SLM (13B) and LLM (70B).
"""

import re
from typing import List, Optional, Union

import numpy as np


class KruschAutomixAdapter:
    """
    Krusch Cascade Router adapter for AutoMix benchmark (Google Research / NeurIPS 2024).
    Determines whether to accept the small model (13B) draft answer or cascade to
    the large frontier model (70B).
    """

    def __init__(self, threshold: float = 0.45, **kwargs):
        self.threshold = threshold

    def detect_knowledge_boundary(self, text: str) -> str:
        """
        Knowledge Boundary Router (arXiv: 2608.23982).
        Flags self-contained, closed-world questions resolvable by lightweight models.
        """
        if not text:
            return "closed"
        clean = text.strip().lower()

        closed_patterns = [
            r"^(?:who|where|when|what year|what date|what time|name of|how many)\b",
            r"\b(?:yes or no|true or false)\b",
        ]
        for pat in closed_patterns:
            if re.search(pat, clean):
                return "closed"
        return "open"

    def estimate_complexity(self, context: str, question: str) -> float:
        """
        Computes composite complexity score C in [0.0, 1.0] from question and context length.
        """
        q_clean = (question or "").strip().lower()
        ctx_len = len(context or "")
        score = 0.0

        # Context length scaling
        if ctx_len > 4000:
            score += 0.30
        elif ctx_len > 1500:
            score += 0.15

        # Question complexity verbs
        high_cognition = (
            "why", "how", "explain", "analyze", "imply", "conclude",
            "relationship between", "cause", "differ", "compare"
        )
        if any(term in q_clean for term in high_cognition):
            score += 0.25

        # Multi-clause or negative constraints
        if any(term in q_clean for term in ["not", "except", "unless", "neither", "both"]):
            score += 0.15

        # Closed boundary discount
        if self.detect_knowledge_boundary(question) == "closed":
            score -= 0.20

        return max(0.0, min(1.0, score))

    def detect_slm_degeneracy(self, slm_answer: str, dataset: str = None) -> bool:
        """
        Detects degenerate, repetitive, or unanswerable outputs from the small model.
        """
        if not slm_answer or len(slm_answer.strip()) == 0:
            return True

        clean = slm_answer.strip().lower()

        # Unanswerable signal (especially prominent in QASPER)
        if "unanswerable" in clean or "cannot be determined" in clean or "not mentioned" in clean:
            return True

        # Repetition detection (e.g. loops in SLM outputs)
        words = clean.split()
        if len(words) >= 6:
            # Check 2-gram repetition
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            if len(bigrams) > 0 and (len(bigrams) - len(set(bigrams))) / len(bigrams) > 0.4:
                return True

        return False

    def should_cascade(
        self,
        context: str,
        question: str,
        slm_answer: str,
        p_ver: Optional[float] = None,
        threshold: Optional[float] = None,
        dataset: Optional[str] = None,
    ) -> bool:
        """
        Decides whether to escalate query to the large model (70B).
        Returns True (to_retry / cascade) or False (accept SLM).
        """
        tau = threshold if threshold is not None else self.threshold

        # 1. Immediate cascade on SLM output degeneracy or unanswerability
        if self.detect_slm_degeneracy(slm_answer, dataset=dataset):
            return True

        # 2. Heuristic complexity & knowledge boundary
        comp = self.estimate_complexity(context, question)

        # 3. If self-verification signal is provided
        if p_ver is not None and not np.isnan(p_ver):
            # Calibrate confidence by prompt complexity:
            # Complex queries require higher verification confidence to avoid cascade
            calibrated_conf = p_ver * (1.0 - 0.30 * comp)
            return bool(calibrated_conf < tau)

        # 4. Pure heuristic mode (zero-token verifier)
        # Cascade on high complexity tasks when boundary is open
        return bool(comp > 0.55)

    def batch_should_cascade(
        self,
        contexts: List[str],
        questions: List[str],
        slm_answers: List[str],
        p_vers: Optional[List[float]] = None,
        threshold: Optional[float] = None,
        dataset: Optional[str] = None,
    ) -> np.ndarray:
        """
        Batch evaluation returning boolean array (True = cascade to LLM).
        """
        n = len(questions)
        p_vals = p_vers if p_vers is not None else [None] * n
        results = [
            self.should_cascade(
                contexts[i], questions[i], slm_answers[i], p_vals[i], threshold=threshold, dataset=dataset
            )
            for i in range(n)
        ]
        return np.array(results, dtype=bool)
