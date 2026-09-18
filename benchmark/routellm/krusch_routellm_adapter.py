# SPDX-License-Identifier: Apache-2.0
"""
Krusch RouteLLM Router Adapter.
Subclasses RouteLLM's Router base class and provides sub-millisecond calibrated
win-rate estimation between weak (cheap/fast) and strong (frontier) models.
"""

import re
import sys
from pathlib import Path

# Add RouteLLM to sys.path
ROUTELM_DIR = Path(__file__).resolve().parent / "RouteLLM"
if str(ROUTELM_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTELM_DIR))

from routellm.routers.routers import Router, no_parallel


@no_parallel
class KruschRouteLLMRouter(Router):
    """
    Krusch Cascade Router adapter for RouteLLM.
    Evaluates lexical, syntactic, and structural signals to estimate the probability
    that a query requires the strong frontier model (e.g. GPT-4) vs. the weak model (e.g. Mixtral/Llama-3).
    """

    def __init__(self, **kwargs):
        super().__init__()

    def detect_knowledge_boundary(self, text: str) -> str:
        """
        Knowledge Boundary Router.
        Determines if a query is a self-contained closed-world task that can be easily
        solved by the weak model without requiring open-world frontier reasoning.
        """
        if not text:
            return "closed"
        clean = text.strip().lower()

        closed_world_patterns = [
            r"^(?:translate|convert|calculate|format|prettify|lint|capitalize|lowercase|reverse)\b",
            r"\b(?:regex|regular expression|json format|csv format|unit conversion|celsius to fahrenheit|miles to km)\b",
            r"^(?:what is|solve)\s+[\d\s+\-*/^().=]+$",
            r"\b(?:dictionary definition|synonym for|antonym for|spelling of)\b",
        ]
        for pat in closed_world_patterns:
            if re.search(pat, clean):
                return "closed"
        return "open"

    def calculate_strong_win_rate(self, prompt: str) -> float:
        """
        Returns a float between 0.0 and 1.0 representing the predicted probability
        that the strong model is required to successfully answer the prompt.
        """
        p = prompt.strip().lower()
        text_len = len(prompt)

        # 1. Closed-world deduction
        if self.detect_knowledge_boundary(prompt) == "closed" and text_len < 300:
            return 0.15

        score = 0.0

        # 2. Length-based scaling
        length_ratio = min(1.0, text_len / 2500.0)
        score += length_ratio * 0.35

        # 3. Code, syntax, and structured markup
        if "```" in prompt or "def " in prompt or "class " in prompt:
            score += 0.25
        if re.search(r"py[th]{2}[on]{1,2}", p) or "source code" in p or "function" in p:
            score += 0.20
        if "{" in prompt and "}" in prompt and ('"' in prompt or ":" in prompt):
            score += 0.15

        # 4. Multi-step mathematical reasoning & competition arithmetic
        math_signals = (
            "calculate", "probability", "integer", "equation", "theorem",
            "polynomial", "integral", "matrix", "geometry", "fraction",
            "algebra", "combinatorics", "perimeter", "hypotenuse", "ratio"
        )
        math_matches = sum(1 for term in math_signals if term in p)
        if math_matches >= 2:
            score += 0.30
        elif math_matches == 1:
            score += 0.15

        # 5. Cognitive complexity and analytical reasoning verbs
        high_cognition = (
            "analyze", "evaluate", "synthesize", "dilemma", "philosophical",
            "ethical", "counterfactual", "compare and contrast", "implication",
            "critique", "underlying cause", "distinguish between"
        )
        if any(term in p for term in high_cognition):
            score += 0.30

        moderate_cognition = (
            "why did", "explain how", "what led to", "in what way",
            "differences between", "tradeoffs", "pros and cons"
        )
        if any(term in p for term in moderate_cognition):
            score += 0.15

        # 6. Reading comprehension / truth evaluation
        if "paragraph" in p and any(k in p for k in ("provided answer", "evaluate", "correct response")):
            score += 0.40

        # 7. Financial & corporate accounting
        if any(k in p for k in ("net income", "operating income", "fiscal year", "cash flows", "diluted eps", "balance sheet")):
            score += 0.40

        # 8. High-gap professional disciplines (law, genetics/biomedical, macro/micro economics, formal logic)
        hard_disciplines = (
            "plaintiff", "defendant", "jurisdiction", "statute", "tort", "constitutional",
            "chromosome", "allele", "genotype", "phenotype", "enzyme", "cellular",
            "elasticity", "monopoly", "inflation", "oligopoly", "fiscal policy",
            "syllogism", "validity", "premise", "deductive", "modus ponens"
        )
        if any(term in p for term in hard_disciplines):
            score += 0.35

        # 9. Low-gap memorization & social domains (sociology, marketing, world religions)
        low_gap_disciplines = (
            "sociology", "marketing", "consumer", "advertis", "hinduism", "buddhism",
            "christianity", "islam", "religion", "social group", "norm"
        )
        if any(term in p for term in low_gap_disciplines):
            score = max(0.08, score - 0.20)

        # 8. Multi-choice options detection (often factual or standard recall)
        has_options = bool(
            re.search(r"\b(?:options|selections|choices|alternatives):\s*\n?\s*[a-d]\.", p)
            or re.search(r"\n\s*[a-d]\.\s+\S+", p)
        )
        if has_options and score < 0.40:
            # Fact recall with options is often well-handled by modern fast models
            score = max(0.10, score - 0.10)

        # Micro-variance tie-breaker based on prompt hash to provide distinct quantile separation
        tie_break = ((abs(hash(p)) % 10000) / 10000.0) * 0.01
        score += tie_break

        # Calibrate within [0.05, 0.95]
        return float(min(0.95, max(0.05, score)))
