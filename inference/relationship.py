"""
Relationship Classification Module
==================================

Classifies the logical relationship between markets in a family:
- DATE_VARIANT: Nested deadlines (e.g., "by March", "by April")
- MUTUALLY_EXCLUSIVE: At most one outcome can be YES
- EXHAUSTIVE: At least one outcome must be YES
- EXCLUSIVE_AND_EXHAUSTIVE: Exactly one outcome will be YES
- CONDITIONAL: Implication chains

The relationship type determines which constraint checks apply.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from inference.models import (
    MarketFamily,
    PolymarketMarket,
    RelationshipType,
)

logger = logging.getLogger("PolyMangoBot.inference.relationship")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RelationshipConfig:
    """Configuration for relationship classification"""
    # Thresholds for heuristic classification
    date_overlap_threshold: float = 0.8  # Token overlap for date variants
    exclusive_keyword_weight: float = 0.3
    exhaustive_keyword_weight: float = 0.3

    # Minimum confidence to classify
    min_confidence: float = 0.6


# =============================================================================
# KEYWORD PATTERNS
# =============================================================================

# Keywords indicating mutual exclusivity
EXCLUSIVE_KEYWORDS = [
    r"\bor\b",           # "A or B"
    r"\bvs\.?\b",        # "A vs B"
    r"\bversus\b",
    r"\bwhich\b",        # "Which candidate will win?"
    r"\bwinner\b",       # Only one winner
    r"\bfirst\b",        # First to achieve something
]

# Keywords indicating exhaustive set
EXHAUSTIVE_KEYWORDS = [
    r"\bwill\s+.+\s+happen\b",
    r"\bwhen\s+will\b",
    r"\bhow\s+many\b",
    r"\btotal\b",
    r"\ball\b",
    r"\bevery\b",
]

# Keywords indicating conditional relationship
CONDITIONAL_KEYWORDS = [
    r"\bif\b",
    r"\bassuming\b",
    r"\bgiven\s+that\b",
    r"\bconditional\s+on\b",
]

# Date-variant patterns
DATE_VARIANT_PATTERNS = [
    r"\bby\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
    r"\bby\s+Q[1-4]\b",
    r"\bby\s+\d{4}\b",
    r"\bby\s+end\s+of\b",
    r"\bbefore\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
]


# =============================================================================
# RELATIONSHIP CLASSIFIER
# =============================================================================

class RelationshipClassifier:
    """
    Classifies the logical relationship between markets in a family.

    Uses a combination of:
    1. Date extraction and comparison
    2. Keyword pattern matching
    3. Question structure analysis
    4. LLM fallback (optional)
    """

    def __init__(self, config: Optional[RelationshipConfig] = None):
        self.config = config or RelationshipConfig()

    def classify(self, family: MarketFamily) -> tuple[RelationshipType, float]:
        """
        Classify the relationship type of a market family.

        Args:
            family: MarketFamily to classify

        Returns:
            Tuple of (RelationshipType, confidence)
        """
        markets = family.markets

        if len(markets) < 2:
            return RelationshipType.UNKNOWN, 0.0

        # Check for date-variant relationship first
        date_score = self._check_date_variant(markets)
        if date_score >= self.config.min_confidence:
            return RelationshipType.DATE_VARIANT, date_score

        # Check for exclusive/exhaustive relationships
        exclusive_score = self._check_exclusive(markets)
        exhaustive_score = self._check_exhaustive(markets)

        # Determine final classification
        if exclusive_score >= self.config.min_confidence and exhaustive_score >= self.config.min_confidence:
            return RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE, min(exclusive_score, exhaustive_score)
        elif exclusive_score >= self.config.min_confidence:
            return RelationshipType.MUTUALLY_EXCLUSIVE, exclusive_score
        elif exhaustive_score >= self.config.min_confidence:
            return RelationshipType.EXHAUSTIVE, exhaustive_score

        # Check for conditional
        conditional_score = self._check_conditional(markets)
        if conditional_score >= self.config.min_confidence:
            return RelationshipType.CONDITIONAL, conditional_score

        return RelationshipType.UNKNOWN, 0.0

    def _check_date_variant(self, markets: list[PolymarketMarket]) -> float:
        """
        Check if markets form a date-variant family.

        Date-variant: Same event with different deadlines.
        E.g., "BTC $100k by March" and "BTC $100k by April"
        """
        # Count markets with deadlines
        markets_with_dates = [m for m in markets if m.deadline]

        if len(markets_with_dates) < 2:
            return 0.0

        # Check if deadlines are distinct
        deadlines = {m.deadline for m in markets_with_dates}
        if len(deadlines) < 2:
            return 0.0

        # Check question similarity (excluding date parts)
        base_questions = []
        for m in markets:
            # Remove date patterns from question
            q = m.question.lower()
            for pattern in DATE_VARIANT_PATTERNS:
                q = re.sub(pattern, "", q, flags=re.IGNORECASE)
            q = re.sub(r"\d{4}", "", q)  # Remove years
            q = re.sub(r"\s+", " ", q).strip()
            base_questions.append(set(q.split()))

        # Calculate pairwise similarity
        if len(base_questions) < 2:
            return 0.0

        similarities = []
        for i in range(len(base_questions)):
            for j in range(i + 1, len(base_questions)):
                q1, q2 = base_questions[i], base_questions[j]
                if q1 and q2:
                    overlap = len(q1 & q2) / len(q1 | q2)
                    similarities.append(overlap)

        if not similarities:
            return 0.0

        avg_similarity = sum(similarities) / len(similarities)

        # High similarity + distinct dates = date variant
        if avg_similarity >= self.config.date_overlap_threshold:
            return min(0.95, avg_similarity + 0.1)

        return avg_similarity * 0.8

    def _check_exclusive(self, markets: list[PolymarketMarket]) -> float:
        """
        Check if markets are mutually exclusive.

        Mutually exclusive: At most one outcome can be YES.
        """
        score = 0.0

        # Check for exclusive keywords in questions
        questions = " ".join(m.question.lower() for m in markets)
        keyword_matches = sum(
            1 for pattern in EXCLUSIVE_KEYWORDS
            if re.search(pattern, questions, re.IGNORECASE)
        )
        score += keyword_matches * self.config.exclusive_keyword_weight

        # Check if questions follow "X or Y" pattern
        for m in markets:
            if re.search(r"\bor\b", m.question, re.IGNORECASE):
                score += 0.2

        # Check if grouped as multiple choice (strong signal)
        if any(m.group_slug for m in markets):
            group_slugs = {m.group_slug for m in markets if m.group_slug}
            if len(group_slugs) == 1:
                # Same group_slug is a very strong signal for exclusivity
                score += 0.5

        # Check YES price sum
        yes_sum = sum(m.yes_price for m in markets)
        # Sum > 1.0 indicates arbitrage opportunity if exclusive
        if yes_sum > 1.02:  # Clear violation
            score += 0.3
        elif yes_sum > 1.0:
            score += 0.2
        # Sum close to 1.0 also suggests exclusive (properly priced)
        elif 0.95 <= yes_sum <= 1.05:
            score += 0.15

        return min(1.0, score)

    def _check_exhaustive(self, markets: list[PolymarketMarket]) -> float:
        """
        Check if markets are exhaustive.

        Exhaustive: At least one outcome must be YES.
        """
        score = 0.0

        # Check for exhaustive keywords
        questions = " ".join(m.question.lower() for m in markets)
        keyword_matches = sum(
            1 for pattern in EXHAUSTIVE_KEYWORDS
            if re.search(pattern, questions, re.IGNORECASE)
        )
        score += keyword_matches * self.config.exhaustive_keyword_weight

        # Check YES price sum (if close to 1.0, likely exhaustive)
        yes_sum = sum(m.yes_price for m in markets)
        if 0.9 <= yes_sum <= 1.1:
            score += 0.3
        elif 0.8 <= yes_sum <= 1.2:
            score += 0.1

        # Check if questions cover all possibilities
        # E.g., "Yes", "No" or "Candidate A", "Candidate B", "Other"
        if self._questions_cover_all(markets):
            score += 0.3

        return min(1.0, score)

    def _check_conditional(self, markets: list[PolymarketMarket]) -> float:
        """
        Check if markets have conditional relationships.

        Conditional: One market implies another.
        E.g., "Trump wins nomination" implies "Trump runs in general"
        """
        score = 0.0

        # Check for conditional keywords
        for m in markets:
            for pattern in CONDITIONAL_KEYWORDS:
                if re.search(pattern, m.question, re.IGNORECASE):
                    score += 0.3
                    break

        # Check for implication structure
        # If market A happening necessarily means market B happens
        # This is hard to detect without LLM, so give low score
        score *= 0.5

        return min(1.0, score)

    def _questions_cover_all(self, markets: list[PolymarketMarket]) -> bool:
        """
        Check if market questions appear to cover all possibilities.
        """
        questions = [m.question.lower() for m in markets]

        # Check for "other" or "none" option
        has_other = any(
            "other" in q or "none" in q or "neither" in q
            for q in questions
        )

        # Check for yes/no pair
        has_yes_no = (
            any("yes" in q for q in questions) and
            any("no" in q for q in questions)
        )

        return has_other or has_yes_no

    def classify_family(self, family: MarketFamily) -> MarketFamily:
        """
        Classify a family and update its relationship field.

        Args:
            family: MarketFamily to classify

        Returns:
            Updated MarketFamily with relationship and confidence set
        """
        relationship, confidence = self.classify(family)
        family.relationship = relationship
        family.confidence = confidence
        return family


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_date_variant_family(family: MarketFamily) -> bool:
    """Check if a family is date-variant."""
    return family.relationship == RelationshipType.DATE_VARIANT


def is_exclusive_family(family: MarketFamily) -> bool:
    """Check if a family has mutually exclusive outcomes."""
    return family.relationship in [
        RelationshipType.MUTUALLY_EXCLUSIVE,
        RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE
    ]


def is_exhaustive_family(family: MarketFamily) -> bool:
    """Check if a family has exhaustive outcomes."""
    return family.relationship in [
        RelationshipType.EXHAUSTIVE,
        RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE
    ]


def get_applicable_constraints(family: MarketFamily) -> list[str]:
    """
    Get list of constraint types that apply to this family.

    Returns:
        List of constraint names that should be checked
    """
    constraints = []

    if family.relationship == RelationshipType.DATE_VARIANT:
        constraints.extend(["monotonicity", "no_sweep"])

    if family.relationship in [
        RelationshipType.MUTUALLY_EXCLUSIVE,
        RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE
    ]:
        constraints.append("exclusive_sum")

    if family.relationship in [
        RelationshipType.EXHAUSTIVE,
        RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE
    ]:
        constraints.append("exhaustive_sum")

    return constraints
