"""
Family Discovery Module
=======================

Discovers market families (groups of related markets) using:
1. Token-based matching (fast, primary method)
2. Metadata grouping (condition_id, group_slug)
3. LLM fallback (for ambiguous cases)

A market family is a set of markets that share the same underlying
event and whose prices should satisfy logical constraints.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from inference.llm_classifier import LLMClassifier
from inference.models import (
    MarketFamily,
    PolymarketMarket,
    RelationshipType,
    TokenGroup,
)

logger = logging.getLogger("PolyMangoBot.inference.family_discovery")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FamilyDiscoveryConfig:
    """Configuration for family discovery"""
    min_family_size: int = 2
    max_family_size: int = 20
    min_token_overlap: float = 0.5  # Minimum Jaccard similarity
    use_metadata_grouping: bool = True
    use_llm_fallback: bool = False
    llm_confidence_threshold: float = 0.7


# =============================================================================
# TOKEN EXTRACTION
# =============================================================================

# Common stop words to exclude
STOP_WORDS = {
    "the", "a", "an", "is", "are", "will", "be", "to", "of", "and", "or",
    "in", "on", "at", "for", "by", "with", "from", "this", "that", "these",
    "it", "its", "as", "if", "than", "more", "less", "before", "after",
    "yes", "no", "market", "prediction"
}

# Date patterns
DATE_PATTERNS = [
    # "by March 2025", "by March 31, 2025"
    r"by\s+(\w+)\s+(\d{1,2},?\s+)?(\d{4})",
    # "before March 2025"
    r"before\s+(\w+)\s+(\d{1,2},?\s+)?(\d{4})",
    # "in Q1 2025", "in Q2"
    r"in\s+(Q[1-4])\s*(\d{4})?",
    # "March 2025", "Jan 2025"
    r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2},?\s+)?(\d{4})",
    # "2025-03-31"
    r"(\d{4})-(\d{2})-(\d{2})",
    # "end of 2025"
    r"end\s+of\s+(\d{4})",
]

# Month name to number mapping
MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12
}

# Entity patterns (names, tickers, etc.)
ENTITY_PATTERNS = [
    r"\$([A-Z]{1,5})\b",  # Stock tickers like $BTC, $ETH
    r"\b([A-Z]{2,5})/USD\b",  # Crypto pairs like BTC/USD
    r"\b(Bitcoin|Ethereum|BTC|ETH|SOL|XRP)\b",  # Crypto names
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",  # Proper names like "Donald Trump"
]

# Event patterns
EVENT_PATTERNS = [
    r"(win|wins|winning)\s+(?:the\s+)?(\w+)",  # "win the election"
    r"(reach|reaches|hit|hits)\s+\$?([\d,]+)",  # "reach $100,000"
    r"(pass|passes|approve|approves)\s+(\w+)",  # "pass legislation"
    r"(make|makes)\s+(?:the\s+)?(\w+)",  # "make playoffs"
]


def extract_tokens(question: str) -> list[str]:
    """
    Extract meaningful tokens from a market question.

    Returns normalized tokens excluding stop words and dates.
    """
    # Lowercase and remove punctuation (keep $ for tickers)
    text = question.lower()
    text = re.sub(r"[^\w\s$/-]", " ", text)

    # Tokenize
    tokens = text.split()

    # Filter stop words and short tokens
    tokens = [
        t for t in tokens
        if t not in STOP_WORDS and len(t) > 1
    ]

    return tokens


def extract_date(question: str) -> Optional[datetime]:
    """
    Extract deadline date from a market question.

    Returns datetime if found, None otherwise.
    """
    text = question

    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                groups = match.groups()

                # Handle different pattern types
                if "Q" in (groups[0] or ""):
                    # Quarter pattern
                    quarter = int(groups[0][1])
                    year = int(groups[1]) if groups[1] else datetime.now().year
                    month = (quarter - 1) * 3 + 3  # End of quarter
                    return datetime(year, month, 28)

                elif groups[0] and groups[0].lower() in MONTH_MAP:
                    # Month name pattern
                    month = MONTH_MAP[groups[0].lower()]
                    day = 28  # Default to end of month
                    year = datetime.now().year

                    # Check for day
                    if len(groups) > 1 and groups[1]:
                        day_str = re.sub(r"[^\d]", "", groups[1])
                        if day_str:
                            day = int(day_str)

                    # Check for year
                    if len(groups) > 2 and groups[2]:
                        year = int(groups[2])
                    elif len(groups) > 1 and groups[1] and groups[1].isdigit():
                        year = int(groups[1])

                    return datetime(year, month, min(day, 28))

                elif len(groups) >= 3 and all(g and g.isdigit() for g in groups[:3]):
                    # ISO date pattern
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]))

                elif "end" in question.lower() and groups[0].isdigit():
                    # End of year pattern
                    return datetime(int(groups[0]), 12, 31)

            except (ValueError, TypeError):
                continue

    return None


def extract_entity(question: str) -> Optional[str]:
    """Extract the main entity/subject from a question."""
    for pattern in ENTITY_PATTERNS:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback: look for capitalized phrases
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", question)
    if caps:
        # Return longest capitalized phrase
        longest: str = max(caps, key=len)
        return longest

    return None


def extract_event(question: str) -> Optional[str]:
    """Extract the main event/predicate from a question."""
    for pattern in EVENT_PATTERNS:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2)}".lower()

    return None


# =============================================================================
# GROUPING FUNCTIONS
# =============================================================================

def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def group_by_metadata(markets: list[PolymarketMarket]) -> dict[str, list[PolymarketMarket]]:
    """Group markets by condition_id or group_slug."""
    groups: dict[str, list[PolymarketMarket]] = defaultdict(list)

    for market in markets:
        # Prefer group_slug, fall back to condition_id
        key = market.group_slug or market.condition_id
        if key:
            groups[key].append(market)

    return dict(groups)


def group_by_tokens(
    markets: list[PolymarketMarket],
    min_overlap: float = 0.5
) -> list[TokenGroup]:
    """
    Group markets by token similarity.

    Uses hierarchical clustering with Jaccard similarity.
    """
    if not markets:
        return []

    # Extract tokens for each market
    market_tokens: list[tuple[PolymarketMarket, set[str]]] = []
    for market in markets:
        tokens = set(extract_tokens(market.question))
        market.tokens = list(tokens)
        market.entity = extract_entity(market.question)
        market.event = extract_event(market.question)
        market.deadline = extract_date(market.question)
        market_tokens.append((market, tokens))

    # Build groups using greedy clustering
    groups: list[TokenGroup] = []
    used = set()

    for i, (market1, tokens1) in enumerate(market_tokens):
        if i in used:
            continue

        group_markets = [market1]
        group_tokens = tokens1.copy()
        used.add(i)

        # Find similar markets
        for j, (market2, tokens2) in enumerate(market_tokens):
            if j in used:
                continue

            sim = jaccard_similarity(group_tokens, tokens2)
            if sim >= min_overlap:
                group_markets.append(market2)
                group_tokens |= tokens2
                used.add(j)

        if len(group_markets) >= 2:
            # Determine group key from shared entity/event
            entities = [m.entity for m in group_markets if m.entity]
            events = [m.event for m in group_markets if m.event]

            entity = max(set(entities), key=entities.count) if entities else None
            event = max(set(events), key=events.count) if events else None

            key = f"{entity or 'unknown'}:{event or 'unknown'}"

            groups.append(TokenGroup(
                key=key,
                markets=group_markets,
                confidence=sum(
                    jaccard_similarity(tokens1, set(m.tokens))
                    for m in group_markets
                ) / len(group_markets)
            ))

    return groups


# =============================================================================
# FAMILY DISCOVERY
# =============================================================================

class FamilyDiscovery:
    """
    Discovers market families using multiple strategies:
    1. Metadata grouping (condition_id, group_slug)
    2. Token-based similarity clustering
    3. LLM fallback for ambiguous cases
    """

    def __init__(
        self,
        config: Optional[FamilyDiscoveryConfig] = None,
        llm_classifier: Optional[LLMClassifier] = None
    ):
        self.config = config or FamilyDiscoveryConfig()
        self.llm = llm_classifier

        # Statistics
        self._families_discovered = 0
        self._llm_fallbacks = 0

    def discover_families(
        self,
        markets: list[PolymarketMarket]
    ) -> list[MarketFamily]:
        """
        Discover market families from a list of markets.

        Args:
            markets: List of markets to analyze

        Returns:
            List of MarketFamily objects
        """
        if len(markets) < self.config.min_family_size:
            return []

        families: list[MarketFamily] = []
        used_market_ids: set[str] = set()

        # Strategy 1: Metadata grouping (highest confidence)
        if self.config.use_metadata_grouping:
            metadata_groups = group_by_metadata(markets)

            for _group_key, group_markets in metadata_groups.items():
                if len(group_markets) >= self.config.min_family_size:
                    family = self._create_family(
                        markets=group_markets,
                        source="metadata",
                        confidence=0.95
                    )
                    families.append(family)
                    used_market_ids.update(m.id for m in group_markets)

        # Strategy 2: Token-based grouping
        remaining = [m for m in markets if m.id not in used_market_ids]
        if len(remaining) >= self.config.min_family_size:
            token_groups = group_by_tokens(remaining, self.config.min_token_overlap)

            for group in token_groups:
                if group.size >= self.config.min_family_size:
                    family = self._create_family(
                        markets=group.markets,
                        source="token_matching",
                        confidence=group.confidence
                    )
                    families.append(family)
                    used_market_ids.update(m.id for m in group.markets)

        self._families_discovered += len(families)
        logger.info(f"Discovered {len(families)} market families from {len(markets)} markets")

        return families

    async def discover_families_with_llm(
        self,
        markets: list[PolymarketMarket]
    ) -> list[MarketFamily]:
        """
        Discover families with LLM fallback for ambiguous cases.

        First uses standard discovery, then uses LLM for remaining
        ungrouped markets if LLM is configured.
        """
        # Run standard discovery first
        families = self.discover_families(markets)
        used_ids = {m.id for f in families for m in f.markets}

        # Check if LLM fallback is needed
        remaining = [m for m in markets if m.id not in used_ids]

        if (
            self.config.use_llm_fallback
            and self.llm
            and len(remaining) >= self.config.min_family_size
        ):
            self._llm_fallbacks += 1
            logger.info(f"Using LLM to group {len(remaining)} remaining markets")

            try:
                # Get LLM groupings
                group_indices = await self.llm.group_markets(remaining)

                for indices in group_indices:
                    if len(indices) >= self.config.min_family_size:
                        group_markets = [remaining[i] for i in indices]

                        # Classify relationship
                        result = await self.llm.classify_relationship(group_markets)

                        if result.confidence >= self.config.llm_confidence_threshold:
                            family = MarketFamily(
                                id=f"llm_{self._families_discovered}",
                                markets=group_markets,
                                relationship=result.relationship,
                                confidence=result.confidence,
                                shared_entity=result.shared_entity,
                                shared_event=result.shared_event,
                                source="llm"
                            )
                            families.append(family)
                            self._families_discovered += 1

            except Exception as e:
                logger.error(f"LLM grouping failed: {e}")

        return families

    def _create_family(
        self,
        markets: list[PolymarketMarket],
        source: str,
        confidence: float
    ) -> MarketFamily:
        """Create a MarketFamily from a list of markets."""
        # Determine shared entity/event
        entities = [m.entity for m in markets if m.entity]
        events = [m.event for m in markets if m.event]

        shared_entity = max(set(entities), key=entities.count) if entities else None
        shared_event = max(set(events), key=events.count) if events else None

        family_id = f"{source}_{self._families_discovered}"

        return MarketFamily(
            id=family_id,
            markets=markets,
            relationship=RelationshipType.UNKNOWN,  # Will be classified later
            confidence=confidence,
            shared_entity=shared_entity,
            shared_event=shared_event,
            source=source
        )

    def get_stats(self) -> dict:
        """Get discovery statistics."""
        return {
            "families_discovered": self._families_discovered,
            "llm_fallbacks": self._llm_fallbacks
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def enrich_market(market: PolymarketMarket) -> PolymarketMarket:
    """
    Enrich a market with extracted tokens, entity, event, and deadline.

    Modifies the market in place and returns it.
    """
    market.tokens = extract_tokens(market.question)
    market.entity = extract_entity(market.question)
    market.event = extract_event(market.question)
    market.deadline = extract_date(market.question)
    return market


def find_date_variant_families(
    markets: list[PolymarketMarket]
) -> list[MarketFamily]:
    """
    Find date-variant families (markets with different deadlines for same event).

    Date-variant families are special because they have monotonicity constraints.
    """
    # Enrich all markets
    for market in markets:
        enrich_market(market)

    # Group by entity+event (excluding date)
    groups: dict[str, list[PolymarketMarket]] = defaultdict(list)

    for market in markets:
        if market.deadline:
            key = f"{market.entity}:{market.event}"
            groups[key].append(market)

    # Create families from groups with multiple dates
    families: list[MarketFamily] = []
    for _key, group in groups.items():
        # Check if multiple distinct deadlines
        deadlines = {m.deadline for m in group}
        if len(deadlines) >= 2:
            family = MarketFamily(
                id=f"date_variant_{len(families)}",
                # Sort by deadline - type: ignore needed due to Optional[datetime]
                markets=sorted(group, key=lambda m: m.deadline),  # type: ignore[arg-type,return-value]
                relationship=RelationshipType.DATE_VARIANT,
                confidence=0.9,
                shared_entity=group[0].entity,
                shared_event=group[0].event,
                source="date_extraction"
            )
            families.append(family)

    return families
