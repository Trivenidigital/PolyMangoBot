"""
Inference Engine Data Models
============================

Data classes for the Cross-Market Inference Engine.
Represents markets, families, violations, trades, and signals.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

# =============================================================================
# ENUMS
# =============================================================================

class RelationshipType(Enum):
    """Types of logical relationships between markets in a family"""
    DATE_VARIANT = "date_variant"              # Nested deadlines (e.g., "by March", "by April")
    MUTUALLY_EXCLUSIVE = "exclusive"           # At most one outcome can be YES
    EXHAUSTIVE = "exhaustive"                  # At least one outcome must be YES
    EXCLUSIVE_AND_EXHAUSTIVE = "exact"         # Exactly one outcome will be YES
    CONDITIONAL = "conditional"                # Implication chains
    UNKNOWN = "unknown"                        # Could not determine relationship


class ViolationType(Enum):
    """Types of constraint violations"""
    MONOTONICITY = "monotonicity"              # Later deadline cheaper than earlier
    MONOTONICITY_NO = "monotonicity_no"        # NO price violation
    DATE_VARIANT_NO_SWEEP = "date_variant_no_sweep"  # Buy all NOs profitable
    EXCLUSIVE_VIOLATION = "exclusive_violation"      # Sum of YES > 1.0
    EXHAUSTIVE_VIOLATION = "exhaustive_violation"    # Sum of YES < 1.0


class TradeSide(Enum):
    """Trade side for a leg"""
    BUY_YES = "BUY_YES"
    SELL_YES = "SELL_YES"
    BUY_NO = "BUY_NO"
    SELL_NO = "SELL_NO"


# =============================================================================
# MARKET DATA MODELS
# =============================================================================

@dataclass
class PolymarketMarket:
    """
    Represents a Polymarket prediction market with full metadata.

    Attributes:
        id: Unique market identifier
        question: The market question text
        slug: URL-friendly identifier
        condition_id: Polymarket condition ID (for grouping)
        group_slug: Optional group identifier
        yes_price: Current YES token price (0.0 - 1.0)
        no_price: Current NO token price (0.0 - 1.0)
        liquidity: Available liquidity in USD
        end_date: Market resolution date
        deadline: Extracted deadline for date-variant markets
    """
    id: str
    question: str
    slug: str
    condition_id: str
    group_slug: Optional[str] = None
    yes_price: float = 0.0
    no_price: float = 0.0
    liquidity: float = 0.0
    end_date: Optional[datetime] = None
    deadline: Optional[datetime] = None

    # Additional metadata
    volume_24h: float = 0.0
    outcome_prices: dict[str, float] = field(default_factory=dict)

    # Extracted tokens for matching
    tokens: list[str] = field(default_factory=list)
    entity: Optional[str] = None
    event: Optional[str] = None
    modifier: Optional[str] = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PolymarketMarket":
        """Create from Polymarket API response"""
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            condition_id=data.get("condition_id", ""),
            group_slug=data.get("group_slug"),
            yes_price=float(data.get("yes_price", 0.0) or 0.0),
            no_price=float(data.get("no_price", 0.0) or 0.0),
            liquidity=float(data.get("liquidity", 0.0) or 0.0),
            volume_24h=float(data.get("volume_24h", 0.0) or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "slug": self.slug,
            "condition_id": self.condition_id,
            "group_slug": self.group_slug,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "liquidity": self.liquidity,
            "deadline": self.deadline.isoformat() if self.deadline else None,
        }


@dataclass
class DateMarket(PolymarketMarket):
    """
    A market with an extracted deadline date.
    Used for date-variant arbitrage detection.
    """
    pass


# =============================================================================
# FAMILY MODELS
# =============================================================================

@dataclass
class MarketFamily:
    """
    A group of logically related markets.

    Markets in a family share the same underlying event and have
    prices that should satisfy certain mathematical constraints.
    """
    id: str
    markets: list[PolymarketMarket]
    relationship: RelationshipType = RelationshipType.UNKNOWN
    confidence: float = 0.0  # Confidence in the grouping (0.0 - 1.0)

    # Grouping metadata
    shared_entity: Optional[str] = None
    shared_event: Optional[str] = None
    source: str = "token_matching"  # "token_matching", "metadata", "llm"

    # Cache
    _sorted_by_deadline: Optional[list[PolymarketMarket]] = field(
        default=None, repr=False
    )

    @property
    def size(self) -> int:
        return len(self.markets)

    @property
    def total_liquidity(self) -> float:
        return sum(m.liquidity for m in self.markets)

    @property
    def min_liquidity(self) -> float:
        if not self.markets:
            return 0.0
        return min(m.liquidity for m in self.markets)

    def get_sorted_by_deadline(self) -> list[PolymarketMarket]:
        """Get markets sorted by deadline (ascending)"""
        if self._sorted_by_deadline is None:
            markets_with_deadline = [m for m in self.markets if m.deadline]
            # Sort by deadline - filter guarantees non-None
            self._sorted_by_deadline = sorted(
                markets_with_deadline,
                key=lambda m: m.deadline  # type: ignore[arg-type,return-value]
            )
        return self._sorted_by_deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "size": self.size,
            "relationship": self.relationship.value,
            "confidence": self.confidence,
            "shared_entity": self.shared_entity,
            "shared_event": self.shared_event,
            "total_liquidity": self.total_liquidity,
            "markets": [m.to_dict() for m in self.markets],
        }


@dataclass
class TokenGroup:
    """
    Intermediate grouping by extracted tokens.
    Used during family discovery before relationship classification.
    """
    key: str  # (entity, event) tuple as string
    markets: list[PolymarketMarket]
    confidence: float = 0.0

    @property
    def size(self) -> int:
        return len(self.markets)


# =============================================================================
# VIOLATION AND OPPORTUNITY MODELS
# =============================================================================

@dataclass
class Trade:
    """A single trade action"""
    side: TradeSide
    market_id: str
    price: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side.value,
            "market_id": self.market_id,
            "price": self.price,
        }


@dataclass
class Violation:
    """
    A constraint violation detected in a market family.
    Contains the raw violation data before trade construction.
    """
    type: ViolationType
    family_id: str
    raw_edge: float  # Raw profit before fees/slippage

    # For monotonicity violations
    buy: Optional[Trade] = None
    sell: Optional[Trade] = None

    # For sweep violations
    legs: list[Trade] = field(default_factory=list)

    # Context
    description: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "family_id": self.family_id,
            "raw_edge": self.raw_edge,
            "buy": self.buy.to_dict() if self.buy else None,
            "sell": self.sell.to_dict() if self.sell else None,
            "legs": [leg.to_dict() for leg in self.legs],
            "description": self.description,
        }


@dataclass
class TradeLeg:
    """
    A single leg of a multi-leg trade.
    Contains all information needed for execution.
    """
    market_id: str
    side: TradeSide
    price: float
    size: float  # Number of shares
    market_liquidity: float

    # Expected outcomes
    expected_payout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "side": self.side.value,
            "price": self.price,
            "size": self.size,
            "market_liquidity": self.market_liquidity,
            "expected_payout": self.expected_payout,
        }


@dataclass
class MultiLegTrade:
    """
    A complete multi-leg trade for structural arbitrage.
    """
    legs: list[TradeLeg]
    raw_edge: float
    worst_pnl: float  # Minimum profit across all scenarios
    best_pnl: float   # Maximum profit across all scenarios

    @property
    def num_legs(self) -> int:
        return len(self.legs)

    @property
    def min_liquidity(self) -> float:
        if not self.legs:
            return 0.0
        return min(leg.market_liquidity for leg in self.legs)

    @property
    def total_cost(self) -> float:
        return sum(
            leg.price * leg.size
            for leg in self.legs
            if leg.side in [TradeSide.BUY_YES, TradeSide.BUY_NO]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_legs": self.num_legs,
            "raw_edge": self.raw_edge,
            "worst_pnl": self.worst_pnl,
            "best_pnl": self.best_pnl,
            "min_liquidity": self.min_liquidity,
            "total_cost": self.total_cost,
            "legs": [leg.to_dict() for leg in self.legs],
        }


@dataclass
class ArbOpportunity:
    """
    A detected arbitrage opportunity with full details.
    """
    type: ViolationType
    family: MarketFamily

    # Trade construction
    legs: list[Trade]
    total_cost: float

    # Profit analysis
    raw_edge: float
    guaranteed_payout: float = 0.0  # Minimum payout
    max_payout: float = 0.0         # Maximum payout

    # Worst-case analysis
    worst_case_profit: float = 0.0
    best_case_profit: float = 0.0
    worst_scenario: str = ""

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "family_id": self.family.id,
            "num_legs": len(self.legs),
            "total_cost": self.total_cost,
            "raw_edge": self.raw_edge,
            "worst_case_profit": self.worst_case_profit,
            "best_case_profit": self.best_case_profit,
            "worst_scenario": self.worst_scenario,
        }


# =============================================================================
# REALIZABLE EDGE MODELS
# =============================================================================

@dataclass
class RealizableEdge:
    """
    Computed realizable edge after fees and slippage.
    """
    edge: float              # Net edge after all costs
    fees: float              # Total estimated fees
    slippage: float          # Total estimated slippage
    execution_risk: float    # Probability of execution failure
    worst_pnl: float         # Worst-case P&L after costs
    best_pnl: float          # Best-case P&L after costs

    @property
    def is_profitable(self) -> bool:
        return self.worst_pnl > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge": self.edge,
            "fees": self.fees,
            "slippage": self.slippage,
            "execution_risk": self.execution_risk,
            "worst_pnl": self.worst_pnl,
            "best_pnl": self.best_pnl,
            "is_profitable": self.is_profitable,
        }


# =============================================================================
# SIGNAL MODELS
# =============================================================================

@dataclass
class ArbSignal:
    """
    Final arbitrage signal ready for execution.
    Output of the inference engine.
    """
    type: str = "structural_arb"
    subtype: str = ""  # ViolationType value
    family_id: str = ""

    # Trade details
    legs: list[TradeLeg] = field(default_factory=list)

    # Edge analysis
    raw_edge: float = 0.0
    realizable_edge: float = 0.0
    worst_case_pnl: float = 0.0
    best_case_pnl: float = 0.0

    # Confidence and risk
    confidence: int = 9  # 1-10, structural arbs are high confidence
    min_liquidity: float = 0.0
    execution_risk: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "subtype": self.subtype,
            "family_id": self.family_id,
            "num_legs": len(self.legs),
            "raw_edge": self.raw_edge,
            "realizable_edge": self.realizable_edge,
            "worst_case_pnl": self.worst_case_pnl,
            "best_case_pnl": self.best_case_pnl,
            "confidence": self.confidence,
            "min_liquidity": self.min_liquidity,
            "execution_risk": self.execution_risk,
        }


# =============================================================================
# MONITOR STATE MODELS
# =============================================================================

@dataclass
class ArbState:
    """
    State tracking for an active arbitrage opportunity.
    Used by ArbMonitor for persistence.
    """
    family_id: str
    violation_type: ViolationType
    first_seen: datetime
    last_seen: datetime
    price_snapshots: list[dict[str, Any]] = field(default_factory=list)
    alert_sent: bool = False

    @property
    def duration_seconds(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def key(self) -> str:
        return f"{self.family_id}:{self.violation_type.value}"


@dataclass
class FamilyPriceSnapshot:
    """
    Price snapshot for a market family at a point in time.
    """
    family_id: str
    timestamp: datetime
    prices: dict[str, dict[str, float]]  # market_id -> {"yes": price, "no": price}

    def get_yes_sum(self) -> float:
        return sum(p.get("yes", 0.0) for p in self.prices.values())

    def get_no_sum(self) -> float:
        return sum(p.get("no", 0.0) for p in self.prices.values())
