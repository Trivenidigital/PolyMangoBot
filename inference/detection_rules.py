"""
Detection Rules Module
======================

Detects constraint violations in market families that represent
arbitrage opportunities:

1. Monotonicity Violations (Date-Variant):
   - Later deadline should have higher YES price
   - Later deadline should have lower NO price

2. NO Sweep Profitability (Date-Variant):
   - Buying NO on all markets may guarantee profit

3. Exclusive Violations:
   - Sum of YES prices > 1.0 (sell all YES = guaranteed profit)

4. Exhaustive Violations:
   - Sum of YES prices < 1.0 (buy all YES = guaranteed profit)
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Optional

from inference.models import (
    MarketFamily,
    RelationshipType,
    Trade,
    TradeSide,
    Violation,
    ViolationType,
)

logger = logging.getLogger("PolyMangoBot.inference.detection_rules")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DetectionConfig:
    """Configuration for violation detection"""
    # Minimum edge to report a violation
    min_edge_pct: float = 0.5  # 0.5% minimum raw edge

    # Monotonicity thresholds
    min_monotonicity_gap: float = 0.01  # 1 cent minimum price gap

    # Sweep profitability thresholds
    min_sweep_edge_pct: float = 1.0  # 1% minimum for sweep trades

    # Exclusive/exhaustive thresholds
    sum_violation_threshold: float = 0.02  # 2% deviation from 1.0

    # Liquidity requirements
    min_leg_liquidity: float = 500.0  # Minimum $500 liquidity per leg


# =============================================================================
# MONOTONICITY DETECTION
# =============================================================================

def detect_monotonicity_violations(
    family: MarketFamily,
    config: DetectionConfig
) -> list[Violation]:
    """
    Detect monotonicity violations in a date-variant family.

    Monotonicity: For nested deadlines, later deadline should have
    higher YES price (more time = more likely to happen).

    If market A deadline < market B deadline, then:
    - YES_A should <= YES_B (violation if YES_A > YES_B)
    - NO_A should >= NO_B (violation if NO_A < NO_B)

    A violation means we can:
    - Buy YES on later deadline (cheaper), sell YES on earlier (expensive)
    - Or buy NO on earlier (cheaper), sell NO on later (expensive)
    """
    if family.relationship != RelationshipType.DATE_VARIANT:
        return []

    violations = []
    sorted_markets = family.get_sorted_by_deadline()

    if len(sorted_markets) < 2:
        return []

    # Check all pairs
    for i in range(len(sorted_markets)):
        for j in range(i + 1, len(sorted_markets)):
            early = sorted_markets[i]  # Earlier deadline
            late = sorted_markets[j]   # Later deadline

            # Check liquidity
            if early.liquidity < config.min_leg_liquidity:
                continue
            if late.liquidity < config.min_leg_liquidity:
                continue

            # YES monotonicity: early.yes should be <= late.yes
            # Violation if early.yes > late.yes
            yes_gap = early.yes_price - late.yes_price
            if yes_gap > config.min_monotonicity_gap:
                edge_pct = yes_gap * 100

                if edge_pct >= config.min_edge_pct:
                    violations.append(Violation(
                        type=ViolationType.MONOTONICITY,
                        family_id=family.id,
                        raw_edge=edge_pct,
                        buy=Trade(
                            side=TradeSide.BUY_YES,
                            market_id=late.id,
                            price=late.yes_price
                        ),
                        sell=Trade(
                            side=TradeSide.SELL_YES,
                            market_id=early.id,
                            price=early.yes_price
                        ),
                        description=(
                            f"YES monotonicity: {early.question[:50]}... @ {early.yes_price:.3f} > "
                            f"{late.question[:50]}... @ {late.yes_price:.3f}"
                        )
                    ))

            # NO monotonicity: early.no should be >= late.no
            # Violation if early.no < late.no
            no_gap = late.no_price - early.no_price
            if no_gap > config.min_monotonicity_gap:
                edge_pct = no_gap * 100

                if edge_pct >= config.min_edge_pct:
                    violations.append(Violation(
                        type=ViolationType.MONOTONICITY_NO,
                        family_id=family.id,
                        raw_edge=edge_pct,
                        buy=Trade(
                            side=TradeSide.BUY_NO,
                            market_id=early.id,
                            price=early.no_price
                        ),
                        sell=Trade(
                            side=TradeSide.SELL_NO,
                            market_id=late.id,
                            price=late.no_price
                        ),
                        description=(
                            f"NO monotonicity: {early.question[:50]}... @ {early.no_price:.3f} < "
                            f"{late.question[:50]}... @ {late.no_price:.3f}"
                        )
                    ))

    return violations


# =============================================================================
# NO SWEEP DETECTION
# =============================================================================

def detect_no_sweep_opportunity(
    family: MarketFamily,
    config: DetectionConfig
) -> Optional[Violation]:
    """
    Detect NO sweep opportunity in a date-variant family.

    NO Sweep: Buy NO on all markets in the family.

    For date-variant markets:
    - If the event happens by the earliest deadline, all NOs pay 0
    - If the event happens between deadlines, some NOs pay 1, some pay 0
    - If the event never happens, all NOs pay 1

    Guaranteed profit if: sum(NO prices) < 1.0

    The payout is always >= 1.0 (at least one NO will pay out),
    so profit = 1.0 - sum(NO prices).
    """
    if family.relationship != RelationshipType.DATE_VARIANT:
        return None

    sorted_markets = family.get_sorted_by_deadline()

    if len(sorted_markets) < 2:
        return None

    # Check liquidity on all legs
    for m in sorted_markets:
        if m.liquidity < config.min_leg_liquidity:
            return None

    # Calculate total NO cost
    total_no_cost = sum(m.no_price for m in sorted_markets)

    # Guaranteed payout is 1.0 (at minimum, the longest deadline NO pays out
    # if the event never happens)
    guaranteed_payout = 1.0

    # Edge calculation
    raw_edge = guaranteed_payout - total_no_cost
    edge_pct = raw_edge * 100

    if edge_pct < config.min_sweep_edge_pct:
        return None

    # Build legs
    legs = [
        Trade(
            side=TradeSide.BUY_NO,
            market_id=m.id,
            price=m.no_price
        )
        for m in sorted_markets
    ]

    return Violation(
        type=ViolationType.DATE_VARIANT_NO_SWEEP,
        family_id=family.id,
        raw_edge=edge_pct,
        legs=legs,
        description=(
            f"NO sweep: Buy NO on {len(sorted_markets)} markets for "
            f"total {total_no_cost:.3f}, guaranteed payout 1.0"
        )
    )


# =============================================================================
# EXCLUSIVE/EXHAUSTIVE DETECTION
# =============================================================================

def detect_exclusive_violation(
    family: MarketFamily,
    config: DetectionConfig
) -> Optional[Violation]:
    """
    Detect exclusive constraint violation.

    For mutually exclusive markets: sum(YES prices) should be <= 1.0

    If sum > 1.0, we can sell YES on all markets:
    - Cost: sum(YES prices) - N (selling N shares at YES prices)
    - Max loss: 1.0 (if one YES wins, we pay 1.0)
    - Profit: sum(YES prices) - 1.0

    Actually for selling, we receive the YES price, so:
    - Revenue: sum(YES prices)
    - Max payout: 1.0
    - Profit: sum(YES prices) - 1.0
    """
    if family.relationship not in [
        RelationshipType.MUTUALLY_EXCLUSIVE,
        RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE
    ]:
        return None

    markets = family.markets

    # Check liquidity
    for m in markets:
        if m.liquidity < config.min_leg_liquidity:
            return None

    yes_sum = sum(m.yes_price for m in markets)
    threshold = 1.0 + config.sum_violation_threshold

    if yes_sum <= threshold:
        return None

    raw_edge = yes_sum - 1.0
    edge_pct = raw_edge * 100

    if edge_pct < config.min_edge_pct:
        return None

    legs = [
        Trade(
            side=TradeSide.SELL_YES,
            market_id=m.id,
            price=m.yes_price
        )
        for m in markets
    ]

    return Violation(
        type=ViolationType.EXCLUSIVE_VIOLATION,
        family_id=family.id,
        raw_edge=edge_pct,
        legs=legs,
        description=(
            f"Exclusive violation: Sum of YES = {yes_sum:.3f} > 1.0, "
            f"sell all YES for {edge_pct:.2f}% profit"
        )
    )


def detect_exhaustive_violation(
    family: MarketFamily,
    config: DetectionConfig
) -> Optional[Violation]:
    """
    Detect exhaustive constraint violation.

    For exhaustive markets: sum(YES prices) should be >= 1.0

    If sum < 1.0, we can buy YES on all markets:
    - Cost: sum(YES prices)
    - Guaranteed payout: 1.0 (at least one YES wins)
    - Profit: 1.0 - sum(YES prices)
    """
    if family.relationship not in [
        RelationshipType.EXHAUSTIVE,
        RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE
    ]:
        return None

    markets = family.markets

    # Check liquidity
    for m in markets:
        if m.liquidity < config.min_leg_liquidity:
            return None

    yes_sum = sum(m.yes_price for m in markets)
    threshold = 1.0 - config.sum_violation_threshold

    if yes_sum >= threshold:
        return None

    raw_edge = 1.0 - yes_sum
    edge_pct = raw_edge * 100

    if edge_pct < config.min_edge_pct:
        return None

    legs = [
        Trade(
            side=TradeSide.BUY_YES,
            market_id=m.id,
            price=m.yes_price
        )
        for m in markets
    ]

    return Violation(
        type=ViolationType.EXHAUSTIVE_VIOLATION,
        family_id=family.id,
        raw_edge=edge_pct,
        legs=legs,
        description=(
            f"Exhaustive violation: Sum of YES = {yes_sum:.3f} < 1.0, "
            f"buy all YES for {edge_pct:.2f}% profit"
        )
    )


# =============================================================================
# UNIFIED DETECTION
# =============================================================================

class ViolationDetector:
    """
    Detects all types of constraint violations in market families.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._violations_found = 0

    def detect_all(self, family: MarketFamily) -> list[Violation]:
        """
        Detect all violations in a market family.

        Args:
            family: MarketFamily to analyze

        Returns:
            List of detected Violations
        """
        violations = []

        # Date-variant checks
        if family.relationship == RelationshipType.DATE_VARIANT:
            # Monotonicity violations
            mono_violations = detect_monotonicity_violations(family, self.config)
            violations.extend(mono_violations)

            # NO sweep opportunity
            sweep = detect_no_sweep_opportunity(family, self.config)
            if sweep:
                violations.append(sweep)

        # Exclusive check
        if family.relationship in [
            RelationshipType.MUTUALLY_EXCLUSIVE,
            RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE
        ]:
            exclusive = detect_exclusive_violation(family, self.config)
            if exclusive:
                violations.append(exclusive)

        # Exhaustive check
        if family.relationship in [
            RelationshipType.EXHAUSTIVE,
            RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE
        ]:
            exhaustive = detect_exhaustive_violation(family, self.config)
            if exhaustive:
                violations.append(exhaustive)

        self._violations_found += len(violations)

        if violations:
            logger.info(
                f"Found {len(violations)} violations in family {family.id} "
                f"(type: {family.relationship.value})"
            )

        return violations

    def detect_all_families(
        self,
        families: list[MarketFamily]
    ) -> Generator[Violation, None, None]:
        """
        Detect violations across all families.

        Args:
            families: List of MarketFamilies to analyze

        Yields:
            Violations as they are found
        """
        for family in families:
            yield from self.detect_all(family)

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "violations_found": self._violations_found
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_worst_case_pnl(violation: Violation) -> tuple[float, str]:
    """
    Calculate worst-case P&L for a violation trade.

    Returns:
        Tuple of (worst_pnl, scenario_description)
    """
    if violation.type == ViolationType.MONOTONICITY:
        # Buy later YES, sell earlier YES
        # Worst case: event happens after earlier deadline but before later
        # Later YES pays 0, earlier YES pays 1 (we owe)
        if violation.buy and violation.sell:
            cost = violation.buy.price
            revenue = violation.sell.price
            worst_pnl = revenue - cost - 1.0  # We owe 1.0 on sell
            return worst_pnl, "Event happens between deadlines"

    elif violation.type == ViolationType.MONOTONICITY_NO:
        # Buy earlier NO, sell later NO
        # Worst case: event never happens
        # Both NOs pay 1, we receive 1 on buy and owe 1 on sell = break even
        # Actually worst is event happens after earlier but before later
        if violation.buy and violation.sell:
            cost = violation.buy.price
            revenue = violation.sell.price
            worst_pnl = revenue - cost  # Both settle same
            return worst_pnl, "Event happens between deadlines"

    elif violation.type == ViolationType.DATE_VARIANT_NO_SWEEP:
        # Buy all NOs
        # Best case: event never happens, all NOs pay 1
        # Worst case: event happens by first deadline, only first NO pays 0
        total_cost = sum(leg.price for leg in violation.legs)
        # Minimum payout is 1 (last NO always pays if event doesn't happen)
        worst_pnl = 1.0 - total_cost
        return worst_pnl, "Event never happens (guaranteed)"

    elif violation.type == ViolationType.EXCLUSIVE_VIOLATION:
        # Sell all YES
        # Worst case: one YES wins, we pay 1.0
        revenue = sum(leg.price for leg in violation.legs)
        worst_pnl = revenue - 1.0
        return worst_pnl, "One outcome wins (expected)"

    elif violation.type == ViolationType.EXHAUSTIVE_VIOLATION:
        # Buy all YES
        # Worst case: only one YES wins (but that's still profit if sum < 1)
        cost = sum(leg.price for leg in violation.legs)
        worst_pnl = 1.0 - cost
        return worst_pnl, "One outcome wins (expected)"

    return 0.0, "Unknown"


def calculate_best_case_pnl(violation: Violation) -> tuple[float, str]:
    """
    Calculate best-case P&L for a violation trade.

    Returns:
        Tuple of (best_pnl, scenario_description)
    """
    if violation.type == ViolationType.MONOTONICITY:
        # Best case: event happens before earlier deadline
        # Both YES pay 1
        if violation.buy and violation.sell:
            cost = violation.buy.price
            revenue = violation.sell.price
            best_pnl = revenue - cost + 1.0 - 1.0  # Both pay out
            return best_pnl, "Event happens before both deadlines"

    elif violation.type == ViolationType.DATE_VARIANT_NO_SWEEP:
        # Best case: event happens after first deadline but before last
        # Multiple NOs pay out
        total_cost = sum(leg.price for leg in violation.legs)
        # Maximum payout is N (all NOs pay if event happens between each)
        # Actually max is still 1 per leg, but timing matters
        best_pnl = len(violation.legs) - total_cost
        return best_pnl, "Event timing maximizes NO payouts"

    elif violation.type == ViolationType.EXCLUSIVE_VIOLATION:
        # Best case: same as worst (one wins)
        revenue = sum(leg.price for leg in violation.legs)
        best_pnl = revenue - 1.0
        return best_pnl, "One outcome wins"

    elif violation.type == ViolationType.EXHAUSTIVE_VIOLATION:
        # Best case: multiple YES win (shouldn't happen if truly exhaustive)
        # But conservatively assume one wins
        cost = sum(leg.price for leg in violation.legs)
        best_pnl = 1.0 - cost
        return best_pnl, "One outcome wins"

    return 0.0, "Unknown"
