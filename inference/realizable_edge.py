"""
Realizable Edge Calculator
==========================

Computes the realizable edge after accounting for:
- Trading fees
- Slippage
- Execution risk
- Market impact

Transforms raw theoretical edge into expected profit.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from inference.models import (
    MultiLegTrade,
    RealizableEdge,
    TradeSide,
)

logger = logging.getLogger("PolyMangoBot.inference.realizable_edge")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EdgeConfig:
    """Configuration for edge calculation"""
    # Fee structure (Polymarket)
    maker_fee_pct: float = 0.0    # 0% maker
    taker_fee_pct: float = 0.02   # 0.02% taker

    # Slippage model
    base_slippage_pct: float = 0.05  # Base slippage
    liquidity_slippage_factor: float = 0.1  # Additional slippage per % of liquidity

    # Execution risk
    base_execution_risk: float = 0.02  # 2% base failure rate
    execution_risk_per_leg: float = 0.01  # 1% additional per leg

    # Minimum thresholds
    min_realizable_edge_pct: float = 0.3  # Minimum 0.3% after costs
    min_worst_case_profit: float = 0.0    # Must be non-negative


# =============================================================================
# EDGE CALCULATOR
# =============================================================================

class RealizableEdgeCalculator:
    """
    Calculates realizable edge after all costs.

    Takes into account:
    - Trading fees (maker/taker)
    - Expected slippage
    - Execution risk
    - Multi-leg complexity
    """

    def __init__(self, config: Optional[EdgeConfig] = None):
        self.config = config or EdgeConfig()

    def calculate(self, trade: MultiLegTrade) -> RealizableEdge:
        """
        Calculate realizable edge for a multi-leg trade.

        Args:
            trade: The MultiLegTrade to analyze

        Returns:
            RealizableEdge with all cost components
        """
        # Calculate fees
        total_fees = self._calculate_fees(trade)

        # Calculate slippage (already included in trade prices, this is estimate)
        total_slippage_pct = self._calculate_slippage(trade)

        # Calculate execution risk
        execution_risk = self._calculate_execution_risk(trade)

        # Calculate notional values
        total_cost = sum(
            leg.price * leg.size
            for leg in trade.legs
            if leg.side in [TradeSide.BUY_YES, TradeSide.BUY_NO]
        )

        total_revenue = sum(
            leg.price * leg.size
            for leg in trade.legs
            if leg.side in [TradeSide.SELL_YES, TradeSide.SELL_NO]
        )

        # Total notional for cost calculation (use either cost or revenue)
        total_notional = max(total_cost, total_revenue, 1.0)

        # Raw edge from trade
        raw_edge = trade.raw_edge

        # Slippage in USD (percentage of notional)
        slippage_usd = total_slippage_pct * total_notional / 100

        # Net edge after costs (as percentage)
        total_costs_usd = total_fees + slippage_usd
        cost_impact_pct = (total_costs_usd / total_notional) * 100
        net_edge = raw_edge - cost_impact_pct

        # Adjust for execution risk
        risk_adjusted_edge = net_edge * (1 - execution_risk)

        # Calculate worst/best P&L after costs
        worst_pnl = trade.worst_pnl - total_fees - slippage_usd
        best_pnl = trade.best_pnl - total_fees - slippage_usd

        return RealizableEdge(
            edge=risk_adjusted_edge,
            fees=total_fees,
            slippage=slippage_usd,
            execution_risk=execution_risk,
            worst_pnl=worst_pnl,
            best_pnl=best_pnl
        )

    def _calculate_fees(self, trade: MultiLegTrade) -> float:
        """Calculate total trading fees in USD."""
        total_fees = 0.0

        for leg in trade.legs:
            notional = leg.price * leg.size

            # Assume taker fees for immediate execution
            fee_rate = self.config.taker_fee_pct / 100
            total_fees += notional * fee_rate

        return total_fees

    def _calculate_slippage(self, trade: MultiLegTrade) -> float:
        """Calculate expected slippage as percentage."""
        total_slippage_pct = 0.0
        total_weight = 0.0

        for leg in trade.legs:
            notional = leg.price * leg.size

            # Base slippage
            leg_slippage = self.config.base_slippage_pct

            # Liquidity-based slippage
            if leg.market_liquidity > 0:
                pct_of_liquidity = (notional / leg.market_liquidity) * 100
                leg_slippage += pct_of_liquidity * self.config.liquidity_slippage_factor

            # Weight by notional
            total_slippage_pct += leg_slippage * notional
            total_weight += notional

        if total_weight > 0:
            return total_slippage_pct / total_weight
        return self.config.base_slippage_pct

    def _calculate_execution_risk(self, trade: MultiLegTrade) -> float:
        """Calculate probability of execution failure."""
        num_legs = len(trade.legs)

        # Base risk + per-leg risk
        risk = (
            self.config.base_execution_risk +
            (num_legs - 1) * self.config.execution_risk_per_leg
        )

        # Cap at 50%
        return min(0.5, risk)

    def is_profitable(self, edge: RealizableEdge) -> bool:
        """Check if edge meets profitability thresholds."""
        return (
            edge.edge >= self.config.min_realizable_edge_pct and
            edge.worst_pnl >= self.config.min_worst_case_profit
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_fees_simple(
    position_usd: float,
    num_legs: int,
    fee_pct: float = 0.02
) -> float:
    """Simple fee estimation without full trade construction."""
    return position_usd * num_legs * (fee_pct / 100)


def estimate_slippage_simple(
    position_usd: float,
    liquidity: float,
    base_pct: float = 0.05,
    liquidity_factor: float = 0.1
) -> float:
    """Simple slippage estimation."""
    if liquidity <= 0:
        return base_pct

    pct_of_liquidity = (position_usd / liquidity) * 100
    return base_pct + pct_of_liquidity * liquidity_factor


def calculate_break_even_edge(
    num_legs: int,
    fee_pct: float = 0.02,
    slippage_pct: float = 0.1,
    execution_risk: float = 0.05
) -> float:
    """
    Calculate minimum edge required to break even.

    Returns:
        Minimum edge percentage needed
    """
    # Total cost as percentage
    total_cost_pct = (
        num_legs * fee_pct +  # Fees
        slippage_pct +        # Slippage
        0                     # Execution risk is multiplicative
    )

    # Adjust for execution risk
    break_even = total_cost_pct / (1 - execution_risk)

    return break_even


def summarize_edge(edge: RealizableEdge, position_usd: float) -> dict:
    """
    Create a summary of edge analysis.

    Returns:
        Dictionary with edge summary
    """
    return {
        "realizable_edge_pct": round(edge.edge, 3),
        "fees_usd": round(edge.fees, 2),
        "slippage_usd": round(edge.slippage, 2),
        "total_costs_usd": round(edge.fees + edge.slippage, 2),
        "execution_risk_pct": round(edge.execution_risk * 100, 1),
        "worst_case_pnl_usd": round(edge.worst_pnl, 2),
        "best_case_pnl_usd": round(edge.best_pnl, 2),
        "expected_pnl_usd": round(
            edge.worst_pnl * 0.5 + edge.best_pnl * 0.5,  # Simple average
            2
        ),
        "position_usd": position_usd,
        "is_profitable": edge.is_profitable,
    }
