"""
Trade Constructor Module
========================

Constructs executable multi-leg trades from detected violations.

Converts Violations into MultiLegTrades with:
- Position sizing based on liquidity
- Slippage estimation
- Execution order optimization
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from inference.models import (
    MarketFamily,
    MultiLegTrade,
    TradeLeg,
    TradeSide,
    Violation,
    ViolationType,
)

logger = logging.getLogger("PolyMangoBot.inference.trade_constructor")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradeConstructorConfig:
    """Configuration for trade construction"""
    # Position sizing
    default_position_usd: float = 100.0
    max_position_pct_of_liquidity: float = 0.05  # Max 5% of market liquidity
    min_position_usd: float = 10.0

    # Slippage estimation
    base_slippage_pct: float = 0.1  # 0.1% base slippage
    slippage_per_pct_of_liquidity: float = 0.5  # Additional slippage per % of liquidity

    # Fee estimation
    maker_fee_pct: float = 0.0  # Polymarket has 0% maker fee
    taker_fee_pct: float = 0.02  # 0.02% taker fee


# =============================================================================
# TRADE CONSTRUCTOR
# =============================================================================

class TradeConstructor:
    """
    Constructs executable trades from violations.

    Handles position sizing, slippage estimation, and execution ordering.
    """

    def __init__(self, config: Optional[TradeConstructorConfig] = None):
        self.config = config or TradeConstructorConfig()

    def construct_trade(
        self,
        violation: Violation,
        family: MarketFamily,
        position_usd: Optional[float] = None
    ) -> Optional[MultiLegTrade]:
        """
        Construct a MultiLegTrade from a Violation.

        Args:
            violation: The detected violation
            family: The market family containing the violation
            position_usd: Desired position size in USD (optional)

        Returns:
            MultiLegTrade if constructable, None otherwise
        """
        # Get market lookup
        market_lookup = {m.id: m for m in family.markets}

        # Calculate position size
        position = self._calculate_position_size(
            violation, market_lookup, position_usd
        )

        if position < self.config.min_position_usd:
            logger.debug(f"Position size {position} below minimum, skipping")
            return None

        # Construct legs based on violation type
        if violation.type in [
            ViolationType.MONOTONICITY,
            ViolationType.MONOTONICITY_NO
        ]:
            return self._construct_pair_trade(
                violation, market_lookup, position
            )
        else:
            return self._construct_sweep_trade(
                violation, market_lookup, position
            )

    def _calculate_position_size(
        self,
        violation: Violation,
        market_lookup: dict[str, Any],
        requested_position: Optional[float]
    ) -> float:
        """Calculate appropriate position size based on liquidity."""
        # Get minimum liquidity across all legs
        min_liquidity = float('inf')

        if violation.buy:
            market = market_lookup.get(violation.buy.market_id)
            if market:
                min_liquidity = min(min_liquidity, market.liquidity)

        if violation.sell:
            market = market_lookup.get(violation.sell.market_id)
            if market:
                min_liquidity = min(min_liquidity, market.liquidity)

        for leg in violation.legs:
            market = market_lookup.get(leg.market_id)
            if market:
                min_liquidity = min(min_liquidity, market.liquidity)

        if min_liquidity == float('inf'):
            min_liquidity = 1000.0  # Default

        # Calculate max position based on liquidity
        max_position = min_liquidity * self.config.max_position_pct_of_liquidity

        # Use requested or default, capped by max
        position = requested_position or self.config.default_position_usd
        position = min(position, max_position)

        return position

    def _construct_pair_trade(
        self,
        violation: Violation,
        market_lookup: dict[str, Any],
        position_usd: float
    ) -> Optional[MultiLegTrade]:
        """Construct a two-leg pair trade (monotonicity violations)."""
        if not violation.buy or not violation.sell:
            return None

        buy_market = market_lookup.get(violation.buy.market_id)
        sell_market = market_lookup.get(violation.sell.market_id)

        if not buy_market or not sell_market:
            return None

        # Calculate share quantities
        buy_price = violation.buy.price
        sell_price = violation.sell.price

        # For pair trades, we want equal exposure
        # Buy side: position_usd / buy_price = shares
        buy_shares = position_usd / buy_price if buy_price > 0 else 0
        sell_shares = buy_shares  # Match share counts

        # Estimate slippage
        buy_slippage = self._estimate_slippage(position_usd, buy_market.liquidity)
        sell_slippage = self._estimate_slippage(
            sell_shares * sell_price, sell_market.liquidity
        )

        # Build legs
        buy_leg = TradeLeg(
            market_id=violation.buy.market_id,
            side=violation.buy.side,
            price=buy_price * (1 + buy_slippage),
            size=buy_shares,
            market_liquidity=buy_market.liquidity,
            expected_payout=buy_shares
        )

        sell_leg = TradeLeg(
            market_id=violation.sell.market_id,
            side=violation.sell.side,
            price=sell_price * (1 - sell_slippage),
            size=sell_shares,
            market_liquidity=sell_market.liquidity,
            expected_payout=sell_shares * sell_price * (1 - sell_slippage)
        )

        # Calculate P&L scenarios
        total_cost = buy_leg.price * buy_leg.size
        total_revenue = sell_leg.price * sell_leg.size

        # For monotonicity trades:
        # Worst case: buy side doesn't pay, sell side does pay (we owe)
        worst_pnl = total_revenue - total_cost - sell_leg.size

        # Best case: both sides pay
        best_pnl = total_revenue - total_cost

        return MultiLegTrade(
            legs=[buy_leg, sell_leg],
            raw_edge=violation.raw_edge,
            worst_pnl=worst_pnl,
            best_pnl=best_pnl
        )

    def _construct_sweep_trade(
        self,
        violation: Violation,
        market_lookup: dict[str, Any],
        position_usd: float
    ) -> Optional[MultiLegTrade]:
        """Construct a multi-leg sweep trade."""
        if not violation.legs:
            return None

        # Determine sizing strategy based on violation type
        # For exclusive/exhaustive violations, we need equal share counts
        # For NO sweeps, we can use equal dollar amounts
        equal_shares = violation.type in [
            ViolationType.EXCLUSIVE_VIOLATION,
            ViolationType.EXHAUSTIVE_VIOLATION
        ]

        trade_legs = []
        total_cost = 0.0  # Cash outflow for buys
        total_revenue = 0.0  # Cash inflow from sells

        if equal_shares:
            # Calculate shares based on total position and sum of prices
            prices = [t.price for t in violation.legs]
            price_sum = sum(prices)
            # shares = position_usd / price_sum gives us N shares where
            # cost to buy N of each = N * price_sum = position_usd
            shares_per_leg = position_usd / price_sum if price_sum > 0 else 0

            for trade in violation.legs:
                market = market_lookup.get(trade.market_id)
                if not market:
                    continue

                price = trade.price
                shares = shares_per_leg
                notional = shares * price

                slippage = self._estimate_slippage(notional, market.liquidity)

                if trade.side in [TradeSide.BUY_YES, TradeSide.BUY_NO]:
                    adj_price = price * (1 + slippage)
                    total_cost += adj_price * shares
                else:
                    adj_price = price * (1 - slippage)
                    total_revenue += adj_price * shares

                leg = TradeLeg(
                    market_id=trade.market_id,
                    side=trade.side,
                    price=adj_price,
                    size=shares,
                    market_liquidity=market.liquidity,
                    expected_payout=shares
                )
                trade_legs.append(leg)
        else:
            # Equal dollar amounts per leg (for NO sweeps)
            num_legs = len(violation.legs)
            position_per_leg = position_usd / num_legs

            for trade in violation.legs:
                market = market_lookup.get(trade.market_id)
                if not market:
                    continue

                price = trade.price
                shares = position_per_leg / price if price > 0 else 0

                slippage = self._estimate_slippage(position_per_leg, market.liquidity)

                if trade.side in [TradeSide.BUY_YES, TradeSide.BUY_NO]:
                    adj_price = price * (1 + slippage)
                    total_cost += adj_price * shares
                else:
                    adj_price = price * (1 - slippage)
                    total_revenue += adj_price * shares

                leg = TradeLeg(
                    market_id=trade.market_id,
                    side=trade.side,
                    price=adj_price,
                    size=shares,
                    market_liquidity=market.liquidity,
                    expected_payout=shares
                )
                trade_legs.append(leg)

        if not trade_legs:
            return None

        # Calculate P&L based on violation type
        if violation.type == ViolationType.DATE_VARIANT_NO_SWEEP:
            # Buying NO on all: guaranteed payout >= 1
            guaranteed_payout = min(leg.size for leg in trade_legs)
            worst_pnl = guaranteed_payout - total_cost
            best_pnl = sum(leg.size for leg in trade_legs) - total_cost

        elif violation.type == ViolationType.EXCLUSIVE_VIOLATION:
            # Selling equal shares of YES on all outcomes
            # Revenue = shares * sum(prices) = shares * (sum > 1.0)
            # Payout when one wins = shares * 1.0
            # Profit = shares * (sum - 1.0)
            shares = trade_legs[0].size if trade_legs else 0
            worst_pnl = total_revenue - shares  # One always wins
            best_pnl = total_revenue - shares

        elif violation.type == ViolationType.EXHAUSTIVE_VIOLATION:
            # Buying equal shares of YES on all outcomes
            # Cost = shares * sum(prices) = shares * (sum < 1.0)
            # Payout = shares * 1.0 (one will win)
            # Profit = shares * (1.0 - sum)
            shares = trade_legs[0].size if trade_legs else 0
            worst_pnl = shares - total_cost  # One always wins
            best_pnl = shares - total_cost

        else:
            worst_pnl = total_revenue - total_cost
            best_pnl = total_revenue - total_cost

        return MultiLegTrade(
            legs=trade_legs,
            raw_edge=violation.raw_edge,
            worst_pnl=worst_pnl,
            best_pnl=best_pnl
        )

    def _estimate_slippage(
        self,
        position_usd: float,
        liquidity: float
    ) -> float:
        """Estimate slippage based on position size and liquidity."""
        if liquidity <= 0:
            return self.config.base_slippage_pct

        pct_of_liquidity = (position_usd / liquidity) * 100

        slippage = (
            self.config.base_slippage_pct +
            pct_of_liquidity * self.config.slippage_per_pct_of_liquidity
        )

        return slippage / 100  # Return as decimal


# =============================================================================
# EXECUTION ORDER OPTIMIZER
# =============================================================================

def optimize_execution_order(trade: MultiLegTrade) -> list[TradeLeg]:
    """
    Optimize the execution order of trade legs.

    Prioritizes:
    1. Buy legs before sell legs (reduce risk of naked shorts)
    2. Higher liquidity legs first (more likely to fill)
    3. Larger positions first (get bulk of trade done)
    """
    legs = trade.legs.copy()

    def sort_key(leg: TradeLeg) -> tuple:
        # Buy legs first (0), then sell (1)
        is_sell = 1 if leg.side in [TradeSide.SELL_YES, TradeSide.SELL_NO] else 0

        # Higher liquidity first (negative for descending)
        liquidity_score = -leg.market_liquidity

        # Larger positions first (negative for descending)
        position_score = -leg.size * leg.price

        return (is_sell, liquidity_score, position_score)

    return sorted(legs, key=sort_key)
