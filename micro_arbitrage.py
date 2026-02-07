"""
Micro-Arbitrage Trading Module
===============================

High-frequency micro-arbitrage strategy with lower spread thresholds.
Designed to capture small price inefficiencies (0.1%+) across venues.

Features:
1. Lower spread thresholds (0.1% vs default 0.3%)
2. Enhanced fee optimization for micro-profits
3. Intelligent opportunity filtering
4. Volume-based position sizing
5. Latency-aware execution
"""

import asyncio
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger("PolyMangoBot.micro_arbitrage")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MicroArbMode(Enum):
    """Micro-arbitrage operating modes"""
    CONSERVATIVE = "conservative"   # 0.15% threshold, strict filters
    STANDARD = "standard"           # 0.10% threshold, normal filters
    AGGRESSIVE = "aggressive"       # 0.05% threshold, relaxed filters


class OpportunityQuality(Enum):
    """Quality rating for opportunities"""
    PREMIUM = "premium"             # Best opportunities
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MARGINAL = "marginal"           # Barely profitable


@dataclass
class MicroArbConfig:
    """Configuration for micro-arbitrage mode"""
    # Thresholds
    min_spread_pct: float = 0.10        # Minimum spread to consider (0.1%)
    min_profit_after_fees_pct: float = 0.02  # Minimum net profit (0.02%)

    # Fee assumptions
    maker_fee_pct: float = 0.05         # Assumed maker fee
    taker_fee_pct: float = 0.10         # Assumed taker fee
    max_slippage_pct: float = 0.05      # Maximum expected slippage

    # Position limits
    max_position_pct: float = 0.05      # Max 5% of capital per micro-arb
    min_position_usd: float = 50.0      # Minimum trade size
    max_position_usd: float = 5000.0    # Maximum trade size

    # Execution
    max_execution_time_ms: float = 2000 # Max time to execute both legs
    require_atomic: bool = True         # Require atomic execution

    # Filtering
    min_liquidity_usd: float = 1000.0   # Minimum depth at price level
    max_spread_age_ms: float = 500      # Maximum age of price data
    min_opportunity_score: float = 0.5  # Minimum quality score

    # Volume/Frequency
    max_trades_per_minute: int = 10     # Rate limit
    cooldown_after_trade_ms: float = 500  # Cooldown between trades

    @classmethod
    def conservative(cls) -> "MicroArbConfig":
        """Conservative configuration"""
        return cls(
            min_spread_pct=0.15,
            min_profit_after_fees_pct=0.05,
            max_position_pct=0.03,
            min_opportunity_score=0.7
        )

    @classmethod
    def standard(cls) -> "MicroArbConfig":
        """Standard configuration"""
        return cls()

    @classmethod
    def aggressive(cls) -> "MicroArbConfig":
        """Aggressive configuration"""
        return cls(
            min_spread_pct=0.05,
            min_profit_after_fees_pct=0.01,
            max_position_pct=0.08,
            min_opportunity_score=0.4,
            max_trades_per_minute=20,
            cooldown_after_trade_ms=200
        )


@dataclass
class MicroArbOpportunity:
    """Micro-arbitrage opportunity"""
    id: str
    market: str

    # Venue information
    buy_venue: str
    sell_venue: str

    # Prices
    buy_price: float
    sell_price: float
    mid_price: float

    # Spreads
    gross_spread_pct: float         # Spread before fees
    net_spread_pct: float           # Spread after fees
    estimated_profit_pct: float     # Including slippage estimate

    # Quality metrics
    quality: OpportunityQuality
    score: float                    # 0-1 quality score
    confidence: float               # Execution confidence

    # Liquidity
    buy_liquidity: float            # Available at buy price
    sell_liquidity: float           # Available at sell price
    executable_size: float          # Max size that can be executed

    # Suggested execution
    suggested_quantity: float
    suggested_position_usd: float
    estimated_profit_usd: float

    # Timing
    price_age_ms: float             # Age of price data
    expected_execution_ms: float

    # Risk metrics
    slippage_risk: float            # 0-1, higher = more risk
    execution_risk: float           # 0-1, higher = more risk

    timestamp: float = field(default_factory=time.time)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "market": self.market,
            "buy_venue": self.buy_venue,
            "sell_venue": self.sell_venue,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "gross_spread_pct": self.gross_spread_pct,
            "net_spread_pct": self.net_spread_pct,
            "estimated_profit_pct": self.estimated_profit_pct,
            "quality": self.quality.value,
            "score": self.score,
            "suggested_quantity": self.suggested_quantity,
            "estimated_profit_usd": self.estimated_profit_usd,
            "timestamp": self.timestamp
        }

    @property
    def is_profitable(self) -> bool:
        """Check if opportunity is profitable after fees"""
        return self.net_spread_pct > 0

    @property
    def risk_adjusted_score(self) -> float:
        """Score adjusted for risk"""
        risk_factor = 1 - (self.slippage_risk * 0.5 + self.execution_risk * 0.5)
        return self.score * risk_factor


@dataclass
class MicroArbResult:
    """Result of a micro-arbitrage trade"""
    opportunity: MicroArbOpportunity
    success: bool

    # Actual execution
    actual_buy_price: float = 0.0
    actual_sell_price: float = 0.0
    actual_quantity: float = 0.0
    actual_profit_usd: float = 0.0
    actual_profit_pct: float = 0.0

    # Fees
    total_fees_usd: float = 0.0

    # Timing
    execution_time_ms: float = 0.0

    # Slippage
    buy_slippage_pct: float = 0.0
    sell_slippage_pct: float = 0.0
    total_slippage_pct: float = 0.0

    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "actual_profit_usd": self.actual_profit_usd,
            "actual_profit_pct": self.actual_profit_pct,
            "total_fees_usd": self.total_fees_usd,
            "execution_time_ms": self.execution_time_ms,
            "total_slippage_pct": self.total_slippage_pct,
            "error": self.error
        }


# =============================================================================
# FEE CALCULATOR
# =============================================================================

class MicroArbFeeCalculator:
    """
    Precise fee calculation for micro-arbitrage.

    Considers:
    - Maker/taker fees per venue
    - Volume tier discounts
    - Rebates for liquidity provision
    - Network/withdrawal fees
    """

    def __init__(self):
        # Venue-specific fee schedules
        self._venue_fees: Dict[str, Dict] = {
            "polymarket": {
                "maker": 0.0,      # No maker fee on Polymarket
                "taker": 0.02,    # 2 cents per dollar (0.02%)
                "withdrawal": 0.0
            },
            "kraken": {
                "maker": 0.16,    # 0.16% maker
                "taker": 0.26,    # 0.26% taker
                "withdrawal": 0.0
            },
            "coinbase": {
                "maker": 0.40,    # 0.40% maker
                "taker": 0.60,    # 0.60% taker
                "withdrawal": 0.0
            }
        }

        # Volume tier adjustments (30-day volume -> fee multiplier)
        self._volume_tiers = [
            (0, 1.0),           # 0-10K: full fees
            (10000, 0.95),      # 10K-50K: 5% discount
            (50000, 0.90),      # 50K-100K: 10% discount
            (100000, 0.80),     # 100K-500K: 20% discount
            (500000, 0.70),     # 500K-1M: 30% discount
            (1000000, 0.60)     # 1M+: 40% discount
        ]

    def calculate_trade_fees(
        self,
        venue: str,
        is_maker: bool,
        trade_value_usd: float,
        thirty_day_volume: float = 0.0
    ) -> float:
        """Calculate fees for a single trade"""
        venue = venue.lower()

        if venue not in self._venue_fees:
            # Use default conservative estimate
            base_fee_pct = 0.25 if not is_maker else 0.15
        else:
            fees = self._venue_fees[venue]
            base_fee_pct = fees["maker"] if is_maker else fees["taker"]

        # Apply volume tier discount
        tier_multiplier = 1.0
        for volume_threshold, multiplier in self._volume_tiers:
            if thirty_day_volume >= volume_threshold:
                tier_multiplier = multiplier

        effective_fee_pct = base_fee_pct * tier_multiplier

        return trade_value_usd * (effective_fee_pct / 100)

    def calculate_round_trip_fees(
        self,
        buy_venue: str,
        sell_venue: str,
        trade_value_usd: float,
        buy_is_maker: bool = False,
        sell_is_maker: bool = False,
        thirty_day_volume: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Calculate round-trip fees for arbitrage.

        Returns:
            (buy_fees, sell_fees, total_fees)
        """
        buy_fees = self.calculate_trade_fees(
            buy_venue, buy_is_maker, trade_value_usd, thirty_day_volume
        )

        sell_fees = self.calculate_trade_fees(
            sell_venue, sell_is_maker, trade_value_usd, thirty_day_volume
        )

        return buy_fees, sell_fees, buy_fees + sell_fees

    def calculate_break_even_spread(
        self,
        buy_venue: str,
        sell_venue: str,
        buy_is_maker: bool = False,
        sell_is_maker: bool = False,
        thirty_day_volume: float = 0.0
    ) -> float:
        """Calculate minimum spread needed to break even"""
        # Use $1000 as reference
        _, _, total_fees = self.calculate_round_trip_fees(
            buy_venue, sell_venue, 1000.0,
            buy_is_maker, sell_is_maker, thirty_day_volume
        )

        return (total_fees / 1000.0) * 100  # As percentage

    def get_venue_fee_info(self, venue: str) -> Dict:
        """Get fee information for a venue"""
        venue = venue.lower()
        if venue in self._venue_fees:
            return self._venue_fees[venue].copy()
        return {"maker": 0.15, "taker": 0.25, "withdrawal": 0.0}


# =============================================================================
# OPPORTUNITY DETECTOR
# =============================================================================

class MicroArbOpportunityDetector:
    """
    Detects micro-arbitrage opportunities.

    Features:
    - Multi-venue price comparison
    - Fee-adjusted spread calculation
    - Liquidity-weighted scoring
    - Real-time opportunity ranking
    """

    def __init__(self, config: MicroArbConfig):
        self.config = config
        self.fee_calculator = MicroArbFeeCalculator()

        # Price cache
        self._prices: Dict[str, Dict[str, Dict]] = {}  # market -> venue -> price data

        # Opportunity history
        self._opportunity_history: deque = deque(maxlen=1000)

        # Statistics
        self._stats = {
            "opportunities_detected": 0,
            "opportunities_filtered": 0,
            "avg_spread_pct": 0.0
        }

    def update_price(
        self,
        market: str,
        venue: str,
        bid: float,
        ask: float,
        bid_size: float = 0.0,
        ask_size: float = 0.0,
        timestamp: Optional[float] = None
    ):
        """Update price data for a market/venue"""
        timestamp = timestamp or time.time()

        if market not in self._prices:
            self._prices[market] = {}

        self._prices[market][venue] = {
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "mid": (bid + ask) / 2,
            "spread": ask - bid,
            "timestamp": timestamp
        }

    def detect_opportunities(
        self,
        markets: Optional[List[str]] = None,
        capital: float = 10000.0
    ) -> List[MicroArbOpportunity]:
        """Detect all micro-arbitrage opportunities"""
        markets = markets or list(self._prices.keys())
        opportunities = []

        for market in markets:
            if market not in self._prices:
                continue

            venues = list(self._prices[market].keys())

            if len(venues) < 2:
                continue

            # Check all venue pairs
            for i, buy_venue in enumerate(venues):
                for sell_venue in venues[i + 1:]:
                    # Check both directions
                    opp = self._check_opportunity(
                        market, buy_venue, sell_venue, capital
                    )
                    if opp:
                        opportunities.append(opp)

                    opp = self._check_opportunity(
                        market, sell_venue, buy_venue, capital
                    )
                    if opp:
                        opportunities.append(opp)

        # Sort by risk-adjusted score
        opportunities.sort(key=lambda x: x.risk_adjusted_score, reverse=True)

        self._stats["opportunities_detected"] += len(opportunities)

        return opportunities

    def _check_opportunity(
        self,
        market: str,
        buy_venue: str,
        sell_venue: str,
        capital: float
    ) -> Optional[MicroArbOpportunity]:
        """Check for opportunity between two venues"""
        buy_data = self._prices[market].get(buy_venue)
        sell_data = self._prices[market].get(sell_venue)

        if not buy_data or not sell_data:
            return None

        # Check price age
        now = time.time()
        buy_age = (now - buy_data["timestamp"]) * 1000
        sell_age = (now - sell_data["timestamp"]) * 1000
        max_age = max(buy_age, sell_age)

        if max_age > self.config.max_spread_age_ms:
            return None

        # Calculate prices
        buy_price = buy_data["ask"]  # Buy at ask
        sell_price = sell_data["bid"]  # Sell at bid
        mid_price = (buy_data["mid"] + sell_data["mid"]) / 2

        # Calculate gross spread
        gross_spread = sell_price - buy_price
        gross_spread_pct = (gross_spread / buy_price) * 100

        # Skip if gross spread is negative or too small
        if gross_spread_pct < self.config.min_spread_pct:
            return None

        # Calculate fees
        trade_value = self.config.max_position_usd
        _, _, total_fees = self.fee_calculator.calculate_round_trip_fees(
            buy_venue, sell_venue, trade_value
        )
        total_fee_pct = (total_fees / trade_value) * 100

        # Calculate net spread (after fees)
        net_spread_pct = gross_spread_pct - total_fee_pct

        # Add slippage estimate
        slippage_pct = self.config.max_slippage_pct
        estimated_profit_pct = net_spread_pct - slippage_pct

        # Skip if not profitable after fees and slippage
        if estimated_profit_pct < self.config.min_profit_after_fees_pct:
            self._stats["opportunities_filtered"] += 1
            return None

        # Calculate liquidity
        buy_liquidity = buy_data["ask_size"] * buy_price
        sell_liquidity = sell_data["bid_size"] * sell_price
        min_liquidity = min(buy_liquidity, sell_liquidity)

        if min_liquidity < self.config.min_liquidity_usd:
            return None

        # Calculate executable size
        max_from_liquidity = min_liquidity * 0.5  # Use at most 50% of available
        max_from_capital = capital * self.config.max_position_pct
        executable_size = min(
            max_from_liquidity,
            max_from_capital,
            self.config.max_position_usd
        )

        # Calculate suggested quantity
        suggested_quantity = executable_size / buy_price

        # Calculate expected profit
        estimated_profit_usd = executable_size * (estimated_profit_pct / 100)

        # Calculate quality score
        score, quality = self._calculate_quality_score(
            gross_spread_pct,
            net_spread_pct,
            estimated_profit_pct,
            min_liquidity,
            max_age
        )

        if score < self.config.min_opportunity_score:
            return None

        # Calculate risk metrics
        slippage_risk = self._calculate_slippage_risk(
            executable_size, min_liquidity, gross_spread_pct
        )
        execution_risk = self._calculate_execution_risk(max_age, min_liquidity)

        # Generate unique ID
        opp_id = f"{market}_{buy_venue}_{sell_venue}_{int(now * 1000)}"

        return MicroArbOpportunity(
            id=opp_id,
            market=market,
            buy_venue=buy_venue,
            sell_venue=sell_venue,
            buy_price=buy_price,
            sell_price=sell_price,
            mid_price=mid_price,
            gross_spread_pct=gross_spread_pct,
            net_spread_pct=net_spread_pct,
            estimated_profit_pct=estimated_profit_pct,
            quality=quality,
            score=score,
            confidence=1 - execution_risk,
            buy_liquidity=buy_liquidity,
            sell_liquidity=sell_liquidity,
            executable_size=executable_size,
            suggested_quantity=suggested_quantity,
            suggested_position_usd=executable_size,
            estimated_profit_usd=estimated_profit_usd,
            price_age_ms=max_age,
            expected_execution_ms=self.config.max_execution_time_ms * 0.5,
            slippage_risk=slippage_risk,
            execution_risk=execution_risk,
            reasons=self._build_reasons(
                gross_spread_pct, net_spread_pct, min_liquidity, buy_venue, sell_venue
            )
        )

    def _calculate_quality_score(
        self,
        gross_spread: float,
        net_spread: float,
        profit_pct: float,
        liquidity: float,
        age_ms: float
    ) -> Tuple[float, OpportunityQuality]:
        """Calculate quality score and rating"""
        score = 0.0

        # Spread contribution (40%)
        if profit_pct > 0.5:
            score += 0.4
        elif profit_pct > 0.2:
            score += 0.3
        elif profit_pct > 0.1:
            score += 0.2
        elif profit_pct > 0.05:
            score += 0.1
        else:
            score += 0.05

        # Liquidity contribution (30%)
        if liquidity > 50000:
            score += 0.3
        elif liquidity > 10000:
            score += 0.25
        elif liquidity > 5000:
            score += 0.2
        elif liquidity > 1000:
            score += 0.15
        else:
            score += 0.1

        # Freshness contribution (20%)
        if age_ms < 100:
            score += 0.2
        elif age_ms < 250:
            score += 0.15
        elif age_ms < 500:
            score += 0.1
        else:
            score += 0.05

        # Fee efficiency (10%)
        fee_efficiency = net_spread / gross_spread if gross_spread > 0 else 0
        score += fee_efficiency * 0.1

        # Determine quality rating
        if score >= 0.8:
            quality = OpportunityQuality.PREMIUM
        elif score >= 0.65:
            quality = OpportunityQuality.HIGH
        elif score >= 0.5:
            quality = OpportunityQuality.MEDIUM
        elif score >= 0.35:
            quality = OpportunityQuality.LOW
        else:
            quality = OpportunityQuality.MARGINAL

        return score, quality

    def _calculate_slippage_risk(
        self,
        trade_size: float,
        liquidity: float,
        spread_pct: float
    ) -> float:
        """Calculate slippage risk (0-1)"""
        # Size relative to liquidity
        size_ratio = trade_size / liquidity if liquidity > 0 else 1.0

        # Base risk from size ratio
        risk = min(1.0, size_ratio * 2)

        # Adjust for spread (tighter spreads = more slippage risk)
        if spread_pct < 0.1:
            risk *= 1.5
        elif spread_pct < 0.2:
            risk *= 1.2

        return min(1.0, risk)

    def _calculate_execution_risk(
        self,
        age_ms: float,
        liquidity: float
    ) -> float:
        """Calculate execution risk (0-1)"""
        # Age risk
        age_risk = min(1.0, age_ms / 1000)

        # Liquidity risk
        if liquidity > 50000:
            liq_risk = 0.1
        elif liquidity > 10000:
            liq_risk = 0.2
        elif liquidity > 5000:
            liq_risk = 0.3
        else:
            liq_risk = 0.5

        return (age_risk * 0.6 + liq_risk * 0.4)

    def _build_reasons(
        self,
        gross_spread: float,
        net_spread: float,
        liquidity: float,
        buy_venue: str,
        sell_venue: str
    ) -> List[str]:
        """Build reasons list"""
        reasons = []

        reasons.append(f"Gross spread: {gross_spread:.3f}%")
        reasons.append(f"Net spread after fees: {net_spread:.3f}%")
        reasons.append(f"Available liquidity: ${liquidity:,.0f}")
        reasons.append(f"Buy on {buy_venue}, sell on {sell_venue}")

        return reasons

    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return self._stats.copy()


# =============================================================================
# MICRO-ARBITRAGE ENGINE
# =============================================================================

class MicroArbitrageEngine:
    """
    Main engine for micro-arbitrage trading.

    Integrates:
    - Opportunity detection
    - Position management
    - Trade execution coordination
    - Performance tracking
    """

    def __init__(
        self,
        mode: MicroArbMode = MicroArbMode.STANDARD,
        capital: float = 10000.0
    ):
        self.mode = mode
        self.capital = capital

        # Set configuration based on mode
        if mode == MicroArbMode.CONSERVATIVE:
            self.config = MicroArbConfig.conservative()
        elif mode == MicroArbMode.AGGRESSIVE:
            self.config = MicroArbConfig.aggressive()
        else:
            self.config = MicroArbConfig.standard()

        self.detector = MicroArbOpportunityDetector(self.config)
        self.fee_calculator = MicroArbFeeCalculator()

        # State
        self._active_opportunities: Dict[str, MicroArbOpportunity] = {}
        self._pending_trades: List[MicroArbOpportunity] = []

        # Rate limiting
        self._trades_this_minute: int = 0
        self._minute_start: float = time.time()
        self._last_trade_time: float = 0

        # Trade history
        self._trade_history: deque = deque(maxlen=1000)

        # Performance tracking
        self._performance = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_profit_usd": 0.0,
            "total_fees_usd": 0.0,
            "total_slippage_pct": 0.0,
            "avg_execution_ms": 0.0
        }

    def update_prices(
        self,
        market: str,
        venue: str,
        bid: float,
        ask: float,
        bid_size: float = 0.0,
        ask_size: float = 0.0
    ):
        """Update price data"""
        self.detector.update_price(
            market, venue, bid, ask, bid_size, ask_size
        )

    def scan_opportunities(
        self,
        markets: Optional[List[str]] = None
    ) -> List[MicroArbOpportunity]:
        """Scan for micro-arbitrage opportunities"""
        opportunities = self.detector.detect_opportunities(markets, self.capital)

        # Filter based on rate limits
        if not self._can_trade():
            return []

        # Filter out opportunities we're already tracking
        active_markets = {opp.market for opp in self._active_opportunities.values()}
        opportunities = [
            opp for opp in opportunities
            if opp.market not in active_markets
        ]

        return opportunities

    def _can_trade(self) -> bool:
        """Check if we can execute more trades"""
        now = time.time()

        # Reset minute counter if needed
        if now - self._minute_start >= 60:
            self._trades_this_minute = 0
            self._minute_start = now

        # Check rate limit
        if self._trades_this_minute >= self.config.max_trades_per_minute:
            return False

        # Check cooldown
        if (now - self._last_trade_time) * 1000 < self.config.cooldown_after_trade_ms:
            return False

        return True

    def queue_trade(self, opportunity: MicroArbOpportunity) -> bool:
        """Queue an opportunity for execution"""
        if not self._can_trade():
            return False

        if opportunity.id in self._active_opportunities:
            return False

        self._pending_trades.append(opportunity)
        self._active_opportunities[opportunity.id] = opportunity

        logger.info(
            f"Queued micro-arb opportunity: {opportunity.market} "
            f"({opportunity.buy_venue} -> {opportunity.sell_venue}) "
            f"spread={opportunity.net_spread_pct:.3f}%"
        )

        return True

    def get_next_trade(self) -> Optional[MicroArbOpportunity]:
        """Get next trade to execute"""
        if not self._pending_trades:
            return None

        if not self._can_trade():
            return None

        # Get highest priority trade
        self._pending_trades.sort(key=lambda x: x.risk_adjusted_score, reverse=True)

        return self._pending_trades.pop(0)

    def record_result(self, result: MicroArbResult):
        """Record trade result"""
        # Update rate limiting
        self._trades_this_minute += 1
        self._last_trade_time = time.time()

        # Update performance
        self._performance["total_trades"] += 1
        if result.success:
            self._performance["successful_trades"] += 1
            self._performance["total_profit_usd"] += result.actual_profit_usd
        self._performance["total_fees_usd"] += result.total_fees_usd
        self._performance["total_slippage_pct"] += result.total_slippage_pct

        # Update average execution time
        n = self._performance["total_trades"]
        self._performance["avg_execution_ms"] = (
            (self._performance["avg_execution_ms"] * (n - 1) + result.execution_time_ms) / n
        )

        # Remove from active
        if result.opportunity.id in self._active_opportunities:
            del self._active_opportunities[result.opportunity.id]

        # Add to history
        self._trade_history.append(result)

        logger.info(
            f"Micro-arb result: {result.opportunity.market} - "
            f"{'SUCCESS' if result.success else 'FAILED'} "
            f"profit=${result.actual_profit_usd:.2f} "
            f"slippage={result.total_slippage_pct:.3f}%"
        )

    def get_performance(self) -> Dict:
        """Get performance statistics"""
        total = self._performance["total_trades"]
        successful = self._performance["successful_trades"]

        return {
            "mode": self.mode.value,
            "total_trades": total,
            "successful_trades": successful,
            "success_rate": successful / total * 100 if total > 0 else 0,
            "total_profit_usd": self._performance["total_profit_usd"],
            "total_fees_usd": self._performance["total_fees_usd"],
            "net_profit_usd": self._performance["total_profit_usd"] - self._performance["total_fees_usd"],
            "avg_profit_per_trade": self._performance["total_profit_usd"] / total if total > 0 else 0,
            "avg_slippage_pct": self._performance["total_slippage_pct"] / total if total > 0 else 0,
            "avg_execution_ms": self._performance["avg_execution_ms"],
            "trades_this_minute": self._trades_this_minute,
            "active_opportunities": len(self._active_opportunities)
        }

    def set_mode(self, mode: MicroArbMode):
        """Change operating mode"""
        self.mode = mode

        if mode == MicroArbMode.CONSERVATIVE:
            self.config = MicroArbConfig.conservative()
        elif mode == MicroArbMode.AGGRESSIVE:
            self.config = MicroArbConfig.aggressive()
        else:
            self.config = MicroArbConfig.standard()

        self.detector.config = self.config

        logger.info(f"Micro-arbitrage mode changed to: {mode.value}")

    def get_break_even_spread(
        self,
        buy_venue: str,
        sell_venue: str
    ) -> float:
        """Get break-even spread for venue pair"""
        return self.fee_calculator.calculate_break_even_spread(
            buy_venue, sell_venue
        )


# =============================================================================
# TEST FUNCTION
# =============================================================================

async def test_micro_arbitrage():
    """Test micro-arbitrage module"""
    print("=" * 70)
    print("MICRO-ARBITRAGE MODULE TEST")
    print("=" * 70)

    # Test different modes
    for mode in [MicroArbMode.CONSERVATIVE, MicroArbMode.STANDARD, MicroArbMode.AGGRESSIVE]:
        print(f"\n--- Testing {mode.value.upper()} mode ---")

        engine = MicroArbitrageEngine(mode=mode, capital=10000.0)

        # Simulate price updates
        # Market 1: Small arbitrage opportunity
        engine.update_prices("BTC/USD", "polymarket", 42000, 42010, 5.0, 5.0)
        engine.update_prices("BTC/USD", "kraken", 42050, 42070, 10.0, 10.0)

        # Market 2: Larger opportunity
        engine.update_prices("ETH/USD", "polymarket", 2300, 2305, 20.0, 20.0)
        engine.update_prices("ETH/USD", "kraken", 2320, 2330, 30.0, 30.0)

        # Scan for opportunities
        opportunities = engine.scan_opportunities()

        print(f"  Config: min_spread={engine.config.min_spread_pct}%, "
              f"min_profit={engine.config.min_profit_after_fees_pct}%")
        print(f"  Break-even spread (poly->kraken): "
              f"{engine.get_break_even_spread('polymarket', 'kraken'):.3f}%")
        print(f"  Opportunities found: {len(opportunities)}")

        for opp in opportunities[:3]:
            print(f"\n  Opportunity: {opp.market}")
            print(f"    Direction: {opp.buy_venue} -> {opp.sell_venue}")
            print(f"    Buy @ ${opp.buy_price:.2f}, Sell @ ${opp.sell_price:.2f}")
            print(f"    Gross spread: {opp.gross_spread_pct:.3f}%")
            print(f"    Net spread: {opp.net_spread_pct:.3f}%")
            print(f"    Est. profit: {opp.estimated_profit_pct:.3f}%")
            print(f"    Quality: {opp.quality.value} (score: {opp.score:.2f})")
            print(f"    Suggested size: ${opp.suggested_position_usd:.2f}")
            print(f"    Expected profit: ${opp.estimated_profit_usd:.2f}")

    # Test fee calculator
    print("\n--- Fee Calculator Test ---")
    calc = MicroArbFeeCalculator()

    venues = ["polymarket", "kraken", "coinbase"]
    for v1 in venues:
        for v2 in venues:
            if v1 != v2:
                be_spread = calc.calculate_break_even_spread(v1, v2)
                print(f"  {v1} -> {v2}: break-even spread = {be_spread:.3f}%")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_micro_arbitrage())
