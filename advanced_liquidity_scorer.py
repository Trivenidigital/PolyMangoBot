"""
Advanced Liquidity-Weighted Scoring Module
Sophisticated opportunity scoring based on market microstructure

Enhanced with full microstructure analysis:
- Order Book Imbalance (OBI) - predictive of short-term price moves
- VWAP distance - realistic fill price estimation
- Book Pressure Gradient - identifies support/resistance
- Toxicity Score (VPIN) - detects informed traders
- Spread Dynamics - timing signals
"""

import math
import statistics
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger("PolyMangoBot.liquidity")


# =============================================================================
# MICROSTRUCTURE ANALYSIS FUNCTIONS
# =============================================================================

def calculate_obi(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], levels: int = 5) -> float:
    """
    Calculate Order Book Imbalance (OBI).
    OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    Strong predictor of next-tick price direction.
    Positive OBI = bid-heavy = bullish signal.

    Args:
        bids: List of (price, quantity) tuples, sorted descending by price
        asks: List of (price, quantity) tuples, sorted ascending by price
        levels: Number of levels to consider

    Returns:
        OBI value between -1 and 1
    """
    bid_vol = sum(b[1] for b in bids[:levels])
    ask_vol = sum(a[1] for a in asks[:levels])
    total = bid_vol + ask_vol

    if total == 0:
        return 0.0

    return (bid_vol - ask_vol) / total


def calculate_depth_weighted_obi(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], levels: int = 10) -> float:
    """
    Calculate depth-weighted OBI.
    Levels closer to mid-price are weighted more heavily.

    Args:
        bids: List of (price, quantity) tuples
        asks: List of (price, quantity) tuples
        levels: Maximum levels to consider

    Returns:
        Depth-weighted OBI between -1 and 1
    """
    # Exponential decay weights
    weights = [0.5 ** i for i in range(levels)]

    weighted_bid = sum(
        bids[i][1] * weights[i]
        for i in range(min(len(bids), levels))
    )
    weighted_ask = sum(
        asks[i][1] * weights[i]
        for i in range(min(len(asks), levels))
    )

    total = weighted_bid + weighted_ask
    if total == 0:
        return 0.0

    return (weighted_bid - weighted_ask) / total


def calculate_vwap(order_book: List[Tuple[float, float]], trade_size: float, mid_price: float, side: str = "buy") -> Tuple[float, float]:
    """
    Calculate Volume-Weighted Average Price and slippage for executing trade_size.

    How far from mid-price will our order actually fill?
    Critical for realistic profit estimation.

    Args:
        order_book: List of (price, quantity) tuples
        trade_size: Size of the intended trade
        mid_price: Current mid price
        side: "buy" or "sell"

    Returns:
        (vwap_price, slippage_bps)
    """
    if not order_book or trade_size <= 0:
        return mid_price, 0.0

    remaining = trade_size
    total_cost = 0.0
    filled = 0.0

    for price, qty in order_book:
        fill_qty = min(remaining, qty)
        total_cost += price * fill_qty
        filled += fill_qty
        remaining -= fill_qty

        if remaining <= 0:
            break

    if filled == 0:
        return mid_price, 0.0

    vwap = total_cost / filled

    # Calculate slippage in basis points
    if side == "buy":
        slippage_bps = ((vwap - mid_price) / mid_price) * 10000
    else:
        slippage_bps = ((mid_price - vwap) / mid_price) * 10000

    return vwap, max(0, slippage_bps)


def calculate_book_pressure_gradient(order_book: List[Tuple[float, float]], levels: int = 5) -> float:
    """
    Calculate rate of change in volume across price levels.
    Steep positive gradient = strong support/resistance.

    Args:
        order_book: List of (price, quantity) tuples
        levels: Number of levels to analyze

    Returns:
        Gradient (positive = increasing depth away from spread)
    """
    if len(order_book) < 2:
        return 0.0

    quantities = [qty for _, qty in order_book[:levels]]

    if len(quantities) < 2:
        return 0.0

    gradients = []
    for i in range(1, len(quantities)):
        gradients.append(quantities[i] - quantities[i-1])

    return np.mean(gradients) if gradients else 0.0


def calculate_toxicity(trades: List[Dict], window_minutes: float = 5) -> float:
    """
    Calculate Volume-synchronized Probability of Informed Trading (VPIN) approximation.
    High toxicity = informed traders present = higher adverse selection risk.

    Args:
        trades: List of trade dicts with 'qty', 'side', 'timestamp' keys
        window_minutes: Time window in minutes

    Returns:
        Toxicity score between 0 and 1
    """
    if not trades:
        return 0.0

    # Filter to recent trades
    import time
    cutoff = time.time() - (window_minutes * 60)
    recent_trades = [t for t in trades if t.get('timestamp', 0) > cutoff]

    if len(recent_trades) < 5:
        return 0.0

    # Classify trades as buy/sell initiated
    buy_volume = sum(t.get('qty', 0) for t in recent_trades if t.get('side') == 'buy')
    sell_volume = sum(t.get('qty', 0) for t in recent_trades if t.get('side') == 'sell')
    total_volume = buy_volume + sell_volume

    if total_volume == 0:
        return 0.0

    # Order imbalance as toxicity proxy
    return abs(buy_volume - sell_volume) / total_volume


def analyze_spread_dynamics(spread_history: List[float]) -> Dict:
    """
    Analyze spread behavior for timing signals.

    Args:
        spread_history: List of historical spread values

    Returns:
        Dict with spread analysis metrics
    """
    if len(spread_history) < 10:
        return {
            "current": spread_history[-1] if spread_history else 0,
            "mean": 0,
            "std": 0,
            "percentile": 0.5,
            "trend": 0
        }

    current = spread_history[-1]
    mean = np.mean(spread_history)
    std = np.std(spread_history)

    # Calculate percentile
    sorted_history = sorted(spread_history)
    percentile = np.searchsorted(sorted_history, current) / len(sorted_history)

    # Calculate trend (last 20 values)
    recent = spread_history[-20:] if len(spread_history) >= 20 else spread_history
    if len(recent) >= 2:
        coeffs = np.polyfit(range(len(recent)), recent, 1)
        trend = coeffs[0]
    else:
        trend = 0

    return {
        "current": current,
        "mean": mean,
        "std": std,
        "percentile": percentile,
        "trend": trend,
        "is_tight": percentile < 0.3,  # Below 30th percentile
        "is_wide": percentile > 0.7,   # Above 70th percentile
        "is_widening": trend > 0,
        "is_tightening": trend < 0
    }


@dataclass
class MicrostructureSignals:
    """Aggregated microstructure signals for trading decisions"""
    # OBI signals
    obi: float = 0.0                     # -1 to 1
    obi_5_level: float = 0.0             # Top 5 levels only
    obi_depth_weighted: float = 0.0      # Weighted by level

    # VWAP metrics
    vwap_buy: float = 0.0
    vwap_sell: float = 0.0
    vwap_slippage_buy_bps: float = 0.0
    vwap_slippage_sell_bps: float = 0.0

    # Pressure gradients
    bid_pressure_gradient: float = 0.0
    ask_pressure_gradient: float = 0.0
    pressure_imbalance: float = 0.0      # bid - ask gradient

    # Toxicity
    toxicity_score: float = 0.0

    # Spread dynamics
    spread_percentile: float = 0.5
    spread_trend: float = 0.0
    spread_is_tight: bool = False
    spread_is_wide: bool = False

    # Derived signals
    price_direction_signal: float = 0.0  # -1 to 1, positive = bullish
    execution_risk_score: float = 0.0    # 0 to 1, lower is better
    timing_score: float = 0.5            # 0 to 1, higher = better timing

    def to_dict(self) -> Dict:
        return {
            "obi": self.obi,
            "obi_5_level": self.obi_5_level,
            "obi_depth_weighted": self.obi_depth_weighted,
            "vwap_slippage_buy_bps": self.vwap_slippage_buy_bps,
            "vwap_slippage_sell_bps": self.vwap_slippage_sell_bps,
            "bid_pressure_gradient": self.bid_pressure_gradient,
            "ask_pressure_gradient": self.ask_pressure_gradient,
            "pressure_imbalance": self.pressure_imbalance,
            "toxicity_score": self.toxicity_score,
            "spread_percentile": self.spread_percentile,
            "spread_trend": self.spread_trend,
            "price_direction_signal": self.price_direction_signal,
            "execution_risk_score": self.execution_risk_score,
            "timing_score": self.timing_score
        }


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics for a venue"""
    venue: str
    symbol: str
    timestamp: float

    # Depth metrics
    bid_depth_1: float = 0.0  # Quantity at best bid
    bid_depth_5: float = 0.0  # Cumulative quantity at top 5 levels
    bid_depth_10: float = 0.0
    ask_depth_1: float = 0.0
    ask_depth_5: float = 0.0
    ask_depth_10: float = 0.0

    # Price impact metrics
    bid_impact_1pct: float = 0.0  # Quantity needed to move price 1%
    ask_impact_1pct: float = 0.0

    # Spread metrics
    spread_bps: float = 0.0  # Spread in basis points
    spread_volatility: float = 0.0  # Spread standard deviation

    # Flow metrics
    imbalance: float = 0.0  # (bid_depth - ask_depth) / total
    turnover_rate: float = 0.0  # How fast liquidity replenishes

    # === NEW MICROSTRUCTURE METRICS ===

    # Order Book Imbalance (OBI)
    obi: float = 0.0                     # -1 to 1, positive = bid-heavy
    obi_5_level: float = 0.0             # OBI for top 5 levels only
    obi_depth_weighted: float = 0.0      # Depth-weighted OBI

    # VWAP metrics (for default trade size)
    vwap_slippage_buy_bps: float = 0.0   # Expected slippage to buy
    vwap_slippage_sell_bps: float = 0.0  # Expected slippage to sell

    # Pressure gradients
    bid_pressure_gradient: float = 0.0   # Rate of depth change on bid side
    ask_pressure_gradient: float = 0.0   # Rate of depth change on ask side
    pressure_imbalance: float = 0.0      # bid_gradient - ask_gradient

    # Toxicity (updated from trade history)
    toxicity_score: float = 0.0          # 0-1, higher = more informed flow

    # Spread dynamics
    spread_percentile: float = 0.5       # Current spread vs historical
    spread_trend: float = 0.0            # Positive = widening

    # Quality score (0-100)
    quality_score: float = 0.0


@dataclass
class ScoredOpportunity:
    """Arbitrage opportunity with comprehensive scoring"""
    market: str
    buy_venue: str
    buy_price: float
    buy_quantity_available: float
    sell_venue: str
    sell_price: float
    sell_quantity_available: float

    # Basic metrics
    spread: float = 0.0
    spread_percent: float = 0.0

    # Liquidity scores
    buy_liquidity_score: float = 0.0
    sell_liquidity_score: float = 0.0
    combined_liquidity_score: float = 0.0

    # Execution estimates
    estimated_slippage_pct: float = 0.0
    estimated_fill_time_ms: float = 0.0
    max_executable_quantity: float = 0.0

    # Risk-adjusted scores
    risk_adjusted_spread: float = 0.0  # Spread after slippage estimate
    sharpe_ratio: float = 0.0  # Risk-adjusted return estimate
    execution_probability: float = 0.0  # Probability of successful fill

    # === NEW MICROSTRUCTURE METRICS ===

    # OBI signals (buy and sell venues)
    buy_obi: float = 0.0                 # OBI at buy venue
    sell_obi: float = 0.0                # OBI at sell venue
    obi_divergence: float = 0.0          # Divergence between venues

    # VWAP-based realistic pricing
    realistic_buy_price: float = 0.0     # VWAP for target quantity
    realistic_sell_price: float = 0.0    # VWAP for target quantity
    realistic_spread_pct: float = 0.0    # After VWAP adjustment

    # Pressure analysis
    buy_pressure_support: float = 0.0    # Strength of bid support
    sell_pressure_resistance: float = 0.0 # Strength of ask resistance

    # Toxicity assessment
    buy_venue_toxicity: float = 0.0      # Toxicity at buy venue
    sell_venue_toxicity: float = 0.0     # Toxicity at sell venue
    combined_toxicity: float = 0.0       # Max of both

    # Timing signals
    spread_timing_score: float = 0.5     # Based on spread dynamics
    entry_confidence: float = 0.5        # Overall entry confidence

    # Final composite score
    composite_score: float = 0.0

    # For compatibility with main_v4.py
    opportunity_id: str = ""
    metrics: Optional['LiquidityMetrics'] = None
    recommended_size: float = 0.0
    expected_profit: float = 0.0
    estimated_cost: float = 0.0
    overall_score: float = 0.0

    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self):
        """Set derived fields after initialization"""
        if not self.opportunity_id:
            self.opportunity_id = f"opp_{self.market}_{self.buy_venue}_{self.sell_venue}"
        if self.overall_score == 0.0:
            self.overall_score = self.composite_score

    def to_dict(self) -> Dict:
        return {
            "market": self.market,
            "buy_venue": self.buy_venue,
            "buy_price": self.buy_price,
            "sell_venue": self.sell_venue,
            "sell_price": self.sell_price,
            "spread_percent": self.spread_percent,
            "composite_score": self.composite_score,
            "estimated_slippage_pct": self.estimated_slippage_pct,
            "max_executable_quantity": self.max_executable_quantity,
            "execution_probability": self.execution_probability,
            # Microstructure metrics
            "buy_obi": self.buy_obi,
            "sell_obi": self.sell_obi,
            "realistic_spread_pct": self.realistic_spread_pct,
            "combined_toxicity": self.combined_toxicity,
            "entry_confidence": self.entry_confidence,
        }


class AdvancedLiquidityScorer:
    """
    Advanced liquidity scoring engine with market microstructure analysis.

    Features:
    - Multi-level order book analysis
    - Order Book Imbalance (OBI) calculation
    - VWAP-based realistic fill estimation
    - Book pressure gradient analysis
    - Toxicity scoring (VPIN approximation)
    - Spread dynamics timing signals
    - Price impact estimation
    - Execution probability modeling
    - Risk-adjusted opportunity scoring
    - Historical liquidity pattern learning
    """

    def __init__(self, history_size: int = 100, spread_history_size: int = 500):
        self.history_size = history_size
        self.spread_history_size = spread_history_size

        # Historical data for pattern learning
        self.liquidity_history: Dict[str, deque] = {}  # venue:symbol -> metrics
        self.spread_history: Dict[str, deque] = {}     # venue:symbol -> spread values
        self.obi_history: Dict[str, deque] = {}        # venue:symbol -> OBI values
        self.execution_history: deque = deque(maxlen=1000)  # Historical executions

        # Trade history for toxicity calculation
        self.trade_history: Dict[str, deque] = {}      # venue:symbol -> trades

        # Learned parameters
        self.venue_reliability: Dict[str, float] = {}  # Venue execution reliability
        self.symbol_volatility: Dict[str, float] = {}  # Symbol-specific volatility

        # Default trade size for VWAP calculations
        self.default_trade_size: float = 1.0

    def analyze_order_book(
        self,
        venue: str,
        symbol: str,
        bids: List[Tuple[float, float]],  # [(price, qty), ...]
        asks: List[Tuple[float, float]],
        mid_price: float,
        trade_size: Optional[float] = None
    ) -> LiquidityMetrics:
        """
        Analyze order book and compute comprehensive liquidity metrics
        including full microstructure analysis.
        """
        metrics = LiquidityMetrics(
            venue=venue,
            symbol=symbol,
            timestamp=datetime.now().timestamp()
        )

        if not bids or not asks:
            return metrics

        if trade_size is None:
            trade_size = self.default_trade_size

        key = f"{venue}:{symbol}"

        # Depth metrics
        metrics.bid_depth_1 = bids[0][1] if bids else 0
        metrics.bid_depth_5 = sum(q for _, q in bids[:5])
        metrics.bid_depth_10 = sum(q for _, q in bids[:10])

        metrics.ask_depth_1 = asks[0][1] if asks else 0
        metrics.ask_depth_5 = sum(q for _, q in asks[:5])
        metrics.ask_depth_10 = sum(q for _, q in asks[:10])

        # Spread metrics
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid
        metrics.spread_bps = (spread / mid_price) * 10000  # Basis points

        # Price impact estimation (quantity needed to move price 1%)
        target_price_bid = best_bid * 0.99  # 1% below best bid
        target_price_ask = best_ask * 1.01  # 1% above best ask

        cumulative_qty = 0
        for price, qty in bids:
            if price >= target_price_bid:
                cumulative_qty += qty
            else:
                break
        metrics.bid_impact_1pct = cumulative_qty

        cumulative_qty = 0
        for price, qty in asks:
            if price <= target_price_ask:
                cumulative_qty += qty
            else:
                break
        metrics.ask_impact_1pct = cumulative_qty

        # Basic imbalance (legacy)
        total_depth = metrics.bid_depth_5 + metrics.ask_depth_5
        if total_depth > 0:
            metrics.imbalance = (metrics.bid_depth_5 - metrics.ask_depth_5) / total_depth

        # === MICROSTRUCTURE ANALYSIS ===

        # 1. Order Book Imbalance (OBI)
        metrics.obi = calculate_obi(bids, asks, levels=10)
        metrics.obi_5_level = calculate_obi(bids, asks, levels=5)
        metrics.obi_depth_weighted = calculate_depth_weighted_obi(bids, asks, levels=10)

        # Store OBI history
        if key not in self.obi_history:
            self.obi_history[key] = deque(maxlen=self.spread_history_size)
        self.obi_history[key].append(metrics.obi)

        # 2. VWAP calculations
        _, metrics.vwap_slippage_buy_bps = calculate_vwap(asks, trade_size, mid_price, "buy")
        _, metrics.vwap_slippage_sell_bps = calculate_vwap(bids, trade_size, mid_price, "sell")

        # 3. Pressure gradients
        metrics.bid_pressure_gradient = calculate_book_pressure_gradient(bids, levels=5)
        metrics.ask_pressure_gradient = calculate_book_pressure_gradient(asks, levels=5)
        metrics.pressure_imbalance = metrics.bid_pressure_gradient - metrics.ask_pressure_gradient

        # 4. Toxicity (from trade history if available)
        trades = list(self.trade_history.get(key, []))
        if trades:
            metrics.toxicity_score = calculate_toxicity(trades, window_minutes=5)

        # 5. Spread dynamics
        if key not in self.spread_history:
            self.spread_history[key] = deque(maxlen=self.spread_history_size)
        self.spread_history[key].append(metrics.spread_bps)

        spread_history_list = list(self.spread_history[key])
        if len(spread_history_list) >= 10:
            spread_dynamics = analyze_spread_dynamics(spread_history_list)
            metrics.spread_percentile = spread_dynamics["percentile"]
            metrics.spread_trend = spread_dynamics["trend"]
            metrics.spread_volatility = spread_dynamics["std"]

        # Compute quality score (enhanced)
        metrics.quality_score = self._compute_quality_score(metrics)

        # Store in history
        self._store_metrics(venue, symbol, metrics)

        return metrics

    def add_trade(self, venue: str, symbol: str, trade: Dict):
        """
        Add observed trade to history for toxicity calculation.

        Args:
            venue: Venue name
            symbol: Symbol name
            trade: Dict with 'qty', 'side', 'timestamp' keys
        """
        key = f"{venue}:{symbol}"
        if key not in self.trade_history:
            self.trade_history[key] = deque(maxlen=1000)
        self.trade_history[key].append(trade)

    def _compute_quality_score(self, metrics: LiquidityMetrics) -> float:
        """
        Compute composite liquidity quality score (0-100).

        Higher is better:
        - Tight spreads
        - Deep order books
        - Balanced order flow
        - Low price impact
        - Low toxicity
        - Favorable spread dynamics
        - Strong support/resistance gradients
        """
        score = 50.0  # Base score

        # Spread component (up to +/-20 points)
        # Tight spread (<10 bps) is excellent, wide spread (>50 bps) is poor
        if metrics.spread_bps < 10:
            score += 20
        elif metrics.spread_bps < 20:
            score += 15
        elif metrics.spread_bps < 50:
            score += 5
        elif metrics.spread_bps > 100:
            score -= 15

        # Depth component (up to +/-15 points)
        depth_ratio = (metrics.bid_depth_5 + metrics.ask_depth_5) / 2
        if depth_ratio > 100:
            score += 15
        elif depth_ratio > 50:
            score += 10
        elif depth_ratio > 10:
            score += 5
        elif depth_ratio < 1:
            score -= 10

        # Imbalance component (up to +/-10 points)
        # Balanced book is better
        imbalance_abs = abs(metrics.imbalance)
        if imbalance_abs < 0.1:
            score += 10
        elif imbalance_abs < 0.3:
            score += 5
        elif imbalance_abs > 0.6:
            score -= 10

        # Price impact component (up to +/-5 points)
        min_impact = min(metrics.bid_impact_1pct, metrics.ask_impact_1pct)
        if min_impact > 50:
            score += 5
        elif min_impact < 5:
            score -= 5

        # === MICROSTRUCTURE SCORING ADJUSTMENTS ===

        # OBI component (up to +/-5 points)
        # Extreme imbalances are warning signs
        obi_abs = abs(metrics.obi)
        if obi_abs < 0.2:
            score += 3  # Balanced book
        elif obi_abs > 0.6:
            score -= 5  # Extreme imbalance - risky

        # Toxicity component (up to +/-10 points)
        # High toxicity = informed traders = higher risk
        if metrics.toxicity_score < 0.2:
            score += 5
        elif metrics.toxicity_score < 0.4:
            score += 2
        elif metrics.toxicity_score > 0.6:
            score -= 10
        elif metrics.toxicity_score > 0.5:
            score -= 5

        # VWAP slippage component (up to +/-5 points)
        total_slippage = metrics.vwap_slippage_buy_bps + metrics.vwap_slippage_sell_bps
        if total_slippage < 5:
            score += 5
        elif total_slippage < 10:
            score += 2
        elif total_slippage > 30:
            score -= 5

        # Spread dynamics component (up to +/-5 points)
        if metrics.spread_percentile < 0.3:
            score += 5  # Spread is tight vs history
        elif metrics.spread_percentile > 0.8:
            score -= 5  # Spread is wide vs history

        # Pressure gradient component (up to +/-3 points)
        # Strong bid support (positive gradient) is good
        if metrics.bid_pressure_gradient > 5:
            score += 2  # Strong support
        if metrics.ask_pressure_gradient > 5:
            score += 1  # Strong resistance (limits upside but stabilizes)

        return max(0, min(100, score))

    def _store_metrics(self, venue: str, symbol: str, metrics: LiquidityMetrics):
        """Store metrics in history for pattern learning"""
        key = f"{venue}:{symbol}"

        if key not in self.liquidity_history:
            self.liquidity_history[key] = deque(maxlen=self.history_size)

        self.liquidity_history[key].append(metrics)

    def estimate_slippage(
        self,
        venue: str,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        order_book: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Estimate slippage for a given order size.

        Returns:
            (slippage_percent, fill_price)
        """
        if not order_book:
            return 10.0, 0.0  # High slippage for empty book

        best_price = order_book[0][0]
        filled = 0.0
        total_cost = 0.0

        for price, qty in order_book:
            fill_qty = min(qty, quantity - filled)
            total_cost += fill_qty * price
            filled += fill_qty

            if filled >= quantity:
                break

        if filled == 0:
            return 10.0, 0.0

        avg_fill_price = total_cost / filled

        if side == "buy":
            slippage_pct = ((avg_fill_price - best_price) / best_price) * 100
        else:
            slippage_pct = ((best_price - avg_fill_price) / best_price) * 100

        return max(0, slippage_pct), avg_fill_price

    def estimate_fill_time(
        self,
        venue: str,
        symbol: str,
        quantity: float,
        liquidity_at_best: float
    ) -> float:
        """
        Estimate time to fill order in milliseconds.

        Based on:
        - Available liquidity
        - Historical fill rates
        - Venue characteristics
        """
        if liquidity_at_best <= 0:
            return 5000.0  # 5 seconds for empty book

        # Base fill rate (quantity per 100ms)
        base_fill_rate = {
            "polymarket": 0.5,
            "kraken": 2.0,
            "coinbase": 1.5,
        }.get(venue, 1.0)

        # Estimate based on quantity vs available
        if quantity <= liquidity_at_best:
            # Can fill immediately at best level
            fill_time = 100.0  # 100ms base
        else:
            # Need multiple levels
            levels_needed = quantity / max(liquidity_at_best, 0.01)
            fill_time = 100.0 + (levels_needed * 50.0)  # 50ms per additional level

        # Adjust by venue reliability
        reliability = self.venue_reliability.get(venue, 0.8)
        fill_time /= reliability

        return min(fill_time, 10000.0)  # Cap at 10 seconds

    def estimate_execution_probability(
        self,
        buy_metrics: LiquidityMetrics,
        sell_metrics: LiquidityMetrics,
        quantity: float,
        spread_percent: float
    ) -> float:
        """
        Estimate probability of successful execution.

        Factors:
        - Liquidity availability
        - Spread stability
        - Historical execution success
        - Market conditions
        - Microstructure signals (OBI, toxicity, spread dynamics)
        """
        # Base probability
        prob = 0.8

        # Liquidity availability factor
        min_liquidity = min(
            buy_metrics.ask_depth_1,
            sell_metrics.bid_depth_1
        )

        if quantity > min_liquidity * 0.5:
            prob *= 0.7  # Reduce if order is large relative to liquidity
        elif quantity < min_liquidity * 0.1:
            prob *= 1.1  # Increase for small orders

        # Spread stability factor
        avg_spread_bps = (buy_metrics.spread_bps + sell_metrics.spread_bps) / 2
        if avg_spread_bps > 50:
            prob *= 0.8  # Wide spreads are less stable
        elif avg_spread_bps < 10:
            prob *= 1.05  # Tight spreads are more predictable

        # Quality score factor
        avg_quality = (buy_metrics.quality_score + sell_metrics.quality_score) / 2
        prob *= (avg_quality / 100)

        # Venue reliability factor
        buy_reliability = self.venue_reliability.get(buy_metrics.venue, 0.85)
        sell_reliability = self.venue_reliability.get(sell_metrics.venue, 0.85)
        prob *= (buy_reliability * sell_reliability) ** 0.5

        # === MICROSTRUCTURE FACTORS ===

        # OBI factor - extreme imbalances reduce execution probability
        max_obi = max(abs(buy_metrics.obi), abs(sell_metrics.obi))
        if max_obi > 0.6:
            prob *= 0.85  # Extreme imbalance - prices may move
        elif max_obi > 0.4:
            prob *= 0.95

        # Toxicity factor - high toxicity means informed traders
        max_toxicity = max(buy_metrics.toxicity_score, sell_metrics.toxicity_score)
        if max_toxicity > 0.6:
            prob *= 0.75  # High toxicity - adverse selection risk
        elif max_toxicity > 0.4:
            prob *= 0.9

        # Spread dynamics factor
        avg_percentile = (buy_metrics.spread_percentile + sell_metrics.spread_percentile) / 2
        if avg_percentile > 0.8:
            prob *= 0.85  # Wide spreads vs history - unstable
        elif avg_percentile < 0.2:
            prob *= 1.05  # Tight spreads - stable conditions

        # Spread trend factor
        avg_trend = (buy_metrics.spread_trend + sell_metrics.spread_trend) / 2
        if avg_trend > 0.5:
            prob *= 0.9  # Spreads widening - conditions deteriorating

        # VWAP slippage factor
        total_slippage = buy_metrics.vwap_slippage_buy_bps + sell_metrics.vwap_slippage_sell_bps
        if total_slippage > 30:
            prob *= 0.8  # High slippage expected
        elif total_slippage > 20:
            prob *= 0.9

        return max(0.1, min(0.99, prob))

    def score_opportunity(
        self,
        market: str,
        buy_venue: str,
        buy_price: float,
        buy_order_book: List[Tuple[float, float]],
        sell_venue: str,
        sell_price: float,
        sell_order_book: List[Tuple[float, float]],
        target_quantity: float
    ) -> ScoredOpportunity:
        """
        Score an arbitrage opportunity comprehensively with full microstructure analysis.

        Includes:
        - Order Book Imbalance (OBI) for price direction prediction
        - VWAP-based realistic fill price estimation
        - Pressure gradient analysis for support/resistance
        - Toxicity scoring for informed trader detection
        - Spread dynamics for timing signals
        """
        # Get mid prices
        buy_mid = (buy_order_book[0][0] + buy_order_book[-1][0]) / 2 if buy_order_book else buy_price
        sell_mid = (sell_order_book[0][0] + sell_order_book[-1][0]) / 2 if sell_order_book else sell_price

        # Analyze order books with microstructure metrics
        buy_metrics = self.analyze_order_book(
            buy_venue, market, buy_order_book, buy_order_book, buy_mid, target_quantity
        )
        sell_metrics = self.analyze_order_book(
            sell_venue, market, sell_order_book, sell_order_book, sell_mid, target_quantity
        )

        # Basic spread calculation
        spread = sell_price - buy_price
        spread_percent = (spread / buy_price) * 100 if buy_price > 0 else 0

        # === VWAP-BASED REALISTIC PRICING ===
        # Calculate realistic fill prices using VWAP
        realistic_buy_price, buy_slippage_bps = calculate_vwap(
            buy_order_book, target_quantity, buy_mid, "buy"
        )
        realistic_sell_price, sell_slippage_bps = calculate_vwap(
            sell_order_book, target_quantity, sell_mid, "sell"
        )

        # Realistic spread after VWAP adjustment
        realistic_spread = realistic_sell_price - realistic_buy_price
        realistic_spread_pct = (realistic_spread / realistic_buy_price) * 100 if realistic_buy_price > 0 else 0

        # Legacy slippage (for compatibility)
        buy_slippage, buy_fill_price = self.estimate_slippage(
            buy_venue, market, "buy", target_quantity, buy_order_book
        )
        sell_slippage, sell_fill_price = self.estimate_slippage(
            sell_venue, market, "sell", target_quantity, sell_order_book
        )
        total_slippage = buy_slippage + sell_slippage

        # Maximum executable quantity
        max_buy_qty = sum(q for _, q in buy_order_book[:5])
        max_sell_qty = sum(q for _, q in sell_order_book[:5])
        max_executable = min(max_buy_qty, max_sell_qty, target_quantity)

        # Fill time estimates
        buy_fill_time = self.estimate_fill_time(
            buy_venue, market, target_quantity, buy_metrics.ask_depth_1
        )
        sell_fill_time = self.estimate_fill_time(
            sell_venue, market, target_quantity, sell_metrics.bid_depth_1
        )
        total_fill_time = max(buy_fill_time, sell_fill_time)  # Parallel execution

        # === MICROSTRUCTURE SIGNALS ===

        # OBI divergence between venues (can signal arbitrage opportunity quality)
        obi_divergence = buy_metrics.obi - sell_metrics.obi

        # Toxicity assessment
        combined_toxicity = max(buy_metrics.toxicity_score, sell_metrics.toxicity_score)

        # Spread timing score (based on spread dynamics)
        spread_timing_score = self._compute_spread_timing_score(buy_metrics, sell_metrics)

        # Entry confidence based on microstructure
        entry_confidence = self._compute_entry_confidence(
            buy_metrics, sell_metrics, combined_toxicity, realistic_spread_pct
        )

        # Enhanced execution probability
        exec_prob = self.estimate_execution_probability(
            buy_metrics, sell_metrics, target_quantity, spread_percent
        )

        # Adjust execution probability based on microstructure
        if combined_toxicity > 0.5:
            exec_prob *= 0.8  # Reduce confidence in toxic environments
        if abs(obi_divergence) > 0.4:
            exec_prob *= 0.9  # Divergent books are less predictable

        # Risk-adjusted spread (use realistic VWAP-based spread)
        risk_adjusted_spread = realistic_spread_pct

        # Sharpe-like ratio (return / risk)
        volatility = self.symbol_volatility.get(market, 1.0)
        sharpe = (risk_adjusted_spread / max(volatility, 0.1)) if risk_adjusted_spread > 0 else 0

        # Liquidity scores
        buy_liquidity_score = buy_metrics.quality_score
        sell_liquidity_score = sell_metrics.quality_score
        combined_liquidity = (buy_liquidity_score * sell_liquidity_score) ** 0.5

        # === ENHANCED COMPOSITE SCORE ===
        # Weighted combination with microstructure factors
        composite = (
            risk_adjusted_spread * 25 +              # Net profitability (25%)
            exec_prob * 20 +                         # Execution confidence (20%)
            combined_liquidity * 0.15 +              # Liquidity quality (15%)
            (100 - total_fill_time / 100) * 0.10 +   # Speed (10%)
            sharpe * 10 +                            # Risk-adjusted (10%)
            entry_confidence * 10 +                  # Microstructure confidence (10%)
            spread_timing_score * 5 +                # Timing (5%)
            (1 - combined_toxicity) * 5              # Low toxicity bonus (5%)
        )

        return ScoredOpportunity(
            market=market,
            buy_venue=buy_venue,
            buy_price=buy_price,
            buy_quantity_available=max_buy_qty,
            sell_venue=sell_venue,
            sell_price=sell_price,
            sell_quantity_available=max_sell_qty,
            spread=spread,
            spread_percent=spread_percent,
            buy_liquidity_score=buy_liquidity_score,
            sell_liquidity_score=sell_liquidity_score,
            combined_liquidity_score=combined_liquidity,
            estimated_slippage_pct=total_slippage,
            estimated_fill_time_ms=total_fill_time,
            max_executable_quantity=max_executable,
            risk_adjusted_spread=risk_adjusted_spread,
            sharpe_ratio=sharpe,
            execution_probability=exec_prob,
            # New microstructure fields
            buy_obi=buy_metrics.obi,
            sell_obi=sell_metrics.obi,
            obi_divergence=obi_divergence,
            realistic_buy_price=realistic_buy_price,
            realistic_sell_price=realistic_sell_price,
            realistic_spread_pct=realistic_spread_pct,
            buy_pressure_support=buy_metrics.bid_pressure_gradient,
            sell_pressure_resistance=sell_metrics.ask_pressure_gradient,
            buy_venue_toxicity=buy_metrics.toxicity_score,
            sell_venue_toxicity=sell_metrics.toxicity_score,
            combined_toxicity=combined_toxicity,
            spread_timing_score=spread_timing_score,
            entry_confidence=entry_confidence,
            composite_score=composite,
            # For compatibility with main_v4.py
            opportunity_id=f"opp_{market}_{buy_venue}_{sell_venue}",
            metrics=buy_metrics,
            recommended_size=max_executable,
            expected_profit=realistic_spread * max_executable,
            estimated_cost=(buy_slippage_bps + sell_slippage_bps) / 100 * max_executable * buy_price,
            overall_score=composite
        )

    def _compute_spread_timing_score(
        self,
        buy_metrics: LiquidityMetrics,
        sell_metrics: LiquidityMetrics
    ) -> float:
        """
        Compute timing score based on spread dynamics.

        Higher score = better timing (spreads are tight vs history)
        """
        score = 0.5  # Neutral

        # Favor tight spreads (low percentile)
        avg_percentile = (buy_metrics.spread_percentile + sell_metrics.spread_percentile) / 2
        if avg_percentile < 0.3:
            score = 0.8  # Great timing - tight spreads
        elif avg_percentile < 0.5:
            score = 0.6  # Good timing
        elif avg_percentile > 0.7:
            score = 0.3  # Poor timing - wide spreads
        elif avg_percentile > 0.8:
            score = 0.2  # Very poor timing

        # Adjust for spread trend
        avg_trend = (buy_metrics.spread_trend + sell_metrics.spread_trend) / 2
        if avg_trend < 0:
            score += 0.1  # Spreads tightening - positive
        elif avg_trend > 0.5:
            score -= 0.1  # Spreads widening - negative

        return max(0, min(1, score))

    def _compute_entry_confidence(
        self,
        buy_metrics: LiquidityMetrics,
        sell_metrics: LiquidityMetrics,
        combined_toxicity: float,
        realistic_spread_pct: float
    ) -> float:
        """
        Compute entry confidence based on microstructure analysis.
        """
        confidence = 0.5  # Start neutral

        # OBI signals
        # For arbitrage, we want to buy where OBI is negative (sellers dominating)
        # and sell where OBI is positive (buyers dominating)
        if buy_metrics.obi < -0.2:
            confidence += 0.1  # Good to buy - sellers present
        if sell_metrics.obi > 0.2:
            confidence += 0.1  # Good to sell - buyers present

        # Depth-weighted OBI (more reliable)
        if buy_metrics.obi_depth_weighted < -0.15:
            confidence += 0.05
        if sell_metrics.obi_depth_weighted > 0.15:
            confidence += 0.05

        # Spread analysis
        avg_percentile = (buy_metrics.spread_percentile + sell_metrics.spread_percentile) / 2
        if avg_percentile < 0.3:
            confidence += 0.1  # Tight spreads
        elif avg_percentile > 0.7:
            confidence -= 0.15  # Wide spreads

        # Toxicity penalty
        if combined_toxicity > 0.5:
            confidence -= 0.2
        elif combined_toxicity < 0.2:
            confidence += 0.05

        # Pressure gradients
        # Strong bid support at buy venue is good
        if buy_metrics.bid_pressure_gradient > 3:
            confidence += 0.05
        # Strong ask resistance at sell venue is good (stops runaway prices)
        if sell_metrics.ask_pressure_gradient > 3:
            confidence += 0.05

        # Realistic spread must be positive
        if realistic_spread_pct <= 0:
            confidence = 0.1  # Very low confidence for negative spreads

        return max(0, min(1, confidence))

    def rank_opportunities(
        self,
        opportunities: List[ScoredOpportunity],
        min_score: float = 0.0
    ) -> List[ScoredOpportunity]:
        """
        Rank opportunities by composite score and filter by minimum.
        """
        filtered = [o for o in opportunities if o.composite_score >= min_score]
        return sorted(filtered, key=lambda o: o.composite_score, reverse=True)

    def record_execution(
        self,
        opportunity: ScoredOpportunity,
        success: bool,
        actual_slippage: float,
        actual_fill_time_ms: float
    ):
        """
        Record execution outcome for model improvement.
        """
        self.execution_history.append({
            "opportunity": opportunity.to_dict(),
            "success": success,
            "actual_slippage": actual_slippage,
            "actual_fill_time_ms": actual_fill_time_ms,
            "predicted_slippage": opportunity.estimated_slippage_pct,
            "predicted_fill_time_ms": opportunity.estimated_fill_time_ms,
            "timestamp": datetime.now().timestamp()
        })

        # Update venue reliability
        venue_key = f"{opportunity.buy_venue}:{opportunity.sell_venue}"
        current = self.venue_reliability.get(venue_key, 0.85)

        if success:
            self.venue_reliability[venue_key] = min(0.99, current * 1.01)
        else:
            self.venue_reliability[venue_key] = max(0.5, current * 0.95)


    def get_microstructure_signals(
        self,
        venue: str,
        symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        trade_size: float = 1.0
    ) -> MicrostructureSignals:
        """
        Get comprehensive microstructure signals for a market.

        This is a convenience method that returns all microstructure
        signals in a structured format.
        """
        if not bids or not asks:
            return MicrostructureSignals()

        mid_price = (bids[0][0] + asks[0][0]) / 2

        signals = MicrostructureSignals()

        # OBI calculations
        signals.obi = calculate_obi(bids, asks, levels=10)
        signals.obi_5_level = calculate_obi(bids, asks, levels=5)
        signals.obi_depth_weighted = calculate_depth_weighted_obi(bids, asks, levels=10)

        # VWAP calculations
        signals.vwap_buy, signals.vwap_slippage_buy_bps = calculate_vwap(
            asks, trade_size, mid_price, "buy"
        )
        signals.vwap_sell, signals.vwap_slippage_sell_bps = calculate_vwap(
            bids, trade_size, mid_price, "sell"
        )

        # Pressure gradients
        signals.bid_pressure_gradient = calculate_book_pressure_gradient(bids, levels=5)
        signals.ask_pressure_gradient = calculate_book_pressure_gradient(asks, levels=5)
        signals.pressure_imbalance = signals.bid_pressure_gradient - signals.ask_pressure_gradient

        # Toxicity from trade history
        key = f"{venue}:{symbol}"
        trades = list(self.trade_history.get(key, []))
        if trades:
            signals.toxicity_score = calculate_toxicity(trades, window_minutes=5)

        # Spread dynamics
        spread_history_list = list(self.spread_history.get(key, []))
        if len(spread_history_list) >= 10:
            spread_dynamics = analyze_spread_dynamics(spread_history_list)
            signals.spread_percentile = spread_dynamics["percentile"]
            signals.spread_trend = spread_dynamics["trend"]
            signals.spread_is_tight = spread_dynamics["is_tight"]
            signals.spread_is_wide = spread_dynamics["is_wide"]

        # Derived signals
        signals.price_direction_signal = self._calculate_price_direction(signals)
        signals.execution_risk_score = self._calculate_execution_risk(signals)
        signals.timing_score = self._calculate_timing_score(signals)

        return signals

    def _calculate_price_direction(self, signals: MicrostructureSignals) -> float:
        """Calculate price direction signal from microstructure"""
        direction = 0.0

        # OBI is primary signal
        direction += signals.obi * 0.4

        # Depth-weighted OBI
        direction += signals.obi_depth_weighted * 0.3

        # Pressure imbalance
        if signals.pressure_imbalance > 2:
            direction += 0.15
        elif signals.pressure_imbalance < -2:
            direction -= 0.15

        return max(-1, min(1, direction))

    def _calculate_execution_risk(self, signals: MicrostructureSignals) -> float:
        """Calculate execution risk score (0-1, lower is better)"""
        risk = 0.2  # Base risk

        # High slippage adds risk
        total_slippage = signals.vwap_slippage_buy_bps + signals.vwap_slippage_sell_bps
        risk += min(0.3, total_slippage / 100)

        # Toxicity adds risk
        risk += signals.toxicity_score * 0.25

        # Wide spreads add risk
        if signals.spread_is_wide:
            risk += 0.15

        # Extreme OBI adds risk
        if abs(signals.obi) > 0.5:
            risk += 0.1

        return max(0, min(1, risk))

    def _calculate_timing_score(self, signals: MicrostructureSignals) -> float:
        """Calculate timing score (0-1, higher is better)"""
        score = 0.5  # Neutral

        # Tight spreads improve timing
        if signals.spread_is_tight:
            score += 0.2
        elif signals.spread_is_wide:
            score -= 0.2

        # Favorable OBI improves timing
        if abs(signals.obi) < 0.2:
            score += 0.1  # Balanced book

        # Low toxicity improves timing
        if signals.toxicity_score < 0.2:
            score += 0.1
        elif signals.toxicity_score > 0.5:
            score -= 0.15

        # Positive pressure gradient improves timing
        if signals.pressure_imbalance > 0:
            score += 0.05

        return max(0, min(1, score))


# Test
def test_liquidity_scorer():
    """Test the liquidity scorer with microstructure analysis"""
    scorer = AdvancedLiquidityScorer()

    # Mock order books with realistic structure
    buy_book = [
        (42500, 1.5),
        (42490, 2.0),
        (42480, 3.0),
        (42470, 4.0),
        (42460, 5.0),
    ]

    sell_book = [
        (42650, 1.0),
        (42660, 2.0),
        (42670, 2.5),
        (42680, 3.0),
        (42690, 4.0),
    ]

    print("=" * 60)
    print("MICROSTRUCTURE ANALYSIS TEST")
    print("=" * 60)

    # Test individual microstructure functions
    print("\n1. Order Book Imbalance (OBI):")
    obi = calculate_obi(buy_book, sell_book, levels=5)
    obi_weighted = calculate_depth_weighted_obi(buy_book, sell_book)
    print(f"   OBI (5 levels): {obi:.3f}")
    print(f"   OBI (depth-weighted): {obi_weighted:.3f}")

    print("\n2. VWAP Analysis:")
    mid_price = (buy_book[0][0] + sell_book[0][0]) / 2
    vwap_buy, slippage_buy = calculate_vwap(sell_book, 2.0, mid_price, "buy")
    vwap_sell, slippage_sell = calculate_vwap(buy_book, 2.0, mid_price, "sell")
    print(f"   Mid price: ${mid_price:.2f}")
    print(f"   VWAP to buy 2 units: ${vwap_buy:.2f} (slippage: {slippage_buy:.1f}bps)")
    print(f"   VWAP to sell 2 units: ${vwap_sell:.2f} (slippage: {slippage_sell:.1f}bps)")

    print("\n3. Pressure Gradients:")
    bid_gradient = calculate_book_pressure_gradient(buy_book)
    ask_gradient = calculate_book_pressure_gradient(sell_book)
    print(f"   Bid pressure gradient: {bid_gradient:.2f}")
    print(f"   Ask pressure gradient: {ask_gradient:.2f}")

    print("\n4. Spread Dynamics:")
    # Simulate spread history
    spread_history = [5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 5.0, 4.7, 5.2, 5.1, 4.8, 5.0]
    dynamics = analyze_spread_dynamics(spread_history)
    print(f"   Current: {dynamics['current']:.2f}bps")
    print(f"   Mean: {dynamics['mean']:.2f}bps")
    print(f"   Percentile: {dynamics['percentile']:.1%}")
    print(f"   Trend: {dynamics['trend']:.4f}")

    print("\n" + "=" * 60)
    print("OPPORTUNITY SCORING")
    print("=" * 60)

    # Score opportunity
    opp = scorer.score_opportunity(
        market="BTC",
        buy_venue="polymarket",
        buy_price=42500,
        buy_order_book=buy_book,
        sell_venue="kraken",
        sell_price=42650,
        sell_order_book=sell_book,
        target_quantity=0.5
    )

    print(f"\nOpportunity: {opp.opportunity_id}")
    print(f"\n  Basic Metrics:")
    print(f"    Spread: {opp.spread_percent:.3f}%")
    print(f"    Realistic spread (VWAP): {opp.realistic_spread_pct:.3f}%")
    print(f"    Estimated slippage: {opp.estimated_slippage_pct:.3f}%")

    print(f"\n  Microstructure Signals:")
    print(f"    Buy OBI: {opp.buy_obi:.3f}")
    print(f"    Sell OBI: {opp.sell_obi:.3f}")
    print(f"    OBI divergence: {opp.obi_divergence:.3f}")
    print(f"    Combined toxicity: {opp.combined_toxicity:.3f}")
    print(f"    Spread timing: {opp.spread_timing_score:.3f}")

    print(f"\n  Execution Metrics:")
    print(f"    Execution probability: {opp.execution_probability:.1%}")
    print(f"    Entry confidence: {opp.entry_confidence:.1%}")
    print(f"    Fill time estimate: {opp.estimated_fill_time_ms:.0f}ms")

    print(f"\n  Final Scores:")
    print(f"    Composite score: {opp.composite_score:.2f}")
    print(f"    Overall score: {opp.overall_score:.2f}")

    print("\n" + "=" * 60)
    print("MICROSTRUCTURE SIGNALS HELPER")
    print("=" * 60)

    signals = scorer.get_microstructure_signals(
        "kraken", "BTC", buy_book, sell_book, trade_size=1.0
    )
    print(f"\n  Signals for Kraken BTC:")
    print(f"    OBI: {signals.obi:.3f}")
    print(f"    VWAP Buy: ${signals.vwap_buy:.2f}")
    print(f"    VWAP Sell: ${signals.vwap_sell:.2f}")
    print(f"    Price direction signal: {signals.price_direction_signal:.3f}")
    print(f"    Execution risk: {signals.execution_risk_score:.3f}")
    print(f"    Timing score: {signals.timing_score:.3f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_liquidity_scorer()
