"""
Order Book Microstructure Analyzer
Priority 2 implementation from analysis report.

Features:
- Order Book Imbalance (OBI) - predictive of short-term price moves
- VWAP distance calculation - realistic fill price estimation
- Book Pressure Gradient - identifies support/resistance
- Toxicity Score (VPIN approximation) - detects informed traders
- Spread Dynamics Analysis - timing signals
- Microstructure-based entry signals
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("PolyMangoBot.microstructure")


class SignalStrength(Enum):
    """Signal strength classification"""
    STRONG_BUY = "strong_buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    STRONG_SELL = "strong_sell"


class MarketCondition(Enum):
    """Current market microstructure condition"""
    HEALTHY = "healthy"          # Normal trading conditions
    STRESSED = "stressed"        # Elevated toxicity/wide spreads
    ILLIQUID = "illiquid"       # Low depth
    MANIPULATED = "manipulated"  # Quote stuffing detected
    TRENDING = "trending"        # Strong directional flow


@dataclass
class OrderBookLevel:
    """Single order book level"""
    price: float
    quantity: float
    order_count: int = 1


@dataclass
class OrderBook:
    """Full order book with bid/ask sides"""
    venue: str
    symbol: str
    bids: List[OrderBookLevel]  # Sorted descending by price (best bid first)
    asks: List[OrderBookLevel]  # Sorted ascending by price (best ask first)
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0

    @property
    def mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return 0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0


@dataclass
class MicrostructureMetrics:
    """Comprehensive microstructure metrics"""
    # Basic metrics
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0

    # Order Book Imbalance
    obi: float = 0.0                    # -1 to 1, positive = bid heavy
    obi_depth_weighted: float = 0.0     # Weighted by depth level
    obi_5_level: float = 0.0            # Top 5 levels only

    # Depth metrics
    bid_depth_total: float = 0.0
    ask_depth_total: float = 0.0
    bid_depth_5: float = 0.0            # Top 5 levels
    ask_depth_5: float = 0.0
    depth_ratio: float = 0.0            # bid/ask ratio

    # VWAP metrics
    vwap_buy_100: float = 0.0           # VWAP to buy 100 units
    vwap_sell_100: float = 0.0          # VWAP to sell 100 units
    vwap_slippage_buy_bps: float = 0.0  # Slippage in bps
    vwap_slippage_sell_bps: float = 0.0

    # Pressure gradients
    bid_pressure_gradient: float = 0.0   # Rate of depth change
    ask_pressure_gradient: float = 0.0
    pressure_imbalance: float = 0.0

    # Toxicity
    toxicity_score: float = 0.0          # 0-1, higher = more informed flow
    order_flow_imbalance: float = 0.0    # Recent buy/sell imbalance

    # Spread dynamics
    spread_percentile: float = 0.0       # Current spread vs history
    spread_trend: float = 0.0            # Widening or tightening
    spread_volatility: float = 0.0

    # Derived signals
    signal_strength: SignalStrength = SignalStrength.NEUTRAL
    market_condition: MarketCondition = MarketCondition.HEALTHY
    entry_confidence: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "mid_price": self.mid_price,
            "spread_bps": self.spread_bps,
            "obi": self.obi,
            "obi_5_level": self.obi_5_level,
            "depth_ratio": self.depth_ratio,
            "vwap_slippage_buy_bps": self.vwap_slippage_buy_bps,
            "vwap_slippage_sell_bps": self.vwap_slippage_sell_bps,
            "pressure_imbalance": self.pressure_imbalance,
            "toxicity_score": self.toxicity_score,
            "spread_percentile": self.spread_percentile,
            "signal_strength": self.signal_strength.value,
            "market_condition": self.market_condition.value,
            "entry_confidence": self.entry_confidence
        }


@dataclass
class TradeRecord:
    """Record of observed trade"""
    price: float
    quantity: float
    side: str  # "buy" or "sell"
    timestamp: float
    is_aggressive: bool = True  # Taker order


class MicrostructureAnalyzer:
    """
    Comprehensive order book microstructure analyzer.

    Provides signals for:
    - Short-term price direction prediction
    - Optimal entry timing
    - Risk assessment
    - Fill price estimation
    """

    def __init__(
        self,
        spread_history_size: int = 500,
        trade_history_size: int = 1000,
        toxicity_window_seconds: float = 300  # 5 minutes
    ):
        self.spread_history_size = spread_history_size
        self.trade_history_size = trade_history_size
        self.toxicity_window_seconds = toxicity_window_seconds

        # History tracking
        self._spread_history: Dict[str, deque] = {}  # venue:symbol -> spreads
        self._trade_history: Dict[str, deque] = {}   # venue:symbol -> trades
        self._obi_history: Dict[str, deque] = {}     # venue:symbol -> OBI values
        self._quote_timestamps: Dict[str, deque] = {} # For quote stuffing detection

        # Cached metrics
        self._cached_metrics: Dict[str, MicrostructureMetrics] = {}
        self._cache_ttl_ms = 100  # 100ms cache

    def analyze(self, order_book: OrderBook) -> MicrostructureMetrics:
        """
        Perform comprehensive microstructure analysis.
        """
        key = f"{order_book.venue}:{order_book.symbol}"

        # Initialize history if needed
        if key not in self._spread_history:
            self._spread_history[key] = deque(maxlen=self.spread_history_size)
            self._trade_history[key] = deque(maxlen=self.trade_history_size)
            self._obi_history[key] = deque(maxlen=self.spread_history_size)
            self._quote_timestamps[key] = deque(maxlen=1000)

        metrics = MicrostructureMetrics()

        # Basic metrics
        metrics.mid_price = order_book.mid_price
        metrics.spread = order_book.spread
        metrics.spread_bps = order_book.spread_bps

        # Order Book Imbalance
        metrics.obi = self._calculate_obi(order_book.bids, order_book.asks, levels=10)
        metrics.obi_5_level = self._calculate_obi(order_book.bids, order_book.asks, levels=5)
        metrics.obi_depth_weighted = self._calculate_depth_weighted_obi(
            order_book.bids, order_book.asks
        )

        # Update OBI history
        self._obi_history[key].append(metrics.obi)

        # Depth metrics
        metrics.bid_depth_total = sum(level.quantity for level in order_book.bids)
        metrics.ask_depth_total = sum(level.quantity for level in order_book.asks)
        metrics.bid_depth_5 = sum(level.quantity for level in order_book.bids[:5])
        metrics.ask_depth_5 = sum(level.quantity for level in order_book.asks[:5])

        if metrics.ask_depth_total > 0:
            metrics.depth_ratio = metrics.bid_depth_total / metrics.ask_depth_total
        else:
            metrics.depth_ratio = 1.0

        # VWAP calculations
        target_size = 100  # Default target size for VWAP calculation
        metrics.vwap_buy_100, metrics.vwap_slippage_buy_bps = self._calculate_vwap(
            order_book.asks, target_size, order_book.mid_price, "buy"
        )
        metrics.vwap_sell_100, metrics.vwap_slippage_sell_bps = self._calculate_vwap(
            order_book.bids, target_size, order_book.mid_price, "sell"
        )

        # Pressure gradients
        metrics.bid_pressure_gradient = self._calculate_pressure_gradient(order_book.bids)
        metrics.ask_pressure_gradient = self._calculate_pressure_gradient(order_book.asks)
        metrics.pressure_imbalance = metrics.bid_pressure_gradient - metrics.ask_pressure_gradient

        # Toxicity from trade history
        trades = list(self._trade_history.get(key, []))
        metrics.toxicity_score = self._calculate_toxicity(trades)
        metrics.order_flow_imbalance = self._calculate_order_flow_imbalance(trades)

        # Spread dynamics
        self._spread_history[key].append(metrics.spread_bps)
        spread_history = list(self._spread_history[key])

        if len(spread_history) >= 10:
            metrics.spread_percentile = self._calculate_percentile(
                spread_history, metrics.spread_bps
            )
            metrics.spread_trend = self._calculate_trend(spread_history[-20:])
            metrics.spread_volatility = np.std(spread_history[-50:]) if len(spread_history) >= 50 else 0

        # Quote stuffing detection
        self._quote_timestamps[key].append(time.time())
        is_stuffing = self._detect_quote_stuffing(key)

        # Determine market condition
        metrics.market_condition = self._classify_market_condition(
            metrics, is_stuffing
        )

        # Generate entry signal
        metrics.signal_strength, metrics.entry_confidence = self._generate_signal(
            metrics
        )

        self._cached_metrics[key] = metrics
        return metrics

    def _calculate_obi(
        self,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
        levels: int = 10
    ) -> float:
        """
        Calculate Order Book Imbalance.
        OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Returns: -1 to 1, positive means bid-heavy (bullish signal)
        """
        bid_volume = sum(level.quantity for level in bids[:levels])
        ask_volume = sum(level.quantity for level in asks[:levels])
        total = bid_volume + ask_volume

        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    def _calculate_depth_weighted_obi(
        self,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
        max_levels: int = 10
    ) -> float:
        """
        Calculate depth-weighted OBI.
        Levels closer to mid-price are weighted more heavily.
        """
        # Weights: first level = 1.0, decaying exponentially
        weights = [0.5 ** i for i in range(max_levels)]

        weighted_bid = sum(
            bids[i].quantity * weights[i]
            for i in range(min(len(bids), max_levels))
        )
        weighted_ask = sum(
            asks[i].quantity * weights[i]
            for i in range(min(len(asks), max_levels))
        )

        total = weighted_bid + weighted_ask
        if total == 0:
            return 0.0

        return (weighted_bid - weighted_ask) / total

    def _calculate_vwap(
        self,
        levels: List[OrderBookLevel],
        target_size: float,
        mid_price: float,
        side: str
    ) -> Tuple[float, float]:
        """
        Calculate VWAP and slippage for executing target_size.

        Returns: (vwap_price, slippage_bps)
        """
        if not levels or target_size <= 0:
            return mid_price, 0.0

        remaining = target_size
        total_cost = 0.0
        filled = 0.0

        for level in levels:
            fill_qty = min(remaining, level.quantity)
            total_cost += level.price * fill_qty
            filled += fill_qty
            remaining -= fill_qty

            if remaining <= 0:
                break

        if filled == 0:
            return mid_price, 0.0

        vwap = total_cost / filled

        # Calculate slippage from mid-price
        if side == "buy":
            slippage = (vwap - mid_price) / mid_price * 10000
        else:
            slippage = (mid_price - vwap) / mid_price * 10000

        return vwap, max(0, slippage)

    def _calculate_pressure_gradient(
        self,
        levels: List[OrderBookLevel],
        num_levels: int = 5
    ) -> float:
        """
        Calculate rate of change in volume across price levels.
        Steep positive gradient = strong support/resistance.
        """
        if len(levels) < 2:
            return 0.0

        quantities = [level.quantity for level in levels[:num_levels]]

        if len(quantities) < 2:
            return 0.0

        # Linear regression slope
        x = np.arange(len(quantities))
        coeffs = np.polyfit(x, quantities, 1)

        return coeffs[0]  # Slope

    def _calculate_toxicity(
        self,
        trades: List[TradeRecord],
        window_seconds: float = None
    ) -> float:
        """
        Calculate toxicity score (VPIN approximation).
        High toxicity = informed traders present = higher adverse selection risk.

        Returns: 0-1, higher is more toxic
        """
        if window_seconds is None:
            window_seconds = self.toxicity_window_seconds

        if not trades:
            return 0.0

        # Filter to recent trades
        cutoff = time.time() - window_seconds
        recent_trades = [t for t in trades if t.timestamp > cutoff]

        if len(recent_trades) < 5:
            return 0.0

        # Calculate buy/sell volume
        buy_volume = sum(t.quantity for t in recent_trades if t.side == "buy")
        sell_volume = sum(t.quantity for t in recent_trades if t.side == "sell")
        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.0

        # Order imbalance as toxicity proxy
        imbalance = abs(buy_volume - sell_volume) / total_volume

        return imbalance

    def _calculate_order_flow_imbalance(
        self,
        trades: List[TradeRecord],
        lookback: int = 50
    ) -> float:
        """
        Calculate recent order flow imbalance.
        Positive = net buying, Negative = net selling
        """
        recent = trades[-lookback:] if len(trades) > lookback else trades

        if not recent:
            return 0.0

        buy_volume = sum(t.quantity for t in recent if t.side == "buy")
        sell_volume = sum(t.quantity for t in recent if t.side == "sell")
        total = buy_volume + sell_volume

        if total == 0:
            return 0.0

        return (buy_volume - sell_volume) / total

    def _calculate_percentile(
        self,
        history: List[float],
        current: float
    ) -> float:
        """Calculate percentile of current value in history"""
        if not history:
            return 0.5

        sorted_history = sorted(history)
        rank = np.searchsorted(sorted_history, current)

        return rank / len(sorted_history)

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (positive = increasing)"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)

        return coeffs[0]  # Slope

    def _detect_quote_stuffing(
        self,
        key: str,
        window_ms: float = 1000,
        threshold_quotes: int = 50
    ) -> bool:
        """
        Detect quote stuffing (rapid quote updates without trades).
        """
        timestamps = self._quote_timestamps.get(key, deque())

        if len(timestamps) < 10:
            return False

        now = time.time()
        cutoff = now - window_ms / 1000

        recent_quotes = sum(1 for t in timestamps if t > cutoff)

        # High quote rate without corresponding trades
        return recent_quotes > threshold_quotes

    def _classify_market_condition(
        self,
        metrics: MicrostructureMetrics,
        is_stuffing: bool
    ) -> MarketCondition:
        """Classify current market condition"""
        if is_stuffing:
            return MarketCondition.MANIPULATED

        if metrics.spread_bps > 50:  # >50bps spread
            return MarketCondition.STRESSED

        if metrics.bid_depth_5 < 10 or metrics.ask_depth_5 < 10:
            return MarketCondition.ILLIQUID

        if metrics.toxicity_score > 0.6:
            return MarketCondition.STRESSED

        if abs(metrics.obi) > 0.4 and abs(metrics.order_flow_imbalance) > 0.3:
            return MarketCondition.TRENDING

        return MarketCondition.HEALTHY

    def _generate_signal(
        self,
        metrics: MicrostructureMetrics
    ) -> Tuple[SignalStrength, float]:
        """
        Generate entry signal based on microstructure analysis.

        Returns: (signal_strength, confidence)
        """
        confidence = 0.5  # Start neutral

        # OBI signal
        if metrics.obi > 0.3:
            confidence += 0.15
        elif metrics.obi < -0.3:
            confidence -= 0.15

        # Depth-weighted OBI (more reliable)
        if metrics.obi_depth_weighted > 0.25:
            confidence += 0.1
        elif metrics.obi_depth_weighted < -0.25:
            confidence -= 0.1

        # Spread analysis
        if metrics.spread_percentile < 0.3:
            # Tight spread - good for entry
            confidence += 0.1
        elif metrics.spread_percentile > 0.7:
            # Wide spread - risky
            confidence -= 0.15

        # Toxicity penalty
        if metrics.toxicity_score > 0.5:
            confidence -= 0.2
        elif metrics.toxicity_score < 0.2:
            confidence += 0.05

        # Pressure gradient
        if metrics.pressure_imbalance > 0.5:
            confidence += 0.1
        elif metrics.pressure_imbalance < -0.5:
            confidence -= 0.1

        # Market condition adjustments
        if metrics.market_condition == MarketCondition.MANIPULATED:
            confidence = 0.0  # Don't trade
        elif metrics.market_condition == MarketCondition.STRESSED:
            confidence *= 0.7
        elif metrics.market_condition == MarketCondition.ILLIQUID:
            confidence *= 0.5

        # Classify signal
        if confidence > 0.7:
            signal = SignalStrength.STRONG_BUY
        elif confidence > 0.6:
            signal = SignalStrength.WEAK_BUY
        elif confidence < 0.3:
            signal = SignalStrength.STRONG_SELL
        elif confidence < 0.4:
            signal = SignalStrength.WEAK_SELL
        else:
            signal = SignalStrength.NEUTRAL

        return signal, max(0, min(1, confidence))

    def add_trade(self, venue: str, symbol: str, trade: TradeRecord):
        """Add observed trade to history"""
        key = f"{venue}:{symbol}"

        if key not in self._trade_history:
            self._trade_history[key] = deque(maxlen=self.trade_history_size)

        self._trade_history[key].append(trade)

    def estimate_fill_price(
        self,
        order_book: OrderBook,
        side: str,
        size: float
    ) -> Dict:
        """
        Estimate realistic fill price for an order.

        Returns detailed breakdown of expected execution.
        """
        levels = order_book.asks if side == "buy" else order_book.bids
        mid_price = order_book.mid_price

        if not levels:
            return {
                "vwap": mid_price,
                "slippage_bps": 0,
                "can_fill": False,
                "fill_levels": 0
            }

        remaining = size
        total_cost = 0.0
        filled = 0.0
        fill_levels = 0

        for level in levels:
            if remaining <= 0:
                break

            fill_qty = min(remaining, level.quantity)
            total_cost += level.price * fill_qty
            filled += fill_qty
            remaining -= fill_qty
            fill_levels += 1

        if filled == 0:
            return {
                "vwap": mid_price,
                "slippage_bps": 0,
                "can_fill": False,
                "fill_levels": 0
            }

        vwap = total_cost / filled

        if side == "buy":
            slippage_bps = (vwap - mid_price) / mid_price * 10000
        else:
            slippage_bps = (mid_price - vwap) / mid_price * 10000

        return {
            "vwap": vwap,
            "slippage_bps": max(0, slippage_bps),
            "can_fill": remaining <= 0,
            "filled_quantity": filled,
            "unfilled_quantity": remaining,
            "fill_levels": fill_levels,
            "market_impact_pct": slippage_bps / 100
        }

    def get_entry_recommendation(
        self,
        order_book: OrderBook,
        intended_side: str,
        size: float
    ) -> Dict:
        """
        Get comprehensive entry recommendation.
        """
        metrics = self.analyze(order_book)
        fill_estimate = self.estimate_fill_price(order_book, intended_side, size)

        # Should we enter?
        should_enter = (
            metrics.entry_confidence > 0.6 and
            metrics.market_condition != MarketCondition.MANIPULATED and
            fill_estimate["can_fill"] and
            fill_estimate["slippage_bps"] < 20  # Max 20bps slippage
        )

        # Optimal size adjustment
        optimal_size = size
        if fill_estimate["slippage_bps"] > 10:
            # Reduce size to limit impact
            optimal_size = size * 0.5
        if metrics.market_condition == MarketCondition.ILLIQUID:
            optimal_size = min(optimal_size, metrics.bid_depth_5 * 0.1)

        return {
            "should_enter": should_enter,
            "confidence": metrics.entry_confidence,
            "signal": metrics.signal_strength.value,
            "market_condition": metrics.market_condition.value,
            "recommended_size": optimal_size,
            "estimated_vwap": fill_estimate["vwap"],
            "estimated_slippage_bps": fill_estimate["slippage_bps"],
            "microstructure": {
                "obi": metrics.obi,
                "obi_5_level": metrics.obi_5_level,
                "spread_percentile": metrics.spread_percentile,
                "toxicity": metrics.toxicity_score,
                "depth_ratio": metrics.depth_ratio
            },
            "warnings": self._get_warnings(metrics, fill_estimate)
        }

    def _get_warnings(
        self,
        metrics: MicrostructureMetrics,
        fill_estimate: Dict
    ) -> List[str]:
        """Generate warning messages"""
        warnings = []

        if metrics.toxicity_score > 0.5:
            warnings.append("High toxicity - informed traders present")

        if metrics.spread_percentile > 0.8:
            warnings.append("Spread unusually wide")

        if not fill_estimate["can_fill"]:
            warnings.append("Insufficient liquidity to fill order")

        if fill_estimate["slippage_bps"] > 10:
            warnings.append(f"High slippage expected: {fill_estimate['slippage_bps']:.1f}bps")

        if metrics.market_condition == MarketCondition.MANIPULATED:
            warnings.append("Quote stuffing detected - avoid trading")

        if abs(metrics.obi) > 0.5:
            direction = "bid" if metrics.obi > 0 else "ask"
            warnings.append(f"Extreme order book imbalance ({direction}-heavy)")

        return warnings


class CrossVenueMicrostructure:
    """
    Analyze microstructure across multiple venues for arbitrage.
    """

    def __init__(self):
        self.analyzers: Dict[str, MicrostructureAnalyzer] = {}

    def get_analyzer(self, venue: str) -> MicrostructureAnalyzer:
        """Get or create analyzer for venue"""
        if venue not in self.analyzers:
            self.analyzers[venue] = MicrostructureAnalyzer()
        return self.analyzers[venue]

    def analyze_arbitrage_opportunity(
        self,
        buy_book: OrderBook,
        sell_book: OrderBook,
        size: float
    ) -> Dict:
        """
        Analyze arbitrage opportunity considering microstructure.
        """
        buy_analyzer = self.get_analyzer(buy_book.venue)
        sell_analyzer = self.get_analyzer(sell_book.venue)

        # Analyze each side
        buy_metrics = buy_analyzer.analyze(buy_book)
        sell_metrics = sell_analyzer.analyze(sell_book)

        # Estimate fills
        buy_fill = buy_analyzer.estimate_fill_price(buy_book, "buy", size)
        sell_fill = sell_analyzer.estimate_fill_price(sell_book, "sell", size)

        # Raw spread
        raw_spread = sell_book.best_bid - buy_book.best_ask
        raw_spread_bps = (raw_spread / buy_book.best_ask) * 10000

        # Realistic spread after slippage
        realistic_buy_price = buy_fill["vwap"]
        realistic_sell_price = sell_fill["vwap"]
        realistic_spread = realistic_sell_price - realistic_buy_price
        realistic_spread_bps = (realistic_spread / realistic_buy_price) * 10000 if realistic_buy_price > 0 else 0

        # Combined confidence
        combined_confidence = (
            buy_metrics.entry_confidence * 0.5 +
            sell_metrics.entry_confidence * 0.5
        )

        # Risk score
        risk_score = self._calculate_risk_score(
            buy_metrics, sell_metrics, buy_fill, sell_fill
        )

        # Should execute?
        should_execute = (
            realistic_spread_bps > 5 and  # Minimum 5bps after costs
            combined_confidence > 0.5 and
            risk_score < 0.7 and
            buy_fill["can_fill"] and
            sell_fill["can_fill"] and
            buy_metrics.market_condition != MarketCondition.MANIPULATED and
            sell_metrics.market_condition != MarketCondition.MANIPULATED
        )

        return {
            "should_execute": should_execute,
            "raw_spread_bps": raw_spread_bps,
            "realistic_spread_bps": realistic_spread_bps,
            "spread_erosion_bps": raw_spread_bps - realistic_spread_bps,
            "combined_confidence": combined_confidence,
            "risk_score": risk_score,
            "buy_side": {
                "venue": buy_book.venue,
                "entry_price": buy_book.best_ask,
                "vwap": buy_fill["vwap"],
                "slippage_bps": buy_fill["slippage_bps"],
                "confidence": buy_metrics.entry_confidence,
                "condition": buy_metrics.market_condition.value,
                "obi": buy_metrics.obi
            },
            "sell_side": {
                "venue": sell_book.venue,
                "entry_price": sell_book.best_bid,
                "vwap": sell_fill["vwap"],
                "slippage_bps": sell_fill["slippage_bps"],
                "confidence": sell_metrics.entry_confidence,
                "condition": sell_metrics.market_condition.value,
                "obi": sell_metrics.obi
            },
            "warnings": self._get_combined_warnings(
                buy_metrics, sell_metrics, buy_fill, sell_fill
            )
        }

    def _calculate_risk_score(
        self,
        buy_metrics: MicrostructureMetrics,
        sell_metrics: MicrostructureMetrics,
        buy_fill: Dict,
        sell_fill: Dict
    ) -> float:
        """Calculate combined risk score (0-1, lower is better)"""
        risk = 0.0

        # Toxicity risk
        risk += (buy_metrics.toxicity_score + sell_metrics.toxicity_score) * 0.15

        # Spread risk
        risk += buy_metrics.spread_percentile * 0.1
        risk += sell_metrics.spread_percentile * 0.1

        # Slippage risk
        total_slippage = buy_fill["slippage_bps"] + sell_fill["slippage_bps"]
        risk += min(1, total_slippage / 50) * 0.2

        # Market condition risk
        for condition in [buy_metrics.market_condition, sell_metrics.market_condition]:
            if condition == MarketCondition.MANIPULATED:
                risk += 0.3
            elif condition == MarketCondition.STRESSED:
                risk += 0.15
            elif condition == MarketCondition.ILLIQUID:
                risk += 0.1

        return min(1, risk)

    def _get_combined_warnings(
        self,
        buy_metrics: MicrostructureMetrics,
        sell_metrics: MicrostructureMetrics,
        buy_fill: Dict,
        sell_fill: Dict
    ) -> List[str]:
        """Get combined warnings from both sides"""
        warnings = []

        if buy_metrics.toxicity_score > 0.5 or sell_metrics.toxicity_score > 0.5:
            warnings.append("High toxicity on one or more venues")

        total_slippage = buy_fill["slippage_bps"] + sell_fill["slippage_bps"]
        if total_slippage > 20:
            warnings.append(f"High combined slippage: {total_slippage:.1f}bps")

        if not buy_fill["can_fill"]:
            warnings.append(f"Insufficient liquidity on {buy_metrics}")

        if not sell_fill["can_fill"]:
            warnings.append(f"Insufficient liquidity on sell side")

        return warnings


# Test function
def test_microstructure():
    """Test the microstructure analyzer"""
    analyzer = MicrostructureAnalyzer()

    # Create sample order book
    bids = [
        OrderBookLevel(price=100.00, quantity=50),
        OrderBookLevel(price=99.95, quantity=100),
        OrderBookLevel(price=99.90, quantity=200),
        OrderBookLevel(price=99.85, quantity=150),
        OrderBookLevel(price=99.80, quantity=300),
    ]

    asks = [
        OrderBookLevel(price=100.05, quantity=30),
        OrderBookLevel(price=100.10, quantity=80),
        OrderBookLevel(price=100.15, quantity=150),
        OrderBookLevel(price=100.20, quantity=200),
        OrderBookLevel(price=100.25, quantity=250),
    ]

    order_book = OrderBook(
        venue="test",
        symbol="BTC",
        bids=bids,
        asks=asks
    )

    print("Order Book Microstructure Analysis")
    print("=" * 50)

    metrics = analyzer.analyze(order_book)

    print(f"Mid Price: ${metrics.mid_price:.2f}")
    print(f"Spread: {metrics.spread_bps:.2f} bps")
    print()
    print(f"Order Book Imbalance (OBI): {metrics.obi:.3f}")
    print(f"OBI (5-level): {metrics.obi_5_level:.3f}")
    print(f"OBI (depth-weighted): {metrics.obi_depth_weighted:.3f}")
    print()
    print(f"Bid Depth (total): {metrics.bid_depth_total:.0f}")
    print(f"Ask Depth (total): {metrics.ask_depth_total:.0f}")
    print(f"Depth Ratio: {metrics.depth_ratio:.2f}")
    print()
    print(f"VWAP Buy 100: ${metrics.vwap_buy_100:.2f} (slippage: {metrics.vwap_slippage_buy_bps:.1f}bps)")
    print(f"VWAP Sell 100: ${metrics.vwap_sell_100:.2f} (slippage: {metrics.vwap_slippage_sell_bps:.1f}bps)")
    print()
    print(f"Bid Pressure Gradient: {metrics.bid_pressure_gradient:.2f}")
    print(f"Ask Pressure Gradient: {metrics.ask_pressure_gradient:.2f}")
    print()
    print(f"Market Condition: {metrics.market_condition.value}")
    print(f"Signal Strength: {metrics.signal_strength.value}")
    print(f"Entry Confidence: {metrics.entry_confidence:.2%}")

    # Test entry recommendation
    print("\n" + "=" * 50)
    print("Entry Recommendation (size=50)")
    recommendation = analyzer.get_entry_recommendation(order_book, "buy", 50)
    print(f"Should Enter: {recommendation['should_enter']}")
    print(f"Confidence: {recommendation['confidence']:.2%}")
    print(f"Recommended Size: {recommendation['recommended_size']:.1f}")
    print(f"Estimated VWAP: ${recommendation['estimated_vwap']:.2f}")
    print(f"Estimated Slippage: {recommendation['estimated_slippage_bps']:.1f}bps")
    if recommendation['warnings']:
        print(f"Warnings: {recommendation['warnings']}")


if __name__ == "__main__":
    test_microstructure()
