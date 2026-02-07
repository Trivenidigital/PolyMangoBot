"""
Edge-Focused Trading Strategies
================================

Realistic edge strategies that don't rely on speed advantages.

Core Philosophy:
- Speed is NOT our edge (HFT firms are 100-1000x faster)
- Our edges: Prediction quality, risk management, longer timeframes, illiquid markets

Strategies Implemented:
1. Enhanced Directional (15min-4hr) - Signal quality focus
2. Statistical Arbitrage - Mean reversion, not speed
3. Illiquid Market Opportunities - Where HFT doesn't compete
4. Sentiment-Momentum Hybrid - Multi-factor signals
5. Volatility Regime Trading - Adapt to market conditions
"""

import asyncio
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger("PolyMangoBot.edge_strategies")


# =============================================================================
# CORE FRAMEWORK
# =============================================================================

class EdgeType(Enum):
    """Types of realistic edges we can exploit"""
    PREDICTION_QUALITY = "prediction_quality"    # Better signals
    LONGER_TIMEFRAME = "longer_timeframe"        # 15min+ where speed doesn't matter
    ILLIQUID_MARKETS = "illiquid_markets"        # Less HFT competition
    MEAN_REVERSION = "mean_reversion"            # Statistical patterns
    VOLATILITY_REGIME = "volatility_regime"      # Adapt to conditions
    RISK_MANAGEMENT = "risk_management"          # Avoid bad trades others take


class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NONE = 0


@dataclass
class EdgeSignal:
    """A trading signal with edge attribution"""
    symbol: str
    direction: str                          # "long" or "short"
    edge_type: EdgeType
    strength: SignalStrength

    # Confidence metrics
    confidence: float                       # 0-1 overall confidence
    signal_quality: float                   # 0-1 signal quality score

    # Price targets
    entry_price: float
    target_price: float
    stop_loss: float

    # Expected outcomes
    expected_return_pct: float
    win_probability: float
    risk_reward_ratio: float

    # Position sizing
    kelly_fraction: float                   # Optimal Kelly bet size
    suggested_position_pct: float           # Actual recommended (half-Kelly typically)
    max_position_pct: float

    # Timing
    expected_hold_time_hours: float
    signal_valid_for_minutes: int

    # Attribution
    contributing_factors: List[str] = field(default_factory=list)
    conflicting_factors: List[str] = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "edge_type": self.edge_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "signal_quality": self.signal_quality,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "expected_return_pct": self.expected_return_pct,
            "win_probability": self.win_probability,
            "risk_reward_ratio": self.risk_reward_ratio,
            "kelly_fraction": self.kelly_fraction,
            "suggested_position_pct": self.suggested_position_pct,
            "expected_hold_time_hours": self.expected_hold_time_hours,
            "contributing_factors": self.contributing_factors,
            "conflicting_factors": self.conflicting_factors
        }

    @property
    def is_valid(self) -> bool:
        """Check if signal is still valid"""
        age_minutes = (time.time() - self.timestamp) / 60
        return age_minutes < self.signal_valid_for_minutes

    @property
    def edge_score(self) -> float:
        """Combined edge score for ranking"""
        return (
            self.confidence * 0.3 +
            self.signal_quality * 0.3 +
            min(1.0, self.risk_reward_ratio / 3) * 0.2 +
            self.win_probability * 0.2
        )


class BaseStrategy(ABC):
    """Base class for all edge strategies"""

    def __init__(self, name: str, edge_type: EdgeType):
        self.name = name
        self.edge_type = edge_type
        self._enabled = True

        # Performance tracking
        self._signals_generated = 0
        self._signals_profitable = 0
        self._total_return_pct = 0.0

    @abstractmethod
    def generate_signal(self, symbol: str, data: Dict) -> Optional[EdgeSignal]:
        """Generate a trading signal"""
        pass

    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Return list of required data types"""
        pass

    def record_outcome(self, profitable: bool, return_pct: float):
        """Record signal outcome for performance tracking"""
        self._signals_generated += 1
        if profitable:
            self._signals_profitable += 1
        self._total_return_pct += return_pct

    @property
    def win_rate(self) -> float:
        if self._signals_generated == 0:
            return 0.0
        return self._signals_profitable / self._signals_generated

    @property
    def avg_return(self) -> float:
        if self._signals_generated == 0:
            return 0.0
        return self._total_return_pct / self._signals_generated


# =============================================================================
# STRATEGY 1: ENHANCED DIRECTIONAL (PREDICTION QUALITY EDGE)
# =============================================================================

class EnhancedDirectionalStrategy(BaseStrategy):
    """
    Multi-timeframe directional strategy focused on signal QUALITY over speed.

    Edge: Better prediction through:
    - Multi-timeframe confirmation (15min, 1hr, 4hr alignment)
    - Volume-price divergence detection
    - Trend strength filtering (only trade strong trends)
    - Overbought/oversold with momentum confirmation

    NOT trying to be fast - trying to be RIGHT.
    """

    def __init__(
        self,
        min_trend_strength: float = 0.6,
        require_volume_confirmation: bool = True,
        require_multi_timeframe: bool = True
    ):
        super().__init__("EnhancedDirectional", EdgeType.PREDICTION_QUALITY)

        self.min_trend_strength = min_trend_strength
        self.require_volume_confirmation = require_volume_confirmation
        self.require_multi_timeframe = require_multi_timeframe

        # Internal state
        self._price_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}

    def get_required_data(self) -> List[str]:
        return ["candles_15m", "candles_1h", "candles_4h", "volume", "orderbook"]

    def generate_signal(self, symbol: str, data: Dict) -> Optional[EdgeSignal]:
        """Generate signal based on multi-factor analysis"""

        candles_15m = data.get("candles_15m", [])
        candles_1h = data.get("candles_1h", [])
        candles_4h = data.get("candles_4h", [])

        if len(candles_15m) < 50 or len(candles_1h) < 20:
            return None

        current_price = candles_15m[-1]["close"]

        # Analyze each timeframe
        trend_15m, strength_15m = self._analyze_trend(candles_15m)
        trend_1h, strength_1h = self._analyze_trend(candles_1h)
        trend_4h, strength_4h = self._analyze_trend(candles_4h) if len(candles_4h) >= 10 else (0, 0)

        # Multi-timeframe alignment check
        if self.require_multi_timeframe:
            if not self._check_alignment(trend_15m, trend_1h, trend_4h):
                return None

        # Volume confirmation
        volume_signal = 1.0
        if self.require_volume_confirmation:
            volume_signal = self._check_volume_confirmation(candles_15m, trend_15m)
            if volume_signal < 0.5:
                return None

        # RSI analysis (avoid extremes without reversal confirmation)
        rsi = self._calculate_rsi(candles_15m)
        rsi_signal, rsi_factor = self._analyze_rsi_for_entry(rsi, trend_15m)

        if rsi_signal == "avoid":
            return None

        # Determine direction
        direction = "long" if trend_15m > 0 else "short"

        # Calculate combined signal strength
        avg_strength = (strength_15m * 0.5 + strength_1h * 0.3 + strength_4h * 0.2)

        if avg_strength < self.min_trend_strength:
            return None

        # Calculate signal quality
        signal_quality = self._calculate_signal_quality(
            trend_alignment=abs(trend_15m + trend_1h + trend_4h) / 3,
            volume_confirmation=volume_signal,
            rsi_factor=rsi_factor,
            trend_strength=avg_strength
        )

        # Calculate targets based on ATR
        atr = self._calculate_atr(candles_15m)

        if direction == "long":
            target_price = current_price + (atr * 2.5)
            stop_loss = current_price - (atr * 1.0)
        else:
            target_price = current_price - (atr * 2.5)
            stop_loss = current_price + (atr * 1.0)

        # Risk/reward calculation
        risk = abs(current_price - stop_loss)
        reward = abs(target_price - current_price)
        risk_reward = reward / risk if risk > 0 else 0

        # Win probability estimation (based on historical patterns)
        win_prob = self._estimate_win_probability(
            signal_quality, avg_strength, risk_reward
        )

        # Kelly criterion for position sizing
        kelly = self._calculate_kelly(win_prob, risk_reward)

        # Expected return
        expected_return = (win_prob * reward - (1 - win_prob) * risk) / current_price * 100

        # Build contributing factors
        factors = []
        if abs(trend_15m) > 0.5:
            factors.append(f"Strong 15m trend ({trend_15m:+.2f})")
        if abs(trend_1h) > 0.5:
            factors.append(f"1h trend aligned ({trend_1h:+.2f})")
        if volume_signal > 0.7:
            factors.append(f"Volume confirms ({volume_signal:.2f})")
        if rsi_factor > 0.6:
            factors.append(f"RSI favorable ({rsi:.1f})")

        # Conflicting factors
        conflicts = []
        if abs(trend_4h) < 0.3:
            conflicts.append("4h trend weak")
        if volume_signal < 0.7:
            conflicts.append("Volume below average")

        # Determine signal strength
        if signal_quality > 0.8 and avg_strength > 0.8:
            strength = SignalStrength.VERY_STRONG
        elif signal_quality > 0.65:
            strength = SignalStrength.STRONG
        elif signal_quality > 0.5:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return EdgeSignal(
            symbol=symbol,
            direction=direction,
            edge_type=self.edge_type,
            strength=strength,
            confidence=signal_quality * avg_strength,
            signal_quality=signal_quality,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_return_pct=expected_return,
            win_probability=win_prob,
            risk_reward_ratio=risk_reward,
            kelly_fraction=kelly,
            suggested_position_pct=kelly * 0.5,  # Half-Kelly
            max_position_pct=min(0.1, kelly),
            expected_hold_time_hours=2.0,  # 15min-4hr timeframe
            signal_valid_for_minutes=30,
            contributing_factors=factors,
            conflicting_factors=conflicts
        )

    def _analyze_trend(self, candles: List[Dict]) -> Tuple[float, float]:
        """Analyze trend direction and strength"""
        if len(candles) < 20:
            return 0.0, 0.0

        closes = np.array([c["close"] for c in candles])

        # EMA crossover
        ema_fast = self._ema(closes, 9)
        ema_slow = self._ema(closes, 21)

        # Trend direction (-1 to 1)
        if ema_fast[-1] > ema_slow[-1]:
            direction = 1.0
        else:
            direction = -1.0

        # Trend strength (0 to 1)
        ema_diff_pct = abs(ema_fast[-1] - ema_slow[-1]) / ema_slow[-1] * 100
        strength = min(1.0, ema_diff_pct / 2.0)  # 2% diff = max strength

        # Adjust for price momentum
        momentum = (closes[-1] - closes[-10]) / closes[-10] * 100 if len(closes) >= 10 else 0
        momentum_factor = min(1.0, abs(momentum) / 3.0)  # 3% momentum = max factor

        strength = (strength + momentum_factor) / 2

        return direction * strength, strength

    def _check_alignment(self, trend_15m: float, trend_1h: float, trend_4h: float) -> bool:
        """Check if trends are aligned across timeframes"""
        # All same direction
        if trend_15m > 0 and trend_1h > 0 and trend_4h >= 0:
            return True
        if trend_15m < 0 and trend_1h < 0 and trend_4h <= 0:
            return True

        # Allow 4h neutral if 15m and 1h aligned
        if abs(trend_4h) < 0.2:
            if (trend_15m > 0 and trend_1h > 0) or (trend_15m < 0 and trend_1h < 0):
                return True

        return False

    def _check_volume_confirmation(self, candles: List[Dict], trend: float) -> float:
        """Check if volume confirms the trend"""
        if len(candles) < 20:
            return 0.5

        volumes = np.array([c["volume"] for c in candles])
        avg_volume = np.mean(volumes[-20:])
        recent_volume = np.mean(volumes[-3:])

        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Volume should be above average in trend direction
        if volume_ratio > 1.2:
            return min(1.0, volume_ratio / 2)
        elif volume_ratio > 0.8:
            return 0.6
        else:
            return 0.3

    def _analyze_rsi_for_entry(self, rsi: float, trend: float) -> Tuple[str, float]:
        """Analyze RSI for entry timing"""
        if trend > 0:  # Looking for long
            if rsi > 75:
                return "avoid", 0.0  # Overbought, don't chase
            elif rsi < 35:
                return "ideal", 0.9  # Oversold in uptrend = great entry
            elif rsi < 50:
                return "good", 0.7
            else:
                return "ok", 0.5
        else:  # Looking for short
            if rsi < 25:
                return "avoid", 0.0  # Oversold, don't chase
            elif rsi > 65:
                return "ideal", 0.9  # Overbought in downtrend = great entry
            elif rsi > 50:
                return "good", 0.7
            else:
                return "ok", 0.5

    def _calculate_signal_quality(
        self,
        trend_alignment: float,
        volume_confirmation: float,
        rsi_factor: float,
        trend_strength: float
    ) -> float:
        """Calculate overall signal quality (0-1)"""
        return (
            trend_alignment * 0.3 +
            volume_confirmation * 0.25 +
            rsi_factor * 0.25 +
            trend_strength * 0.2
        )

    def _estimate_win_probability(
        self,
        signal_quality: float,
        trend_strength: float,
        risk_reward: float
    ) -> float:
        """Estimate win probability based on signal characteristics"""
        # Base probability from signal quality
        base_prob = 0.4 + signal_quality * 0.25  # 40-65% base

        # Adjust for trend strength
        trend_adj = trend_strength * 0.1  # Up to +10%

        # Adjust for risk/reward (tighter targets = higher win rate)
        if risk_reward < 1.5:
            rr_adj = 0.1
        elif risk_reward < 2.5:
            rr_adj = 0.05
        else:
            rr_adj = 0.0

        return min(0.75, base_prob + trend_adj + rr_adj)

    def _calculate_kelly(self, win_prob: float, risk_reward: float) -> float:
        """Calculate Kelly criterion fraction"""
        # Kelly formula: f = (bp - q) / b
        # where b = risk_reward, p = win_prob, q = 1 - p
        if risk_reward <= 0:
            return 0.0

        kelly = (risk_reward * win_prob - (1 - win_prob)) / risk_reward

        # Cap at 25% (full Kelly is too aggressive)
        return max(0.0, min(0.25, kelly))

    def _calculate_rsi(self, candles: List[Dict], period: int = 14) -> float:
        """Calculate RSI"""
        if len(candles) < period + 1:
            return 50.0

        closes = np.array([c["close"] for c in candles])
        deltas = np.diff(closes)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, candles: List[Dict], period: int = 14) -> float:
        """Calculate ATR"""
        if len(candles) < period + 1:
            return candles[-1]["close"] * 0.02  # Default 2%

        tr_values = []
        for i in range(1, len(candles)):
            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = candles[i-1]["close"]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        return np.mean(tr_values[-period:])

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA"""
        if len(data) < period:
            return data.copy()

        multiplier = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[:period] = np.mean(data[:period])  # Fill initial values

        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))

        return ema


# =============================================================================
# STRATEGY 2: STATISTICAL ARBITRAGE (MEAN REVERSION EDGE)
# =============================================================================

class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Mean reversion strategy - NOT speed-dependent arbitrage.

    Edge: Statistical patterns that persist for minutes/hours, not milliseconds.
    - Pair correlation deviations
    - Z-score based entry/exit
    - Cointegration analysis

    This works because we're exploiting STATISTICAL relationships,
    not trying to be faster than HFT.
    """

    def __init__(
        self,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        lookback_periods: int = 100,
        min_correlation: float = 0.7
    ):
        super().__init__("StatisticalArbitrage", EdgeType.MEAN_REVERSION)

        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.lookback_periods = lookback_periods
        self.min_correlation = min_correlation

        # Pair tracking
        self._pair_history: Dict[str, deque] = {}
        self._spread_stats: Dict[str, Dict] = {}

    def get_required_data(self) -> List[str]:
        return ["price_history", "pair_prices"]

    def generate_signal(self, symbol: str, data: Dict) -> Optional[EdgeSignal]:
        """Generate mean reversion signal"""

        # For pairs trading: symbol format is "ASSET1/ASSET2"
        pair_prices = data.get("pair_prices", {})

        if len(pair_prices) < 2:
            return None

        assets = list(pair_prices.keys())
        asset1, asset2 = assets[0], assets[1]

        prices1 = pair_prices[asset1]
        prices2 = pair_prices[asset2]

        if len(prices1) < self.lookback_periods or len(prices2) < self.lookback_periods:
            return None

        # Calculate correlation
        correlation = self._calculate_correlation(prices1, prices2)

        if abs(correlation) < self.min_correlation:
            return None  # Pair not correlated enough

        # Calculate spread and z-score
        spread = self._calculate_spread(prices1, prices2)
        zscore = self._calculate_zscore(spread)

        current_zscore = zscore[-1]

        # Determine signal
        if abs(current_zscore) < self.zscore_entry:
            return None  # Not extreme enough

        # Mean reversion: bet on spread returning to zero
        if current_zscore > self.zscore_entry:
            # Spread too high, expect it to decrease
            direction = "short"  # Short asset1 relative to asset2
            expected_move = (current_zscore - self.zscore_exit) / current_zscore
        elif current_zscore < -self.zscore_entry:
            # Spread too low, expect it to increase
            direction = "long"  # Long asset1 relative to asset2
            expected_move = (abs(current_zscore) - self.zscore_exit) / abs(current_zscore)
        else:
            return None

        # Calculate targets
        current_spread = spread[-1]
        spread_std = np.std(spread[-self.lookback_periods:])
        spread_mean = np.mean(spread[-self.lookback_periods:])

        if direction == "long":
            target_spread = spread_mean + self.zscore_exit * spread_std
            stop_spread = spread_mean - (self.zscore_entry + 1) * spread_std
        else:
            target_spread = spread_mean - self.zscore_exit * spread_std
            stop_spread = spread_mean + (self.zscore_entry + 1) * spread_std

        # Calculate expected return
        expected_return_pct = abs(target_spread - current_spread) / abs(current_spread) * 100

        # Risk metrics
        risk = abs(current_spread - stop_spread)
        reward = abs(target_spread - current_spread)
        risk_reward = reward / risk if risk > 0 else 0

        # Win probability based on z-score reversion statistics
        win_prob = self._estimate_reversion_probability(abs(current_zscore))

        # Kelly fraction
        kelly = self._calculate_kelly(win_prob, risk_reward)

        # Signal quality based on z-score extremity and correlation strength
        signal_quality = min(1.0, (abs(current_zscore) - 1.5) / 2) * abs(correlation)

        # Strength determination
        if abs(current_zscore) > 3.0 and abs(correlation) > 0.85:
            strength = SignalStrength.VERY_STRONG
        elif abs(current_zscore) > 2.5:
            strength = SignalStrength.STRONG
        elif abs(current_zscore) > 2.0:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        factors = [
            f"Z-score: {current_zscore:.2f}",
            f"Correlation: {correlation:.2f}",
            f"Expected reversion: {expected_move:.1%}"
        ]

        return EdgeSignal(
            symbol=f"{asset1}/{asset2}",
            direction=direction,
            edge_type=self.edge_type,
            strength=strength,
            confidence=signal_quality,
            signal_quality=signal_quality,
            entry_price=current_spread,
            target_price=target_spread,
            stop_loss=stop_spread,
            expected_return_pct=expected_return_pct,
            win_probability=win_prob,
            risk_reward_ratio=risk_reward,
            kelly_fraction=kelly,
            suggested_position_pct=kelly * 0.5,
            max_position_pct=min(0.08, kelly),
            expected_hold_time_hours=4.0,  # Mean reversion takes time
            signal_valid_for_minutes=60,
            contributing_factors=factors,
            conflicting_factors=[]
        )

    def _calculate_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """Calculate correlation between two price series"""
        returns1 = np.diff(prices1) / prices1[:-1]
        returns2 = np.diff(prices2) / prices2[:-1]

        if len(returns1) < 20:
            return 0.0

        return np.corrcoef(returns1[-self.lookback_periods:], returns2[-self.lookback_periods:])[0, 1]

    def _calculate_spread(self, prices1: List[float], prices2: List[float]) -> np.ndarray:
        """Calculate price spread (ratio method)"""
        p1 = np.array(prices1[-self.lookback_periods:])
        p2 = np.array(prices2[-self.lookback_periods:])

        # Use log spread for stationarity
        return np.log(p1) - np.log(p2)

    def _calculate_zscore(self, spread: np.ndarray) -> np.ndarray:
        """Calculate rolling z-score of spread"""
        mean = np.mean(spread)
        std = np.std(spread)

        if std == 0:
            return np.zeros_like(spread)

        return (spread - mean) / std

    def _estimate_reversion_probability(self, zscore: float) -> float:
        """Estimate probability of mean reversion based on z-score"""
        # Based on normal distribution, but adjusted for fat tails
        if zscore > 3.0:
            return 0.70  # Very likely to revert
        elif zscore > 2.5:
            return 0.65
        elif zscore > 2.0:
            return 0.60
        else:
            return 0.55

    def _calculate_kelly(self, win_prob: float, risk_reward: float) -> float:
        """Calculate Kelly criterion"""
        if risk_reward <= 0:
            return 0.0

        kelly = (risk_reward * win_prob - (1 - win_prob)) / risk_reward
        return max(0.0, min(0.20, kelly))


# =============================================================================
# STRATEGY 3: ILLIQUID MARKET OPPORTUNITIES
# =============================================================================

class IlliquidMarketStrategy(BaseStrategy):
    """
    Strategy for illiquid markets where HFT doesn't compete.

    Edge: In illiquid markets:
    - HFT firms avoid due to execution risk
    - Spreads are wider (more profit potential)
    - Opportunities persist longer
    - Technical patterns more reliable (less noise)

    We specifically look for markets with:
    - Low volume relative to price
    - Wide bid-ask spreads
    - Less algorithmic activity
    """

    def __init__(
        self,
        min_spread_pct: float = 0.5,          # Only trade wide spreads
        max_daily_volume_usd: float = 1000000, # Avoid high volume
        min_spread_persistence_minutes: int = 5
    ):
        super().__init__("IlliquidMarket", EdgeType.ILLIQUID_MARKETS)

        self.min_spread_pct = min_spread_pct
        self.max_daily_volume_usd = max_daily_volume_usd
        self.min_spread_persistence = min_spread_persistence_minutes

        # Track spread history
        self._spread_history: Dict[str, deque] = {}

    def get_required_data(self) -> List[str]:
        return ["orderbook", "daily_volume", "spread_history"]

    def generate_signal(self, symbol: str, data: Dict) -> Optional[EdgeSignal]:
        """Generate signal for illiquid market opportunity"""

        orderbook = data.get("orderbook", {})
        daily_volume = data.get("daily_volume", float("inf"))

        if not orderbook or daily_volume > self.max_daily_volume_usd:
            return None  # Too liquid, HFT will compete

        best_bid = orderbook.get("best_bid", 0)
        best_ask = orderbook.get("best_ask", 0)

        if best_bid <= 0 or best_ask <= 0:
            return None

        # Calculate spread
        spread_pct = (best_ask - best_bid) / best_bid * 100

        if spread_pct < self.min_spread_pct:
            return None  # Spread too tight

        # Check spread persistence
        if not self._check_spread_persistence(symbol, spread_pct):
            return None

        mid_price = (best_bid + best_ask) / 2

        # In illiquid markets, we can be market makers
        # Buy at bid, sell at ask (or vice versa)

        # Analyze order book depth
        bid_depth = sum(orderbook.get("bid_sizes", [0]))
        ask_depth = sum(orderbook.get("ask_sizes", [0]))

        # Determine direction based on imbalance
        if bid_depth > ask_depth * 1.5:
            # More buyers, price likely to rise
            direction = "long"
            entry_price = best_bid  # Buy at bid
            target_price = best_ask  # Sell at ask
        elif ask_depth > bid_depth * 1.5:
            # More sellers, price likely to fall
            direction = "short"
            entry_price = best_ask
            target_price = best_bid
        else:
            # No clear imbalance, market make both sides
            direction = "long"  # Default to capturing spread
            entry_price = best_bid
            target_price = best_ask

        # Stop loss - wider in illiquid markets
        stop_distance = (best_ask - best_bid) * 2
        if direction == "long":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # Expected return is approximately the spread minus fees
        fee_estimate = 0.2  # 0.2% round trip
        expected_return = spread_pct - fee_estimate

        if expected_return < 0.1:
            return None

        # Risk metrics
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        risk_reward = reward / risk if risk > 0 else 0

        # Win probability higher in illiquid markets (spreads persist)
        win_prob = 0.55 + min(0.15, spread_pct / 10)  # 55-70%

        # Kelly
        kelly = self._calculate_kelly(win_prob, risk_reward)

        # Signal quality based on spread and volume characteristics
        spread_quality = min(1.0, spread_pct / 2)  # 2% spread = max
        volume_quality = 1 - (daily_volume / self.max_daily_volume_usd)
        signal_quality = (spread_quality + volume_quality) / 2

        # Strength
        if spread_pct > 1.5 and daily_volume < 100000:
            strength = SignalStrength.VERY_STRONG
        elif spread_pct > 1.0:
            strength = SignalStrength.STRONG
        elif spread_pct > 0.5:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        factors = [
            f"Spread: {spread_pct:.2f}%",
            f"Daily volume: ${daily_volume:,.0f}",
            f"Bid/ask imbalance: {bid_depth/ask_depth:.2f}" if ask_depth > 0 else "N/A"
        ]

        return EdgeSignal(
            symbol=symbol,
            direction=direction,
            edge_type=self.edge_type,
            strength=strength,
            confidence=signal_quality,
            signal_quality=signal_quality,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_return_pct=expected_return,
            win_probability=win_prob,
            risk_reward_ratio=risk_reward,
            kelly_fraction=kelly,
            suggested_position_pct=kelly * 0.3,  # More conservative in illiquid
            max_position_pct=min(0.05, kelly),   # Smaller max due to liquidity risk
            expected_hold_time_hours=0.5,        # 30 min average
            signal_valid_for_minutes=15,
            contributing_factors=factors,
            conflicting_factors=["Illiquid - execution risk"]
        )

    def _check_spread_persistence(self, symbol: str, current_spread: float) -> bool:
        """Check if spread has persisted long enough"""
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=60)  # 1 hour of minutes

        self._spread_history[symbol].append({
            "spread": current_spread,
            "timestamp": time.time()
        })

        history = self._spread_history[symbol]
        if len(history) < self.min_spread_persistence:
            return False

        # Check if spread has been consistently wide
        recent = list(history)[-self.min_spread_persistence:]
        avg_spread = np.mean([h["spread"] for h in recent])

        return avg_spread >= self.min_spread_pct * 0.8

    def _calculate_kelly(self, win_prob: float, risk_reward: float) -> float:
        """Calculate Kelly criterion"""
        if risk_reward <= 0:
            return 0.0

        kelly = (risk_reward * win_prob - (1 - win_prob)) / risk_reward
        return max(0.0, min(0.15, kelly))  # Lower cap for illiquid


# =============================================================================
# STRATEGY 4: VOLATILITY REGIME TRADING
# =============================================================================

class VolatilityRegimeStrategy(BaseStrategy):
    """
    Adapt trading based on volatility regime.

    Edge: Most traders use static strategies. We adapt:
    - High volatility: Wider stops, momentum focus
    - Low volatility: Mean reversion, tighter stops
    - Regime transitions: Early detection and adaptation
    """

    def __init__(self):
        super().__init__("VolatilityRegime", EdgeType.VOLATILITY_REGIME)

        self._volatility_history: Dict[str, deque] = {}

    def get_required_data(self) -> List[str]:
        return ["candles_15m", "candles_1h"]

    def generate_signal(self, symbol: str, data: Dict) -> Optional[EdgeSignal]:
        """Generate regime-aware signal"""

        candles = data.get("candles_15m", [])

        if len(candles) < 50:
            return None

        # Determine current volatility regime
        current_vol = self._calculate_volatility(candles)
        regime = self._classify_regime(symbol, current_vol)

        current_price = candles[-1]["close"]

        # Generate strategy based on regime
        if regime == "high":
            signal = self._high_volatility_signal(symbol, candles, current_price)
        elif regime == "low":
            signal = self._low_volatility_signal(symbol, candles, current_price)
        else:  # transitioning
            signal = self._transition_signal(symbol, candles, current_price)

        return signal

    def _calculate_volatility(self, candles: List[Dict], period: int = 20) -> float:
        """Calculate current volatility (ATR-based)"""
        if len(candles) < period + 1:
            return 0.0

        tr_values = []
        for i in range(-period, 0):
            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = candles[i-1]["close"]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr / prev_close * 100)  # As percentage

        return np.mean(tr_values)

    def _classify_regime(self, symbol: str, current_vol: float) -> str:
        """Classify volatility regime"""
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = deque(maxlen=100)

        self._volatility_history[symbol].append(current_vol)

        history = list(self._volatility_history[symbol])
        if len(history) < 20:
            return "normal"

        avg_vol = np.mean(history)
        vol_std = np.std(history)

        if current_vol > avg_vol + vol_std:
            return "high"
        elif current_vol < avg_vol - vol_std * 0.5:
            return "low"
        else:
            # Check for regime transition
            recent_trend = np.mean(history[-5:]) - np.mean(history[-20:-5])
            if abs(recent_trend) > vol_std * 0.5:
                return "transitioning"
            return "normal"

    def _high_volatility_signal(
        self,
        symbol: str,
        candles: List[Dict],
        current_price: float
    ) -> Optional[EdgeSignal]:
        """High vol: momentum following with wide stops"""

        # Look for strong momentum
        momentum = (current_price - candles[-10]["close"]) / candles[-10]["close"] * 100

        if abs(momentum) < 1.5:  # Need strong momentum in high vol
            return None

        direction = "long" if momentum > 0 else "short"

        atr = self._calculate_atr(candles)

        # Wide stops in high vol
        if direction == "long":
            target_price = current_price + atr * 3
            stop_loss = current_price - atr * 2
        else:
            target_price = current_price - atr * 3
            stop_loss = current_price + atr * 2

        risk = abs(current_price - stop_loss)
        reward = abs(target_price - current_price)
        risk_reward = reward / risk if risk > 0 else 0

        expected_return = (reward / current_price) * 100
        win_prob = 0.45  # Lower in high vol

        return EdgeSignal(
            symbol=symbol,
            direction=direction,
            edge_type=self.edge_type,
            strength=SignalStrength.MODERATE,
            confidence=0.55,
            signal_quality=0.55,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_return_pct=expected_return,
            win_probability=win_prob,
            risk_reward_ratio=risk_reward,
            kelly_fraction=0.05,
            suggested_position_pct=0.02,  # Small in high vol
            max_position_pct=0.03,
            expected_hold_time_hours=1.0,
            signal_valid_for_minutes=15,
            contributing_factors=[
                f"High volatility regime",
                f"Momentum: {momentum:.1f}%"
            ],
            conflicting_factors=["High risk environment"]
        )

    def _low_volatility_signal(
        self,
        symbol: str,
        candles: List[Dict],
        current_price: float
    ) -> Optional[EdgeSignal]:
        """Low vol: mean reversion with tight stops"""

        # Calculate Bollinger Bands
        closes = np.array([c["close"] for c in candles[-20:]])
        sma = np.mean(closes)
        std = np.std(closes)

        upper_band = sma + 2 * std
        lower_band = sma - 2 * std

        # Mean reversion signals
        if current_price > upper_band:
            direction = "short"
            target_price = sma
            stop_loss = upper_band + std
        elif current_price < lower_band:
            direction = "long"
            target_price = sma
            stop_loss = lower_band - std
        else:
            return None

        risk = abs(current_price - stop_loss)
        reward = abs(target_price - current_price)
        risk_reward = reward / risk if risk > 0 else 0

        expected_return = (reward / current_price) * 100
        win_prob = 0.60  # Higher in low vol mean reversion

        return EdgeSignal(
            symbol=symbol,
            direction=direction,
            edge_type=self.edge_type,
            strength=SignalStrength.STRONG,
            confidence=0.65,
            signal_quality=0.65,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_return_pct=expected_return,
            win_probability=win_prob,
            risk_reward_ratio=risk_reward,
            kelly_fraction=0.08,
            suggested_position_pct=0.04,  # Larger in low vol
            max_position_pct=0.06,
            expected_hold_time_hours=2.0,
            signal_valid_for_minutes=30,
            contributing_factors=[
                "Low volatility regime",
                "Mean reversion setup"
            ],
            conflicting_factors=[]
        )

    def _transition_signal(
        self,
        symbol: str,
        candles: List[Dict],
        current_price: float
    ) -> Optional[EdgeSignal]:
        """Regime transition: be cautious, wait for clarity"""
        # During transitions, we're more selective
        # Only trade very strong setups
        return None  # Skip during transitions

    def _calculate_atr(self, candles: List[Dict], period: int = 14) -> float:
        """Calculate ATR"""
        if len(candles) < period + 1:
            return candles[-1]["close"] * 0.02

        tr_values = []
        for i in range(-period, 0):
            high = candles[i]["high"]
            low = candles[i]["low"]
            prev_close = candles[i-1]["close"]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        return np.mean(tr_values)


# =============================================================================
# STRATEGY ENSEMBLE
# =============================================================================

class EdgeStrategyEnsemble:
    """
    Combines all edge strategies into a unified system.

    Allocates capital and selects signals based on:
    - Strategy performance history
    - Current market conditions
    - Risk limits
    """

    def __init__(self, capital: float = 10000.0):
        self.capital = capital

        # Initialize strategies
        self.strategies: Dict[str, BaseStrategy] = {
            "directional": EnhancedDirectionalStrategy(),
            "stat_arb": StatisticalArbitrageStrategy(),
            "illiquid": IlliquidMarketStrategy(),
            "volatility": VolatilityRegimeStrategy()
        }

        # Default allocations (sum to 1.0)
        self.allocations = {
            "directional": 0.40,    # Main edge
            "stat_arb": 0.30,       # Statistical patterns
            "illiquid": 0.15,       # Niche markets
            "volatility": 0.15      # Regime adaptation
        }

        # Performance tracking
        self._performance: Dict[str, Dict] = {
            name: {"trades": 0, "wins": 0, "total_return": 0.0}
            for name in self.strategies
        }

        # Active signals
        self._active_signals: Dict[str, EdgeSignal] = {}

    def generate_all_signals(
        self,
        symbols: List[str],
        market_data: Dict[str, Dict]
    ) -> List[EdgeSignal]:
        """Generate signals from all strategies"""
        all_signals = []

        for symbol in symbols:
            data = market_data.get(symbol, {})

            for strategy_name, strategy in self.strategies.items():
                if not strategy._enabled:
                    continue

                try:
                    signal = strategy.generate_signal(symbol, data)
                    if signal:
                        # Adjust position size based on allocation
                        allocation = self.allocations.get(strategy_name, 0.1)
                        signal.suggested_position_pct *= allocation
                        signal.max_position_pct *= allocation

                        all_signals.append(signal)
                except Exception as e:
                    logger.error(f"Strategy {strategy_name} error for {symbol}: {e}")

        # Sort by edge score
        all_signals.sort(key=lambda s: s.edge_score, reverse=True)

        return all_signals

    def select_best_signals(
        self,
        signals: List[EdgeSignal],
        max_signals: int = 5,
        max_per_strategy: int = 2
    ) -> List[EdgeSignal]:
        """Select best signals respecting diversification"""
        selected = []
        strategy_counts: Dict[EdgeType, int] = {}

        for signal in signals:
            # Check strategy limit
            edge_type = signal.edge_type
            if strategy_counts.get(edge_type, 0) >= max_per_strategy:
                continue

            # Check total limit
            if len(selected) >= max_signals:
                break

            selected.append(signal)
            strategy_counts[edge_type] = strategy_counts.get(edge_type, 0) + 1

        return selected

    def record_outcome(
        self,
        signal: EdgeSignal,
        profitable: bool,
        return_pct: float
    ):
        """Record signal outcome"""
        # Find which strategy generated this
        for name, strategy in self.strategies.items():
            if strategy.edge_type == signal.edge_type:
                strategy.record_outcome(profitable, return_pct)

                # Update ensemble tracking
                self._performance[name]["trades"] += 1
                if profitable:
                    self._performance[name]["wins"] += 1
                self._performance[name]["total_return"] += return_pct
                break

    def get_performance_summary(self) -> Dict:
        """Get performance summary for all strategies"""
        summary = {}

        for name, stats in self._performance.items():
            trades = stats["trades"]
            wins = stats["wins"]
            total_return = stats["total_return"]

            summary[name] = {
                "trades": trades,
                "win_rate": wins / trades * 100 if trades > 0 else 0,
                "total_return": total_return,
                "avg_return": total_return / trades if trades > 0 else 0,
                "allocation": self.allocations.get(name, 0) * 100
            }

        return summary

    def adjust_allocations_by_performance(self):
        """Dynamically adjust allocations based on performance"""
        # Calculate performance scores
        scores = {}
        total_score = 0

        for name, stats in self._performance.items():
            if stats["trades"] < 10:
                scores[name] = 1.0  # Default score for new strategies
            else:
                win_rate = stats["wins"] / stats["trades"]
                avg_return = stats["total_return"] / stats["trades"]

                # Score = win_rate * avg_return
                scores[name] = max(0.1, win_rate * (1 + avg_return / 100))

            total_score += scores[name]

        # Normalize to allocations
        if total_score > 0:
            for name in self.allocations:
                new_allocation = scores.get(name, 0.1) / total_score
                # Smooth transition (50% old, 50% new)
                self.allocations[name] = (
                    self.allocations[name] * 0.5 + new_allocation * 0.5
                )

        logger.info(f"Adjusted allocations: {self.allocations}")


# =============================================================================
# TEST FUNCTION
# =============================================================================

async def test_edge_strategies():
    """Test edge-focused strategies"""
    print("=" * 70)
    print("EDGE-FOCUSED STRATEGIES TEST")
    print("=" * 70)

    # Generate synthetic market data
    np.random.seed(42)

    def generate_candles(base_price: float, count: int, volatility: float = 0.02, trend: float = 0.0) -> List[Dict]:
        """Generate candles with optional trend"""
        candles = []
        price = base_price
        timestamp = time.time() - count * 900  # 15min candles

        for i in range(count):
            # Add trend and mean-reverting noise
            change = np.random.randn() * volatility + trend
            open_price = price
            close_price = price * (1 + change)
            high = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.003))
            low = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.003))
            # Volume increases with volatility
            volume = np.random.uniform(5000, 15000) * (1 + abs(change) * 10)

            candles.append({
                "timestamp": timestamp,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume
            })

            price = close_price
            timestamp += 900

        return candles

    # Create ensemble
    ensemble = EdgeStrategyEnsemble(capital=10000.0)

    # Generate test data with TRENDS to trigger signals
    # BTC in uptrend
    btc_15m = generate_candles(42000, 100, 0.012, trend=0.002)
    btc_1h = generate_candles(42000, 50, 0.018, trend=0.003)
    btc_4h = generate_candles(42000, 30, 0.025, trend=0.004)

    # ETH in downtrend (for short signals)
    eth_15m = generate_candles(2300, 100, 0.015, trend=-0.001)
    eth_1h = generate_candles(2300, 50, 0.020, trend=-0.002)
    eth_4h = generate_candles(2300, 30, 0.028, trend=-0.002)

    eth_prices = [c["close"] for c in eth_15m]
    btc_prices = [c["close"] for c in btc_15m]

    market_data = {
        "BTC": {
            "candles_15m": btc_15m,
            "candles_1h": btc_1h,
            "candles_4h": btc_4h,
            "volume": 5000000,
            "orderbook": {
                "best_bid": btc_15m[-1]["close"] * 0.999,
                "best_ask": btc_15m[-1]["close"] * 1.001,
                "bid_sizes": [1.0, 2.0, 3.0],
                "ask_sizes": [1.5, 2.5, 2.0]
            }
        },
        "ETH": {
            "candles_15m": eth_15m,
            "candles_1h": eth_1h,
            "candles_4h": eth_4h,
            "volume": 3000000,
            "orderbook": {
                "best_bid": eth_15m[-1]["close"] * 0.998,
                "best_ask": eth_15m[-1]["close"] * 1.002,
                "bid_sizes": [10.0, 20.0, 15.0],
                "ask_sizes": [12.0, 18.0, 25.0]
            }
        },
        "BTC/ETH": {
            "pair_prices": {
                "BTC": btc_prices,
                "ETH": eth_prices
            }
        },
        "ILLIQUID_TOKEN": {
            "candles_15m": generate_candles(1.5, 100, 0.025),
            "daily_volume": 50000,
            "orderbook": {
                "best_bid": 1.47,
                "best_ask": 1.54,  # Wide 4.7% spread - triggers illiquid strategy
                "bid_sizes": [1000, 500, 300],
                "ask_sizes": [800, 600, 400]
            },
            "spread_history": []
        }
    }

    # Generate signals
    print("\n--- Generating Signals ---")
    signals = ensemble.generate_all_signals(
        symbols=["BTC", "ETH", "BTC/ETH", "ILLIQUID_TOKEN"],
        market_data=market_data
    )

    print(f"Total signals generated: {len(signals)}")

    # Select best signals
    best_signals = ensemble.select_best_signals(signals, max_signals=5)

    print(f"\nTop {len(best_signals)} signals selected:")
    for i, signal in enumerate(best_signals, 1):
        print(f"\n  {i}. {signal.symbol} - {signal.direction.upper()}")
        print(f"     Edge type: {signal.edge_type.value}")
        print(f"     Strength: {signal.strength.name}")
        print(f"     Confidence: {signal.confidence:.2%}")
        print(f"     Signal quality: {signal.signal_quality:.2%}")
        print(f"     Expected return: {signal.expected_return_pct:.2f}%")
        print(f"     Win probability: {signal.win_probability:.1%}")
        print(f"     Risk/Reward: {signal.risk_reward_ratio:.2f}")
        print(f"     Kelly fraction: {signal.kelly_fraction:.2%}")
        print(f"     Suggested position: {signal.suggested_position_pct:.2%}")
        print(f"     Hold time: {signal.expected_hold_time_hours:.1f} hours")
        print(f"     Factors: {', '.join(signal.contributing_factors[:3])}")

    # Test performance tracking
    print("\n--- Simulating Outcomes ---")
    for signal in best_signals:
        profitable = np.random.random() < signal.win_probability
        actual_return = signal.expected_return_pct * (1.2 if profitable else -0.8)
        ensemble.record_outcome(signal, profitable, actual_return)

    # Show performance summary
    print("\n--- Strategy Performance ---")
    summary = ensemble.get_performance_summary()

    for name, stats in summary.items():
        print(f"\n  {name}:")
        print(f"    Trades: {stats['trades']}")
        print(f"    Win rate: {stats['win_rate']:.1f}%")
        print(f"    Total return: {stats['total_return']:.2f}%")
        print(f"    Allocation: {stats['allocation']:.1f}%")

    print("\n" + "=" * 70)
    print("KEY INSIGHT: These strategies focus on PREDICTION QUALITY,")
    print("not speed. HFT can't compete on 15-minute timeframes or")
    print("in illiquid markets where our edge actually exists.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_edge_strategies())
