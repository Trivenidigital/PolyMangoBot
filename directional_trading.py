"""
15-Minute Directional Trading Module
=====================================

Short-term directional trading strategy focusing on 15-minute price movements.
Complements the existing arbitrage strategy by adding momentum-based trades.

Features:
1. Technical indicators (RSI, MACD, EMA crossovers)
2. 15-minute candlestick pattern analysis
3. Volume-weighted momentum signals
4. Multi-timeframe confirmation
5. Risk-adjusted position sizing
"""

import asyncio
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger("PolyMangoBot.directional_trading")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class TradingSignal(Enum):
    """Trading signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class Timeframe(Enum):
    """Supported timeframes"""
    M1 = 1       # 1 minute
    M5 = 5       # 5 minutes
    M15 = 15     # 15 minutes (primary)
    M30 = 30     # 30 minutes
    H1 = 60      # 1 hour
    H4 = 240     # 4 hours


class TrendDirection(Enum):
    """Market trend direction"""
    STRONG_UP = "strong_up"
    UP = "up"
    SIDEWAYS = "sideways"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Timeframe = Timeframe.M15

    @property
    def body_size(self) -> float:
        """Size of candle body"""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        """Upper wick size"""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Lower wick size"""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish"""
        return self.close > self.open

    @property
    def range(self) -> float:
        """Total candle range"""
        return self.high - self.low


@dataclass
class TechnicalIndicators:
    """Collection of technical indicators"""
    # RSI
    rsi_14: float = 50.0
    rsi_7: float = 50.0

    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Moving Averages
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    sma_20: float = 0.0

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0

    # Momentum
    momentum_10: float = 0.0
    roc_10: float = 0.0  # Rate of Change

    # Volume
    volume_sma: float = 0.0
    volume_ratio: float = 1.0  # Current volume / average

    # ATR (Average True Range)
    atr_14: float = 0.0

    # Trend
    adx: float = 0.0  # Average Directional Index

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "rsi_14": self.rsi_14,
            "rsi_7": self.rsi_7,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "ema_9": self.ema_9,
            "ema_21": self.ema_21,
            "ema_50": self.ema_50,
            "bb_upper": self.bb_upper,
            "bb_middle": self.bb_middle,
            "bb_lower": self.bb_lower,
            "bb_width": self.bb_width,
            "momentum_10": self.momentum_10,
            "volume_ratio": self.volume_ratio,
            "atr_14": self.atr_14,
            "adx": self.adx,
            "timestamp": self.timestamp
        }


@dataclass
class DirectionalSignal:
    """Signal from directional trading strategy"""
    symbol: str
    signal: TradingSignal
    direction: str  # "long" or "short"
    confidence: float  # 0-1

    # Price targets
    entry_price: float
    target_price: float
    stop_loss: float

    # Expected metrics
    expected_return_pct: float
    risk_reward_ratio: float
    win_probability: float

    # Position sizing
    suggested_position_pct: float  # % of capital
    max_position_pct: float

    # Indicator signals
    rsi_signal: str
    macd_signal: str
    ema_signal: str
    volume_signal: str

    # Trend context
    trend_direction: TrendDirection
    trend_strength: float

    # Timing
    expected_duration_minutes: int
    signal_expiry_minutes: int = 15

    timestamp: float = field(default_factory=time.time)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "direction": self.direction,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "expected_return_pct": self.expected_return_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "win_probability": self.win_probability,
            "suggested_position_pct": self.suggested_position_pct,
            "trend_direction": self.trend_direction.value,
            "trend_strength": self.trend_strength,
            "expected_duration_minutes": self.expected_duration_minutes,
            "reasons": self.reasons,
            "timestamp": self.timestamp
        }

    @property
    def is_valid(self) -> bool:
        """Check if signal is still valid"""
        age_minutes = (time.time() - self.timestamp) / 60
        return age_minutes < self.signal_expiry_minutes


# =============================================================================
# TECHNICAL INDICATOR CALCULATOR
# =============================================================================

class TechnicalIndicatorCalculator:
    """
    Calculates technical indicators from candle data.

    Supports:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - EMA/SMA (Exponential/Simple Moving Averages)
    - Bollinger Bands
    - ATR (Average True Range)
    - ADX (Average Directional Index)
    - Volume analysis
    """

    def __init__(self):
        # EMA smoothing factors
        self._ema_multipliers = {}

    def calculate_all(self, candles: List[Candle]) -> TechnicalIndicators:
        """Calculate all indicators from candle data"""
        if len(candles) < 50:
            logger.warning(f"Insufficient candles for calculation: {len(candles)}")
            return TechnicalIndicators()

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        indicators = TechnicalIndicators()

        # RSI
        indicators.rsi_14 = self._calculate_rsi(closes, 14)
        indicators.rsi_7 = self._calculate_rsi(closes, 7)

        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(closes)
        indicators.macd_line = macd_line
        indicators.macd_signal = signal_line
        indicators.macd_histogram = histogram

        # Moving Averages
        indicators.ema_9 = self._calculate_ema(closes, 9)
        indicators.ema_21 = self._calculate_ema(closes, 21)
        indicators.ema_50 = self._calculate_ema(closes, 50)
        indicators.sma_20 = self._calculate_sma(closes, 20)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
        indicators.bb_upper = bb_upper
        indicators.bb_middle = bb_middle
        indicators.bb_lower = bb_lower
        indicators.bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0

        # Momentum
        indicators.momentum_10 = self._calculate_momentum(closes, 10)
        indicators.roc_10 = self._calculate_roc(closes, 10)

        # Volume
        indicators.volume_sma = self._calculate_sma(volumes, 20)
        current_volume = volumes[-1] if len(volumes) > 0 else 0
        indicators.volume_ratio = current_volume / indicators.volume_sma if indicators.volume_sma > 0 else 1.0

        # ATR
        indicators.atr_14 = self._calculate_atr(highs, lows, closes, 14)

        # ADX
        indicators.adx = self._calculate_adx(highs, lows, closes, 14)

        indicators.timestamp = time.time()

        return indicators

    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD"""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0

        ema_fast = self._calculate_ema_series(prices, fast)
        ema_slow = self._calculate_ema_series(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema_series(macd_line, signal)
        histogram = macd_line - signal_line

        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate current EMA value"""
        ema_series = self._calculate_ema_series(prices, period)
        return float(ema_series[-1]) if len(ema_series) > 0 else 0.0

    def _calculate_ema_series(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA series"""
        if len(prices) < period:
            return prices

        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[period - 1] = np.mean(prices[:period])

        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))

        return ema

    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """Calculate SMA"""
        if len(prices) < period:
            return float(np.mean(prices)) if len(prices) > 0 else 0.0
        return float(np.mean(prices[-period:]))

    def _calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return 0.0, 0.0, 0.0

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        return float(upper), float(sma), float(lower)

    def _calculate_momentum(self, prices: np.ndarray, period: int) -> float:
        """Calculate momentum"""
        if len(prices) < period + 1:
            return 0.0
        return float(prices[-1] - prices[-period - 1])

    def _calculate_roc(self, prices: np.ndarray, period: int) -> float:
        """Calculate Rate of Change"""
        if len(prices) < period + 1:
            return 0.0

        prev_price = prices[-period - 1]
        if prev_price == 0:
            return 0.0

        return float((prices[-1] - prev_price) / prev_price * 100)

    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int
    ) -> float:
        """Calculate Average True Range"""
        if len(closes) < period + 1:
            return 0.0

        tr = np.zeros(len(closes) - 1)

        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i-1] = max(hl, hc, lc)

        return float(np.mean(tr[-period:]))

    def _calculate_adx(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int
    ) -> float:
        """Calculate Average Directional Index"""
        if len(closes) < period * 2:
            return 25.0  # Default neutral

        # Calculate +DM and -DM
        plus_dm = np.zeros(len(highs) - 1)
        minus_dm = np.zeros(len(highs) - 1)

        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i-1] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i-1] = down_move

        # Calculate TR
        tr = np.zeros(len(closes) - 1)
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i-1] = max(hl, hc, lc)

        # Smooth
        atr = np.mean(tr[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0

        # Calculate DX and ADX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        dx = 100 * abs(plus_di - minus_di) / di_sum

        return float(dx)


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class DirectionalSignalGenerator:
    """
    Generates trading signals based on technical analysis.

    Signal Generation Logic:
    1. Trend identification using multiple timeframes
    2. Entry signals from indicator combinations
    3. Confirmation from volume analysis
    4. Risk-adjusted position sizing
    """

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        min_confidence: float = 0.6,
        default_target_pct: float = 1.0,
        default_stop_pct: float = 0.5
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_confidence = min_confidence
        self.default_target_pct = default_target_pct
        self.default_stop_pct = default_stop_pct

        self.indicator_calculator = TechnicalIndicatorCalculator()

    def generate_signal(
        self,
        symbol: str,
        candles: List[Candle],
        current_price: float
    ) -> Optional[DirectionalSignal]:
        """Generate trading signal from candle data"""
        if len(candles) < 50:
            return None

        # Calculate indicators
        indicators = self.indicator_calculator.calculate_all(candles)

        # Analyze individual indicator signals
        rsi_signal, rsi_strength = self._analyze_rsi(indicators)
        macd_signal, macd_strength = self._analyze_macd(indicators)
        ema_signal, ema_strength = self._analyze_ema(indicators, current_price)
        volume_signal, volume_strength = self._analyze_volume(indicators)
        bb_signal, bb_strength = self._analyze_bollinger(indicators, current_price)

        # Determine trend
        trend_direction, trend_strength = self._determine_trend(
            indicators, candles, current_price
        )

        # Combine signals
        combined_signal, confidence = self._combine_signals(
            rsi_signal, rsi_strength,
            macd_signal, macd_strength,
            ema_signal, ema_strength,
            volume_signal, volume_strength,
            bb_signal, bb_strength,
            trend_direction, trend_strength
        )

        if confidence < self.min_confidence:
            return None

        # Determine direction
        direction = "long" if combined_signal in [TradingSignal.STRONG_BUY, TradingSignal.BUY] else "short"

        # Calculate targets
        atr = indicators.atr_14 if indicators.atr_14 > 0 else current_price * 0.01

        if direction == "long":
            target_price = current_price + (atr * 2)
            stop_loss = current_price - atr
        else:
            target_price = current_price - (atr * 2)
            stop_loss = current_price + atr

        # Calculate expected metrics
        risk = abs(current_price - stop_loss)
        reward = abs(target_price - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        expected_return_pct = (reward / current_price) * 100

        # Estimate win probability based on confidence and trend alignment
        win_probability = min(0.75, confidence * 0.8 + trend_strength * 0.15)

        # Position sizing
        suggested_position = self._calculate_position_size(
            confidence, risk_reward_ratio, win_probability, indicators.atr_14, current_price
        )

        # Build reasons list
        reasons = self._build_reasons(
            rsi_signal, macd_signal, ema_signal, volume_signal, bb_signal,
            trend_direction, indicators
        )

        return DirectionalSignal(
            symbol=symbol,
            signal=combined_signal,
            direction=direction,
            confidence=confidence,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            expected_return_pct=expected_return_pct,
            risk_reward_ratio=risk_reward_ratio,
            win_probability=win_probability,
            suggested_position_pct=suggested_position,
            max_position_pct=suggested_position * 1.5,
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            ema_signal=ema_signal,
            volume_signal=volume_signal,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            expected_duration_minutes=15,
            reasons=reasons
        )

    def _analyze_rsi(self, indicators: TechnicalIndicators) -> Tuple[str, float]:
        """Analyze RSI for signals"""
        rsi = indicators.rsi_14

        if rsi < 20:
            return "strong_buy", 0.9
        elif rsi < self.rsi_oversold:
            return "buy", 0.7
        elif rsi > 80:
            return "strong_sell", 0.9
        elif rsi > self.rsi_overbought:
            return "sell", 0.7
        elif 45 <= rsi <= 55:
            return "neutral", 0.3
        elif rsi < 45:
            return "slight_buy", 0.4
        else:
            return "slight_sell", 0.4

    def _analyze_macd(self, indicators: TechnicalIndicators) -> Tuple[str, float]:
        """Analyze MACD for signals"""
        histogram = indicators.macd_histogram
        macd = indicators.macd_line
        signal = indicators.macd_signal

        # Crossover detection
        if macd > signal and histogram > 0:
            strength = min(1.0, abs(histogram) * 100)
            if histogram > 0.001:  # Strong bullish
                return "strong_buy", strength
            return "buy", strength * 0.8
        elif macd < signal and histogram < 0:
            strength = min(1.0, abs(histogram) * 100)
            if histogram < -0.001:  # Strong bearish
                return "strong_sell", strength
            return "sell", strength * 0.8

        return "neutral", 0.3

    def _analyze_ema(
        self,
        indicators: TechnicalIndicators,
        current_price: float
    ) -> Tuple[str, float]:
        """Analyze EMA crossovers and price position"""
        ema_9 = indicators.ema_9
        ema_21 = indicators.ema_21
        ema_50 = indicators.ema_50

        # Price above/below EMAs
        above_9 = current_price > ema_9
        above_21 = current_price > ema_21
        above_50 = current_price > ema_50

        # EMA alignment
        bullish_alignment = ema_9 > ema_21 > ema_50
        bearish_alignment = ema_9 < ema_21 < ema_50

        if above_9 and above_21 and above_50 and bullish_alignment:
            return "strong_buy", 0.85
        elif above_9 and above_21:
            return "buy", 0.65
        elif not above_9 and not above_21 and not above_50 and bearish_alignment:
            return "strong_sell", 0.85
        elif not above_9 and not above_21:
            return "sell", 0.65

        return "neutral", 0.3

    def _analyze_volume(self, indicators: TechnicalIndicators) -> Tuple[str, float]:
        """Analyze volume for confirmation"""
        volume_ratio = indicators.volume_ratio

        if volume_ratio > 2.0:
            return "high", 0.9
        elif volume_ratio > 1.5:
            return "elevated", 0.7
        elif volume_ratio > 1.0:
            return "normal", 0.5
        elif volume_ratio > 0.5:
            return "low", 0.3
        else:
            return "very_low", 0.2

    def _analyze_bollinger(
        self,
        indicators: TechnicalIndicators,
        current_price: float
    ) -> Tuple[str, float]:
        """Analyze Bollinger Bands"""
        if indicators.bb_upper == 0:
            return "neutral", 0.3

        bb_position = (current_price - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)

        if bb_position < 0.1:
            return "oversold", 0.8
        elif bb_position < 0.2:
            return "low", 0.6
        elif bb_position > 0.9:
            return "overbought", 0.8
        elif bb_position > 0.8:
            return "high", 0.6

        return "neutral", 0.4

    def _determine_trend(
        self,
        indicators: TechnicalIndicators,
        candles: List[Candle],
        current_price: float
    ) -> Tuple[TrendDirection, float]:
        """Determine overall trend direction and strength"""
        # ADX for trend strength
        adx = indicators.adx

        # Price vs EMAs
        ema_21 = indicators.ema_21
        ema_50 = indicators.ema_50

        # Recent price action (last 10 candles)
        recent_closes = [c.close for c in candles[-10:]]
        price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100 if recent_closes[0] > 0 else 0

        # Determine direction
        if adx > 40 and current_price > ema_21 > ema_50 and price_change > 1:
            return TrendDirection.STRONG_UP, min(1.0, adx / 50)
        elif current_price > ema_21 and price_change > 0.5:
            return TrendDirection.UP, 0.6
        elif adx > 40 and current_price < ema_21 < ema_50 and price_change < -1:
            return TrendDirection.STRONG_DOWN, min(1.0, adx / 50)
        elif current_price < ema_21 and price_change < -0.5:
            return TrendDirection.DOWN, 0.6

        return TrendDirection.SIDEWAYS, 0.3

    def _combine_signals(
        self,
        rsi_signal: str, rsi_strength: float,
        macd_signal: str, macd_strength: float,
        ema_signal: str, ema_strength: float,
        volume_signal: str, volume_strength: float,
        bb_signal: str, bb_strength: float,
        trend: TrendDirection, trend_strength: float
    ) -> Tuple[TradingSignal, float]:
        """Combine all signals into final decision"""
        # Score buy and sell signals
        buy_score = 0.0
        sell_score = 0.0

        # RSI contribution (weight: 0.2)
        if "buy" in rsi_signal:
            buy_score += rsi_strength * 0.2
        elif "sell" in rsi_signal:
            sell_score += rsi_strength * 0.2

        # MACD contribution (weight: 0.25)
        if "buy" in macd_signal:
            buy_score += macd_strength * 0.25
        elif "sell" in macd_signal:
            sell_score += macd_strength * 0.25

        # EMA contribution (weight: 0.25)
        if "buy" in ema_signal:
            buy_score += ema_strength * 0.25
        elif "sell" in ema_signal:
            sell_score += ema_strength * 0.25

        # Bollinger contribution (weight: 0.15)
        if bb_signal == "oversold":
            buy_score += bb_strength * 0.15
        elif bb_signal == "overbought":
            sell_score += bb_strength * 0.15

        # Trend alignment (weight: 0.15)
        if trend in [TrendDirection.UP, TrendDirection.STRONG_UP]:
            buy_score += trend_strength * 0.15
        elif trend in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
            sell_score += trend_strength * 0.15

        # Volume confirmation boost
        if volume_signal in ["high", "elevated"]:
            volume_boost = 1.1
        else:
            volume_boost = 1.0

        buy_score *= volume_boost
        sell_score *= volume_boost

        # Determine final signal
        if buy_score > sell_score + 0.1:
            confidence = min(0.95, buy_score)
            if buy_score > 0.7:
                return TradingSignal.STRONG_BUY, confidence
            return TradingSignal.BUY, confidence
        elif sell_score > buy_score + 0.1:
            confidence = min(0.95, sell_score)
            if sell_score > 0.7:
                return TradingSignal.STRONG_SELL, confidence
            return TradingSignal.SELL, confidence

        return TradingSignal.NEUTRAL, 0.3

    def _calculate_position_size(
        self,
        confidence: float,
        risk_reward: float,
        win_prob: float,
        atr: float,
        price: float
    ) -> float:
        """Calculate suggested position size as % of capital"""
        # Base position
        base_position = 0.02  # 2% default

        # Adjust for confidence
        confidence_multiplier = confidence / 0.6  # 1.0 at 60% confidence

        # Adjust for risk/reward
        rr_multiplier = min(1.5, risk_reward / 2)  # 1.0 at 2:1 R:R

        # Adjust for win probability
        wp_multiplier = win_prob / 0.5  # 1.0 at 50% win rate

        # Volatility adjustment (lower size for high volatility)
        vol_ratio = atr / price if price > 0 else 0.01
        vol_multiplier = max(0.5, 1 - vol_ratio * 10)

        position = base_position * confidence_multiplier * rr_multiplier * wp_multiplier * vol_multiplier

        # Cap at 10%
        return min(0.10, max(0.005, position))

    def _build_reasons(
        self,
        rsi_signal: str,
        macd_signal: str,
        ema_signal: str,
        volume_signal: str,
        bb_signal: str,
        trend: TrendDirection,
        indicators: TechnicalIndicators
    ) -> List[str]:
        """Build human-readable reasons for the signal"""
        reasons = []

        if "buy" in rsi_signal:
            reasons.append(f"RSI at {indicators.rsi_14:.1f} indicates oversold conditions")
        elif "sell" in rsi_signal:
            reasons.append(f"RSI at {indicators.rsi_14:.1f} indicates overbought conditions")

        if "buy" in macd_signal:
            reasons.append(f"MACD bullish crossover (histogram: {indicators.macd_histogram:.4f})")
        elif "sell" in macd_signal:
            reasons.append(f"MACD bearish crossover (histogram: {indicators.macd_histogram:.4f})")

        if "buy" in ema_signal:
            reasons.append("Price above key EMAs with bullish alignment")
        elif "sell" in ema_signal:
            reasons.append("Price below key EMAs with bearish alignment")

        if bb_signal == "oversold":
            reasons.append("Price near lower Bollinger Band - potential bounce")
        elif bb_signal == "overbought":
            reasons.append("Price near upper Bollinger Band - potential pullback")

        if trend in [TrendDirection.STRONG_UP, TrendDirection.UP]:
            reasons.append(f"Overall trend is bullish (ADX: {indicators.adx:.1f})")
        elif trend in [TrendDirection.STRONG_DOWN, TrendDirection.DOWN]:
            reasons.append(f"Overall trend is bearish (ADX: {indicators.adx:.1f})")

        if volume_signal in ["high", "elevated"]:
            reasons.append(f"Volume confirmation ({indicators.volume_ratio:.1f}x average)")

        return reasons


# =============================================================================
# CANDLE MANAGER
# =============================================================================

class CandleManager:
    """
    Manages candle data for multiple symbols and timeframes.

    Features:
    - Real-time candle building from tick data
    - Historical candle storage
    - Multi-timeframe support
    """

    def __init__(self, max_candles: int = 500):
        self.max_candles = max_candles

        # Storage: {symbol: {timeframe: deque of candles}}
        self._candles: Dict[str, Dict[Timeframe, deque]] = {}

        # Current building candle: {symbol: {timeframe: Candle}}
        self._current_candles: Dict[str, Dict[Timeframe, Candle]] = {}

    def add_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: Optional[float] = None
    ):
        """Add tick data and update candles"""
        timestamp = timestamp or time.time()

        # Initialize if needed
        if symbol not in self._candles:
            self._candles[symbol] = {}
            self._current_candles[symbol] = {}

        # Update each timeframe
        for timeframe in [Timeframe.M1, Timeframe.M5, Timeframe.M15]:
            self._update_candle(symbol, timeframe, price, volume, timestamp)

    def _update_candle(
        self,
        symbol: str,
        timeframe: Timeframe,
        price: float,
        volume: float,
        timestamp: float
    ):
        """Update candle for specific timeframe"""
        if timeframe not in self._candles[symbol]:
            self._candles[symbol][timeframe] = deque(maxlen=self.max_candles)

        # Calculate candle period
        period_seconds = timeframe.value * 60
        candle_start = (int(timestamp) // period_seconds) * period_seconds

        current = self._current_candles[symbol].get(timeframe)

        if current is None or current.timestamp != candle_start:
            # New candle period
            if current is not None:
                # Save completed candle
                self._candles[symbol][timeframe].append(current)

            # Start new candle
            self._current_candles[symbol][timeframe] = Candle(
                timestamp=candle_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                timeframe=timeframe
            )
        else:
            # Update current candle
            current.high = max(current.high, price)
            current.low = min(current.low, price)
            current.close = price
            current.volume += volume

    def get_candles(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.M15,
        count: Optional[int] = None
    ) -> List[Candle]:
        """Get historical candles for symbol"""
        if symbol not in self._candles or timeframe not in self._candles[symbol]:
            return []

        candles = list(self._candles[symbol][timeframe])

        # Add current building candle
        if symbol in self._current_candles and timeframe in self._current_candles[symbol]:
            candles.append(self._current_candles[symbol][timeframe])

        if count:
            candles = candles[-count:]

        return candles

    def add_historical_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        candles: List[Candle]
    ):
        """Add historical candles (e.g., from API)"""
        if symbol not in self._candles:
            self._candles[symbol] = {}

        if timeframe not in self._candles[symbol]:
            self._candles[symbol][timeframe] = deque(maxlen=self.max_candles)

        for candle in candles:
            self._candles[symbol][timeframe].append(candle)


# =============================================================================
# DIRECTIONAL TRADING ENGINE
# =============================================================================

class DirectionalTradingEngine:
    """
    Main engine for 15-minute directional trading.

    Integrates:
    - Candle management
    - Signal generation
    - Trade tracking
    - Performance metrics
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_concurrent_signals: int = 5
    ):
        self.min_confidence = min_confidence
        self.max_concurrent_signals = max_concurrent_signals

        self.candle_manager = CandleManager()
        self.signal_generator = DirectionalSignalGenerator(min_confidence=min_confidence)

        # Active signals
        self._active_signals: Dict[str, DirectionalSignal] = {}

        # Signal history
        self._signal_history: deque = deque(maxlen=1000)

        # Performance tracking
        self._performance = {
            "total_signals": 0,
            "winning_signals": 0,
            "total_return_pct": 0.0,
            "avg_confidence": 0.0
        }

    def update_price(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
        timestamp: Optional[float] = None
    ):
        """Update with new price tick"""
        self.candle_manager.add_tick(symbol, price, volume, timestamp)

        # Check if active signal hit targets
        if symbol in self._active_signals:
            signal = self._active_signals[symbol]
            self._check_signal_outcome(signal, price)

    def generate_signals(self, symbols: List[str]) -> List[DirectionalSignal]:
        """Generate signals for all symbols"""
        signals = []

        for symbol in symbols:
            candles = self.candle_manager.get_candles(symbol, Timeframe.M15)

            if len(candles) < 50:
                continue

            current_price = candles[-1].close

            signal = self.signal_generator.generate_signal(
                symbol, candles, current_price
            )

            if signal and signal.confidence >= self.min_confidence:
                # Check if we already have an active signal for this symbol
                if symbol not in self._active_signals:
                    signals.append(signal)
                    self._active_signals[symbol] = signal
                    self._performance["total_signals"] += 1

                    logger.info(
                        f"New {signal.signal.value} signal for {symbol}: "
                        f"confidence={signal.confidence:.2%}, "
                        f"direction={signal.direction}"
                    )

        # Limit concurrent signals
        if len(signals) > self.max_concurrent_signals:
            signals = sorted(signals, key=lambda s: s.confidence, reverse=True)
            signals = signals[:self.max_concurrent_signals]

        return signals

    def _check_signal_outcome(self, signal: DirectionalSignal, current_price: float):
        """Check if signal hit target or stop loss"""
        if not signal.is_valid:
            # Signal expired
            self._complete_signal(signal, "expired", current_price)
            return

        if signal.direction == "long":
            if current_price >= signal.target_price:
                self._complete_signal(signal, "target_hit", current_price)
            elif current_price <= signal.stop_loss:
                self._complete_signal(signal, "stop_hit", current_price)
        else:  # short
            if current_price <= signal.target_price:
                self._complete_signal(signal, "target_hit", current_price)
            elif current_price >= signal.stop_loss:
                self._complete_signal(signal, "stop_hit", current_price)

    def _complete_signal(
        self,
        signal: DirectionalSignal,
        outcome: str,
        exit_price: float
    ):
        """Complete and record signal outcome"""
        symbol = signal.symbol

        # Calculate actual return
        if signal.direction == "long":
            actual_return = (exit_price - signal.entry_price) / signal.entry_price * 100
        else:
            actual_return = (signal.entry_price - exit_price) / signal.entry_price * 100

        # Update performance
        if outcome == "target_hit" or actual_return > 0:
            self._performance["winning_signals"] += 1

        self._performance["total_return_pct"] += actual_return

        # Update average confidence
        n = self._performance["total_signals"]
        self._performance["avg_confidence"] = (
            (self._performance["avg_confidence"] * (n - 1) + signal.confidence) / n
            if n > 0 else signal.confidence
        )

        # Record to history
        self._signal_history.append({
            "signal": signal.to_dict(),
            "outcome": outcome,
            "exit_price": exit_price,
            "actual_return_pct": actual_return,
            "timestamp": time.time()
        })

        # Remove from active
        if symbol in self._active_signals:
            del self._active_signals[symbol]

        logger.info(
            f"Signal completed for {symbol}: {outcome}, "
            f"return={actual_return:.2f}%"
        )

    def get_active_signals(self) -> Dict[str, DirectionalSignal]:
        """Get currently active signals"""
        # Clean expired signals
        expired = [
            symbol for symbol, signal in self._active_signals.items()
            if not signal.is_valid
        ]
        for symbol in expired:
            del self._active_signals[symbol]

        return self._active_signals.copy()

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        total = self._performance["total_signals"]
        wins = self._performance["winning_signals"]

        return {
            "total_signals": total,
            "winning_signals": wins,
            "win_rate": wins / total * 100 if total > 0 else 0,
            "total_return_pct": self._performance["total_return_pct"],
            "avg_return_per_signal": self._performance["total_return_pct"] / total if total > 0 else 0,
            "avg_confidence": self._performance["avg_confidence"],
            "active_signals": len(self._active_signals)
        }

    def load_historical_candles(
        self,
        symbol: str,
        candles: List[Dict],
        timeframe: Timeframe = Timeframe.M15
    ):
        """Load historical candle data"""
        candle_objects = [
            Candle(
                timestamp=c["timestamp"],
                open=c["open"],
                high=c["high"],
                low=c["low"],
                close=c["close"],
                volume=c.get("volume", 0),
                timeframe=timeframe
            )
            for c in candles
        ]

        self.candle_manager.add_historical_candles(symbol, timeframe, candle_objects)


# =============================================================================
# TEST FUNCTION
# =============================================================================

async def test_directional_trading():
    """Test the directional trading module"""
    print("=" * 70)
    print("DIRECTIONAL TRADING MODULE TEST")
    print("=" * 70)

    engine = DirectionalTradingEngine(min_confidence=0.5)

    # Generate synthetic price data
    print("\nGenerating synthetic 15-minute candle data...")

    np.random.seed(42)
    base_price = 100.0

    # Generate 200 candles (about 2 days of 15-min data)
    candles = []
    timestamp = time.time() - (200 * 15 * 60)  # Start 200 candles ago

    for i in range(200):
        # Random walk with trend
        trend = 0.0001 * np.sin(i / 20)  # Slight trend
        change = np.random.randn() * 0.005 + trend

        open_price = base_price
        close_price = base_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.002))
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.002))
        volume = np.random.uniform(1000, 5000)

        candles.append({
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        })

        base_price = close_price
        timestamp += 15 * 60

    # Load candles
    engine.load_historical_candles("BTC", candles, Timeframe.M15)
    print(f"Loaded {len(candles)} candles for BTC")

    # Generate signals
    print("\n" + "-" * 50)
    print("GENERATING SIGNALS")
    print("-" * 50)

    signals = engine.generate_signals(["BTC"])

    if signals:
        for signal in signals:
            print(f"\nSignal: {signal.signal.value}")
            print(f"  Direction: {signal.direction}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Entry: ${signal.entry_price:.2f}")
            print(f"  Target: ${signal.target_price:.2f}")
            print(f"  Stop: ${signal.stop_loss:.2f}")
            print(f"  R:R Ratio: {signal.risk_reward_ratio:.2f}")
            print(f"  Position Size: {signal.suggested_position_pct:.2%}")
            print(f"  Reasons:")
            for reason in signal.reasons:
                print(f"    - {reason}")
    else:
        print("No signals generated (confidence too low)")

    # Show performance
    print("\n" + "-" * 50)
    print("PERFORMANCE STATS")
    print("-" * 50)

    stats = engine.get_performance_stats()
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Win rate: {stats['win_rate']:.1f}%")
    print(f"  Active signals: {stats['active_signals']}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_directional_trading())
