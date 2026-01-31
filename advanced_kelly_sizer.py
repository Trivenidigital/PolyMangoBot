"""
Advanced Kelly Criterion Position Sizer
Sophisticated position sizing with drawdown protection and regime detection
"""

import math
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger("PolyMangoBot.kelly")


class MarketRegime(Enum):
    """Current market regime"""
    BULL = "bull"           # Trending up, high win rate
    BEAR = "bear"           # Trending down, low win rate
    SIDEWAYS = "sideways"   # Range-bound
    VOLATILE = "volatile"   # High volatility, uncertain


class DrawdownState(Enum):
    """Current drawdown state"""
    NORMAL = "normal"           # Within acceptable drawdown
    CAUTION = "caution"         # Approaching max drawdown
    CRITICAL = "critical"       # At or near max drawdown
    RECOVERY = "recovery"       # Recovering from drawdown


@dataclass
class TradeRecord:
    """Record of a single trade"""
    timestamp: float
    profit_loss: float
    profit_loss_pct: float
    position_size: float
    win: bool
    market: str
    strategy: str = "arbitrage"

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "profit_loss": self.profit_loss,
            "profit_loss_pct": self.profit_loss_pct,
            "position_size": self.position_size,
            "win": self.win,
            "market": self.market
        }


@dataclass
class KellyRecommendation:
    """Position sizing recommendation"""
    # Core Kelly calculation
    raw_kelly_fraction: float = 0.0    # Full Kelly
    adjusted_kelly_fraction: float = 0.0  # After adjustments

    # Position sizing
    recommended_position_pct: float = 0.0   # % of capital
    recommended_position_usd: float = 0.0   # Absolute amount
    max_position_usd: float = 0.0           # Hard cap

    # Risk metrics
    expected_return_pct: float = 0.0
    expected_risk_pct: float = 0.0
    risk_reward_ratio: float = 0.0

    # Adjustments applied
    drawdown_adjustment: float = 1.0    # Multiplier for drawdown state
    volatility_adjustment: float = 1.0  # Multiplier for market volatility
    correlation_adjustment: float = 1.0 # Multiplier for correlation risk
    regime_adjustment: float = 1.0      # Multiplier for market regime

    # Confidence
    confidence: float = 0.0
    data_quality: float = 0.0
    sample_size: int = 0

    # State info
    drawdown_state: DrawdownState = DrawdownState.NORMAL
    market_regime: MarketRegime = MarketRegime.SIDEWAYS

    def to_dict(self) -> Dict:
        return {
            "raw_kelly_fraction": self.raw_kelly_fraction,
            "adjusted_kelly_fraction": self.adjusted_kelly_fraction,
            "recommended_position_pct": self.recommended_position_pct,
            "recommended_position_usd": self.recommended_position_usd,
            "confidence": self.confidence,
            "drawdown_state": self.drawdown_state.value,
            "market_regime": self.market_regime.value
        }


class AdvancedKellySizer:
    """
    Advanced Kelly Criterion position sizer with sophisticated adjustments.

    Features:
    - Dynamic Kelly calculation from trade history
    - Drawdown-based position reduction
    - Market regime detection
    - Volatility adjustment
    - Correlation risk management
    - Recovery mode after losses
    """

    def __init__(
        self,
        capital: float = 10000.0,
        max_position_pct: float = 10.0,     # Max 10% per trade
        kelly_fraction: float = 0.5,         # Use half-Kelly by default
        max_drawdown_pct: float = 20.0,      # Stop trading at 20% drawdown
        history_size: int = 1000
    ):
        self.capital = capital
        self.initial_capital = capital
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.max_drawdown_pct = max_drawdown_pct
        self.history_size = history_size

        # Trade history
        self.trade_history: deque = deque(maxlen=history_size)

        # Performance tracking
        self.peak_capital = capital
        self.current_drawdown_pct = 0.0
        self.max_realized_drawdown = 0.0

        # Rolling statistics
        self.rolling_win_rate: deque = deque(maxlen=100)
        self.rolling_profit_factor: deque = deque(maxlen=100)
        self.rolling_returns: deque = deque(maxlen=100)

        # Regime detection
        self.price_history: Dict[str, deque] = {}
        self.volatility_history: deque = deque(maxlen=50)

        # State
        self.is_trading_enabled = True
        self.drawdown_state = DrawdownState.NORMAL
        self.market_regime = MarketRegime.SIDEWAYS

    def add_trade(self, trade: TradeRecord):
        """Add a trade record and update statistics"""
        self.trade_history.append(trade)

        # Update capital
        self.capital += trade.profit_loss

        # Update peak and drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
            self.current_drawdown_pct = 0.0
        else:
            self.current_drawdown_pct = ((self.peak_capital - self.capital) / self.peak_capital) * 100
            self.max_realized_drawdown = max(self.max_realized_drawdown, self.current_drawdown_pct)

        # Update rolling statistics
        self.rolling_win_rate.append(1 if trade.win else 0)
        self.rolling_returns.append(trade.profit_loss_pct)

        # Update drawdown state
        self._update_drawdown_state()

        logger.debug(f"Trade recorded: {trade.profit_loss:.2f} | Drawdown: {self.current_drawdown_pct:.1f}%")

    def _update_drawdown_state(self):
        """Update drawdown state based on current metrics"""
        dd = self.current_drawdown_pct
        max_dd = self.max_drawdown_pct

        if dd >= max_dd:
            self.drawdown_state = DrawdownState.CRITICAL
            self.is_trading_enabled = False
            logger.warning(f"CRITICAL drawdown reached: {dd:.1f}%. Trading disabled.")
        elif dd >= max_dd * 0.75:
            self.drawdown_state = DrawdownState.CAUTION
        elif dd < max_dd * 0.5 and self.drawdown_state == DrawdownState.CRITICAL:
            self.drawdown_state = DrawdownState.RECOVERY
            self.is_trading_enabled = True
            logger.info("Recovered from critical drawdown. Trading re-enabled.")
        elif dd < max_dd * 0.25:
            self.drawdown_state = DrawdownState.NORMAL

    def get_statistics(self) -> Dict:
        """Get comprehensive trading statistics"""
        if len(self.trade_history) < 2:
            return {
                "total_trades": len(self.trade_history),
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "expectancy": 0
            }

        trades = list(self.trade_history)
        wins = [t for t in trades if t.win]
        losses = [t for t in trades if not t.win]

        total_profit = sum(t.profit_loss for t in wins)
        total_loss = abs(sum(t.profit_loss for t in losses))

        avg_win = total_profit / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0

        win_rate = len(wins) / len(trades)
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Expectancy (average profit per trade)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Sharpe-like ratio
        returns = [t.profit_loss_pct for t in trades]
        if len(returns) > 1:
            sharpe = statistics.mean(returns) / statistics.stdev(returns) if statistics.stdev(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_drawdown_pct": self.max_realized_drawdown,
            "capital": self.capital,
            "total_return_pct": ((self.capital - self.initial_capital) / self.initial_capital) * 100
        }

    def calculate_raw_kelly(self) -> Tuple[float, float]:
        """
        Calculate raw Kelly fraction from trade statistics.

        Kelly formula: f* = (bp - q) / b

        Where:
        - f* = fraction of capital to risk
        - b = odds (avg_win / avg_loss)
        - p = probability of winning
        - q = probability of losing (1 - p)

        Returns:
            (kelly_fraction, confidence)
        """
        stats = self.get_statistics()

        if stats["total_trades"] < 10:
            return 0.02, 0.1  # Default conservative

        win_rate = stats["win_rate"]
        avg_win = stats["avg_win"]
        avg_loss = stats["avg_loss"]

        if avg_loss == 0:
            return 0.02, 0.1

        # Calculate odds
        b = avg_win / avg_loss if avg_loss > 0 else 1

        # Kelly formula
        p = win_rate
        q = 1 - win_rate

        kelly = ((b * p) - q) / b if b > 0 else 0

        # Confidence based on sample size
        confidence = min(stats["total_trades"] / 50, 1.0)

        # Clamp to reasonable range
        kelly = max(0, min(kelly, 0.25))  # Max 25% even full Kelly

        return kelly, confidence

    def detect_market_regime(self, symbol: str) -> MarketRegime:
        """Detect current market regime from price history"""
        key = symbol
        if key not in self.price_history or len(self.price_history[key]) < 20:
            return MarketRegime.SIDEWAYS

        prices = list(self.price_history[key])

        # Calculate trend
        recent_prices = prices[-20:]
        old_prices = prices[-40:-20] if len(prices) >= 40 else prices[:20]

        recent_avg = statistics.mean(recent_prices)
        old_avg = statistics.mean(old_prices)

        trend = (recent_avg - old_avg) / old_avg

        # Calculate volatility
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0

        # Classify regime
        if volatility > 0.02:  # High volatility
            return MarketRegime.VOLATILE
        elif trend > 0.02:  # Strong uptrend
            return MarketRegime.BULL
        elif trend < -0.02:  # Strong downtrend
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def calculate_drawdown_adjustment(self) -> float:
        """Calculate position size adjustment based on drawdown"""
        dd = self.current_drawdown_pct
        max_dd = self.max_drawdown_pct

        if self.drawdown_state == DrawdownState.CRITICAL:
            return 0.0  # No trading

        elif self.drawdown_state == DrawdownState.CAUTION:
            # Linear reduction from 50% to 0%
            reduction = 0.5 * (1 - (dd - max_dd * 0.5) / (max_dd * 0.5))
            return max(0.1, reduction)

        elif self.drawdown_state == DrawdownState.RECOVERY:
            # Gradual increase during recovery
            recovery_pct = 1 - (dd / (max_dd * 0.75))
            return 0.3 + (0.7 * recovery_pct)

        else:  # NORMAL
            return 1.0

    def calculate_volatility_adjustment(self) -> float:
        """Calculate position size adjustment based on market volatility"""
        if len(self.volatility_history) < 5:
            return 1.0

        current_vol = self.volatility_history[-1] if self.volatility_history else 0
        avg_vol = statistics.mean(self.volatility_history)

        if avg_vol == 0:
            return 1.0

        vol_ratio = current_vol / avg_vol

        # Reduce size when volatility is high
        if vol_ratio > 2.0:
            return 0.5
        elif vol_ratio > 1.5:
            return 0.7
        elif vol_ratio < 0.5:
            return 1.2  # Slight increase in low vol
        else:
            return 1.0

    def calculate_regime_adjustment(self) -> float:
        """Calculate position size adjustment based on market regime"""
        regime = self.market_regime

        adjustments = {
            MarketRegime.BULL: 1.1,      # Slightly larger in bull market
            MarketRegime.BEAR: 0.7,      # Smaller in bear market
            MarketRegime.SIDEWAYS: 1.0,  # Normal in sideways
            MarketRegime.VOLATILE: 0.6,  # Smaller in volatile
        }

        return adjustments.get(regime, 1.0)

    def calculate_streak_adjustment(self) -> float:
        """Calculate adjustment based on win/loss streaks"""
        if len(self.trade_history) < 5:
            return 1.0

        # Check for streak
        recent = list(self.trade_history)[-10:]
        streak = 0
        streak_type = None

        for trade in reversed(recent):
            if streak_type is None:
                streak_type = trade.win
                streak = 1
            elif trade.win == streak_type:
                streak += 1
            else:
                break

        # Adjust based on streak
        if streak_type:  # Winning streak
            if streak >= 5:
                return 0.8  # Reduce after long win streak (mean reversion)
            else:
                return 1.0
        else:  # Losing streak
            if streak >= 5:
                return 0.3  # Significant reduction after losses
            elif streak >= 3:
                return 0.5
            else:
                return 0.8

    def get_position_recommendation(
        self,
        opportunity_edge_pct: float = 0.5,
        opportunity_confidence: float = 0.7,
        market: str = "BTC"
    ) -> KellyRecommendation:
        """
        Get comprehensive position sizing recommendation.

        Args:
            opportunity_edge_pct: Expected edge of the opportunity
            opportunity_confidence: Confidence in the opportunity
            market: Market symbol for regime detection
        """
        rec = KellyRecommendation()

        # Check if trading is enabled
        if not self.is_trading_enabled:
            rec.recommended_position_pct = 0
            rec.recommended_position_usd = 0
            rec.drawdown_state = self.drawdown_state
            rec.confidence = 0
            return rec

        # Calculate raw Kelly
        raw_kelly, kelly_confidence = self.calculate_raw_kelly()
        rec.raw_kelly_fraction = raw_kelly
        rec.sample_size = len(self.trade_history)

        # Apply Kelly fraction (half Kelly by default)
        base_kelly = raw_kelly * self.kelly_fraction

        # Calculate adjustments
        rec.drawdown_adjustment = self.calculate_drawdown_adjustment()
        rec.volatility_adjustment = self.calculate_volatility_adjustment()
        rec.regime_adjustment = self.calculate_regime_adjustment()
        streak_adjustment = self.calculate_streak_adjustment()

        # Detect current state
        rec.drawdown_state = self.drawdown_state
        rec.market_regime = self.detect_market_regime(market)

        # Opportunity-specific adjustment
        edge_adjustment = min(opportunity_edge_pct / 1.0, 1.5)  # Cap at 1.5x
        confidence_adjustment = opportunity_confidence

        # Calculate adjusted Kelly
        total_adjustment = (
            rec.drawdown_adjustment *
            rec.volatility_adjustment *
            rec.regime_adjustment *
            streak_adjustment *
            edge_adjustment *
            confidence_adjustment
        )

        rec.adjusted_kelly_fraction = base_kelly * total_adjustment

        # Calculate position size
        position_pct = rec.adjusted_kelly_fraction * 100
        position_pct = min(position_pct, self.max_position_pct)  # Apply cap

        rec.recommended_position_pct = position_pct
        rec.recommended_position_usd = self.capital * (position_pct / 100)
        rec.max_position_usd = self.capital * (self.max_position_pct / 100)

        # Calculate expected return/risk
        stats = self.get_statistics()
        if stats["total_trades"] > 0:
            rec.expected_return_pct = stats["expectancy"] / (stats["avg_loss"] or 1) * position_pct
            rec.expected_risk_pct = position_pct * (1 - stats["win_rate"])
            rec.risk_reward_ratio = rec.expected_return_pct / rec.expected_risk_pct if rec.expected_risk_pct > 0 else 0

        # Calculate confidence
        rec.confidence = kelly_confidence * opportunity_confidence * (1 - self.current_drawdown_pct / 100)
        rec.data_quality = min(len(self.trade_history) / 50, 1.0)

        return rec

    def update_volatility(self, volatility: float):
        """Update volatility observation"""
        self.volatility_history.append(volatility)

    def update_price(self, symbol: str, price: float):
        """Update price for regime detection"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)
        self.price_history[symbol].append(price)

    def reset_drawdown_state(self):
        """Manually reset drawdown state (use with caution)"""
        self.drawdown_state = DrawdownState.NORMAL
        self.is_trading_enabled = True
        self.peak_capital = self.capital
        self.current_drawdown_pct = 0
        logger.info("Drawdown state manually reset")


# Test
def test_kelly_sizer():
    """Test the Kelly sizer"""
    sizer = AdvancedKellySizer(capital=10000, max_drawdown_pct=20)

    # Simulate trades
    import random
    random.seed(42)

    for i in range(50):
        # 60% win rate with 1.5:1 reward ratio
        is_win = random.random() < 0.6
        profit_loss = 150 if is_win else -100
        profit_pct = profit_loss / 10000 * 100

        trade = TradeRecord(
            timestamp=datetime.now().timestamp(),
            profit_loss=profit_loss,
            profit_loss_pct=profit_pct,
            position_size=1000,
            win=is_win,
            market="BTC"
        )
        sizer.add_trade(trade)

    # Get recommendation
    rec = sizer.get_position_recommendation(
        opportunity_edge_pct=0.5,
        opportunity_confidence=0.8,
        market="BTC"
    )

    print("Kelly Sizer Statistics:")
    stats = sizer.get_statistics()
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  Profit factor: {stats['profit_factor']:.2f}")
    print(f"  Expectancy: ${stats['expectancy']:.2f}")
    print(f"  Sharpe ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Current drawdown: {stats['current_drawdown_pct']:.1f}%")
    print(f"  Total return: {stats['total_return_pct']:.1f}%")

    print("\nPosition Recommendation:")
    print(f"  Raw Kelly: {rec.raw_kelly_fraction:.2%}")
    print(f"  Adjusted Kelly: {rec.adjusted_kelly_fraction:.2%}")
    print(f"  Position size: {rec.recommended_position_pct:.1f}% (${rec.recommended_position_usd:.2f})")
    print(f"  Drawdown state: {rec.drawdown_state.value}")
    print(f"  Market regime: {rec.market_regime.value}")
    print(f"  Confidence: {rec.confidence:.1%}")


if __name__ == "__main__":
    test_kelly_sizer()
