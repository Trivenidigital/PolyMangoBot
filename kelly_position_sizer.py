"""
Kelly Criterion Position Sizer Module
Dynamic position sizing based on trading statistics and win rate
Implements Kelly formula for optimal position sizing with conservative safety factor
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KellySizeMode(Enum):
    """Kelly sizing modes"""
    FULL_KELLY = 1.0  # f* = (bp - q) / b
    HALF_KELLY = 0.5  # Conservative: use 50% of Kelly fraction
    QUARTER_KELLY = 0.25  # Very conservative: use 25% of Kelly fraction


@dataclass
class TradeStatistics:
    """Trading performance statistics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0  # 0-1 scale
    avg_win: float = 0.0  # Average profit per winning trade
    avg_loss: float = 0.0  # Average loss per losing trade
    profit_factor: float = 0.0  # Total wins / Total losses
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0

    def __str__(self) -> str:
        return (
            f"Win rate: {self.win_rate*100:.1f}% | "
            f"Profit factor: {self.profit_factor:.2f} | "
            f"Avg win: ${self.avg_win:.2f} | "
            f"Avg loss: ${self.avg_loss:.2f}"
        )


@dataclass
class KellyFraction:
    """Kelly criterion calculation result"""
    kelly_percent: float  # Kelly as percentage (0-100)
    kelly_fraction: float  # Kelly as fraction (0-1)
    safe_kelly_fraction: float  # Conservative Kelly (usually 50% of kelly_fraction)
    estimated_position_size: float  # Position size in USD
    confidence: float  # 0-1 scale: how confident in calculation
    details: str = ""

    def to_dict(self) -> Dict:
        return {
            'kelly_percent': self.kelly_percent,
            'kelly_fraction': self.kelly_fraction,
            'safe_kelly_fraction': self.safe_kelly_fraction,
            'estimated_position_size': self.estimated_position_size,
            'confidence': self.confidence,
            'details': self.details
        }


class KellyPositionSizer:
    """
    Calculates optimal position sizing using Kelly Criterion

    Kelly Formula:
    f* = (bp - q) / b

    Where:
    - f* = Kelly fraction (fraction of bankroll to risk)
    - b = odds (win_amount / loss_amount)
    - p = probability of winning (win_rate)
    - q = probability of losing (1 - win_rate)

    For trading:
    - b = avg_win / avg_loss
    - p = win_rate
    - q = 1 - win_rate

    The result f* tells us what fraction of our bankroll to risk on each trade
    """

    def __init__(self, capital: float = 10000.0, kelly_mode: KellySizeMode = KellySizeMode.HALF_KELLY):
        """
        Initialize Kelly position sizer

        Args:
            capital: Total trading capital (bankroll)
            kelly_mode: Which Kelly variant to use (Full, Half, Quarter)
        """
        self.capital = capital
        self.kelly_mode = kelly_mode
        self.trade_history: List[Dict] = []
        self.stats: TradeStatistics = TradeStatistics()

    def add_trade(self, is_winning: bool, profit_loss: float):
        """
        Record a trade result

        Args:
            is_winning: True if trade was profitable
            profit_loss: Profit/loss amount (positive or negative)
        """
        self.trade_history.append({
            'is_winning': is_winning,
            'profit_loss': profit_loss,
            'timestamp': datetime.now()
        })

        # Update statistics
        self._update_statistics()

    def _update_statistics(self):
        """Recalculate statistics from trade history"""
        if not self.trade_history:
            return

        self.stats.total_trades = len(self.trade_history)
        self.stats.winning_trades = sum(1 for t in self.trade_history if t['is_winning'])
        self.stats.losing_trades = self.stats.total_trades - self.stats.winning_trades

        # Win rate
        self.stats.win_rate = self.stats.winning_trades / self.stats.total_trades if self.stats.total_trades > 0 else 0

        # Separate wins and losses
        wins = [t['profit_loss'] for t in self.trade_history if t['is_winning']]
        losses = [-t['profit_loss'] for t in self.trade_history if not t['is_winning']]

        # Average win and loss
        self.stats.avg_win = statistics.mean(wins) if wins else 0
        self.stats.avg_loss = statistics.mean(losses) if losses else 0

        # Profit factor (total wins / total losses)
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        self.stats.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Largest win/loss
        self.stats.largest_win = max(wins) if wins else 0
        self.stats.largest_loss = max(losses) if losses else 0

        # Consecutive losses
        self.stats.consecutive_losses = 0
        self.stats.max_consecutive_losses = 0
        for trade in reversed(self.trade_history):
            if not trade['is_winning']:
                self.stats.consecutive_losses += 1
                self.stats.max_consecutive_losses = max(
                    self.stats.max_consecutive_losses,
                    self.stats.consecutive_losses
                )
            else:
                self.stats.consecutive_losses = 0

    def calculate_kelly_fraction(self) -> KellyFraction:
        """
        Calculate Kelly fraction from trade statistics

        Returns:
            KellyFraction object with sizing recommendations
        """

        # Minimum trades needed for confidence
        min_trades = 20
        if self.stats.total_trades < min_trades:
            confidence = self.stats.total_trades / min_trades
            logger.warning(f"Low trade count ({self.stats.total_trades}/{min_trades}) - reduced confidence")
        else:
            confidence = 1.0

        # Handle edge cases
        if self.stats.total_trades == 0:
            logger.warning("No trade history - cannot calculate Kelly fraction")
            return KellyFraction(
                kelly_percent=0,
                kelly_fraction=0,
                safe_kelly_fraction=0,
                estimated_position_size=self.capital * 0.02,  # Default to 2% of capital
                confidence=0,
                details="No trade history available"
            )

        if self.stats.win_rate == 0:
            logger.warning("0% win rate - Kelly fraction is 0")
            return KellyFraction(
                kelly_percent=0,
                kelly_fraction=0,
                safe_kelly_fraction=0,
                estimated_position_size=0,
                confidence=confidence,
                details="0% win rate means Kelly fraction is 0"
            )

        if self.stats.avg_loss == 0:
            logger.warning("Average loss is 0 - cannot calculate Kelly")
            return KellyFraction(
                kelly_percent=0,
                kelly_fraction=0,
                safe_kelly_fraction=0,
                estimated_position_size=self.capital * 0.02,
                confidence=0,
                details="Cannot divide by zero (avg_loss = 0)"
            )

        # Calculate odds (b = avg_win / avg_loss)
        b = self.stats.avg_win / self.stats.avg_loss if self.stats.avg_loss > 0 else 1

        # Calculate Kelly fraction: f* = (bp - q) / b
        p = self.stats.win_rate
        q = 1 - self.stats.win_rate

        kelly_fraction = ((b * p) - q) / b if b > 0 else 0

        # Clamp to reasonable range (0-100%)
        kelly_fraction = max(0, min(kelly_fraction, 1.0))

        # Apply kelly mode (full, half, quarter)
        safe_kelly_fraction = kelly_fraction * self.kelly_mode.value

        # Calculate position size in USD
        position_size = self.capital * safe_kelly_fraction

        # Additional safety: limit to 10% of capital max
        max_position = self.capital * 0.10
        position_size = min(position_size, max_position)

        kelly_percent = kelly_fraction * 100
        safe_kelly_percent = safe_kelly_fraction * 100

        details = (
            f"Kelly formula: f* = (bp - q) / b = "
            f"({b:.2f} * {p:.3f} - {q:.3f}) / {b:.2f} = {kelly_percent:.2f}% "
            f"| Conservative ({self.kelly_mode.name}): {safe_kelly_percent:.2f}% "
            f"| Position: ${position_size:.2f}"
        )

        return KellyFraction(
            kelly_percent=kelly_percent,
            kelly_fraction=kelly_fraction,
            safe_kelly_fraction=safe_kelly_fraction,
            estimated_position_size=position_size,
            confidence=confidence,
            details=details
        )

    def get_recommended_position_size(self) -> float:
        """Get recommended position size in USD based on current statistics"""
        kelly = self.calculate_kelly_fraction()
        return kelly.estimated_position_size

    def get_statistics(self) -> TradeStatistics:
        """Get current trading statistics"""
        return self.stats

    def print_analysis(self):
        """Print detailed Kelly analysis"""
        kelly = self.calculate_kelly_fraction()

        print(f"\n{'='*70}")
        print(f"KELLY CRITERION POSITION SIZING ANALYSIS")
        print(f"{'='*70}")

        print(f"\nTrade Statistics:")
        print(f"  Total trades: {self.stats.total_trades}")
        print(f"  Winning trades: {self.stats.winning_trades}")
        print(f"  Losing trades: {self.stats.losing_trades}")
        print(f"  Win rate: {self.stats.win_rate*100:.1f}%")
        print(f"  Profit factor: {self.stats.profit_factor:.2f}")
        print(f"  Avg win: ${self.stats.avg_win:.2f}")
        print(f"  Avg loss: ${self.stats.avg_loss:.2f}")
        print(f"  Max consecutive losses: {self.stats.max_consecutive_losses}")

        print(f"\nKelly Calculation:")
        print(f"  Capital: ${self.capital:.2f}")
        print(f"  Kelly mode: {self.kelly_mode.name}")
        print(f"  Full Kelly fraction: {kelly.kelly_percent:.2f}%")
        print(f"  Safe Kelly fraction ({self.kelly_mode.name}): {kelly.kelly_percent * self.kelly_mode.value:.2f}%")
        print(f"  Recommended position size: ${kelly.estimated_position_size:.2f}")
        print(f"  Confidence: {kelly.confidence*100:.0f}%")

        print(f"\nRecommendation:")
        if kelly.estimated_position_size > 0:
            percent_of_capital = (kelly.estimated_position_size / self.capital) * 100
            print(f"  Risk ${kelly.estimated_position_size:.2f} per trade ({percent_of_capital:.1f}% of capital)")
        else:
            print(f"  Do not trade - insufficient statistical edge")

        print(f"\n{kelly.details}")
        print(f"{'='*70}\n")

    def get_kelly_history_trend(self, window: int = 10) -> Dict:
        """
        Analyze Kelly fraction trend over recent trades
        Shows if position sizing should be increasing or decreasing
        """
        if len(self.trade_history) < window:
            return {"trend": "insufficient_data", "message": "Need more trades for trend analysis"}

        recent_trades = self.trade_history[-window:]
        recent_wins = sum(1 for t in recent_trades if t['is_winning'])
        recent_win_rate = recent_wins / window

        # Compare to overall
        trend = "increasing" if recent_win_rate > self.stats.win_rate else "decreasing"

        return {
            "trend": trend,
            "recent_win_rate": recent_win_rate,
            "overall_win_rate": self.stats.win_rate,
            "message": f"Performance {trend}: {recent_win_rate*100:.1f}% vs {self.stats.win_rate*100:.1f}%"
        }


class PositionSizerWithRiskValidator:
    """
    Combines Kelly sizing with risk validation
    Used by risk_validator.py for integrated position sizing
    """

    def __init__(self, capital: float, kelly_mode: KellySizeMode = KellySizeMode.HALF_KELLY):
        self.sizer = KellyPositionSizer(capital, kelly_mode)

    def calculate_size_for_trade(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_position_size: float = None
    ) -> float:
        """
        Calculate optimal position size for a trade

        Args:
            capital: Available capital
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount
            max_position_size: Maximum allowed position (safety limit)

        Returns:
            Recommended position size in USD
        """

        if avg_loss == 0 or avg_win == 0 or win_rate == 0:
            # Fallback to conservative 2% of capital
            fallback_size = capital * 0.02
            if max_position_size:
                return min(fallback_size, max_position_size)
            return fallback_size

        # Calculate Kelly fraction
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly_fraction = ((b * p) - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 1.0))

        # Apply conservative mode (50% of Kelly)
        safe_kelly = kelly_fraction * 0.5

        # Position size
        position_size = capital * safe_kelly

        # Hard cap at 10% of capital
        position_size = min(position_size, capital * 0.10)

        # Apply max if specified
        if max_position_size:
            position_size = min(position_size, max_position_size)

        return position_size


# Test
def test_kelly_sizer():
    """Test Kelly position sizer"""

    sizer = KellyPositionSizer(capital=10000, kelly_mode=KellySizeMode.HALF_KELLY)

    # Simulate trades
    trades = [
        (True, 150),    # Win $150
        (True, 200),    # Win $200
        (False, -100),  # Lose $100
        (True, 180),    # Win $180
        (False, -90),   # Lose $90
        (True, 220),    # Win $220
        (True, 160),    # Win $160
        (False, -110),  # Lose $110
        (True, 190),    # Win $190
        (True, 210),    # Win $210
        (False, -95),   # Lose $95
        (True, 175),    # Win $175
        (True, 185),    # Win $185
        (True, 205),    # Win $205
        (False, -100),  # Lose $100
        (True, 195),    # Win $195
        (True, 215),    # Win $215
        (False, -105),  # Lose $105
        (True, 170),    # Win $170
        (True, 200),    # Win $200
    ]

    print("Recording trades...")
    for is_win, pl in trades:
        sizer.add_trade(is_win, pl)

    # Print analysis
    sizer.print_analysis()

    # Test trend analysis
    trend = sizer.get_kelly_history_trend()
    print(f"Trend analysis: {trend['message']}")

    # Test with various Kelly modes
    print("\n" + "="*70)
    print("KELLY MODE COMPARISON")
    print("="*70)

    for mode in [KellySizeMode.FULL_KELLY, KellySizeMode.HALF_KELLY, KellySizeMode.QUARTER_KELLY]:
        sizer_test = KellyPositionSizer(capital=10000, kelly_mode=mode)
        for is_win, pl in trades:
            sizer_test.add_trade(is_win, pl)

        kelly = sizer_test.calculate_kelly_fraction()
        print(f"{mode.name:15} -> Position: ${kelly.estimated_position_size:.2f} "
              f"({kelly.kelly_percent * mode.value:.2f}%)")


if __name__ == "__main__":
    test_kelly_sizer()
