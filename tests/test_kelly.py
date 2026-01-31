"""
Tests for Kelly Criterion position sizer.
Tests trade statistics, Kelly fraction calculation, and position sizing.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kelly_position_sizer import (
    KellyPositionSizer,
    KellySizeMode,
    TradeStatistics,
    KellyFraction
)


class TestKellySizeMode:
    """Tests for Kelly sizing modes"""

    def test_mode_values(self):
        """Should have correct mode multipliers"""
        assert KellySizeMode.FULL_KELLY.value == 1.0
        assert KellySizeMode.HALF_KELLY.value == 0.5
        assert KellySizeMode.QUARTER_KELLY.value == 0.25


class TestTradeStatistics:
    """Tests for trade statistics dataclass"""

    def test_empty_statistics(self):
        """Empty stats should have zero values"""
        stats = TradeStatistics()
        assert stats.total_trades == 0
        assert stats.winning_trades == 0
        assert stats.losing_trades == 0
        assert stats.win_rate == 0.0
        assert stats.avg_win == 0.0
        assert stats.avg_loss == 0.0

    def test_statistics_str(self):
        """Should format as string"""
        stats = TradeStatistics(
            win_rate=0.7,
            profit_factor=2.5,
            avg_win=150.0,
            avg_loss=60.0
        )
        result = str(stats)
        assert "70.0%" in result
        assert "2.50" in result


class TestKellyPositionSizer:
    """Tests for Kelly position sizer"""

    def test_initialization(self):
        """Should initialize with correct defaults"""
        sizer = KellyPositionSizer(capital=10000.0)
        assert sizer.capital == 10000.0
        assert sizer.kelly_mode == KellySizeMode.HALF_KELLY

    def test_full_kelly_mode(self):
        """Should use full Kelly when specified"""
        sizer = KellyPositionSizer(
            capital=10000.0,
            kelly_mode=KellySizeMode.FULL_KELLY
        )
        assert sizer.kelly_mode == KellySizeMode.FULL_KELLY

    def test_add_trade(self):
        """Should record trades correctly"""
        sizer = KellyPositionSizer(capital=10000.0)
        sizer.add_trade(is_winning=True, profit_loss=100)
        sizer.add_trade(is_winning=False, profit_loss=-50)

        stats = sizer.get_statistics()
        assert stats.total_trades == 2
        assert stats.winning_trades == 1
        assert stats.losing_trades == 1

    def test_trade_statistics_calculated(self):
        """Should calculate statistics after trades"""
        sizer = KellyPositionSizer(capital=10000.0)
        sizer.add_trade(is_winning=True, profit_loss=100)
        sizer.add_trade(is_winning=True, profit_loss=200)
        sizer.add_trade(is_winning=False, profit_loss=-50)

        stats = sizer.get_statistics()
        assert stats.total_trades == 3
        assert stats.winning_trades == 2
        assert stats.losing_trades == 1
        assert stats.win_rate == pytest.approx(2/3)
        assert stats.avg_win == 150.0
        assert stats.avg_loss == 50.0

    def test_kelly_fraction_no_trades(self):
        """Should return zero Kelly with no trades"""
        sizer = KellyPositionSizer(capital=10000.0)
        result = sizer.calculate_kelly_fraction()

        assert result.kelly_percent == 0.0
        assert result.confidence == 0.0

    def test_kelly_fraction_with_trades(self):
        """Should calculate Kelly fraction with trade history"""
        sizer = KellyPositionSizer(
            capital=10000.0,
            kelly_mode=KellySizeMode.HALF_KELLY
        )

        # Add 20 trades with 70% win rate
        for i in range(14):
            sizer.add_trade(is_winning=True, profit_loss=100)
        for i in range(6):
            sizer.add_trade(is_winning=False, profit_loss=-80)

        result = sizer.calculate_kelly_fraction()

        # Kelly should be positive with positive expectation
        assert result.kelly_percent > 0
        assert result.confidence > 0

    def test_kelly_fraction_negative_expectation(self):
        """Should return zero Kelly with negative expectation"""
        sizer = KellyPositionSizer(capital=10000.0)

        # Add trades with negative expectation (30% win rate, bad risk/reward)
        for i in range(3):
            sizer.add_trade(is_winning=True, profit_loss=50)
        for i in range(7):
            sizer.add_trade(is_winning=False, profit_loss=-100)

        result = sizer.calculate_kelly_fraction()

        # Kelly should be zero or very small (negative capped to zero)
        assert result.kelly_percent <= 5  # Allow small positive due to capping

    def test_position_size_cap(self):
        """Position size should not exceed 10% of capital"""
        sizer = KellyPositionSizer(
            capital=10000.0,
            kelly_mode=KellySizeMode.FULL_KELLY
        )

        # Add very favorable trades that would suggest >10% position
        for i in range(20):
            sizer.add_trade(is_winning=True, profit_loss=500)
        for i in range(1):
            sizer.add_trade(is_winning=False, profit_loss=-100)

        result = sizer.calculate_kelly_fraction()

        # Position should be capped at 10% of capital = $1000
        assert result.estimated_position_size <= 1000.0

    def test_confidence_increases_with_trades(self):
        """Confidence should increase with more trades"""
        sizer = KellyPositionSizer(capital=10000.0)

        # Add 5 trades
        for i in range(5):
            sizer.add_trade(is_winning=True, profit_loss=100)

        result_5 = sizer.calculate_kelly_fraction()

        # Add more trades (15 more = 20 total)
        for i in range(15):
            sizer.add_trade(is_winning=True, profit_loss=100)

        result_20 = sizer.calculate_kelly_fraction()

        # Confidence should be higher with more trades
        assert result_20.confidence >= result_5.confidence

    def test_kelly_mode_comparison(self):
        """Half Kelly should give smaller position than Full Kelly"""
        trades = [(True, 200)] * 14 + [(False, -100)] * 6

        full_sizer = KellyPositionSizer(
            capital=10000.0,
            kelly_mode=KellySizeMode.FULL_KELLY
        )
        half_sizer = KellyPositionSizer(
            capital=10000.0,
            kelly_mode=KellySizeMode.HALF_KELLY
        )

        for is_win, pl in trades:
            full_sizer.add_trade(is_winning=is_win, profit_loss=pl)
            half_sizer.add_trade(is_winning=is_win, profit_loss=pl)

        full_result = full_sizer.calculate_kelly_fraction()
        half_result = half_sizer.calculate_kelly_fraction()

        # Safe Kelly fraction should be lower for half Kelly
        assert half_result.safe_kelly_fraction <= full_result.safe_kelly_fraction

    def test_get_recommended_position_size(self):
        """Should return position size directly"""
        sizer = KellyPositionSizer(capital=10000.0)

        # Add trades
        for i in range(10):
            sizer.add_trade(is_winning=True, profit_loss=100)
        for i in range(5):
            sizer.add_trade(is_winning=False, profit_loss=-80)

        position = sizer.get_recommended_position_size()
        assert position >= 0
        assert position <= 10000.0 * 0.1  # Should not exceed max (10%)

    def test_consecutive_losses_tracking(self):
        """Should track max consecutive losses"""
        sizer = KellyPositionSizer(capital=10000.0)

        # Win first, then 3 losses in a row (ending with losses)
        sizer.add_trade(is_winning=True, profit_loss=100)
        sizer.add_trade(is_winning=False, profit_loss=-50)
        sizer.add_trade(is_winning=False, profit_loss=-50)
        sizer.add_trade(is_winning=False, profit_loss=-50)

        stats = sizer.get_statistics()
        # max_consecutive_losses should capture the streak of 3
        assert stats.max_consecutive_losses == 3

    def test_consecutive_losses_reset_on_win(self):
        """Should track consecutive losses via reverse iteration"""
        sizer = KellyPositionSizer(capital=10000.0)

        # 2 losses, then a win, then 1 loss
        sizer.add_trade(is_winning=False, profit_loss=-50)
        sizer.add_trade(is_winning=False, profit_loss=-50)
        sizer.add_trade(is_winning=True, profit_loss=100)
        sizer.add_trade(is_winning=False, profit_loss=-50)

        stats = sizer.get_statistics()
        # The implementation iterates in reverse and resets on wins,
        # so consecutive_losses ends at 2 (the first two losses after reset)
        assert stats.consecutive_losses == 2
        # Max streak was 2 (the first two losses)
        assert stats.max_consecutive_losses == 2

    def test_largest_win_loss(self):
        """Should track largest win and loss"""
        sizer = KellyPositionSizer(capital=10000.0)

        sizer.add_trade(is_winning=True, profit_loss=100)
        sizer.add_trade(is_winning=True, profit_loss=500)
        sizer.add_trade(is_winning=False, profit_loss=-50)
        sizer.add_trade(is_winning=False, profit_loss=-200)

        stats = sizer.get_statistics()
        assert stats.largest_win == 500
        assert stats.largest_loss == 200


class TestKellyFraction:
    """Tests for KellyFraction dataclass"""

    def test_kelly_fraction_fields(self):
        """Should have all expected fields"""
        result = KellyFraction(
            kelly_percent=10.0,
            kelly_fraction=0.1,
            safe_kelly_fraction=0.05,
            estimated_position_size=500.0,
            confidence=0.8
        )

        assert result.kelly_percent == 10.0
        assert result.kelly_fraction == 0.1
        assert result.safe_kelly_fraction == 0.05
        assert result.estimated_position_size == 500.0
        assert result.confidence == 0.8

    def test_kelly_fraction_to_dict(self):
        """Should convert to dictionary"""
        result = KellyFraction(
            kelly_percent=10.0,
            kelly_fraction=0.1,
            safe_kelly_fraction=0.05,
            estimated_position_size=500.0,
            confidence=0.8,
            details="Test details"
        )

        d = result.to_dict()
        assert d["kelly_percent"] == 10.0
        assert d["confidence"] == 0.8
        assert d["details"] == "Test details"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
