"""
Dynamic Fee and Slippage Estimator
Predicts actual fees and slippage based on market conditions
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics


@dataclass
class FeeEstimate:
    """Fee estimation result"""
    maker_fee_percent: float
    taker_fee_percent: float
    total_fee_percent: float
    confidence: float  # 0-1, how confident in estimate


@dataclass
class SlippageEstimate:
    """Slippage estimation result"""
    slippage_percent: float
    confidence: float  # 0-1
    reasoning: str


class FeeEstimator:
    """Estimates trading fees based on volume tiers and venue"""

    # Polymarket fee tiers (estimated)
    POLYMARKET_FEES = {
        'default': {'maker': 0.05, 'taker': 0.10},  # 5%, 10% spreads typical
        'high_volume': {'maker': 0.02, 'taker': 0.05},  # Volume discounts
    }

    # Kraken fee tiers (actual as of 2026)
    KRAKEN_FEES = {
        0: {'maker': 0.16, 'taker': 0.26},  # <$50k
        50000: {'maker': 0.14, 'taker': 0.24},  # $50k-100k
        100000: {'maker': 0.12, 'taker': 0.22},  # $100k-500k
        500000: {'maker': 0.10, 'taker': 0.20},  # $500k-1M
        1000000: {'maker': 0.08, 'taker': 0.18},  # $1M-5M
        5000000: {'maker': 0.06, 'taker': 0.16},  # $5M-10M
        10000000: {'maker': 0.04, 'taker': 0.14},  # $10M-25M
        25000000: {'maker': 0.02, 'taker': 0.12},  # >$25M
    }

    def __init__(self):
        self.volume_30day: Dict[str, float] = {}  # venue -> volume

    def update_volume(self, venue: str, volume: float):
        """Update 30-day volume for a venue"""
        self.volume_30day[venue] = volume

    def estimate_kraken_fees(self, volume_30day: float) -> FeeEstimate:
        """Estimate Kraken fees based on 30-day volume"""

        # Find applicable tier
        maker_fee = 0.16
        taker_fee = 0.26

        for min_volume in sorted(self.KRAKEN_FEES.keys(), reverse=True):
            if volume_30day >= min_volume:
                fees = self.KRAKEN_FEES[min_volume]
                maker_fee = fees['maker']
                taker_fee = fees['taker']
                break

        # Convert basis points to percentage (Kraken uses basis points)
        maker_percent = maker_fee / 100
        taker_percent = taker_fee / 100

        return FeeEstimate(
            maker_fee_percent=maker_percent,
            taker_fee_percent=taker_percent,
            total_fee_percent=(maker_percent + taker_percent) / 2,
            confidence=0.95  # Kraken fees are public
        )

    def estimate_polymarket_fees(self, position_size: float,
                                 total_volume_24h: float) -> FeeEstimate:
        """
        Estimate Polymarket fees

        Polymarket uses bid-ask spreads as fee mechanism
        Actual cost depends on market maker spreads
        """

        # Polymarket doesn't charge explicit fees but uses spreads
        # For prediction markets, typical spreads are 2-5%
        # But for liquid markets, spreads are 0.5-2%

        if total_volume_24h < 10000:
            # Low volume market, wide spreads
            spread_percent = 3.0
        elif total_volume_24h < 100000:
            spread_percent = 1.5
        elif total_volume_24h < 1000000:
            spread_percent = 0.8
        else:
            # Highly liquid market
            spread_percent = 0.3

        # Maker/taker split (prediction markets often charge on the order book side)
        maker_fee = spread_percent * 0.3
        taker_fee = spread_percent * 0.7

        return FeeEstimate(
            maker_fee_percent=maker_fee,
            taker_fee_percent=taker_fee,
            total_fee_percent=spread_percent,
            confidence=0.70  # Less confident without real data
        )

    def estimate_fees(self, venue: str, position_size: float = None,
                     volume_24h: float = None) -> FeeEstimate:
        """
        Estimate trading fees for a specific venue

        Args:
            venue: 'kraken', 'polymarket', etc.
            position_size: Size of trade
            volume_24h: Market's 24h volume
        """

        if venue.lower() == 'kraken':
            volume_30d = self.volume_30day.get('kraken', 0)
            return self.estimate_kraken_fees(volume_30d)

        elif venue.lower() == 'polymarket':
            return self.estimate_polymarket_fees(position_size or 1000, volume_24h or 100000)

        else:
            # Default conservative estimate
            return FeeEstimate(
                maker_fee_percent=0.10,
                taker_fee_percent=0.15,
                total_fee_percent=0.125,
                confidence=0.30
            )


class SlippageEstimator:
    """Estimates order slippage based on market conditions"""

    def __init__(self):
        self.volatility_cache: Dict[str, float] = {}
        self.liquidity_cache: Dict[str, float] = {}

    def update_volatility(self, symbol: str, volatility: float):
        """Update recent volatility for symbol"""
        self.volatility_cache[symbol] = volatility

    def update_liquidity(self, symbol: str, depth: float):
        """Update order book depth for symbol"""
        self.liquidity_cache[symbol] = depth

    def estimate_slippage(self, symbol: str, position_size: float,
                         market_volume_24h: float, volatility: float = None) -> SlippageEstimate:
        """
        Estimate slippage for an order

        Formula:
        base_slippage = 0.03% (baseline)
        volume_impact = (position_size / market_volume_24h) * 0.5
        volatility_multiplier = 1.0 + (volatility * 0.2)

        Args:
            symbol: Trading pair
            position_size: Size of order in USD
            market_volume_24h: Total market volume in USD
            volatility: Recent volatility (0-1 scale)
        """

        # Base slippage
        base_slippage = 0.03

        # Volume impact: larger orders relative to market slippage more
        volume_ratio = position_size / max(market_volume_24h, 1000)
        volume_slippage = min(volume_ratio * 0.5, 0.5)  # Cap at 0.5%

        # Volatility impact: higher volatility = more slippage
        if volatility is None:
            volatility = self.volatility_cache.get(symbol, 0.5)

        volatility_mult = 1.0 + (volatility * 0.2)

        # Time of day impact
        now = datetime.now()
        hour = now.hour

        # Markets are typically less liquid 2-6 AM UTC
        if 2 <= hour <= 6:
            time_mult = 1.3  # +30% slippage during low liquidity hours
        elif 8 <= hour <= 16:  # US business hours
            time_mult = 0.9  # -10% slippage during high liquidity
        else:
            time_mult = 1.0

        # Final slippage calculation
        slippage = (base_slippage + volume_slippage) * volatility_mult * time_mult

        reasoning = (
            f"Base: {base_slippage}% + "
            f"Volume impact: {volume_slippage:.3f}% + "
            f"Volatility mult: {volatility_mult:.2f}x + "
            f"Time mult: {time_mult:.2f}x"
        )

        return SlippageEstimate(
            slippage_percent=slippage,
            confidence=0.75,
            reasoning=reasoning
        )

    def estimate_min_slippage(self) -> float:
        """Best case slippage in ideal conditions"""
        return 0.02

    def estimate_max_slippage(self) -> float:
        """Worst case slippage in terrible conditions"""
        return 1.5


class CombinedCostEstimator:
    """Combines fees and slippage for total transaction cost"""

    def __init__(self):
        self.fee_estimator = FeeEstimator()
        self.slippage_estimator = SlippageEstimator()

    def estimate_total_cost(self, venue: str, symbol: str,
                           position_size: float,
                           market_volume_24h: float,
                           volatility: float = None) -> Dict:
        """
        Estimate complete transaction cost

        Returns:
            {
                'total_cost_percent': float,
                'fee_percent': float,
                'slippage_percent': float,
                'confidence': float,
                'details': str
            }
        """

        fee_est = self.fee_estimator.estimate_fees(
            venue, position_size, market_volume_24h
        )
        slip_est = self.slippage_estimator.estimate_slippage(
            symbol, position_size, market_volume_24h, volatility
        )

        # Total cost is combination (not additive due to partial fills)
        total_cost = fee_est.total_fee_percent + slip_est.slippage_percent

        confidence = (fee_est.confidence + slip_est.confidence) / 2

        return {
            'total_cost_percent': total_cost,
            'fee_percent': fee_est.total_fee_percent,
            'slippage_percent': slip_est.slippage_percent,
            'confidence': confidence,
            'fee_breakdown': f"Maker: {fee_est.maker_fee_percent:.3f}%, Taker: {fee_est.taker_fee_percent:.3f}%",
            'slippage_details': slip_est.reasoning
        }


# Test
def test_fee_and_slippage():
    """Test estimators"""

    fee_est = FeeEstimator()
    slip_est = SlippageEstimator()
    combined = CombinedCostEstimator()

    print("=== Fee Estimation ===")
    kraken_fees = fee_est.estimate_kraken_fees(volume_30day=500000)
    print(f"Kraken ($500k volume): {kraken_fees.total_fee_percent:.3f}%")

    poly_fees = fee_est.estimate_polymarket_fees(position_size=1000, total_volume_24h=1000000)
    print(f"Polymarket ($1M volume): {poly_fees.total_fee_percent:.3f}%")

    print("\n=== Slippage Estimation ===")
    slip1 = slip_est.estimate_slippage("BTC", position_size=10000, market_volume_24h=1000000)
    print(f"BTC, $10k order, $1M volume: {slip1.slippage_percent:.3f}%")

    slip2 = slip_est.estimate_slippage("BTC", position_size=100000, market_volume_24h=100000)
    print(f"BTC, $100k order, $100k volume: {slip2.slippage_percent:.3f}%")

    print("\n=== Combined Cost ===")
    cost = combined.estimate_total_cost(
        venue="kraken",
        symbol="BTC",
        position_size=10000,
        market_volume_24h=1000000,
        volatility=0.5
    )
    print(f"Total cost: {cost['total_cost_percent']:.3f}%")
    print(f"Breakdown: {cost['fee_breakdown']}")
    print(f"Slippage details: {cost['slippage_details']}")


if __name__ == "__main__":
    test_fee_and_slippage()
