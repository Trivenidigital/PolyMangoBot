"""
Risk Validator Module
Ensures trades meet safety requirements
Now includes Kelly Criterion position sizing for optimal sizing
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
from fee_estimator import CombinedCostEstimator, FeeEstimator, SlippageEstimator
from kelly_position_sizer import KellyPositionSizer, PositionSizerWithRiskValidator, KellySizeMode


class RiskLevel(Enum):
    """Risk assessment levels"""
    SAFE = "safe"
    CAUTION = "caution"
    REJECT = "reject"


@dataclass
class RiskReport:
    """Risk assessment result"""
    is_safe: bool
    risk_level: RiskLevel
    reasons: List[str]
    estimated_profit_after_fees: float
    
    def print_report(self):
        """Pretty print the report"""
        print(f"\n{'='*50}")
        print(f"Risk Level: {self.risk_level.value.upper()}")
        print(f"Safe to trade: {' YES' if self.is_safe else ' NO'}")
        print(f"Estimated profit after fees: ${self.estimated_profit_after_fees:.4f}")
        print(f"\nReasons:")
        for reason in self.reasons:
            print(f"   {reason}")
        print(f"{'='*50}\n")


class RiskValidator:
    """Validates trades against risk rules with dynamic fee/slippage estimation and Kelly sizing"""

    def __init__(
        self,
        max_position_size: float = 1000.0,          # Max $ per trade
        max_daily_loss: float = 5000.0,             # Max daily loss
        min_profit_margin: float = 0.5,             # Min % profit after fees
        max_spread_percent: float = 50.0,           # Don't arb if spread > 50%
        maker_fee_percent: float = 0.1,             # Maker fee % (fallback)
        taker_fee_percent: float = 0.15,            # Taker fee % (fallback)
        slippage_percent: float = 0.25,             # Expected slippage % (fallback)
        use_dynamic_estimation: bool = True,        # Use ML-based fee/slippage estimation
        capital: float = 10000.0,                   # Trading capital for Kelly sizing
        enable_kelly_sizing: bool = True,           # Use Kelly Criterion for position sizing
        kelly_mode: KellySizeMode = KellySizeMode.HALF_KELLY,
    ):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.min_profit_margin = min_profit_margin
        self.max_spread_percent = max_spread_percent
        self.maker_fee = maker_fee_percent
        self.taker_fee = taker_fee_percent
        self.slippage = slippage_percent

        self.daily_loss = 0.0
        self.open_positions = []

        # Dynamic estimation
        self.use_dynamic = use_dynamic_estimation
        self.cost_estimator = CombinedCostEstimator() if use_dynamic_estimation else None

        # Kelly Criterion position sizing
        self.enable_kelly = enable_kelly_sizing
        self.kelly_sizer = KellyPositionSizer(capital=capital, kelly_mode=kelly_mode) if enable_kelly_sizing else None
        self.capital = capital
    
    def validate_trade(
        self,
        market: str,
        buy_venue: str,
        buy_price: float,
        sell_venue: str,
        sell_price: float,
        position_size: float = None,
        market_volume_24h: float = None,
        volatility: float = None
    ) -> RiskReport:
        """
        Validate if a trade is safe to execute.

        Uses dynamic fee/slippage estimation if available.
        Uses Kelly Criterion for position sizing if enabled.

        Returns RiskReport with safety assessment.
        """

        reasons = []
        is_safe = True
        risk_level = RiskLevel.SAFE

        # 1. Check spread doesn't exceed threshold
        spread_percent = ((sell_price - buy_price) / buy_price) * 100
        if spread_percent > self.max_spread_percent:
            reasons.append(f"Spread {spread_percent:.2f}% exceeds max {self.max_spread_percent}%")
            is_safe = False
            risk_level = RiskLevel.REJECT

        # 2. Calculate fees and slippage (dynamically or static)
        if self.use_dynamic and market_volume_24h:
            # Use ML-based estimation
            cost = self.cost_estimator.estimate_total_cost(
                venue=buy_venue,
                symbol=market,
                position_size=position_size or 1000,
                market_volume_24h=market_volume_24h,
                volatility=volatility or 0.5
            )
            total_cost_percent = cost['total_cost_percent']
            fee_percent = cost['fee_percent']
            slippage_percent = cost['slippage_percent']

            reasons.append(f"Dynamic fee est: {fee_percent:.3f}%, slippage: {slippage_percent:.3f}%")
        else:
            # Use static fallback
            fee_percent = (self.taker_fee + self.maker_fee) / 2
            slippage_percent = self.slippage
            total_cost_percent = fee_percent + slippage_percent

        buy_cost_with_fee = buy_price * (1 + self.taker_fee / 100)
        sell_revenue_after_fee = sell_price * (1 - self.maker_fee / 100)
        sell_revenue_after_cost = sell_revenue_after_fee * (1 - total_cost_percent / 100)

        profit_per_unit = sell_revenue_after_cost - buy_cost_with_fee
        profit_percent = (profit_per_unit / buy_price) * 100

        # 3. Check minimum profit threshold
        if profit_percent < self.min_profit_margin:
            reasons.append(f"Profit {profit_percent:.2f}% below minimum {self.min_profit_margin}%")
            is_safe = False
            risk_level = RiskLevel.REJECT

        # 4. Determine position size - use Kelly if enabled, else static
        if position_size is None:
            if self.enable_kelly and self.kelly_sizer:
                # Use Kelly Criterion for sizing
                kelly_size = self.kelly_sizer.get_recommended_position_size()
                position_size = kelly_size
                reasons.append(f"Kelly sizing: ${position_size:.2f} ({kelly_size/self.capital*100:.1f}% of capital)")
            else:
                # Default to 50% of max
                position_size = self.max_position_size * 0.5
        if position_size > self.max_position_size:
            reasons.append(f"Position ${position_size} exceeds max ${self.max_position_size}")
            is_safe = False
            risk_level = RiskLevel.CAUTION

        # 5. Check daily loss limit
        potential_loss = position_size * 0.01  # Assume 1% max loss
        if (self.daily_loss + potential_loss) > self.max_daily_loss:
            reasons.append(f"Daily loss would exceed ${self.max_daily_loss}")
            is_safe = False
            risk_level = RiskLevel.CAUTION

        # 6. Positive signals
        if is_safe:
            reasons.append(f"Spread is {spread_percent:.2f}% (healthy)")
            reasons.append(f"Profit after fees/slippage: {profit_percent:.3f}%")
            reasons.append(f"Position size ${position_size} is within limits")
            reasons.append(f"Daily loss impact: ${self.daily_loss + potential_loss:.2f}/${self.max_daily_loss}")

        estimated_profit = position_size * (profit_percent / 100)

        return RiskReport(
            is_safe=is_safe,
            risk_level=risk_level,
            reasons=reasons,
            estimated_profit_after_fees=estimated_profit
        )
    
    def update_daily_loss(self, loss: float):
        """Update daily loss tracker"""
        self.daily_loss += loss
    
    def reset_daily_loss(self):
        """Reset daily loss (call at end of trading day)"""
        self.daily_loss = 0.0
    
    def add_position(self, trade_id: str, position_size: float):
        """Track open position"""
        self.open_positions.append({"id": trade_id, "size": position_size})
    
    def remove_position(self, trade_id: str):
        """Remove closed position"""
        self.open_positions = [p for p in self.open_positions if p["id"] != trade_id]

    def get_total_exposure(self) -> float:
        """Get total open position size"""
        return sum(p["size"] for p in self.open_positions)

    def record_trade_result(self, is_profitable: bool, profit_loss: float):
        """Record trade result for Kelly sizing"""
        if self.kelly_sizer:
            self.kelly_sizer.add_trade(is_profitable, profit_loss)

    def get_kelly_statistics(self):
        """Get Kelly sizing statistics"""
        if self.kelly_sizer:
            return self.kelly_sizer.get_statistics()
        return None

    def get_kelly_recommendation(self):
        """Get current Kelly position sizing recommendation"""
        if self.kelly_sizer:
            return self.kelly_sizer.calculate_kelly_fraction()
        return None

    def print_kelly_analysis(self):
        """Print Kelly criterion analysis"""
        if self.kelly_sizer:
            self.kelly_sizer.print_analysis()
        else:
            print("[INFO] Kelly sizing not enabled")


# Test the module
def test_risk_validator():
    """Test risk validation"""
    
    validator = RiskValidator(
        max_position_size=1000,
        min_profit_margin=0.3,
        maker_fee_percent=0.1,
        taker_fee_percent=0.15,
    )
    
    print("Testing risk scenarios...\n")
    
    # Scenario 1: Good trade
    print("SCENARIO 1: Healthy arb opportunity")
    report1 = validator.validate_trade(
        market="BTC",
        buy_venue="kraken",
        buy_price=42500,
        sell_venue="polymarket",
        sell_price=42700,
        position_size=500
    )
    report1.print_report()
    
    # Scenario 2: Low profit margin
    print("SCENARIO 2: Tight margin (likely unprofitable)")
    report2 = validator.validate_trade(
        market="ETH",
        buy_venue="kraken",
        buy_price=2300,
        sell_venue="coinbase",
        sell_price=2310,  # Only 0.43% difference
        position_size=500
    )
    report2.print_report()
    
    # Scenario 3: Position too large
    print("SCENARIO 3: Position size too large")
    report3 = validator.validate_trade(
        market="DOGE",
        buy_venue="kraken",
        buy_price=0.50,
        sell_venue="polymarket",
        sell_price=0.75,
        position_size=5000  # Way too large
    )
    report3.print_report()


if __name__ == "__main__":
    test_risk_validator()