"""
Capital Efficiency and Compounding Strategy Manager
Advanced capital management for optimal growth and risk control.

Features:
1. Capital Allocation Framework
   - Multi-strategy capital allocation
   - Reserve capital management
   - Dynamic rebalancing
   - Opportunity cost optimization

2. Compounding Rules
   - Profit reinvestment strategies
   - Milestone-based withdrawals
   - Growth targets
   - Compound interest optimization

3. Performance-Based Sizing
   - Performance tier system
   - Adaptive position limits
   - Reward for consistency
   - Penalty for drawdowns
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("PolyMangoBot.capital_efficiency")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class AllocationStrategy(Enum):
    """Capital allocation strategies"""
    EQUAL_WEIGHT = "equal_weight"           # Equal allocation to all strategies
    RISK_PARITY = "risk_parity"             # Allocate based on inverse volatility
    KELLY_OPTIMAL = "kelly_optimal"         # Kelly-based optimal allocation
    PERFORMANCE_WEIGHTED = "performance_weighted"  # Weight by recent performance
    MOMENTUM = "momentum"                   # Allocate more to winning strategies


class CompoundingMode(Enum):
    """Compounding modes"""
    FULL_REINVEST = "full_reinvest"         # Reinvest all profits
    PARTIAL_REINVEST = "partial_reinvest"   # Reinvest percentage of profits
    MILESTONE_WITHDRAW = "milestone_withdraw"  # Withdraw at milestones
    TARGET_GROWTH = "target_growth"         # Compound until target reached
    HYBRID = "hybrid"                       # Combination approach


class PerformanceTier(Enum):
    """Performance tier levels"""
    ELITE = "elite"           # Top tier - maximum privileges
    ADVANCED = "advanced"     # High performer
    STANDARD = "standard"     # Normal operation
    PROBATION = "probation"   # Underperforming - reduced sizing
    RESTRICTED = "restricted" # Poor performance - minimum sizing


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy"""
    strategy_id: str
    strategy_name: str

    # Allocation
    target_allocation_pct: float = 0.0      # Target % of capital
    current_allocation_pct: float = 0.0     # Current % of capital
    allocated_capital: float = 0.0          # Absolute allocation

    # Performance
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0                     # Value at Risk 95%
    expected_shortfall: float = 0.0

    # Status
    is_active: bool = True
    performance_tier: PerformanceTier = PerformanceTier.STANDARD
    trades_today: int = 0
    last_trade_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "target_allocation_pct": self.target_allocation_pct,
            "current_allocation_pct": self.current_allocation_pct,
            "allocated_capital": self.allocated_capital,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "performance_tier": self.performance_tier.value,
            "is_active": self.is_active
        }


@dataclass
class CompoundingState:
    """Current compounding state"""
    mode: CompoundingMode = CompoundingMode.FULL_REINVEST

    # Capital tracking
    initial_capital: float = 0.0
    current_capital: float = 0.0
    peak_capital: float = 0.0

    # Profit tracking
    total_profit: float = 0.0
    reinvested_profit: float = 0.0
    withdrawn_profit: float = 0.0
    reserved_profit: float = 0.0

    # Targets
    next_milestone: float = 0.0
    growth_target_pct: float = 0.0

    # Rates
    reinvestment_rate: float = 1.0          # % of profit to reinvest
    withdrawal_rate: float = 0.0            # % of profit to withdraw
    reserve_rate: float = 0.0               # % of profit to reserve

    # Timing
    last_compound_time: float = 0.0
    compound_frequency_hours: float = 24.0

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode.value,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "total_profit": self.total_profit,
            "reinvested_profit": self.reinvested_profit,
            "withdrawn_profit": self.withdrawn_profit,
            "reserved_profit": self.reserved_profit,
            "growth_pct": ((self.current_capital - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0,
            "next_milestone": self.next_milestone
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for tier calculation"""
    # Returns
    total_return_pct: float = 0.0
    daily_return_pct: float = 0.0
    weekly_return_pct: float = 0.0
    monthly_return_pct: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Consistency
    win_rate: float = 0.0
    profit_factor: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Drawdown
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0

    # Activity
    total_trades: int = 0
    trades_per_day: float = 0.0
    avg_trade_duration_minutes: float = 0.0

    def calculate_tier_score(self) -> float:
        """Calculate a composite score for tier determination"""
        score = 0.0

        # Sharpe contribution (0-30 points)
        score += min(30, max(0, self.sharpe_ratio * 10))

        # Win rate contribution (0-20 points)
        score += self.win_rate * 20

        # Profit factor contribution (0-20 points)
        pf_score = min(2.0, self.profit_factor) / 2.0 * 20
        score += pf_score

        # Drawdown penalty (-30 to 0 points)
        dd_penalty = min(30, self.max_drawdown_pct)
        score -= dd_penalty

        # Consistency bonus (0-15 points)
        if self.consecutive_losses == 0:
            score += min(15, self.consecutive_wins * 3)
        else:
            score -= min(15, self.consecutive_losses * 5)

        # Activity bonus (0-15 points)
        if self.total_trades >= 50:
            score += 15
        elif self.total_trades >= 20:
            score += 10
        elif self.total_trades >= 10:
            score += 5

        return max(0, score)


# =============================================================================
# CAPITAL ALLOCATION FRAMEWORK
# =============================================================================

class CapitalAllocationFramework:
    """
    Multi-strategy capital allocation with dynamic rebalancing.

    Features:
    - Allocate capital across multiple strategies
    - Dynamic rebalancing based on performance
    - Reserve management for safety
    - Opportunity cost optimization
    """

    def __init__(
        self,
        total_capital: float,
        reserve_pct: float = 10.0,          # Keep 10% in reserve
        max_single_strategy_pct: float = 40.0,  # Max 40% to any single strategy
        rebalance_threshold_pct: float = 5.0,   # Rebalance if drift > 5%
        min_allocation_pct: float = 5.0     # Minimum allocation to active strategy
    ):
        self.total_capital = total_capital
        self.reserve_pct = reserve_pct
        self.max_single_strategy_pct = max_single_strategy_pct
        self.rebalance_threshold_pct = rebalance_threshold_pct
        self.min_allocation_pct = min_allocation_pct

        # Strategy allocations
        self.allocations: Dict[str, StrategyAllocation] = {}

        # Reserve tracking
        self.reserve_capital = total_capital * (reserve_pct / 100)
        self.deployable_capital = total_capital - self.reserve_capital

        # Allocation strategy
        self.allocation_strategy = AllocationStrategy.PERFORMANCE_WEIGHTED

        # Performance history
        self._strategy_returns: Dict[str, deque] = {}
        self._rebalance_history: deque = deque(maxlen=100)

        # Last rebalance
        self._last_rebalance_time = time.time()
        self._rebalance_cooldown_hours = 1.0

    def register_strategy(
        self,
        strategy_id: str,
        strategy_name: str,
        initial_allocation_pct: float = 0.0
    ):
        """Register a new strategy"""
        if strategy_id in self.allocations:
            logger.warning(f"Strategy {strategy_id} already registered")
            return

        allocation = StrategyAllocation(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            target_allocation_pct=initial_allocation_pct,
            current_allocation_pct=initial_allocation_pct,
            allocated_capital=self.deployable_capital * (initial_allocation_pct / 100)
        )

        self.allocations[strategy_id] = allocation
        self._strategy_returns[strategy_id] = deque(maxlen=1000)

        logger.info(f"Registered strategy: {strategy_name} with {initial_allocation_pct:.1f}% allocation")

    def update_strategy_performance(
        self,
        strategy_id: str,
        pnl: float,
        win: bool,
        sharpe: float = 0.0,
        volatility: float = 0.0
    ):
        """Update strategy performance metrics"""
        if strategy_id not in self.allocations:
            logger.warning(f"Unknown strategy: {strategy_id}")
            return

        alloc = self.allocations[strategy_id]

        # Update PnL
        alloc.total_pnl += pnl
        alloc.allocated_capital += pnl

        # Update win rate (rolling)
        returns = self._strategy_returns[strategy_id]
        returns.append({"pnl": pnl, "win": win, "timestamp": time.time()})

        if len(returns) > 0:
            wins = sum(1 for r in returns if r["win"])
            alloc.win_rate = wins / len(returns)

        # Update other metrics
        alloc.sharpe_ratio = sharpe
        alloc.volatility = volatility
        alloc.last_trade_time = time.time()
        alloc.trades_today += 1

        # Update current allocation percentage
        self._recalculate_allocations()

        # Check if rebalance needed
        if self._should_rebalance():
            self.rebalance()

    def _recalculate_allocations(self):
        """Recalculate current allocation percentages"""
        total_allocated = sum(a.allocated_capital for a in self.allocations.values())

        if total_allocated > 0:
            for alloc in self.allocations.values():
                alloc.current_allocation_pct = (alloc.allocated_capital / total_allocated) * 100

    def _should_rebalance(self) -> bool:
        """Check if rebalancing is needed"""
        # Check cooldown
        hours_since = (time.time() - self._last_rebalance_time) / 3600
        if hours_since < self._rebalance_cooldown_hours:
            return False

        # Check drift from targets
        for alloc in self.allocations.values():
            if not alloc.is_active:
                continue
            drift = abs(alloc.current_allocation_pct - alloc.target_allocation_pct)
            if drift > self.rebalance_threshold_pct:
                return True

        return False

    def calculate_target_allocations(self) -> Dict[str, float]:
        """Calculate target allocations based on strategy"""
        active_strategies = [a for a in self.allocations.values() if a.is_active]

        if not active_strategies:
            return {}

        targets = {}

        if self.allocation_strategy == AllocationStrategy.EQUAL_WEIGHT:
            # Equal allocation to all active strategies
            equal_pct = 100.0 / len(active_strategies)
            for alloc in active_strategies:
                targets[alloc.strategy_id] = min(equal_pct, self.max_single_strategy_pct)

        elif self.allocation_strategy == AllocationStrategy.RISK_PARITY:
            # Allocate inversely proportional to volatility
            total_inv_vol = sum(1.0 / (a.volatility + 0.001) for a in active_strategies)
            for alloc in active_strategies:
                inv_vol = 1.0 / (alloc.volatility + 0.001)
                pct = (inv_vol / total_inv_vol) * 100
                targets[alloc.strategy_id] = min(pct, self.max_single_strategy_pct)

        elif self.allocation_strategy == AllocationStrategy.PERFORMANCE_WEIGHTED:
            # Weight by Sharpe ratio and win rate
            scores = {}
            for alloc in active_strategies:
                score = max(0.1, alloc.sharpe_ratio * 0.5 + alloc.win_rate * 0.5)
                scores[alloc.strategy_id] = score

            total_score = sum(scores.values())
            for alloc in active_strategies:
                pct = (scores[alloc.strategy_id] / total_score) * 100
                targets[alloc.strategy_id] = min(pct, self.max_single_strategy_pct)

        elif self.allocation_strategy == AllocationStrategy.MOMENTUM:
            # Allocate more to recent winners
            for alloc in active_strategies:
                returns = self._strategy_returns.get(alloc.strategy_id, deque())
                recent = list(returns)[-20:]

                if recent:
                    recent_pnl = sum(r["pnl"] for r in recent)
                    momentum = 1.0 + (recent_pnl / (self.deployable_capital + 1)) * 10
                else:
                    momentum = 1.0

                scores[alloc.strategy_id] = max(0.1, momentum)

            total_score = sum(scores.values())
            for alloc in active_strategies:
                pct = (scores[alloc.strategy_id] / total_score) * 100
                targets[alloc.strategy_id] = min(pct, self.max_single_strategy_pct)

        else:  # KELLY_OPTIMAL
            # Use Kelly-based allocation (simplified)
            for alloc in active_strategies:
                kelly = alloc.win_rate - ((1 - alloc.win_rate) / max(0.1, alloc.sharpe_ratio))
                kelly = max(0, min(0.25, kelly))  # Cap at 25%
                targets[alloc.strategy_id] = min(kelly * 100, self.max_single_strategy_pct)

        # Normalize to sum to 100%
        total_target = sum(targets.values())
        if total_target > 0:
            for sid in targets:
                targets[sid] = (targets[sid] / total_target) * 100

        # Apply minimum allocation
        for sid in targets:
            if targets[sid] < self.min_allocation_pct:
                targets[sid] = self.min_allocation_pct

        return targets

    def rebalance(self):
        """Rebalance allocations to targets"""
        targets = self.calculate_target_allocations()

        if not targets:
            return

        rebalance_actions = []

        for strategy_id, target_pct in targets.items():
            alloc = self.allocations[strategy_id]
            current_pct = alloc.current_allocation_pct

            diff_pct = target_pct - current_pct
            diff_capital = self.deployable_capital * (diff_pct / 100)

            alloc.target_allocation_pct = target_pct
            alloc.allocated_capital += diff_capital

            rebalance_actions.append({
                "strategy_id": strategy_id,
                "from_pct": current_pct,
                "to_pct": target_pct,
                "capital_change": diff_capital
            })

        self._recalculate_allocations()
        self._last_rebalance_time = time.time()

        self._rebalance_history.append({
            "timestamp": time.time(),
            "actions": rebalance_actions
        })

        logger.info(f"Rebalanced {len(rebalance_actions)} strategies")

    def get_allocation(self, strategy_id: str) -> float:
        """Get current capital allocation for a strategy"""
        if strategy_id not in self.allocations:
            return 0.0
        return self.allocations[strategy_id].allocated_capital

    def get_allocation_pct(self, strategy_id: str) -> float:
        """Get current allocation percentage for a strategy"""
        if strategy_id not in self.allocations:
            return 0.0
        return self.allocations[strategy_id].current_allocation_pct

    def update_total_capital(self, new_capital: float):
        """Update total capital and recalculate allocations"""
        old_capital = self.total_capital
        self.total_capital = new_capital
        self.reserve_capital = new_capital * (self.reserve_pct / 100)
        self.deployable_capital = new_capital - self.reserve_capital

        # Scale all allocations proportionally
        scale = new_capital / old_capital if old_capital > 0 else 1.0
        for alloc in self.allocations.values():
            alloc.allocated_capital *= scale

        self._recalculate_allocations()

    def get_summary(self) -> Dict:
        """Get allocation summary"""
        return {
            "total_capital": self.total_capital,
            "reserve_capital": self.reserve_capital,
            "deployable_capital": self.deployable_capital,
            "allocation_strategy": self.allocation_strategy.value,
            "strategies": {
                sid: alloc.to_dict()
                for sid, alloc in self.allocations.items()
            },
            "last_rebalance": self._last_rebalance_time
        }


# =============================================================================
# COMPOUNDING RULES ENGINE
# =============================================================================

class CompoundingEngine:
    """
    Profit reinvestment and compounding strategy engine.

    Features:
    - Multiple compounding modes
    - Milestone-based profit taking
    - Growth targets
    - Reserve building
    """

    def __init__(
        self,
        initial_capital: float,
        mode: CompoundingMode = CompoundingMode.PARTIAL_REINVEST,
        reinvestment_rate: float = 0.7,     # Reinvest 70% of profits
        reserve_rate: float = 0.2,          # Keep 20% as reserve
        withdrawal_rate: float = 0.1,       # Withdraw 10%
        growth_target_pct: float = 100.0,   # Target 100% growth
        milestone_increment_pct: float = 25.0  # Milestone every 25% growth
    ):
        self.state = CompoundingState(
            mode=mode,
            initial_capital=initial_capital,
            current_capital=initial_capital,
            peak_capital=initial_capital,
            reinvestment_rate=reinvestment_rate,
            reserve_rate=reserve_rate,
            withdrawal_rate=withdrawal_rate,
            growth_target_pct=growth_target_pct,
            next_milestone=initial_capital * (1 + milestone_increment_pct / 100)
        )

        self.milestone_increment_pct = milestone_increment_pct

        # Profit history
        self._profit_history: deque = deque(maxlen=1000)
        self._milestone_history: List[Dict] = []

        # Compound interest tracking
        self._compound_periods = 0
        self._effective_annual_rate = 0.0

    def record_profit(self, profit: float, timestamp: Optional[float] = None):
        """Record a profit and apply compounding rules"""
        ts = timestamp or time.time()

        self._profit_history.append({
            "profit": profit,
            "timestamp": ts,
            "capital_before": self.state.current_capital
        })

        self.state.total_profit += profit

        # Apply compounding based on mode
        if profit > 0:
            self._apply_compounding(profit)
        else:
            # Losses always reduce capital
            self.state.current_capital += profit

        # Update peak
        if self.state.current_capital > self.state.peak_capital:
            self.state.peak_capital = self.state.current_capital

        # Check milestones
        self._check_milestones()

    def _apply_compounding(self, profit: float):
        """Apply compounding rules to profit"""
        mode = self.state.mode

        if mode == CompoundingMode.FULL_REINVEST:
            # Reinvest all profits
            reinvest = profit
            reserve = 0.0
            withdraw = 0.0

        elif mode == CompoundingMode.PARTIAL_REINVEST:
            # Split according to rates
            reinvest = profit * self.state.reinvestment_rate
            reserve = profit * self.state.reserve_rate
            withdraw = profit * self.state.withdrawal_rate

        elif mode == CompoundingMode.MILESTONE_WITHDRAW:
            # Reinvest until milestone, then withdraw
            if self.state.current_capital + profit >= self.state.next_milestone:
                # At milestone - withdraw excess
                excess = (self.state.current_capital + profit) - self.state.next_milestone
                withdraw = min(excess, profit * 0.5)  # Withdraw up to 50% of profit
                reinvest = profit - withdraw
                reserve = 0.0
            else:
                reinvest = profit
                reserve = 0.0
                withdraw = 0.0

        elif mode == CompoundingMode.TARGET_GROWTH:
            # Aggressive reinvest until target
            target_capital = self.state.initial_capital * (1 + self.state.growth_target_pct / 100)
            if self.state.current_capital >= target_capital:
                # Target reached - switch to partial reinvest
                reinvest = profit * 0.5
                withdraw = profit * 0.5
                reserve = 0.0
            else:
                reinvest = profit
                reserve = 0.0
                withdraw = 0.0

        else:  # HYBRID
            # Dynamic based on performance
            growth_pct = ((self.state.current_capital - self.state.initial_capital)
                         / self.state.initial_capital * 100)

            if growth_pct < 20:
                # Early stage - aggressive reinvest
                reinvest = profit * 0.9
                reserve = profit * 0.1
                withdraw = 0.0
            elif growth_pct < 50:
                # Growth stage - balanced
                reinvest = profit * 0.7
                reserve = profit * 0.2
                withdraw = profit * 0.1
            else:
                # Mature stage - profit taking
                reinvest = profit * 0.5
                reserve = profit * 0.2
                withdraw = profit * 0.3

        # Apply allocations
        self.state.current_capital += reinvest
        self.state.reinvested_profit += reinvest
        self.state.reserved_profit += reserve
        self.state.withdrawn_profit += withdraw

        self._compound_periods += 1

    def _check_milestones(self):
        """Check and record milestones"""
        if self.state.current_capital >= self.state.next_milestone:
            milestone = {
                "milestone_capital": self.state.next_milestone,
                "actual_capital": self.state.current_capital,
                "timestamp": time.time(),
                "growth_pct": ((self.state.current_capital - self.state.initial_capital)
                              / self.state.initial_capital * 100),
                "milestone_number": len(self._milestone_history) + 1
            }

            self._milestone_history.append(milestone)

            logger.info(
                f"Milestone {milestone['milestone_number']} reached! "
                f"Capital: ${self.state.current_capital:.2f} "
                f"(+{milestone['growth_pct']:.1f}%)"
            )

            # Set next milestone
            self.state.next_milestone = self.state.current_capital * (1 + self.milestone_increment_pct / 100)

    def calculate_cagr(self) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(self._profit_history) < 2:
            return 0.0

        first_entry = self._profit_history[0]
        last_entry = self._profit_history[-1]

        time_diff_years = (last_entry["timestamp"] - first_entry["timestamp"]) / (365.25 * 24 * 3600)

        if time_diff_years <= 0:
            return 0.0

        initial = self.state.initial_capital
        final = self.state.current_capital

        if initial <= 0:
            return 0.0

        # CAGR = (Final/Initial)^(1/years) - 1
        cagr = math.pow(final / initial, 1 / time_diff_years) - 1

        return cagr * 100  # Return as percentage

    def calculate_effective_compound_rate(self) -> float:
        """Calculate effective compound rate"""
        if self._compound_periods == 0:
            return 0.0

        total_return = (self.state.current_capital - self.state.initial_capital) / self.state.initial_capital

        # Effective rate per period
        rate_per_period = math.pow(1 + total_return, 1 / self._compound_periods) - 1

        return rate_per_period * 100

    def get_optimal_reinvestment_rate(self) -> float:
        """Calculate optimal reinvestment rate based on performance"""
        if len(self._profit_history) < 20:
            return 0.7  # Default

        profits = [p["profit"] for p in self._profit_history]

        # Calculate metrics
        total_profit = sum(profits)
        avg_profit = total_profit / len(profits)

        wins = sum(1 for p in profits if p > 0)
        win_rate = wins / len(profits)

        # Higher win rate -> higher reinvestment
        # Lower volatility -> higher reinvestment
        import statistics
        volatility = statistics.stdev(profits) if len(profits) > 1 else 0
        avg_volatility = abs(avg_profit) if avg_profit != 0 else 1
        vol_ratio = volatility / avg_volatility

        # Base rate
        optimal = 0.5

        # Adjust for win rate
        if win_rate > 0.7:
            optimal += 0.2
        elif win_rate > 0.6:
            optimal += 0.1
        elif win_rate < 0.4:
            optimal -= 0.2

        # Adjust for volatility
        if vol_ratio < 0.5:
            optimal += 0.1
        elif vol_ratio > 2:
            optimal -= 0.15

        return max(0.3, min(0.95, optimal))

    def project_growth(self, periods: int, avg_return_per_period: float) -> List[float]:
        """Project capital growth over periods"""
        projections = [self.state.current_capital]
        capital = self.state.current_capital

        for _ in range(periods):
            profit = capital * (avg_return_per_period / 100)
            reinvested = profit * self.state.reinvestment_rate
            capital += reinvested
            projections.append(capital)

        return projections

    def get_state(self) -> Dict:
        """Get current compounding state"""
        return {
            **self.state.to_dict(),
            "cagr": self.calculate_cagr(),
            "effective_compound_rate": self.calculate_effective_compound_rate(),
            "optimal_reinvestment_rate": self.get_optimal_reinvestment_rate(),
            "milestones_reached": len(self._milestone_history),
            "compound_periods": self._compound_periods
        }


# =============================================================================
# PERFORMANCE-BASED SIZING
# =============================================================================

class PerformanceBasedSizer:
    """
    Performance tier-based position sizing.

    Features:
    - Tier system based on performance
    - Adaptive position limits
    - Reward consistency
    - Penalty for drawdowns
    """

    # Tier thresholds (score ranges)
    TIER_THRESHOLDS = {
        PerformanceTier.ELITE: 80,
        PerformanceTier.ADVANCED: 60,
        PerformanceTier.STANDARD: 40,
        PerformanceTier.PROBATION: 20,
        PerformanceTier.RESTRICTED: 0
    }

    # Position size multipliers by tier
    TIER_MULTIPLIERS = {
        PerformanceTier.ELITE: 1.5,         # 150% of base
        PerformanceTier.ADVANCED: 1.2,      # 120% of base
        PerformanceTier.STANDARD: 1.0,      # 100% of base
        PerformanceTier.PROBATION: 0.6,     # 60% of base
        PerformanceTier.RESTRICTED: 0.3     # 30% of base
    }

    # Max position by tier (as % of capital)
    TIER_MAX_POSITION = {
        PerformanceTier.ELITE: 15.0,
        PerformanceTier.ADVANCED: 12.0,
        PerformanceTier.STANDARD: 10.0,
        PerformanceTier.PROBATION: 5.0,
        PerformanceTier.RESTRICTED: 2.0
    }

    def __init__(
        self,
        base_position_pct: float = 5.0,
        evaluation_window: int = 50,        # Trades to evaluate
        tier_update_frequency: int = 10     # Update tier every N trades
    ):
        self.base_position_pct = base_position_pct
        self.evaluation_window = evaluation_window
        self.tier_update_frequency = tier_update_frequency

        # Current state
        self.current_tier = PerformanceTier.STANDARD
        self.current_score = 50.0
        self.metrics = PerformanceMetrics()

        # History
        self._trade_history: deque = deque(maxlen=1000)
        self._tier_history: deque = deque(maxlen=100)
        self._trades_since_update = 0

    def record_trade(
        self,
        pnl: float,
        pnl_pct: float,
        win: bool,
        duration_minutes: float = 0.0
    ):
        """Record a trade and update metrics"""
        self._trade_history.append({
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "win": win,
            "duration_minutes": duration_minutes,
            "timestamp": time.time()
        })

        self._trades_since_update += 1

        # Update metrics
        self._update_metrics()

        # Update tier if needed
        if self._trades_since_update >= self.tier_update_frequency:
            self._update_tier()
            self._trades_since_update = 0

    def _update_metrics(self):
        """Update performance metrics from trade history"""
        trades = list(self._trade_history)

        if not trades:
            return

        recent = trades[-self.evaluation_window:] if len(trades) >= self.evaluation_window else trades

        # Calculate returns
        pnls = [t["pnl_pct"] for t in recent]
        self.metrics.total_return_pct = sum(pnls)

        # Daily/weekly/monthly returns (approximated)
        if len(trades) >= 1:
            first_ts = trades[0]["timestamp"]
            last_ts = trades[-1]["timestamp"]
            days = max(1, (last_ts - first_ts) / 86400)

            self.metrics.daily_return_pct = self.metrics.total_return_pct / days
            self.metrics.weekly_return_pct = self.metrics.daily_return_pct * 7
            self.metrics.monthly_return_pct = self.metrics.daily_return_pct * 30

        # Win rate and profit factor
        wins = [t for t in recent if t["win"]]
        losses = [t for t in recent if not t["win"]]

        self.metrics.win_rate = len(wins) / len(recent) if recent else 0

        total_wins = sum(t["pnl"] for t in wins)
        total_losses = abs(sum(t["pnl"] for t in losses))
        self.metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Consecutive wins/losses
        streak = 0
        streak_type = None
        for trade in reversed(recent):
            if streak_type is None:
                streak_type = trade["win"]
                streak = 1
            elif trade["win"] == streak_type:
                streak += 1
            else:
                break

        if streak_type:
            self.metrics.consecutive_wins = streak
            self.metrics.consecutive_losses = 0
        else:
            self.metrics.consecutive_wins = 0
            self.metrics.consecutive_losses = streak

        # Risk metrics
        if len(pnls) > 1:
            import statistics
            returns_std = statistics.stdev(pnls)
            returns_mean = statistics.mean(pnls)

            self.metrics.sharpe_ratio = (returns_mean / returns_std * math.sqrt(252)) if returns_std > 0 else 0

            # Sortino (downside deviation)
            downside = [r for r in pnls if r < 0]
            if downside:
                downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
                self.metrics.sortino_ratio = (returns_mean / downside_std * math.sqrt(252)) if downside_std > 0 else 0

        # Drawdown
        peak = 0
        max_dd = 0
        running = 0

        for t in recent:
            running += t["pnl_pct"]
            if running > peak:
                peak = running
            dd = (peak - running)
            if dd > max_dd:
                max_dd = dd

        self.metrics.current_drawdown_pct = peak - running if running < peak else 0
        self.metrics.max_drawdown_pct = max_dd

        # Activity
        self.metrics.total_trades = len(trades)

        if len(trades) >= 2:
            time_span_days = (trades[-1]["timestamp"] - trades[0]["timestamp"]) / 86400
            self.metrics.trades_per_day = len(trades) / max(1, time_span_days)

        durations = [t["duration_minutes"] for t in recent if t["duration_minutes"] > 0]
        if durations:
            self.metrics.avg_trade_duration_minutes = sum(durations) / len(durations)

    def _update_tier(self):
        """Update performance tier based on current metrics"""
        old_tier = self.current_tier
        self.current_score = self.metrics.calculate_tier_score()

        # Determine tier from score
        if self.current_score >= self.TIER_THRESHOLDS[PerformanceTier.ELITE]:
            self.current_tier = PerformanceTier.ELITE
        elif self.current_score >= self.TIER_THRESHOLDS[PerformanceTier.ADVANCED]:
            self.current_tier = PerformanceTier.ADVANCED
        elif self.current_score >= self.TIER_THRESHOLDS[PerformanceTier.STANDARD]:
            self.current_tier = PerformanceTier.STANDARD
        elif self.current_score >= self.TIER_THRESHOLDS[PerformanceTier.PROBATION]:
            self.current_tier = PerformanceTier.PROBATION
        else:
            self.current_tier = PerformanceTier.RESTRICTED

        # Record tier change
        if old_tier != self.current_tier:
            self._tier_history.append({
                "from_tier": old_tier.value,
                "to_tier": self.current_tier.value,
                "score": self.current_score,
                "timestamp": time.time()
            })

            logger.info(
                f"Performance tier changed: {old_tier.value} -> {self.current_tier.value} "
                f"(score: {self.current_score:.1f})"
            )

    def get_position_multiplier(self) -> float:
        """Get position size multiplier for current tier"""
        return self.TIER_MULTIPLIERS.get(self.current_tier, 1.0)

    def get_max_position_pct(self) -> float:
        """Get maximum position size for current tier"""
        return self.TIER_MAX_POSITION.get(self.current_tier, 10.0)

    def calculate_position_size(
        self,
        capital: float,
        kelly_recommendation: float = 1.0,
        opportunity_confidence: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate position size based on tier.

        Returns:
            (position_size_usd, position_pct)
        """
        # Base position
        base_pct = self.base_position_pct

        # Apply tier multiplier
        tier_mult = self.get_position_multiplier()

        # Apply Kelly recommendation (capped at 2x)
        kelly_mult = min(2.0, kelly_recommendation)

        # Apply confidence
        conf_mult = opportunity_confidence

        # Calculate final percentage
        final_pct = base_pct * tier_mult * kelly_mult * conf_mult

        # Apply tier max cap
        max_pct = self.get_max_position_pct()
        final_pct = min(final_pct, max_pct)

        # Calculate absolute size
        position_size = capital * (final_pct / 100)

        return position_size, final_pct

    def get_sizing_summary(self) -> Dict:
        """Get comprehensive sizing summary"""
        return {
            "current_tier": self.current_tier.value,
            "tier_score": self.current_score,
            "position_multiplier": self.get_position_multiplier(),
            "max_position_pct": self.get_max_position_pct(),
            "metrics": {
                "win_rate": self.metrics.win_rate,
                "profit_factor": self.metrics.profit_factor,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "max_drawdown_pct": self.metrics.max_drawdown_pct,
                "consecutive_wins": self.metrics.consecutive_wins,
                "consecutive_losses": self.metrics.consecutive_losses,
                "total_trades": self.metrics.total_trades
            },
            "tier_history": list(self._tier_history)[-5:]
        }


# =============================================================================
# INTEGRATED CAPITAL EFFICIENCY MANAGER
# =============================================================================

class CapitalEfficiencyManager:
    """
    Integrated capital efficiency manager combining all components.

    Provides unified interface for:
    - Capital allocation across strategies
    - Profit compounding
    - Performance-based sizing
    """

    def __init__(
        self,
        initial_capital: float,
        reserve_pct: float = 10.0,
        compounding_mode: CompoundingMode = CompoundingMode.PARTIAL_REINVEST,
        reinvestment_rate: float = 0.7
    ):
        # Initialize components
        self.allocation_framework = CapitalAllocationFramework(
            total_capital=initial_capital,
            reserve_pct=reserve_pct
        )

        self.compounding_engine = CompoundingEngine(
            initial_capital=initial_capital,
            mode=compounding_mode,
            reinvestment_rate=reinvestment_rate
        )

        self.performance_sizer = PerformanceBasedSizer()

        # Register default arbitrage strategy
        self.allocation_framework.register_strategy(
            strategy_id="arbitrage",
            strategy_name="Cross-Venue Arbitrage",
            initial_allocation_pct=100.0
        )

        # Tracking
        self._total_trades = 0
        self._last_update_time = time.time()

    def record_trade(
        self,
        strategy_id: str,
        pnl: float,
        pnl_pct: float,
        win: bool,
        duration_minutes: float = 0.0,
        sharpe: float = 0.0,
        volatility: float = 0.0
    ):
        """Record a trade across all components"""
        # Update allocation framework
        self.allocation_framework.update_strategy_performance(
            strategy_id=strategy_id,
            pnl=pnl,
            win=win,
            sharpe=sharpe,
            volatility=volatility
        )

        # Update compounding engine
        self.compounding_engine.record_profit(pnl)

        # Update performance sizer
        self.performance_sizer.record_trade(
            pnl=pnl,
            pnl_pct=pnl_pct,
            win=win,
            duration_minutes=duration_minutes
        )

        self._total_trades += 1
        self._last_update_time = time.time()

        # Sync capital across components
        self._sync_capital()

    def _sync_capital(self):
        """Synchronize capital across components"""
        current_capital = self.compounding_engine.state.current_capital
        self.allocation_framework.update_total_capital(current_capital)

    def get_position_recommendation(
        self,
        strategy_id: str,
        kelly_fraction: float = 1.0,
        opportunity_confidence: float = 1.0
    ) -> Dict:
        """Get comprehensive position recommendation"""
        # Get allocation for strategy
        allocation = self.allocation_framework.get_allocation(strategy_id)

        # Get performance-based sizing
        position_size, position_pct = self.performance_sizer.calculate_position_size(
            capital=allocation,
            kelly_recommendation=kelly_fraction,
            opportunity_confidence=opportunity_confidence
        )

        return {
            "strategy_id": strategy_id,
            "allocated_capital": allocation,
            "recommended_position_size": position_size,
            "recommended_position_pct": position_pct,
            "performance_tier": self.performance_sizer.current_tier.value,
            "tier_multiplier": self.performance_sizer.get_position_multiplier(),
            "max_position_pct": self.performance_sizer.get_max_position_pct()
        }

    def get_capital_stats(self) -> Dict:
        """Get comprehensive capital statistics"""
        return {
            "allocation": self.allocation_framework.get_summary(),
            "compounding": self.compounding_engine.get_state(),
            "performance_sizing": self.performance_sizer.get_sizing_summary(),
            "total_trades": self._total_trades,
            "last_update": self._last_update_time
        }

    def project_growth(self, months: int = 12) -> Dict:
        """Project capital growth"""
        current = self.compounding_engine.state.current_capital

        # Estimate average monthly return
        cagr = self.compounding_engine.calculate_cagr()
        monthly_return = cagr / 12 if cagr > 0 else 2.0  # Default 2% monthly

        projections = self.compounding_engine.project_growth(
            periods=months,
            avg_return_per_period=monthly_return
        )

        return {
            "current_capital": current,
            "projected_capital": projections[-1],
            "projected_growth_pct": ((projections[-1] - current) / current * 100),
            "monthly_projections": projections,
            "assumed_monthly_return": monthly_return,
            "cagr": cagr
        }


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_capital_efficiency():
    """Test the capital efficiency manager"""
    print("Testing Capital Efficiency Manager...\n")

    manager = CapitalEfficiencyManager(
        initial_capital=10000,
        reserve_pct=10.0,
        compounding_mode=CompoundingMode.PARTIAL_REINVEST,
        reinvestment_rate=0.7
    )

    # Simulate trades
    import random
    random.seed(42)

    print("Simulating 100 trades...")
    for i in range(100):
        # 60% win rate with varying profit
        win = random.random() < 0.6
        if win:
            pnl = random.uniform(50, 200)
        else:
            pnl = random.uniform(-100, -30)

        pnl_pct = pnl / 10000 * 100

        manager.record_trade(
            strategy_id="arbitrage",
            pnl=pnl,
            pnl_pct=pnl_pct,
            win=win,
            duration_minutes=random.uniform(1, 30),
            sharpe=random.uniform(0.5, 2.0),
            volatility=random.uniform(0.01, 0.05)
        )

    # Get results
    stats = manager.get_capital_stats()

    print("\n" + "=" * 60)
    print("CAPITAL EFFICIENCY REPORT")
    print("=" * 60)

    print("\n--- Allocation ---")
    alloc = stats["allocation"]
    print(f"Total Capital: ${alloc['total_capital']:.2f}")
    print(f"Reserve: ${alloc['reserve_capital']:.2f}")
    print(f"Deployable: ${alloc['deployable_capital']:.2f}")

    print("\n--- Compounding ---")
    comp = stats["compounding"]
    print(f"Initial: ${comp['initial_capital']:.2f}")
    print(f"Current: ${comp['current_capital']:.2f}")
    print(f"Growth: {comp['growth_pct']:.1f}%")
    print(f"CAGR: {comp['cagr']:.1f}%")
    print(f"Reinvested: ${comp['reinvested_profit']:.2f}")
    print(f"Milestones: {comp['milestones_reached']}")

    print("\n--- Performance Tier ---")
    perf = stats["performance_sizing"]
    print(f"Current Tier: {perf['current_tier']}")
    print(f"Tier Score: {perf['tier_score']:.1f}")
    print(f"Position Multiplier: {perf['position_multiplier']:.2f}x")
    print(f"Max Position: {perf['max_position_pct']:.1f}%")

    print("\n--- Metrics ---")
    metrics = perf["metrics"]
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")

    print("\n--- Position Recommendation ---")
    rec = manager.get_position_recommendation(
        strategy_id="arbitrage",
        kelly_fraction=1.2,
        opportunity_confidence=0.8
    )
    print(f"Allocated Capital: ${rec['allocated_capital']:.2f}")
    print(f"Recommended Size: ${rec['recommended_position_size']:.2f}")
    print(f"Position %: {rec['recommended_position_pct']:.1f}%")

    print("\n--- 12-Month Projection ---")
    proj = manager.project_growth(months=12)
    print(f"Current: ${proj['current_capital']:.2f}")
    print(f"Projected: ${proj['projected_capital']:.2f}")
    print(f"Projected Growth: {proj['projected_growth_pct']:.1f}%")

    print("\nTest complete!")


if __name__ == "__main__":
    test_capital_efficiency()
