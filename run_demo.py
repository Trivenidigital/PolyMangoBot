"""
Demonstration Script for PolyMangoBot Edge-Based Trading System
================================================================

This script demonstrates the edge-focused trading strategies and generates
sample output for documentation purposes.
"""

import asyncio
import numpy as np
import time
import json
from datetime import datetime

# Import strategies
from edge_strategies import (
    EdgeStrategyEnsemble,
    EnhancedDirectionalStrategy,
    StatisticalArbitrageStrategy,
    IlliquidMarketStrategy,
    VolatilityRegimeStrategy,
    EdgeType,
    SignalStrength
)

from realistic_trading_engine import (
    RealisticTradingEngine,
    RealisticEngineConfig,
    RiskLevel
)


def generate_trending_candles(base_price: float, count: int, trend: float = 0.003, volatility: float = 0.01):
    """Generate candles with a clear trend for demonstration"""
    candles = []
    price = base_price
    timestamp = time.time() - count * 900

    for i in range(count):
        # Strong trend with some noise
        change = trend + np.random.randn() * volatility
        open_price = price
        close_price = price * (1 + change)
        high = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.002))
        low = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.002))
        # Volume increases with trend strength
        volume = np.random.uniform(8000, 12000) * (1 + abs(change) * 5)

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


def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


async def demo_individual_strategies():
    """Demonstrate each strategy individually"""
    print_header("INDIVIDUAL STRATEGY DEMONSTRATIONS")

    results = {}

    # 1. Enhanced Directional Strategy
    print("\n[1] ENHANCED DIRECTIONAL STRATEGY")
    print("-" * 50)
    print("Edge: Signal quality through multi-timeframe analysis")
    print("Timeframe: 15min - 4hr")
    print("Focus: Being RIGHT, not fast")

    strategy = EnhancedDirectionalStrategy(
        min_trend_strength=0.3,  # Lower for demo
        require_volume_confirmation=True,
        require_multi_timeframe=False  # Disable for simpler demo
    )

    # Generate trending data
    btc_15m = generate_trending_candles(42000, 100, trend=0.004)
    btc_1h = generate_trending_candles(42000, 50, trend=0.005)
    btc_4h = generate_trending_candles(42000, 30, trend=0.006)

    data = {
        "candles_15m": btc_15m,
        "candles_1h": btc_1h,
        "candles_4h": btc_4h
    }

    signal = strategy.generate_signal("BTC", data)

    if signal:
        print(f"\n[OK] Signal Generated!")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Entry: ${signal.entry_price:,.2f}")
        print(f"   Target: ${signal.target_price:,.2f}")
        print(f"   Stop: ${signal.stop_loss:,.2f}")
        print(f"   Expected Return: {signal.expected_return_pct:.2f}%")
        print(f"   Win Probability: {signal.win_probability:.1%}")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")
        print(f"   Signal Quality: {signal.signal_quality:.2f}")
        print(f"   Kelly Fraction: {signal.kelly_fraction:.2%}")
        print(f"   Suggested Position: {signal.suggested_position_pct:.2%} of capital")
        print(f"   Contributing Factors:")
        for factor in signal.contributing_factors:
            print(f"      - {factor}")

        results["directional"] = signal.to_dict()
    else:
        print("\n[!] No signal (conditions not met)")
        results["directional"] = None

    # 2. Statistical Arbitrage Strategy
    print("\n[2] 2. STATISTICAL ARBITRAGE STRATEGY")
    print("-" * 50)
    print("Edge: Mean reversion of correlated pairs")
    print("Timeframe: Hours to days")
    print("Focus: Statistical patterns, not speed")

    strategy = StatisticalArbitrageStrategy(
        zscore_entry=1.5,  # Lower for demo
        zscore_exit=0.3,
        lookback_periods=80,
        min_correlation=0.5
    )

    # Generate correlated price data with divergence
    np.random.seed(123)
    base_btc = 42000
    base_eth = 2300

    btc_prices = []
    eth_prices = []

    for i in range(100):
        # Correlated movement with some divergence
        common_factor = np.random.randn() * 0.01
        btc_prices.append(base_btc * (1 + common_factor + np.random.randn() * 0.005))
        # ETH diverges more towards the end (creates z-score opportunity)
        divergence = 0.002 * (i / 100) if i > 50 else 0
        eth_prices.append(base_eth * (1 + common_factor + np.random.randn() * 0.005 - divergence))
        base_btc = btc_prices[-1]
        base_eth = eth_prices[-1]

    data = {
        "pair_prices": {
            "BTC": btc_prices,
            "ETH": eth_prices
        }
    }

    signal = strategy.generate_signal("BTC/ETH", data)

    if signal:
        print(f"\n[OK] Signal Generated!")
        print(f"   Pair: {signal.symbol}")
        print(f"   Direction: {signal.direction.upper()} (relative position)")
        print(f"   Expected Return: {signal.expected_return_pct:.2f}%")
        print(f"   Win Probability: {signal.win_probability:.1%}")
        print(f"   Signal Quality: {signal.signal_quality:.2f}")
        print(f"   Contributing Factors:")
        for factor in signal.contributing_factors:
            print(f"      - {factor}")

        results["stat_arb"] = signal.to_dict()
    else:
        print("\n[!] No signal (z-score not extreme enough)")
        results["stat_arb"] = None

    # 3. Illiquid Market Strategy
    print("\n[3] 3. ILLIQUID MARKET STRATEGY")
    print("-" * 50)
    print("Edge: Wide spreads in markets HFT avoids")
    print("Timeframe: Minutes to hours")
    print("Focus: Markets with low competition")

    strategy = IlliquidMarketStrategy(
        min_spread_pct=2.0,  # 2% minimum spread
        max_daily_volume_usd=500000,
        min_spread_persistence_minutes=1  # Lower for demo
    )

    # Simulate spread persistence
    for _ in range(5):
        strategy._check_spread_persistence("SMALL_CAP", 4.5)

    data = {
        "orderbook": {
            "best_bid": 1.45,
            "best_ask": 1.52,  # 4.8% spread
            "bid_sizes": [5000, 3000, 2000],
            "ask_sizes": [4000, 3500, 2500]
        },
        "daily_volume": 75000  # Low volume
    }

    signal = strategy.generate_signal("SMALL_CAP", data)

    if signal:
        print(f"\n[OK] Signal Generated!")
        print(f"   Symbol: {signal.symbol}")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Entry: ${signal.entry_price:.4f}")
        print(f"   Target: ${signal.target_price:.4f}")
        print(f"   Expected Return: {signal.expected_return_pct:.2f}%")
        print(f"   Win Probability: {signal.win_probability:.1%}")
        print(f"   Contributing Factors:")
        for factor in signal.contributing_factors:
            print(f"      - {factor}")

        results["illiquid"] = signal.to_dict()
    else:
        print("\n[!] No signal")
        results["illiquid"] = None

    # 4. Volatility Regime Strategy
    print("\n[4] 4. VOLATILITY REGIME STRATEGY")
    print("-" * 50)
    print("Edge: Adapting to market conditions")
    print("Timeframe: Varies by regime")
    print("Focus: Right strategy for right conditions")

    strategy = VolatilityRegimeStrategy()

    # Generate low volatility data with mean reversion setup
    candles = []
    price = 100
    timestamp = time.time() - 100 * 900

    # Low volatility, price near upper band
    for i in range(100):
        change = np.random.randn() * 0.003  # Low volatility
        if i > 80:  # Push price high in recent candles
            change += 0.002
        open_price = price
        close_price = price * (1 + change)
        high = max(open_price, close_price) * 1.001
        low = min(open_price, close_price) * 0.999
        volume = np.random.uniform(1000, 2000)

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

    data = {"candles_15m": candles}

    signal = strategy.generate_signal("TEST", data)

    if signal:
        print(f"\n[OK] Signal Generated!")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Expected Return: {signal.expected_return_pct:.2f}%")
        print(f"   Contributing Factors:")
        for factor in signal.contributing_factors:
            print(f"      - {factor}")

        results["volatility"] = signal.to_dict()
    else:
        print("\n[!] No signal (regime may be transitioning)")
        results["volatility"] = None

    return results


async def demo_full_engine():
    """Demonstrate the full trading engine"""
    print_header("REALISTIC TRADING ENGINE DEMONSTRATION")

    print("\n[*] Initializing Engine...")
    engine = RealisticTradingEngine(
        capital=10000.0,
        config=RealisticEngineConfig.moderate()
    )

    print(f"   Capital: $10,000")
    print(f"   Risk Level: MODERATE")
    print(f"   Max Daily Loss: {engine.config.max_daily_loss_pct}%")
    print(f"   Max Position Size: {engine.config.max_capital_per_trade_pct}%")
    print(f"   Min Signal Quality: {engine.config.min_signal_quality}")
    print(f"   Min Win Probability: {engine.config.min_win_probability}")
    print(f"   Min Risk/Reward: {engine.config.min_risk_reward}")

    # Load market data
    print("\n[+] Loading Market Data...")

    # BTC in uptrend
    btc_15m = generate_trending_candles(42000, 100, trend=0.003)
    btc_1h = generate_trending_candles(42000, 50, trend=0.004)
    btc_4h = generate_trending_candles(42000, 30, trend=0.005)

    engine.update_market_data(
        symbol="BTC",
        candles_15m=btc_15m,
        candles_1h=btc_1h,
        candles_4h=btc_4h,
        orderbook={
            "best_bid": btc_15m[-1]["close"] * 0.999,
            "best_ask": btc_15m[-1]["close"] * 1.001,
            "bid_sizes": [1.0, 2.0, 3.0],
            "ask_sizes": [1.5, 2.0, 2.5]
        },
        daily_volume=5000000
    )
    print("   [v] BTC loaded (uptrend)")

    # ETH in downtrend
    eth_15m = generate_trending_candles(2300, 100, trend=-0.002)
    eth_1h = generate_trending_candles(2300, 50, trend=-0.003)
    eth_4h = generate_trending_candles(2300, 30, trend=-0.003)

    engine.update_market_data(
        symbol="ETH",
        candles_15m=eth_15m,
        candles_1h=eth_1h,
        candles_4h=eth_4h,
        orderbook={
            "best_bid": eth_15m[-1]["close"] * 0.998,
            "best_ask": eth_15m[-1]["close"] * 1.002,
            "bid_sizes": [10.0, 20.0, 15.0],
            "ask_sizes": [12.0, 18.0, 25.0]
        },
        daily_volume=3000000
    )
    print("   [v] ETH loaded (downtrend)")

    # Run trading simulation
    print("\n[~] Running Trading Simulation (5 cycles)...")
    print("-" * 50)

    cycle_results = []
    np.random.seed(42)

    for cycle in range(5):
        # Update prices with movement
        for symbol in ["BTC", "ETH"]:
            candles = engine._market_data[symbol]["candles_15m"]
            last_price = candles[-1]["close"]
            trend = 0.003 if symbol == "BTC" else -0.002
            new_price = last_price * (1 + np.random.randn() * 0.01 + trend)

            new_candle = {
                "timestamp": time.time(),
                "open": last_price,
                "high": max(last_price, new_price) * 1.002,
                "low": min(last_price, new_price) * 0.998,
                "close": new_price,
                "volume": np.random.uniform(5000, 10000)
            }
            candles.append(new_candle)
            engine._price_cache[symbol] = new_price

        # Force signal check
        engine._last_signal_check = 0

        result = await engine.run_cycle()
        cycle_results.append(result)

        print(f"\n   Cycle {cycle + 1}:")
        print(f"      Signals found: {result['signals_found']}")
        print(f"      Positions opened: {result['positions_opened']}")
        print(f"      Positions closed: {result['positions_closed']}")
        print(f"      Cycle PnL: ${result['pnl']:.2f}")

    # Get final performance
    perf = engine.get_performance()

    print("\n" + "=" * 50)
    print("[#] FINAL PERFORMANCE REPORT")
    print("=" * 50)

    summary = perf["summary"]
    print(f"\n   Total Trades: {summary['total_trades']}")
    print(f"   Winning Trades: {summary['winning_trades']}")
    print(f"   Win Rate: {summary['win_rate']:.1f}%")
    print(f"   Total PnL: ${summary['total_pnl']:.2f}")
    print(f"   Avg PnL/Trade: ${summary['avg_pnl_per_trade']:.2f}")
    print(f"   Best Trade: ${summary['best_trade']:.2f}")
    print(f"   Worst Trade: ${summary['worst_trade']:.2f}")

    risk = perf["risk_status"]
    print(f"\n   Current Capital: ${risk['capital']:.2f}")
    print(f"   Drawdown: {risk['drawdown_pct']:.2f}%")
    print(f"   Daily PnL: ${risk['daily_pnl']:.2f}")
    print(f"   Exposure: {risk['exposure_pct']:.1f}%")
    print(f"   Open Positions: {risk['open_positions']}")

    return {
        "cycles": cycle_results,
        "performance": perf
    }


async def main():
    """Run full demonstration"""
    print("=" * 70)
    print("  PolyMangoBot - Edge-Based Trading System Demonstration")
    print("=" * 70)
    print("\nPhilosophy: We compete on SIGNAL QUALITY, not SPEED.")
    print("HFT is 100-1000x faster. Our edge is being RIGHT, not fast.")

    # Demo individual strategies
    strategy_results = await demo_individual_strategies()

    # Demo full engine
    engine_results = await demo_full_engine()

    print_header("DEMONSTRATION COMPLETE")

    print("""
Key Takeaways:
======================================================================

1. NO SPEED COMPETITION
   - We don't try to be faster than HFT (impossible)
   - Focus on 15-minute+ timeframes where speed doesn't matter

2. SIGNAL QUALITY IS THE EDGE
   - Multi-timeframe confirmation
   - Volume analysis
   - Statistical patterns

3. RISK MANAGEMENT IS PARAMOUNT
   - Daily loss limits
   - Position sizing via Kelly criterion
   - Consecutive loss protection

4. MULTIPLE UNCORRELATED EDGES
   - Directional (trend following)
   - Statistical arbitrage (mean reversion)
   - Illiquid markets (low competition)
   - Volatility adaptation

5. REALISTIC EXPECTATIONS
   - Win rate: 50-60% (not 90%)
   - Monthly returns: 5-15% (not 100%)
   - Drawdowns: 10-20% expected
""")

    return {
        "strategies": strategy_results,
        "engine": engine_results
    }


if __name__ == "__main__":
    results = asyncio.run(main())
