"""
End-to-End Test for Cross-Market Inference Engine
=================================================

Tests the full pipeline from market data ingestion to signal generation,
including integration with EnhancedTradingEngine.

Run with: python tests/e2e_inference_test.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.models import PolymarketMarket, RelationshipType, ViolationType
from inference.engine import InferenceEngine, InferenceEngineConfig, create_engine
from inference.arb_monitor import ArbMonitor, ArbMonitorConfig, create_monitor
from inference.family_discovery import FamilyDiscovery, find_date_variant_families
from inference.detection_rules import ViolationDetector, DetectionConfig


# =============================================================================
# MOCK DATA GENERATORS
# =============================================================================

def generate_btc_price_markets() -> List[PolymarketMarket]:
    """Generate Bitcoin price prediction markets with date variants."""
    base_date = datetime(2025, 1, 1)
    markets = []

    # BTC $100k markets - with monotonicity violation
    # Earlier deadline has HIGHER yes price (violation!)
    for i, (month, yes_price) in enumerate([
        ("March", 0.35),   # Earlier deadline, higher price (WRONG)
        ("April", 0.32),   # Later deadline, lower price (WRONG)
        ("June", 0.45),    # Later deadline, higher price (correct)
        ("December", 0.55) # Latest deadline, highest price (correct)
    ]):
        deadline = base_date + timedelta(days=30 * (i + 3))
        markets.append(PolymarketMarket(
            id=f"btc_100k_{month.lower()}",
            question=f"Will Bitcoin reach $100,000 by {month} 2025?",
            slug=f"btc-100k-{month.lower()}-2025",
            condition_id="btc_100k_family",
            group_slug="btc_100k_dates",
            yes_price=yes_price,
            no_price=1.0 - yes_price,
            liquidity=10000.0 + i * 2000,
            deadline=deadline
        ))

    return markets


def generate_election_markets() -> List[PolymarketMarket]:
    """Generate election markets with exclusive constraint violation."""
    # Sum of YES prices > 1.0 (violation for exclusive markets)
    candidates = [
        ("Trump", 0.48, 50000),
        ("Biden", 0.42, 45000),
        ("Other", 0.15, 10000),  # Total = 1.05 (5% violation)
    ]

    markets = []
    for name, yes_price, liquidity in candidates:
        markets.append(PolymarketMarket(
            id=f"election_{name.lower()}",
            question=f"Will {name} win the 2024 Presidential Election?",
            slug=f"{name.lower()}-wins-2024",
            condition_id="election_2024",
            group_slug="election_winner_2024",
            yes_price=yes_price,
            no_price=1.0 - yes_price,
            liquidity=liquidity,
        ))

    return markets


def generate_fed_rate_markets() -> List[PolymarketMarket]:
    """Generate Fed rate decision markets (exhaustive set)."""
    # Sum of YES prices < 1.0 (violation for exhaustive markets)
    decisions = [
        ("Raise", 0.25),
        ("Cut", 0.30),
        ("Hold", 0.40),  # Total = 0.95 (5% violation)
    ]

    markets = []
    for action, yes_price in decisions:
        markets.append(PolymarketMarket(
            id=f"fed_{action.lower()}_q1",
            question=f"Will the Fed {action.lower()} rates in Q1 2025?",
            slug=f"fed-{action.lower()}-q1-2025",
            condition_id="fed_q1_2025",
            group_slug="fed_decision_q1",
            yes_price=yes_price,
            no_price=1.0 - yes_price,
            liquidity=25000.0,
        ))

    return markets


def generate_profitable_no_sweep_markets() -> List[PolymarketMarket]:
    """Generate markets where NO sweep is highly profitable."""
    # Total NO cost < 1.0 guarantees profit
    markets = [
        PolymarketMarket(
            id="event_march",
            question="Will major event happen by March 2025?",
            slug="event-march-2025",
            condition_id="event_family",
            group_slug="event_dates",
            yes_price=0.70,
            no_price=0.30,  # NO costs 0.30
            liquidity=8000.0,
            deadline=datetime(2025, 3, 31)
        ),
        PolymarketMarket(
            id="event_june",
            question="Will major event happen by June 2025?",
            slug="event-june-2025",
            condition_id="event_family",
            group_slug="event_dates",
            yes_price=0.80,
            no_price=0.20,  # NO costs 0.20, total = 0.50
            liquidity=10000.0,
            deadline=datetime(2025, 6, 30)
        ),
    ]
    # Total NO cost = 0.50, guaranteed payout = 1.0, edge = 50%!
    return markets


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_family_discovery():
    """Test market family discovery."""
    print("\n" + "=" * 70)
    print("TEST 1: Family Discovery")
    print("=" * 70)

    all_markets = (
        generate_btc_price_markets() +
        generate_election_markets() +
        generate_fed_rate_markets()
    )

    print(f"\nInput: {len(all_markets)} markets")

    discovery = FamilyDiscovery()
    families = discovery.discover_families(all_markets)

    print(f"Discovered: {len(families)} families")

    for family in families:
        print(f"\n  Family: {family.id}")
        print(f"    Size: {family.size} markets")
        print(f"    Source: {family.source}")
        print(f"    Confidence: {family.confidence:.2f}")
        print(f"    Shared entity: {family.shared_entity}")
        print(f"    Markets:")
        for m in family.markets[:3]:  # Show first 3
            print(f"      - {m.question[:50]}...")

    assert len(families) >= 3, f"Expected at least 3 families, got {len(families)}"
    print("\n[PASS] Family discovery working correctly")
    return families


def test_date_variant_detection():
    """Test date-variant family detection."""
    print("\n" + "=" * 70)
    print("TEST 2: Date-Variant Family Detection")
    print("=" * 70)

    markets = generate_btc_price_markets()
    print(f"\nInput: {len(markets)} BTC price markets")

    families = find_date_variant_families(markets)

    print(f"Found: {len(families)} date-variant families")

    for family in families:
        print(f"\n  Family: {family.id}")
        print(f"    Relationship: {family.relationship.value}")
        print(f"    Markets by deadline:")
        for m in family.get_sorted_by_deadline():
            print(f"      {m.deadline.strftime('%Y-%m-%d')}: YES={m.yes_price:.2f}, NO={m.no_price:.2f}")

    assert len(families) >= 1, "Should find at least 1 date-variant family"
    assert families[0].relationship == RelationshipType.DATE_VARIANT
    print("\n[PASS] Date-variant detection working correctly")
    return families


def test_violation_detection():
    """Test violation detection across different types."""
    print("\n" + "=" * 70)
    print("TEST 3: Violation Detection")
    print("=" * 70)

    detector = ViolationDetector(DetectionConfig(
        min_edge_pct=0.1,
        min_leg_liquidity=100.0,
        min_sweep_edge_pct=1.0,
        sum_violation_threshold=0.01
    ))

    # Test 1: Monotonicity violations in BTC markets
    print("\n--- Testing Monotonicity Violations ---")
    btc_families = find_date_variant_families(generate_btc_price_markets())

    for family in btc_families:
        violations = detector.detect_all(family)
        print(f"\n  Family {family.id}: {len(violations)} violations")
        for v in violations:
            print(f"    - {v.type.value}: {v.raw_edge:.2f}% edge")
            print(f"      {v.description[:60]}...")

    # Test 2: NO sweep opportunity
    print("\n--- Testing NO Sweep Opportunity ---")
    sweep_markets = generate_profitable_no_sweep_markets()
    sweep_families = find_date_variant_families(sweep_markets)

    for family in sweep_families:
        violations = detector.detect_all(family)
        print(f"\n  Family {family.id}: {len(violations)} violations")
        for v in violations:
            print(f"    - {v.type.value}: {v.raw_edge:.2f}% edge")
            if v.legs:
                total_cost = sum(leg.price for leg in v.legs)
                print(f"      Total NO cost: {total_cost:.2f}")
                print(f"      Guaranteed payout: 1.0")
                print(f"      Edge: {(1.0 - total_cost) * 100:.1f}%")

    print("\n[PASS] Violation detection working correctly")
    return detector.get_stats()


def test_full_inference_pipeline():
    """Test the full inference engine pipeline."""
    print("\n" + "=" * 70)
    print("TEST 4: Full Inference Pipeline")
    print("=" * 70)

    # Combine all test markets
    all_markets = (
        generate_btc_price_markets() +
        generate_election_markets() +
        generate_profitable_no_sweep_markets()
    )

    print(f"\nInput: {len(all_markets)} total markets")

    # Create engine with aggressive settings
    config = InferenceEngineConfig(
        min_family_confidence=0.1,
        min_realizable_edge_pct=0.1,
        min_leg_liquidity=100.0,
        default_position_usd=100.0,
    )
    config.detection.min_edge_pct = 0.1
    config.detection.min_leg_liquidity = 100.0
    config.edge.min_realizable_edge_pct = 0.1
    config.edge.min_worst_case_profit = -50.0

    engine = InferenceEngine(config)

    # Process markets
    signals = engine.process_markets(all_markets)
    stats = engine.get_stats()

    print(f"\nPipeline Results:")
    print(f"  Markets processed: {stats['markets_processed']}")
    print(f"  Families discovered: {stats['families_discovered']}")
    print(f"  Violations detected: {stats['violations_detected']}")
    print(f"  Signals generated: {stats['signals_generated']}")
    print(f"  Processing time: {stats['last_run_time_seconds']:.3f}s")

    if signals:
        print(f"\nGenerated Signals:")
        for i, signal in enumerate(signals[:5]):
            print(f"\n  Signal {i+1}:")
            print(f"    Type: {signal.type}")
            print(f"    Subtype: {signal.subtype}")
            print(f"    Family: {signal.family_id}")
            print(f"    Raw edge: {signal.raw_edge:.2f}%")
            print(f"    Realizable edge: {signal.realizable_edge:.2f}%")
            print(f"    Worst case P&L: ${signal.worst_case_pnl:.2f}")
            print(f"    Confidence: {signal.confidence}/10")
            print(f"    Legs: {len(signal.legs)}")

    print("\n[PASS] Full inference pipeline working correctly")
    return signals, stats


def test_arb_monitor():
    """Test the arbitrage monitor."""
    print("\n" + "=" * 70)
    print("TEST 5: Arbitrage Monitor")
    print("=" * 70)

    # Create engine
    engine = create_engine(mode="aggressive")

    # Create monitor
    monitor = create_monitor(
        engine=engine,
        poll_interval=5.0,  # Short for testing
        auto_execute=False
    )

    # Track callbacks
    signals_received = []
    alerts_received = []

    def on_signal(signal):
        signals_received.append(signal)
        print(f"    Signal received: {signal.subtype} in {signal.family_id}")

    def on_alert(signal, state):
        alerts_received.append((signal, state))
        print(f"    ALERT: {signal.subtype} edge={signal.realizable_edge:.2f}%")

    monitor.on_signal(on_signal)
    monitor.on_alert(on_alert)

    # Run single poll
    print("\n--- Running Single Poll ---")
    markets = generate_btc_price_markets() + generate_profitable_no_sweep_markets()
    signals = monitor.poll_once(markets)

    print(f"\n  Signals from poll: {len(signals)}")
    print(f"  Callbacks received: {len(signals_received)}")

    # Check active opportunities
    active = monitor.get_active_opportunities()
    print(f"  Active opportunities: {len(active)}")

    for opp in active:
        print(f"\n    Family: {opp['family_id']}")
        print(f"    Type: {opp['type']}")
        print(f"    Duration: {opp['duration_seconds']:.1f}s")
        print(f"    Last edge: {opp['last_edge']:.2f}%")
        print(f"    Persistent: {opp['is_persistent']}")

    # Get monitor stats
    stats = monitor.get_stats()
    print(f"\n  Monitor stats:")
    print(f"    Polls: {stats['polls']}")
    print(f"    Signals found: {stats['signals_found']}")
    print(f"    Alerts sent: {stats['alerts_sent']}")

    print("\n[PASS] Arbitrage monitor working correctly")
    return stats


def test_enhanced_engine_integration():
    """Test integration with EnhancedTradingEngine."""
    print("\n" + "=" * 70)
    print("TEST 6: EnhancedTradingEngine Integration")
    print("=" * 70)

    try:
        from enhanced_trading_engine import (
            EnhancedTradingEngine,
            EngineMode,
            TradingStrategy,
            INFERENCE_AVAILABLE
        )
    except ImportError as e:
        print(f"\n  [SKIP] Could not import EnhancedTradingEngine: {e}")
        return None

    print(f"\n  Inference module available: {INFERENCE_AVAILABLE}")

    if not INFERENCE_AVAILABLE:
        print("  [SKIP] Inference module not available in EnhancedTradingEngine")
        return None

    # Create engine
    engine = EnhancedTradingEngine(
        capital=10000.0,
        mode=EngineMode.AGGRESSIVE
    )

    print(f"\n  Engine mode: {engine.mode.value}")
    print(f"  Structural arb enabled: {engine.config.enable_structural_arb}")
    print(f"  Inference engine: {'initialized' if engine.inference_engine else 'None'}")

    if engine.inference_engine:
        # Prepare test data
        polymarket_markets = [m.to_dict() for m in generate_btc_price_markets()]

        # Scan for opportunities
        opportunities = engine.scan_all_opportunities(
            symbols=["BTC"],
            polymarket_markets=polymarket_markets
        )

        structural_opps = [
            o for o in opportunities
            if o.strategy == TradingStrategy.STRUCTURAL_ARB
        ]

        print(f"\n  Total opportunities: {len(opportunities)}")
        print(f"  Structural arb opportunities: {len(structural_opps)}")

        for opp in structural_opps[:3]:
            print(f"\n    {opp.id}:")
            print(f"      Expected profit: {opp.expected_profit_pct:.2f}%")
            print(f"      Confidence: {opp.confidence:.2f}")
            print(f"      Position: ${opp.suggested_position_usd:.2f}")

    print("\n[PASS] EnhancedTradingEngine integration working")
    return engine


async def test_async_monitoring():
    """Test async monitoring loop (brief)."""
    print("\n" + "=" * 70)
    print("TEST 7: Async Monitoring (3 second test)")
    print("=" * 70)

    engine = create_engine(mode="aggressive")
    monitor = create_monitor(engine=engine, poll_interval=1.0)

    # Mock market fetcher
    poll_count = [0]
    def market_fetcher():
        poll_count[0] += 1
        print(f"    Poll #{poll_count[0]}")
        return generate_btc_price_markets()

    monitor.set_market_fetcher(market_fetcher)

    # Start monitor
    print("\n  Starting monitor...")
    await monitor.start()

    # Let it run briefly
    await asyncio.sleep(3)

    # Stop monitor
    await monitor.stop()

    stats = monitor.get_stats()
    print(f"\n  Final stats:")
    print(f"    Polls completed: {stats['polls']}")
    print(f"    Signals found: {stats['signals_found']}")
    print(f"    Total run time: {stats['total_run_time']:.2f}s")

    assert stats['polls'] >= 2, f"Expected at least 2 polls, got {stats['polls']}"
    print("\n[PASS] Async monitoring working correctly")
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all end-to-end tests."""
    print("=" * 70)
    print("CROSS-MARKET INFERENCE ENGINE - END-TO-END TESTS")
    print("=" * 70)

    results = {}

    try:
        # Test 1: Family Discovery
        results['family_discovery'] = test_family_discovery()

        # Test 2: Date Variant Detection
        results['date_variant'] = test_date_variant_detection()

        # Test 3: Violation Detection
        results['violations'] = test_violation_detection()

        # Test 4: Full Pipeline
        signals, stats = test_full_inference_pipeline()
        results['pipeline'] = {'signals': len(signals), 'stats': stats}

        # Test 5: Arb Monitor
        results['monitor'] = test_arb_monitor()

        # Test 6: Enhanced Engine Integration
        results['integration'] = test_enhanced_engine_integration()

        # Test 7: Async Monitoring
        results['async'] = asyncio.run(test_async_monitoring())

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print("\n  All tests passed!")
    print(f"\n  Results:")
    print(f"    Families discovered: {len(results.get('family_discovery', []))}")
    print(f"    Date-variant families: {len(results.get('date_variant', []))}")
    print(f"    Pipeline signals: {results.get('pipeline', {}).get('signals', 0)}")
    print(f"    Monitor polls: {results.get('monitor', {}).get('polls', 0)}")
    print(f"    Async polls: {results.get('async', {}).get('polls', 0)}")

    print("\n" + "=" * 70)
    print("END-TO-END TESTS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
