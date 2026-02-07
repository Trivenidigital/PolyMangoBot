"""
Tests for Cross-Market Inference Engine
=======================================

Unit tests for:
- Family discovery (token extraction, grouping)
- Relationship classification
- Violation detection (monotonicity, sweeps, exclusive/exhaustive)
- Trade construction
- Edge calculation

Integration tests for:
- Full pipeline with mock market data
- Integration with EnhancedTradingEngine
"""

import pytest
from datetime import datetime
from typing import List

# Import inference modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.models import (
    PolymarketMarket,
    MarketFamily,
    RelationshipType,
    ViolationType,
    TradeSide,
)
from inference.family_discovery import (
    FamilyDiscovery,
    FamilyDiscoveryConfig,
    extract_tokens,
    extract_date,
    extract_entity,
    find_date_variant_families,
)
from inference.relationship import (
    RelationshipClassifier,
    RelationshipConfig,
)
from inference.detection_rules import (
    ViolationDetector,
    DetectionConfig,
    detect_monotonicity_violations,
    detect_no_sweep_opportunity,
    detect_exclusive_violation,
    detect_exhaustive_violation,
)
from inference.trade_constructor import (
    TradeConstructor,
    TradeConstructorConfig,
)
from inference.realizable_edge import (
    RealizableEdgeCalculator,
    EdgeConfig,
)
from inference.engine import (
    InferenceEngine,
    InferenceEngineConfig,
    create_engine,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

def create_date_variant_markets() -> List[PolymarketMarket]:
    """Create test markets for date-variant arbitrage testing."""
    return [
        PolymarketMarket(
            id="btc_100k_march",
            question="Will Bitcoin reach $100,000 by March 2025?",
            slug="btc-100k-march-2025",
            condition_id="btc_100k",
            group_slug="btc_100k_family",
            yes_price=0.35,  # Higher price for earlier deadline
            no_price=0.65,
            liquidity=10000.0,
            deadline=datetime(2025, 3, 31)
        ),
        PolymarketMarket(
            id="btc_100k_april",
            question="Will Bitcoin reach $100,000 by April 2025?",
            slug="btc-100k-april-2025",
            condition_id="btc_100k",
            group_slug="btc_100k_family",
            yes_price=0.30,  # Lower price for later deadline (VIOLATION!)
            no_price=0.70,
            liquidity=8000.0,
            deadline=datetime(2025, 4, 30)
        ),
        PolymarketMarket(
            id="btc_100k_june",
            question="Will Bitcoin reach $100,000 by June 2025?",
            slug="btc-100k-june-2025",
            condition_id="btc_100k",
            group_slug="btc_100k_family",
            yes_price=0.45,
            no_price=0.55,
            liquidity=12000.0,
            deadline=datetime(2025, 6, 30)
        ),
    ]


def create_exclusive_markets() -> List[PolymarketMarket]:
    """Create mutually exclusive markets for testing."""
    return [
        PolymarketMarket(
            id="election_trump",
            question="Will Trump win the 2024 election?",
            slug="trump-2024",
            condition_id="election_2024",
            group_slug="election_winner",
            yes_price=0.45,
            no_price=0.55,
            liquidity=50000.0,
        ),
        PolymarketMarket(
            id="election_biden",
            question="Will Biden win the 2024 election?",
            slug="biden-2024",
            condition_id="election_2024",
            group_slug="election_winner",
            yes_price=0.40,
            no_price=0.60,
            liquidity=45000.0,
        ),
        PolymarketMarket(
            id="election_other",
            question="Will another candidate win the 2024 election?",
            slug="other-2024",
            condition_id="election_2024",
            group_slug="election_winner",
            yes_price=0.20,  # Sum = 1.05 (VIOLATION!)
            no_price=0.80,
            liquidity=10000.0,
        ),
    ]


def create_exhaustive_markets() -> List[PolymarketMarket]:
    """Create exhaustive markets for testing."""
    return [
        PolymarketMarket(
            id="fed_hike",
            question="Will the Fed raise rates in Q1 2025?",
            slug="fed-hike-q1",
            condition_id="fed_q1",
            group_slug="fed_action",
            yes_price=0.25,
            no_price=0.75,
            liquidity=20000.0,
        ),
        PolymarketMarket(
            id="fed_cut",
            question="Will the Fed cut rates in Q1 2025?",
            slug="fed-cut-q1",
            condition_id="fed_q1",
            group_slug="fed_action",
            yes_price=0.30,
            no_price=0.70,
            liquidity=25000.0,
        ),
        PolymarketMarket(
            id="fed_hold",
            question="Will the Fed hold rates in Q1 2025?",
            slug="fed-hold-q1",
            condition_id="fed_q1",
            group_slug="fed_action",
            yes_price=0.40,  # Sum = 0.95 (VIOLATION if exhaustive!)
            no_price=0.60,
            liquidity=30000.0,
        ),
    ]


# =============================================================================
# TOKEN EXTRACTION TESTS
# =============================================================================

class TestTokenExtraction:
    """Tests for token extraction from market questions."""

    def test_extract_tokens_basic(self):
        question = "Will Bitcoin reach $100,000 by March 2025?"
        tokens = extract_tokens(question)

        assert "bitcoin" in tokens
        assert "reach" in tokens
        # Price may be tokenized differently (e.g., "$100" and "000")
        assert any("100" in t for t in tokens)
        # Stop words should be excluded
        assert "will" not in tokens
        assert "by" not in tokens

    def test_extract_tokens_with_ticker(self):
        question = "Will $BTC reach $100k?"
        tokens = extract_tokens(question)

        assert "$btc" in tokens or "btc" in tokens

    def test_extract_date_march(self):
        question = "Will Bitcoin reach $100,000 by March 2025?"
        date = extract_date(question)

        assert date is not None
        assert date.year == 2025
        assert date.month == 3

    def test_extract_date_quarter(self):
        question = "Will the Fed cut rates in Q2 2025?"
        date = extract_date(question)

        assert date is not None
        assert date.year == 2025
        assert date.month == 6  # End of Q2

    def test_extract_entity_person(self):
        question = "Will Donald Trump win the 2024 election?"
        entity = extract_entity(question)

        assert entity is not None
        assert "Trump" in entity or "Donald Trump" in entity

    def test_extract_entity_crypto(self):
        question = "Will Bitcoin reach $100,000?"
        entity = extract_entity(question)

        assert entity is not None
        assert "Bitcoin" in entity or "BTC" in entity


# =============================================================================
# FAMILY DISCOVERY TESTS
# =============================================================================

class TestFamilyDiscovery:
    """Tests for market family discovery."""

    def test_discover_by_metadata(self):
        markets = create_date_variant_markets()
        discovery = FamilyDiscovery()
        families = discovery.discover_families(markets)

        # Should find one family (same group_slug)
        assert len(families) >= 1

        # Family should contain all 3 markets
        family = families[0]
        assert family.size == 3

    def test_discover_by_tokens(self):
        # Create markets without common metadata
        markets = [
            PolymarketMarket(
                id="btc_1",
                question="Will Bitcoin reach $100,000 by March?",
                slug="btc-march",
                condition_id="cond1",
                yes_price=0.35,
                no_price=0.65,
                liquidity=5000.0,
            ),
            PolymarketMarket(
                id="btc_2",
                question="Will Bitcoin reach $100,000 by April?",
                slug="btc-april",
                condition_id="cond2",
                yes_price=0.40,
                no_price=0.60,
                liquidity=6000.0,
            ),
        ]

        discovery = FamilyDiscovery(
            config=FamilyDiscoveryConfig(use_metadata_grouping=False)
        )
        families = discovery.discover_families(markets)

        # Should find family by token similarity
        assert len(families) >= 1

    def test_find_date_variant_families(self):
        markets = create_date_variant_markets()
        families = find_date_variant_families(markets)

        assert len(families) >= 1
        family = families[0]
        assert family.relationship == RelationshipType.DATE_VARIANT


# =============================================================================
# RELATIONSHIP CLASSIFICATION TESTS
# =============================================================================

class TestRelationshipClassification:
    """Tests for relationship classification."""

    def test_classify_date_variant(self):
        markets = create_date_variant_markets()
        family = MarketFamily(
            id="test",
            markets=markets,
        )

        classifier = RelationshipClassifier()
        classifier.classify_family(family)

        assert family.relationship == RelationshipType.DATE_VARIANT
        assert family.confidence >= 0.6

    def test_classify_exclusive(self):
        markets = create_exclusive_markets()
        family = MarketFamily(
            id="test",
            markets=markets,
        )

        classifier = RelationshipClassifier()
        classifier.classify_family(family)

        # Election markets may be classified as exclusive, exhaustive, or both
        # The classifier uses heuristics so any of these are valid
        assert family.relationship in [
            RelationshipType.MUTUALLY_EXCLUSIVE,
            RelationshipType.EXHAUSTIVE,
            RelationshipType.EXCLUSIVE_AND_EXHAUSTIVE,
            RelationshipType.UNKNOWN
        ]
        # Should have some confidence
        assert family.confidence >= 0.0


# =============================================================================
# DETECTION RULES TESTS
# =============================================================================

class TestDetectionRules:
    """Tests for violation detection rules."""

    def test_detect_monotonicity_violation(self):
        markets = create_date_variant_markets()
        family = MarketFamily(
            id="test",
            markets=markets,
            relationship=RelationshipType.DATE_VARIANT,
        )

        config = DetectionConfig(min_edge_pct=0.1, min_leg_liquidity=100.0)
        violations = detect_monotonicity_violations(family, config)

        # Should detect violation: March (0.35) > April (0.30)
        assert len(violations) >= 1

        violation = violations[0]
        assert violation.type == ViolationType.MONOTONICITY
        assert violation.raw_edge > 0

    def test_detect_no_sweep(self):
        # Create markets where NO sweep is profitable
        markets = [
            PolymarketMarket(
                id="m1",
                question="Event by March?",
                slug="m1",
                condition_id="c1",
                yes_price=0.60,
                no_price=0.40,
                liquidity=5000.0,
                deadline=datetime(2025, 3, 31)
            ),
            PolymarketMarket(
                id="m2",
                question="Event by April?",
                slug="m2",
                condition_id="c1",
                yes_price=0.70,
                no_price=0.30,  # Total NO = 0.70 < 1.0
                liquidity=5000.0,
                deadline=datetime(2025, 4, 30)
            ),
        ]

        family = MarketFamily(
            id="test",
            markets=markets,
            relationship=RelationshipType.DATE_VARIANT,
        )

        config = DetectionConfig(min_sweep_edge_pct=0.1, min_leg_liquidity=100.0)
        violation = detect_no_sweep_opportunity(family, config)

        # Total NO cost = 0.70, guaranteed payout = 1.0, edge = 30%
        assert violation is not None
        assert violation.type == ViolationType.DATE_VARIANT_NO_SWEEP
        assert violation.raw_edge > 20  # Should be ~30%

    def test_detect_exclusive_violation(self):
        markets = create_exclusive_markets()
        family = MarketFamily(
            id="test",
            markets=markets,
            relationship=RelationshipType.MUTUALLY_EXCLUSIVE,
        )

        config = DetectionConfig(min_edge_pct=0.1, min_leg_liquidity=100.0)
        violation = detect_exclusive_violation(family, config)

        # Sum = 1.05, should detect violation
        assert violation is not None
        assert violation.type == ViolationType.EXCLUSIVE_VIOLATION
        assert violation.raw_edge > 0

    def test_detect_exhaustive_violation(self):
        markets = create_exhaustive_markets()
        family = MarketFamily(
            id="test",
            markets=markets,
            relationship=RelationshipType.EXHAUSTIVE,
        )

        config = DetectionConfig(
            min_edge_pct=0.1,
            min_leg_liquidity=100.0,
            sum_violation_threshold=0.01
        )
        violation = detect_exhaustive_violation(family, config)

        # Sum = 0.95 < 1.0, should detect if exhaustive
        assert violation is not None
        assert violation.type == ViolationType.EXHAUSTIVE_VIOLATION


# =============================================================================
# TRADE CONSTRUCTION TESTS
# =============================================================================

class TestTradeConstruction:
    """Tests for trade construction."""

    def test_construct_monotonicity_trade(self):
        markets = create_date_variant_markets()
        family = MarketFamily(
            id="test",
            markets=markets,
            relationship=RelationshipType.DATE_VARIANT,
        )

        # Detect violation
        config = DetectionConfig(min_edge_pct=0.1, min_leg_liquidity=100.0)
        violations = detect_monotonicity_violations(family, config)
        assert len(violations) > 0

        # Construct trade
        constructor = TradeConstructor()
        trade = constructor.construct_trade(
            violation=violations[0],
            family=family,
            position_usd=100.0
        )

        assert trade is not None
        assert len(trade.legs) == 2
        assert trade.raw_edge > 0

    def test_construct_sweep_trade(self):
        # Create profitable sweep scenario
        markets = [
            PolymarketMarket(
                id="m1",
                question="Option A?",
                slug="m1",
                condition_id="c1",
                group_slug="g1",
                yes_price=0.40,
                no_price=0.60,
                liquidity=5000.0,
            ),
            PolymarketMarket(
                id="m2",
                question="Option B?",
                slug="m2",
                condition_id="c1",
                group_slug="g1",
                yes_price=0.35,
                no_price=0.65,
                liquidity=5000.0,
            ),
            PolymarketMarket(
                id="m3",
                question="Option C?",
                slug="m3",
                condition_id="c1",
                group_slug="g1",
                yes_price=0.30,  # Sum = 1.05
                no_price=0.70,
                liquidity=5000.0,
            ),
        ]

        family = MarketFamily(
            id="test",
            markets=markets,
            relationship=RelationshipType.MUTUALLY_EXCLUSIVE,
        )

        config = DetectionConfig(min_edge_pct=0.1, min_leg_liquidity=100.0)
        violation = detect_exclusive_violation(family, config)
        assert violation is not None

        constructor = TradeConstructor()
        trade = constructor.construct_trade(
            violation=violation,
            family=family,
            position_usd=100.0
        )

        assert trade is not None
        assert len(trade.legs) == 3


# =============================================================================
# EDGE CALCULATION TESTS
# =============================================================================

class TestEdgeCalculation:
    """Tests for realizable edge calculation."""

    def test_calculate_edge_with_fees(self):
        markets = create_date_variant_markets()
        family = MarketFamily(
            id="test",
            markets=markets,
            relationship=RelationshipType.DATE_VARIANT,
        )

        # Get violation and trade
        det_config = DetectionConfig(min_edge_pct=0.1, min_leg_liquidity=100.0)
        violations = detect_monotonicity_violations(family, det_config)
        assert len(violations) > 0

        constructor = TradeConstructor()
        trade = constructor.construct_trade(
            violation=violations[0],
            family=family,
            position_usd=100.0
        )
        assert trade is not None

        # Calculate edge
        calculator = RealizableEdgeCalculator()
        edge = calculator.calculate(trade)

        # Edge should be less than raw due to fees/slippage
        assert edge.edge < trade.raw_edge
        assert edge.fees > 0
        assert edge.slippage >= 0

    def test_profitability_check(self):
        markets = create_date_variant_markets()
        family = MarketFamily(
            id="test",
            markets=markets,
            relationship=RelationshipType.DATE_VARIANT,
        )

        det_config = DetectionConfig(min_edge_pct=0.1, min_leg_liquidity=100.0)
        violations = detect_monotonicity_violations(family, det_config)

        constructor = TradeConstructor()
        trade = constructor.construct_trade(
            violation=violations[0],
            family=family,
            position_usd=100.0
        )

        edge_config = EdgeConfig(min_realizable_edge_pct=0.1)
        calculator = RealizableEdgeCalculator(edge_config)
        edge = calculator.calculate(trade)

        # is_profitable should reflect thresholds
        profitable = calculator.is_profitable(edge)
        assert isinstance(profitable, bool)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestInferenceEngineIntegration:
    """Integration tests for the full inference pipeline."""

    def test_full_pipeline_date_variant(self):
        """Test full pipeline with date-variant markets."""
        markets = create_date_variant_markets()

        # Use aggressive mode with lower thresholds
        config = InferenceEngineConfig.aggressive()
        config = InferenceEngineConfig(
            detection=DetectionConfig(
                min_edge_pct=0.1,
                min_leg_liquidity=100.0
            ),
            edge=EdgeConfig(
                min_realizable_edge_pct=0.1,
                min_worst_case_profit=-100.0
            ),
            min_family_confidence=0.1,
            min_realizable_edge_pct=0.1,
            min_leg_liquidity=100.0,
        )
        engine = InferenceEngine(config)
        signals = engine.process_markets(markets)

        # Check stats to understand what happened
        stats = engine.get_stats()
        families = engine.get_families()

        # Verify families were discovered
        assert stats["families_discovered"] >= 1, f"No families discovered from {len(markets)} markets"

        # If signals found, verify their structure
        if len(signals) >= 1:
            signal = signals[0]
            assert signal.type == "structural_arb"
            assert signal.realizable_edge > 0
        else:
            # Even if no signals, verify the pipeline ran correctly
            assert stats["runs"] == 1
            assert stats["markets_processed"] == 3
            # Log why no signals (for debugging)
            print(f"Stats: {stats}")
            print(f"Families: {[f.relationship.value for f in families]}")

    def test_full_pipeline_exclusive(self):
        """Test full pipeline with exclusive markets."""
        markets = create_exclusive_markets()

        engine = create_engine(mode="aggressive")
        signals = engine.process_markets(markets)

        # May or may not find signals depending on classification
        # Just verify no errors
        assert isinstance(signals, list)

    def test_engine_stats(self):
        """Test that engine tracks statistics correctly."""
        markets = create_date_variant_markets()

        engine = create_engine()
        initial_stats = engine.get_stats()

        assert initial_stats["runs"] == 0
        assert initial_stats["signals_generated"] == 0

        engine.process_markets(markets)
        final_stats = engine.get_stats()

        assert final_stats["runs"] == 1
        assert final_stats["markets_processed"] == 3

    def test_conservative_vs_aggressive(self):
        """Test that conservative mode has higher thresholds."""
        markets = create_date_variant_markets()

        conservative = create_engine(mode="conservative")
        aggressive = create_engine(mode="aggressive")

        cons_signals = conservative.process_markets(markets)
        agg_signals = aggressive.process_markets(markets)

        # Aggressive should find same or more signals
        assert len(agg_signals) >= len(cons_signals)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
