"""
Inference Engine
================

Main orchestration module for the Cross-Market Inference Engine.

Pipeline:
1. Market ingestion and enrichment
2. Family discovery (grouping related markets)
3. Relationship classification
4. Violation detection
5. Trade construction
6. Edge calculation
7. Signal generation

Outputs ArbSignal objects ready for execution.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from inference.detection_rules import (
    DetectionConfig,
    ViolationDetector,
)
from inference.family_discovery import (
    FamilyDiscovery,
    FamilyDiscoveryConfig,
    enrich_market,
)
from inference.llm_classifier import (
    LLMClassifier,
    LLMConfig,
)
from inference.models import (
    ArbSignal,
    MarketFamily,
    MultiLegTrade,
    PolymarketMarket,
    RealizableEdge,
    RelationshipType,
    Violation,
)
from inference.realizable_edge import (
    EdgeConfig,
    RealizableEdgeCalculator,
)
from inference.relationship import (
    RelationshipClassifier,
    RelationshipConfig,
)
from inference.trade_constructor import (
    TradeConstructor,
    TradeConstructorConfig,
)

logger = logging.getLogger("PolyMangoBot.inference.engine")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InferenceEngineConfig:
    """Master configuration for the inference engine"""
    # Component configs
    family_discovery: FamilyDiscoveryConfig = field(
        default_factory=FamilyDiscoveryConfig
    )
    relationship: RelationshipConfig = field(
        default_factory=RelationshipConfig
    )
    detection: DetectionConfig = field(
        default_factory=DetectionConfig
    )
    trade_constructor: TradeConstructorConfig = field(
        default_factory=TradeConstructorConfig
    )
    edge: EdgeConfig = field(
        default_factory=EdgeConfig
    )

    # LLM settings
    use_llm: bool = False
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-haiku-20240307"

    # Filtering thresholds
    min_family_confidence: float = 0.6
    min_realizable_edge_pct: float = 0.3
    min_leg_liquidity: float = 500.0

    # Position sizing
    default_position_usd: float = 100.0
    max_position_usd: float = 1000.0

    # Output settings
    max_signals_per_run: int = 10

    @classmethod
    def conservative(cls) -> "InferenceEngineConfig":
        """Conservative configuration with higher thresholds"""
        return cls(
            detection=DetectionConfig(
                min_edge_pct=1.0,
                min_leg_liquidity=1000.0
            ),
            edge=EdgeConfig(
                min_realizable_edge_pct=0.5,
                min_worst_case_profit=1.0
            ),
            min_realizable_edge_pct=0.5,
            min_leg_liquidity=1000.0,
            default_position_usd=50.0,
            max_position_usd=500.0
        )

    @classmethod
    def aggressive(cls) -> "InferenceEngineConfig":
        """Aggressive configuration with lower thresholds"""
        return cls(
            detection=DetectionConfig(
                min_edge_pct=0.3,
                min_leg_liquidity=200.0
            ),
            edge=EdgeConfig(
                min_realizable_edge_pct=0.2,
                min_worst_case_profit=-10.0  # Allow small losses
            ),
            min_realizable_edge_pct=0.2,
            min_leg_liquidity=200.0,
            default_position_usd=200.0,
            max_position_usd=2000.0
        )


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """
    Cross-Market Inference Engine for structural arbitrage detection.

    Orchestrates the full pipeline from market data to trading signals.
    """

    def __init__(self, config: Optional[InferenceEngineConfig] = None):
        self.config = config or InferenceEngineConfig()

        # Initialize components
        self._init_components()

        # State
        self._markets: dict[str, PolymarketMarket] = {}
        self._families: list[MarketFamily] = []
        self._last_run_time: float = 0.0

        # Statistics
        self._stats = {
            "runs": 0,
            "markets_processed": 0,
            "families_discovered": 0,
            "violations_detected": 0,
            "signals_generated": 0,
            "total_edge_detected": 0.0,
        }

        logger.info("Inference Engine initialized")

    def _init_components(self):
        """Initialize sub-components"""
        # LLM classifier (optional)
        self.llm = None
        if self.config.use_llm:
            try:
                llm_config = LLMConfig(
                    provider=self.config.llm_provider,
                    model=self.config.llm_model
                )
                self.llm = LLMClassifier(llm_config)
                logger.info(f"LLM classifier enabled: {self.config.llm_provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")

        # Family discovery
        self.family_discovery = FamilyDiscovery(
            config=self.config.family_discovery,
            llm_classifier=self.llm
        )

        # Relationship classifier
        self.relationship_classifier = RelationshipClassifier(
            config=self.config.relationship
        )

        # Violation detector
        self.violation_detector = ViolationDetector(
            config=self.config.detection
        )

        # Trade constructor
        self.trade_constructor = TradeConstructor(
            config=self.config.trade_constructor
        )

        # Edge calculator
        self.edge_calculator = RealizableEdgeCalculator(
            config=self.config.edge
        )

    def process_markets(
        self,
        markets: list[PolymarketMarket]
    ) -> list[ArbSignal]:
        """
        Process a batch of markets and generate arbitrage signals.

        This is the main entry point for the inference engine.

        Args:
            markets: List of PolymarketMarket objects with current prices

        Returns:
            List of ArbSignal objects for execution
        """
        start_time = time.time()
        self._stats["runs"] += 1

        logger.info(f"Processing {len(markets)} markets...")

        # Step 1: Enrich markets
        for market in markets:
            enrich_market(market)
            self._markets[market.id] = market

        self._stats["markets_processed"] += len(markets)

        # Step 2: Discover families
        self._families = self.family_discovery.discover_families(markets)
        self._stats["families_discovered"] += len(self._families)

        if not self._families:
            logger.info("No market families discovered")
            return []

        # Step 3: Classify relationships
        for family in self._families:
            if family.confidence >= self.config.min_family_confidence:
                self.relationship_classifier.classify_family(family)

        # Filter families with known relationships
        classified_families = [
            f for f in self._families
            if f.relationship != RelationshipType.UNKNOWN
        ]

        logger.info(
            f"Discovered {len(self._families)} families, "
            f"{len(classified_families)} with known relationships"
        )

        # Step 4: Detect violations
        all_violations: list[Violation] = []
        for family in classified_families:
            violations = self.violation_detector.detect_all(family)
            all_violations.extend(violations)

        self._stats["violations_detected"] += len(all_violations)

        if not all_violations:
            logger.info("No violations detected")
            return []

        logger.info(f"Detected {len(all_violations)} violations")

        # Step 5: Construct trades and calculate edge
        signals: list[ArbSignal] = []

        for violation in all_violations:
            # Find the family
            target_family = next(
                (f for f in self._families if f.id == violation.family_id),
                None
            )
            if not target_family:
                continue

            # Construct trade
            trade = self.trade_constructor.construct_trade(
                violation=violation,
                family=target_family,
                position_usd=self.config.default_position_usd
            )

            if not trade:
                continue

            # Calculate realizable edge
            edge = self.edge_calculator.calculate(trade)

            if not self.edge_calculator.is_profitable(edge):
                continue

            # Generate signal
            signal = self._create_signal(violation, family, trade, edge)
            signals.append(signal)

            self._stats["total_edge_detected"] += edge.edge

        # Sort by edge and limit
        signals.sort(key=lambda s: s.realizable_edge, reverse=True)
        signals = signals[:self.config.max_signals_per_run]

        self._stats["signals_generated"] += len(signals)
        self._last_run_time = time.time() - start_time

        logger.info(
            f"Generated {len(signals)} signals in {self._last_run_time:.2f}s"
        )

        return signals

    async def process_markets_async(
        self,
        markets: list[PolymarketMarket]
    ) -> list[ArbSignal]:
        """
        Async version with LLM fallback support.

        Use this when LLM classification is enabled.
        """
        start_time = time.time()
        self._stats["runs"] += 1

        # Enrich markets
        for market in markets:
            enrich_market(market)
            self._markets[market.id] = market

        self._stats["markets_processed"] += len(markets)

        # Discover families (with LLM if enabled)
        if self.llm and self.config.use_llm:
            self._families = await self.family_discovery.discover_families_with_llm(
                markets
            )
        else:
            self._families = self.family_discovery.discover_families(markets)

        self._stats["families_discovered"] += len(self._families)

        # Rest of pipeline is same as sync version
        if not self._families:
            return []

        # Classify relationships
        for family in self._families:
            if family.confidence >= self.config.min_family_confidence:
                self.relationship_classifier.classify_family(family)

        classified_families = [
            f for f in self._families
            if f.relationship != RelationshipType.UNKNOWN
        ]

        # Detect violations
        all_violations = []
        for family in classified_families:
            violations = self.violation_detector.detect_all(family)
            all_violations.extend(violations)

        self._stats["violations_detected"] += len(all_violations)

        if not all_violations:
            return []

        # Generate signals
        signals = []
        for violation in all_violations:
            found_family = next(
                (f for f in self._families if f.id == violation.family_id),
                None
            )
            if not found_family:
                continue

            trade = self.trade_constructor.construct_trade(
                violation=violation,
                family=found_family,
                position_usd=self.config.default_position_usd
            )

            if not trade:
                continue

            edge = self.edge_calculator.calculate(trade)

            if not self.edge_calculator.is_profitable(edge):
                continue

            signal = self._create_signal(violation, family, trade, edge)
            signals.append(signal)

        signals.sort(key=lambda s: s.realizable_edge, reverse=True)
        signals = signals[:self.config.max_signals_per_run]

        self._stats["signals_generated"] += len(signals)
        self._last_run_time = time.time() - start_time

        return signals

    def _create_signal(
        self,
        violation: Violation,
        family: MarketFamily,
        trade: MultiLegTrade,
        edge: RealizableEdge
    ) -> ArbSignal:
        """Create an ArbSignal from violation, trade, and edge analysis."""
        return ArbSignal(
            type="structural_arb",
            subtype=violation.type.value,
            family_id=family.id,
            legs=trade.legs,
            raw_edge=trade.raw_edge,
            realizable_edge=edge.edge,
            worst_case_pnl=edge.worst_pnl,
            best_case_pnl=edge.best_pnl,
            confidence=9,  # Structural arbs are high confidence
            min_liquidity=trade.min_liquidity,
            execution_risk=edge.execution_risk
        )

    def get_families(self) -> list[MarketFamily]:
        """Get discovered market families."""
        return self._families

    def get_market(self, market_id: str) -> Optional[PolymarketMarket]:
        """Get a market by ID."""
        return self._markets.get(market_id)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "last_run_time_seconds": self._last_run_time,
            "active_markets": len(self._markets),
            "active_families": len(self._families),
        }

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "runs": 0,
            "markets_processed": 0,
            "families_discovered": 0,
            "violations_detected": 0,
            "signals_generated": 0,
            "total_edge_detected": 0.0,
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_engine(
    mode: str = "balanced",
    use_llm: bool = False
) -> InferenceEngine:
    """
    Factory function to create an InferenceEngine.

    Args:
        mode: "conservative", "balanced", or "aggressive"
        use_llm: Whether to enable LLM classification

    Returns:
        Configured InferenceEngine
    """
    if mode == "conservative":
        config = InferenceEngineConfig.conservative()
    elif mode == "aggressive":
        config = InferenceEngineConfig.aggressive()
    else:
        config = InferenceEngineConfig()

    config.use_llm = use_llm

    return InferenceEngine(config)
