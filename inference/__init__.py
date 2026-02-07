"""
Cross-Market Inference Engine
=============================

Detects structural arbitrage opportunities across related Polymarket markets.

Main components:
- InferenceEngine: Main orchestration
- FamilyDiscovery: Groups related markets
- ViolationDetector: Finds constraint violations
- TradeConstructor: Builds executable trades
- ArbMonitor: Continuous monitoring

Usage:
    from inference import InferenceEngine, create_engine

    engine = create_engine(mode="balanced")
    signals = engine.process_markets(markets)
"""

from inference.arb_monitor import (
    ArbMonitor,
    ArbMonitorConfig,
    create_monitor,
)
from inference.detection_rules import (
    DetectionConfig,
    ViolationDetector,
)
from inference.engine import (
    InferenceEngine,
    InferenceEngineConfig,
    create_engine,
)
from inference.family_discovery import (
    FamilyDiscovery,
    FamilyDiscoveryConfig,
    find_date_variant_families,
)
from inference.models import (
    ArbSignal,
    MarketFamily,
    MultiLegTrade,
    PolymarketMarket,
    RealizableEdge,
    RelationshipType,
    TradeLeg,
    Violation,
    ViolationType,
)

__all__ = [
    # Monitor
    "ArbMonitor",
    "ArbMonitorConfig",
    "ArbSignal",
    "DetectionConfig",
    # Discovery
    "FamilyDiscovery",
    "FamilyDiscoveryConfig",
    # Engine
    "InferenceEngine",
    "InferenceEngineConfig",
    "MarketFamily",
    "MultiLegTrade",
    # Models
    "PolymarketMarket",
    "RealizableEdge",
    "RelationshipType",
    "TradeLeg",
    "Violation",
    # Detection
    "ViolationDetector",
    "ViolationType",
    "create_engine",
    "create_monitor",
    "find_date_variant_families",
]
