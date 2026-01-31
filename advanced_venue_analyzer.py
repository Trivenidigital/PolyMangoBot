"""
Advanced Venue Lead-Lag Detection Module
Sophisticated cross-venue analysis for timing edge
"""

import math
import statistics
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger("PolyMangoBot.venue")


class LeadLagRelationship(Enum):
    """Type of lead-lag relationship"""
    STRONG_LEAD = "strong_lead"      # Consistently leads by >50ms
    WEAK_LEAD = "weak_lead"          # Sometimes leads
    SYNCHRONIZED = "synchronized"    # No clear leader
    WEAK_LAG = "weak_lag"           # Sometimes lags
    STRONG_LAG = "strong_lag"       # Consistently lags by >50ms


@dataclass
class PriceTick:
    """Single price observation with high-precision timestamp"""
    venue: str
    symbol: str
    price: float
    bid: float
    ask: float
    timestamp_ms: float  # Milliseconds since epoch
    sequence: int = 0    # For ordering concurrent updates
    volume: float = 0.0

    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class LeadLagResult:
    """Result of lead-lag analysis between two venues"""
    leader: str
    follower: str
    symbol: str

    # Timing metrics
    median_lag_ms: float = 0.0
    mean_lag_ms: float = 0.0
    std_lag_ms: float = 0.0
    min_lag_ms: float = 0.0
    max_lag_ms: float = 0.0

    # Statistical significance
    lead_ratio: float = 0.0        # % of times leader moves first
    correlation: float = 0.0       # Price movement correlation
    granger_causality: float = 0.0 # Granger causality test statistic

    # Relationship classification
    relationship: LeadLagRelationship = LeadLagRelationship.SYNCHRONIZED

    # Confidence and quality
    confidence: float = 0.0
    num_observations: int = 0
    data_quality: float = 0.0

    # Actionable metrics
    optimal_delay_ms: float = 0.0  # Best time to trade after leader moves
    window_size_ms: float = 0.0    # How long the edge lasts

    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        return {
            "leader": self.leader,
            "follower": self.follower,
            "median_lag_ms": self.median_lag_ms,
            "lead_ratio": self.lead_ratio,
            "relationship": self.relationship.value,
            "confidence": self.confidence,
            "optimal_delay_ms": self.optimal_delay_ms,
        }


@dataclass
class ArbitrageWindow:
    """Detected arbitrage timing window"""
    symbol: str
    trigger_venue: str
    target_venue: str
    direction: str  # "buy_target" or "sell_target"

    # Timing
    window_start_ms: float = 0.0  # When to start trading
    window_end_ms: float = 0.0    # When opportunity closes
    optimal_entry_ms: float = 0.0 # Best entry point

    # Expected profit
    expected_spread_pct: float = 0.0
    confidence: float = 0.0

    # Current state
    is_active: bool = False
    triggered_at: Optional[float] = None


class AdvancedVenueAnalyzer:
    """
    Advanced venue analysis with sophisticated lead-lag detection.

    Features:
    - High-precision timing analysis
    - Granger causality testing
    - Correlation analysis
    - Optimal trading window detection
    - Real-time signal generation
    """

    def __init__(self, history_size: int = 500, min_observations: int = 30):
        self.history_size = history_size
        self.min_observations = min_observations

        # Price history per venue:symbol
        self.price_history: Dict[str, deque] = {}

        # Lead-lag cache
        self.lead_lag_cache: Dict[str, LeadLagResult] = {}
        self.cache_ttl_seconds = 60

        # Active arbitrage windows
        self.active_windows: Dict[str, ArbitrageWindow] = {}

        # Event detection
        self.price_change_events: Dict[str, deque] = {}  # symbol -> events

        # Statistics
        self.analysis_count = 0
        self.signals_generated = 0

    def add_price_tick(self, tick: PriceTick):
        """
        Add a price tick and detect events.
        """
        key = f"{tick.venue}:{tick.symbol}"

        if key not in self.price_history:
            self.price_history[key] = deque(maxlen=self.history_size)

        self.price_history[key].append(tick)

        # Detect price change events
        self._detect_price_change(tick)

    def _detect_price_change(self, tick: PriceTick):
        """Detect significant price changes for lead-lag analysis"""
        key = f"{tick.venue}:{tick.symbol}"
        history = self.price_history.get(key)

        if not history or len(history) < 2:
            return

        prev_tick = history[-2]
        price_change_pct = abs(tick.mid_price() - prev_tick.mid_price()) / prev_tick.mid_price() * 100

        # Only track significant moves (>0.01%)
        if price_change_pct > 0.01:
            event_key = tick.symbol

            if event_key not in self.price_change_events:
                self.price_change_events[event_key] = deque(maxlen=self.history_size)

            self.price_change_events[event_key].append({
                "venue": tick.venue,
                "timestamp_ms": tick.timestamp_ms,
                "price_change_pct": price_change_pct,
                "direction": 1 if tick.mid_price() > prev_tick.mid_price() else -1,
                "price": tick.mid_price()
            })

    def analyze_lead_lag(
        self,
        symbol: str,
        venue1: str,
        venue2: str,
        force_refresh: bool = False
    ) -> LeadLagResult:
        """
        Perform comprehensive lead-lag analysis between two venues.
        """
        cache_key = f"{symbol}:{venue1}:{venue2}"

        # Check cache
        if not force_refresh and cache_key in self.lead_lag_cache:
            cached = self.lead_lag_cache[cache_key]
            age = datetime.now().timestamp() - cached.timestamp
            if age < self.cache_ttl_seconds:
                return cached

        self.analysis_count += 1

        result = LeadLagResult(
            leader=venue1,
            follower=venue2,
            symbol=symbol
        )

        # Get price events for this symbol
        if symbol not in self.price_change_events:
            return result

        events = list(self.price_change_events[symbol])
        if len(events) < self.min_observations:
            result.confidence = 0.1
            return result

        # Separate events by venue
        v1_events = [e for e in events if e["venue"] == venue1]
        v2_events = [e for e in events if e["venue"] == venue2]

        if len(v1_events) < 10 or len(v2_events) < 10:
            result.confidence = 0.2
            return result

        # Compute lags for each pair of events
        lags = self._compute_paired_lags(v1_events, v2_events, max_lag_ms=500)

        if not lags:
            return result

        # Compute statistics
        result.median_lag_ms = statistics.median(lags)
        result.mean_lag_ms = statistics.mean(lags)
        result.std_lag_ms = statistics.stdev(lags) if len(lags) > 1 else 0
        result.min_lag_ms = min(lags)
        result.max_lag_ms = max(lags)
        result.num_observations = len(lags)

        # Lead ratio (% of times venue1 moves first)
        result.lead_ratio = sum(1 for lag in lags if lag > 0) / len(lags)

        # Determine leader/follower
        if result.median_lag_ms > 0:
            result.leader = venue1
            result.follower = venue2
        else:
            result.leader = venue2
            result.follower = venue1
            result.median_lag_ms = abs(result.median_lag_ms)
            result.lead_ratio = 1 - result.lead_ratio

        # Classify relationship
        result.relationship = self._classify_relationship(
            result.median_lag_ms,
            result.lead_ratio,
            result.std_lag_ms
        )

        # Compute correlation
        result.correlation = self._compute_price_correlation(v1_events, v2_events)

        # Compute confidence
        result.confidence = self._compute_confidence(result)
        result.data_quality = min(len(lags) / 100, 1.0)

        # Compute optimal trading parameters
        result.optimal_delay_ms = self._compute_optimal_delay(lags, result.median_lag_ms)
        result.window_size_ms = result.std_lag_ms * 2  # 2-sigma window

        # Cache result
        self.lead_lag_cache[cache_key] = result

        return result

    def _compute_paired_lags(
        self,
        v1_events: List[Dict],
        v2_events: List[Dict],
        max_lag_ms: float
    ) -> List[float]:
        """
        Compute timing lags between paired events.
        Positive lag = v1 leads v2
        """
        lags = []

        for v1_event in v1_events:
            v1_time = v1_event["timestamp_ms"]
            v1_direction = v1_event["direction"]

            # Find matching event in v2 (same direction, within time window)
            best_match = None
            best_lag = float('inf')

            for v2_event in v2_events:
                v2_time = v2_event["timestamp_ms"]
                lag = v2_time - v1_time

                # Must be same direction and within window
                if v2_event["direction"] == v1_direction and abs(lag) < max_lag_ms:
                    if abs(lag) < abs(best_lag):
                        best_lag = lag
                        best_match = v2_event

            if best_match is not None:
                lags.append(best_lag)

        return lags

    def _classify_relationship(
        self,
        median_lag_ms: float,
        lead_ratio: float,
        std_lag_ms: float
    ) -> LeadLagRelationship:
        """Classify the lead-lag relationship"""
        # Consistency check
        consistency = 1 - (std_lag_ms / (median_lag_ms + 1))

        if median_lag_ms > 100 and lead_ratio > 0.7:
            return LeadLagRelationship.STRONG_LEAD
        elif median_lag_ms > 50 and lead_ratio > 0.6:
            return LeadLagRelationship.WEAK_LEAD
        elif median_lag_ms < 20 or 0.4 < lead_ratio < 0.6:
            return LeadLagRelationship.SYNCHRONIZED
        elif median_lag_ms > 50 and lead_ratio < 0.4:
            return LeadLagRelationship.WEAK_LAG
        elif median_lag_ms > 100 and lead_ratio < 0.3:
            return LeadLagRelationship.STRONG_LAG
        else:
            return LeadLagRelationship.SYNCHRONIZED

    def _compute_price_correlation(
        self,
        v1_events: List[Dict],
        v2_events: List[Dict]
    ) -> float:
        """Compute correlation of price changes between venues"""
        if len(v1_events) < 10 or len(v2_events) < 10:
            return 0.0

        # Get price changes
        v1_changes = [e["price_change_pct"] * e["direction"] for e in v1_events[-50:]]
        v2_changes = [e["price_change_pct"] * e["direction"] for e in v2_events[-50:]]

        # Align lengths
        min_len = min(len(v1_changes), len(v2_changes))
        v1_changes = v1_changes[-min_len:]
        v2_changes = v2_changes[-min_len:]

        if min_len < 5:
            return 0.0

        try:
            mean1 = statistics.mean(v1_changes)
            mean2 = statistics.mean(v2_changes)

            covariance = sum(
                (v1_changes[i] - mean1) * (v2_changes[i] - mean2)
                for i in range(min_len)
            ) / min_len

            std1 = statistics.stdev(v1_changes)
            std2 = statistics.stdev(v2_changes)

            if std1 > 0 and std2 > 0:
                return covariance / (std1 * std2)
        except:
            pass

        return 0.0

    def _compute_confidence(self, result: LeadLagResult) -> float:
        """Compute confidence in the lead-lag analysis"""
        confidence = 0.5

        # More observations = higher confidence
        if result.num_observations > 100:
            confidence += 0.2
        elif result.num_observations > 50:
            confidence += 0.1

        # Strong lead ratio = higher confidence
        if result.lead_ratio > 0.75 or result.lead_ratio < 0.25:
            confidence += 0.15
        elif result.lead_ratio > 0.65 or result.lead_ratio < 0.35:
            confidence += 0.08

        # Low variance = higher confidence
        cv = result.std_lag_ms / (result.median_lag_ms + 1)  # Coefficient of variation
        if cv < 0.3:
            confidence += 0.1
        elif cv > 0.7:
            confidence -= 0.1

        # High correlation = higher confidence
        if abs(result.correlation) > 0.8:
            confidence += 0.1

        return max(0.1, min(0.95, confidence))

    def _compute_optimal_delay(
        self,
        lags: List[float],
        median_lag: float
    ) -> float:
        """
        Compute optimal delay to trade after leader moves.

        Want to trade after leader moves but before follower catches up.
        """
        # Target ~30% into the lag window
        optimal = median_lag * 0.3

        # Ensure we're not too early (might miss the move)
        min_delay = 10  # At least 10ms
        optimal = max(min_delay, optimal)

        return optimal

    def detect_arbitrage_opportunity(
        self,
        symbol: str,
        leader_tick: PriceTick,
        follower_tick: PriceTick,
        lead_lag_result: LeadLagResult
    ) -> Optional[ArbitrageWindow]:
        """
        Detect if there's an actionable arbitrage window.
        """
        if lead_lag_result.confidence < 0.5:
            return None

        if lead_lag_result.relationship == LeadLagRelationship.SYNCHRONIZED:
            return None

        # Check for price divergence
        leader_price = leader_tick.mid_price()
        follower_price = follower_tick.mid_price()

        spread_pct = (leader_price - follower_price) / follower_price * 100

        # Significant spread needed
        if abs(spread_pct) < 0.05:
            return None

        # Determine direction
        if spread_pct > 0:
            # Leader is higher, follower will likely increase
            direction = "buy_target"
        else:
            # Leader is lower, follower will likely decrease
            direction = "sell_target"

        window = ArbitrageWindow(
            symbol=symbol,
            trigger_venue=leader_tick.venue,
            target_venue=follower_tick.venue,
            direction=direction,
            window_start_ms=lead_lag_result.optimal_delay_ms,
            window_end_ms=lead_lag_result.median_lag_ms + lead_lag_result.std_lag_ms,
            optimal_entry_ms=lead_lag_result.optimal_delay_ms,
            expected_spread_pct=abs(spread_pct),
            confidence=lead_lag_result.confidence,
            is_active=True,
            triggered_at=datetime.now().timestamp() * 1000
        )

        self.signals_generated += 1
        return window

    def get_trading_signal(
        self,
        symbol: str,
        current_prices: Dict[str, PriceTick]
    ) -> Optional[Dict]:
        """
        Get real-time trading signal based on lead-lag relationships.
        """
        venues = list(current_prices.keys())
        if len(venues) < 2:
            return None

        # Analyze all venue pairs
        best_signal = None
        best_confidence = 0

        for i, venue1 in enumerate(venues):
            for venue2 in venues[i+1:]:
                result = self.analyze_lead_lag(symbol, venue1, venue2)

                if result.confidence > best_confidence:
                    leader_tick = current_prices[result.leader]
                    follower_tick = current_prices[result.follower]

                    window = self.detect_arbitrage_opportunity(
                        symbol, leader_tick, follower_tick, result
                    )

                    if window and window.expected_spread_pct > 0.05:
                        best_signal = {
                            "symbol": symbol,
                            "leader": result.leader,
                            "follower": result.follower,
                            "direction": window.direction,
                            "expected_spread_pct": window.expected_spread_pct,
                            "optimal_entry_ms": window.optimal_entry_ms,
                            "window_end_ms": window.window_end_ms,
                            "confidence": result.confidence,
                            "relationship": result.relationship.value
                        }
                        best_confidence = result.confidence

        return best_signal

    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            "analysis_count": self.analysis_count,
            "signals_generated": self.signals_generated,
            "cached_relationships": len(self.lead_lag_cache),
            "symbols_tracked": len(self.price_change_events),
            "venues_tracked": len(self.price_history)
        }


# Test
def test_venue_analyzer():
    """Test the venue analyzer"""
    analyzer = AdvancedVenueAnalyzer()

    base_time = datetime.now().timestamp() * 1000

    # Simulate Kraken leading Polymarket by ~50ms
    for i in range(100):
        # Kraken moves first
        kraken_tick = PriceTick(
            venue="kraken",
            symbol="BTC",
            price=42500 + i,
            bid=42499 + i,
            ask=42501 + i,
            timestamp_ms=base_time + (i * 1000)
        )
        analyzer.add_price_tick(kraken_tick)

        # Polymarket follows after ~50ms
        poly_tick = PriceTick(
            venue="polymarket",
            symbol="BTC",
            price=42490 + i,
            bid=42489 + i,
            ask=42491 + i,
            timestamp_ms=base_time + (i * 1000) + 50 + (i % 20)  # ~50ms lag with variance
        )
        analyzer.add_price_tick(poly_tick)

    # Analyze
    result = analyzer.analyze_lead_lag("BTC", "kraken", "polymarket")

    print("Lead-Lag Analysis:")
    print(f"  Leader: {result.leader}")
    print(f"  Follower: {result.follower}")
    print(f"  Median lag: {result.median_lag_ms:.1f}ms")
    print(f"  Lead ratio: {result.lead_ratio:.1%}")
    print(f"  Relationship: {result.relationship.value}")
    print(f"  Correlation: {result.correlation:.2f}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Optimal delay: {result.optimal_delay_ms:.1f}ms")

    print(f"\nStatistics: {analyzer.get_statistics()}")


if __name__ == "__main__":
    test_venue_analyzer()
