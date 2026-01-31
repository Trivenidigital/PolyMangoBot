"""
Advanced Market Maker Tracker
Sophisticated MM inventory tracking with pattern recognition:
- Market maker identification and profiling
- Inventory position estimation
- Trading pattern detection
- Behavior prediction
- Lead-lag analysis with MMs
"""

import asyncio
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger("PolyMangoBot.advanced_mm_tracker")


class MMBehavior(Enum):
    """Market maker behavior patterns"""
    ACCUMULATING = "accumulating"  # Building inventory
    DISTRIBUTING = "distributing"  # Reducing inventory
    NEUTRAL = "neutral"  # Balanced
    REPOSITIONING = "repositioning"  # Shifting levels
    WIDENING = "widening"  # Increasing spreads
    TIGHTENING = "tightening"  # Decreasing spreads
    AGGRESSIVE = "aggressive"  # Taking liquidity
    PASSIVE = "passive"  # Providing liquidity


class InventoryState(Enum):
    """Inventory position states"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    EXTREME_LONG = "extreme_long"
    EXTREME_SHORT = "extreme_short"


@dataclass
class OrderBookLevel:
    """Single level in order book"""
    price: float
    quantity: float
    order_count: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderBookSnapshot:
    """Point-in-time order book snapshot"""
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float
    mid_price: float = 0.0
    spread: float = 0.0
    imbalance: float = 0.0

    def __post_init__(self):
        if self.bids and self.asks:
            self.mid_price = (self.bids[0].price + self.asks[0].price) / 2
            self.spread = self.asks[0].price - self.bids[0].price

            # Calculate imbalance
            bid_volume = sum(b.quantity for b in self.bids[:5])
            ask_volume = sum(a.quantity for a in self.asks[:5])
            total = bid_volume + ask_volume
            if total > 0:
                self.imbalance = (bid_volume - ask_volume) / total


@dataclass
class MMProfile:
    """Profile of a detected market maker"""
    mm_id: str
    venue: str
    first_seen: float
    last_seen: float

    # Inventory estimation
    estimated_inventory: float = 0.0
    inventory_state: InventoryState = InventoryState.NEUTRAL
    inventory_confidence: float = 0.0

    # Trading patterns
    avg_spread: float = 0.0
    avg_quote_size: float = 0.0
    avg_quote_refresh_ms: float = 0.0
    fill_rate: float = 0.0

    # Behavior
    current_behavior: MMBehavior = MMBehavior.NEUTRAL
    behavior_confidence: float = 0.0

    # Historical metrics
    total_trades_observed: int = 0
    total_volume_observed: float = 0.0
    price_levels_used: Set[float] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "mm_id": self.mm_id,
            "venue": self.venue,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "estimated_inventory": self.estimated_inventory,
            "inventory_state": self.inventory_state.value,
            "inventory_confidence": self.inventory_confidence,
            "avg_spread": self.avg_spread,
            "avg_quote_size": self.avg_quote_size,
            "avg_quote_refresh_ms": self.avg_quote_refresh_ms,
            "fill_rate": self.fill_rate,
            "current_behavior": self.current_behavior.value,
            "behavior_confidence": self.behavior_confidence,
            "total_trades_observed": self.total_trades_observed,
            "total_volume_observed": self.total_volume_observed,
        }


@dataclass
class TradeObservation:
    """Observed trade data"""
    trade_id: str
    venue: str
    market: str
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    timestamp: float
    is_maker: Optional[bool] = None
    taker_id: Optional[str] = None
    maker_id: Optional[str] = None


class InventoryEstimator:
    """Estimates MM inventory from order flow"""

    def __init__(self, decay_factor: float = 0.995):
        self.decay_factor = decay_factor
        self._inventory_estimate: float = 0.0
        self._confidence: float = 0.0
        self._observations: deque = deque(maxlen=1000)

    def update(self, trade: TradeObservation, is_mm_trade: bool):
        """Update inventory estimate from trade"""
        if not is_mm_trade:
            return

        # Decay old estimates
        self._inventory_estimate *= self.decay_factor

        # Update based on trade side
        if trade.side == "buy":
            # MM bought = inventory increased
            self._inventory_estimate += trade.quantity
        else:
            # MM sold = inventory decreased
            self._inventory_estimate -= trade.quantity

        self._observations.append({
            "timestamp": trade.timestamp,
            "delta": trade.quantity if trade.side == "buy" else -trade.quantity
        })

        # Update confidence based on observation count
        self._confidence = min(0.9, len(self._observations) / 100)

    @property
    def estimate(self) -> float:
        return self._inventory_estimate

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def state(self) -> InventoryState:
        """Determine inventory state"""
        if self._confidence < 0.3:
            return InventoryState.NEUTRAL

        # Normalize estimate
        if len(self._observations) < 10:
            return InventoryState.NEUTRAL

        recent_deltas = [o["delta"] for o in list(self._observations)[-100:]]
        avg_volume = np.mean(np.abs(recent_deltas)) if recent_deltas else 1

        normalized = self._inventory_estimate / (avg_volume * 10)

        if normalized > 2:
            return InventoryState.EXTREME_LONG
        elif normalized > 0.5:
            return InventoryState.LONG
        elif normalized < -2:
            return InventoryState.EXTREME_SHORT
        elif normalized < -0.5:
            return InventoryState.SHORT
        else:
            return InventoryState.NEUTRAL


class BehaviorDetector:
    """Detects MM behavior patterns"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._spread_history: deque = deque(maxlen=window_size)
        self._imbalance_history: deque = deque(maxlen=window_size)
        self._quote_update_times: deque = deque(maxlen=window_size)
        self._trade_directions: deque = deque(maxlen=window_size)

    def update(self, snapshot: OrderBookSnapshot, trade: Optional[TradeObservation] = None):
        """Update behavior detection with new data"""
        self._spread_history.append(snapshot.spread)
        self._imbalance_history.append(snapshot.imbalance)
        self._quote_update_times.append(snapshot.timestamp)

        if trade:
            self._trade_directions.append(1 if trade.side == "buy" else -1)

    def detect_behavior(self) -> Tuple[MMBehavior, float]:
        """Detect current MM behavior pattern"""
        if len(self._spread_history) < 20:
            return MMBehavior.NEUTRAL, 0.0

        behaviors = []
        confidences = []

        # Spread analysis
        spread_behavior, spread_conf = self._analyze_spread_behavior()
        if spread_behavior:
            behaviors.append(spread_behavior)
            confidences.append(spread_conf)

        # Imbalance analysis
        imbalance_behavior, imbalance_conf = self._analyze_imbalance_behavior()
        if imbalance_behavior:
            behaviors.append(imbalance_behavior)
            confidences.append(imbalance_conf)

        # Trade direction analysis
        trade_behavior, trade_conf = self._analyze_trade_behavior()
        if trade_behavior:
            behaviors.append(trade_behavior)
            confidences.append(trade_conf)

        if not behaviors:
            return MMBehavior.NEUTRAL, 0.0

        # Return most confident behavior
        max_idx = np.argmax(confidences)
        return behaviors[max_idx], confidences[max_idx]

    def _analyze_spread_behavior(self) -> Tuple[Optional[MMBehavior], float]:
        """Analyze spread changes"""
        spreads = list(self._spread_history)

        if len(spreads) < 20:
            return None, 0.0

        recent = spreads[-20:]
        older = spreads[-40:-20] if len(spreads) >= 40 else spreads[:20]

        recent_avg = np.mean(recent)
        older_avg = np.mean(older)

        change_pct = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        if change_pct > 0.1:
            return MMBehavior.WIDENING, min(0.8, abs(change_pct))
        elif change_pct < -0.1:
            return MMBehavior.TIGHTENING, min(0.8, abs(change_pct))

        return None, 0.0

    def _analyze_imbalance_behavior(self) -> Tuple[Optional[MMBehavior], float]:
        """Analyze order book imbalance"""
        imbalances = list(self._imbalance_history)

        if len(imbalances) < 20:
            return None, 0.0

        recent = imbalances[-20:]
        avg_imbalance = np.mean(recent)
        trend = np.polyfit(range(len(recent)), recent, 1)[0]

        # Significant positive imbalance trend = accumulating
        if avg_imbalance > 0.2 and trend > 0.01:
            return MMBehavior.ACCUMULATING, min(0.7, abs(avg_imbalance))
        # Significant negative imbalance trend = distributing
        elif avg_imbalance < -0.2 and trend < -0.01:
            return MMBehavior.DISTRIBUTING, min(0.7, abs(avg_imbalance))

        return None, 0.0

    def _analyze_trade_behavior(self) -> Tuple[Optional[MMBehavior], float]:
        """Analyze trade flow"""
        if len(self._trade_directions) < 20:
            return None, 0.0

        directions = list(self._trade_directions)[-20:]
        net_direction = sum(directions) / len(directions)

        if abs(net_direction) > 0.5:
            # One-sided trading = aggressive behavior
            return MMBehavior.AGGRESSIVE, min(0.8, abs(net_direction))
        elif abs(net_direction) < 0.1:
            # Balanced trading = passive/neutral
            return MMBehavior.PASSIVE, 0.5

        return None, 0.0


class PatternRecognizer:
    """Recognizes trading patterns in MM behavior"""

    def __init__(self):
        self._patterns: Dict[str, List[Dict]] = {}
        self._pattern_templates = self._init_templates()

    def _init_templates(self) -> Dict[str, np.ndarray]:
        """Initialize pattern templates"""
        return {
            "fade": np.array([1, 1, 1, -1, -1]),  # Buy then sell
            "momentum": np.array([1, 1, 1, 1, 1]),  # Consistent buying
            "reversal": np.array([-1, -1, 0, 1, 1]),  # Sell then buy
            "oscillation": np.array([1, -1, 1, -1, 1]),  # Back and forth
        }

    def match_pattern(self, trade_sequence: List[int]) -> Tuple[str, float]:
        """Match trade sequence to known patterns"""
        if len(trade_sequence) < 5:
            return "unknown", 0.0

        sequence = np.array(trade_sequence[-5:])

        best_match = "unknown"
        best_score = 0.0

        for name, template in self._pattern_templates.items():
            # Calculate correlation
            correlation = np.corrcoef(sequence, template)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            if correlation > best_score:
                best_score = correlation
                best_match = name

        return best_match if best_score > 0.5 else "unknown", max(0, best_score)

    def predict_next_trade(self, trade_sequence: List[int]) -> Tuple[int, float]:
        """Predict next trade direction"""
        pattern, confidence = self.match_pattern(trade_sequence)

        if pattern == "unknown" or confidence < 0.5:
            return 0, 0.0

        # Predict based on pattern
        if pattern == "fade":
            return -1, confidence  # Expect continued selling
        elif pattern == "momentum":
            return 1, confidence  # Expect continued buying
        elif pattern == "reversal":
            return 1, confidence  # Expect buying after reversal
        elif pattern == "oscillation":
            last = trade_sequence[-1] if trade_sequence else 0
            return -last, confidence  # Expect opposite

        return 0, 0.0


class QuoteAnalyzer:
    """Analyzes quote patterns for MM identification"""

    def __init__(self):
        self._quote_history: Dict[str, deque] = {}  # venue -> quotes
        self._identified_mms: Dict[str, MMProfile] = {}

    def process_orderbook(self, venue: str, market: str, snapshot: OrderBookSnapshot):
        """Process order book snapshot"""
        key = f"{venue}_{market}"

        if key not in self._quote_history:
            self._quote_history[key] = deque(maxlen=1000)

        self._quote_history[key].append(snapshot)

        # Analyze for MM signatures
        self._detect_mm_signatures(venue, market)

    def _detect_mm_signatures(self, venue: str, market: str):
        """Detect market maker signatures in quotes"""
        key = f"{venue}_{market}"
        history = self._quote_history.get(key, deque())

        if len(history) < 10:
            return

        recent = list(history)[-50:]

        # Look for consistent quote patterns
        # 1. Symmetric quotes around mid
        # 2. Consistent quote sizes
        # 3. Regular refresh intervals

        bid_sizes = [s.bids[0].quantity for s in recent if s.bids]
        ask_sizes = [s.asks[0].quantity for s in recent if s.asks]

        if not bid_sizes or not ask_sizes:
            return

        # Check for symmetric sizing
        size_ratio = np.mean(bid_sizes) / np.mean(ask_sizes) if np.mean(ask_sizes) > 0 else 0
        is_symmetric = 0.8 < size_ratio < 1.2

        # Check for consistent sizing (low variance)
        bid_cv = np.std(bid_sizes) / np.mean(bid_sizes) if np.mean(bid_sizes) > 0 else 1
        ask_cv = np.std(ask_sizes) / np.mean(ask_sizes) if np.mean(ask_sizes) > 0 else 1
        is_consistent = bid_cv < 0.3 and ask_cv < 0.3

        # If patterns match MM behavior, create/update profile
        if is_symmetric and is_consistent:
            mm_id = f"mm_{venue}_{market}"

            if mm_id not in self._identified_mms:
                self._identified_mms[mm_id] = MMProfile(
                    mm_id=mm_id,
                    venue=venue,
                    first_seen=time.time(),
                    last_seen=time.time()
                )

            mm = self._identified_mms[mm_id]
            mm.last_seen = time.time()
            mm.avg_quote_size = (np.mean(bid_sizes) + np.mean(ask_sizes)) / 2
            mm.avg_spread = np.mean([s.spread for s in recent])

    def get_identified_mms(self) -> List[MMProfile]:
        """Get list of identified market makers"""
        return list(self._identified_mms.values())


class AdvancedMMTracker:
    """
    Advanced market maker tracking with pattern recognition.

    Features:
    - MM identification from order flow
    - Inventory position estimation
    - Behavior pattern detection
    - Trade prediction
    - Lead-lag analysis
    """

    def __init__(self):
        self._mm_profiles: Dict[str, MMProfile] = {}
        self._inventory_estimators: Dict[str, InventoryEstimator] = {}
        self._behavior_detectors: Dict[str, BehaviorDetector] = {}
        self._pattern_recognizer = PatternRecognizer()
        self._quote_analyzer = QuoteAnalyzer()

        # Tracking data
        self._trade_history: Dict[str, deque] = {}
        self._orderbook_history: Dict[str, deque] = {}

        # Configuration
        self.mm_detection_threshold = 0.7
        self.inventory_decay = 0.995

    def process_trade(self, trade: TradeObservation):
        """Process an observed trade"""
        key = f"{trade.venue}_{trade.market}"

        # Initialize tracking if needed
        if key not in self._trade_history:
            self._trade_history[key] = deque(maxlen=10000)
            self._inventory_estimators[key] = InventoryEstimator(self.inventory_decay)
            self._behavior_detectors[key] = BehaviorDetector()

        self._trade_history[key].append(trade)

        # Check if this looks like MM trade
        is_mm_trade = self._classify_trade(trade)

        # Update inventory estimate
        self._inventory_estimators[key].update(trade, is_mm_trade)

        # Update MM profile if exists
        mm_id = self._identify_mm(trade)
        if mm_id and mm_id in self._mm_profiles:
            mm = self._mm_profiles[mm_id]
            mm.last_seen = trade.timestamp
            mm.total_trades_observed += 1
            mm.total_volume_observed += trade.quantity
            mm.price_levels_used.add(trade.price)

    def process_orderbook(self, venue: str, market: str, bids: List[Dict], asks: List[Dict]):
        """Process order book update"""
        # Convert to OrderBookLevels
        bid_levels = [
            OrderBookLevel(
                price=b["price"],
                quantity=b["quantity"],
                order_count=b.get("count", 1)
            )
            for b in bids
        ]
        ask_levels = [
            OrderBookLevel(
                price=a["price"],
                quantity=a["quantity"],
                order_count=a.get("count", 1)
            )
            for a in asks
        ]

        snapshot = OrderBookSnapshot(
            bids=bid_levels,
            asks=ask_levels,
            timestamp=time.time()
        )

        key = f"{venue}_{market}"

        if key not in self._orderbook_history:
            self._orderbook_history[key] = deque(maxlen=1000)
            self._behavior_detectors[key] = BehaviorDetector()

        self._orderbook_history[key].append(snapshot)

        # Update behavior detector
        self._behavior_detectors[key].update(snapshot)

        # Analyze for MM signatures
        self._quote_analyzer.process_orderbook(venue, market, snapshot)

    def _classify_trade(self, trade: TradeObservation) -> bool:
        """Classify if trade is likely from MM"""
        # Simple heuristics - in production would use more sophisticated methods
        key = f"{trade.venue}_{trade.market}"
        history = self._trade_history.get(key, deque())

        if len(history) < 10:
            return False

        recent = list(history)[-10:]

        # Check for MM-like patterns:
        # 1. Symmetric trading (buys ~ sells)
        buys = sum(1 for t in recent if t.side == "buy")
        sells = len(recent) - buys
        is_balanced = 0.3 < buys / len(recent) < 0.7 if recent else False

        # 2. Consistent sizing
        sizes = [t.quantity for t in recent]
        size_cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 1
        is_consistent = size_cv < 0.5

        return is_balanced and is_consistent

    def _identify_mm(self, trade: TradeObservation) -> Optional[str]:
        """Try to identify which MM made the trade"""
        if trade.maker_id:
            return trade.maker_id

        # Use heuristics
        key = f"{trade.venue}_{trade.market}"

        # Check identified MMs from quote analysis
        for mm in self._quote_analyzer.get_identified_mms():
            if mm.venue == trade.venue:
                # Check if price matches MM levels
                if trade.price in mm.price_levels_used:
                    return mm.mm_id

        return None

    def get_mm_profile(self, mm_id: str) -> Optional[MMProfile]:
        """Get profile for a specific MM"""
        # Check our profiles
        if mm_id in self._mm_profiles:
            return self._mm_profiles[mm_id]

        # Check quote analyzer
        for mm in self._quote_analyzer.get_identified_mms():
            if mm.mm_id == mm_id:
                return mm

        return None

    def get_inventory_estimate(self, venue: str, market: str) -> Dict:
        """Get estimated MM inventory for a market"""
        key = f"{venue}_{market}"
        estimator = self._inventory_estimators.get(key)

        if not estimator:
            return {
                "estimate": 0,
                "state": InventoryState.NEUTRAL.value,
                "confidence": 0
            }

        return {
            "estimate": estimator.estimate,
            "state": estimator.state.value,
            "confidence": estimator.confidence
        }

    def get_behavior(self, venue: str, market: str) -> Dict:
        """Get detected MM behavior"""
        key = f"{venue}_{market}"
        detector = self._behavior_detectors.get(key)

        if not detector:
            return {
                "behavior": MMBehavior.NEUTRAL.value,
                "confidence": 0
            }

        behavior, confidence = detector.detect_behavior()

        return {
            "behavior": behavior.value,
            "confidence": confidence
        }

    def predict_mm_action(self, venue: str, market: str) -> Dict:
        """Predict next MM action"""
        key = f"{venue}_{market}"
        history = self._trade_history.get(key, deque())

        if len(history) < 5:
            return {
                "prediction": "neutral",
                "confidence": 0,
                "pattern": "unknown"
            }

        # Get trade directions
        directions = [1 if t.side == "buy" else -1 for t in list(history)[-20:]]

        # Match pattern
        pattern, pattern_conf = self._pattern_recognizer.match_pattern(directions)

        # Predict next
        direction, pred_conf = self._pattern_recognizer.predict_next_trade(directions)

        prediction = "buy" if direction > 0 else ("sell" if direction < 0 else "neutral")

        return {
            "prediction": prediction,
            "confidence": pred_conf,
            "pattern": pattern,
            "pattern_confidence": pattern_conf
        }

    def get_all_mm_profiles(self) -> List[Dict]:
        """Get all identified MM profiles"""
        profiles = []

        # From quote analyzer
        for mm in self._quote_analyzer.get_identified_mms():
            # Add inventory and behavior
            inventory = self.get_inventory_estimate(mm.venue, mm.mm_id.split("_")[-1])
            behavior = self.get_behavior(mm.venue, mm.mm_id.split("_")[-1])

            mm.estimated_inventory = inventory["estimate"]
            mm.inventory_state = InventoryState(inventory["state"])
            mm.inventory_confidence = inventory["confidence"]
            mm.current_behavior = MMBehavior(behavior["behavior"])
            mm.behavior_confidence = behavior["confidence"]

            profiles.append(mm.to_dict())

        return profiles

    def get_market_summary(self, venue: str, market: str) -> Dict:
        """Get comprehensive market summary"""
        inventory = self.get_inventory_estimate(venue, market)
        behavior = self.get_behavior(venue, market)
        prediction = self.predict_mm_action(venue, market)

        key = f"{venue}_{market}"
        ob_history = self._orderbook_history.get(key, deque())

        # Calculate additional metrics
        if ob_history:
            recent = list(ob_history)[-20:]
            avg_spread = np.mean([s.spread for s in recent])
            avg_imbalance = np.mean([s.imbalance for s in recent])
            spread_volatility = np.std([s.spread for s in recent])
        else:
            avg_spread = 0
            avg_imbalance = 0
            spread_volatility = 0

        return {
            "venue": venue,
            "market": market,
            "mm_inventory": inventory,
            "mm_behavior": behavior,
            "mm_prediction": prediction,
            "market_metrics": {
                "avg_spread": avg_spread,
                "avg_imbalance": avg_imbalance,
                "spread_volatility": spread_volatility
            },
            "timestamp": time.time()
        }

    def get_trading_signal(self, venue: str, market: str) -> Dict:
        """Get trading signal based on MM analysis"""
        summary = self.get_market_summary(venue, market)

        # Generate signal
        inventory_state = summary["mm_inventory"]["state"]
        behavior = summary["mm_behavior"]["behavior"]
        prediction = summary["mm_prediction"]["prediction"]

        signal = "neutral"
        confidence = 0.0

        # Trading logic based on MM analysis
        if inventory_state == "extreme_long" and behavior == "distributing":
            signal = "sell"  # MM likely to push prices down
            confidence = 0.7
        elif inventory_state == "extreme_short" and behavior == "accumulating":
            signal = "buy"  # MM likely to push prices up
            confidence = 0.7
        elif behavior == "aggressive" and prediction != "neutral":
            signal = prediction
            confidence = summary["mm_prediction"]["confidence"] * 0.8
        elif inventory_state in ["long", "extreme_long"]:
            signal = "sell"  # Fade the MM
            confidence = 0.5
        elif inventory_state in ["short", "extreme_short"]:
            signal = "buy"  # Fade the MM
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "reasoning": {
                "inventory_state": inventory_state,
                "behavior": behavior,
                "prediction": prediction
            },
            "timestamp": time.time()
        }


# Test function
async def test_advanced_mm_tracker():
    """Test the advanced MM tracker"""
    tracker = AdvancedMMTracker()

    print("Testing Advanced MM Tracker...\n")

    # Simulate order book updates
    for i in range(100):
        mid_price = 100 + np.sin(i / 10) * 2

        bids = [
            {"price": mid_price - 0.1 * (j + 1), "quantity": 100 + np.random.rand() * 50}
            for j in range(5)
        ]
        asks = [
            {"price": mid_price + 0.1 * (j + 1), "quantity": 100 + np.random.rand() * 50}
            for j in range(5)
        ]

        tracker.process_orderbook("kraken", "BTC", bids, asks)

        # Simulate trades
        if i % 3 == 0:
            trade = TradeObservation(
                trade_id=f"trade_{i}",
                venue="kraken",
                market="BTC",
                side="buy" if np.random.rand() > 0.4 else "sell",
                price=mid_price + np.random.randn() * 0.1,
                quantity=10 + np.random.rand() * 5,
                timestamp=time.time()
            )
            tracker.process_trade(trade)

        await asyncio.sleep(0.01)

    # Get results
    print("Market Summary:")
    summary = tracker.get_market_summary("kraken", "BTC")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nTrading Signal:")
    signal = tracker.get_trading_signal("kraken", "BTC")
    for key, value in signal.items():
        print(f"  {key}: {value}")

    print("\nIdentified MMs:")
    profiles = tracker.get_all_mm_profiles()
    for profile in profiles:
        print(f"  {profile}")


if __name__ == "__main__":
    asyncio.run(test_advanced_mm_tracker())
