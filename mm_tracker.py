"""
Market-Maker Tracking Module
Tracks MM behavior and predicts their next moves
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import statistics


@dataclass
class MMSnapshot:
    """Single observation of MM behavior"""
    timestamp: float
    bid_quantity: float
    ask_quantity: float
    spread: float
    bid_price: float
    ask_price: float
    inventory_ratio: float  # bid_qty / ask_qty


class MarketMakerTracker:
    """Tracks individual market-maker behavior"""

    def __init__(self, mm_id: str, lookback: int = 50):
        """
        Args:
            mm_id: Identifier for this MM (e.g., "mm_polymarket_1")
            lookback: How many snapshots to keep
        """
        self.mm_id = mm_id
        self.lookback = lookback
        self.snapshots: deque = deque(maxlen=lookback)

    def add_snapshot(self, snap: MMSnapshot):
        """Record MM state"""
        self.snapshots.append(snap)

    def get_inventory_trend(self) -> Dict:
        """
        Detect if MM is accumulating or distributing

        Returns:
            {
                'trend': 'accumulating_buy' | 'accumulating_sell' | 'neutral',
                'momentum': float (-1 to +1),
                'confidence': float (0-1)
            }
        """

        if len(self.snapshots) < 3:
            return {'trend': 'neutral', 'momentum': 0, 'confidence': 0}

        # Get inventory ratios over time
        ratios = [s.inventory_ratio for s in self.snapshots]

        # Trend: are they buying more (ratio decreasing) or selling more (ratio increasing)?
        recent_ratio = statistics.mean(ratios[-5:])
        old_ratio = statistics.mean(ratios[:5])

        momentum = (recent_ratio - old_ratio) / old_ratio if old_ratio > 0 else 0

        if momentum > 0.1:
            trend = 'accumulating_sell'
        elif momentum < -0.1:
            trend = 'accumulating_buy'
        else:
            trend = 'neutral'

        confidence = min(abs(momentum), 1.0)

        return {
            'trend': trend,
            'momentum': momentum,
            'confidence': confidence
        }

    def predict_next_spread_move(self) -> Dict:
        """
        Predict what MM will do with spreads next

        Logic:
        - If MM is buying heavily (short inventory): they'll widen asks to accumulate
        - If MM is selling heavily (long inventory): they'll widen bids to sell
        - If inventory is balanced: spreads should stay tight

        Returns:
            {
                'next_move': 'widen_bid' | 'widen_ask' | 'tighten' | 'neutral',
                'expected_spread_change': float,  # %
                'time_horizon_ms': int
            }
        """

        inventory = self.get_inventory_trend()

        if inventory['confidence'] < 0.3:
            return {
                'next_move': 'neutral',
                'expected_spread_change': 0,
                'time_horizon_ms': 5000
            }

        if inventory['trend'] == 'accumulating_buy':
            # MM wants to buy more, will make asks worse (wider)
            return {
                'next_move': 'widen_ask',
                'expected_spread_change': 0.3,  # Expect 0.3% wider ask
                'time_horizon_ms': 2000
            }

        elif inventory['trend'] == 'accumulating_sell':
            # MM wants to sell more, will make bids worse (wider)
            return {
                'next_move': 'widen_bid',
                'expected_spread_change': 0.3,
                'time_horizon_ms': 2000
            }

        return {
            'next_move': 'neutral',
            'expected_spread_change': 0,
            'time_horizon_ms': 5000
        }

    def get_volatility(self) -> float:
        """How volatile is this MM's spreads?"""

        if len(self.snapshots) < 2:
            return 0

        spreads = [s.spread for s in self.snapshots]
        return statistics.stdev(spreads) if len(spreads) > 1 else 0


class MultiMMAnalyzer:
    """
    Tracks multiple MMs on a venue and identifies patterns

    On Polymarket, there are probably 5-20 main MMs
    This module identifies them and tracks behavior
    """

    def __init__(self):
        self.mms: Dict[str, MarketMakerTracker] = {}

    def update_mm(self, mm_id: str, snap: MMSnapshot):
        """Update state of specific MM"""

        if mm_id not in self.mms:
            self.mms[mm_id] = MarketMakerTracker(mm_id)

        self.mms[mm_id].add_snapshot(snap)

    def identify_dominant_mm(self) -> Optional[str]:
        """
        Find which MM has largest inventory

        Returns: mm_id with most activity
        """

        if not self.mms:
            return None

        max_activity = 0
        dominant = None

        for mm_id, tracker in self.mms.items():
            if tracker.snapshots:
                activity = sum(s.bid_quantity + s.ask_quantity for s in tracker.snapshots)
                if activity > max_activity:
                    max_activity = activity
                    dominant = mm_id

        return dominant

    def detect_coordinated_action(self) -> Dict:
        """
        Detect if multiple MMs are moving together

        Example: If all MMs simultaneously widen spreads,
        there might be external event causing it.

        Returns:
            {
                'is_coordinated': bool,
                'action': 'spread_widening' | 'spread_tightening' | 'none',
                'num_mms_involved': int,
                'confidence': float
            }
        """

        if len(self.mms) < 2:
            return {'is_coordinated': False, 'confidence': 0}

        # Check spread trends across all MMs
        spreads = {}

        for mm_id, tracker in self.mms.items():
            if tracker.snapshots:
                recent_spread = tracker.snapshots[-1].spread
                old_spread = tracker.snapshots[0].spread if len(tracker.snapshots) > 0 else recent_spread
                spreads[mm_id] = (recent_spread - old_spread) / old_spread if old_spread > 0 else 0

        if not spreads:
            return {'is_coordinated': False, 'confidence': 0}

        # If >60% of MMs moved in same direction, it's coordinated
        spread_changes = list(spreads.values())
        widening = sum(1 for x in spread_changes if x > 0.01)
        tightening = sum(1 for x in spread_changes if x < -0.01)

        total = len(spread_changes)
        max_direction = max(widening, tightening)

        is_coordinated = max_direction / total > 0.6 if total > 0 else False

        action = 'spread_widening' if widening > tightening else 'spread_tightening' if tightening > widening else 'none'

        return {
            'is_coordinated': is_coordinated,
            'action': action,
            'num_mms_involved': max_direction,
            'confidence': max_direction / total if total > 0 else 0
        }

    def get_market_health(self) -> Dict:
        """
        Overall assessment of MM market conditions

        Returns:
            {
                'liquidity': 'high' | 'medium' | 'low',
                'stability': 'stable' | 'volatile' | 'chaotic',
                'health_score': float (0-1)
            }
        """

        if not self.mms:
            return {'liquidity': 'unknown', 'stability': 'unknown', 'health_score': 0}

        # Average volatility across all MMs
        volatilities = [tracker.get_volatility() for tracker in self.mms.values()]
        avg_volatility = statistics.mean(volatilities) if volatilities else 0

        # Average inventory balance
        balances = []

        for tracker in self.mms.values():
            if tracker.snapshots:
                recent = tracker.snapshots[-1]
                balance = recent.inventory_ratio  # 1.0 = perfectly balanced
                balances.append(balance)

        avg_balance = statistics.mean(balances) if balances else 1.0

        # Stability assessment
        if avg_volatility < 0.01:
            stability = 'stable'
        elif avg_volatility < 0.05:
            stability = 'volatile'
        else:
            stability = 'chaotic'

        # Liquidity assessment (based on number of MMs)
        if len(self.mms) < 3:
            liquidity = 'low'
        elif len(self.mms) < 10:
            liquidity = 'medium'
        else:
            liquidity = 'high'

        # Health score (0-1)
        # Higher is better: less volatility, more MMs, balanced inventory
        health = (
            (1.0 - min(avg_volatility * 100, 1.0)) * 0.5 +  # Low volatility is good
            (len(self.mms) / 20.0) * 0.3 +  # More MMs is good
            (1.0 / (abs(avg_balance - 1.0) + 0.1)) * 0.2  # Balanced is good
        )

        return {
            'liquidity': liquidity,
            'stability': stability,
            'health_score': min(health, 1.0),
            'num_active_mms': len(self.mms)
        }


class MMBehaviorPredictor:
    """
    Predicts what MMs will do based on recent patterns

    High-level logic for trading:
    - If MM is accumulating, they'll make unfavorable quotes soon
    - Better to trade NOW before quotes worsen
    """

    def __init__(self):
        self.mm_tracker = MultiMMAnalyzer()

    def should_trade_now(self, symbol: str, current_spread: float) -> Dict:
        """
        Determine if now is good time to trade based on MM behavior

        Returns:
            {
                'trade_signal': bool,
                'urgency': float (0-1),  # How urgent to execute
                'reason': str,
                'confidence': float
            }
        """

        # Check if MMs are coordinating (may indicate external event)
        coord = self.mm_tracker.detect_coordinated_action()

        if coord['action'] == 'spread_widening':
            # Spreads are widening - better to trade NOW before they get worse
            return {
                'trade_signal': True,
                'urgency': 0.8,
                'reason': 'Spreads widening across multiple MMs',
                'confidence': coord['confidence']
            }

        health = self.mm_tracker.get_market_health()

        if health['stability'] == 'stable' and health['liquidity'] == 'high':
            # Good conditions, spreads won't change much
            return {
                'trade_signal': True,
                'urgency': 0.5,
                'reason': 'Market stable with good liquidity',
                'confidence': 0.7
            }

        elif health['stability'] == 'chaotic':
            # Bad conditions, avoid trading
            return {
                'trade_signal': False,
                'urgency': 0,
                'reason': 'Market too chaotic, high MM volatility',
                'confidence': 0.8
            }

        return {
            'trade_signal': True,
            'urgency': 0.5,
            'reason': 'Neutral market conditions',
            'confidence': 0.5
        }


# Test
def test_mm_tracker():
    """Test MM tracking"""

    tracker = MarketMakerTracker("mm_1")

    # Simulate MM buying (inventory ratio decreasing)
    for i in range(10):
        snap = MMSnapshot(
            timestamp=datetime.now().timestamp(),
            bid_quantity=100 - i * 5,  # Decreasing bids
            ask_quantity=100,  # Stable asks
            spread=0.5,
            bid_price=100,
            ask_price=100.5,
            inventory_ratio=(100 - i * 5) / 100  # Decreasing
        )
        tracker.add_snapshot(snap)

    inventory = tracker.get_inventory_trend()
    print(f"Inventory trend: {inventory}")

    prediction = tracker.predict_next_spread_move()
    print(f"Predicted next move: {prediction}")


if __name__ == "__main__":
    test_mm_tracker()
