"""
Order Book Analysis Module
Analyzes order book depth, patterns, and predicts liquidity changes
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import statistics


@dataclass
class OrderBookSnapshot:
    """Single point-in-time order book state"""
    exchange: str
    symbol: str
    timestamp: float
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]
    mid_price: float
    spread: float

    @property
    def bid_depth_at_level(self, level: int = 0) -> Tuple[float, float]:
        """Get bid price and quantity at specific level"""
        if level < len(self.bids):
            return self.bids[level]
        return (0, 0)

    @property
    def ask_depth_at_level(self, level: int = 0) -> Tuple[float, float]:
        """Get ask price and quantity at specific level"""
        if level < len(self.asks):
            return self.asks[level]
        return (0, 0)


class OrderBookAnalyzer:
    """Analyzes order book patterns and predicts liquidity changes"""

    def __init__(self, max_history: int = 100):
        """
        Args:
            max_history: Keep last N snapshots for analysis
        """
        self.max_history = max_history
        self.history: Dict[str, deque] = {}  # symbol -> snapshots
        self.spread_history: Dict[str, deque] = {}  # symbol -> spread changes

    def add_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Add new order book snapshot"""
        key = f"{snapshot.exchange}:{snapshot.symbol}"

        if key not in self.history:
            self.history[key] = deque(maxlen=self.max_history)
            self.spread_history[key] = deque(maxlen=self.max_history)

        self.history[key].append(snapshot)
        self.spread_history[key].append(snapshot.spread)

    def get_total_depth(self, exchange: str, symbol: str, side: str = "bid",
                       levels: int = 5) -> float:
        """
        Calculate total quantity available at best prices

        Args:
            side: "bid" or "ask"
            levels: How many price levels to sum
        """
        key = f"{exchange}:{symbol}"
        if key not in self.history or not self.history[key]:
            return 0

        snapshot = self.history[key][-1]
        orders = snapshot.bids if side == "bid" else snapshot.asks

        total = sum(qty for _, qty in orders[:levels])
        return total

    def detect_accumulation_pattern(self, exchange: str, symbol: str) -> Dict:
        """
        Detect if market-maker is accumulating inventory

        Patterns:
        - Placing large buy orders at support levels
        - Removing sell orders
        → Next: MM will likely widen asks to rebalance

        Returns:
            {
                'is_accumulating': bool,
                'confidence': float (0-1),
                'side': 'buy' or 'sell',
                'predicted_next_move': 'spread_widen' or 'spread_tighten'
            }
        """
        key = f"{exchange}:{symbol}"
        if key not in self.history or len(self.history[key]) < 5:
            return {'is_accumulating': False, 'confidence': 0.0}

        recent = list(self.history[key])[-5:]

        # Check bid depth trend
        bid_depths = []
        ask_depths = []

        for snap in recent:
            bid_depth = self.get_total_depth(exchange, symbol, "bid", levels=3)
            ask_depth = self.get_total_depth(exchange, symbol, "ask", levels=3)
            bid_depths.append(bid_depth)
            ask_depths.append(ask_depth)

        # Detect pattern: bids growing, asks shrinking
        bid_trend = bid_depths[-1] > statistics.mean(bid_depths[:-1]) * 1.2
        ask_trend = ask_depths[-1] < statistics.mean(ask_depths[:-1]) * 0.8

        is_accumulating = bid_trend and ask_trend
        confidence = 0.7 if is_accumulating else 0.2

        return {
            'is_accumulating': is_accumulating,
            'confidence': confidence,
            'side': 'buy',
            'predicted_next_move': 'spread_widen' if is_accumulating else 'neutral'
        }

    def calculate_spread_volatility(self, exchange: str, symbol: str,
                                    window: int = 20) -> float:
        """
        Calculate how volatile spreads are

        High volatility = spreads widening/tightening rapidly
        Low volatility = stable spreads
        """
        key = f"{exchange}:{symbol}"
        if key not in self.spread_history or len(self.spread_history[key]) < window:
            return 0.0

        recent_spreads = list(self.spread_history[key])[-window:]
        if len(recent_spreads) < 2:
            return 0.0

        return statistics.stdev(recent_spreads)

    def predict_spread_movement(self, exchange: str, symbol: str) -> Dict:
        """
        Predict if spread will widen or tighten in near future

        Signals:
        - Increasing spread volatility = unpredictable (avoid)
        - MM accumulation = spread widening (trade now before it widens)
        - Order book imbalance = spread tightening (wait)

        Returns:
            {
                'prediction': 'widen' | 'tighten' | 'stable',
                'confidence': float (0-1),
                'time_horizon_ms': expected time to move
            }
        """
        volatility = self.calculate_spread_volatility(exchange, symbol)
        accumulation = self.detect_accumulation_pattern(exchange, symbol)

        if accumulation.get('predicted_next_move') == 'spread_widen':
            return {
                'prediction': 'widen',
                'confidence': accumulation['confidence'],
                'time_horizon_ms': 500  # Within 500ms
            }

        # If very high volatility, spreads are unpredictable
        if volatility > 0.05:  # > 0.05% std dev
            return {
                'prediction': 'volatile',
                'confidence': 0.5,
                'time_horizon_ms': 1000
            }

        return {
            'prediction': 'stable',
            'confidence': 0.6,
            'time_horizon_ms': 2000
        }

    def get_liquidity_density(self, exchange: str, symbol: str,
                             side: str = "bid", price_range: float = 0.01) -> float:
        """
        How much liquidity is available per unit price change

        High density = large trades won't slip much
        Low density = orders will slippage significantly
        """
        key = f"{exchange}:{symbol}"
        if key not in self.history or not self.history[key]:
            return 0

        snapshot = self.history[key][-1]
        orders = snapshot.bids if side == "bid" else snapshot.asks

        if not orders:
            return 0

        best_price = orders[0][0]
        range_min = best_price - (best_price * price_range / 100)
        range_max = best_price

        qty_in_range = sum(
            qty for price, qty in orders
            if range_min <= price <= range_max
        )

        # Return qty per 1% price move
        return qty_in_range / (price_range or 1)

    def estimate_fill_time(self, exchange: str, symbol: str, side: str,
                          quantity: float) -> float:
        """
        Estimate how long it will take to fill an order

        Returns: milliseconds
        """
        key = f"{exchange}:{symbol}"
        if key not in self.history or not self.history[key]:
            return 1000  # Default 1s

        snapshot = self.history[key][-1]
        orders = snapshot.bids if side == "buy" else snapshot.asks

        filled = 0.0
        for price, qty in orders:
            filled += qty
            if filled >= quantity:
                # Estimate: roughly 100ms per level + 50ms base
                levels_used = len([1 for p, q in orders if p >= orders[0][0] - (orders[0][0] * 0.001)])
                return max(50, levels_used * 100)

        # Can't fill entire quantity
        return 5000  # 5 seconds = can't fill

    def get_best_venue_for_side(self, venues: Dict[str, OrderBookSnapshot],
                                side: str) -> Tuple[str, float]:
        """
        Among available venues, which has best liquidity for this side?

        Returns: (venue_name, liquidity_score)
        """
        best_venue = None
        best_score = 0

        for venue_name, snapshot in venues.items():
            orders = snapshot.bids if side == "buy" else snapshot.asks

            # Score: quantity at best level
            qty_at_best = orders[0][1] if orders else 0

            # Penalize if spread is too wide
            spread_penalty = 1.0 - min(snapshot.spread * 10, 0.5)

            score = qty_at_best * spread_penalty

            if score > best_score:
                best_score = score
                best_venue = venue_name

        return best_venue, best_score


# Test
def test_order_book_analyzer():
    """Test the analyzer"""

    analyzer = OrderBookAnalyzer()

    # Create mock snapshots
    snap1 = OrderBookSnapshot(
        exchange="polymarket",
        symbol="BTC",
        timestamp=datetime.now().timestamp(),
        bids=[(42500, 1.5), (42490, 2.0), (42480, 3.0)],
        asks=[(42510, 1.0), (42520, 2.0), (42530, 2.5)],
        mid_price=42505,
        spread=10
    )

    snap2 = OrderBookSnapshot(
        exchange="kraken",
        symbol="BTC",
        timestamp=datetime.now().timestamp(),
        bids=[(42490, 5.0), (42480, 6.0), (42470, 7.0)],
        asks=[(42510, 4.0), (42520, 5.0), (42530, 6.0)],
        mid_price=42500,
        spread=20
    )

    analyzer.add_snapshot(snap1)
    analyzer.add_snapshot(snap2)

    print("Order Book Analysis:")
    print(f"Polymarket bid depth (3 levels): {analyzer.get_total_depth('polymarket', 'BTC', 'bid', 3)}")
    print(f"Kraken ask depth (3 levels): {analyzer.get_total_depth('kraken', 'BTC', 'ask', 3)}")

    accum = analyzer.detect_accumulation_pattern("polymarket", "BTC")
    print(f"Accumulation pattern: {accum}")

    movement = analyzer.predict_spread_movement("polymarket", "BTC")
    print(f"Spread prediction: {movement}")


if __name__ == "__main__":
    test_order_book_analyzer()
