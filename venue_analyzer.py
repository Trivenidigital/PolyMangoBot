"""
Venue Analysis Module
Detects lead-lag relationships between venues and predicts price movements
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import statistics


@dataclass
class PriceEvent:
    """Single price observation"""
    venue: str
    symbol: str
    price: float
    timestamp: float  # milliseconds


class VenueAnalyzer:
    """Analyzes relationships between trading venues"""

    def __init__(self, lookback_period: int = 100):
        """
        Args:
            lookback_period: How many price events to keep for analysis
        """
        self.lookback_period = lookback_period
        self.price_history: Dict[str, deque] = {}  # symbol -> events
        self.lead_lag_cache: Dict[Tuple[str, str, str], float] = {}  # (symbol, lead_venue, lag_venue) -> lag_ms

    def add_price_event(self, event: PriceEvent):
        """Record a price change event"""
        key = event.symbol

        if key not in self.price_history:
            self.price_history[key] = deque(maxlen=self.lookback_period)

        self.price_history[key].append(event)

    def detect_lead_lag(self, symbol: str) -> Dict:
        """
        Detect which venue leads and which lags

        Example: If Polymarket price always updates 80ms after Kraken,
        then Kraken leads and Polymarket lags by 80ms.

        Returns:
            {
                'lead_venue': 'kraken',
                'lag_venue': 'polymarket',
                'lag_ms': 80,
                'confidence': 0.8,
                'lead_ratio': 0.75  # How often lead venue moves first
            }
        """
        if symbol not in self.price_history:
            return {'lead_venue': None, 'lag_venue': None, 'lag_ms': 0, 'confidence': 0}

        events = list(self.price_history[symbol])
        if len(events) < 10:
            return {'lead_venue': None, 'lag_venue': None, 'lag_ms': 0, 'confidence': 0}

        # Group events by venue
        venues = {}
        for event in events:
            if event.venue not in venues:
                venues[event.venue] = []
            venues[event.venue].append(event)

        if len(venues) < 2:
            return {'lead_venue': None, 'lag_venue': None, 'lag_ms': 0, 'confidence': 0}

        # Compare timing between venues
        venue_names = list(venues.keys())
        venue1, venue2 = venue_names[0], venue_names[1]

        # For each price change in venue1, find corresponding change in venue2
        lags = []

        for v1_event in venues[venue1]:
            # Find closest price change in venue2 (within 500ms)
            for v2_event in venues[venue2]:
                if abs(v2_event.timestamp - v1_event.timestamp) < 500:
                    lag = v2_event.timestamp - v1_event.timestamp
                    lags.append(lag)
                    break

        if not lags:
            return {'lead_venue': None, 'lag_venue': None, 'lag_ms': 0, 'confidence': 0}

        # Median lag is most reliable
        median_lag = statistics.median(lags)
        mean_lag = statistics.mean(lags)

        # Determine lead/lag
        if median_lag > 0:
            lead_venue = venue1
            lag_venue = venue2
            lag_ms = median_lag
        else:
            lead_venue = venue2
            lag_venue = venue1
            lag_ms = abs(median_lag)

        # Confidence: how consistent is the lag?
        if len(lags) > 1:
            stdev = statistics.stdev(lags)
            consistency = 1.0 - min(stdev / lag_ms, 1.0) if lag_ms > 0 else 0
        else:
            consistency = 0

        lead_count = sum(1 for lag in lags if lag > 0)
        lead_ratio = lead_count / len(lags) if lags else 0

        return {
            'lead_venue': lead_venue,
            'lag_venue': lag_venue,
            'lag_ms': lag_ms,
            'confidence': consistency,
            'lead_ratio': lead_ratio,
            'num_observations': len(lags)
        }

    def predict_next_price(self, symbol: str, venue: str) -> Dict:
        """
        Using lead-lag relationships, predict where price will be next

        If venue A leads venue B by 50ms, and venue A just moved up 0.5%,
        then venue B is likely to follow with similar move.

        Returns:
            {
                'predicted_move': float,  # % expected change
                'confidence': float,  # 0-1
                'time_horizon_ms': int,
                'trigger_venue': str  # Which venue moved first
            }
        """

        if symbol not in self.price_history:
            return {'predicted_move': 0, 'confidence': 0}

        events = list(self.price_history[symbol])
        if len(events) < 2:
            return {'predicted_move': 0, 'confidence': 0}

        # Get most recent event
        latest = events[-1]

        # Check if this is the lead or lag venue
        lead_lag = self.detect_lead_lag(symbol)

        if lead_lag['lag_ms'] == 0:
            return {'predicted_move': 0, 'confidence': 0}

        if latest.venue == lead_lag['lead_venue']:
            # Lead venue just moved - predict lag venue will follow
            prev_lead = events[-2] if len(events) > 1 else None

            if prev_lead:
                move_pct = ((latest.price - prev_lead.price) / prev_lead.price) * 100
            else:
                move_pct = 0

            return {
                'predicted_move': move_pct,
                'confidence': lead_lag['confidence'],
                'time_horizon_ms': lead_lag['lag_ms'],
                'trigger_venue': latest.venue,
                'lag_venue': lead_lag['lag_venue']
            }

        return {'predicted_move': 0, 'confidence': 0}

    def detect_arbitrage_window(self, symbol: str) -> Dict:
        """
        Detect when an arbitrage opportunity is about to appear

        If lead venue creates a spread, lag venue will replicate it.
        You can trade BEFORE the lag venue updates.

        Returns:
            {
                'arbitrage_signal': bool,
                'direction': 'buy_lag_sell_lead' | 'buy_lead_sell_lag',
                'expected_spread': float,  # %
                'window_ms': float,  # How long before it disappears
                'confidence': float
            }
        """

        lead_lag = self.detect_lead_lag(symbol)

        if lead_lag['lag_ms'] == 0 or lead_lag['confidence'] < 0.5:
            return {'arbitrage_signal': False, 'confidence': 0}

        # Check if there's a recent price divergence
        if symbol not in self.price_history:
            return {'arbitrage_signal': False, 'confidence': 0}

        events = list(self.price_history[symbol])

        # Get last prices from each venue
        lead_price = None
        lag_price = None

        for event in reversed(events):
            if event.venue == lead_lag['lead_venue'] and lead_price is None:
                lead_price = event.price
            elif event.venue == lead_lag['lag_venue'] and lag_price is None:
                lag_price = event.price

            if lead_price and lag_price:
                break

        if not lead_price or not lag_price:
            return {'arbitrage_signal': False, 'confidence': 0}

        spread_pct = ((lead_price - lag_price) / lag_price) * 100

        # Signal if spread is meaningful
        has_signal = abs(spread_pct) > 0.05  # > 0.05% spread

        direction = 'buy_lag_sell_lead' if spread_pct > 0 else 'buy_lead_sell_lag'

        return {
            'arbitrage_signal': has_signal,
            'direction': direction,
            'expected_spread': abs(spread_pct),
            'window_ms': lead_lag['lag_ms'] * 0.8,  # Arbitrage window is ~80% of lag time
            'confidence': lead_lag['confidence'] if has_signal else 0
        }


class MultiVenueCorrelation:
    """Analyzes correlations between venues"""

    def __init__(self):
        self.venue_prices: Dict[str, List[float]] = {}
        self.correlation_cache: Dict[Tuple[str, str], float] = {}

    def update_prices(self, symbol: str, venue_prices: Dict[str, float]):
        """
        Update prices for symbol across venues

        Args:
            symbol: Trading pair
            venue_prices: {'kraken': 42500.00, 'polymarket': 42450.00}
        """

        key = f"{symbol}_prices"
        if key not in self.venue_prices:
            self.venue_prices[key] = []

        self.venue_prices[key].append(venue_prices)

        # Keep last 100 snapshots
        if len(self.venue_prices[key]) > 100:
            self.venue_prices[key] = self.venue_prices[key][-100:]

    def calculate_correlation(self, symbol: str, venue1: str, venue2: str) -> float:
        """
        Calculate price correlation between two venues

        Returns: -1 to 1 (1 = perfectly correlated, -1 = inversely correlated)
        """

        cache_key = (symbol, venue1, venue2)
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]

        key = f"{symbol}_prices"
        if key not in self.venue_prices:
            return 0

        snapshots = self.venue_prices[key]
        if len(snapshots) < 2:
            return 0

        # Extract price series for each venue
        v1_prices = [s.get(venue1, 0) for s in snapshots if venue1 in s and s[venue1] > 0]
        v2_prices = [s.get(venue2, 0) for s in snapshots if venue2 in s and s[venue2] > 0]

        if len(v1_prices) < 2 or len(v2_prices) < 2:
            return 0

        # Simple correlation calculation
        if len(v1_prices) != len(v2_prices):
            min_len = min(len(v1_prices), len(v2_prices))
            v1_prices = v1_prices[-min_len:]
            v2_prices = v2_prices[-min_len:]

        try:
            mean1 = statistics.mean(v1_prices)
            mean2 = statistics.mean(v2_prices)

            covariance = sum(
                (v1_prices[i] - mean1) * (v2_prices[i] - mean2)
                for i in range(len(v1_prices))
            ) / len(v1_prices)

            std1 = statistics.stdev(v1_prices)
            std2 = statistics.stdev(v2_prices)

            correlation = covariance / (std1 * std2) if std1 > 0 and std2 > 0 else 0

            self.correlation_cache[cache_key] = correlation
            return correlation

        except:
            return 0


# Test
def test_venue_analyzer():
    """Test venue analysis"""

    analyzer = VenueAnalyzer()

    # Simulate price events: Kraken leads by ~50ms
    base_time = datetime.now().timestamp() * 1000

    # Kraken moves first
    analyzer.add_price_event(PriceEvent("kraken", "BTC", 42500, base_time))
    # Polymarket follows 50ms later
    analyzer.add_price_event(PriceEvent("polymarket", "BTC", 42450, base_time + 50))

    # Kraken moves again
    analyzer.add_price_event(PriceEvent("kraken", "BTC", 42510, base_time + 1000))
    # Polymarket follows
    analyzer.add_price_event(PriceEvent("polymarket", "BTC", 42460, base_time + 1050))

    lead_lag = analyzer.detect_lead_lag("BTC")
    print(f"Lead-lag detection: {lead_lag}")

    arb = analyzer.detect_arbitrage_window("BTC")
    print(f"Arbitrage window: {arb}")


if __name__ == "__main__":
    test_venue_analyzer()
