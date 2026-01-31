"""
Opportunity Detector Module
Finds arbitrage opportunities between venues
Simple: buy cheap, sell expensive
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ArbitrageOpportunity:
    """Represents one potential arb trade"""
    market: str
    buy_venue: str
    buy_price: float
    buy_liquidity: float  # How much liquidity available at buy price
    sell_venue: str
    sell_price: float
    sell_liquidity: float  # How much liquidity available at sell price
    spread: float  # Difference between prices
    spread_percent: float  # As percentage
    potential_profit: float  # Before fees
    liquidity_score: float  # Weighted by available liquidity
    fill_time_estimate_ms: float  # How long to fill this order
    timestamp: float

    def to_dict(self) -> Dict:
        return {
            "market": self.market,
            "buy_venue": self.buy_venue,
            "buy_price": self.buy_price,
            "buy_liquidity": self.buy_liquidity,
            "sell_venue": self.sell_venue,
            "sell_price": self.sell_price,
            "sell_liquidity": self.sell_liquidity,
            "spread": self.spread,
            "spread_percent": self.spread_percent,
            "potential_profit": self.potential_profit,
            "liquidity_score": self.liquidity_score,
            "fill_time_estimate_ms": self.fill_time_estimate_ms,
            "timestamp": self.timestamp,
        }


class OpportunityDetector:
    """Scans for arbitrage opportunities between venues"""
    
    def __init__(self, min_spread_percent: float = 0.5):
        """
        Args:
            min_spread_percent: Only flag opportunities above this % threshold
        """
        self.min_spread_percent = min_spread_percent
        self.opportunities = []
    
    def detect_opportunities(self, prices: Dict[str, Dict], liquidity_data: Dict = None) -> List[ArbitrageOpportunity]:
        """
        Detect arb opportunities from normalized prices with liquidity weighting.

        Input format:
        {
            "BTC": {
                "polymarket": {"price": 42500.00, "bid_qty": 5.0, "ask_qty": 4.5},
                "kraken": {"price": 42650.00, "bid_qty": 50.0, "ask_qty": 45.0},
            }
        }

        This version weighs opportunities by available liquidity.
        A 1% spread with 0.1 BTC liquidity << 1% spread with 100 BTC liquidity
        """

        opportunities = []

        for symbol, venue_data in prices.items():
            if len(venue_data) < 2:
                continue  # Need at least 2 venues to arb

            # Get all venue prices and liquidity
            venues = list(venue_data.items())

            # Compare every pair of venues
            for i in range(len(venues)):
                for j in range(i + 1, len(venues)):
                    venue1_name, venue1_data = venues[i]
                    venue2_name, venue2_data = venues[j]

                    # Handle both dict and simple float formats
                    price1 = venue1_data.get("price", venue1_data) if isinstance(venue1_data, dict) else venue1_data
                    price2 = venue2_data.get("price", venue2_data) if isinstance(venue2_data, dict) else venue2_data

                    liquidity1 = venue1_data.get("ask_qty", 1.0) if isinstance(venue1_data, dict) else 1.0
                    liquidity2 = venue2_data.get("bid_qty", 1.0) if isinstance(venue2_data, dict) else 1.0

                    # Check both directions
                    if price1 < price2:
                        arb = self._create_opportunity(
                            symbol, venue1_name, price1, liquidity1, venue2_name, price2, liquidity2
                        )
                    else:
                        arb = self._create_opportunity(
                            symbol, venue2_name, price2, liquidity2, venue1_name, price1, liquidity1
                        )

                    if arb:
                        opportunities.append(arb)

        # Sort by liquidity-weighted score (not just raw spread)
        opportunities.sort(key=lambda x: x.liquidity_score, reverse=True)
        self.opportunities = opportunities

        return opportunities
    
    def _create_opportunity(self, symbol: str, buy_venue: str, buy_price: float,
                           buy_liquidity: float, sell_venue: str, sell_price: float,
                           sell_liquidity: float) -> Optional[ArbitrageOpportunity]:
        """Create an opportunity if spread meets threshold"""

        if buy_price <= 0 or sell_price <= 0:
            return None

        spread = sell_price - buy_price
        spread_percent = (spread / buy_price) * 100

        # Only return if meets minimum threshold
        if spread_percent < self.min_spread_percent:
            return None

        # Liquidity-weighted scoring
        # Available liquidity limits position size
        available_liquidity = min(buy_liquidity, sell_liquidity)

        # Liquidity score: prefer high spread with good liquidity over tiny spread with huge liquidity
        # Score = spread%  available_liquidity / fill_time
        fill_time_estimate = self._estimate_fill_time(available_liquidity, symbol)
        liquidity_score = (spread_percent * available_liquidity) / max(fill_time_estimate, 0.1)

        return ArbitrageOpportunity(
            market=symbol,
            buy_venue=buy_venue,
            buy_price=buy_price,
            buy_liquidity=buy_liquidity,
            sell_venue=sell_venue,
            sell_price=sell_price,
            sell_liquidity=sell_liquidity,
            spread=spread,
            spread_percent=spread_percent,
            potential_profit=spread,  # Before fees
            liquidity_score=liquidity_score,
            fill_time_estimate_ms=fill_time_estimate,
            timestamp=datetime.now().timestamp(),
        )

    @staticmethod
    def _estimate_fill_time(available_liquidity: float, symbol: str) -> float:
        """
        Estimate fill time in milliseconds based on liquidity

        Conservative estimate: assume 1 BTC filled per 500ms
        """

        if symbol in ["BTC"]:
            fill_rate = 1.0  # 1 BTC per 500ms
        elif symbol in ["ETH"]:
            fill_rate = 10.0  # 10 ETH per 500ms
        else:
            fill_rate = 100.0  # 100 tokens per 500ms

        fill_time = (available_liquidity / fill_rate) * 500
        return max(fill_time, 50)  # Min 50ms, max estimate
    
    def filter_by_venue(self, venue_name: str) -> List[ArbitrageOpportunity]:
        """Get opportunities involving specific venue"""
        return [
            opp for opp in self.opportunities
            if opp.buy_venue == venue_name or opp.sell_venue == venue_name
        ]
    
    def filter_by_threshold(self, min_percent: float) -> List[ArbitrageOpportunity]:
        """Get opportunities above spread threshold"""
        return [
            opp for opp in self.opportunities
            if opp.spread_percent >= min_percent
        ]
    
    def get_top_opportunities(self, limit: int = 10) -> List[ArbitrageOpportunity]:
        """Get top N opportunities by spread"""
        return self.opportunities[:limit]


# Test the module
def test_opportunity_detector():
    """Test with sample price data"""
    
    detector = OpportunityDetector(min_spread_percent=0.5)
    
    # Sample price data from different venues
    test_prices = {
        "BTC": {
            "polymarket": 42500.00,
            "kraken": 42650.00,      # 0.35% spread
            "coinbase": 42600.00,    # 0.23% spread from kraken
        },
        "ETH": {
            "polymarket": 2300.00,
            "kraken": 2330.00,       # 1.30% spread
            "coinbase": 2320.00,     # 0.43% spread from kraken
        },
        "DOGE": {
            "polymarket": 0.45,
            "kraken": 0.52,          # 15.56% spread (very profitable!)
        }
    }
    
    print("Scanning for opportunities...\n")
    opportunities = detector.detect_opportunities(test_prices)
    
    print(f"Found {len(opportunities)} opportunities:\n")
    
    for opp in detector.get_top_opportunities(5):
        print(f" {opp.market}")
        print(f"   Buy on {opp.buy_venue}: ${opp.buy_price:.2f}")
        print(f"   Sell on {opp.sell_venue}: ${opp.sell_price:.2f}")
        print(f"   Spread: {opp.spread_percent:.2f}% (${opp.spread:.2f})")
        print(f"   Profit (before fees): ${opp.potential_profit:.2f}\n")
    
    # Filter by threshold
    print("---")
    print("Top opportunities above 1% spread:")
    high_spread = detector.filter_by_threshold(1.0)
    for opp in high_spread:
        print(f"  {opp.market}: {opp.spread_percent:.2f}% on {opp.buy_venue}  {opp.sell_venue}")


if __name__ == "__main__":
    test_opportunity_detector()