"""
Data Normalizer Module
Converts different API responses into standardized format
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class NormalizedOrderBook:
    """Standard format for any order book"""
    exchange: str
    symbol: str
    timestamp: float
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]  # [(price, quantity), ...]
    mid_price: float
    spread: float
    
    def to_dict(self) -> Dict:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "bids": self.bids,
            "asks": self.asks,
            "mid_price": self.mid_price,
            "spread": self.spread,
        }


@dataclass
class NormalizedPrice:
    """Standard price point"""
    exchange: str
    symbol: str
    price: float
    timestamp: float
    volume_24h: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "price": self.price,
            "timestamp": self.timestamp,
            "volume_24h": self.volume_24h,
        }


class DataNormalizer:
    """Converts various API responses to standard format"""
    
    @staticmethod
    def normalize_polymarket_orderbook(raw_data: Dict) -> Optional[NormalizedOrderBook]:
        """Convert Polymarket order book to standard format"""
        
        if not raw_data or "bids" not in raw_data:
            return None
        
        try:
            bids = [(float(b["price"]), float(b["quantity"])) for b in raw_data.get("bids", [])]
            asks = [(float(a["price"]), float(a["quantity"])) for a in raw_data.get("asks", [])]
            
            # Sort: highest bid first, lowest ask first
            bids.sort(reverse=True, key=lambda x: x[0])
            asks.sort(key=lambda x: x[0])
            
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            
            return NormalizedOrderBook(
                exchange="polymarket",
                symbol=raw_data.get("market_id", "unknown"),
                timestamp=datetime.now().timestamp(),
                bids=bids[:10],  # Top 10 levels
                asks=asks[:10],
                mid_price=mid_price,
                spread=spread,
            )
        except Exception as e:
            print(f"Normalization error: {e}")
            return None
    
    @staticmethod
    def normalize_kraken_ticker(raw_data: Dict, symbol: str) -> Optional[NormalizedPrice]:
        """Convert Kraken ticker to standard format"""
        
        if not raw_data:
            return None
        
        try:
            # Kraken format: {"pair": {"c": [price, count], ...}}
            pair_data = raw_data.get(symbol, {})
            last_price = float(pair_data.get("c", [0])[0])
            volume = float(pair_data.get("v", [0])[1]) if "v" in pair_data else None
            
            return NormalizedPrice(
                exchange="kraken",
                symbol=symbol,
                price=last_price,
                timestamp=datetime.now().timestamp(),
                volume_24h=volume,
            )
        except Exception as e:
            print(f"Kraken normalization error: {e}")
            return None
    
    @staticmethod
    def normalize_generic_orderbook(raw_data: Dict, exchange: str, symbol: str) -> Optional[NormalizedOrderBook]:
        """Generic order book normalizer (CCXT format)"""
        
        if not raw_data:
            return None
        
        try:
            bids = [(float(b[0]), float(b[1])) for b in raw_data.get("bids", [])]
            asks = [(float(a[0]), float(a[1])) for a in raw_data.get("asks", [])]
            
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread = best_ask - best_bid if best_bid and best_ask else 0
            
            return NormalizedOrderBook(
                exchange=exchange,
                symbol=symbol,
                timestamp=datetime.now().timestamp(),
                bids=bids[:10],
                asks=asks[:10],
                mid_price=mid_price,
                spread=spread,
            )
        except Exception as e:
            print(f"Generic normalization error: {e}")
            return None


class DataCache:
    """Simple in-memory cache for latest normalized data"""
    
    def __init__(self):
        self.data = {}
    
    def set(self, key: str, value: NormalizedOrderBook | NormalizedPrice):
        """Store normalized data"""
        self.data[key] = value
    
    def get(self, key: str) -> Optional[NormalizedOrderBook | NormalizedPrice]:
        """Retrieve latest data"""
        return self.data.get(key)
    
    def get_all(self) -> Dict:
        """Get all cached data"""
        return {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in self.data.items()}
    
    def clear(self):
        """Clear cache"""
        self.data.clear()


# Test the module
def test_data_normalizer():
    """Test normalizer with sample data"""
    
    normalizer = DataNormalizer()
    cache = DataCache()
    
    # Test Polymarket normalization
    sample_poly_data = {
        "market_id": "POLY-BTC",
        "bids": [
            {"price": "0.45", "quantity": "100"},
            {"price": "0.44", "quantity": "200"},
        ],
        "asks": [
            {"price": "0.46", "quantity": "150"},
            {"price": "0.47", "quantity": "300"},
        ]
    }
    
    poly_orderbook = normalizer.normalize_polymarket_orderbook(sample_poly_data)
    if poly_orderbook:
        print(f"✓ Polymarket normalized: {poly_orderbook.symbol}")
        print(f"  Mid price: {poly_orderbook.mid_price}, Spread: {poly_orderbook.spread}")
        cache.set("POLY-BTC", poly_orderbook)
    
    # Test Kraken normalization
    sample_kraken_data = {
        "XXBTZUSD": {
            "c": ["42500.50", "1000"],
            "v": ["1234567.89", "9876543.21"]
        }
    }
    
    kraken_price = normalizer.normalize_kraken_ticker(sample_kraken_data, "XXBTZUSD")
    if kraken_price:
        print(f"✓ Kraken normalized: {kraken_price.symbol} @ ${kraken_price.price}")
        cache.set("KRAKEN-BTC", kraken_price)
    
    # Display cached data
    print(f"\nCache contents: {len(cache.get_all())} items")
    for key, value in cache.get_all().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_data_normalizer()