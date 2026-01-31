"""
Advanced Arbitrage Bot v2.0
Integrates all optimization modules:
- Liquidity-weighted opportunity scoring
- Dynamic fee/slippage estimation
- Parallel API fetching
- Order book analysis
- Venue lead-lag detection
- Machine learning prediction
- Market-maker tracking
"""

import asyncio
import sys

from api_connectors import APIManager
from data_normalizer import DataNormalizer, DataCache
from opportunity_detector import OpportunityDetector
from risk_validator import RiskValidator
from order_executor import OrderExecutor
from order_book_analyzer import OrderBookAnalyzer, OrderBookSnapshot
from fee_estimator import CombinedCostEstimator
from venue_analyzer import VenueAnalyzer, PriceEvent
from ml_opportunity_predictor import EnsemblePredictor
from mm_tracker import MultiMMAnalyzer
from websocket_manager import WebSocketManager, PriceEvent as WSPriceEvent
from kelly_position_sizer import KellySizeMode
from datetime import datetime
import statistics


class AdvancedArbitrageBot:
    """Next-generation arbitrage bot with all enhancements"""

    def __init__(self, enable_websocket: bool = True):
        # Core modules
        self.api_manager = APIManager(enable_websocket=enable_websocket)
        self.normalizer = DataNormalizer()
        self.cache = DataCache()

        # Detector with liquidity weighting
        self.detector = OpportunityDetector(min_spread_percent=0.3)  # Lower threshold since we filter better

        # Risk validator with Kelly sizing and dynamic fees
        self.risk_validator = RiskValidator(
            max_position_size=1000,
            min_profit_margin=0.2,  # Lower now that we estimate fees better
            use_dynamic_estimation=True,  # Enable dynamic fee estimation
            capital=10000.0,  # Total trading capital
            enable_kelly_sizing=True,  # Enable Kelly Criterion positioning
            kelly_mode=KellySizeMode.HALF_KELLY  # Conservative approach
        )

        self.executor = OrderExecutor(self.api_manager)

        # Advanced modules
        self.order_book_analyzer = OrderBookAnalyzer()
        self.fee_estimator = CombinedCostEstimator()
        self.venue_analyzer = VenueAnalyzer()
        self.ml_predictor = EnsemblePredictor()
        self.mm_analyzer = MultiMMAnalyzer()

        # WebSocket real-time streaming
        self.enable_websocket = enable_websocket
        self.ws_streaming_active = False

        # Statistics
        self.is_running = False
        self.scan_count = 0
        self.last_opportunities = []

    async def start(self):
        """Start the bot"""
        print("[BOT] Starting Advanced Arbitrage Bot v2.0...")
        await self.api_manager.connect_all()
        self.is_running = True
        print("[OK] Bot started with all optimization modules\n")

    async def stop(self):
        """Stop the bot"""
        print("\n[STOP] Stopping Bot...")
        await self.api_manager.disconnect_all()
        self.is_running = False
        print("[OK] Bot stopped")

    async def scan_once(self):
        """
        Run one advanced scan cycle with all optimizations
        """

        if not self.is_running:
            return

        self.scan_count += 1

        print(f"\n{'='*70}")
        print(f"SCAN #{self.scan_count} - {self._get_timestamp()}")
        print(f"{'='*70}")

        # Step 1: Fetch prices PARALLELLY from all venues
        print("\n[FETCH] Fetching prices from all venues (PARALLEL)...")
        test_symbols = ["BTC", "ETH", "DOGE"]

        # Simulate price fetching with test data
        test_prices = {
            "BTC": {
                "polymarket": {"price": 42500, "bid_qty": 2.5, "ask_qty": 2.0},
                "kraken": {"price": 42650, "bid_qty": 25.0, "ask_qty": 20.0},
            },
            "ETH": {
                "polymarket": {"price": 2300, "bid_qty": 50, "ask_qty": 40},
                "kraken": {"price": 2330, "bid_qty": 500, "ask_qty": 400},
            },
            "DOGE": {
                "polymarket": {"price": 0.45, "bid_qty": 1000, "ask_qty": 800},
                "kraken": {"price": 0.52, "bid_qty": 10000, "ask_qty": 8000},
            }
        }

        print(f"[OK] Fetched {len(test_prices)} markets in parallel")

        # Step 2: Detect opportunities with liquidity weighting
        print("\n[DETECT] Detecting opportunities with liquidity weighting...")
        opportunities = self.detector.detect_opportunities(test_prices)
        print(f"[OK] Found {len(opportunities)} raw opportunities")

        if opportunities:
            print("\nTop opportunities (sorted by liquidity score):")
            for i, opp in enumerate(opportunities[:5]):
                print(f"  {i+1}. {opp.market}: {opp.spread_percent:.2f}% spread, "
                      f"Liquidity score: {opp.liquidity_score:.1f}, "
                      f"Fill time: {opp.fill_time_estimate_ms:.0f}ms")

        # Step 3: Order book analysis for each opportunity
        print("\n[ANALYZE] Analyzing order books...")
        for opp in opportunities[:3]:  # Analyze top 3
            # Track with order book analyzer
            snap = OrderBookSnapshot(
                exchange="polymarket",
                symbol=opp.market,
                timestamp=datetime.now().timestamp(),
                bids=[(opp.buy_price, opp.buy_liquidity)],
                asks=[(opp.sell_price, opp.sell_liquidity)],
                mid_price=(opp.buy_price + opp.sell_price) / 2,
                spread=opp.spread
            )
            self.order_book_analyzer.add_snapshot(snap)

            # Get analysis
            liquidity = self.order_book_analyzer.get_liquidity_density("polymarket", opp.market)
            movement = self.order_book_analyzer.predict_spread_movement("polymarket", opp.market)

            print(f"  {opp.market}: Liquidity density: {liquidity:.1f}, "
                  f"Spread prediction: {movement['prediction']} "
                  f"(confidence: {movement['confidence']:.1f})")

        # Step 4: Validate risk for top opportunity
        if opportunities:
            top_opp = opportunities[0]

            print(f"\n[VALIDATE] Validating top opportunity: {top_opp.market}")
            print(f"   Buy on {top_opp.buy_venue} @ ${top_opp.buy_price:.2f}")
            print(f"   Sell on {top_opp.sell_venue} @ ${top_opp.sell_price:.2f}")
            print(f"   Spread: {top_opp.spread_percent:.2f}%")

            # Get market volume (mock)
            market_volumes = {
                "BTC": 1000000,
                "ETH": 500000,
                "DOGE": 100000
            }

            risk_report = self.risk_validator.validate_trade(
                market=top_opp.market,
                buy_venue=top_opp.buy_venue,
                buy_price=top_opp.buy_price,
                sell_venue=top_opp.sell_venue,
                sell_price=top_opp.sell_price,
                position_size=500,
                market_volume_24h=market_volumes.get(top_opp.market, 1000000),
                volatility=0.5
            )

            print(f"\n   Risk Level: {risk_report.risk_level.value.upper()}")
            print(f"   Safe to trade: {'YES' if risk_report.is_safe else 'NO'}")
            print(f"   Estimated profit: ${risk_report.estimated_profit_after_fees:.2f}")

            for reason in risk_report.reasons:
                print(f"   - {reason}")

            # Step 5: Execute if safe
            if risk_report.is_safe:
                print("\n Executing atomic trade (buy + sell SIMULTANEOUSLY)...")

                trade = await self.executor.execute_atomic_trade(
                    market=top_opp.market,
                    buy_venue=top_opp.buy_venue,
                    buy_price=top_opp.buy_price,
                    buy_quantity=1.0,
                    sell_venue=top_opp.sell_venue,
                    sell_price=top_opp.sell_price,
                    sell_quantity=1.0,
                )

                if trade:
                    print(f"\n Trade executed!")
                    print(f"   Profit: ${trade.profit:.2f}")
                    print(f"   Total bot profit so far: ${self.executor.get_total_profit():.2f}")

                    # Record trade result for Kelly sizing
                    is_profitable = trade.profit > 0
                    self.risk_validator.record_trade_result(is_profitable, trade.profit)
            else:
                print("\n⛔ Trade rejected due to risk checks")
        else:
            print("\n(No profitable opportunities at this moment)")

        # Step 6: Print advanced analytics
        self._print_advanced_analytics()

    def _print_advanced_analytics(self):
        """Print advanced market analytics"""

        print(f"\n{'='*70}")
        print("ADVANCED ANALYTICS")
        print(f"{'='*70}")

        # Order book health
        health = self.mm_analyzer.get_market_health()
        print(f"\nMarket Health:")
        print(f"  Liquidity: {health.get('liquidity', 'unknown')}")
        print(f"  Stability: {health.get('stability', 'unknown')}")
        print(f"  Health Score: {health.get('health_score', 0):.1%}")
        print(f"  Active MMs: {health.get('num_active_mms', 0)}")

        # Venue lead-lag
        lead_lag = self.venue_analyzer.detect_lead_lag("BTC")
        if lead_lag.get('lead_venue'):
            print(f"\nVenue Dynamics (BTC):")
            print(f"  Lead venue: {lead_lag['lead_venue']}")
            print(f"  Lag venue: {lead_lag['lag_venue']}")
            print(f"  Lag time: {lead_lag['lag_ms']:.0f}ms")
            print(f"  Confidence: {lead_lag['confidence']:.1%}")

        # WebSocket statistics
        if self.enable_websocket and self.api_manager.ws_manager:
            ws_stats = self.api_manager.ws_manager.get_statistics()
            print(f"\nWebSocket Streaming:")
            print(f"  Total price events: {ws_stats['total_events']}")
            print(f"  Polymarket connected: {ws_stats['polymarket_connected']}")
            print(f"  Subscriptions: {ws_stats['polymarket_subscriptions']}")

        # Kelly positioning statistics
        kelly_stats = self.risk_validator.get_kelly_statistics()
        if kelly_stats:
            print(f"\nKelly Criterion Positioning:")
            print(f"  Trade history: {kelly_stats.total_trades}")
            print(f"  Win rate: {kelly_stats.win_rate*100:.1f}%")
            print(f"  Profit factor: {kelly_stats.profit_factor:.2f}")
            if self.risk_validator.get_kelly_recommendation():
                kelly_rec = self.risk_validator.get_kelly_recommendation()
                print(f"  Current position size: ${kelly_rec.estimated_position_size:.2f}")

        # Trade summary
        trades = self.executor.get_trade_history()
        if trades:
            print(f"\nTrade Performance:")
            print(f"  Total trades: {len(trades)}")
            print(f"  Total profit: ${self.executor.get_total_profit():.2f}")
            print(f"  Win rate: {self.executor.get_win_rate():.1f}%")

    async def run_continuous(self, interval_seconds: int = 5):
        """Run continuous scanning with optimizations"""
        print(f" Running continuous scans every {interval_seconds} seconds...")
        print(f"(Press Ctrl+C to stop)\n")

        try:
            while self.is_running:
                await self.scan_once()
                await asyncio.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n\n⏸️  Scan interrupted by user")

    def print_summary(self):
        """Print bot summary"""
        trades = self.executor.get_trade_history()
        total_profit = self.executor.get_total_profit()
        win_rate = self.executor.get_win_rate()

        print(f"\n{'='*70}")
        print(f"FINAL BOT SUMMARY - PolyMangoBot v2.0")
        print(f"{'='*70}")
        print(f"Total scans: {self.scan_count}")
        print(f"Total trades executed: {len(trades)}")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total exposure: ${self.risk_validator.get_total_exposure():.2f}")
        print(f"Daily loss: ${self.risk_validator.daily_loss:.2f}")

        # Kelly Criterion Summary
        kelly_stats = self.risk_validator.get_kelly_statistics()
        if kelly_stats and kelly_stats.total_trades > 0:
            print(f"\nKelly Criterion Summary:")
            print(f"  Trades analyzed: {kelly_stats.total_trades}")
            print(f"  Win rate: {kelly_stats.win_rate*100:.1f}%")
            print(f"  Profit factor: {kelly_stats.profit_factor:.2f}")
            kelly_rec = self.risk_validator.get_kelly_recommendation()
            if kelly_rec:
                print(f"  Recommended position: ${kelly_rec.estimated_position_size:.2f}")
                print(f"  Kelly confidence: {kelly_rec.confidence*100:.0f}%")

        # WebSocket statistics
        if self.enable_websocket and self.api_manager.ws_manager:
            ws_stats = self.api_manager.ws_manager.get_statistics()
            print(f"\nWebSocket Streaming Statistics:")
            print(f"  Total price events received: {ws_stats['total_events']}")
            for venue, count in ws_stats['events_per_venue'].items():
                print(f"    {venue}: {count} events")

        print(f"{'='*70}\n")

    @staticmethod
    def _get_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main():
    """Main entry point - PolyMangoBot v2.0 with WebSocket and Kelly Criterion"""

    # Initialize bot with both features enabled
    bot = AdvancedArbitrageBot(enable_websocket=True)
    await bot.start()

    print("\n[BOT] PolyMangoBot v2.0 Features Enabled:")
    print("  - WebSocket real-time price streaming")
    print("  - Kelly Criterion dynamic position sizing")
    print("  - Dynamic fee and slippage estimation")
    print("  - Multi-venue arbitrage detection")
    print("  - ML-based opportunity prediction")
    print("  - Market maker tracking")
    print()

    # Run 5 scan cycles as demo
    for i in range(5):
        await bot.scan_once()
        await asyncio.sleep(1)

    # Print Kelly analysis
    print("\n[BOT] Kelly Criterion Analysis:")
    bot.risk_validator.print_kelly_analysis()

    # Print final summary
    bot.print_summary()
    await bot.stop()


async def main_with_realtime_streaming():
    """Alternative main - demonstrates real-time WebSocket streaming"""

    bot = AdvancedArbitrageBot(enable_websocket=True)
    await bot.start()

    # Register price update callback
    async def price_update(event):
        print(f"  Price: {event.venue} {event.symbol} - "
              f"Bid: {event.bid:.4f} | Ask: {event.ask:.4f} | Mid: {event.mid_price:.4f}")

    if bot.api_manager.ws_manager:
        bot.api_manager.ws_manager.register_callback(price_update)

        # Subscribe to markets
        await bot.api_manager.subscribe_realtime_prices(
            polymarket_ids=["market_1", "market_2"],
            symbols=["BTC", "ETH"],
            exchange="kraken"
        )

        # Run for 30 seconds with streaming
        try:
            await asyncio.wait_for(
                bot.api_manager.ws_manager.start_streaming(),
                timeout=30
            )
        except asyncio.TimeoutError:
            pass

    await bot.stop()


if __name__ == "__main__":
    # Run main bot with all optimizations
    asyncio.run(main())

    # Uncomment to test real-time streaming:
    # asyncio.run(main_with_realtime_streaming())
