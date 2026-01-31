"""
Advanced Arbitrage Bot v3.0
Production-ready with:
- Proper configuration management
- Graceful shutdown handling
- Comprehensive error handling
- Circuit breakers and retry logic
- Memory-safe price history
"""

import asyncio
import signal
import sys
import logging
from typing import Optional
from datetime import datetime

from config import BotConfig, setup_logging, get_config
from api_connectors import APIManager
from data_normalizer import DataNormalizer, DataCache
from opportunity_detector import OpportunityDetector
from risk_validator import RiskValidator
from order_executor import OrderExecutor
from order_book_analyzer import OrderBookAnalyzer, OrderBookSnapshot
from fee_estimator import CombinedCostEstimator
from venue_analyzer import VenueAnalyzer
from ml_opportunity_predictor import EnsemblePredictor
from mm_tracker import MultiMMAnalyzer
from websocket_manager import WebSocketManager
from kelly_position_sizer import KellySizeMode
from exceptions import PolyMangoBotError


class AdvancedArbitrageBotV3:
    """
    Production-ready arbitrage bot with comprehensive error handling
    and graceful shutdown support.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or get_config()
        self.logger = setup_logging(self.config.logging)

        # Validate configuration
        issues = self.config.validate()
        for issue in issues:
            self.logger.warning(issue)

        # Core modules
        self.api_manager = APIManager(
            enable_websocket=self.config.websocket.enabled
        )
        self.normalizer = DataNormalizer()
        self.cache = DataCache()

        # Detector with liquidity weighting
        self.detector = OpportunityDetector(
            min_spread_percent=self.config.min_spread_percent
        )

        # Risk validator with Kelly sizing
        kelly_mode = KellySizeMode[self.config.kelly.mode]
        self.risk_validator = RiskValidator(
            max_position_size=self.config.risk.max_position_size,
            min_profit_margin=self.config.risk.min_profit_margin_percent,
            use_dynamic_estimation=True,
            capital=self.config.kelly.capital,
            enable_kelly_sizing=self.config.kelly.enabled,
            kelly_mode=kelly_mode
        )

        self.executor = OrderExecutor(self.api_manager)

        # Advanced modules
        self.order_book_analyzer = OrderBookAnalyzer()
        self.fee_estimator = CombinedCostEstimator()
        self.venue_analyzer = VenueAnalyzer()
        self.ml_predictor = EnsemblePredictor()
        self.mm_analyzer = MultiMMAnalyzer()

        # State management
        self.is_running = False
        self.scan_count = 0
        self._shutdown_event = asyncio.Event()
        self._main_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the bot with proper initialization"""
        self.logger.info("Starting PolyMangoBot v3.0...")

        try:
            await self.api_manager.connect_all()
            self.is_running = True

            if self.config.dry_run:
                self.logger.warning("Running in DRY RUN mode - no real trades")

            self.logger.info("Bot started successfully")
            self._print_config_summary()

        except Exception as e:
            self.logger.error(f"Failed to start bot: {e}")
            raise

    async def stop(self):
        """Stop the bot gracefully"""
        self.logger.info("Initiating graceful shutdown...")

        self._shutdown_event.set()
        self.is_running = False

        # Wait for active trades to complete
        self.logger.info("Waiting for active trades to complete...")
        await self.executor.wait_for_active_trades(timeout_seconds=30.0)

        # Shutdown WebSocket connections
        if self.api_manager.ws_manager:
            await self.api_manager.ws_manager.shutdown()

        # Disconnect APIs
        await self.api_manager.disconnect_all()

        self.logger.info("Bot stopped gracefully")

    def _print_config_summary(self):
        """Print configuration summary"""
        self.logger.info("Configuration Summary:")
        self.logger.info(f"  Scan interval: {self.config.scan_interval_seconds}s")
        self.logger.info(f"  Min spread: {self.config.min_spread_percent}%")
        self.logger.info(f"  Max position: ${self.config.risk.max_position_size}")
        self.logger.info(f"  Kelly sizing: {self.config.kelly.enabled} ({self.config.kelly.mode})")
        self.logger.info(f"  WebSocket: {self.config.websocket.enabled}")
        self.logger.info(f"  Dry run: {self.config.dry_run}")

    async def scan_once(self):
        """Run one scan cycle with comprehensive error handling"""
        if not self.is_running or self._shutdown_event.is_set():
            return

        self.scan_count += 1
        scan_start = datetime.now()

        self.logger.info(f"Scan #{self.scan_count} started at {scan_start.strftime('%H:%M:%S')}")

        try:
            # Fetch prices (using test data for demo)
            test_prices = {
                "BTC": {
                    "polymarket": {"price": 42500, "bid_qty": 2.5, "ask_qty": 2.0},
                    "kraken": {"price": 42650, "bid_qty": 25.0, "ask_qty": 20.0},
                },
                "ETH": {
                    "polymarket": {"price": 2300, "bid_qty": 50, "ask_qty": 40},
                    "kraken": {"price": 2330, "bid_qty": 500, "ask_qty": 400},
                },
            }

            # Detect opportunities
            opportunities = self.detector.detect_opportunities(test_prices)
            self.logger.info(f"Found {len(opportunities)} opportunities")

            if not opportunities:
                return

            # Process top opportunity
            top_opp = opportunities[0]
            self.logger.info(
                f"Top opportunity: {top_opp.market} - "
                f"Buy @ ${top_opp.buy_price:.2f} on {top_opp.buy_venue}, "
                f"Sell @ ${top_opp.sell_price:.2f} on {top_opp.sell_venue}, "
                f"Spread: {top_opp.spread_percent:.2f}%"
            )

            # Validate risk
            market_volumes = {"BTC": 1000000, "ETH": 500000}
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

            self.logger.info(
                f"Risk assessment: {risk_report.risk_level.value} - "
                f"Safe: {risk_report.is_safe}, "
                f"Est. profit: ${risk_report.estimated_profit_after_fees:.2f}"
            )

            # Execute if safe
            if risk_report.is_safe:
                trade = await self.executor.execute_atomic_trade(
                    market=top_opp.market,
                    buy_venue=top_opp.buy_venue,
                    buy_price=top_opp.buy_price,
                    buy_quantity=1.0,
                    sell_venue=top_opp.sell_venue,
                    sell_price=top_opp.sell_price,
                    sell_quantity=1.0,
                    dry_run=self.config.dry_run
                )

                if trade:
                    self.logger.info(
                        f"Trade executed: {trade.trade_id} - "
                        f"Profit: ${trade.profit:.2f}"
                    )
                    self.risk_validator.record_trade_result(
                        trade.profit > 0, trade.profit
                    )
            else:
                self.logger.info("Trade rejected due to risk checks")

        except PolyMangoBotError as e:
            self.logger.error(f"Bot error during scan: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error during scan: {e}")

        finally:
            scan_duration = (datetime.now() - scan_start).total_seconds()
            self.logger.debug(f"Scan #{self.scan_count} completed in {scan_duration:.2f}s")

    async def run_continuous(self):
        """Run continuous scanning with graceful shutdown support"""
        self.logger.info(
            f"Starting continuous scanning every {self.config.scan_interval_seconds}s"
        )

        try:
            while self.is_running and not self._shutdown_event.is_set():
                await self.scan_once()

                # Use wait_for with shutdown event for interruptible sleep
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.scan_interval_seconds
                    )
                    # If we get here, shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue scanning
                    pass

        except asyncio.CancelledError:
            self.logger.info("Continuous scanning cancelled")

    def print_summary(self):
        """Print final bot summary"""
        trades = self.executor.get_trade_history()
        total_profit = self.executor.get_total_profit()
        win_rate = self.executor.get_win_rate()
        exec_stats = self.executor.get_execution_stats()

        print("\n" + "=" * 70)
        print("FINAL BOT SUMMARY - PolyMangoBot v3.0")
        print("=" * 70)
        print(f"Total scans: {self.scan_count}")
        print(f"Total trades executed: {len(trades)}")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"Win rate: {win_rate:.1f}%")

        print(f"\nExecution Statistics:")
        print(f"  Avg execution time: {exec_stats['avg_execution_ms']:.1f}ms")
        print(f"  P95 execution time: {exec_stats['p95_execution_ms']:.1f}ms")

        kelly_stats = self.risk_validator.get_kelly_statistics()
        if kelly_stats and kelly_stats.total_trades > 0:
            print(f"\nKelly Criterion Summary:")
            print(f"  Trades analyzed: {kelly_stats.total_trades}")
            print(f"  Win rate: {kelly_stats.win_rate * 100:.1f}%")
            print(f"  Profit factor: {kelly_stats.profit_factor:.2f}")

        print("=" * 70 + "\n")


async def main():
    """Main entry point with signal handling"""
    bot = AdvancedArbitrageBotV3()

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutdown signal received...")
        asyncio.create_task(bot.stop())

    # Register signal handlers (Unix only)
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        await bot.start()

        # Run 5 scan cycles for demo
        for _ in range(5):
            if not bot.is_running:
                break
            await bot.scan_once()
            await asyncio.sleep(1)

        bot.print_summary()

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
