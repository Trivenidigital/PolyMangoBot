"""
Main Orchestrator
Brings all modules together into a working bot
"""

import asyncio
from api_connectors import APIManager
from data_normalizer import DataNormalizer, DataCache
from opportunity_detector import OpportunityDetector
from risk_validator import RiskValidator
from order_executor import OrderExecutor


class ArbitrageBot:
    """Main bot that orchestrates all modules"""
    
    def __init__(self):
        self.api_manager = APIManager()
        self.normalizer = DataNormalizer()
        self.cache = DataCache()
        self.detector = OpportunityDetector(min_spread_percent=0.5)
        self.risk_validator = RiskValidator(
            max_position_size=1000,
            min_profit_margin=0.3,
        )
        self.executor = OrderExecutor(self.api_manager)
        self.is_running = False
    
    async def start(self):
        """Start the bot"""
        print(" Starting Arbitrage Bot...")
        await self.api_manager.connect_all()
        self.is_running = True
        print(" Bot started\n")
    
    async def stop(self):
        """Stop the bot"""
        print("\n Stopping Arbitrage Bot...")
        await self.api_manager.disconnect_all()
        self.is_running = False
        print(" Bot stopped")
    
    async def scan_once(self):
        """
        Run one scan cycle:
        1. Fetch prices from all venues
        2. Normalize data
        3. Detect opportunities
        4. Validate risk
        5. Execute if safe
        """
        
        if not self.is_running:
            return
        
        print(f"\n{'='*60}")
        print(f"SCAN CYCLE - {self._get_timestamp()}")
        print(f"{'='*60}")
        
        # Step 1: Fetch prices (simulated with test data for demo)
        print("\n Fetching prices from venues...")
        
        # In real version, fetch from APIs:
        # poly_data = await self.api_manager.polymarket.get_order_book("BTC")
        # kraken_data = await self.api_manager.exchange.get_ticker("XXBTZUSD")
        
        # For demo, use test data
        test_prices = {
            "BTC": {
                "polymarket": 42500,
                "kraken": 42650,
            },
            "ETH": {
                "polymarket": 2300,
                "kraken": 2330,
            },
            "DOGE": {
                "polymarket": 0.45,
                "kraken": 0.52,
            }
        }
        
        print(f" Fetched prices for {len(test_prices)} markets")
        
        # Step 2: Detect opportunities
        print("\n Scanning for opportunities...")
        opportunities = self.detector.detect_opportunities(test_prices)
        print(f" Found {len(opportunities)} opportunities")
        
        if not opportunities:
            print("   (No profitable opportunities at this moment)")
            return
        
        # Step 3: Validate and execute top opportunity
        print("\n  Validating top opportunity...")
        top_opp = opportunities[0]
        
        print(f"   Market: {top_opp.market}")
        print(f"   Buy on {top_opp.buy_venue} @ ${top_opp.buy_price:.2f}")
        print(f"   Sell on {top_opp.sell_venue} @ ${top_opp.sell_price:.2f}")
        print(f"   Spread: {top_opp.spread_percent:.2f}%")
        
        # Validate risk
        risk_report = self.risk_validator.validate_trade(
            market=top_opp.market,
            buy_venue=top_opp.buy_venue,
            buy_price=top_opp.buy_price,
            sell_venue=top_opp.sell_venue,
            sell_price=top_opp.sell_price,
            position_size=500
        )
        
        print(f"\n Risk assessment: {risk_report.risk_level.value.upper()}")
        print(f"   Safe to trade: {'YES ' if risk_report.is_safe else 'NO '}")
        print(f"   Estimated profit: ${risk_report.estimated_profit_after_fees:.2f}")
        
        # Step 4: Execute if safe
        if risk_report.is_safe:
            print("\n Executing trade...")
            
            trade = await self.executor.execute_atomic_trade(
                market=top_opp.market,
                buy_venue=top_opp.buy_venue,
                buy_price=top_opp.buy_price,
                buy_quantity=1.0,  # Simplified quantity
                sell_venue=top_opp.sell_venue,
                sell_price=top_opp.sell_price,
                sell_quantity=1.0,
            )
            
            if trade:
                print(f"\n Trade executed successfully!")
                print(f"   Profit: ${trade.profit:.2f}")
                print(f"   Total bot profit so far: ${self.executor.get_total_profit():.2f}")
        else:
            print("\n Trade rejected due to risk checks")
            for reason in risk_report.reasons:
                print(f"    {reason}")
    
    async def run_continuous(self, interval_seconds: int = 10):
        """Run continuous scanning"""
        print(f" Running continuous scans every {interval_seconds} seconds...")
        print(f"(Press Ctrl+C to stop)\n")
        
        try:
            while self.is_running:
                await self.scan_once()
                await asyncio.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n\n  Scan interrupted by user")
    
    def print_summary(self):
        """Print bot summary"""
        trades = self.executor.get_trade_history()
        total_profit = self.executor.get_total_profit()
        win_rate = self.executor.get_win_rate()
        
        print(f"\n{'='*60}")
        print(f"BOT SUMMARY")
        print(f"{'='*60}")
        print(f"Total trades executed: {len(trades)}")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total exposure: ${self.risk_validator.get_total_exposure():.2f}")
        print(f"Daily loss: ${self.risk_validator.daily_loss:.2f}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def _get_timestamp():
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main():
    """Main entry point"""
    
    bot = ArbitrageBot()
    await bot.start()
    
    # Run 3 scan cycles as demo
    for i in range(3):
        await bot.scan_once()
        await asyncio.sleep(2)
    
    bot.print_summary()
    await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())