"""
Enhanced Trading Engine
========================

Integrates all new trading strategies into a unified engine:
1. Original arbitrage strategy
2. 15-minute directional trading
3. Micro-arbitrage mode
4. Trade frequency targeting

Provides a single interface for running all strategies concurrently.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# Import new modules
from directional_trading import (
    DirectionalTradingEngine,
    DirectionalSignal,
    TradingSignal,
    Candle,
    Timeframe
)
from micro_arbitrage import (
    MicroArbitrageEngine,
    MicroArbMode,
    MicroArbOpportunity,
    MicroArbResult
)
from trade_frequency_manager import (
    TradeFrequencyManager,
    FrequencyMode,
    FrequencyTarget,
    TradePriority,
    DynamicThresholds
)

# Import inference engine for structural arbitrage
try:
    from inference.engine import InferenceEngine, InferenceEngineConfig
    from inference.models import PolymarketMarket, ArbSignal
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False

logger = logging.getLogger("PolyMangoBot.enhanced_engine")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class TradingStrategy(Enum):
    """Available trading strategies"""
    ARBITRAGE = "arbitrage"             # Original cross-venue arbitrage
    DIRECTIONAL = "directional"         # 15-minute directional
    MICRO_ARB = "micro_arb"             # Low-threshold micro-arbitrage
    STRUCTURAL_ARB = "structural_arb"   # Cross-market structural arbitrage
    ALL = "all"                         # Run all strategies


class EngineMode(Enum):
    """Engine operating modes"""
    CONSERVATIVE = "conservative"       # Low risk, low frequency
    BALANCED = "balanced"               # Balanced risk/reward
    AGGRESSIVE = "aggressive"           # Higher risk, high frequency
    MAX_FREQUENCY = "max_frequency"     # Maximum trade frequency (500+/week)


@dataclass
class UnifiedOpportunity:
    """Unified opportunity from any strategy"""
    id: str
    strategy: TradingStrategy
    symbol: str

    # Direction
    direction: str                      # "long", "short", or "arbitrage"
    buy_venue: Optional[str] = None
    sell_venue: Optional[str] = None

    # Prices
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_loss: float = 0.0

    # Expected metrics
    expected_profit_pct: float = 0.0
    expected_profit_usd: float = 0.0
    confidence: float = 0.0

    # Position
    suggested_quantity: float = 0.0
    suggested_position_usd: float = 0.0
    max_position_pct: float = 0.0

    # Priority
    priority: TradePriority = TradePriority.MEDIUM
    score: float = 0.0

    # Source data
    source_data: Optional[Any] = None

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "expected_profit_pct": self.expected_profit_pct,
            "confidence": self.confidence,
            "priority": self.priority.value,
            "score": self.score,
            "timestamp": self.timestamp
        }


@dataclass
class TradeExecution:
    """Record of an executed trade"""
    opportunity: UnifiedOpportunity
    success: bool
    actual_profit_usd: float = 0.0
    actual_profit_pct: float = 0.0
    execution_time_ms: float = 0.0
    slippage_pct: float = 0.0
    fees_usd: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class EngineConfig:
    """Configuration for the enhanced engine"""
    # Strategy enables
    enable_arbitrage: bool = True
    enable_directional: bool = True
    enable_micro_arb: bool = True
    enable_structural_arb: bool = True

    # Capital allocation (must sum to 1.0)
    arbitrage_capital_pct: float = 0.4
    directional_capital_pct: float = 0.25
    micro_arb_capital_pct: float = 0.15
    structural_arb_capital_pct: float = 0.2

    # Frequency settings
    frequency_mode: FrequencyMode = FrequencyMode.ACTIVE
    custom_trades_per_day: Optional[int] = None

    # Risk settings
    max_concurrent_positions: int = 10
    max_daily_loss_pct: float = 5.0

    # Scanning intervals (seconds)
    arbitrage_scan_interval: float = 5.0
    directional_scan_interval: float = 60.0  # 1 minute for 15-min candles
    micro_arb_scan_interval: float = 1.0
    structural_arb_scan_interval: float = 45.0  # Structural arb polling

    @classmethod
    def conservative(cls) -> "EngineConfig":
        return cls(
            enable_micro_arb=False,
            enable_structural_arb=True,
            arbitrage_capital_pct=0.5,
            directional_capital_pct=0.25,
            micro_arb_capital_pct=0.0,
            structural_arb_capital_pct=0.25,
            frequency_mode=FrequencyMode.CONSERVATIVE,
            max_concurrent_positions=5
        )

    @classmethod
    def balanced(cls) -> "EngineConfig":
        return cls(
            frequency_mode=FrequencyMode.MODERATE
        )

    @classmethod
    def aggressive(cls) -> "EngineConfig":
        return cls(
            enable_structural_arb=True,
            frequency_mode=FrequencyMode.ACTIVE,
            max_concurrent_positions=15,
            structural_arb_scan_interval=30.0
        )

    @classmethod
    def max_frequency(cls) -> "EngineConfig":
        """Target 500+ trades per week"""
        return cls(
            enable_structural_arb=True,
            frequency_mode=FrequencyMode.HIGH_FREQUENCY,
            arbitrage_capital_pct=0.25,
            directional_capital_pct=0.25,
            micro_arb_capital_pct=0.3,
            structural_arb_capital_pct=0.2,
            micro_arb_scan_interval=0.5,
            structural_arb_scan_interval=20.0,
            max_concurrent_positions=20
        )


# =============================================================================
# ENHANCED TRADING ENGINE
# =============================================================================

class EnhancedTradingEngine:
    """
    Unified trading engine combining all strategies.

    Features:
    - Multi-strategy coordination
    - Unified opportunity management
    - Dynamic capital allocation
    - Frequency-aware trading
    - Consolidated performance tracking
    """

    def __init__(
        self,
        capital: float = 10000.0,
        mode: EngineMode = EngineMode.BALANCED,
        config: Optional[EngineConfig] = None
    ):
        self.total_capital = capital
        self.mode = mode

        # Set configuration based on mode or use custom
        if config:
            self.config = config
        elif mode == EngineMode.CONSERVATIVE:
            self.config = EngineConfig.conservative()
        elif mode == EngineMode.AGGRESSIVE:
            self.config = EngineConfig.aggressive()
        elif mode == EngineMode.MAX_FREQUENCY:
            self.config = EngineConfig.max_frequency()
        else:
            self.config = EngineConfig.balanced()

        # Initialize sub-engines
        self._init_sub_engines()

        # Frequency manager
        self.frequency_manager = TradeFrequencyManager(
            mode=self.config.frequency_mode
        )

        # Register threshold change callback
        self.frequency_manager.on_threshold_change(self._on_threshold_change)

        # State
        self._active_positions: Dict[str, UnifiedOpportunity] = {}
        self._pending_opportunities: List[UnifiedOpportunity] = []

        # Execution history
        self._execution_history: deque = deque(maxlen=5000)

        # Performance tracking
        self._performance = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_profit_usd": 0.0,
            "total_fees_usd": 0.0,
            "by_strategy": {
                TradingStrategy.ARBITRAGE.value: {"trades": 0, "profit": 0.0},
                TradingStrategy.DIRECTIONAL.value: {"trades": 0, "profit": 0.0},
                TradingStrategy.MICRO_ARB.value: {"trades": 0, "profit": 0.0},
                TradingStrategy.STRUCTURAL_ARB.value: {"trades": 0, "profit": 0.0}
            }
        }

        # Running state
        self._is_running = False
        self._scan_tasks: List[asyncio.Task] = []

        logger.info(
            f"Enhanced Trading Engine initialized: mode={mode.value}, "
            f"capital=${capital:,.2f}, frequency={self.config.frequency_mode.value}"
        )

    def _init_sub_engines(self):
        """Initialize sub-engines with allocated capital"""
        # Directional trading engine
        if self.config.enable_directional:
            directional_capital = self.total_capital * self.config.directional_capital_pct
            self.directional_engine = DirectionalTradingEngine(
                min_confidence=0.6
            )
            logger.info(f"Directional engine: ${directional_capital:,.2f} allocated")
        else:
            self.directional_engine = None

        # Micro-arbitrage engine
        if self.config.enable_micro_arb:
            micro_arb_capital = self.total_capital * self.config.micro_arb_capital_pct

            # Set micro-arb mode based on engine mode
            if self.mode == EngineMode.CONSERVATIVE:
                micro_arb_mode = MicroArbMode.CONSERVATIVE
            elif self.mode in [EngineMode.AGGRESSIVE, EngineMode.MAX_FREQUENCY]:
                micro_arb_mode = MicroArbMode.AGGRESSIVE
            else:
                micro_arb_mode = MicroArbMode.STANDARD

            self.micro_arb_engine = MicroArbitrageEngine(
                mode=micro_arb_mode,
                capital=micro_arb_capital
            )
            logger.info(f"Micro-arb engine: ${micro_arb_capital:,.2f} allocated, mode={micro_arb_mode.value}")
        else:
            self.micro_arb_engine = None

        # Structural arbitrage (inference) engine
        self.inference_engine = None
        if self.config.enable_structural_arb and INFERENCE_AVAILABLE:
            structural_capital = self.total_capital * self.config.structural_arb_capital_pct

            # Configure based on mode
            if self.mode == EngineMode.CONSERVATIVE:
                inference_config = InferenceEngineConfig.conservative()
            elif self.mode in [EngineMode.AGGRESSIVE, EngineMode.MAX_FREQUENCY]:
                inference_config = InferenceEngineConfig.aggressive()
            else:
                inference_config = InferenceEngineConfig()

            inference_config.default_position_usd = structural_capital * 0.1

            self.inference_engine = InferenceEngine(inference_config)
            logger.info(f"Structural arb engine: ${structural_capital:,.2f} allocated")
        elif self.config.enable_structural_arb and not INFERENCE_AVAILABLE:
            logger.warning("Structural arb enabled but inference module not available")

    def _on_threshold_change(self, thresholds: DynamicThresholds):
        """Handle threshold changes from frequency manager"""
        logger.info(
            f"Thresholds updated: min_spread={thresholds.min_spread_pct:.3f}%, "
            f"min_profit={thresholds.min_profit_pct:.3f}%"
        )

        # Update micro-arb engine thresholds
        if self.micro_arb_engine:
            self.micro_arb_engine.config.min_spread_pct = thresholds.min_spread_pct
            self.micro_arb_engine.config.min_profit_after_fees_pct = thresholds.min_profit_pct

    # =========================================================================
    # PRICE UPDATES
    # =========================================================================

    def update_tick(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
        venue: Optional[str] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        bid_size: float = 0.0,
        ask_size: float = 0.0
    ):
        """Update with new price tick"""
        timestamp = time.time()

        # Update directional engine
        if self.directional_engine:
            self.directional_engine.update_price(symbol, price, volume, timestamp)

        # Update micro-arb engine (needs venue and bid/ask)
        if self.micro_arb_engine and venue and bid and ask:
            self.micro_arb_engine.update_prices(
                symbol, venue, bid, ask, bid_size, ask_size
            )

    def update_candle(
        self,
        symbol: str,
        candle_data: Dict,
        timeframe: Timeframe = Timeframe.M15
    ):
        """Update with new candle data"""
        if self.directional_engine:
            self.directional_engine.candle_manager.add_tick(
                symbol,
                candle_data["close"],
                candle_data.get("volume", 0)
            )

    def load_historical_candles(
        self,
        symbol: str,
        candles: List[Dict],
        timeframe: Timeframe = Timeframe.M15
    ):
        """Load historical candle data for directional trading"""
        if self.directional_engine:
            self.directional_engine.load_historical_candles(symbol, candles, timeframe)

    # =========================================================================
    # OPPORTUNITY SCANNING
    # =========================================================================

    def scan_all_opportunities(
        self,
        symbols: List[str],
        arbitrage_prices: Optional[Dict] = None,
        polymarket_markets: Optional[List[Dict]] = None
    ) -> List[UnifiedOpportunity]:
        """Scan all strategies for opportunities"""
        opportunities = []

        # Check if we can trade
        can_trade, reason = self.frequency_manager.can_trade()
        if not can_trade:
            return []

        # Scan directional opportunities
        if self.config.enable_directional and self.directional_engine:
            directional_opps = self._scan_directional(symbols)
            opportunities.extend(directional_opps)

        # Scan micro-arb opportunities
        if self.config.enable_micro_arb and self.micro_arb_engine:
            micro_arb_opps = self._scan_micro_arb(symbols)
            opportunities.extend(micro_arb_opps)

        # Convert arbitrage opportunities if provided
        if self.config.enable_arbitrage and arbitrage_prices:
            arb_opps = self._scan_arbitrage(arbitrage_prices)
            opportunities.extend(arb_opps)

        # Scan structural arbitrage opportunities
        if self.config.enable_structural_arb and self.inference_engine and polymarket_markets:
            structural_opps = self._scan_structural_arb(polymarket_markets)
            opportunities.extend(structural_opps)

        # Prioritize all opportunities
        if opportunities:
            opportunities = self._prioritize_opportunities(opportunities)

        return opportunities

    def _scan_directional(self, symbols: List[str]) -> List[UnifiedOpportunity]:
        """Scan for directional trading opportunities"""
        opportunities = []

        if not self.directional_engine:
            return opportunities

        signals = self.directional_engine.generate_signals(symbols)

        for signal in signals:
            opp = UnifiedOpportunity(
                id=f"dir_{signal.symbol}_{int(time.time() * 1000)}",
                strategy=TradingStrategy.DIRECTIONAL,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                target_price=signal.target_price,
                stop_loss=signal.stop_loss,
                expected_profit_pct=signal.expected_return_pct,
                expected_profit_usd=signal.expected_return_pct / 100 * self.total_capital * self.config.directional_capital_pct * signal.suggested_position_pct,
                confidence=signal.confidence,
                suggested_position_usd=self.total_capital * self.config.directional_capital_pct * signal.suggested_position_pct,
                max_position_pct=signal.max_position_pct,
                source_data=signal
            )
            opportunities.append(opp)

        return opportunities

    def _scan_micro_arb(self, symbols: List[str]) -> List[UnifiedOpportunity]:
        """Scan for micro-arbitrage opportunities"""
        opportunities = []

        if not self.micro_arb_engine:
            return opportunities

        micro_arb_opps = self.micro_arb_engine.scan_opportunities(symbols)

        for micro_opp in micro_arb_opps:
            opp = UnifiedOpportunity(
                id=micro_opp.id,
                strategy=TradingStrategy.MICRO_ARB,
                symbol=micro_opp.market,
                direction="arbitrage",
                buy_venue=micro_opp.buy_venue,
                sell_venue=micro_opp.sell_venue,
                entry_price=micro_opp.buy_price,
                target_price=micro_opp.sell_price,
                expected_profit_pct=micro_opp.estimated_profit_pct,
                expected_profit_usd=micro_opp.estimated_profit_usd,
                confidence=micro_opp.confidence,
                suggested_quantity=micro_opp.suggested_quantity,
                suggested_position_usd=micro_opp.suggested_position_usd,
                score=micro_opp.score,
                source_data=micro_opp
            )
            opportunities.append(opp)

        return opportunities

    def _scan_arbitrage(self, prices: Dict) -> List[UnifiedOpportunity]:
        """Scan for traditional arbitrage opportunities"""
        opportunities = []
        thresholds = self.frequency_manager.get_current_thresholds()

        for market, venue_prices in prices.items():
            venues = list(venue_prices.keys())

            for i, buy_venue in enumerate(venues):
                for sell_venue in venues[i + 1:]:
                    buy_price = venue_prices[buy_venue]
                    sell_price = venue_prices[sell_venue]

                    # Check both directions
                    for bp, sp, bv, sv in [
                        (buy_price, sell_price, buy_venue, sell_venue),
                        (sell_price, buy_price, sell_venue, buy_venue)
                    ]:
                        spread_pct = (sp - bp) / bp * 100

                        if spread_pct >= thresholds.min_spread_pct:
                            position_usd = self.total_capital * self.config.arbitrage_capital_pct * 0.1
                            expected_profit = position_usd * (spread_pct - 0.3) / 100  # Estimate fees

                            opp = UnifiedOpportunity(
                                id=f"arb_{market}_{bv}_{sv}_{int(time.time() * 1000)}",
                                strategy=TradingStrategy.ARBITRAGE,
                                symbol=market,
                                direction="arbitrage",
                                buy_venue=bv,
                                sell_venue=sv,
                                entry_price=bp,
                                target_price=sp,
                                expected_profit_pct=spread_pct,
                                expected_profit_usd=expected_profit,
                                confidence=min(0.9, spread_pct / 2),
                                suggested_position_usd=position_usd
                            )
                            opportunities.append(opp)

        return opportunities

    def _scan_structural_arb(
        self,
        polymarket_markets: List[Dict]
    ) -> List[UnifiedOpportunity]:
        """Scan for structural arbitrage opportunities using inference engine"""
        opportunities = []

        if not self.inference_engine or not INFERENCE_AVAILABLE:
            return opportunities

        # Convert dict markets to PolymarketMarket objects
        markets = []
        for m in polymarket_markets:
            market = PolymarketMarket.from_api(m)
            markets.append(market)

        if not markets:
            return opportunities

        # Run inference engine
        try:
            signals = self.inference_engine.process_markets(markets)

            for signal in signals:
                position_usd = self.total_capital * self.config.structural_arb_capital_pct * 0.1

                opp = UnifiedOpportunity(
                    id=f"struct_{signal.family_id}_{signal.subtype}_{int(time.time() * 1000)}",
                    strategy=TradingStrategy.STRUCTURAL_ARB,
                    symbol=signal.family_id,
                    direction="structural_arb",
                    expected_profit_pct=signal.realizable_edge,
                    expected_profit_usd=signal.worst_case_pnl,
                    confidence=signal.confidence / 10.0,  # Convert 1-10 to 0-1
                    suggested_position_usd=position_usd,
                    score=signal.realizable_edge * signal.confidence,
                    source_data=signal
                )
                opportunities.append(opp)

        except Exception as e:
            logger.error(f"Structural arb scan error: {e}")

        return opportunities

    def _prioritize_opportunities(
        self,
        opportunities: List[UnifiedOpportunity]
    ) -> List[UnifiedOpportunity]:
        """Prioritize opportunities using frequency manager"""
        # Convert to dicts for prioritizer
        opp_dicts = []
        for opp in opportunities:
            opp_dicts.append({
                "id": opp.id,
                "estimated_profit_pct": opp.expected_profit_pct,
                "expected_return_pct": opp.expected_profit_pct,
                "confidence": opp.confidence,
                "liquidity": opp.suggested_position_usd * 10,  # Estimate
                "price_age_ms": (time.time() - opp.timestamp) * 1000,
                "original": opp
            })

        prioritized = self.frequency_manager.prioritize_opportunities(opp_dicts)

        # Update opportunities with priority and score
        result = []
        for opp_dict, priority, score in prioritized:
            original = opp_dict["original"]
            original.priority = priority
            original.score = score
            result.append(original)

        return result

    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================

    def get_next_trade(self) -> Optional[UnifiedOpportunity]:
        """Get the next opportunity to execute"""
        if not self._pending_opportunities:
            return None

        # Check position limits
        if len(self._active_positions) >= self.config.max_concurrent_positions:
            return None

        # Get highest priority opportunity
        self._pending_opportunities.sort(key=lambda x: x.score, reverse=True)

        return self._pending_opportunities.pop(0)

    def queue_opportunity(self, opportunity: UnifiedOpportunity) -> bool:
        """Queue an opportunity for execution"""
        if opportunity.id in self._active_positions:
            return False

        self._pending_opportunities.append(opportunity)
        return True

    def record_execution(
        self,
        opportunity: UnifiedOpportunity,
        success: bool,
        actual_profit_usd: float = 0.0,
        actual_profit_pct: float = 0.0,
        execution_time_ms: float = 0.0,
        fees_usd: float = 0.0,
        slippage_pct: float = 0.0,
        error: Optional[str] = None
    ):
        """Record trade execution result"""
        execution = TradeExecution(
            opportunity=opportunity,
            success=success,
            actual_profit_usd=actual_profit_usd,
            actual_profit_pct=actual_profit_pct,
            execution_time_ms=execution_time_ms,
            fees_usd=fees_usd,
            slippage_pct=slippage_pct,
            error=error
        )

        self._execution_history.append(execution)

        # Update performance
        self._performance["total_trades"] += 1
        if success:
            self._performance["successful_trades"] += 1
            self._performance["total_profit_usd"] += actual_profit_usd
        self._performance["total_fees_usd"] += fees_usd

        # Update strategy-specific stats
        strategy = opportunity.strategy.value
        self._performance["by_strategy"][strategy]["trades"] += 1
        self._performance["by_strategy"][strategy]["profit"] += actual_profit_usd

        # Record in frequency manager
        self.frequency_manager.record_trade(
            source=strategy,
            opportunity_id=opportunity.id,
            profit_pct=actual_profit_pct
        )

        # Record in micro-arb engine if applicable
        if opportunity.strategy == TradingStrategy.MICRO_ARB and self.micro_arb_engine:
            result = MicroArbResult(
                opportunity=opportunity.source_data,
                success=success,
                actual_profit_usd=actual_profit_usd,
                actual_profit_pct=actual_profit_pct,
                execution_time_ms=execution_time_ms,
                total_fees_usd=fees_usd,
                total_slippage_pct=slippage_pct,
                error=error
            )
            self.micro_arb_engine.record_result(result)

        # Remove from active positions
        if opportunity.id in self._active_positions:
            del self._active_positions[opportunity.id]

        logger.info(
            f"Trade executed: {opportunity.strategy.value} {opportunity.symbol} - "
            f"{'SUCCESS' if success else 'FAILED'} "
            f"profit=${actual_profit_usd:.2f} ({actual_profit_pct:.2f}%)"
        )

    # =========================================================================
    # CONTINUOUS SCANNING
    # =========================================================================

    async def start(
        self,
        symbols: List[str],
        polymarket_fetcher: Optional[Callable[[], List[Dict]]] = None
    ):
        """Start continuous scanning"""
        if self._is_running:
            return

        self._is_running = True
        logger.info("Enhanced Trading Engine starting...")

        # Start scan tasks
        if self.config.enable_directional:
            task = asyncio.create_task(
                self._directional_scan_loop(symbols)
            )
            self._scan_tasks.append(task)

        if self.config.enable_micro_arb:
            task = asyncio.create_task(
                self._micro_arb_scan_loop(symbols)
            )
            self._scan_tasks.append(task)

        if self.config.enable_structural_arb and self.inference_engine and polymarket_fetcher:
            task = asyncio.create_task(
                self._structural_arb_scan_loop(polymarket_fetcher)
            )
            self._scan_tasks.append(task)

        logger.info(f"Started {len(self._scan_tasks)} scan tasks")

    async def stop(self):
        """Stop continuous scanning"""
        self._is_running = False

        for task in self._scan_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._scan_tasks.clear()
        logger.info("Enhanced Trading Engine stopped")

    async def _directional_scan_loop(self, symbols: List[str]):
        """Continuous directional scanning loop"""
        while self._is_running:
            try:
                opps = self._scan_directional(symbols)
                for opp in opps:
                    self.queue_opportunity(opp)

                await asyncio.sleep(self.config.directional_scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Directional scan error: {e}")
                await asyncio.sleep(5)

    async def _micro_arb_scan_loop(self, symbols: List[str]):
        """Continuous micro-arb scanning loop"""
        while self._is_running:
            try:
                opps = self._scan_micro_arb(symbols)
                for opp in opps:
                    self.queue_opportunity(opp)

                await asyncio.sleep(self.config.micro_arb_scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Micro-arb scan error: {e}")
                await asyncio.sleep(5)

    async def _structural_arb_scan_loop(
        self,
        market_fetcher: Callable[[], List[Dict]]
    ):
        """Continuous structural arbitrage scanning loop"""
        while self._is_running:
            try:
                # Fetch Polymarket markets
                if asyncio.iscoroutinefunction(market_fetcher):
                    markets = await market_fetcher()
                else:
                    markets = market_fetcher()

                if markets:
                    opps = self._scan_structural_arb(markets)
                    for opp in opps:
                        self.queue_opportunity(opp)

                await asyncio.sleep(self.config.structural_arb_scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Structural arb scan error: {e}")
                await asyncio.sleep(10)

    # =========================================================================
    # STATISTICS AND REPORTING
    # =========================================================================

    def get_performance(self) -> Dict:
        """Get comprehensive performance statistics"""
        total = self._performance["total_trades"]
        successful = self._performance["successful_trades"]

        freq_stats = self.frequency_manager.get_stats()
        weekly_projection = self.frequency_manager.get_weekly_projection()

        return {
            "mode": self.mode.value,
            "capital": self.total_capital,
            "total_trades": total,
            "successful_trades": successful,
            "success_rate": successful / total * 100 if total > 0 else 0,
            "total_profit_usd": self._performance["total_profit_usd"],
            "total_fees_usd": self._performance["total_fees_usd"],
            "net_profit_usd": self._performance["total_profit_usd"] - self._performance["total_fees_usd"],
            "roi_pct": (self._performance["total_profit_usd"] / self.total_capital) * 100,
            "by_strategy": self._performance["by_strategy"],
            "frequency": {
                "target_per_day": freq_stats["target_per_day"],
                "trades_today": freq_stats["trades_today"],
                "progress_pct": freq_stats["progress_pct"],
                "projected_weekly": weekly_projection["projected_weekly"],
                "on_track_for_500": weekly_projection["projected_weekly"] >= 500
            },
            "thresholds": freq_stats["thresholds"],
            "active_positions": len(self._active_positions),
            "pending_opportunities": len(self._pending_opportunities)
        }

    def get_strategy_breakdown(self) -> Dict:
        """Get performance breakdown by strategy"""
        breakdown = {}

        for strategy, stats in self._performance["by_strategy"].items():
            trades = stats["trades"]
            profit = stats["profit"]

            breakdown[strategy] = {
                "trades": trades,
                "profit_usd": profit,
                "avg_profit_per_trade": profit / trades if trades > 0 else 0,
                "pct_of_total_trades": trades / self._performance["total_trades"] * 100 if self._performance["total_trades"] > 0 else 0,
                "pct_of_total_profit": profit / self._performance["total_profit_usd"] * 100 if self._performance["total_profit_usd"] > 0 else 0
            }

        return breakdown

    def set_mode(self, mode: EngineMode):
        """Change engine operating mode"""
        self.mode = mode

        if mode == EngineMode.CONSERVATIVE:
            self.config = EngineConfig.conservative()
        elif mode == EngineMode.AGGRESSIVE:
            self.config = EngineConfig.aggressive()
        elif mode == EngineMode.MAX_FREQUENCY:
            self.config = EngineConfig.max_frequency()
        else:
            self.config = EngineConfig.balanced()

        # Update frequency manager
        self.frequency_manager.set_mode(self.config.frequency_mode)

        # Reinitialize sub-engines
        self._init_sub_engines()

        logger.info(f"Engine mode changed to: {mode.value}")

    def set_frequency_target(self, trades_per_day: int):
        """Set custom trades per day target"""
        self.frequency_manager.set_custom_target(trades_per_day)

        if trades_per_day >= 72:  # 500+ per week
            logger.info(f"High-frequency target set: {trades_per_day}/day = {trades_per_day * 7}/week")


# =============================================================================
# TEST FUNCTION
# =============================================================================

async def test_enhanced_engine():
    """Test the enhanced trading engine"""
    print("=" * 70)
    print("ENHANCED TRADING ENGINE TEST")
    print("=" * 70)

    # Test different modes
    for mode in [EngineMode.CONSERVATIVE, EngineMode.BALANCED, EngineMode.MAX_FREQUENCY]:
        print(f"\n--- Testing {mode.value.upper()} mode ---")

        engine = EnhancedTradingEngine(
            capital=10000.0,
            mode=mode
        )

        # Load some test data
        import numpy as np
        np.random.seed(42)

        # Generate synthetic candles
        candles = []
        base_price = 100.0
        timestamp = time.time() - (200 * 15 * 60)

        for i in range(200):
            change = np.random.randn() * 0.005
            open_price = base_price
            close_price = base_price * (1 + change)
            high_price = max(open_price, close_price) * 1.002
            low_price = min(open_price, close_price) * 0.998
            volume = np.random.uniform(1000, 5000)

            candles.append({
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })

            base_price = close_price
            timestamp += 15 * 60

        engine.load_historical_candles("BTC", candles)

        # Update micro-arb prices
        if engine.micro_arb_engine:
            engine.update_tick("BTC/USD", 42000, 1.0, "polymarket", 41990, 42010, 5.0, 5.0)
            engine.update_tick("BTC/USD", 42050, 2.0, "kraken", 42040, 42060, 10.0, 10.0)

        # Scan for opportunities
        test_prices = {
            "BTC": {"polymarket": 42000, "kraken": 42150},
            "ETH": {"polymarket": 2300, "kraken": 2330}
        }

        opportunities = engine.scan_all_opportunities(
            symbols=["BTC", "ETH", "BTC/USD"],
            arbitrage_prices=test_prices
        )

        print(f"  Config: arb={engine.config.enable_arbitrage}, "
              f"dir={engine.config.enable_directional}, "
              f"micro={engine.config.enable_micro_arb}")
        print(f"  Frequency target: {engine.frequency_manager.target.trades_per_day}/day "
              f"({engine.frequency_manager.target.trades_per_day * 7}/week)")
        print(f"  Opportunities found: {len(opportunities)}")

        if opportunities:
            print("\n  Top opportunities:")
            for opp in opportunities[:3]:
                print(f"    - {opp.strategy.value}: {opp.symbol}")
                print(f"      Direction: {opp.direction}")
                print(f"      Expected profit: {opp.expected_profit_pct:.3f}% (${opp.expected_profit_usd:.2f})")
                print(f"      Priority: {opp.priority.value} (score: {opp.score:.2f})")

        # Simulate some trades
        for i, opp in enumerate(opportunities[:5]):
            engine.record_execution(
                opportunity=opp,
                success=np.random.random() > 0.3,
                actual_profit_usd=opp.expected_profit_usd * np.random.uniform(0.5, 1.2),
                actual_profit_pct=opp.expected_profit_pct * np.random.uniform(0.5, 1.2),
                execution_time_ms=np.random.uniform(100, 500),
                fees_usd=opp.suggested_position_usd * 0.002
            )

        # Show performance
        perf = engine.get_performance()
        print(f"\n  Performance:")
        print(f"    Total trades: {perf['total_trades']}")
        print(f"    Success rate: {perf['success_rate']:.1f}%")
        print(f"    Net profit: ${perf['net_profit_usd']:.2f}")
        print(f"    ROI: {perf['roi_pct']:.2f}%")
        print(f"    On track for 500/week: {perf['frequency']['on_track_for_500']}")

    # Test MAX_FREQUENCY mode in detail
    print("\n" + "=" * 70)
    print("MAX FREQUENCY MODE - 500+ TRADES/WEEK TARGET")
    print("=" * 70)

    engine = EnhancedTradingEngine(
        capital=10000.0,
        mode=EngineMode.MAX_FREQUENCY
    )

    print(f"\nConfiguration:")
    print(f"  Target: {engine.frequency_manager.target.trades_per_day}/day = "
          f"{engine.frequency_manager.target.trades_per_day * 7}/week")
    print(f"  Frequency mode: {engine.config.frequency_mode.value}")
    print(f"  Capital allocation:")
    print(f"    - Arbitrage: {engine.config.arbitrage_capital_pct * 100}%")
    print(f"    - Directional: {engine.config.directional_capital_pct * 100}%")
    print(f"    - Micro-arb: {engine.config.micro_arb_capital_pct * 100}%")
    print(f"  Scan intervals:")
    print(f"    - Arbitrage: {engine.config.arbitrage_scan_interval}s")
    print(f"    - Directional: {engine.config.directional_scan_interval}s")
    print(f"    - Micro-arb: {engine.config.micro_arb_scan_interval}s")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_enhanced_engine())
