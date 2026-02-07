# PolyMangoBot

Cryptocurrency arbitrage trading bot (v2.0) that detects and executes profitable opportunities across Polymarket, Kraken, and Coinbase.

## Tech Stack

- **Language**: Python 3.9+
- **Architecture**: Async-based (asyncio) with modular design
- **ML**: scikit-learn, LightGBM (ensemble prediction)
- **HTTP/WebSocket**: aiohttp, websockets (sub-100ms latency)
- **Code Quality**: ruff (linter), black (formatter), mypy (type checker), bandit (security)

## Key Files

- `main_v2.py` - Production bot (recommended)
- `main.py` - Original bot v1.0
- `config.py` - Centralized configuration (includes `InferenceConfig`)
- `constants.py` - Trading constants (min spread 0.3%, max daily loss 5%)
- `enhanced_trading_engine.py` - Unified engine with all strategies

## Core Modules

| Module | Purpose |
|--------|---------|
| `api_connectors.py` | API management for all venues |
| `opportunity_detector.py` | Identifies arbitrage opportunities |
| `order_executor.py` | Atomic order execution |
| `risk_validator.py` | Pre-execution validation |
| `kelly_position_sizer.py` | Kelly Criterion position sizing |
| `websocket_manager.py` | Real-time price streaming |
| `ml_opportunity_predictor.py` | ML spread prediction |
| `enhanced_trading_engine.py` | Unified multi-strategy engine |
| `inference/` | Cross-market structural arbitrage |

## Inference Engine (inference/)

Detects structural arbitrage in related Polymarket markets by finding constraint violations.

### Modules

| Module | Purpose |
|--------|---------|
| `models.py` | Data classes (MarketFamily, Violation, ArbSignal, TradeLeg) |
| `family_discovery.py` | Groups related markets by tokens/metadata/group_slug |
| `relationship.py` | Classifies relationships (date_variant, exclusive, exhaustive, exact) |
| `detection_rules.py` | Detects constraint violations |
| `trade_constructor.py` | Builds multi-leg trades with proper sizing |
| `realizable_edge.py` | Calculates net edge after fees/slippage |
| `engine.py` | Main orchestration pipeline |
| `arb_monitor.py` | Continuous monitoring (45s polling) |
| `llm_classifier.py` | Optional LLM fallback (Claude/GPT) |

### Violation Types

| Type | Description | Trade |
|------|-------------|-------|
| `monotonicity` | Later deadline cheaper than earlier (YES) | Buy later, sell earlier |
| `monotonicity_no` | Earlier deadline cheaper than later (NO) | Buy earlier, sell later |
| `exclusive_violation` | Sum of YES > 1.0 for exclusive outcomes | Sell all YES |
| `exhaustive_violation` | Sum of YES < 1.0 for exhaustive outcomes | Buy all YES |
| `date_variant_no_sweep` | Sum of NO < 1.0 for date variants | Buy all NO |

### Usage

```python
from inference import create_engine, create_monitor

# Create engine
engine = create_engine(mode="balanced")  # or "conservative", "aggressive"

# Process markets
signals = engine.process_markets(polymarket_markets)

for signal in signals:
    print(f"{signal.subtype}: {signal.realizable_edge:.2f}% edge")

# Continuous monitoring
monitor = create_monitor(engine, poll_interval=45.0)
monitor.on_signal(lambda s: print(f"Signal: {s.family_id}"))
await monitor.start()
```

### Key Formulas

- **Exclusive arb**: Profit = (sum of YES prices - 1.0) × shares
- **Exhaustive arb**: Profit = (1.0 - sum of YES prices) × shares
- **Equal share sizing**: shares = position_usd / sum(prices)

## Commands

```bash
# Run bot (recommended)
python main_v2.py

# Run original bot
python main.py

# Run demo (3 cycles)
python run_demo.py

# Run all tests (155 tests)
pytest tests/

# Run inference tests only
pytest tests/test_inference.py -v

# Run E2E inference test
python tests/e2e_inference_test.py

# Verify dependencies
python -c "import websockets, sklearn, numpy, aiohttp; print('OK')"
```

## Architecture

```
APIs (Parallel) -> WebSocket Stream -> Data Normalization -> Analysis Pipeline -> Risk Validation -> Atomic Execution
```

### Analysis Pipeline
Order Book -> Fee Estimation -> ML Prediction -> Venue Timing -> MM Tracking

### Inference Pipeline
Market Ingestion -> Family Discovery -> Relationship Classification -> Violation Detection -> Trade Construction -> Edge Calculation -> Signal Generation

## Trading Strategies (EnhancedTradingEngine)

| Strategy | Description | Capital % |
|----------|-------------|-----------|
| `ARBITRAGE` | Cross-venue price arbitrage | 40% |
| `DIRECTIONAL` | 15-minute directional trading | 25% |
| `MICRO_ARB` | Low-threshold micro-arbitrage | 15% |
| `STRUCTURAL_ARB` | Cross-market inference engine | 20% |

## Safety Features

- Atomic execution (both orders or neither)
- Circuit breaker (5 failures)
- Kelly position sizing (prevents over-leverage)
- Daily loss limits (5% cap)
- Risk validation on every trade
- Equal share sizing for exclusive/exhaustive trades

## Code Conventions

- Async/await patterns throughout
- Type hints required (mypy strict)
- Constants in `constants.py`, config in `config.py`
- Custom exceptions in `exceptions.py`
- Tests in `tests/` directory with pytest-asyncio
- Inference module uses dataclasses for all models

## Configuration

Environment variables for API credentials. See `config.py` for:
- `PolymarketConfig` - Polymarket API settings
- `ExchangeConfig` - Kraken/Coinbase settings
- `RiskConfig` - Position sizing and limits
- `KellyConfig` - Kelly Criterion parameters
- `InferenceConfig` - Structural arbitrage settings

### Inference Config Options

```python
InferenceConfig(
    enabled=True,
    min_edge_pct=0.5,           # Minimum raw edge to report
    min_leg_liquidity=500.0,    # Minimum $500 per leg
    min_realizable_edge_pct=0.3, # After fees/slippage
    default_position_usd=100.0,
    poll_interval_seconds=45.0,
    use_llm=False,              # Enable LLM classification
    auto_execute=False,         # Auto-execute signals
)
```

## Test Coverage

- **Total tests**: 155 (all passing)
- **Inference tests**: 23 unit + 7 E2E scenarios
- **Coverage**: Family discovery, relationship classification, violation detection, trade construction, edge calculation, full pipeline integration
