# PolyMangoBot - Dependencies & Setup Guide

## Prerequisites Installed

All required dependencies have been installed and verified:

```
websockets     - WebSocket protocol support
scikit-learn   - Machine learning library
numpy          - Numerical computing
aiohttp        - Async HTTP requests
asyncio        - Built-in Python async support (Python 3.7+)
```

### Installation Complete

```bash
pip install websockets scikit-learn numpy aiohttp
```

Status: [OK] All dependencies installed and verified

---

## Running the Bots

### Option 1: Original Bot (v1.0)
```bash
python main.py
```

What it does:
- Scans every ~2 seconds
- Detects spread opportunities
- Executes atomic trades
- Uses basic spread threshold

Expected output: 5 scan cycles with atomic execution demonstrations

---

### Option 2: Advanced Bot (v2.0) - RECOMMENDED
```bash
python main_v2.py
```

What it does:
- Continuous scanning
- Liquidity-weighted opportunities
- Dynamic fee estimation
- Atomic parallel execution
- ML predictions
- Market-maker tracking
- Kelly position sizing
- WebSocket ready

Expected output: 5+ scan cycles with advanced analytics

---

### Option 3: Test Suite
```bash
python test_features_v2.py
```

What it tests:
- Kelly Criterion position sizing
- Kelly modes comparison (Full, Half, Quarter)
- Risk validator with Kelly integration
- WebSocket manager structure
- Component integration

Expected result: **5/5 tests PASSING**

---

## Troubleshooting

### If you see: `ModuleNotFoundError: No module named 'websockets'`

**Solution:**
```bash
pip install websockets
```

### If you see: `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
pip install scikit-learn
```

### If you see: `ModuleNotFoundError: No module named 'numpy'`

**Solution:**
```bash
pip install numpy
```

### If you see: `UnicodeEncodeError` with emoji characters

**Solution:** This should be fixed now - we cleaned all emoji characters. If you still see it, please report it.

**Manual fix if needed:**
```bash
python << 'EOF'
import os
import re

for filename in ['main.py', 'main_v2.py', 'api_connectors.py', 'order_executor.py']:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove emoji and non-ASCII characters
        content = ''.join(c if ord(c) < 128 else '' for c in content)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned {filename}")
EOF
```

### If WebSocket connection fails in test mode

**This is expected!** The warnings about WebSocket connection failures are normal in test mode because:
- Polymarket CLOB requires real authentication
- Kraken/Coinbase APIs require real credentials
- Test mode uses mock data instead

**For production:** Replace API credentials in the bot configuration to connect to real endpoints.

### If tests show low confidence warnings

**This is expected!** At startup:
- Kelly position sizer hasn't recorded trades yet
- ML model hasn't seen real data
- Confidence gradually improves with real market data

Once you run with real APIs, these warnings will disappear as trade history accumulates.

---

## Environment Setup (Optional)

### Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install websockets scikit-learn numpy aiohttp
```

### Check Python Version
```bash
python --version
```

Required: Python 3.7 or higher (3.8+ recommended)

---

## Quick Start Verification

Run this to verify everything is working:

```bash
# 1. Check dependencies
python -c "import asyncio, websockets, aiohttp, numpy, sklearn; print('[OK] All dependencies installed')"

# 2. Run the bot
python main_v2.py

# 3. Run the tests
python test_features_v2.py
```

Expected: All three commands succeed without errors

---

## Configuration Files

No external configuration files needed! Everything is hardcoded:

### In main_v2.py:
- API credentials (mock for testing)
- Position sizes ($500 default)
- Scan frequency (2-3 seconds)
- Risk limits

### For Production:
Edit `main_v2.py` and update:
```python
# API Credentials
POLYMARKET_API_KEY = "your_key_here"
KRAKEN_API_KEY = "your_key_here"
KRAKEN_API_SECRET = "your_secret_here"

# Position sizing
POSITION_SIZE = 500  # Start small, increase after testing

# Risk parameters
MAX_POSITION = 1000
MIN_PROFIT_MARGIN = 0.3
KELLY_MODE = "HALF_KELLY"  # Recommended: 50% Kelly
```

---

## File Structure

```
C:\Projects\PolyMangoBot\

CORE BOT CODE:
  main.py                    - Original bot (v1.0)
  main_v2.py                 - Advanced bot (v2.0) - USE THIS
  order_book_analyzer.py     - Order book intelligence
  fee_estimator.py           - Dynamic fee estimation
  venue_analyzer.py          - Venue timing analysis
  ml_opportunity_predictor.py - ML prediction
  mm_tracker.py              - Market-maker tracking
  websocket_manager.py       - Real-time streaming
  kelly_position_sizer.py    - Position sizing

SUPPORT MODULES:
  api_connectors.py          - API management
  opportunity_detector.py    - Opportunity detection
  risk_validator.py          - Risk validation
  order_executor.py          - Order execution
  data_normalizer.py         - Data normalization

TESTING:
  test_features_v2.py        - Test suite

DOCUMENTATION:
  START_HERE.md                          - Quick overview
  QUICKSTART_v2.0.md                     - 30-second setup
  FEATURES_v2.0.md                       - Feature guide
  IMPLEMENTATION_SUMMARY.md              - Technical details
  IMPLEMENTATION_CHECKLIST.md            - What was implemented
  README_V2.md                           - Comprehensive guide
  DEPLOYMENT_SUMMARY.txt                 - Deployment guide
  PROJECT_COMPLETION_VERIFICATION.md     - Final verification
  DEPENDENCIES_AND_SETUP.md              - This file
```

---

## Next Steps

### Immediate (Today)
1. [x] Install dependencies
2. [x] Run main_v2.py to verify it works
3. [x] Run test_features_v2.py to verify tests pass
4. [x] Read START_HERE.md

### This Week
5. Read FEATURES_v2.0.md (feature overview)
6. Read IMPLEMENTATION_SUMMARY.md (technical details)
7. Review code in your IDE
8. Plan API integration

### Next Week
9. Get real Polymarket CLOB WebSocket URL
10. Get real Kraken/Coinbase API credentials
11. Update main_v2.py with real credentials
12. Test with small position sizes ($100-500)
13. Monitor and adjust parameters

### Ongoing
14. Collect historical data for ML training
15. Scale position sizes gradually
16. Optimize Kelly parameters
17. Track live profitability

---

## Support & Documentation

- **Quick questions?** → START_HERE.md
- **How to run?** → QUICKSTART_v2.0.md
- **Features explained?** → FEATURES_v2.0.md
- **Technical details?** → IMPLEMENTATION_SUMMARY.md
- **Troubleshooting?** → QUICK_REFERENCE.md
- **Deployment?** → DEPLOYMENT_SUMMARY.txt

---

## Final Verification

Run these commands to verify everything is ready:

```bash
# Test 1: Dependencies
python -c "import websockets, aiohttp, numpy, sklearn; print('[OK] Dependencies installed')"

# Test 2: Bot runs
timeout 5 python main_v2.py && echo "[OK] Bot runs successfully"

# Test 3: Tests pass
python test_features_v2.py | grep "ALL TESTS PASSED"
```

Expected output: All three succeed

---

## Production Checklist

Before deploying with real APIs:

- [ ] API credentials obtained
- [ ] WebSocket URL verified
- [ ] Position size decided (start with $100-500)
- [ ] Risk limits configured
- [ ] Kelly parameters reviewed
- [ ] Monitoring setup
- [ ] Backup plan in place
- [ ] Small position test passed

---

## Summary

✓ All dependencies installed
✓ Both bots (v1 and v2) working
✓ Test suite passing
✓ Documentation complete
✓ Ready for real API integration

**You're all set to begin real API integration and live trading!**

---

## Questions?

Check the documentation files in the project directory, or review the code comments in each module.

**Happy trading!**
