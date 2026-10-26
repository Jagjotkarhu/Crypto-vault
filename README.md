# 🛡️ Crypto-Vault: Closed-Loop Quantitative Alpha & Control Systems Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An industrial-grade, closed-loop quantitative trading intelligence engine fusing **Kaggle-grade Machine Learning** with **Deterministic Classical & Modern Control Systems Engineering (ECE)**.

---

## 🏛️ System Architecture

Instead of heuristic indicators or static percentage betting, Crypto-Vault models market dynamics as a physical dynamical system:

`
[Live Market Data (ccxt - 15m OHLCV + Orderbook)]
                     │
                     ▼
  [Discrete-Time Kalman Filter (core_math/kalman.py)]
  • Strips microstructure bounce & spoofing noise
  • Extracts unobserved Fair Value & True Momentum
                     │
                     ▼
 [Stationary Feature Engine (core_math/fractional_diff.py)]
  • Marcos Lopez de Prado Fractional Differentiation (d ≈ 0.38)
  • Preserves long-range trend memory while passing ADF (p < 0.05)
                     │
                     ▼
  [Deep Learning Alpha Pipeline (models/temporal_tcn.py)]
  • Primary: Directional Probability P(Up) vs P(Down)
  • Meta-Labeling: Triple-Barrier bet sizing confidence (0.0 to 1.0)
  • Validated via Purged & Embargoed Cross-Validation (Zero lookahead)
                     │
                     ▼
  [Closed-Loop PID / LQR Control Engine (risk_control/pid_sizer.py)]
  • Error e[k] = Target Portfolio Delta - Current Exposure
  • Proportional (P): Scales allocation to ML conviction
  • Integral (I): Accumulates on persistent trends (Anti-Windup Clamped)
  • Derivative (D): Active brake on volatility spikes & orderbook flips
  • Stability Guardrail: Enforces Phase Margin > 35 deg against latency z^-d
                     │
                     ▼
         [Actionable Execution Plan]
         • Exact position sizing in INR (₹) / BTC
         • Dynamic de-risking & trailing stop thresholds
`

---

## 🔒 The 7 Constitutional Golden Rules

1. **Pure Math & ECE Control Foundations**: Strictly deterministic control algorithms and statistical grounding. Zero arbitrary magic indicators.
2. **Reused Verified Math Core**: Inherits verified LTI math representations (StateSpace, PID, LQR, stability criteria) from verified control systems.
3. **PyTorch Deep Learning Integration**: Temporal architectures trained exclusively with **Purged & Embargoed Cross-Validation**.
4. **Strict Discrete-Time (z-domain) Math**: All filters and controllers operate in discrete time. Closed-loop poles must satisfy $|z_i| < 1$.
5. **Actuator Saturation & Anti-Windup Clamping**: Hard leverage cap clamped at **5x**. Integral term halts accumulation immediately when active tranche hits limit.
6. **Transport Delay & Latency Margins**: Models exchange round-trip delay (^{-d}$) and enforces Phase Margin $> 35^\circ$ to prevent destabilizing execution whipsaws.
7. **Fractional Differentiation Stationarity Gate**: Raw prices never enter ML models. Series are transformed with optimal fractional order ^*$ passing the ADF test ( < 0.05$).

---

---

## ⚡ Digital Signal Processing (DSP) & DSD Principles

To bridge ECE hardware theory with high-frequency quantitative computing, the mathematical core adopts 3 digital systems invariants:

1. **Windowed Linear-Phase FIR Topology (core_math/fractional_diff.py)**:
   - Fractional differentiation weights are truncated via an expanding/fixed-window **FIR filter** with a weight threshold (e.g.,  < 10^{-4}$).
   - **Guarantees zero phase distortion**, ensuring filtered momentum peaks and troughs remain strictly synchronized with raw real-time price discovery.

2. **Quantization Noise Modeling (core_math/kalman.py)**:
   - Models market microstructure orderbook granularity as an **ADC quantization error floor**:
     \sigma_{quant}^2 = \frac{\Delta^2}{12}
     where $\Delta$ is the minimum asset tick size.
   - Automatically tunes the Kalman measurement covariance matrix ($) dynamically per asset without manual guesswork.

3. **Deterministic Execution FSM (
isk_control/pid_sizer.py)**:
   - Capital allocation transitions through a formal **Moore/Mealy Finite State Machine**:
     IDLE $\to$ ARMED $\to$ IN_POSITION $\to$ ANTI_WINDUP_CLAMPED $\to$ EMERGENCY_DAMP.
   - Rejects illegal state jumps, double-order execution, or race conditions during rapid volatility regime shifts.

## ⚙️ Configuration & Risk Controls

* **Primary Assets**: BTC/USDT, ETH/USDT, SOL/USDT
* **Account Capital**: 5,000 INR
* **Single Trade Active Margin**: 500 INR (10% allocation)
* **Maximum Leverage**: 5x (Maximum exposure: 2,500 INR notional)
* **Cloud Storage**: Supabase PostgreSQL time-series cache (zero local laptop disk bloat)
* **Cloud GPU Compute**: Kaggle API (~/.kaggle/kaggle.json)

---

## 🚀 Quickstart

### 1. Installation
`ash
git clone https://github.com/Jagjotkarhu/Crypto-vault.git
cd Crypto-vault
pip install -r requirements.txt
`

### 2. Environment Variables
Create a .env file in the root directory:
`env
SUPABASE_URL=https://buctapkmengzlijcmeyg.supabase.co
SUPABASE_KEY=your_supabase_secret_key
DEFAULT_PAIR=BTC/USDT
MAX_LEVERAGE=5.0
TRANCHE_INR=500.0
`

### 3. Run Live Analysis
`ash
python main.py --pair BTC/USDT --timeframe 15m
`

---


---

## 🎯 Completed Milestones (Sprint Delivered)
1. ✅ **Paper-Trading Simulated Ledger (`execution/paper_ledger.py`):** Local state machine tracking simulated executions, fills, slippage, and cumulative PnL over time without real capital risk.
2. ✅ **Persistent WebSocket Data Ingestion (`data/ws_streamer.py`):** Replaced polling with continuous Binance WebSocket orderbook feeds to handle microstructure flash wicks with sub-20ms latency.
3. ✅ **Volatility-Adaptive Trailing Stop-Loss (`risk_control/adaptive_stops.py`):** Linked closed-loop PID error, spread, and dynamic Kalman covariance directly to automated trailing stop-loss triggers.

### 4. Autonomous Paper Trading Bot
Run the continuous live execution loop:
```bash
python run_paper_bot.py BTC/USDT
```

## 📜 License
Distributed under the MIT License. See LICENSE for more information.
