# PROJECT BLUEPRINT: ALPHA-ML-RESEARCHER
**Autonomous Quantitative Signal & Closed-Loop Control Systems Engine (Crypto Vault)**

---

## 1. Executive Overview & Mission
The **Alpha ML Researcher** is an industrial-grade trading intelligence engine fusing **Kaggle-grade Machine Learning** with **Classical & Modern Control Systems Engineering (ECE)**. It replaces heuristic indicators with deterministic state estimation, stationarity transformations, and closed-loop position sizing.

---

## 2. The 7 Constitutional Golden Rules
* **Rule 1: Pure Math & ECE Control Foundations**: Market microstructure, state estimation, and capital sizing are governed strictly by deterministic control theory and rigorous statistics. Zero arbitrary indicator magic numbers.
* **Rule 2: Reuse Verified Math Core**: Directly import and interface verified LTI representations (StateSpace, PID, LQR, stability analysis, root locus, Kalman) from the tested control_system project.
* **Rule 3: PyTorch Deep Learning Integration**: Deploy PyTorch temporal architectures (TCN / PatchTST) trained exclusively via Purged & Embargoed Cross-Validation to prevent lookahead bias.
* **Rule 4: Strict Discrete-Time (z-domain) Math**: All execution and filtering operate in discrete time. All closed-loop system poles must lie strictly inside the unit circle (|z| < 1).
* **Rule 5: Actuator Saturation & Anti-Windup Clamping**: Position size and leverage represent physical actuators. The integral term halts accumulation immediately upon hitting risk allocation caps.
* **Rule 6: Transport Delay & Latency Margins**: Explicitly model network and exchange latency (z^-d). Maintain a Phase Margin > 35 deg to prevent whipsaws and oscillatory instability.
* **Rule 7: Fractional Differentiation Stationarity Gate**: Raw prices are prohibited from entering ML models. Input data must be transformed with the minimum fractional order d* passing the ADF stationarity test (p < 0.05).

---

## 3. The 4 Production-Grade Upgrades
* **--anti-windup**: Clamps the integral term to prevent explosive leverage growth during prolonged sideways/consolidation regimes.
* **--volatility-adaptive-gains**: Dynamically schedules Kp, Ki, Kd and LQR penalty matrices (Q, R) based on Garman-Klass / ATR volatility to damp aggressive sizing during liquidation wicks.
* **--meta-labeling**: Uses the Triple-Barrier Method. Decouples direction prediction (Up/Down) from bet sizing confidence (0.0 to 1.0) to filter false breakouts.
* **--zero-leakage-cv**: Enforces strict walk-forward cross-validation with purging and embargo buffers, guaranteeing zero future data leakage in training.

---

## 4. End-to-End Architectural Dataflow
1. **Live Ingestion (ccxt)**: Extracts 15m OHLCV, 20-level orderbook, and funding rates.
2. **State Denoising (Discrete Kalman Filter)**: Strips microstructure bounce; extracts latent fair price & velocity.
3. **Feature Transform (FracDiff d ≈ 0.38)**: Preserves trend memory while ensuring ADF stationarity (p < 0.05).
4. **Alpha Inference (PyTorch TCN / Meta-Labeling)**: Outputs directional probability & bet sizing confidence score.
5. **Sizing & Risk (Closed-Loop PID / LQR Engine)**: Scales position via conviction, damps via D-term, guards phase margin.
6. **Action Output (Trading Action Plan Brief)**: Recommends exact position size, entry bracket, and dynamic de-risk rule.

---

## 5. User Action Checklist (What We Need From You)
* **[ ] SQLite Vault Placement**: Confirm where local historical market tick/candle databases should live (e.g., C:/Users/gotog/quant_data/crypto_vault.db).
* **[ ] Kaggle API Credentials (Optional GPU Compute)**: Ensure kaggle.json is present in ~/.kaggle/ if you want the agent to offload heavy multi-epoch PyTorch training jobs directly to free cloud GPUs.
* **[ ] Exchange Target Selection**: Confirm your primary target pairs (e.g., BTC/USDT, ETH/USDT on Binance or Bybit public endpoints). Zero exchange API keys or account balances are required for data fetching.
* **[ ] Risk Boundaries Configuration**: Define your account safety parameters: Maximum leverage cap (e.g., 3x), maximum single-position equity risk (e.g., 2% of portfolio), and daily drawdown cutoff.
