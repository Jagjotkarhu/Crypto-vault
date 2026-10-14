---
name: quantitative-signal-engine
description: Autonomous Quantitative Market Signal & Closed-Loop Control Systems Engine. Ingests live orderbook depth and OHLCV, runs discrete Kalman denoising and FIR FracDiff, and outputs optimal PID position sizes.
---

# Quantitative Signal Engine Skill

## Operational Workflow
1. **Live Ingestion**:
   - Stream top-20 order book depth and last 250 15m OHLCV candles via ccxt.
   - Calculate Order Book Imbalance (OBI) and bid/ask volume ratios.
2. **State Denoising & Regime Check**:
   - Run discrete Kalman filter (core_math/kalman.py) with ADC quantization noise modeling to extract true latent fair price and velocity.
   - Filter macro trend using the 200 EMA boundary condition.
3. **Stationary Memory Transformation**:
   - Apply windowed Linear-Phase FIR Fractional Differentiation (core_math/fractional_diff.py) with d = 0.38 to preserve long-term trend memory while ensuring ADF stationarity.
4. **Closed-Loop Position Sizing (PID Sizer)**:
   - Calculate multi-factor conviction score (0.35 to 0.83).
   - Evaluate discrete PID feedback terms:
     - P (Proportional): Immediate conviction allocation.
     - I (Integral): Persistent trend scaling (Anti-windup clamped at 5.0x / 2,500 INR).
     - D (Derivative): Dynamic damping on volatility acceleration.
5. **Execution Reporting**:
   - Output structured actionable trading briefs with exact entry, stop-loss, and de-risking rules.
