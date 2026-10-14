---
name: crypto-microstructure-cleaner
description: Real-time market microstructure cleaning, anomaly/flash-wick rejection, and dynamic Kalman covariance calibration for high-accuracy alpha signals.
---

# Crypto Microstructure Cleaner Skill

## Operational Responsibilities
1. **Flash Wick & Outlier Sanitization**:
   - Uses Median Absolute Deviation (MAD) to detect and bound price spikes caused by liquidity vacuums or API latency jitter.
2. **Order Book Depth Density (OBDD)**:
   - Evaluates weighted volume across 20 bid/ask levels instead of raw unweighted counts, weighting near-the-spread depth exponentially higher.
3. **Adaptive Sensor Covariance ($) Tuning**:
   - Instead of static quantization error floor, scales measurement covariance $ dynamically:
     R_v[k] = \frac{\Delta^2}{12} \times \left(1 + \alpha \frac{\text{Spread}[k]}{\text{TickSize}}\right)
   - When the spread widens (uncertainty spikes), the Kalman filter automatically reduces trust in incoming measurements and relies on dynamic momentum.
4. **Volume-Synchronized Sampling Verification**:
   - Discards zero-volume heartbeat candles to prevent artificial flat-line distortion in fractional differentiation.
