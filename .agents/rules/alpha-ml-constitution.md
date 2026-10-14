# 🛡️ Constitutional Invariants: Alpha ML Researcher

These rules are universally enforced across all development and autonomous executions in this repository.

## 1. The 7 Constitutional Golden Rules
1. **Pure Math & ECE Control Foundations**: Strictly deterministic control algorithms and statistical grounding. Zero arbitrary magic indicators.
2. **Reused Verified Math Core**: Inherits verified LTI math representations (StateSpace, PID, LQR, stability criteria) from verified control systems.
3. **PyTorch Deep Learning Integration**: Temporal architectures trained exclusively with **Purged & Embargoed Cross-Validation**.
4. **Strict Discrete-Time (z-domain) Math**: All filters and controllers operate in discrete time. Closed-loop poles must satisfy |z_i| < 1.
5. **Actuator Saturation & Anti-Windup Clamping**: Hard leverage cap clamped at **5x**. Integral term halts accumulation immediately when active tranche hits limit.
6. **Transport Delay & Latency Margins**: Models exchange round-trip delay (z^-d) and enforces Phase Margin > 35 deg to prevent destabilizing execution whipsaws.
7. **Fractional Differentiation Stationarity Gate**: Raw prices never enter ML models. Series are transformed with optimal fractional order d* passing the ADF test (p < 0.05).

## 2. DSP & DSD Mathematical Invariants
- **Linear-Phase FIR Topology**: Windowed Fractional Differentiation operates with zero phase distortion to keep momentum synchronized with real-time price discovery.
- **Quantization Noise Modeling**: Sensor measurement variance Rv scales dynamically with asset tick size: Rv = (Delta^2) / 12.
- **Execution Finite State Machine (FSM)**: State transitions follow strict Moore/Mealy rules: IDLE -> ARMED -> IN_POSITION -> ANTI_WINDUP_CLAMPED -> EMERGENCY_DAMP.

## 3. Account Capital & Risk Allocations
- Account Capital: 5,000 INR
- Active Margin Tranche: 500 INR (10% allocation)
- Hard Leverage Cap: 5.0x
- Maximum Position Notional: 2,500 INR
- Max Risk Per Trade (1R): 1.5% of notional (~15 to 37.50 INR)
