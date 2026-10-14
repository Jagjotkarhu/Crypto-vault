from enum import Enum
import numpy as np

class ExecutionState(Enum):
    IDLE = 'IDLE'
    ARMED = 'ARMED'
    IN_POSITION = 'IN_POSITION'
    ANTI_WINDUP_CLAMPED = 'ANTI_WINDUP_CLAMPED'
    EMERGENCY_DAMP = 'EMERGENCY_DAMP'

class DiscretePIDSizer:
    def __init__(self, Kp: float = 0.5, Ki: float = 0.1, Kd: float = 0.05, max_leverage: float = 5.0, tranche_inr: float = 500.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_leverage = max_leverage
        self.tranche_inr = tranche_inr
        self.max_notional_inr = tranche_inr * max_leverage
        self.integral = 0.0
        self.prev_error = 0.0
        self.state = ExecutionState.IDLE

    def compute(self, target_conviction: float, current_exposure_inr: float, dt: float = 1.0) -> tuple:
        target_inr = target_conviction * self.max_notional_inr
        error = target_inr - current_exposure_inr
        P = self.Kp * error
        D = self.Kd * (error - self.prev_error) / dt if dt > 0 else 0.0
        trial_integral = self.integral + error * dt
        trial_output = P + (self.Ki * trial_integral) + D
        if abs(trial_output) >= self.max_notional_inr:
            self.state = ExecutionState.ANTI_WINDUP_CLAMPED
            output = np.clip(trial_output, -self.max_notional_inr, self.max_notional_inr)
        else:
            self.integral = trial_integral
            output = trial_output
            self.state = ExecutionState.IN_POSITION if abs(output) > 10 else ExecutionState.IDLE
        self.prev_error = error
        return float(output), self.state
