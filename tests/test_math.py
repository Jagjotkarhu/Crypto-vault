import pytest
import numpy as np
import pandas as pd
from core_math.kalman import MarketKalmanFilter
from core_math.fractional_diff import get_weights_ffd, fractional_differentiation_ffd
from risk_control.pid_sizer import DiscretePIDSizer, ExecutionState

def test_kalman_filter():
    kf = MarketKalmanFilter(dt=1.0, tick_size=0.01)
    prices = [100.0, 100.02, 100.01, 100.05, 100.08]
    for p in prices:
        fair, vel = kf.update(p)
    assert 99.0 < fair < 101.0
    assert isinstance(vel, float)

def test_fractional_diff_weights():
    w = get_weights_ffd(d=0.4, thres=1e-3)
    assert len(w) > 1
    assert abs(w[-1] - 1.0) < 1e-6

def test_pid_anti_windup_clamp():
    sizer = DiscretePIDSizer(Kp=1.0, Ki=0.5, Kd=0.1, max_leverage=5.0, tranche_inr=500.0)
    # Target 100% conviction with 0 exposure
    target_inr, state = sizer.compute(target_conviction=1.0, current_exposure_inr=0.0)
    assert target_inr <= 2500.0
    assert state in [ExecutionState.IN_POSITION, ExecutionState.ANTI_WINDUP_CLAMPED]
