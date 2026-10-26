import math
from typing import Dict, Any, Tuple


class VolatilityAdaptiveTrailingStop:
    """
    Volatility and Kalman-Covariance Adaptive Trailing Stop-Loss Engine.
    Replaces static fixed stops with a mathematically derived dynamic boundary:
      D_stop = max(min_stop_pct * entry, alpha * ATR + beta * sqrt(P_00) + gamma * spread)
    Guarantees monotonic ratcheting (profits locked in, risk never widened).
    """

    def __init__(
        self,
        alpha_atr: float = 1.5,
        beta_cov: float = 1.0,
        gamma_spread: float = 2.0,
        min_stop_pct: float = 0.005,  # 0.5% minimum hard cushion
        max_stop_pct: float = 0.030   # 3.0% maximum safety ceiling
    ):
        self.alpha_atr = alpha_atr
        self.beta_cov = beta_cov
        self.gamma_spread = gamma_spread
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct

        # Active state
        self.is_active = False
        self.side: str = "LONG"
        self.entry_price: float = 0.0
        self.current_stop: float = 0.0
        self.peak_favorable_price: float = 0.0
        self.last_distance: float = 0.0

    def arm(self, side: str, entry_price: float, initial_atr: float = 0.0, kalman_p00: float = 0.0, spread: float = 0.0):
        """Arms the trailing stop upon order fill."""
        self.side = side.upper()
        self.entry_price = float(entry_price)
        self.peak_favorable_price = float(entry_price)
        self.is_active = True

        initial_distance = self._compute_stop_distance(entry_price, initial_atr, kalman_p00, spread)
        self.last_distance = initial_distance

        if self.side == "LONG":
            self.current_stop = self.entry_price - initial_distance
        else:
            self.current_stop = self.entry_price + initial_distance

    def _compute_stop_distance(self, price: float, atr: float, kalman_p00: float, spread: float) -> float:
        """Derives volatility & estimation-uncertainty scaled stop buffer."""
        uncert = math.sqrt(max(0.0, kalman_p00))
        dynamic_dist = (self.alpha_atr * atr) + (self.beta_cov * uncert) + (self.gamma_spread * spread)

        min_allowed = price * self.min_stop_pct
        max_allowed = price * self.max_stop_pct

        # Clamp between min cushion and max safety ceiling
        return max(min_allowed, min(dynamic_dist, max_allowed))

    def update(
        self,
        current_price: float,
        atr: float = 0.0,
        kalman_p00: float = 0.0,
        spread: float = 0.0,
        kalman_velocity: float = 0.0
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Updates the trailing stop level monotonically and checks for exit breaches.
        Returns:
            should_exit: bool
            reason: str ('STOP_BREACHED', 'INNOVATION_SHOCK', 'HOLD')
            metrics: dict
        """
        if not self.is_active:
            return False, 'INACTIVE', {}

        distance = self._compute_stop_distance(current_price, atr, kalman_p00, spread)
        self.last_distance = distance

        should_exit = False
        reason = 'HOLD'

        if self.side == "LONG":
            # Track peak high
            if current_price > self.peak_favorable_price:
                self.peak_favorable_price = current_price

            candidate_stop = current_price - distance
            # Monotonic ratchet: stop can ONLY move UP
            if candidate_stop > self.current_stop:
                self.current_stop = candidate_stop

            # Check breach
            if current_price <= self.current_stop:
                should_exit = True
                reason = 'TRAILING_STOP_BREACHED'

        elif self.side == "SHORT":
            # Track lowest low
            if current_price < self.peak_favorable_price:
                self.peak_favorable_price = current_price

            candidate_stop = current_price + distance
            # Monotonic ratchet: stop can ONLY move DOWN
            if candidate_stop < self.current_stop:
                self.current_stop = candidate_stop

            # Check breach
            if current_price >= self.current_stop:
                should_exit = True
                reason = 'TRAILING_STOP_BREACHED'

        metrics = {
            'is_active': self.is_active,
            'side': self.side,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'current_stop': self.current_stop,
            'stop_distance': distance,
            'stop_distance_pct': (distance / current_price) * 100.0 if current_price > 0 else 0.0,
            'peak_price': self.peak_favorable_price,
            'unrealized_pnl_pct': ((current_price - self.entry_price) / self.entry_price * 100.0) if self.side == "LONG" else ((self.entry_price - current_price) / self.entry_price * 100.0)
        }

        if should_exit:
            self.disarm()

        return should_exit, reason, metrics

    def disarm(self):
        """Disarms the stop engine once trade exits."""
        self.is_active = False
