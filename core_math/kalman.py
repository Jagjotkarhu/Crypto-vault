import numpy as np

class MarketKalmanFilter:
    def __init__(self, dt: float = 1.0, tick_size: float = 0.01, process_var: float = 1e-4):
        self.dt = dt
        self.tick_size = tick_size
        self.A = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.C = np.array([[1.0, 0.0]], dtype=np.float64)
        quant_var = (tick_size ** 2) / 12.0
        self.R = np.array([[quant_var]], dtype=np.float64)
        q = process_var
        self.Q = np.array([[(dt**4)/4 * q, (dt**3)/2 * q], [(dt**3)/2 * q, (dt**2) * q]], dtype=np.float64)
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64)
        self.initialized = False

    def reset(self, initial_price: float):
        self.x = np.array([[initial_price], [0.0]], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * (self.tick_size ** 2)
        self.initialized = True

    def update(self, measured_price: float) -> tuple:
        if not self.initialized:
            self.reset(measured_price)
            return measured_price, 0.0
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q
        y = np.array([[measured_price]], dtype=np.float64) - (self.C @ x_pred)
        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        I = np.eye(2, dtype=np.float64)
        self.P = (I - K @ self.C) @ P_pred
        return float(self.x[0, 0]), float(self.x[1, 0])
