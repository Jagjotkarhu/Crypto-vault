import numpy as np
import pandas as pd

def get_weights_ffd(d: float, thres: float = 1e-4, max_lags: int = 100) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_lags:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def fractional_differentiation_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    weights = get_weights_ffd(d, thres)
    width = len(weights) - 1
    res = {}
    for i in range(width, len(series)):
        window = series.iloc[i - width : i + 1]
        res[series.index[i]] = np.dot(weights, window)
    return pd.Series(res)
