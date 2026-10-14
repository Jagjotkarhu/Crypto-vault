import numpy as np
import pandas as pd

class MicrostructureCleaner:
    def __init__(self, tick_size=0.01, mad_threshold=3.5):
        self.tick_size = tick_size
        self.mad_threshold = mad_threshold

    def sanitize_candles(self, df):
        clean_df = df.copy()
        clean_df = clean_df[clean_df[df.columns[5]] > 0].reset_index(drop=True)
        ret = clean_df[df.columns[4]].pct_change().fillna(0)
        med = ret.median()
        mad = (ret - med).abs().median()
        if mad > 0:
            z = 0.6745 * (ret - med).abs() / mad
            mask = z > self.mad_threshold
            if mask.any():
                clean_df.loc[mask, df.columns[4]] = clean_df[df.columns[4]].shift(1)[mask]
        return clean_df

    def compute_weighted_obi(self, orderbook):
        bids = orderbook.get('bids', [])[:20]
        asks = orderbook.get('asks', [])[:20]
        if not bids or not asks:
            return 0.0, 50.0, (self.tick_size**2)/12.0
        best_b = bids[0][0]
        best_a = asks[0][0]
        spread = max(best_a - best_b, self.tick_size)
        d_b = np.exp(-0.15 * np.arange(len(bids)))
        w_b = sum(b[1] * d for b, d in zip(bids, d_b))
        d_a = np.exp(-0.15 * np.arange(len(asks)))
        w_a = sum(a[1] * d for a, d in zip(asks, d_a))
        tot = w_b + w_a
        obi = (w_b - w_a) / tot if tot > 0 else 0.0
        ratio = (w_b / tot) * 100.0 if tot > 0 else 50.0
        dyn_rv = ((self.tick_size**2)/12.0) * (1.0 + (spread / self.tick_size))
        return float(obi), float(ratio), float(dyn_rv)
