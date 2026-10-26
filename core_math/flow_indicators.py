import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class FlowIndicators:
    """
    Mathematical engine for institutional order flow indicators:
    1. Cumulative Volume Delta (CVD) and Volume Absorption Divergences.
    2. Event/Session Anchored VWAP (AVWAP) with statistical variance bands (+/-1sigma, +/-2sigma).
    3. Multi-factor structural direction resolver.
    """

    @staticmethod
    def compute_cvd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates bar-by-bar volume delta and cumulative volume delta (CVD).
        Uses taker_buy_volume if present; otherwise applies the exact institutional
        price-action volume-distribution delta proxy.
        """
        result = df.copy()

        if 'taker_buy_volume' in result.columns and result['taker_buy_volume'].sum() > 0:
            buy_vol = result['taker_buy_volume']
            sell_vol = result['volume'] - buy_vol
            delta = buy_vol - sell_vol
        else:
            # High-fidelity volume distribution proxy:
            # delta = volume * ((close - low) - (high - close)) / (high - low)
            hl_range = result['high'] - result['low']
            hl_range = hl_range.replace(0, np.nan)
            ratio = (2.0 * result['close'] - result['high'] - result['low']) / hl_range
            ratio = ratio.fillna(0.0).clip(-1.0, 1.0)
            delta = result['volume'] * ratio

        result['volume_delta'] = delta
        result['cvd'] = delta.cumsum()
        return result

    @staticmethod
    def detect_cvd_divergence(df_with_cvd: pd.DataFrame, window: int = 14) -> Dict[str, Any]:
        """
        Detects institutional absorption divergences between price action and CVD.
        - Bullish Absorption: Price makes a lower low while CVD forms a higher low.
        - Bearish Absorption: Price makes a higher high while CVD forms a lower high.
        """
        if len(df_with_cvd) < window + 2:
            return {'has_divergence': False, 'type': 'NONE', 'strength': 0.0}

        sub = df_with_cvd.iloc[-window:].copy()
        current_close = float(sub['close'].iloc[-1])
        current_cvd = float(sub['cvd'].iloc[-1])
        prev_closes = sub['close'].iloc[:-1]
        prev_cvds = sub['cvd'].iloc[:-1]

        min_prev_close = float(prev_closes.min())
        min_prev_cvd = float(prev_cvds.min())
        max_prev_close = float(prev_closes.max())
        max_prev_cvd = float(prev_cvds.max())

        # Bullish Divergence: Price lower, CVD higher (Limit buyers absorbing market selling)
        if current_close <= min_prev_close and current_cvd > min_prev_cvd:
            cvd_delta = current_cvd - min_prev_cvd
            norm_strength = min(1.0, abs(cvd_delta) / (abs(min_prev_cvd) + 1e-6))
            return {
                'has_divergence': True,
                'type': 'BULLISH_ABSORPTION',
                'description': 'Price tested new lows but CVD formed higher low (Institutional Limit Bids absorbed selling)',
                'strength': float(norm_strength)
            }

        # Bearish Divergence: Price higher, CVD lower (Limit sellers absorbing market buying)
        if current_close >= max_prev_close and current_cvd < max_prev_cvd:
            cvd_delta = max_prev_cvd - current_cvd
            norm_strength = min(1.0, abs(cvd_delta) / (abs(max_prev_cvd) + 1e-6))
            return {
                'has_divergence': True,
                'type': 'BEARISH_ABSORPTION',
                'description': 'Price tested new highs but CVD formed lower high (Institutional Limit Asks absorbed buying)',
                'strength': float(norm_strength)
            }

        return {'has_divergence': False, 'type': 'NONE', 'strength': 0.0}

    @staticmethod
    def compute_anchored_vwap(df: pd.DataFrame, anchor_index: Optional[int] = None, lookback: int = 50) -> pd.DataFrame:
        """
        Computes Anchored VWAP and statistical variance bands (+/-1sigma, +/-2sigma).
        Anchors from a specific index or looks back a fixed number of candles.
        """
        result = df.copy()
        if anchor_index is None or anchor_index < 0 or anchor_index >= len(result):
            start_idx = max(0, len(result) - lookback)
        else:
            start_idx = anchor_index

        typ_price = (result['high'] + result['low'] + result['close']) / 3.0
        pv = typ_price * result['volume']

        avwap = np.full(len(result), np.nan)
        upper_1s = np.full(len(result), np.nan)
        lower_1s = np.full(len(result), np.nan)
        upper_2s = np.full(len(result), np.nan)
        lower_2s = np.full(len(result), np.nan)

        cum_pv = pv.iloc[start_idx:].cumsum()
        cum_vol = result['volume'].iloc[start_idx:].cumsum()

        vwap_series = cum_pv / cum_vol.replace(0, np.nan)
        avwap[start_idx:] = vwap_series.values

        # Compute volume-weighted standard deviation bands
        dev_sq_v = (typ_price.iloc[start_idx:] - vwap_series) ** 2 * result['volume'].iloc[start_idx:]
        cum_dev_sq_v = dev_sq_v.cumsum()
        variance = cum_dev_sq_v / cum_vol.replace(0, np.nan)
        std_dev = np.sqrt(variance.clip(lower=0.0)).values

        upper_1s[start_idx:] = vwap_series.values + std_dev
        lower_1s[start_idx:] = vwap_series.values - std_dev
        upper_2s[start_idx:] = vwap_series.values + (2.0 * std_dev)
        lower_2s[start_idx:] = vwap_series.values - (2.0 * std_dev)

        result['avwap'] = avwap
        result['avwap_upper_1s'] = upper_1s
        result['avwap_lower_1s'] = lower_1s
        result['avwap_upper_2s'] = upper_2s
        result['avwap_lower_2s'] = lower_2s
        return result

    @classmethod
    def analyze_market_flow(
        cls,
        df: pd.DataFrame,
        current_price: float,
        anchor_lookback: int = 50
    ) -> Dict[str, Any]:
        """
        Executes end-to-end flow intelligence combining CVD, absorption divergences,
        and Anchored VWAP relative positions.
        """
        df_cvd = cls.compute_cvd(df)
        divergence = cls.detect_cvd_divergence(df_cvd, window=14)
        df_flow = cls.compute_anchored_vwap(df_cvd, lookback=anchor_lookback)

        latest_avwap = float(df_flow['avwap'].iloc[-1])
        latest_upper_1s = float(df_flow['avwap_upper_1s'].iloc[-1])
        latest_lower_1s = float(df_flow['avwap_lower_1s'].iloc[-1])
        latest_upper_2s = float(df_flow['avwap_upper_2s'].iloc[-1])
        latest_lower_2s = float(df_flow['avwap_lower_2s'].iloc[-1])
        latest_cvd = float(df_flow['cvd'].iloc[-1])

        # Regime relative to AVWAP
        if current_price > latest_upper_2s:
            vwap_regime = "OVEREXTENDED_BULLISH (+2 Sigma)"
            vwap_bias = "MEAN_REVERSION_DOWN"
        elif current_price < latest_lower_2s:
            vwap_regime = "OVEREXTENDED_BEARISH (-2 Sigma)"
            vwap_bias = "MEAN_REVERSION_UP"
        elif current_price >= latest_avwap:
            vwap_regime = "BULLISH (Above Institutional VWAP)"
            vwap_bias = "BULLISH_TREND"
        else:
            vwap_regime = "BEARISH (Below Institutional VWAP)"
            vwap_bias = "BEARISH_TREND"

        # Directional synthesis
        bull_weight = 0.0
        bear_weight = 0.0

        if current_price >= latest_avwap:
            bull_weight += 0.35
        else:
            bear_weight += 0.35

        if divergence['has_divergence']:
            if divergence['type'] == 'BULLISH_ABSORPTION':
                bull_weight += 0.45
            elif divergence['type'] == 'BEARISH_ABSORPTION':
                bear_weight += 0.45

        # Net conviction
        total = bull_weight + bear_weight
        if total > 0:
            bull_prob = bull_weight / total
        else:
            bull_prob = 0.50

        if bull_prob >= 0.65:
            direction = "STRONG_BULLISH (UP)"
        elif bull_prob >= 0.55:
            direction = "BULLISH (UP)"
        elif bull_prob <= 0.35:
            direction = "STRONG_BEARISH (DOWN)"
        elif bull_prob <= 0.45:
            direction = "BEARISH (DOWN)"
        else:
            direction = "NEUTRAL (BALANCED)"

        return {
            'direction': direction,
            'directional_probability': bull_prob,
            'current_price': current_price,
            'avwap': latest_avwap,
            'avwap_dist_pct': ((current_price - latest_avwap) / latest_avwap) * 100.0,
            'upper_band_1s': latest_upper_1s,
            'lower_band_1s': latest_lower_1s,
            'upper_band_2s': latest_upper_2s,
            'lower_band_2s': latest_lower_2s,
            'vwap_regime': vwap_regime,
            'cvd_value': latest_cvd,
            'divergence': divergence
        }
