import pytest
import numpy as np
import pandas as pd
from core_math.flow_indicators import FlowIndicators


def test_cvd_calculation():
    # Synthetic data: 5 green bars with high close
    df = pd.DataFrame({
        'open': [100.0, 102.0, 104.0, 106.0, 108.0],
        'high': [103.0, 105.0, 107.0, 109.0, 111.0],
        'low': [99.0, 101.0, 103.0, 105.0, 107.0],
        'close': [102.5, 104.5, 106.5, 108.5, 110.5],
        'volume': [10.0, 10.0, 10.0, 10.0, 10.0]
    })

    res = FlowIndicators.compute_cvd(df)
    assert 'volume_delta' in res.columns
    assert 'cvd' in res.columns
    # Green candles with close near high should produce positive delta
    assert res['volume_delta'].iloc[-1] > 0
    assert res['cvd'].iloc[-1] > res['cvd'].iloc[0]


def test_cvd_bullish_absorption_divergence():
    # Construct a scenario where price makes a new lower low, but CVD forms a higher low
    n = 20
    closes = np.linspace(100, 90, n).tolist()
    # At the end, price makes a lower low (85), but taker volume surges positive
    closes.append(85.0)

    # Fake taker buy volume that surges at the end
    vols = [100.0] * n + [200.0]
    taker_buys = [30.0] * n + [180.0]  # Massive buy absorption at the bottom

    df = pd.DataFrame({
        'open': [c + 0.5 for c in closes],
        'high': [c + 1.0 for c in closes],
        'low': [c - 1.0 for c in closes],
        'close': closes,
        'volume': vols,
        'taker_buy_volume': taker_buys
    })

    df_cvd = FlowIndicators.compute_cvd(df)
    div = FlowIndicators.detect_cvd_divergence(df_cvd, window=10)
    assert div['has_divergence'] is True
    assert div['type'] == 'BULLISH_ABSORPTION'


def test_anchored_vwap_and_bands():
    n = 30
    prices = np.linspace(100, 130, n)
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 2.0,
        'low': prices - 2.0,
        'close': prices + 1.0,
        'volume': [50.0] * n
    })

    res = FlowIndicators.compute_anchored_vwap(df, lookback=20)
    assert 'avwap' in res.columns
    assert 'avwap_upper_1s' in res.columns
    assert 'avwap_lower_1s' in res.columns
    assert 'avwap_upper_2s' in res.columns
    assert 'avwap_lower_2s' in res.columns

    latest_vwap = res['avwap'].iloc[-1]
    u1 = res['avwap_upper_1s'].iloc[-1]
    l1 = res['avwap_lower_1s'].iloc[-1]
    u2 = res['avwap_upper_2s'].iloc[-1]
    l2 = res['avwap_lower_2s'].iloc[-1]

    assert l2 < l1 < latest_vwap < u1 < u2


def test_analyze_market_flow_synthesis():
    n = 40
    prices = np.linspace(100, 120, n)
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 1.5,
        'low': prices - 1.5,
        'close': prices + 0.5,
        'volume': [100.0] * n
    })

    analysis = FlowIndicators.analyze_market_flow(df, current_price=125.0)
    assert 'direction' in analysis
    assert 'avwap' in analysis
    assert 'vwap_regime' in analysis
    assert analysis['current_price'] == 125.0
    assert analysis['avwap_dist_pct'] > 0
