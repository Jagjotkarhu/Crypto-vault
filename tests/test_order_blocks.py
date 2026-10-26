import pytest
import pandas as pd
import numpy as np
from core_math.order_blocks import OrderBlockDetector, OrderBlockType


def test_order_block_detection_bullish():
    # Build synthetic OHLCV dataframe with a clear bullish impulse
    # 20 flat candles, then a red candle, then a massive green impulse candle
    candles = []
    base_price = 80000.0
    for i in range(20):
        candles.append({
            'timestamp': 1000 + i * 60000,
            'open': base_price,
            'high': base_price + 20.0,
            'low': base_price - 20.0,
            'close': base_price + 5.0,
            'volume': 10.0
        })

    # Preceding bearish origin candle (index 20)
    candles.append({
        'timestamp': 1000 + 20 * 60000,
        'open': 80010.0,
        'high': 80020.0,
        'low': 79980.0,
        'close': 79990.0,  # Red candle
        'volume': 25.0
    })

    # Violent Bullish Displacement Impulse candle (index 21)
    candles.append({
        'timestamp': 1000 + 21 * 60000,
        'open': 80000.0,
        'high': 80500.0,  # 500 range (massive relative to ~40 baseline)
        'low': 79990.0,
        'close': 80480.0, # Strong green body
        'volume': 150.0
    })

    # Follow-through candle
    candles.append({
        'timestamp': 1000 + 22 * 60000,
        'open': 80480.0,
        'high': 80550.0,
        'low': 80450.0,
        'close': 80520.0,
        'volume': 50.0
    })

    df = pd.DataFrame(candles)
    detector = OrderBlockDetector(atr_multiplier=1.8, atr_period=14)
    blocks = detector.detect_order_blocks(df)

    assert len(blocks) >= 1
    bullish_blocks = [b for b in blocks if b.block_type == OrderBlockType.BULLISH]
    assert len(bullish_blocks) >= 1

    ob = bullish_blocks[0]
    assert ob.top == 80020.0
    assert ob.bottom == 79980.0
    assert ob.mitigated is False


def test_order_block_mitigation_and_invalidation():
    # Setup Bullish OB and test breach
    detector = OrderBlockDetector(atr_multiplier=1.5, atr_period=5)
    candles = [
        {'open': 100, 'high': 102, 'low': 98, 'close': 101, 'volume': 10} for _ in range(8)
    ]
    # Origin red candle
    candles.append({'open': 100, 'high': 101, 'low': 96, 'close': 97, 'volume': 15})
    # Massive impulse green
    candles.append({'open': 98, 'high': 130, 'low': 97, 'close': 128, 'volume': 100})
    # Re-test into zone
    candles.append({'open': 128, 'high': 128, 'low': 99, 'close': 105, 'volume': 20})
    # Breakdown below low (invalidation)
    candles.append({'open': 105, 'high': 105, 'low': 90, 'close': 92, 'volume': 30})
    
    df = pd.DataFrame(candles)
    blocks = detector.detect_order_blocks(df)
    assert len(blocks) >= 1
    assert blocks[0].mitigated is True  # Should be marked mitigated/invalidated


def test_live_reversal_evaluation():
    detector = OrderBlockDetector(atr_multiplier=1.8, atr_period=14)
    candles = [
        {'open': 100, 'high': 102, 'low': 98, 'close': 100, 'volume': 10} for _ in range(20)
    ]
    # Origin red candle
    candles.append({'open': 100, 'high': 102, 'low': 96, 'close': 98, 'volume': 20})
    # Green impulse
    candles.append({'open': 99, 'high': 140, 'low': 98, 'close': 138, 'volume': 100})
    df = pd.DataFrame(candles)

    blocks = detector.detect_order_blocks(df)
    assert len(blocks) == 1
    ob = blocks[0]

    # Test price entering block at 100 with supportive OBI and slowing velocity
    setup = detector.evaluate_live_reversal(
        current_price=100.0,
        active_blocks=blocks,
        kalman_velocity=-2.0,
        weighted_obi=0.25
    )

    assert setup is not None
    assert setup['signal'] == 'BULLISH_REVERSAL'
    assert setup['entry_price'] == 100.0
    assert setup['stop_loss'] < 96.0  # Below bottom
    assert setup['take_profit_3r'] > 100.0
    assert setup['risk_reward_ratio'] == 3.0
