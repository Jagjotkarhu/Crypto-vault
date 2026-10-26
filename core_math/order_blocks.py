import numpy as np
import pandas as pd
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class OrderBlockType(Enum):
    BULLISH = "BULLISH_OB"
    BEARISH = "BEARISH_OB"


class OrderBlock:
    """
    Physical representation of an institutional Order Block zone.
    Stores the structural boundaries, origin impulse metrics, and mitigation state.
    """

    def __init__(
        self,
        index: int,
        timestamp: Any,
        block_type: OrderBlockType,
        top: float,
        bottom: float,
        volume: float,
        impulse_range: float
    ):
        self.index = int(index)
        self.timestamp = timestamp
        self.block_type = block_type
        self.top = float(top)
        self.bottom = float(bottom)
        self.volume = float(volume)
        self.impulse_range = float(impulse_range)
        self.mean_price = (self.top + self.bottom) / 2.0  # 50% Equilibrium

        # Lifecycle tracking
        self.mitigated = False
        self.mitigation_index: Optional[int] = None
        self.mitigation_price: Optional[float] = None
        self.touch_count = 0

    def check_interaction(self, candle_high: float, candle_low: float, candle_close: float, index: int) -> Tuple[bool, bool]:
        """
        Evaluates whether a subsequent candle touches or invalidates this Order Block.
        Returns:
            (is_touched, is_invalidated)
        """
        if self.mitigated:
            return False, False

        is_touched = False
        is_invalidated = False

        if self.block_type == OrderBlockType.BULLISH:
            # Bullish OB is penetrated when price dips into [bottom, top]
            if candle_low <= self.top and candle_high >= self.bottom:
                is_touched = True
                self.touch_count += 1
            # Invalidation: candle closes strictly below the structural floor
            if candle_close < self.bottom:
                is_invalidated = True
                self.mitigated = True
                self.mitigation_index = index
                self.mitigation_price = candle_close

        elif self.block_type == OrderBlockType.BEARISH:
            # Bearish OB is penetrated when price rises into [bottom, top]
            if candle_high >= self.bottom and candle_low <= self.top:
                is_touched = True
                self.touch_count += 1
            # Invalidation: candle closes strictly above the structural ceiling
            if candle_close > self.top:
                is_invalidated = True
                self.mitigated = True
                self.mitigation_index = index
                self.mitigation_price = candle_close

        return is_touched, is_invalidated


class OrderBlockDetector:
    """
    Deterministic quantitative engine for discovering unmitigated Order Blocks,
    evaluating elastic retests, and validating reversal setups using Kalman velocity and OBI.
    """

    def __init__(self, atr_multiplier: float = 1.8, atr_period: int = 14, body_ratio_min: float = 0.55):
        self.atr_multiplier = float(atr_multiplier)
        self.atr_period = int(atr_period)
        self.body_ratio_min = float(body_ratio_min)

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Computes true range and Wilder's ATR."""
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period, min_periods=1).mean()

    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Scans OHLCV history to isolate institutional impulse candles and their origin blocks.
        Tracks forward retests to mark mitigation status.
        """
        if len(df) < self.atr_period + 2:
            return []

        atr = self.compute_atr(df)
        order_blocks: List[OrderBlock] = []

        # 1. Identify displacement candles and form Order Blocks
        for i in range(1, len(df)):
            c_open = float(df['open'].iloc[i])
            c_high = float(df['high'].iloc[i])
            c_low = float(df['low'].iloc[i])
            c_close = float(df['close'].iloc[i])
            c_range = c_high - c_low
            c_body = abs(c_close - c_open)
            current_atr = float(atr.iloc[i])

            if current_atr <= 0 or c_range <= 0:
                continue

            # Check if this candle is an institutional displacement impulse
            is_impulse = (c_range >= self.atr_multiplier * current_atr) and ((c_body / c_range) >= self.body_ratio_min)

            if not is_impulse:
                continue

            prev_open = float(df['open'].iloc[i - 1])
            prev_close = float(df['close'].iloc[i - 1])
            prev_high = float(df['high'].iloc[i - 1])
            prev_low = float(df['low'].iloc[i - 1])
            prev_vol = float(df['volume'].iloc[i - 1])
            prev_time = df['timestamp'].iloc[i - 1] if 'timestamp' in df.columns else i - 1

            # BULLISH IMPULSE (Upward Breakout) -> Origin is preceding Bearish candle
            if c_close > c_open and prev_close <= prev_open:
                ob = OrderBlock(
                    index=i - 1,
                    timestamp=prev_time,
                    block_type=OrderBlockType.BULLISH,
                    top=prev_high,
                    bottom=prev_low,
                    volume=prev_vol,
                    impulse_range=c_range
                )
                order_blocks.append(ob)

            # BEARISH IMPULSE (Downward Breakout) -> Origin is preceding Bullish candle
            elif c_close < c_open and prev_close >= prev_open:
                ob = OrderBlock(
                    index=i - 1,
                    timestamp=prev_time,
                    block_type=OrderBlockType.BEARISH,
                    top=prev_high,
                    bottom=prev_low,
                    volume=prev_vol,
                    impulse_range=c_range
                )
                order_blocks.append(ob)

        # 2. Forward pass to simulate mitigation & invalidation
        for ob in order_blocks:
            start_scan = ob.index + 2  # Skip the impulse candle itself
            for j in range(start_scan, len(df)):
                k_high = float(df['high'].iloc[j])
                k_low = float(df['low'].iloc[j])
                k_close = float(df['close'].iloc[j])
                touched, invalidated = ob.check_interaction(k_high, k_low, k_close, j)
                if invalidated:
                    break

        return order_blocks

    def evaluate_live_reversal(
        self,
        current_price: float,
        active_blocks: List[OrderBlock],
        kalman_velocity: float = 0.0,
        weighted_obi: float = 0.0,
        buffer_pct: float = 0.001
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates whether current price is testing an unmitigated Order Block
        with dynamic confluence from Kalman velocity and Order Book Imbalance.
        """
        unmitigated = [ob for ob in active_blocks if not ob.mitigated]

        for ob in sorted(unmitigated, key=lambda x: x.index, reverse=True):
            if ob.block_type == OrderBlockType.BULLISH:
                # Price is inside or at the boundary of a Bullish OB
                if (ob.bottom * (1.0 - buffer_pct)) <= current_price <= (ob.top * (1.0 + buffer_pct)):
                    # Check reversal confirmation: Kalman velocity turning non-negative or OBI showing buy bids
                    is_velocity_turning = kalman_velocity >= -15.0  # Downward velocity decelerating
                    is_obi_supportive = weighted_obi > 0.05

                    confluence = int(is_velocity_turning) + int(is_obi_supportive) + int(current_price >= ob.mean_price)
                    stop_loss = ob.bottom * (1.0 - buffer_pct)
                    risk_dist = current_price - stop_loss
                    target_tp = current_price + (risk_dist * 3.0)  # Standard 1:3 R:R

                    return {
                        'block_type': ob.block_type.value,
                        'signal': 'BULLISH_REVERSAL',
                        'entry_price': current_price,
                        'zone_top': ob.top,
                        'zone_bottom': ob.bottom,
                        'equilibrium_50': ob.mean_price,
                        'stop_loss': stop_loss,
                        'take_profit_3r': target_tp,
                        'risk_reward_ratio': 3.0,
                        'kalman_velocity': kalman_velocity,
                        'weighted_obi': weighted_obi,
                        'confluence_score': confluence
                    }

            elif ob.block_type == OrderBlockType.BEARISH:
                # Price is inside or at the boundary of a Bearish OB
                if (ob.bottom * (1.0 - buffer_pct)) <= current_price <= (ob.top * (1.0 + buffer_pct)):
                    is_velocity_turning = kalman_velocity <= 15.0  # Upward velocity decelerating
                    is_obi_supportive = weighted_obi < -0.05

                    confluence = int(is_velocity_turning) + int(is_obi_supportive) + int(current_price <= ob.mean_price)
                    stop_loss = ob.top * (1.0 + buffer_pct)
                    risk_dist = stop_loss - current_price
                    target_tp = current_price - (risk_dist * 3.0)

                    return {
                        'block_type': ob.block_type.value,
                        'signal': 'BEARISH_REVERSAL',
                        'entry_price': current_price,
                        'zone_top': ob.top,
                        'zone_bottom': ob.bottom,
                        'equilibrium_50': ob.mean_price,
                        'stop_loss': stop_loss,
                        'take_profit_3r': target_tp,
                        'risk_reward_ratio': 3.0,
                        'kalman_velocity': kalman_velocity,
                        'weighted_obi': weighted_obi,
                        'confluence_score': confluence
                    }

        return None
