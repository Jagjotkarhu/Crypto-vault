import sys
import pandas as pd
from data.ingestion import fetch_live_market_data
from core_math.kalman import MarketKalmanFilter
from core_math.fractional_diff import fractional_differentiation_ffd
from core_math.flow_indicators import FlowIndicators
from core_math.order_blocks import OrderBlockDetector
from risk_control.pid_sizer import DiscretePIDSizer, ExecutionState
from data.cleaner import MicrostructureCleaner
from risk_control.scoring import score_factors, size_for_risk
from config.settings import (MAX_LEVERAGE, TRANCHE_MARGIN_INR, MAX_POSITION_NOTIONAL_INR,
                             TOTAL_CAPITAL_INR, RISK_PER_TRADE_PCT, MIN_EXCHANGE_NOTIONAL_INR)


def run_analysis(pair='BTC/USDT'):
    print('Fetching live 15m candles and order book for ' + str(pair) + '...')
    raw_df, liq = fetch_live_market_data(pair, timeframe='15m', limit=250)
    
    # 0. Microstructure Cleaning & Pre-processing
    cleaner = MicrostructureCleaner(tick_size=0.01, mad_threshold=3.5)
    df = cleaner.sanitize_candles(raw_df)
    current_close = float(df['close'].iloc[-1])
    
    # Dynamic Order Book Imbalance & Kalman Measurement Noise R_v
    raw_ob = liq.get('raw_orderbook', {})
    weighted_obi, bid_ratio, dyn_rv = cleaner.compute_weighted_obi(raw_ob)
    
    # 1. 200 EMA Macro Regime Filter
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    ema_200 = float(df['ema_200'].iloc[-1])
    is_macro_bull = bool(current_close > ema_200)
    regime_str = 'BULLISH (Above 200 EMA)' if is_macro_bull else 'BEARISH (Below 200 EMA)'
    
    # 2. Discrete Kalman Denoising with Dynamic Microstructure Covariance R_v
    kf = MarketKalmanFilter(dt=1.0, tick_size=0.01)
    for i, p in enumerate(df['close']):
        is_last = (i == len(df['close']) - 1)
        r_step = dyn_rv if is_last else None
        fair_price, velocity = kf.update(p, measurement_var=r_step)
        
    # 3. FracDiff Stationary Momentum
    frac_series = fractional_differentiation_ffd(df['close'], d=0.38)
    recent_mom = float(frac_series.iloc[-1] - frac_series.iloc[0]) if len(frac_series) > 1 else 0.0

    # 4. Institutional Flow: CVD & Anchored VWAP
    flow = FlowIndicators.analyze_market_flow(df, current_price=current_close, anchor_lookback=50)

    # 5. Institutional Order Blocks & Reversal Engine
    ob_detector = OrderBlockDetector(atr_multiplier=1.8, atr_period=14)
    detected_blocks = ob_detector.detect_order_blocks(df)
    active_unmitigated = [b for b in detected_blocks if not b.mitigated]
    reversal_setup = ob_detector.evaluate_live_reversal(current_close, detected_blocks, kalman_velocity=velocity, weighted_obi=weighted_obi)
    
    # 6. Multi-Factor Flow & Physics Confluence Scoring (6 Factors)
    flow_bull = bool(current_close >= flow['avwap'])
    cvd_bull_div = bool(flow['divergence'].get('has_divergence') and flow['divergence'].get('type') == 'BULLISH_ABSORPTION')
    
    cvd_vote = 1 if cvd_bull_div else (-1 if (flow['divergence'].get('has_divergence') and flow['divergence'].get('type') == 'BEARISH_ABSORPTION') else 0)
    reversal_vote = 0
    if reversal_setup is not None:
        reversal_vote = 1 if reversal_setup['signal'] == 'BULLISH_REVERSAL' else (-1 if reversal_setup['signal'] == 'BEARISH_REVERSAL' else 0)
    score = score_factors(velocity, recent_mom, weighted_obi, is_macro_bull, flow_bull, cvd_vote, reversal_vote)
    conviction = score.conviction
    direction_str = {'BULLISH': 'BULLISH (UP)', 'BEARISH': 'BEARISH (DOWN)', 'NEUTRAL': 'NEUTRAL (NO EDGE)'}[score.direction]

    # 7. Closed-Loop PID Sizing
    sizer = DiscretePIDSizer(Kp=0.6, Ki=0.15, Kd=0.05, max_leverage=MAX_LEVERAGE, tranche_inr=TRANCHE_MARGIN_INR)
    target_inr, state = sizer.compute(target_conviction=conviction, current_exposure_inr=0.0)

    # Equity-based risk cap: notional * stop_distance <= 1.5% of ACCOUNT EQUITY
    atr14 = float((df['high'].iloc[-14:] - df['low'].iloc[-14:]).mean())
    if reversal_setup is not None:
        stop_price = float(reversal_setup['stop_loss'])
    else:
        stop_price = current_close + (1.5 * atr14 if score.direction == 'BEARISH' else -1.5 * atr14)
    sizing = size_for_risk(target_inr, current_close, stop_price, TOTAL_CAPITAL_INR,
                           RISK_PER_TRADE_PCT, MAX_POSITION_NOTIONAL_INR, MIN_EXCHANGE_NOTIONAL_INR)
    if score.direction == 'NEUTRAL':
        sizing.notional_inr = 0.0
        sizing.risk_inr = 0.0
    
    print('==================================================================')
    print('[*] LIVE QUANTITATIVE INTELLIGENCE BRIEF: ' + str(pair))
    print('------------------------------------------------------------------')
    print('- Data As Of (UTC):      ' + str(df['datetime'].iloc[-1]) + ' (last 15m candle open)')
    print('- Market Raw Price:      $' + format(current_close, ',.2f'))
    print('- Kalman Denoised Fair:   $' + format(fair_price, ',.2f') + ' (Velocity: ' + format(velocity, '+.4f') + ', Dyn R_v: ' + format(dyn_rv, '.2e') + ')')
    print('- 200 EMA Macro Filter:   $' + format(ema_200, ',.2f') + ' -> ' + regime_str)
    print('- Order Book Liquidity:   ' + format(bid_ratio, '.1f') + '% Bids vs ' + format(100.0 - bid_ratio, '.1f') + '% Asks (W-OBI: ' + format(weighted_obi, '+.3f') + ')')
    print('- FracDiff Stationary:    ' + format(recent_mom, '+.4f'))
    print('------------------------------------------------------------------')
    print('[FLOW] INSTITUTIONAL ORDER FLOW & VOLUMETRIC SIGNALS:')
    print('- Anchored VWAP (50-bar): $' + format(flow['avwap'], ',.2f') + ' (' + format(flow['avwap_dist_pct'], '+.2f') + '% from VWAP)')
    print('  * 1-Sigma Band:         $' + format(flow['lower_band_1s'], ',.2f') + ' to $' + format(flow['upper_band_1s'], ',.2f'))
    print('  * 2-Sigma Band:         $' + format(flow['lower_band_2s'], ',.2f') + ' to $' + format(flow['upper_band_2s'], ',.2f'))
    print('  * Flow Regime:          ' + flow['vwap_regime'])
    print('- Cumulative Delta (CVD): ' + format(flow['cvd_value'], '+,.1f') + ' contracts')
    if flow['divergence']['has_divergence']:
        print('  * [!] CVD DIVERGENCE:   ' + flow['divergence']['type'] + ' (Strength: ' + format(flow['divergence']['strength'] * 100, '.1f') + '%)')
        print('                          ' + flow['divergence']['description'])
    else:
        print('  * CVD Divergence:       None (Volume and price in phase)')
    print('- Active Order Blocks:    ' + str(len(active_unmitigated)) + ' unmitigated zones mapped')
    if reversal_setup:
        print('  * [!] ORDER BLOCK SETUP: ' + reversal_setup['signal'] + ' @ $' + format(reversal_setup['entry_price'], ',.2f'))
        print('                          Zone: [$' + format(reversal_setup['zone_bottom'], ',.2f') + ' - $' + format(reversal_setup['zone_top'], ',.2f') + '], Stop: $' + format(reversal_setup['stop_loss'], ',.2f') + ', 3R Target: $' + format(reversal_setup['take_profit_3r'], ',.2f'))
    print('------------------------------------------------------------------')
    print('[DECISION] DIRECTIONAL SYNTHESIS & CONFLUENCE:')
    print('- Forecasted Direction:   ' + direction_str)
    print('- Factor Votes:           ' + str(score.bear) + ' bearish / ' + str(score.bull) + ' bullish / ' + str(score.neutral) + ' neutral (of ' + str(score.total) + ')')
    for f in score.factors:
        print('  * ' + f.name.ljust(30) + ('BULLISH' if f.vote > 0 else 'BEARISH' if f.vote < 0 else 'NEUTRAL'))
    print('- Signal Strength:        ' + format(conviction * 100.0, '.1f') + '% (uncalibrated vote margin, not a win probability)')
    print('------------------------------------------------------------------')
    print('[CONFIG] CLOSED-LOOP PID POSITION SIZING AND RISK PLAN:')
    print('- PID Target Notional:    INR ' + format(abs(target_inr), ',.2f') + ' (Max Cap: INR ' + format(MAX_POSITION_NOTIONAL_INR, ',.2f') + ')')
    print('- Risk-Capped Notional:   INR ' + format(sizing.notional_inr, ',.2f') + ' (limit set by: ' + sizing.binding + ')')
    print('- Recommended Leverage:   ' + format(sizing.notional_inr / TRANCHE_MARGIN_INR, '.2f') + 'x (Hard Limit: ' + str(MAX_LEVERAGE) + 'x)')
    print('- Stop Distance:          ' + format(sizing.stop_pct * 100.0, '.2f') + '% (stop $' + format(stop_price, ',.2f') + ')')
    print('- Risk Per Trade:         INR ' + format(sizing.risk_inr, ',.2f') + ' of INR ' + format(sizing.risk_budget_inr, ',.2f') + ' budget (' + format(RISK_PER_TRADE_PCT * 100.0, '.1f') + '% of INR ' + format(TOTAL_CAPITAL_INR, ',.0f') + ' equity)')
    if sizing.below_min_notional and sizing.notional_inr > 0:
        print('- [!] WARNING:            Notional below exchange minimum (INR ' + format(MIN_EXCHANGE_NOTIONAL_INR, ',.0f') + '); order would be rejected.')
    print('- Execution State:        IDLE (analysis only; no position is assumed. Sizer internal state: ' + str(state.value) + ')')
    print('==================================================================')


if __name__ == '__main__':
    p_arg = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
    run_analysis(p_arg)
