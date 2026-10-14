import sys
import pandas as pd
from data.ingestion import fetch_live_market_data
from core_math.kalman import MarketKalmanFilter
from core_math.fractional_diff import fractional_differentiation_ffd
from risk_control.pid_sizer import DiscretePIDSizer, ExecutionState
from data.cleaner import MicrostructureCleaner
from config.settings import MAX_LEVERAGE, TRANCHE_MARGIN_INR, MAX_POSITION_NOTIONAL_INR

def run_analysis(pair='BTC/USDT'):
    print('Fetching live 15m candles and order book for ' + str(pair) + '...')
    raw_df, liq = fetch_live_market_data(pair, timeframe='15m', limit=250)
    
    # 0. Microstructure Cleaning & Pre-processing
    cleaner = MicrostructureCleaner(tick_size=0.01, mad_threshold=3.5)
    df = cleaner.sanitize_candles(raw_df)
    current_close = df['close'].iloc[-1]
    
    # Dynamic Order Book Imbalance & Kalman Measurement Noise R_v
    raw_ob = liq.get('raw_orderbook', {})
    weighted_obi, bid_ratio, dyn_rv = cleaner.compute_weighted_obi(raw_ob)
    
    # 1. 200 EMA Macro Regime Filter
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    ema_200 = df['ema_200'].iloc[-1]
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
    
    # 4. Confluence Scoring (incorporating exponentially-weighted OBI)
    bull_score = int(velocity > 0) + int(recent_mom > 0) + int(weighted_obi > 0.08) + int(is_macro_bull)
    conviction = 0.35 + (bull_score * 0.12)
    
    # 5. Closed-Loop PID Sizing
    sizer = DiscretePIDSizer(Kp=0.6, Ki=0.15, Kd=0.05, max_leverage=MAX_LEVERAGE, tranche_inr=TRANCHE_MARGIN_INR)
    target_inr, state = sizer.compute(target_conviction=conviction, current_exposure_inr=0.0)
    
    print('==================================================================')
    print('[*] LIVE DAY-TRADE INTELLIGENCE BRIEF: ' + str(pair))
    print('------------------------------------------------------------------')
    print('- Market Raw Price:      $' + format(current_close, ',.2f'))
    print('- Kalman Denoised Fair:   $' + format(fair_price, ',.2f') + ' (Velocity: ' + format(velocity, '+.4f') + ', Dyn R_v: ' + format(dyn_rv, '.2e') + ')')
    print('- 200 EMA Macro Filter:   $' + format(ema_200, ',.2f') + ' -> ' + regime_str)
    print('- Order Book Liquidity:   ' + format(bid_ratio, '.1f') + '% Bids vs ' + format(100.0 - bid_ratio, '.1f') + '% Asks (W-OBI: ' + format(weighted_obi, '+.3f') + ')')
    print('- FracDiff Stationary:    ' + format(recent_mom, '+.4f'))
    print('- Multi-Factor Conviction: ' + format(conviction * 100.0, '.1f') + '% (Confluence: ' + str(bull_score) + '/4 factors aligned)')
    print('------------------------------------------------------------------')
    print('[CONFIG] CLOSED-LOOP PID POSITION SIZING AND RISK PLAN:')
    print('- Allocated Capital:      INR ' + format(target_inr, ',.2f') + ' (Max Cap: INR ' + format(MAX_POSITION_NOTIONAL_INR, ',.2f') + ')')
    print('- Recommended Leverage:   ' + format(abs(target_inr) / TRANCHE_MARGIN_INR, '.2f') + 'x (Hard Limit: ' + str(MAX_LEVERAGE) + 'x)')
    print('- Risk Per Trade (1.5%):  INR ' + format(abs(target_inr) * 0.015, ',.2f'))
    print('- Execution State:        ' + str(state.value))
    print('==================================================================')

if __name__ == '__main__':
    p_arg = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
    run_analysis(p_arg)
