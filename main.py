import sys
import pandas as pd
from data.ingestion import fetch_live_market_data
from core_math.kalman import MarketKalmanFilter
from core_math.fractional_diff import fractional_differentiation_ffd
from risk_control.pid_sizer import DiscretePIDSizer, ExecutionState
from config.settings import MAX_LEVERAGE, TRANCHE_MARGIN_INR, MAX_POSITION_NOTIONAL_INR

def run_analysis(pair: str = 'BTC/USDT'):
    print(f'Fetching live 15m data for {pair}...')
    df = fetch_live_market_data(pair, timeframe='15m', limit=250)
    current_close = df['close'].iloc[-1]
    
    # 1. Kalman Denoising
    kf = MarketKalmanFilter(dt=1.0, tick_size=0.01)
    for p in df['close']:
        fair_price, velocity = kf.update(p)
        
    # 2. FracDiff (Memory + Stationarity)
    frac_series = fractional_differentiation_ffd(df['close'], d=0.38)
    recent_momentum = float(frac_series.iloc[-1] - frac_series.iloc[0]) if len(frac_series) > 1 else 0.0
    
    # 3. Conviction mapping (Demo heuristic pending PyTorch model training)
    conviction = 0.74 if velocity > 0 and recent_momentum > 0 else 0.40
    
    # 4. PID Position Sizing
    sizer = DiscretePIDSizer(Kp=0.6, Ki=0.15, Kd=0.05, max_leverage=MAX_LEVERAGE, tranche_inr=TRANCHE_MARGIN_INR)
    target_inr, state = sizer.compute(target_conviction=conviction, current_exposure_inr=0.0)
    
    print('========================================================')
    print(f'[*] LIVE ANALYSIS BRIEF: {pair}')
    print(f'- Market Raw Close:     ')
    print(f'- Kalman Denoised Fair:   (Velocity: {velocity:+.4f})')
    print(f'- FracDiff (d=0.38):     {recent_momentum:+.4f} (Stationary Momentum)')
    print(f'- Model Conviction:     {conviction*100:.1f}%')
    print('--------------------------------------------------------')
    print(f'[CONFIG] CLOSED-LOOP PID POSITION SIZING:')
    print(f'- Target Sizing (INR):   INR {target_inr:,.2f} / INR {MAX_POSITION_NOTIONAL_INR:,.2f} Max Cap')
    print(f'- Effective Leverage:    {abs(target_inr)/TRANCHE_MARGIN_INR:.2f}x (Clamped at {MAX_LEVERAGE}x)')
    print(f'- Execution State:       {state.value}')
    print('========================================================')

if __name__ == '__main__':
    pair = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
    run_analysis(pair)
