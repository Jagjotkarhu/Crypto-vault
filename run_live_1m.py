import json
import datetime
import pandas as pd
from data.ingestion import fetch_live_market_data
from core_math.kalman import MarketKalmanFilter
from core_math.fractional_diff import fractional_differentiation_ffd
from risk_control.pid_sizer import DiscretePIDSizer
from config.settings import MAX_LEVERAGE, TRANCHE_MARGIN_INR, MAX_POSITION_NOTIONAL_INR

def run_live_1m():
    pair = 'BTC/USDT'
    print(f"Fetching live 1m candles and L2 order book for {pair}...")
    df, liq = fetch_live_market_data(pair, timeframe='1m', limit=300)
    current_close = float(df['close'].iloc[-1])
    current_timestamp = int(df['timestamp'].iloc[-1])
    
    # 1. 200 EMA Filter (1m)
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    ema_200 = float(df['ema_200'].iloc[-1])
    is_macro_bull = bool(current_close > ema_200)
    regime_str = 'BULLISH (Above 200 EMA)' if is_macro_bull else 'BEARISH (Below 200 EMA)'
    
    # 2. Discrete Kalman Denoising
    # On 1m candles, dt = 1.0 (min)
    kf = MarketKalmanFilter(dt=1.0, tick_size=0.01)
    for p in df['close']:
        fair_price, velocity = kf.update(p)
    
    # 3. FracDiff Stationary Momentum (d=0.38)
    frac_series = fractional_differentiation_ffd(df['close'], d=0.38)
    recent_mom = float(frac_series.iloc[-1] - frac_series.iloc[0]) if len(frac_series) > 1 else 0.0
    
    # 4. Multi-Factor Confluence & Directional Prediction
    bull_factors = {
        "kalman_velocity_positive": bool(velocity > 0),
        "stationary_momentum_positive": bool(recent_mom > 0),
        "obi_buy_pressure": bool(liq['obi'] > 0.05),
        "above_200_ema": is_macro_bull
    }
    bull_score = sum(bull_factors.values())
    conviction = 0.35 + (bull_score * 0.12)
    
    # Clear directional prediction
    if bull_score >= 3:
        direction = "BULLISH (UP)"
    elif bull_score <= 1:
        direction = "BEARISH (DOWN)"
    else:
        # Tie breaker / nuanced: based on Kalman velocity + OBI
        if velocity > 0 and liq['obi'] > 0:
            direction = "BULLISH (UP)"
        elif velocity < 0 and liq['obi'] < 0:
            direction = "BEARISH (DOWN)"
        elif velocity > 0:
            direction = "BULLISH (UP)"
        else:
            direction = "BEARISH (DOWN)"
            
    # 5. Closed-Loop PID Sizing
    sizer = DiscretePIDSizer(Kp=0.6, Ki=0.15, Kd=0.05, max_leverage=MAX_LEVERAGE, tranche_inr=TRANCHE_MARGIN_INR)
    target_inr, state = sizer.compute(target_conviction=conviction, current_exposure_inr=0.0)
    
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    check_time_utc = now_utc + datetime.timedelta(minutes=5)
    
    record = {
        "timestamp_utc": now_utc.isoformat(),
        "candle_timestamp_ms": current_timestamp,
        "candle_datetime_utc": str(df['datetime'].iloc[-1]),
        "target_check_time_utc": check_time_utc.isoformat(),
        "pair": pair,
        "timeframe": "1m",
        "baseline_price": current_close,
        "kalman_fair_price": round(fair_price, 2),
        "kalman_velocity": round(velocity, 4),
        "ema_200": round(ema_200, 2),
        "macro_regime": regime_str,
        "orderbook": {
            "obi": round(liq['obi'], 4),
            "bid_ratio_pct": round(liq['bid_ratio'], 2),
            "bid_volume": round(liq['bid_volume'], 4),
            "ask_volume": round(liq['ask_volume'], 4),
            "best_bid": liq['best_bid'],
            "best_ask": liq['best_ask']
        },
        "stationary_momentum": round(recent_mom, 6),
        "bull_confluence_score": f"{bull_score}/4",
        "conviction_pct": round(conviction * 100.0, 1),
        "directional_prediction": direction,
        "pid_position_inr": round(target_inr, 2),
        "recommended_leverage": round(abs(target_inr) / TRANCHE_MARGIN_INR, 2),
        "factors": bull_factors
    }
    
    output_path = r"C:\Users\gotog\Crypto-vault\live_test_1m.json"
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
        
    print("\n--- RESULTS SAVED TO live_test_1m.json ---")
    print(json.dumps(record, indent=2))
    return record

if __name__ == '__main__':
    run_live_1m()
