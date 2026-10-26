import sys
import time
import pandas as pd
from data.ws_streamer import BinanceWebSocketStreamer
from data.cleaner import MicrostructureCleaner
from data.ingestion import fetch_live_market_data
from core_math.kalman import MarketKalmanFilter
from core_math.fractional_diff import fractional_differentiation_ffd
from risk_control.pid_sizer import DiscretePIDSizer
from risk_control.adaptive_stops import VolatilityAdaptiveTrailingStop
from execution.paper_ledger import PaperTradingLedger, LedgerState
from config.settings import MAX_LEVERAGE, TRANCHE_MARGIN_INR, TOTAL_CAPITAL_INR


def run_paper_bot(pair: str = 'BTC/USDT', duration_seconds: int = 60, poll_interval: float = 1.0):
    print("=" * 70)
    print(f"[*] INITIALIZING REAL-TIME PAPER TRADING BOT: {pair}")
    print(f"[*] Capital: INR {TOTAL_CAPITAL_INR:.2f} | Tranche: INR {TRANCHE_MARGIN_INR:.2f} | Max Leverage: {MAX_LEVERAGE:.1f}x")
    print("=" * 70)

    # 1. Warm up baseline historical candles for 200 EMA & FracDiff
    print("Fetching initial baseline candles for filter initialization...")
    raw_df, init_liq = fetch_live_market_data(pair, timeframe='1m', limit=250)
    cleaner = MicrostructureCleaner(tick_size=0.01, mad_threshold=3.5)
    df = cleaner.sanitize_candles(raw_df)

    # 2. Initialize Persistent WebSocket Streamer
    streamer = BinanceWebSocketStreamer(pair=pair, timeframe='1m')
    streamer.start()
    print("Persistent Binance WebSocket connected. Awaiting stream sync...")
    time.sleep(2.0)  # Wait for first depth20 & kline packet

    # 3. Engines
    kf = MarketKalmanFilter(dt=1.0, tick_size=0.01)
    sizer = DiscretePIDSizer(Kp=0.6, Ki=0.15, Kd=0.05, max_leverage=MAX_LEVERAGE, tranche_inr=TRANCHE_MARGIN_INR)
    trailing_stop = VolatilityAdaptiveTrailingStop(alpha_atr=1.5, beta_cov=1.0, gamma_spread=2.0, min_stop_pct=0.005)
    ledger = PaperTradingLedger(initial_capital_inr=TOTAL_CAPITAL_INR)

    # Prime Kalman with historical close prices
    for p in df['close']:
        fair_price, velocity = kf.update(p)

    start_time = time.time()
    iteration = 0

    try:
        while (time.time() - start_time) < duration_seconds:
            iteration += 1
            live_price, ob, is_connected = streamer.get_market_state()

            if not live_price or not ob.get('bids') or not ob.get('asks'):
                time.sleep(poll_interval)
                continue

            best_bid = ob['bids'][0][0]
            best_ask = ob['asks'][0][0]
            spread = max(best_ask - best_bid, 0.01)

            # Microstructure Cleaner: Dynamic OBI & spread-adaptive noise Rv
            weighted_obi, bid_ratio, dyn_rv = cleaner.compute_weighted_obi(ob)
            fair_price, velocity = kf.update(live_price, measurement_var=dyn_rv)
            kalman_p00 = float(kf.P[0, 0])

            # ATR baseline from recent candle range
            atr = float((df['high'].iloc[-14:] - df['low'].iloc[-14:]).mean()) if len(df) >= 14 else spread * 5.0

            # 200 EMA Macro Regime
            ema_200 = float(df['close'].ewm(span=200, adjust=False).mean().iloc[-1])
            is_macro_bull = bool(live_price > ema_200)

            # FracDiff Momentum
            frac_series = fractional_differentiation_ffd(df['close'], d=0.38)
            recent_mom = float(frac_series.iloc[-1] - frac_series.iloc[0]) if len(frac_series) > 1 else 0.0

            # Multi-factor conviction
            bull_score = int(velocity > 0) + int(recent_mom > 0) + int(weighted_obi > 0.08) + int(is_macro_bull)
            conviction = 0.35 + (bull_score * 0.12)

            # State Machine & Paper Trading Execution Logic
            if ledger.state == LedgerState.IDLE:
                # Opportunity scan
                if conviction >= 0.55 and bull_score >= 2:
                    side = "LONG" if velocity > 0 else "SHORT"
                    target_inr, pid_state = sizer.compute(target_conviction=conviction, current_exposure_inr=0.0)
                    rec_leverage = min(abs(target_inr) / TRANCHE_MARGIN_INR, MAX_LEVERAGE)
                    
                    success, msg, pos = ledger.open_position(
                        symbol=pair,
                        side=side,
                        allocated_inr=TRANCHE_MARGIN_INR,
                        leverage=rec_leverage,
                        orderbook=ob
                    )
                    if success:
                        trailing_stop.arm(
                            side=side,
                            entry_price=pos['entry_price'],
                            initial_atr=atr,
                            kalman_p00=kalman_p00,
                            spread=spread
                        )
                        print(f"\n[EXECUTION] OPENED PAPER {side} @ ${pos['entry_price']:,.2f} | Leverage: {rec_leverage:.2f}x | Margin: INR {TRANCHE_MARGIN_INR:.2f}")

            elif ledger.state == LedgerState.IN_POSITION:
                # Live Mark-to-Market
                mtm = ledger.update_mtm(ob)
                should_exit, stop_reason, stop_metrics = trailing_stop.update(
                    current_price=live_price,
                    atr=atr,
                    kalman_p00=kalman_p00,
                    spread=spread,
                    kalman_velocity=velocity
                )

                if should_exit:
                    closed = ledger.close_position(ob, reason=stop_reason)
                    print(f"\n[STOP-EXIT] {stop_reason} triggered @ ${closed['exit_price']:,.2f} | Net PnL: INR {closed['net_pnl_inr']:+.2f} ({closed['return_pct']:+.2f}%)")

            # Periodic Status Line
            if iteration % 5 == 0:
                pos_status = f"IN_POS (MTM: INR {ledger.update_mtm(ob).get('unrealized_pnl_inr', 0.0):+.2f})" if ledger.state == LedgerState.IN_POSITION else "SCANNING (IDLE)"
                print(f"[{time.strftime('%H:%M:%S')}] P: ${live_price:,.2f} | Fair: ${fair_price:,.2f} | W-OBI: {weighted_obi:+.2f} | Conv: {conviction*100:.0f}% | State: {pos_status}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nStopping bot gracefully...")

    finally:
        streamer.stop()
        print("\n" + "=" * 70)
        print("[*] PAPER TRADING SESSION SUMMARY:")
        summary = ledger.get_summary()
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  - {k}: {v:,.2f}")
            else:
                print(f"  - {k}: {v}")
        print("=" * 70)


if __name__ == '__main__':
    pair_arg = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
    run_paper_bot(pair_arg, duration_seconds=15, poll_interval=1.0)
