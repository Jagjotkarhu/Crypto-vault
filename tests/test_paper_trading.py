import pytest
from risk_control.adaptive_stops import VolatilityAdaptiveTrailingStop
from execution.paper_ledger import PaperTradingLedger, LedgerState


def test_trailing_stop_monotonic_ratchet_long():
    stop = VolatilityAdaptiveTrailingStop(alpha_atr=1.0, beta_cov=1.0, gamma_spread=1.0, min_stop_pct=0.01)
    
    # Arm Long at 80,000 with initial stop distance ~ 800 (1%) -> initial stop = 79,200
    stop.arm(side="LONG", entry_price=80000.0, initial_atr=500.0, kalman_p00=100.0, spread=10.0)
    assert stop.is_active is True
    initial_stop = stop.current_stop
    assert initial_stop < 80000.0

    # Price rises to 82,000 -> Stop must ratchet UP
    should_exit, reason, metrics = stop.update(current_price=82000.0, atr=500.0, kalman_p00=100.0, spread=10.0)
    assert should_exit is False
    assert stop.current_stop > initial_stop
    ratcheted_stop = stop.current_stop

    # Price drops to 81,500 -> Stop must NOT move DOWN (monotonic ratchet invariant)
    should_exit, reason, metrics = stop.update(current_price=81500.0, atr=500.0, kalman_p00=100.0, spread=10.0)
    assert should_exit is False
    assert stop.current_stop == ratcheted_stop

    # Price collapses below ratcheted stop -> Must trigger exit
    breach_price = ratcheted_stop - 10.0
    should_exit, reason, metrics = stop.update(current_price=breach_price, atr=500.0, kalman_p00=100.0, spread=10.0)
    assert should_exit is True
    assert reason == 'TRAILING_STOP_BREACHED'
    assert stop.is_active is False


def test_trailing_stop_monotonic_ratchet_short():
    stop = VolatilityAdaptiveTrailingStop(alpha_atr=1.0, beta_cov=1.0, gamma_spread=1.0, min_stop_pct=0.01)
    
    # Arm Short at 80,000 -> initial stop > 80,000
    stop.arm(side="SHORT", entry_price=80000.0, initial_atr=500.0, kalman_p00=100.0, spread=10.0)
    initial_stop = stop.current_stop
    assert initial_stop > 80000.0

    # Price drops to 78,000 -> Stop must ratchet DOWN (lock profit for short)
    should_exit, reason, metrics = stop.update(current_price=78000.0, atr=500.0, kalman_p00=100.0, spread=10.0)
    assert should_exit is False
    assert stop.current_stop < initial_stop
    ratcheted_stop = stop.current_stop

    # Price rallies above ratcheted stop -> Must trigger short exit
    should_exit, reason, metrics = stop.update(current_price=ratcheted_stop + 10.0, atr=500.0, kalman_p00=100.0, spread=10.0)
    assert should_exit is True
    assert reason == 'TRAILING_STOP_BREACHED'


def test_paper_trading_ledger_long_flow():
    ledger = PaperTradingLedger(initial_capital_inr=5000.0, taker_fee_pct=0.0005, base_slippage_pct=0.0002)
    mock_ob = {
        'bids': [[80000.0, 2.0]],
        'asks': [[80005.0, 3.0]],
        'timestamp': 1000
    }

    # Open Long: 500 INR margin with 3x leverage -> 1500 INR notional
    success, msg, pos = ledger.open_position(
        symbol='BTC/USDT',
        side='LONG',
        allocated_inr=500.0,
        leverage=3.0,
        orderbook=mock_ob
    )
    assert success is True
    assert ledger.state == LedgerState.IN_POSITION
    assert pos['margin_inr'] == 500.0
    assert pos['notional_inr'] == 1500.0
    # Entry price should be ask * (1 + slippage)
    expected_entry = 80005.0 * 1.0002
    assert abs(pos['entry_price'] - expected_entry) < 0.01

    # Check live MTM when price rises to 81,000
    rising_ob = {'bids': [[81000.0, 2.0]], 'asks': [[81005.0, 3.0]]}
    mtm = ledger.update_mtm(rising_ob)
    assert mtm['unrealized_pnl_inr'] > 0

    # Close position
    trade = ledger.close_position(rising_ob, reason='TAKE_PROFIT')
    assert ledger.state == LedgerState.IDLE
    assert trade['net_pnl_inr'] > 0
    assert trade['fees_inr'] > 0
    assert ledger.current_capital_inr > 5000.0

    summary = ledger.get_summary()
    assert summary['total_trades'] == 1
    assert summary['winning_trades'] == 1
    assert summary['win_rate_pct'] == 100.0
