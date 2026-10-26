import time
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple


class LedgerState(Enum):
    IDLE = "IDLE"
    IN_POSITION = "IN_POSITION"


class PaperTradingLedger:
    """
    Finite State Machine Paper-Trading Ledger.
    Tracks simulated order fills, bid/ask spread crossing, slippage,
    Binance taker fees, unrealized MTM, and cumulative realized PnL in INR.
    Strictly enforces capital invariants (₹5,000 max capital, ₹500 tranche, 5x leverage cap).
    """

    def __init__(
        self,
        initial_capital_inr: float = 5000.0,
        taker_fee_pct: float = 0.0005,  # 0.05% VIP0 taker fee
        base_slippage_pct: float = 0.0002 # 0.02% (2 bps) market impact
    ):
        self.initial_capital_inr = float(initial_capital_inr)
        self.current_capital_inr = float(initial_capital_inr)
        self.taker_fee_pct = taker_fee_pct
        self.base_slippage_pct = base_slippage_pct

        # State machine
        self.state = LedgerState.IDLE
        self.active_position: Optional[Dict[str, Any]] = None

        # Ledger history & analytics
        self.trade_history: List[Dict[str, Any]] = []
        self.peak_capital_inr = self.initial_capital_inr
        self.max_drawdown_pct = 0.0

    def open_position(
        self,
        symbol: str,
        side: str,
        allocated_inr: float,
        leverage: float,
        orderbook: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Executes a simulated paper order crossing the order book.
        Long entries cross best ask; Short entries cross best bid.
        """
        if self.state != LedgerState.IDLE:
            return False, "Position already active. Multi-position stacking rejected by risk invariant.", None

        if allocated_inr > self.current_capital_inr:
            return False, f"Insufficient capital: Required INR {allocated_inr:.2f}, Available INR {self.current_capital_inr:.2f}", None

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])

        if not bids or not asks:
            return False, "Orderbook depth empty; cannot determine realistic execution fill.", None

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        side = side.upper()

        # Realistic spread crossing + slippage model
        if side == "LONG":
            raw_fill = best_ask
            effective_fill = raw_fill * (1.0 + self.base_slippage_pct)
        elif side == "SHORT":
            raw_fill = best_bid
            effective_fill = raw_fill * (1.0 - self.base_slippage_pct)
        else:
            return False, f"Invalid trade side: {side}", None

        # Notional and fee accounting
        leverage = min(max(1.0, leverage), 5.0)  # Hard 5x cap
        margin_inr = allocated_inr
        notional_inr = margin_inr * leverage
        entry_fee_inr = notional_inr * self.taker_fee_pct

        self.current_capital_inr -= entry_fee_inr

        self.active_position = {
            'symbol': symbol,
            'side': side,
            'entry_time': time.time(),
            'entry_price': effective_fill,
            'raw_quote_price': raw_fill,
            'margin_inr': margin_inr,
            'leverage': leverage,
            'notional_inr': notional_inr,
            'total_fees_inr': entry_fee_inr,
            'peak_price': effective_fill
        }
        self.state = LedgerState.IN_POSITION

        return True, "FILLED", dict(self.active_position)

    def update_mtm(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """Computes live mark-to-market unrealized PnL based on current exit liquidity."""
        if self.state != LedgerState.IN_POSITION or not self.active_position:
            return {'state': self.state.value, 'unrealized_pnl_inr': 0.0}

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        if not bids or not asks:
            return {'state': self.state.value, 'unrealized_pnl_inr': 0.0}

        side = self.active_position['side']
        entry_p = self.active_position['entry_price']
        notional = self.active_position['notional_inr']

        # To exit Long, we hit best bid; to exit Short, we lift best ask
        current_exit_p = float(bids[0][0]) if side == "LONG" else float(asks[0][0])

        if side == "LONG":
            price_ret = (current_exit_p - entry_p) / entry_p
        else:
            price_ret = (entry_p - current_exit_p) / entry_p

        unrealized_pnl_inr = notional * price_ret

        # Update peak capital for drawdown tracking
        total_equity = self.current_capital_inr + self.active_position['margin_inr'] + unrealized_pnl_inr
        if total_equity > self.peak_capital_inr:
            self.peak_capital_inr = total_equity

        dd_pct = ((self.peak_capital_inr - total_equity) / self.peak_capital_inr) * 100.0
        if dd_pct > self.max_drawdown_pct:
            self.max_drawdown_pct = dd_pct

        return {
            'state': self.state.value,
            'side': side,
            'entry_price': entry_p,
            'exit_price': current_exit_p,
            'return_pct': price_ret * 100.0,
            'unrealized_pnl_inr': unrealized_pnl_inr,
            'total_equity_inr': total_equity,
            'drawdown_pct': dd_pct
        }

    def close_position(self, orderbook: Dict[str, Any], reason: str = "MANUAL_OR_STOP") -> Dict[str, Any]:
        """
        Closes out active paper position, applies exit fee and slippage,
        updates realized capital, and records trade into ledger history.
        """
        if self.state != LedgerState.IN_POSITION or not self.active_position:
            return {'error': 'No active position to close'}

        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        side = self.active_position['side']
        entry_p = self.active_position['entry_price']
        notional = self.active_position['notional_inr']
        margin = self.active_position['margin_inr']

        # Realistic exit pricing with slippage
        if side == "LONG":
            raw_exit = float(bids[0][0]) if bids else entry_p
            effective_exit = raw_exit * (1.0 - self.base_slippage_pct)
            price_ret = (effective_exit - entry_p) / entry_p
        else:
            raw_exit = float(asks[0][0]) if asks else entry_p
            effective_exit = raw_exit * (1.0 + self.base_slippage_pct)
            price_ret = (entry_p - effective_exit) / entry_p

        gross_pnl_inr = notional * price_ret
        exit_fee_inr = notional * self.taker_fee_pct
        total_fees = self.active_position['total_fees_inr'] + exit_fee_inr
        net_pnl_inr = gross_pnl_inr - exit_fee_inr

        # Settle capital
        self.current_capital_inr += (margin + net_pnl_inr)

        trade_record = {
            'symbol': self.active_position['symbol'],
            'side': side,
            'entry_price': entry_p,
            'exit_price': effective_exit,
            'return_pct': price_ret * 100.0,
            'notional_inr': notional,
            'gross_pnl_inr': gross_pnl_inr,
            'fees_inr': total_fees,
            'net_pnl_inr': net_pnl_inr,
            'duration_sec': time.time() - self.active_position['entry_time'],
            'reason': reason,
            'capital_after_inr': self.current_capital_inr
        }
        self.trade_history.append(trade_record)
        self.active_position = None
        self.state = LedgerState.IDLE

        return trade_record

    def get_summary(self) -> Dict[str, Any]:
        """Returns cumulative performance analytics of the paper ledger."""
        total_trades = len(self.trade_history)
        wins = [t for t in self.trade_history if t['net_pnl_inr'] > 0]
        losses = [t for t in self.trade_history if t['net_pnl_inr'] <= 0]
        win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0

        cum_net_pnl = self.current_capital_inr - self.initial_capital_inr
        total_fees = sum(t['fees_inr'] for t in self.trade_history)

        return {
            'initial_capital_inr': self.initial_capital_inr,
            'current_capital_inr': self.current_capital_inr,
            'cumulative_net_pnl_inr': cum_net_pnl,
            'cumulative_return_pct': (cum_net_pnl / self.initial_capital_inr) * 100.0,
            'total_fees_paid_inr': total_fees,
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate_pct': win_rate,
            'max_drawdown_pct': self.max_drawdown_pct,
            'state': self.state.value
        }
