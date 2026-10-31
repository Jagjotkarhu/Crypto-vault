# Core Configuration for Crypto-Vault
import os

# Secrets come from the environment (.env is already git-ignored). Never hardcode keys.
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')

# Risk Boundaries
TOTAL_CAPITAL_INR = 5000.0
RISK_PER_TRADE_PCT = 0.015                      # 1R = 1.5% of EQUITY (INR 75 on INR 5,000)
RISK_BUDGET_INR = TOTAL_CAPITAL_INR * RISK_PER_TRADE_PCT

# Exchange minimums (verify against your venue's current MIN_NOTIONAL filter)
MIN_EXCHANGE_NOTIONAL_USD = 10.0
USD_INR_RATE = float(os.environ.get('USD_INR_RATE', '88.0'))
MIN_EXCHANGE_NOTIONAL_INR = MIN_EXCHANGE_NOTIONAL_USD * USD_INR_RATE

TRANCHE_MARGIN_INR = max(1000.0, MIN_EXCHANGE_NOTIONAL_INR * 1.15)  # ~INR 1,012
MAX_LEVERAGE = 5.0
MAX_POSITION_NOTIONAL_INR = TRANCHE_MARGIN_INR * MAX_LEVERAGE

DEFAULT_TIMEFRAME = '15m'
DEFAULT_PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
