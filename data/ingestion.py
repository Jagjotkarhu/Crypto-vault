import ccxt
import pandas as pd

def fetch_live_market_data(pair: str = 'BTC/USDT', timeframe: str = '15m', limit: int = 100) -> pd.DataFrame:
    exchange = ccxt.binance({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
