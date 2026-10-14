import ccxt
import pandas as pd

def fetch_live_market_data(pair: str = 'BTC/USDT', timeframe: str = '15m', limit: int = 250) -> tuple:
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # 1. Fetch OHLCV Candles
    ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 2. Fetch Order Book Depth (Top 20 bids & asks for Liquidity Imbalance)
    orderbook = exchange.fetch_order_book(pair, limit=20)
    bid_vol = sum([b[1] for b in orderbook['bids'][:20]])
    ask_vol = sum([a[1] for a in orderbook['asks'][:20]])
    total_vol = bid_vol + ask_vol
    
    # Order Book Imbalance (OBI): Ranges from -1.0 (heavy sell wall) to +1.0 (heavy buy support)
    obi = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0
    bid_ratio = (bid_vol / total_vol) * 100.0 if total_vol > 0 else 50.0
    
    liquidity_data = {
        'bid_volume': bid_vol,
        'ask_volume': ask_vol,
        'obi': obi,
        'bid_ratio': bid_ratio,
        'best_bid': orderbook['bids'][0][0] if orderbook['bids'] else 0.0,
        'best_ask': orderbook['asks'][0][0] if orderbook['asks'] else 0.0,
        'raw_orderbook': orderbook
    }
    
    return df, liquidity_data
