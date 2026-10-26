import asyncio
import json
import threading
import time
from typing import Dict, Any, Optional, Tuple
import websockets
import pandas as pd


class BinanceWebSocketStreamer:
    """
    Persistent, low-latency WebSocket client for Binance public market data.
    Streams live top-20 L2 orderbook depth and real-time kline candles (<20ms latency).
    Operates in a background daemon thread with automatic reconnect logic.
    """

    def __init__(self, pair: str = 'BTC/USDT', timeframe: str = '1m'):
        self.raw_pair = pair
        # Binance stream names require lowercase without slash (e.g., btcusdt)
        self.symbol = pair.replace('/', '').lower()
        self.timeframe = timeframe
        self.ws_url = f"wss://stream.binance.com:9443/stream?streams={self.symbol}@depth20@100ms/{self.symbol}@kline_{self.timeframe}"

        # Thread-safe cached states
        self._lock = threading.Lock()
        self._latest_orderbook: Dict[str, Any] = {'bids': [], 'asks': [], 'timestamp': 0}
        self._latest_candle: Dict[str, Any] = {}
        self._candle_history: list = []
        self._is_running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self):
        """Starts the WebSocket streamer in a background daemon thread."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True, name="BinanceWSStreamer")
        self._thread.start()

    def stop(self):
        """Signals the WebSocket stream to shut down gracefully."""
        self._is_running = False
        if self._loop and self._loop.is_running():
            def _cancel_tasks():
                for t in asyncio.all_tasks(self._loop):
                    t.cancel()
            self._loop.call_soon_threadsafe(_cancel_tasks)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run_event_loop(self):
        """Runs the dedicated asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except (asyncio.CancelledError, RuntimeError):
            pass
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()
            except Exception:
                pass

    async def _connect_and_listen(self):
        backoff = 1.0
        while self._is_running:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self._connected = True
                    backoff = 1.0
                    while self._is_running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=1.5)
                            data = json.loads(message)
                            self._handle_stream_payload(data)
                        except asyncio.TimeoutError:
                            continue
            except asyncio.CancelledError:
                break
            except Exception:
                self._connected = False
                if not self._is_running:
                    break
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    break
                backoff = min(backoff * 2.0, 15.0)

    def _handle_stream_payload(self, payload: Dict[str, Any]):
        stream = payload.get('stream', '')
        data = payload.get('data', {})

        if 'depth20' in stream:
            # depth20 delivers bids and asks as [[price, qty], ...]
            bids = [[float(b[0]), float(b[1])] for b in data.get('bids', [])]
            asks = [[float(a[0]), float(a[1])] for a in data.get('asks', [])]
            with self._lock:
                self._latest_orderbook = {
                    'bids': bids,
                    'asks': asks,
                    'timestamp': time.time()
                }

        elif 'kline' in stream:
            k = data.get('k', {})
            candle = {
                'timestamp': k.get('t'),
                'open': float(k.get('o', 0.0)),
                'high': float(k.get('h', 0.0)),
                'low': float(k.get('l', 0.0)),
                'close': float(k.get('c', 0.0)),
                'volume': float(k.get('v', 0.0)),
                'is_closed': k.get('x', False)
            }
            with self._lock:
                self._latest_candle = candle
                if candle['is_closed']:
                    self._candle_history.append(candle)
                    if len(self._candle_history) > 500:
                        self._candle_history.pop(0)

    def get_orderbook(self) -> Dict[str, Any]:
        """Returns the latest top-20 order book snapshot."""
        with self._lock:
            return {
                'bids': list(self._latest_orderbook['bids']),
                'asks': list(self._latest_orderbook['asks']),
                'timestamp': self._latest_orderbook['timestamp']
            }

    def get_latest_price(self) -> Optional[float]:
        """Returns the current mid-price or best ask/bid."""
        with self._lock:
            ob = self._latest_orderbook
            if ob['bids'] and ob['asks']:
                return (ob['bids'][0][0] + ob['asks'][0][0]) / 2.0
            if self._latest_candle:
                return self._latest_candle['close']
            return None

    def get_market_state(self) -> Tuple[Optional[float], Dict[str, Any], bool]:
        """
        Returns (latest_price, orderbook_dict, is_connected).
        """
        with self._lock:
            price = None
            if self._latest_orderbook['bids'] and self._latest_orderbook['asks']:
                price = (self._latest_orderbook['bids'][0][0] + self._latest_orderbook['asks'][0][0]) / 2.0
            elif self._latest_candle:
                price = self._latest_candle['close']
            return price, dict(self._latest_orderbook), self._connected
