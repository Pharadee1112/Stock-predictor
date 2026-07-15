"""Bundled fallback price data, used when the live TradingView fetch fails,
times out, or comes back empty. Keeps the app usable (e.g. for a public demo
on a flaky free-tier network) instead of just erroring out.

Refresh with scripts/generate_demo_data.py.
"""
import os
import threading

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

DEMO_SYMBOLS = {
    'AAPL': os.path.join(DATA_DIR, 'AAPL.csv'),
    'BTCUSD': os.path.join(DATA_DIR, 'BTCUSD.csv'),
}

_cache = {}
_cache_lock = threading.Lock()


def is_demo_symbol(symbol):
    return symbol.upper() in DEMO_SYMBOLS


def load_demo_data(symbol, n_bars=None):
    """Returns a (close, volume) DataFrame indexed by datetime, shaped like
    tvDatafeed's get_hist output, or None if there's no bundled data for
    this symbol."""
    symbol = symbol.upper()
    path = DEMO_SYMBOLS.get(symbol)
    if path is None:
        return None

    with _cache_lock:
        df = _cache.get(symbol)
        if df is None:
            df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
            _cache[symbol] = df

    if n_bars is not None and n_bars < len(df):
        df = df.iloc[-n_bars:]
    return df.copy()
