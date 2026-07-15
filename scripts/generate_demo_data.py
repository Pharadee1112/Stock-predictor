"""Refresh the bundled demo/fallback datasets under data/.

Run this occasionally (e.g. every few months) to keep the demo data from
getting too stale. Requires a working TradingView connection - it's a
one-off maintenance script, not something the app runs at request time.

Usage: python scripts/generate_demo_data.py
"""
import os

from tvDatafeed import TvDatafeed, Interval

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# (filename, symbol, exchange)
DEMO_SOURCES = [
    ('AAPL.csv', 'AAPL', 'NASDAQ'),
    ('BTCUSD.csv', 'BTCUSD', 'BITSTAMP'),
]
N_BARS = 500


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    tv = TvDatafeed()

    for filename, symbol, exchange in DEMO_SOURCES:
        df = tv.get_hist(symbol=symbol, exchange=exchange, n_bars=N_BARS, interval=Interval.in_daily)
        if df is None or df.empty:
            print(f"skipped {symbol} ({exchange}): no data returned")
            continue

        df = df[['close', 'volume']]
        path = os.path.join(DATA_DIR, filename)
        df.to_csv(path, index_label='datetime')
        print(f"wrote {path} ({len(df)} rows, {df.index[0].date()} to {df.index[-1].date()})")


if __name__ == '__main__':
    main()
