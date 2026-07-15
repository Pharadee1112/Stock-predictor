# Stock Predictor

A small Flask web app that predicts a stock's future closing price using several
classic ML models (or an LSTM), pulling historical price data live from
TradingView via [tvDatafeed](https://github.com/rongardF/tvdatafeed).

## Why this project

The starting point was simple: given a stock's price history, can a handful of
common ML models say something useful about where the price is headed? The
first version did exactly that — one feature (a day counter), one model
(linear regression), fit on the whole dataset, no validation.

That's an easy trap in any ML project: numbers that look good because the
model is graded on data it already memorized. The project evolved through
four phases (tracked in [`IMPLEMENT.md`](IMPLEMENT.md)) specifically to close
that gap — so the error numbers and warnings shown on screen reflect what the
model would actually have gotten right on *unseen* data, not how well it fit
the past:

1. **Methodology** — chronological train/test split, out-of-sample MAE/MSE/MAPE,
   a naive "tomorrow = today" baseline to check the model is actually earning
   its keep, richer features (moving averages, volatility, volume) instead of
   just a day index, and an uncertainty band that widens the further out you
   forecast.
2. **Error handling & validation** — bad input (typos, malformed dates, huge
   date ranges, a network hiccup on the TradingView side) fails with a clear
   message instead of crashing the server.
3. **UX & performance** — a loading indicator, plots returned as base64 so the
   `static/` folder doesn't fill up with throwaway PNGs, a short-lived cache
   so re-running the same symbol/model doesn't retrain from scratch, and a
   selectable exchange instead of a hardcoded `NASDAQ`.
4. **Deploy prep** — a real WSGI server (`waitress`) for anything other than
   local dev, `.env`-driven config, and a pytest suite that mocks TradingView
   so tests don't depend on the network or market hours. (An optional LLM
   layer to explain results in plain language is scoped but not yet built.)

## How it works

1. You pick a symbol, exchange, a future date, a model, and how many days of
   history to train on.
2. The app pulls that history from TradingView.
3. It builds features (day index, 5/20-day moving averages, rolling
   volatility, volume), splits the data chronologically (80% train / 20%
   test — no shuffling, since shuffling time series data leaks the future
   into training), and fits the chosen model on the training slice.
4. Accuracy (MAE, MSE, MAPE) is measured only on the held-out test slice, and
   compared against a naive baseline. If the model doesn't beat the naive
   baseline, you'll see a warning.
5. To forecast your target date, it recursively rolls the model forward one
   day at a time (using its own prior predictions to build the next day's
   features), which is how it supports forecasting further ahead than a
   single step.

**Supported models:** `linear`, `ridge`, `polynomial`, `svr`, `randomforest`,
`gradientboosting`, `mlp`, `lstm`.

## Project structure

```
app.py                     Flask routes, request validation, caching, plotting
stock_analyzer.py          Data collection + model fitting/forecasting logic
config.py                  Loads .env and exposes typed settings
templates/index.html       Single-page UI
tests/test_stock_analyzer.py  pytest suite (TradingView calls are mocked)
IMPLEMENT.md               Phase-by-phase implementation checklist/log
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env          # then edit .env if you want non-default settings
```

## Running it

```bash
python app.py
```

Behavior depends on `DEBUG` in `.env`:

- `DEBUG=True` (default) — Flask's dev server at `http://127.0.0.1:5000`,
  with auto-reload and the debugger.
- `DEBUG=False` — serves the same app through `waitress`, a production-ready
  WSGI server. If you're deploying somewhere other than your own machine,
  also set `HOST=0.0.0.0` in `.env` so the server accepts external
  connections (the default `127.0.0.1` only accepts local traffic).

Then open the page in a browser, fill in the form, and click **Predict**:

- **Stock symbol** — e.g. `AAPL`
- **Exchange** — e.g. `NASDAQ`, `NYSE`, `HOSE`, `BINANCE` (defaults to `NASDAQ`)
- **Date to predict** — must be after the last available trading date
- **Model type** — pick from the dropdown
- **Number of data points** — how many past bars to train on (30–500)

The popup shows the predicted price, its uncertainty band, MAE/MSE/MAPE,
how it compares to the naive baseline, any warnings (e.g. long forecast
horizon), and a plot of the out-of-sample test predictions vs. actuals.

## Configuration (`.env`)

| Variable                       | Default   | Meaning                                              |
|---------------------------------|-----------|-------------------------------------------------------|
| `DEBUG`                         | `True`    | Flask dev server vs. waitress                         |
| `HOST`                          | `127.0.0.1` | Bind address                                         |
| `PORT`                          | `5000`    | Bind port                                              |
| `DEFAULT_EXCHANGE`               | `NASDAQ`  | Used when no exchange is given                         |
| `COLLECT_DATA_TIMEOUT_SECONDS`   | `15`      | Timeout for the TradingView data fetch                |
| `CACHE_TTL_SECONDS`              | `900`     | How long a fitted model is reused before retraining    |
| `MIN_DATA_POINTS` / `MAX_DATA_POINTS` | `30` / `500` | Allowed range for the data-points input       |

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The test suite mocks `TvDatafeed` entirely, so it runs offline and doesn't
depend on TradingView being reachable or the market being open.

## Known limitations

- Anonymous TradingView access (no login) can be rate-limited or return
  incomplete data.
- A naive baseline sometimes wins, especially over short, noisy horizons —
  that's expected and the app tells you when it happens rather than hiding it.
- Forecasts more than ~7 days out get considerably less reliable; the app
  flags this but the underlying models don't fundamentally get better at long
  horizons.
