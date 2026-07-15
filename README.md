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

`lstm` needs `tensorflow`, which is a heavy dependency. It's imported lazily —
if tensorflow isn't installed, the app still starts and every other model
still works; the LSTM option is just hidden from the UI (and returns a clear
400 if requested directly). See [Why three requirements files](#why-three-requirements-files).

## Demo data / offline fallback

Live data comes from TradingView with an anonymous (no-login) session, which
can be rate-limited, slow, or unreachable — especially on a free-tier host.
For `AAPL` and `BTCUSD`, the app has bundled historical data under `data/`
(committed to the repo, refreshed via `scripts/generate_demo_data.py`) as a
fallback: if the live fetch fails, times out, or comes back empty for one of
those two symbols, the app transparently serves the bundled data instead of
erroring out.

This is never silent — the API response includes `"demo_data": true` and a
`"demo_data_reason"` explaining why, and the UI shows an orange **DEMO DATA**
badge with that reason next to the result. Any other symbol just gets the
normal error if the live fetch fails.

The homepage form is pre-filled with a working example (`AAPL` / `NASDAQ` /
`Linear`) so a first-time visitor can click **Predict** immediately.

## Project structure

```
app.py                     Flask routes, request validation, caching, plotting
stock_analyzer.py          Data collection + model fitting/forecasting logic
demo_data.py                Bundled fallback data loader
data/                       Bundled CSV data for AAPL, BTCUSD
scripts/generate_demo_data.py  Refreshes the bundled CSVs from live data
config.py                  Loads .env and exposes typed settings
templates/index.html       Single-page UI
tests/                      pytest suite (TradingView calls are mocked)
IMPLEMENT.md               Phase-by-phase implementation checklist/log
Procfile                    Production start command (waitress)
runtime.txt / .python-version  Pinned Python version for deploy platforms
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt        # full install, includes LSTM support
# or: pip install -r requirements-slim.txt   # smaller/lower-memory, no LSTM

copy .env.example .env          # then edit .env if you want non-default settings
```

### Why three requirements files

Every package here is version-pinned (`flask==3.1.2`, not just `flask`), so
`pip install -r requirements.txt` gives anyone — you next month, a teammate,
a deploy platform — the exact same versions this was built and tested
against, instead of silently picking up a newer release that changed
behavior.

There are three variants because "install everything, everywhere" isn't
actually what you want in every situation:

| File | Contents | Use it when |
|---|---|---|
| `requirements.txt` | Everything, including `tensorflow`/`keras` | Local dev, or deploying somewhere with enough RAM for LSTM support |
| `requirements-slim.txt` | Same as above, minus `tensorflow`/`keras` | Deploying to a low-memory free tier |
| `requirements-dev.txt` | `requirements.txt` + `pytest` | Only when you're running the test suite |

- **`-slim` exists because `tensorflow` is heavy** — hundreds of MB and a lot
  of RAM at import time, for one model out of eight (`lstm`). Free hosting
  tiers (e.g. 512MB RAM) can fail to even start the app if tensorflow is in
  the mix. Since the app imports tensorflow lazily (see above), dropping it
  from `requirements-slim.txt` doesn't break anything — LSTM just becomes
  unavailable and hides itself from the UI.
- **`-dev` exists because `pytest` has no reason to be on a production
  server** — it's a tool for the person writing the code, not something the
  running app needs. Keeping it out of `requirements.txt` means a real
  deploy doesn't install test tooling it'll never use.

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

## Deploying

Built and tested against Render/Railway-style platforms (a `Procfile`,
env-var-driven `PORT`, no filesystem writes needed at runtime), but there's
nothing platform-specific — it'll run anywhere that can `pip install` and
run a WSGI app.

1. **Pick a requirements file.** Free/low-memory tiers: use
   `requirements-slim.txt` (see [Why three requirements files](#why-three-requirements-files)
   above). Otherwise use `requirements.txt` for LSTM support too.
2. **Set environment variables** on the platform (not in a committed `.env`
   — `.env` is git-ignored on purpose):
   - `DEBUG=False` — required; runs the app through `waitress` instead of
     Flask's dev server.
   - `HOST=0.0.0.0` — required for the server to accept external traffic.
   - `PORT` — most platforms (Render, Railway, Heroku) inject this
     automatically; `config.py` and the `Procfile` both already read it.
3. **Start command.** The included `Procfile` covers this:
   ```
   web: waitress-serve --host=0.0.0.0 --port=$PORT app:app
   ```
   Platforms that read `Procfile` (Heroku-style buildpacks, some Render/Railway
   setups) will pick this up automatically. If your platform wants an explicit
   start command instead, use the same line.
4. **Python version** is pinned via `runtime.txt` / `.python-version`
   (currently 3.13.0) so the build matches what this was developed against.
5. **No persistent disk needed** — plots are returned as base64 in the JSON
   response (nothing written to `static/`), and the in-memory model cache is
   fine to lose on restart; it just means the next request retrains.
6. If outbound network access to TradingView is flaky or blocked on your
   host, the [demo data fallback](#demo-data--offline-fallback) means `AAPL`
   and `BTCUSD` still work and are clearly labeled as demo data — useful for
   a public demo link that needs to work reliably.

## Known limitations

- Anonymous TradingView access (no login) can be rate-limited or return
  incomplete data.
- A naive baseline sometimes wins, especially over short, noisy horizons —
  that's expected and the app tells you when it happens rather than hiding it.
- Forecasts more than ~7 days out get considerably less reliable; the app
  flags this but the underlying models don't fundamentally get better at long
  horizons.
- Bundled demo data (`AAPL`, `BTCUSD`) is a static snapshot from whenever
  `scripts/generate_demo_data.py` was last run — it's clearly labeled as demo
  data when served, but it will drift from the real market over time.
