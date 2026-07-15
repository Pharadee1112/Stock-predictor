from flask import Flask, render_template, request, jsonify
from stock_analyzer import StockAnalyzer
import config
import datetime
import re
import time
import threading
import base64
import io
import concurrent.futures
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)

ALLOWED_MODEL_TYPES = {
    'linear', 'ridge', 'polynomial', 'svr',
    'randomforest', 'gradientboosting', 'mlp', 'lstm'
}
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.\-]{1,15}$')
EXCHANGE_PATTERN = re.compile(r'^[A-Z0-9_\-]{1,20}$')
DEFAULT_EXCHANGE = config.DEFAULT_EXCHANGE
MIN_DATA_POINTS = config.MIN_DATA_POINTS
MAX_DATA_POINTS = config.MAX_DATA_POINTS
COLLECT_DATA_TIMEOUT_SECONDS = config.COLLECT_DATA_TIMEOUT_SECONDS
CACHE_TTL_SECONDS = config.CACHE_TTL_SECONDS

# Cache of (analyzer, fit_result) keyed by (symbol, exchange, data_points, model_type),
# so repeat requests that only differ by target date don't retrain the model.
_fit_cache = {}
_fit_cache_lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')


def _collect_data_with_timeout(analyzer, symbol, exchange, data_points, timeout=COLLECT_DATA_TIMEOUT_SECONDS):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(analyzer.collect_data, symbol, exchange, data_points)
    try:
        return future.result(timeout=timeout)
    finally:
        executor.shutdown(wait=False)


def _get_cached_fit(cache_key):
    with _fit_cache_lock:
        cached = _fit_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
            return cached[1], cached[2]
    return None, None


def _store_cached_fit(cache_key, analyzer, fit_result):
    with _fit_cache_lock:
        _fit_cache[cache_key] = (time.time(), analyzer, fit_result)


def _plot_to_base64(actual_prices, pred_prices, symbol, model_type):
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label='Actual Prices', linewidth=2)
    plt.plot(pred_prices, label='Model Predictions', linewidth=2)
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.title(f'{symbol} - Actual vs Predicted ({model_type.upper()})')
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


@app.route('/analyze', methods=['POST'])
def analyze():
    symbol = request.form.get('symbol', '').strip().upper()
    exchange = request.form.get('exchange', '').strip().upper() or DEFAULT_EXCHANGE
    date_input = request.form.get('date', '').strip()
    model_type = request.form.get('model_type', '').strip().lower()
    data_points_raw = request.form.get('data_points', '').strip()

    if not SYMBOL_PATTERN.match(symbol):
        return jsonify({"error": "Invalid symbol. Use letters/numbers only (e.g. AAPL)."}), 400

    if not EXCHANGE_PATTERN.match(exchange):
        return jsonify({"error": "Invalid exchange. Use letters/numbers only (e.g. NASDAQ)."}), 400

    try:
        future_date = datetime.datetime.strptime(date_input, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date. Use the format YYYY-MM-DD."}), 400

    if model_type not in ALLOWED_MODEL_TYPES:
        return jsonify({
            "error": f"Invalid model_type. Choose one of: {', '.join(sorted(ALLOWED_MODEL_TYPES))}."
        }), 400

    try:
        data_points = int(data_points_raw)
    except ValueError:
        return jsonify({"error": "data_points must be a whole number."}), 400
    if not (MIN_DATA_POINTS <= data_points <= MAX_DATA_POINTS):
        return jsonify({
            "error": f"data_points must be between {MIN_DATA_POINTS} and {MAX_DATA_POINTS}."
        }), 400

    cache_key = (symbol, exchange, data_points, model_type)
    analyzer, fit_result = _get_cached_fit(cache_key)

    if analyzer is None:
        analyzer = StockAnalyzer()
        try:
            data = _collect_data_with_timeout(analyzer, symbol, exchange, data_points)
        except concurrent.futures.TimeoutError:
            return jsonify({"error": f"Timed out fetching data for {symbol}. Please try again."}), 502
        except Exception as e:
            return jsonify({"error": f"Failed to fetch data for {symbol}: {e}"}), 502

        if data is None or data.empty:
            return jsonify({
                "error": f"No data found for symbol '{symbol}' on exchange '{exchange}'. Check the symbol/exchange and try again."
            }), 502

        try:
            fit_result = analyzer.fit(model_type)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Model fitting failed: {e}"}), 500

        _store_cached_fit(cache_key, analyzer, fit_result)

    last_date = analyzer.data.index[-1]
    if future_date <= last_date:
        return jsonify({
            "error": f"Date must be after the last available trading date ({last_date.strftime('%Y-%m-%d')})."
        }), 400
    days_ahead = (future_date - last_date).days

    try:
        result = analyzer.rollout(fit_result, days_ahead)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    predicted = result['predicted']
    mae, mse, mape = result['mae'], result['mse'], result['mape']
    baseline_mae, baseline_mse, baseline_mape = (
        result['baseline_mae'], result['baseline_mse'], result['baseline_mape']
    )
    better_than_baseline = result['better_than_baseline']
    uncertainty_band = result['uncertainty_band']
    warning = result['warning']
    actual_prices, pred_prices = result['actual_prices'], result['predicted_prices']

    plot_data_uri = _plot_to_base64(actual_prices, pred_prices, symbol, model_type)

    return jsonify({
        "symbol": symbol,
        "exchange": exchange,
        "predicted_close": round(predicted, 2),
        "last_date": last_date.strftime('%d-%m-%Y'),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "mape": round(mape, 4),
        "baseline_mae": round(baseline_mae, 4),
        "baseline_mse": round(baseline_mse, 4),
        "baseline_mape": round(baseline_mape, 4),
        "better_than_baseline": better_than_baseline,
        "uncertainty_band": round(uncertainty_band, 4),
        "warning": warning,
        "plot": plot_data_uri
    })


if __name__ == '__main__':
    if config.DEBUG:
        app.run(host=config.HOST, port=config.PORT, debug=True)
    else:
        from waitress import serve
        serve(app, host=config.HOST, port=config.PORT)
