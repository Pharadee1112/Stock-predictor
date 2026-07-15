from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import stock_analyzer
from stock_analyzer import StockAnalyzer

CLASSICAL_MODEL_TYPES = ["linear", "ridge", "polynomial", "svr", "randomforest", "gradientboosting", "mlp"]


def make_synthetic_data(n=80, seed=0, start_price=100.0, drift=0.4, noise_std=1.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = start_price + np.arange(n) * drift
    noise = rng.normal(0, noise_std, n)
    close = trend + noise
    volume = rng.integers(1000, 5000, n)
    return pd.DataFrame({"close": close, "volume": volume}, index=dates)


@pytest.fixture
def mocked_tv(monkeypatch):
    """Patch TvDatafeed so StockDataCollector never touches the network."""
    mock_tv_instance = MagicMock()
    mock_tv_class = MagicMock(return_value=mock_tv_instance)
    monkeypatch.setattr(stock_analyzer, "TvDatafeed", mock_tv_class)
    return mock_tv_instance


def make_analyzer(mocked_tv, n=80, **kwargs):
    synthetic = make_synthetic_data(n=n, **kwargs)
    mocked_tv.get_hist.return_value = synthetic
    analyzer = StockAnalyzer()
    analyzer.collect_data("AAPL", n_bars=n)
    return analyzer


def test_collect_data_uses_tvdatafeed(mocked_tv):
    synthetic = make_synthetic_data(n=60)
    mocked_tv.get_hist.return_value = synthetic

    analyzer = StockAnalyzer()
    data = analyzer.collect_data("AAPL", exchange="NASDAQ", n_bars=60)

    mocked_tv.get_hist.assert_called_once()
    assert data is synthetic
    assert analyzer.data is synthetic


def test_analyze_trend(mocked_tv):
    analyzer = make_analyzer(mocked_tv, n=10)
    trend_df = analyzer.analyze_trend()

    assert "Trend" in trend_df.columns
    assert set(trend_df["Trend"].unique()) <= {"Up", "Down", "Stable"}


@pytest.mark.parametrize("model_type", CLASSICAL_MODEL_TYPES)
def test_fit_classical_all_model_types(mocked_tv, model_type):
    analyzer = make_analyzer(mocked_tv, n=80)
    fit_result = analyzer.fit(model_type)

    assert fit_result["kind"] == "classical"
    for key in ("mae", "mse", "mape", "baseline_mae", "baseline_mse", "baseline_mape", "residual_std"):
        assert np.isfinite(fit_result[key])
        assert fit_result[key] >= 0
    assert isinstance(fit_result["better_than_baseline"], bool)
    assert len(fit_result["actual_prices"]) == len(fit_result["predicted_prices"])
    assert len(fit_result["actual_prices"]) > 0


def test_fit_lstm(mocked_tv):
    analyzer = make_analyzer(mocked_tv, n=80)
    fit_result = analyzer.fit("lstm")

    assert fit_result["kind"] == "lstm"
    assert fit_result["window"] > 0
    assert np.isfinite(fit_result["mae"])
    assert len(fit_result["actual_prices"]) == len(fit_result["predicted_prices"])


@pytest.mark.parametrize("model_type", ["linear", "randomforest", "lstm"])
@pytest.mark.parametrize("days_ahead", [1, 5, 15])
def test_rollout_different_horizons(mocked_tv, model_type, days_ahead):
    analyzer = make_analyzer(mocked_tv, n=80)
    fit_result = analyzer.fit(model_type)
    result = analyzer.rollout(fit_result, days_ahead)

    assert np.isfinite(result["predicted"])
    assert result["predicted"] > 0
    assert result["uncertainty_band"] >= 0
    if days_ahead > 7:
        assert result["warning"] is not None
        assert "7" in result["warning"]


def test_predict_future_close_end_to_end(mocked_tv):
    analyzer = make_analyzer(mocked_tv, n=80)
    result = analyzer.predict_future_close(5, model_type="linear")

    for key in ("predicted", "mae", "mse", "mape", "baseline_mae", "baseline_mse",
                "baseline_mape", "better_than_baseline", "uncertainty_band",
                "warning", "actual_prices", "predicted_prices"):
        assert key in result


def test_cached_fit_reused_across_rollouts(mocked_tv):
    """Same fit_result used for two different horizons: identical eval metrics
    (no retraining happened) but different predictions (rollout recomputed)."""
    analyzer = make_analyzer(mocked_tv, n=80)
    fit_result = analyzer.fit("linear")

    result_5 = analyzer.rollout(fit_result, 5)
    result_10 = analyzer.rollout(fit_result, 10)

    assert result_5["mae"] == result_10["mae"]
    assert result_5["predicted"] != result_10["predicted"]


def test_insufficient_data_raises_for_classical(mocked_tv):
    analyzer = make_analyzer(mocked_tv, n=5)
    with pytest.raises(ValueError):
        analyzer.fit("linear")


def test_insufficient_data_raises_for_lstm(mocked_tv):
    analyzer = make_analyzer(mocked_tv, n=5)
    with pytest.raises(ValueError):
        analyzer.fit("lstm")


def test_naive_baseline_is_yesterdays_close(mocked_tv):
    """With a noiseless linear trend, the 'tomorrow = today' baseline's error
    should equal exactly the daily drift, confirming it's computed correctly."""
    analyzer = make_analyzer(mocked_tv, n=80, noise_std=0.0)
    fit_result = analyzer.fit("linear")

    assert abs(fit_result["baseline_mae"] - 0.4) < 0.05
