import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import app as flask_app
from stock_analyzer import StockAnalyzer


@pytest.fixture
def client():
    flask_app.app.config['TESTING'] = True
    flask_app._fit_cache.clear()
    with flask_app.app.test_client() as c:
        yield c
    flask_app._fit_cache.clear()


def future_date_str(days=5):
    return (datetime.date.today() + datetime.timedelta(days=days)).strftime('%Y-%m-%d')


def post_analyze(client, **overrides):
    payload = {
        'symbol': 'AAPL',
        'exchange': 'NASDAQ',
        'date': future_date_str(),
        'model_type': 'linear',
        'data_points': '60',
    }
    payload.update(overrides)
    return client.post('/analyze', data=payload)


def test_live_fetch_failure_falls_back_to_demo_data_for_bundled_symbol(client):
    with patch.object(StockAnalyzer, 'collect_data', side_effect=RuntimeError('network down')):
        resp = post_analyze(client, symbol='AAPL')

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['demo_data'] is True
    assert 'AAPL' in body['demo_data_reason']
    assert body['predicted_close'] is not None


def test_live_fetch_failure_without_bundled_data_still_errors(client):
    with patch.object(StockAnalyzer, 'collect_data', side_effect=RuntimeError('network down')):
        resp = post_analyze(client, symbol='MSFT')

    assert resp.status_code == 502
    assert 'error' in resp.get_json()


def test_empty_live_data_also_falls_back_to_demo_data(client):
    def fake_collect_data(self, symbol, exchange='NASDAQ', n_bars=None, interval=None):
        self.data = pd.DataFrame(columns=['close', 'volume'])
        return self.data

    with patch.object(StockAnalyzer, 'collect_data', fake_collect_data):
        resp = post_analyze(client, symbol='BTCUSD', exchange='BITSTAMP')

    assert resp.status_code == 200
    assert resp.get_json()['demo_data'] is True


def test_successful_live_fetch_does_not_use_demo_data(client):
    dates = pd.date_range('2024-01-01', periods=80, freq='D')
    fake_df = pd.DataFrame(
        {'close': np.linspace(100, 130, 80), 'volume': np.arange(80)},
        index=dates,
    )

    def fake_collect_data(self, symbol, exchange='NASDAQ', n_bars=None, interval=None):
        self.data = fake_df
        return self.data

    with patch.object(StockAnalyzer, 'collect_data', fake_collect_data):
        resp = post_analyze(client, symbol='AAPL')

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['demo_data'] is False
    assert body['demo_data_reason'] is None


def test_lstm_returns_400_when_unavailable(client, monkeypatch):
    monkeypatch.setattr(flask_app, 'LSTM_AVAILABLE', False)

    resp = post_analyze(client, model_type='lstm')

    assert resp.status_code == 400
    assert 'tensorflow' in resp.get_json()['error'].lower()
