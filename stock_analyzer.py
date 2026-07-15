from tvDatafeed import TvDatafeed, Interval
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLUMNS = ['day', 'MA5', 'MA20', 'volatility', 'volume']
TRAIN_RATIO = 0.8
LSTM_WINDOW = 20
LONG_HORIZON_WARNING_DAYS = 7


class StockDataCollector:
    def __init__(self):
        self.tv = TvDatafeed()
        self.data = None

    def collect_data(self, symbol, exchange='NASDAQ', n_bars=None, interval=Interval.in_daily):
        self.data = self.tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            n_bars=n_bars,
            interval=interval
        )
        return self.data


class StockAnalyzer(StockDataCollector):
    def analyze_trend(self):
        self.data['Trend'] = self.data['close'].diff().apply(
            lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'Stable')
        )
        return self.data[['close', 'Trend']]

    def fit(self, model_type):
        """Train/evaluate a model on the currently loaded data. Expensive; safe to cache
        and reuse across requests that only differ by target date (days_ahead)."""
        model_type = model_type.lower()
        if model_type == 'lstm':
            return self._fit_lstm()
        return self._fit_classical(model_type)

    def rollout(self, fit_result, days_ahead):
        """Cheap: turn an already-fitted model into a forecast for a specific horizon."""
        if fit_result['kind'] == 'lstm':
            predicted = self._rollout_lstm(fit_result, days_ahead)
        else:
            predicted = self._recursive_forecast(fit_result['model'], days_ahead, poly=fit_result['poly'])

        uncertainty_band = fit_result['residual_std'] * np.sqrt(max(days_ahead, 1))

        return {
            'predicted': predicted,
            'mae': fit_result['mae'],
            'mse': fit_result['mse'],
            'mape': fit_result['mape'],
            'baseline_mae': fit_result['baseline_mae'],
            'baseline_mse': fit_result['baseline_mse'],
            'baseline_mape': fit_result['baseline_mape'],
            'better_than_baseline': fit_result['better_than_baseline'],
            'uncertainty_band': uncertainty_band,
            'warning': self._build_warning(days_ahead, fit_result['better_than_baseline']),
            'actual_prices': fit_result['actual_prices'],
            'predicted_prices': fit_result['predicted_prices'],
        }

    def predict_future_close(self, days_ahead: int, model_type='linear'):
        return self.rollout(self.fit(model_type), days_ahead)

    # ---------- shared helpers ----------

    @staticmethod
    def _metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
        return float(mae), float(mse), mape

    @staticmethod
    def _chronological_split(df, train_ratio=TRAIN_RATIO):
        split_idx = max(1, int(len(df) * train_ratio))
        split_idx = min(split_idx, len(df) - 1)
        return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)

    @staticmethod
    def _build_warning(days_ahead, better_than_baseline):
        parts = []
        if days_ahead > LONG_HORIZON_WARNING_DAYS:
            parts.append(
                f"ทำนายล่วงหน้า {days_ahead} วัน (เกิน {LONG_HORIZON_WARNING_DAYS} วัน) "
                "ความแม่นยำจะลดลงมาก ควรใช้ผลลัพธ์อย่างระมัดระวัง"
            )
        if not better_than_baseline:
            parts.append(
                "โมเดลนี้แม่นยำไม่ดีไปกว่าการทำนายแบบพื้นฐาน (พรุ่งนี้ = ราคาวันนี้) "
                "ควรพิจารณาผลลัพธ์อย่างระมัดระวังหรือเลือกโมเดลอื่น"
            )
        return " ".join(parts) if parts else None

    # ---------- classical (non-LSTM) models ----------

    def _build_feature_frame(self):
        df = self.data.reset_index()
        df['day'] = np.arange(len(df))

        ma_short = min(5, max(2, len(df) // 6))
        ma_long = min(20, max(ma_short + 1, len(df) // 3))
        df['MA5'] = df['close'].rolling(window=ma_short).mean()
        df['MA20'] = df['close'].rolling(window=ma_long).mean()
        df['volatility'] = df['close'].rolling(window=ma_short).std()
        if 'volume' not in df.columns:
            df['volume'] = 0.0

        df['next_close'] = df['close'].shift(-1)
        return df.dropna().reset_index(drop=True)

    @staticmethod
    def _fit_model(model_type, X_train, y_train):
        poly = None
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge()
        elif model_type == 'polynomial':
            poly = PolynomialFeatures(degree=2)
            X_train = poly.fit_transform(X_train)
            model = LinearRegression()
        elif model_type == 'svr':
            model = SVR(kernel='rbf')
        elif model_type == 'randomforest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradientboosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'mlp':
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)
        return model, poly

    @staticmethod
    def _transform(X, poly):
        return poly.transform(X) if poly is not None else X

    def _recursive_forecast(self, model, days_ahead, poly=None):
        closes = list(self.data['close'].values.astype(float))
        last_volume = float(self.data['volume'].values[-1]) if 'volume' in self.data.columns else 0.0

        ma_short = min(5, max(2, len(closes) // 6))
        ma_long = min(20, max(ma_short + 1, len(closes) // 3))
        start_day = len(closes) - 1

        predicted = closes[-1]
        for step in range(days_ahead):
            ma5 = float(np.mean(closes[-ma_short:]))
            ma20 = float(np.mean(closes[-ma_long:]))
            volatility = float(np.std(closes[-ma_short:]))
            day_idx = start_day + step + 1

            feat = self._transform(np.array([[day_idx, ma5, ma20, volatility, last_volume]]), poly)
            predicted = float(model.predict(feat)[0])
            closes.append(predicted)

        return predicted

    def _fit_classical(self, model_type):
        df = self._build_feature_frame()
        if len(df) < 4:
            raise ValueError("Not enough data points after feature engineering to train/test split. Increase data_points.")

        train_df, test_df = self._chronological_split(df)

        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df['next_close'].values
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df['next_close'].values
        naive_test = test_df['close'].values

        eval_model, eval_poly = self._fit_model(model_type, X_train, y_train)
        pred_test = eval_model.predict(self._transform(X_test, eval_poly))

        mae, mse, mape = self._metrics(y_test, pred_test)
        baseline_mae, baseline_mse, baseline_mape = self._metrics(y_test, naive_test)
        better_than_baseline = mae < baseline_mae

        final_model, final_poly = self._fit_model(model_type, df[FEATURE_COLUMNS].values, df['next_close'].values)
        residual_std = float(np.std(y_test - pred_test)) if len(y_test) > 1 else 0.0

        return {
            'kind': 'classical',
            'model_type': model_type,
            'model': final_model,
            'poly': final_poly,
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'baseline_mae': baseline_mae,
            'baseline_mse': baseline_mse,
            'baseline_mape': baseline_mape,
            'better_than_baseline': bool(better_than_baseline),
            'residual_std': residual_std,
            'actual_prices': y_test,
            'predicted_prices': pred_test,
        }

    # ---------- LSTM ----------

    @staticmethod
    def _lstm_windows(series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i + window])
            y.append(series[i + window])
        return np.array(X), np.array(y)

    def _fit_lstm(self, window=LSTM_WINDOW):
        closes = self.data['close'].values.astype(float)
        window = max(2, min(window, len(closes) - 5))
        if len(closes) - window < 4:
            raise ValueError("Not enough data points to build LSTM sliding windows. Increase data_points.")

        scaler = MinMaxScaler()
        scaled_closes = scaler.fit_transform(closes.reshape(-1, 1)).flatten()

        X_seq, y_seq = self._lstm_windows(scaled_closes, window)
        split_idx = max(1, int(len(X_seq) * TRAIN_RATIO))
        split_idx = min(split_idx, len(X_seq) - 1)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        def make_model():
            m = Sequential([LSTM(50, input_shape=(window, 1)), Dense(1)])
            m.compile(optimizer='adam', loss='mse')
            return m

        eval_model = make_model()
        eval_model.fit(X_train.reshape(-1, window, 1), y_train, epochs=50, verbose=0)
        pred_test_scaled = eval_model.predict(X_test.reshape(-1, window, 1), verbose=0).flatten()

        pred_test = scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        naive_test = scaler.inverse_transform(X_test[:, -1].reshape(-1, 1)).flatten()

        mae, mse, mape = self._metrics(y_test_actual, pred_test)
        baseline_mae, baseline_mse, baseline_mape = self._metrics(y_test_actual, naive_test)
        better_than_baseline = mae < baseline_mae

        final_model = make_model()
        final_model.fit(X_seq.reshape(-1, window, 1), y_seq, epochs=50, verbose=0)

        residual_std = float(np.std(y_test_actual - pred_test)) if len(y_test_actual) > 1 else 0.0

        return {
            'kind': 'lstm',
            'model_type': 'lstm',
            'model': final_model,
            'scaler': scaler,
            'window': window,
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'baseline_mae': baseline_mae,
            'baseline_mse': baseline_mse,
            'baseline_mape': baseline_mape,
            'better_than_baseline': bool(better_than_baseline),
            'residual_std': residual_std,
            'actual_prices': y_test_actual,
            'predicted_prices': pred_test,
        }

    def _rollout_lstm(self, fit_result, days_ahead):
        model = fit_result['model']
        scaler = fit_result['scaler']
        window = fit_result['window']

        closes = self.data['close'].values.astype(float)
        scaled_closes = scaler.transform(closes.reshape(-1, 1)).flatten()

        rolling_window = list(scaled_closes[-window:])
        for _ in range(days_ahead):
            seq = np.array(rolling_window[-window:]).reshape(1, window, 1)
            next_scaled = float(model.predict(seq, verbose=0)[0][0])
            rolling_window.append(next_scaled)

        return float(scaler.inverse_transform([[rolling_window[-1]]])[0][0])
