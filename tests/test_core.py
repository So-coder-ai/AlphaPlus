import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.validators import (
    parse_side, parse_order_type, parse_quantity, parse_price, parse_symbol
)
from bot.orders import RiskManager
from ml.features import build_features, FEATURE_COLS


class TestParseSide:
    def test_valid_buy(self):
        assert parse_side("buy") == "BUY"

    def test_valid_sell(self):
        assert parse_side("  SELL  ") == "SELL"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid side"):
            parse_side("HOLD")


class TestParseOrderType:
    def test_market(self):
        assert parse_order_type("market") == "MARKET"

    def test_limit(self):
        assert parse_order_type("LIMIT") == "LIMIT"

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_order_type("STOP")


class TestParseQuantity:
    def test_valid_float(self):
        assert parse_quantity("0.01") == 0.01

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_quantity("0")

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            parse_quantity("-5")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_quantity("abc")


class TestParsePrice:
    def test_limit_requires_price(self):
        with pytest.raises(ValueError, match="required"):
            parse_price(None, "LIMIT")

    def test_limit_valid_price(self):
        assert parse_price("60000", "LIMIT") == 60000.0

    def test_market_returns_none(self):
        assert parse_price(None, "MARKET") is None

    def test_limit_zero_price_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            parse_price("0", "LIMIT")


class TestParseSymbol:
    def test_valid_symbol(self):
        assert parse_symbol("btcusdt") == "BTCUSDT"

    def test_short_symbol_raises(self):
        with pytest.raises(ValueError):
            parse_symbol("BTC")


class TestRiskManager:
    def test_quantity_within_limit(self):
        rm = RiskManager(max_quantity=5.0)
        rm.validate("BTCUSDT", 4.9)

    def test_quantity_exceeds_limit(self):
        rm = RiskManager(max_quantity=1.0)
        with pytest.raises(ValueError, match="risk limit"):
            rm.validate("BTCUSDT", 1.5)

    def test_symbol_not_allowed(self):
        rm = RiskManager(allowed_symbols=["BTCUSDT"])
        with pytest.raises(ValueError, match="not in allowed"):
            rm.validate("DOGEUSDT", 0.1)

    def test_symbol_allowed(self):
        rm = RiskManager(allowed_symbols=["BTCUSDT", "ETHUSDT"])
        rm.validate("ETHUSDT", 0.5)


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    prices = 30000 * np.exp(np.cumsum(np.random.normal(0, 0.005, n)))
    return pd.DataFrame({
        "open":   prices * np.random.uniform(0.998, 1.002, n),
        "high":   prices * np.random.uniform(1.001, 1.01, n),
        "low":    prices * np.random.uniform(0.99, 0.999, n),
        "close":  prices,
        "volume": np.random.lognormal(10, 0.5, n),
    }, index=dates)


class TestFeatureEngineering:
    def test_build_features_returns_dataframe(self, sample_ohlcv):
        df = build_features(sample_ohlcv)
        assert isinstance(df, pd.DataFrame)

    def test_all_feature_cols_present(self, sample_ohlcv):
        df = build_features(sample_ohlcv)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_target_column_present(self, sample_ohlcv):
        df = build_features(sample_ohlcv, include_target=True)
        assert "target" in df.columns

    def test_no_nan_after_drop(self, sample_ohlcv):
        df = build_features(sample_ohlcv, drop_na=True)
        assert df[FEATURE_COLS].isnull().sum().sum() == 0

    def test_target_is_binary(self, sample_ohlcv):
        df = build_features(sample_ohlcv, include_target=True)
        assert set(df["target"].unique()).issubset({0, 1})

    def test_rsi_in_valid_range(self, sample_ohlcv):
        df = build_features(sample_ohlcv)
        assert df["rsi"].between(0, 100).all()

    def test_bb_position_roughly_bounded(self, sample_ohlcv):
        df = build_features(sample_ohlcv)
        assert df["bb_position"].between(-0.5, 1.5).all()

    def test_minimum_rows_after_processing(self, sample_ohlcv):
        df = build_features(sample_ohlcv, drop_na=True)
        assert len(df) > 100, "Too many rows dropped during feature engineering"

