import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_access import DataAccess, DatabaseConfig, DataQualityError
from import_initial_data import run as import_data

pytestmark = pytest.mark.skipif(not os.environ.get("CARLSON_TEST_DATABASE_URL"), reason="Dedicated imported DB required")
ROOT = Path(__file__).resolve().parents[1]


def test_initial_import_counts_idempotency_and_raw_holdings():
    stats = import_data()
    assert stats["import_skipped"]
    assert stats["instruments"] == 1133
    assert stats["csv_price_rows"] == 281569
    db = DataAccess()
    raw = db.get_holdings("r2500g_proxy", priced_only=False)
    priced = db.get_holdings("r2500g_proxy")
    assert len(raw) == 1123 and len(priced) == 1018
    assert np.isclose(raw["Weight (%)"].sum(), 100)
    assert np.isclose(priced["Weight (%)"].sum(), 100)
    assert np.isclose(priced.attrs["price_coverage_pct"], 94.332)
    assert len(priced.attrs["excluded_tickers"]) == 105
    assert len(db.get_price_universe()) == 1028
    assert len(db.get_holdings()) == 45
    assert int(db._query("SELECT count(*) FROM mart.v_current_prices").iloc[0,0]) == 563138


def test_full_imported_prices_and_inclusive_dates():
    csv = pd.read_csv(ROOT / "data/initial_daily_prices.csv", parse_dates=["trade_date"])
    db = DataAccess()
    symbols = sorted(csv.ticker.unique())
    for kind, column in [("close", "close"), ("adjusted_close", "adj_close")]:
        expected = csv.pivot(index="trade_date", columns="ticker", values=column)
        expected.index = expected.index.as_unit("ns")
        expected.columns.name = None
        actual = db.get_prices(symbols, price_type=kind, source="yahoo_finance")
        assert_frame_equal(actual, expected, rtol=1e-13, atol=1e-12, check_freq=False, check_dtype=False)
    prices = db.get_prices(["ADUS", "SIZE"], "2026-11-03", "2026-11-05")
    assert prices.index[0] == pd.Timestamp("2026-11-03")
    assert prices.index[-1] == pd.Timestamp("2026-11-05")
    assert list(prices.columns) == ["ADUS", "SIZE"]
    with pytest.raises(DataQualityError):
        db.get_prices(["ADUS"], "2024-01-01", "2024-12-31")


def test_readonly_snapshot_and_parameter_binding():
    db = DataAccess()
    with db.session() as reader:
        assert reader._query("SHOW transaction_read_only").iloc[0,0] == "on"
        assert reader._query("SHOW transaction_isolation").iloc[0,0] == "repeatable read"
    assert db.get_prices(["x'); DROP TABLE core.holdings; --"], strict=False).empty
    assert len(db.get_holdings()) == 45


def test_windows_env_port_and_hidden_password(tmp_path):
    env = tmp_path / "test.env"
    env.write_text("POSTGRES_PORT=25432\nPOSTGRES_PASSWORD=dummy-local-test\n")
    config = DatabaseConfig.from_environment(env, environ={})
    assert config.connection["port"] == 25432
    assert "dummy-local-test" not in repr(config)
