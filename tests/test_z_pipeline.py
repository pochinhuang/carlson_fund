from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pandas_market_calendars as calendars
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from data_access import DataAccess, DatabaseConfig
from import_initial_data import LOCK_ID
from market_providers import Price
from update_daily_prices import run

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not os.environ.get("CARLSON_TEST_DATABASE_URL"), reason="Disposable imported DB required")
NOW = datetime(2026,11,8,12,tzinfo=timezone.utc)
TARGET = date(2026,11,6)


@pytest.fixture
def isolated_database(monkeypatch):
    params = conninfo_to_dict(os.environ["CARLSON_TEST_DATABASE_URL"])
    template = params["dbname"]
    name = "pipeline_" + uuid.uuid4().hex
    admin_params = {**params, "dbname": "template1"}
    with psycopg.connect(**admin_params, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(sql.Identifier(name), sql.Identifier(template)))
    params["dbname"] = name
    url = make_conninfo(**params)
    monkeypatch.setenv("DATABASE_URL", url)
    try:
        yield DatabaseConfig({"conninfo": url, "connect_timeout": 5})
    finally:
        with psycopg.connect(**admin_params, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


class FakeProvider:
    """Deterministic provider replay using the actual supplied input histories."""
    def __init__(self, table, market=False, missing_adjusted=False):
        self.source = "test_yfinance" if market else "test_alpaca"
        self.table, self.market, self.missing_adjusted = table, market, missing_adjusted
        self.groups = {symbol: group.sort_values("trade_date") for symbol,group in table.groupby("ticker")}
        self.calendar = calendars.get_calendar("LSE" if market else "NYSE")
        self.cached_dates = {}
    def fetch(self, symbols, start, end, kind):
        for symbol in symbols:
            group = self.groups[symbol]
            column = "close" if kind == "close" else "adj_close"
            scale = Decimal("0.7") if kind == "adjusted_close" else Decimal(1)
            for idx,row in enumerate(group.itertuples(index=False)):
                if self.missing_adjusted and symbol == "ADUS" and kind == "adjusted_close" and idx == 10:
                    continue
                yield Price(symbol,row.trade_date.date(),kind,Decimal(str(getattr(row,column)))*scale,self.source)
            last = group.iloc[-1]
            calendar_key = (last.trade_date.date(), end)
            if calendar_key not in self.cached_dates:
                self.cached_dates[calendar_key] = self.calendar.valid_days(last.trade_date.date()+timedelta(days=1),end)
            days = self.cached_dates[calendar_key]
            for step, stamp in enumerate(days, start=1):
                value = Decimal(str(last[column]))*scale*Decimal("1.01")**step
                yield Price(symbol,stamp.date(),kind,value,self.source)


@pytest.fixture(scope="module")
def table():
    return pd.read_csv(ROOT / "data/initial_daily_prices.csv",parse_dates=["trade_date"])


def small_universe(config):
    with psycopg.connect(**config.connection) as conn:
        conn.execute("DELETE FROM core.price_universe WHERE instrument_id NOT IN (SELECT instrument_id FROM core.instruments WHERE ticker IN ('ADUS','CSUS.L'))")


def test_complete_update_repeat_and_open_dashboard_refresh(isolated_database, table):
    from streamlit.testing.v1 import AppTest
    db = DataAccess(isolated_database)
    before_revision = db.get_data_revision()
    app = AppTest.from_file(str(ROOT / "portfolio_dashboard.py")).run(timeout=90)
    assert not app.exception and not app.error
    before_chart = app.get("plotly_chart")[0].proto.spec
    providers = (FakeProvider(table), FakeProvider(table,market=True))
    result = run(TARGET,config=isolated_database,providers=providers,now=NOW,write_report=False)
    assert result["status"] == "succeeded"
    assert result["updated_tickers"] == 1028
    assert result["accepted_series"] == 2056
    assert db.get_data_revision() == before_revision+1
    prices = db.get_prices(["ADUS"])
    assert prices.index[-1] == pd.Timestamp(TARGET)
    # Entire adjusted history moves to one coherent vintage, avoiding a false
    # -30% boundary return when providers use different adjusted price levels.
    assert np.isclose(prices.iloc[-1,0]/prices.iloc[-2,0]-1, 0.01)
    assert prices.attrs["sources"]["ADUS"] == "test_alpaca"
    with psycopg.connect(**isolated_database.connection) as conn:
        count = conn.execute("SELECT count(*) FROM core.daily_prices").fetchone()[0]
    repeated = run(TARGET,config=isolated_database,providers=providers,now=NOW,write_report=False)
    assert repeated["status"] == "succeeded"
    with psycopg.connect(**isolated_database.connection) as conn:
        assert conn.execute("SELECT count(*) FROM core.daily_prices").fetchone()[0] == count
    app.run(timeout=90)
    assert not app.exception and not app.error
    assert app.session_state["_carlson_data_revision"] == db.get_data_revision()
    assert app.get("plotly_chart")[0].proto.spec != before_chart


def test_incomplete_adjusted_series_never_overwrites_published_history(isolated_database, table):
    small_universe(isolated_database)
    db = DataAccess(isolated_database)
    before = db.get_prices(["ADUS"])
    result = run(TARGET,config=isolated_database,providers=(FakeProvider(table,missing_adjusted=True),FakeProvider(table,market=True)),now=NOW,write_report=False)
    assert result["status"] == "partial"
    assert result["incomplete_tickers"] == ["ADUS"]
    after = db.get_prices(["ADUS"])
    pd.testing.assert_frame_equal(before,after)
    assert after.attrs["sources"]["ADUS"] == "yahoo_finance"
    assert db.get_prices(["ADUS"],price_type="close").index[-1] == pd.Timestamp(TARGET)


def test_failed_publish_rolls_back_prices_preferences_and_revision(isolated_database, table):
    small_universe(isolated_database)
    db = DataAccess(isolated_database)
    before_revision = db.get_data_revision()
    with psycopg.connect(**isolated_database.connection) as conn:
        count = conn.execute("SELECT count(*) FROM core.daily_prices").fetchone()[0]
        conn.execute("""CREATE FUNCTION reject_test_prices() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN IF NEW.source='test_alpaca' THEN RAISE EXCEPTION 'intentional test failure'; END IF; RETURN NEW; END $$""")
        conn.execute("CREATE TRIGGER reject_test_prices BEFORE INSERT ON core.daily_prices FOR EACH ROW EXECUTE FUNCTION reject_test_prices()")
    with pytest.raises(psycopg.errors.RaiseException):
        run(TARGET,config=isolated_database,providers=(FakeProvider(table),FakeProvider(table,market=True)),now=NOW,write_report=False)
    with psycopg.connect(**isolated_database.connection) as conn:
        assert conn.execute("SELECT count(*) FROM core.daily_prices").fetchone()[0] == count
        assert conn.execute("SELECT status FROM ops.pipeline_runs WHERE source='nightly_prices' ORDER BY finished_at DESC LIMIT 1").fetchone()[0] == "failed"
        assert conn.execute("SELECT count(*) FROM core.preferred_price_sources WHERE source LIKE 'test_%'").fetchone()[0] == 0
    assert db.get_data_revision() == before_revision


def test_concurrent_update_is_skipped(isolated_database, table):
    with psycopg.connect(**isolated_database.connection,autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)",(LOCK_ID,))
        result = run(TARGET,config=isolated_database,providers=(FakeProvider(table),FakeProvider(table,market=True)),now=NOW,write_report=False)
        assert result["status"] == "skipped"
