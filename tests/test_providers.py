from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from market_providers import AlpacaProvider, YahooProvider, ProviderError
from scheduler import build_trigger, credentials_present
from update_daily_prices import target_date, last_session


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}
    def json(self, **kwargs):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = iter(responses), [], {}
    def get(self, url, **kwargs):
        self.calls.append((url, {**kwargs, "params": kwargs["params"].copy()}))
        return next(self.responses)


def test_alpaca_follows_all_pages_and_uses_exchange_date():
    session = Session([
        Response({"bars": {"A": [{"t": "2026-11-05T05:00:00Z", "c": 100}]}, "next_page_token": "next"}),
        Response({"bars": {"B": [{"t": "2026-11-06T00:00:00Z", "c": 50}]}, "next_page_token": None}),
    ])
    client = AlpacaProvider("dummy-key", "dummy-secret", session=session, sleep=lambda _: None)
    rows = list(client.fetch(["A", "B"], date(2026,11,5), date(2026,11,5), "close"))
    assert [row.ticker for row in rows] == ["A", "B"]
    assert all(row.trade_date == date(2026,11,5) for row in rows)
    assert session.calls[1][1]["params"]["page_token"] == "next"
    assert session.calls[0][1]["params"]["adjustment"] == "raw"
    assert session.calls[0][1]["params"]["feed"] == "sip"
    assert "dummy-secret" not in str(session.calls)
    assert session.headers["APCA-API-SECRET-KEY"] == "dummy-secret"


def test_alpaca_retry_and_adjusted_request():
    session = Session([Response({},429,{"Retry-After":"0"}),Response({},503),Response({"bars":{}})])
    sleeps = []
    client = AlpacaProvider("dummy", "dummy", session=session, sleep=sleeps.append)
    assert list(client.fetch(["A"], date(2026,11,5), date(2026,11,5), "adjusted_close")) == []
    assert len(session.calls) == 3
    assert session.calls[0][1]["params"]["adjustment"] == "all"


def test_alpaca_authentication_error_and_pagination_loop():
    client = AlpacaProvider("dummy-key", "secret-should-not-appear", session=Session([Response({},403)]), sleep=lambda _:None)
    with pytest.raises(ProviderError, match="HTTP 403") as error:
        list(client.fetch(["A"], date(2026,11,5), date(2026,11,5), "close"))
    assert "secret-should-not-appear" not in str(error.value)
    client = AlpacaProvider("dummy", "dummy", session=Session([
        Response({"bars":{},"next_page_token":"repeat"}),Response({"bars":{},"next_page_token":"repeat"})]), sleep=lambda _:None)
    with pytest.raises(ProviderError, match="pagination"):
        list(client.fetch(["A"], date(2026,11,5), date(2026,11,5), "close"))


def test_yahoo_uses_unadjusted_close_and_exclusive_end():
    calls = []
    class Ticker:
        def __init__(self, ticker):
            assert ticker == "CSUS.L"
        def history(self, **kwargs):
            calls.append(kwargs)
            return pd.DataFrame({"Close":[100],"Adj Close":[95]},index=pd.DatetimeIndex(["2026-11-05"],tz="Europe/London"))
    client = YahooProvider(Ticker)
    raw = list(client.fetch(["CSUS.L"],date(2026,11,5),date(2026,11,5),"close"))
    adjusted = list(client.fetch(["CSUS.L"],date(2026,11,5),date(2026,11,5),"adjusted_close"))
    assert raw[0].price == Decimal(100) and adjusted[0].price == Decimal(95)
    assert calls[0]["end"] == "2026-11-06" and calls[0]["auto_adjust"] is False


def test_missing_yahoo_adjusted_column_never_uses_close():
    class Ticker:
        def __init__(self, _):pass
        def history(self, **kwargs):return pd.DataFrame({"Close":[10]},index=pd.DatetimeIndex(["2026-11-05"]))
    with pytest.raises(ProviderError,match="no Adj Close"):
        list(YahooProvider(Ticker).fetch(["CSUS.L"],date(2026,11,5),date(2026,11,5),"adjusted_close"))


def test_market_holidays_yesterday_and_schedule_timezone():
    assert target_date(datetime(2026,11,7,2,tzinfo=timezone.utc)) == date(2026,11,5)
    assert last_session(date(2026,11,7),"NYSE") == date(2026,11,6)
    assert last_session(date(2026,12,25),"NYSE") == date(2026,12,24)
    trigger = build_trigger({})
    fire = trigger.get_next_fire_time(None, datetime(2026,11,6,20,tzinfo=timezone.utc))
    assert fire.hour == 22 and str(fire.tzinfo) == "America/Chicago"
    assert credentials_present({"ALPACA_API_KEY":"x","ALPACA_SECRET_KEY":"y"})
    assert not credentials_present({"ALPACA_API_KEY":"x"})
