"""Historical market-data clients only; this module has no trading operations."""
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
import time as clock
from zoneinfo import ZoneInfo

import pandas as pd
import requests


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Price:
    ticker: str
    trade_date: object
    price_type: str
    price: Decimal
    source: str
    currency: str = "USD"


def positive(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ProviderError("Provider returned an invalid numeric price.") from None
    if not result.is_finite() or result <= 0:
        raise ProviderError("Provider returned a nonpositive/nonfinite price.")
    return result


class AlpacaProvider:
    def __init__(self, key, secret, feed="sip", *, session=None, sleep=clock.sleep, aliases=None):
        if not key or not secret:
            raise ProviderError("Fill ALPACA_API_KEY and ALPACA_SECRET_KEY in .env first.")
        if feed not in {"sip", "iex"}:
            raise ProviderError("ALPACA_FEED must be sip or iex.")
        self.source = "alpaca_" + feed
        self.feed, self.sleep = feed, sleep
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.aliases = aliases or {}

    def _json(self, params):
        for attempt in range(5):
            try:
                response = self.session.get("https://data.alpaca.markets/v2/stocks/bars", params=params, timeout=(10, 90))
            except requests.RequestException:
                if attempt == 4:
                    raise ProviderError("Alpaca network request failed after retries.") from None
                self.sleep(min(2 ** attempt, 30)); continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == 4:
                    raise ProviderError(f"Alpaca HTTP {response.status_code} after retries.")
                try:
                    delay = float(response.headers.get("Retry-After", 2 ** attempt))
                except (ValueError, TypeError):
                    delay = 2 ** attempt
                self.sleep(max(0, min(delay, 60))); continue
            if response.status_code != 200:
                # Never echo authentication headers or arbitrary error bodies.
                raise ProviderError(f"Alpaca HTTP {response.status_code}; check credentials/feed access.")
            try:
                return response.json(parse_float=Decimal)
            except (ValueError, TypeError):
                raise ProviderError("Alpaca returned invalid JSON.") from None
        raise ProviderError("Alpaca request failed.")

    def fetch(self, symbols, start, end, price_type):
        external = {self.aliases.get(symbol, symbol): symbol for symbol in symbols}
        if len(external) != len(symbols):
            raise ProviderError("Symbol aliases must be one-to-one.")
        timezone = ZoneInfo("America/New_York")
        params = {
            "symbols": ",".join(external), "timeframe": "1Day", "limit": 10000,
            "start": datetime.combine(start, time.min, timezone).isoformat(),
            "end": datetime.combine(end, time.max, timezone).isoformat(),
            "adjustment": "raw" if price_type == "close" else "all",
            "feed": self.feed, "sort": "asc",
        }
        seen_tokens = set()
        while True:
            self.sleep(0.35)  # Stay below the Basic plan's request budget.
            payload = self._json(params)
            bars = payload.get("bars") or {}
            if not isinstance(bars, dict):
                raise ProviderError("Alpaca bars must be grouped by symbol.")
            for external_symbol, values in bars.items():
                if external_symbol not in external:
                    raise ProviderError("Unexpected symbol in Alpaca response.")
                for bar in values or []:
                    stamp = pd.Timestamp(bar["t"])
                    if stamp.tzinfo is None:
                        raise ProviderError("Alpaca bar timestamp has no timezone.")
                    day = stamp.tz_convert("America/New_York").date()
                    if start <= day <= end:
                        yield Price(external[external_symbol], day, price_type, positive(bar["c"]), self.source)
            token = payload.get("next_page_token")
            if not token:
                break
            if token in seen_tokens:
                raise ProviderError("Repeated Alpaca pagination token.")
            seen_tokens.add(token)
            params["page_token"] = token


class YahooProvider:
    source = "yfinance"

    def __init__(self, ticker_factory=None, sleep=clock.sleep):
        if ticker_factory is None:
            import yfinance as yf
            ticker_factory = yf.Ticker
        self.ticker_factory, self.sleep = ticker_factory, sleep

    def fetch(self, symbols, start, end, price_type):
        if list(symbols) != ["CSUS.L"]:
            raise ProviderError("Yahoo is used only for CSUS.L in this pipeline.")
        frame = None
        for attempt in range(3):
            try:
                frame = self.ticker_factory("CSUS.L").history(
                    start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                    interval="1d", auto_adjust=False, actions=False, raise_errors=True,
                )
                break
            except Exception as exc:
                if attempt == 2:
                    raise ProviderError("Yahoo CSUS.L request failed (" + type(exc).__name__ + ").") from None
                self.sleep(2 ** attempt)
        column = "Close" if price_type == "close" else "Adj Close"
        if frame is None or frame.empty:
            return
        if column not in frame:
            raise ProviderError("Yahoo response has no " + column + "; no fallback is permitted.")
        for stamp, value in frame[column].items():
            day = pd.Timestamp(stamp).date()  # yfinance history index is exchange-local.
            if start <= day <= end and not pd.isna(value):
                yield Price("CSUS.L", day, price_type, positive(value), self.source)
