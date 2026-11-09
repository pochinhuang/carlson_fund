"""Read Carlson Fund's existing PostgreSQL schema; never download or mutate data.

Prices have a sorted DatetimeIndex named ``trade_date`` and caller-ordered ticker
columns. Date bounds are inclusive. Holdings retain the display columns used by
the Streamlit pages; ``Weight (%)`` is in percent, not a decimal fraction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import psycopg
from dotenv import dotenv_values
from psycopg import IsolationLevel


class DataAccessError(RuntimeError):
    """A safe, user-facing database/configuration error."""


class DataQualityError(DataAccessError):
    """Required financial data is absent or invalid."""


@dataclass(frozen=True)
class DatabaseConfig:
    # Hide connection credentials from repr, diagnostics, and page output.
    connection: dict = field(repr=False)

    @classmethod
    def from_environment(cls, env_file=None, environ: Mapping | None = None):
        env = dict(os.environ if environ is None else environ)
        explicit = env_file or env.get("CARLSON_DB_ENV")
        candidates = [Path(explicit).expanduser()] if explicit else [
            Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"
        ]
        values = {}
        for path in candidates:
            if path.is_file():
                values = dict(dotenv_values(path, encoding="utf-8-sig"))
                break
        else:
            if explicit:
                raise DataAccessError("CARLSON_DB_ENV / --env-file does not point to an existing .env file.")
        values.update({k: v for k, v in env.items() if v is not None})
        settings = {"connect_timeout": 10, "application_name": "carlson_fund_app",
                    "options": "-c statement_timeout=30000"}
        if values.get("DATABASE_URL"):
            settings["conninfo"] = values["DATABASE_URL"]
        else:
            try:
                port = int(values.get("PGPORT") or values.get("POSTGRES_PORT") or "5432")
                if not 1 <= port <= 65535:
                    raise ValueError
            except (ValueError, TypeError) as exc:
                raise DataAccessError("The PostgreSQL port must be an integer from 1 to 65535.") from exc
            settings.update(
                host=values.get("PGHOST") or values.get("POSTGRES_HOST") or "127.0.0.1",
                port=port,
                dbname=values.get("PGDATABASE") or values.get("POSTGRES_DB") or "carlson_fund",
                user=values.get("PGUSER") or values.get("POSTGRES_USER") or "carlson",
            )
            password = values.get("PGPASSWORD") or values.get("POSTGRES_PASSWORD")
            if password:
                settings["password"] = password
        if values.get("PGSSLMODE"):
            settings["sslmode"] = values["PGSSLMODE"]
        return cls(settings)


def _tickers(values):
    if isinstance(values, str):
        values = [values]
    if values is None:
        raise ValueError("Supply the required tickers explicitly.")
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Tickers must be nonempty strings.")
        ticker = value.strip().upper()
        if ticker not in result:
            result.append(ticker)
    return result


def _date(value):
    if value is None:
        return None
    if not isinstance(value, (str, date, pd.Timestamp)):
        raise ValueError("Dates must be YYYY-MM-DD strings or date objects.")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Invalid date.")
    return stamp.date()


@dataclass
class AnalyticsData:
    holdings: pd.DataFrame
    benchmark_holdings: pd.DataFrame
    adjusted_prices: pd.DataFrame
    close_prices: pd.DataFrame
    proxies: dict

    @property
    def portfolio_prices(self):
        return self.adjusted_prices.reindex(columns=self.holdings["Ticker"].tolist()).dropna(how="all")

    @property
    def benchmark_prices(self):
        return self.adjusted_prices.reindex(columns=self.benchmark_holdings["Ticker"].tolist()).dropna(how="all")

    @property
    def factor_prices(self):
        return self.adjusted_prices.reindex(columns=list(self.proxies.values())).dropna(how="all")


class DataAccess:
    def __init__(self, config=None, *, connection=None):
        self.config = config
        self._connection = connection

    @contextmanager
    def session(self):
        """One consistent, read-only snapshot; borrowed test connections stay owned by the caller."""
        if self._connection is not None:
            yield self
            return
        config = self.config or DatabaseConfig.from_environment()
        try:
            with psycopg.connect(**config.connection) as connection:
                connection.read_only = True
                connection.isolation_level = IsolationLevel.REPEATABLE_READ
                yield DataAccess(config, connection=connection)
        except psycopg.Error as exc:
            raise DataAccessError(
                "PostgreSQL connection/query failed. Check that Docker is running, the .env path/port "
                "is correct, and the initial import has completed."
            ) from exc

    def _query(self, sql, params=()):
        if self._connection is None:
            with self.session() as reader:
                return reader._query(sql, params)
        with self._connection.cursor() as cursor:
            cursor.execute(sql, params)
            columns = [col.name for col in cursor.description]
            return pd.DataFrame(cursor.fetchall(), columns=columns)

    def get_holdings(self, portfolio_id="fund", *, priced_only=True, normalize=True):
        result = self._query('''
            SELECT i.ticker AS "Ticker", i.security_name AS "Security Name",
                   h.shares::double precision AS "Shares",
                   (h.market_value / nullif(h.shares, 0))::double precision AS "Price [USD]",
                   h.market_value::double precision AS "Market Value [USD]",
                   (100*h.weight)::double precision AS "Weight (%%)",
                   coalesce(i.sector,'Unknown') AS "Sector", coalesce(i.industry,'Unknown') AS "Industry",
                   h.as_of_date, EXISTS(SELECT 1 FROM core.price_universe u WHERE u.instrument_id=h.instrument_id) AS priced
            FROM core.holdings h JOIN core.instruments i USING(instrument_id)
            WHERE h.portfolio_id=%s ORDER BY i.ticker
        ''', (portfolio_id,))
        if result.empty:
            raise DataQualityError(f"No holdings for {portfolio_id!r}.")
        raw_count = len(result)
        raw_total = float(result["Weight (%)"].sum())
        excluded = result.loc[~result.priced, "Ticker"].tolist()
        coverage = float(result.loc[result.priced, "Weight (%)"].sum())
        if priced_only:
            if portfolio_id == "fund" and excluded:
                raise DataQualityError("Fund holdings without prices: " + ", ".join(excluded))
            result = result.loc[result.priced].copy()
            if result.empty:
                raise DataQualityError("No priced holdings remain.")
            if normalize and portfolio_id == "r2500g_proxy":
                result["Weight (%)"] *= 100 / result["Weight (%)"].sum()
        result = result.drop(columns="priced").reset_index(drop=True)
        weights = result["Weight (%)"].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or result.Ticker.duplicated().any():
            raise DataQualityError("Invalid holdings.")
        if (normalize or not priced_only) and not np.isclose(weights.sum(),100,atol=1e-6,rtol=0):
            raise DataQualityError("Holdings weights do not sum to 100%.")
        result.attrs.update(raw_count=raw_count, raw_weight_sum_pct=raw_total,
                            price_coverage_pct=coverage, excluded_tickers=excluded)
        return result

    def get_data_revision(self):
        return int(self._query("SELECT revision FROM ops.data_revision WHERE singleton").iloc[0,0])

    def get_price_universe(self):
        return self._query('''SELECT i.ticker FROM core.price_universe u
            JOIN core.instruments i USING(instrument_id) ORDER BY i.ticker''')["ticker"].tolist()

    def get_factor_proxies(self):
        rows = self._query('''
            SELECT f.factor_name, i.ticker FROM core.factor_proxies f
            JOIN core.instruments i ON i.instrument_id = f.proxy_instrument_id
        ''')
        mapping = dict(zip(rows["factor_name"], rows["ticker"]))
        order = ["market", "size", "value", "momentum", "quality"]
        if set(mapping) != set(order) or len(set(mapping.values())) != 5:
            raise DataQualityError("All five distinct market/style proxy mappings are required.")
        return {key: mapping[key] for key in order}

    def get_prices(self, tickers, start=None, end=None, *, price_type="adjusted_close",
                   source=None, strict=True):
        """Read one complete preferred provider series per ticker, with inclusive dates.

        The nightly pipeline changes a preferred source only after checking
        complete coverage of the previously displayed history. No adjusted level
        splicing or Close-to-adjusted fallback is performed.
        """
        symbols = _tickers(tickers)
        if price_type not in ("close", "adjusted_close"):
            raise ValueError("price_type must be 'close' or 'adjusted_close'.")
        start, end = _date(start), _date(end)
        if start and end and start > end:
            raise ValueError("Start date must be on or before end date.")
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="trade_date"), columns=symbols, dtype=float)
        if not symbols:
            return empty
        conditions = ["i.ticker = ANY(%s::text[])", "p.price_type = %s"]
        params = [symbols, price_type]
        if start:
            conditions.append("p.trade_date >= %s")
            params.append(start)
        if end:
            conditions.append("p.trade_date <= %s")
            params.append(end)
        if source is not None:
            conditions.append("p.source = %s")
            params.append(source)
        preferred_join = ""
        if source is None:
            preferred_join = """ JOIN core.preferred_price_sources ps
                ON ps.instrument_id=p.instrument_id AND ps.price_type=p.price_type AND ps.source=p.source """
        rows = self._query('''
            SELECT i.ticker,p.trade_date,p.source,p.price::double precision AS price,p.currency
            FROM core.daily_prices p JOIN core.instruments i USING(instrument_id)
        ''' + preferred_join + " WHERE " + " AND ".join(conditions) + " ORDER BY p.trade_date,i.ticker", params)
        present = set(rows["ticker"])
        missing = [ticker for ticker in symbols if ticker not in present]
        if missing and strict:
            raise DataQualityError("No " + price_type + " history in this window for: " + ", ".join(missing))
        if rows.empty:
            return empty
        values = rows["price"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise DataQualityError("Prices must be finite and strictly positive.")
        if rows.duplicated(["ticker", "trade_date"]).any():
            raise DataQualityError("Duplicate ticker/date after source selection.")
        if (rows.groupby("ticker")["currency"].nunique() > 1).any():
            raise DataQualityError("An instrument changes currency within the requested window.")
        rows["trade_date"] = pd.to_datetime(rows["trade_date"]).astype("datetime64[ns]")
        result = rows.pivot(index="trade_date", columns="ticker", values="price").reindex(columns=symbols)
        result.columns.name = None
        result = result.sort_index().astype(float)
        result.attrs["sources"] = rows.groupby("ticker")["source"].first().to_dict()
        result.attrs["currencies"] = rows.groupby("ticker")["currency"].first().to_dict()
        result.attrs["price_type"] = price_type
        return result

    def get_history_bounds(self, tickers, *, price_type="adjusted_close", source=None):
        prices = self.get_prices(tickers, price_type=price_type, source=source)
        complete = prices.dropna(how="any")
        if complete.empty:
            raise DataQualityError("No dates have prices for every required ticker.")
        return complete.index[0].date(), complete.index[-1].date()

    def load_analytics_data(self, *, include_benchmark=True, start=None, end=None):
        with self.session() as reader:
            holdings = reader.get_holdings("fund")
            benchmark = reader.get_holdings("r2500g_proxy") if include_benchmark else holdings.iloc[:0].copy()
            proxies = reader.get_factor_proxies()
            symbols = sorted(set(holdings["Ticker"]) | set(benchmark["Ticker"]) | set(proxies.values()))
            adjusted = reader.get_prices(symbols, start, end)
            close = reader.get_prices(holdings["Ticker"], start, end, price_type="close")
        return AnalyticsData(holdings, benchmark, adjusted, close, proxies)


def describe_data(data):
    return {
        "portfolio_holdings": len(data.holdings), "benchmark_holdings": len(data.benchmark_holdings),
        "price_tickers": len(data.adjusted_prices.columns),
        "price_dates": len(data.adjusted_prices),
        "first_date": str(data.adjusted_prices.index.min().date()),
        "last_date": str(data.adjusted_prices.index.max().date()),
        "fund_weight_sum_pct": float(data.holdings["Weight (%)"].sum()),
        "benchmark_weight_sum_pct": float(data.benchmark_holdings["Weight (%)"].sum()),
        "factor_proxies": data.proxies,
        "source_counts": pd.Series(data.adjusted_prices.attrs.get("sources", {})).value_counts().to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Path to the existing PostgreSQL bootstrap .env")
    parser.add_argument("--check", action="store_true", help="Validate the existing database without changing it")
    args = parser.parse_args()
    try:
        reader = DataAccess(DatabaseConfig.from_environment(args.env_file))
        data = reader.load_analytics_data()
        print(json.dumps(describe_data(data), indent=2))
        print("\nLatest portfolio adjusted prices:")
        print(data.portfolio_prices.tail(3).iloc[:, :6].to_string())
    except (DataAccessError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
