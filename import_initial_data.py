"""Idempotent, transactional import of the three supplied input files."""
import csv
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
import uuid

import psycopg
from psycopg.types.json import Jsonb

from data_access import DatabaseConfig
from settings import ROOT, data_dir, environment

LOCK_ID = 81450928
HOLDINGS = [("fund", "CG001 Portfolio", "portfolio", "portfolio_holdings.csv"),
            ("r2500g_proxy", "R2500G Proxy", "benchmark", "benchmark_proxy_holdings.csv")]
PROXIES = {"market": "CSUS.L", "size": "SIZE", "value": "VLUE", "momentum": "MTUM", "quality": "QUAL"}


def number(value):
    if value is None or not str(value).strip():
        return None
    result = Decimal(str(value).replace(",", ""))
    if not result.is_finite():
        raise ValueError("Nonfinite input number")
    return result


def load_holdings(folder):
    result, instruments = {}, {}
    for portfolio_id, _, _, filename in HOLDINGS:
        with (folder / filename).open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        symbols = [row["Ticker"].strip().upper() for row in rows]
        weights = [number(row["Weight (%)"]) / 100 for row in rows]
        if len(set(symbols)) != len(symbols) or any(not x for x in symbols):
            raise ValueError(f"Duplicate/empty ticker in {filename}")
        if any(w < 0 or w > 1 for w in weights) or abs(sum(weights) - 1) > Decimal("1e-8"):
            raise ValueError(f"Weights must sum to 100% in {filename}")
        result[portfolio_id] = list(zip(rows, symbols, weights))
        for row, symbol in zip(rows, symbols):
            instruments.setdefault(symbol, (symbol, row.get("Security Name"),
                row.get("Sector") or row.get("sector"), row.get("Industry") or row.get("industry")))
    for symbol in PROXIES.values():
        instruments.setdefault(symbol, (symbol, None, None, None))
    return result, instruments


def run(folder=None, config=None):
    folder = folder or data_dir()
    files = [filename for _, _, _, filename in HOLDINGS] + 	["initial_daily_prices.csv"]
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files}
    holdings, instruments = load_holdings(folder)
    snapshot = environment().get("HOLDINGS_AS_OF", "").strip()
    as_of = date.fromisoformat(snapshot) if snapshot else None
    hashes["holdings_as_of"] = snapshot
    with psycopg.connect(**(config or DatabaseConfig.from_environment()).connection, autocommit=True) as conn:
        conn.execute((ROOT / "schema.sql").read_text(), prepare=False)
        conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_ID,))
        existing = conn.execute("SELECT stats FROM ops.pipeline_runs WHERE source='initial_csv' AND status='succeeded' ORDER BY finished_at DESC LIMIT 1").fetchone()
        if existing and existing[0].get("input_hashes") == hashes:
            return {**existing[0], "import_skipped": True}
        run_id = uuid.uuid4()
        conn.execute("INSERT INTO ops.pipeline_runs(run_id,source,status) VALUES (%s,'initial_csv','running')", (run_id,))
        try:
            with conn.transaction():
                conn.execute("""CREATE TEMP TABLE stage_prices(
                    ticker text NOT NULL, yahoo_ticker text, trade_date date NOT NULL,
                    close numeric NOT NULL, adj_close numeric, currency text NOT NULL,
                    source text NOT NULL, retrieved_at timestamptz NOT NULL,
                    UNIQUE(ticker,trade_date,source)
                ) ON COMMIT DROP""")
                price_symbols, count, adjusted_count = set(), 0, 0
                with (folder / "initial_daily_prices.csv").open(encoding="utf-8-sig", newline="") as handle, conn.cursor() as cursor:
                    with cursor.copy("COPY stage_prices FROM STDIN") as copy:
                        for row in csv.DictReader(handle):
                            symbol = row["ticker"].strip().upper()
                            close, adjusted = number(row["close"]), number(row["adj_close"])
                            if not symbol or close is None or close <= 0 or (adjusted is not None and adjusted <= 0):
                                raise ValueError("Invalid price")
                            retrieved = datetime.fromisoformat(row["retrieved_at"].replace("Z", "+00:00"))
                            if retrieved.tzinfo is None:
                                raise ValueError("Price retrieval time must have a timezone")
                            copy.write_row((symbol, row.get("yahoo_ticker") or symbol, date.fromisoformat(row["trade_date"]), close,
                                            adjusted, row["currency"], row["source"], retrieved))
                            price_symbols.add(symbol); count += 1; adjusted_count += adjusted is not None
                required = set(PROXIES.values()) | {s for _, s, _ in holdings["fund"]}
                if required - price_symbols:
                    raise ValueError("Missing fund/proxy prices: " + ", ".join(sorted(required - price_symbols)))
                if price_symbols - set(instruments):
                    raise ValueError("Unexpected symbols in price input")
                with conn.cursor() as cursor:
                    cursor.executemany("""INSERT INTO core.instruments(ticker,security_name,sector,industry,instrument_type)
                        VALUES (%s,%s,%s,%s,%s) ON CONFLICT(ticker) DO UPDATE SET security_name=EXCLUDED.security_name,
                        sector=EXCLUDED.sector,industry=EXCLUDED.industry,instrument_type=EXCLUDED.instrument_type""",
                        [(*values, "etf" if symbol in set(PROXIES.values()) | {"IWO", "IWP", "VTWG"} else "unknown")
                         for symbol, values in instruments.items()])
                ids = dict(conn.execute("SELECT ticker,instrument_id FROM core.instruments").fetchall())
                conn.execute("DELETE FROM core.price_universe")
                with conn.cursor() as cursor:
                    cursor.executemany("INSERT INTO core.price_universe VALUES (%s)", [(ids[s],) for s in sorted(price_symbols)])
                for portfolio_id, name, kind, filename in HOLDINGS:
                    conn.execute("""INSERT INTO core.portfolios VALUES (%s,%s,%s) ON CONFLICT(portfolio_id)
                        DO UPDATE SET name=EXCLUDED.name,portfolio_type=EXCLUDED.portfolio_type""", (portfolio_id, name, kind))
                    conn.execute("DELETE FROM core.holdings WHERE portfolio_id=%s", (portfolio_id,))
                    with conn.cursor() as cursor:
                        cursor.executemany("INSERT INTO core.holdings VALUES (%s,%s,%s,%s,%s,%s,%s)", [
                            (portfolio_id, ids[symbol], as_of, number(row["Shares"]), number(row["Market Value [USD]"]), weight, run_id)
                            for row, symbol, weight in holdings[portfolio_id]])
                with conn.cursor() as cursor:
                    cursor.executemany("""INSERT INTO core.factor_proxies VALUES (%s,%s) ON CONFLICT(factor_name)
                        DO UPDATE SET proxy_instrument_id=EXCLUDED.proxy_instrument_id""", [(key, ids[symbol]) for key, symbol in PROXIES.items()])
                conn.execute("""UPDATE core.instruments i SET yahoo_ticker=s.yahoo_ticker,currency=s.currency
                    FROM (SELECT DISTINCT ticker,yahoo_ticker,currency FROM stage_prices) s WHERE i.ticker=s.ticker""")
                conn.execute("""INSERT INTO core.daily_prices
                    SELECT i.instrument_id,s.trade_date,s.source,v.price_type,v.price,s.currency,s.retrieved_at,%s
                    FROM stage_prices s JOIN core.instruments i ON i.ticker=s.ticker
                    CROSS JOIN LATERAL (VALUES ('close',s.close),('adjusted_close',s.adj_close)) v(price_type,price)
                    WHERE v.price IS NOT NULL ON CONFLICT(instrument_id,trade_date,source,price_type) DO UPDATE
                    SET price=EXCLUDED.price,retrieved_at=EXCLUDED.retrieved_at,batch_id=EXCLUDED.batch_id""", (run_id,))
                conn.execute("""INSERT INTO core.preferred_price_sources
                    SELECT DISTINCT ON (p.instrument_id,p.price_type) p.instrument_id,p.price_type,p.source
                    FROM core.daily_prices p GROUP BY p.instrument_id,p.price_type,p.source
                    ORDER BY p.instrument_id,p.price_type,count(*) DESC,p.source
                    ON CONFLICT DO NOTHING""")
                coverage = conn.execute("""SELECT count(*),sum(h.weight) FROM core.holdings h
                    JOIN core.price_universe u USING(instrument_id) WHERE h.portfolio_id='r2500g_proxy'""").fetchone()
                stats = {"input_hashes": hashes, "instruments": len(instruments), "price_universe": len(price_symbols),
                         "fund_holdings": len(holdings['fund']), "raw_benchmark_holdings": len(holdings['r2500g_proxy']),
                         "priced_benchmark_holdings": coverage[0], "benchmark_coverage_pct": float(coverage[1] * 100),
                         "csv_price_rows": count, "initial_core_price_rows": count + adjusted_count}
                with conn.cursor() as cursor:
                    cursor.executemany("INSERT INTO raw.import_files VALUES (%s,%s,%s,%s)", [
                        (run_id, name, hashes[name], count if name == 'initial_daily_prices.csv' else len(holdings[HOLDINGS[files.index(name)][0]]))
                        for name in files])
                conn.execute("UPDATE ops.data_revision SET revision=revision+1,updated_at=now()")
                conn.execute("UPDATE ops.pipeline_runs SET status='succeeded',finished_at=now(),stats=%s WHERE run_id=%s", (Jsonb(stats), run_id))
            return stats
        except Exception as exc:
            conn.execute("UPDATE ops.pipeline_runs SET status='failed',finished_at=now(),error_summary=%s WHERE run_id=%s", (type(exc).__name__, run_id))
            raise


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
