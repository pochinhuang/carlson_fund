"""Refresh yesterday's closes and coherent provider histories, then publish atomically."""
import argparse
import csv
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import uuid
from zoneinfo import ZoneInfo

import pandas_market_calendars as calendars
import psycopg
from psycopg.types.json import Jsonb

from data_access import DatabaseConfig
from import_initial_data import LOCK_ID
from market_providers import AlpacaProvider, YahooProvider, ProviderError, positive
from settings import data_dir, environment, reports_dir


def target_date(now=None, timezone_name="America/Chicago"):
    now = now or datetime.now(timezone.utc)
    return now.astimezone(ZoneInfo(timezone_name)).date() - timedelta(days=1)


def last_session(day, calendar):
    days = calendars.get_calendar(calendar).valid_days(day - timedelta(days=30), day)
    if len(days) == 0:
        raise ValueError("No market sessions in the last 30 days.")
    return days[-1].date()


def default_providers():
    env = environment()
    aliases_path = data_dir() / "symbol_aliases.json"
    aliases = json.loads(aliases_path.read_text()) if aliases_path.exists() else {}
    return (AlpacaProvider(env.get("ALPACA_API_KEY") or env.get("APCA_API_KEY_ID"),
                           env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY"),
                           env.get("ALPACA_FEED") or "sip", aliases=aliases), YahooProvider())


def run(day=None, *, config=None, providers=None, now=None, write_report=True):
    env = environment()
    day = day or target_date(now, env.get("UPDATE_TIMEZONE") or "America/Chicago")
    if isinstance(day, str):
        day = date.fromisoformat(day)
    today = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo(env.get("UPDATE_TIMEZONE") or "America/Chicago")).date()
    if day >= today:
        raise ValueError("Request yesterday or an earlier date, not an unfinished current session.")
    alpaca, yahoo = providers or default_providers()
    config = config or DatabaseConfig.from_environment()
    with psycopg.connect(**config.connection, autocommit=True) as conn:
        if not conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_ID,)).fetchone()[0]:
            return {"status": "skipped", "reason": "Another import/update is running."}
        conn.execute("""UPDATE ops.pipeline_runs SET status='failed',finished_at=now(),error_summary='Interrupted worker; retry is safe'
            WHERE source='nightly_prices' AND status='running'""")
        run_id = uuid.uuid4()
        conn.execute("INSERT INTO ops.pipeline_runs(run_id,source,status) VALUES (%s,'nightly_prices','running')", (run_id,))
        try:
            universe = dict(conn.execute("""SELECT i.ticker,i.instrument_id FROM core.price_universe u
                JOIN core.instruments i USING(instrument_id) ORDER BY i.ticker""").fetchall())
            if "CSUS.L" not in universe:
                raise ValueError("CSUS.L is missing from the configured update universe.")
            start = conn.execute("SELECT min(trade_date) FROM core.daily_prices p JOIN core.price_universe USING(instrument_id)").fetchone()[0]
            if day < start:
                raise ValueError("Target date predates the imported price history.")
            required = {}
            for instrument_id, kind, trade_date in conn.execute("""SELECT p.instrument_id,p.price_type,p.trade_date
                FROM core.daily_prices p JOIN core.preferred_price_sources s
                ON (p.instrument_id,p.price_type,p.source)=(s.instrument_id,s.price_type,s.source)
                JOIN core.price_universe u ON u.instrument_id=p.instrument_id"""):
                required.setdefault((instrument_id, kind), set()).add(trade_date)
            expected = {"us": last_session(day, "NYSE"), "uk": last_session(day, "LSE")}
            us = [symbol for symbol in universe if symbol != "CSUS.L"]
            observed, errors, stage_count = {}, {}, 0
            with tempfile.TemporaryDirectory(prefix="carlson_prices_") as temp:
                stage_file = Path(temp) / "quotes.csv"
                with stage_file.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    groups = [(alpaca, us[idx:idx+50]) for idx in range(0, len(us), 50)] + [(yahoo, ["CSUS.L"])]
                    for provider, symbols in groups:
                        for kind in ["close", "adjusted_close"]:
                            try:
                                for quote in provider.fetch(symbols, start, day, kind):
                                    if quote.ticker not in symbols or quote.price_type != kind or quote.source != provider.source:
                                        raise ProviderError("Unexpected price key in provider output.")
                                    if not start <= quote.trade_date <= day or quote.currency != "USD":
                                        raise ProviderError("Unexpected quote date/currency.")
                                    value = positive(quote.price)
                                    key = (universe[quote.ticker], quote.price_type, quote.source)
                                    if quote.trade_date in observed.get(key, set()):
                                        raise ProviderError("Duplicate quote key from provider.")
                                    observed.setdefault(key, set()).add(quote.trade_date)
                                    writer.writerow([universe[quote.ticker], quote.trade_date, quote.source, kind, value, quote.currency])
                                    stage_count += 1
                            except (ProviderError, KeyError, ValueError) as exc:
                                for symbol in symbols:
                                    errors.setdefault(symbol, []).append(str(exc) if isinstance(exc, ProviderError) else type(exc).__name__)
                accepted, symbol_stats = [], {}
                for symbol, instrument_id in universe.items():
                    provider = yahoo if symbol == "CSUS.L" else alpaca
                    expected_end = expected["uk" if symbol == "CSUS.L" else "us"]
                    stats = {"expected_last_session": str(expected_end), "types": {}, "errors": errors.get(symbol, [])}
                    for kind in ["close", "adjusted_close"]:
                        key = (instrument_id, kind, provider.source)
                        dates = observed.get(key, set())
                        missing_history = required.get((instrument_id, kind), set()) - dates
                        complete = bool(dates) and not missing_history and max(dates) >= expected_end
                        stats["types"][kind] = {"rows": len(dates), "last_date": str(max(dates)) if dates else None,
                            "missing_history_dates": len(missing_history), "accepted": complete}
                        if complete:
                            accepted.append(key)
                    stats["status"] = "updated" if all(v["accepted"] for v in stats["types"].values()) else "incomplete"
                    stats["requested_date_is_session"] = expected_end == day
                    symbol_stats[symbol] = (provider.source, stats)
                with conn.transaction():
                    conn.execute("""CREATE TEMP TABLE stage_updates(instrument_id bigint,trade_date date,source text,
                        price_type text,price numeric,currency text,UNIQUE(instrument_id,trade_date,source,price_type)) ON COMMIT DROP""")
                    with stage_file.open("rb") as handle, conn.cursor() as cursor:
                        with cursor.copy("COPY stage_updates FROM STDIN WITH (FORMAT CSV)") as copy:
                            while block := handle.read(1024*1024):
                                copy.write(block)
                    conn.execute("CREATE TEMP TABLE accepted_series(instrument_id bigint,price_type text,source text,PRIMARY KEY(instrument_id,price_type)) ON COMMIT DROP")
                    with conn.cursor() as cursor:
                        cursor.executemany("INSERT INTO accepted_series VALUES (%s,%s,%s)", accepted)
                    written = conn.execute("""INSERT INTO core.daily_prices
                        SELECT s.instrument_id,s.trade_date,s.source,s.price_type,s.price,s.currency,now(),%s
                        FROM stage_updates s JOIN accepted_series a USING(instrument_id,price_type,source)
                        ON CONFLICT(instrument_id,trade_date,source,price_type) DO UPDATE
                        SET price=EXCLUDED.price,currency=EXCLUDED.currency,retrieved_at=EXCLUDED.retrieved_at,batch_id=EXCLUDED.batch_id""", (run_id,)).rowcount
                    conn.execute("""INSERT INTO core.preferred_price_sources SELECT * FROM accepted_series
                        ON CONFLICT(instrument_id,price_type) DO UPDATE SET source=EXCLUDED.source""")
                    incomplete = [symbol for symbol, (_, stats) in symbol_stats.items() if stats["status"] != "updated"]
                    status = "partial" if incomplete or errors else "succeeded"
                    result = {"run_id": str(run_id), "status": status, "requested_date": str(day),
                        "history_start": str(start), "effective_last_sessions": {k: str(v) for k,v in expected.items()},
                        "universe_tickers": len(universe), "downloaded_price_rows": stage_count, "published_price_rows": written,
                        "accepted_series": len(accepted), "updated_tickers": len(universe)-len(incomplete),
                        "incomplete_tickers": incomplete, "symbols": {s: stats for s, (_, stats) in symbol_stats.items()}}
                    with conn.cursor() as cursor:
                        cursor.executemany("INSERT INTO ops.update_symbol_status VALUES (%s,%s,%s,%s,%s)", [
                            (run_id, symbol, provider, stats["status"], Jsonb(stats)) for symbol,(provider,stats) in symbol_stats.items()])
                    if written:
                        conn.execute("UPDATE ops.data_revision SET revision=revision+1,updated_at=now()")
                    conn.execute("UPDATE ops.pipeline_runs SET status=%s,finished_at=now(),stats=%s WHERE run_id=%s", (status, Jsonb(result), run_id))
            if write_report:
                try:
                    folder = reports_dir(); folder.mkdir(parents=True, exist_ok=True)
                    target = folder / "update_latest.json"; temporary = target.with_suffix(".tmp")
                    temporary.write_text(json.dumps(result, indent=2)); temporary.replace(target)
                except OSError:
                    print("Database commit succeeded; writing the optional local report failed.")
            return result
        except Exception as exc:
            conn.execute("UPDATE ops.pipeline_runs SET status='failed',finished_at=now(),error_summary=%s WHERE run_id=%s", (type(exc).__name__, run_id))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="YYYY-MM-DD; default is yesterday in UPDATE_TIMEZONE")
    parser.add_argument("--check-config", action="store_true", help="Check key presence without sending a request")
    args = parser.parse_args()
    if args.check_config:
        env = environment()
        print(json.dumps({"alpaca_key_present": bool(env.get("ALPACA_API_KEY") or env.get("APCA_API_KEY_ID")),
                          "alpaca_secret_present": bool(env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")),
                          "feed": env.get("ALPACA_FEED") or "sip"}, indent=2))
        return 0
    try:
        result = run(args.date)
        print(json.dumps({k:v for k,v in result.items() if k != "symbols"}, indent=2))
        return 2 if result["status"] == "partial" else 0
    except (ProviderError, ValueError, psycopg.Error) as exc:
        print(str(exc) if isinstance(exc, (ProviderError, ValueError)) else "Database update failed; see ops.pipeline_runs.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
