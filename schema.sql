CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id uuid PRIMARY KEY,
    source text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status text NOT NULL CHECK (status IN ('running', 'succeeded', 'partial', 'failed')),
    stats jsonb,
    error_summary text
);

-- Original CSVs stay in data/; the database records each import's hashes and row counts.
CREATE TABLE IF NOT EXISTS raw.import_files (
    run_id uuid NOT NULL REFERENCES ops.pipeline_runs(run_id),
    file_name text NOT NULL,
    sha256 text NOT NULL,
    row_count bigint NOT NULL CHECK (row_count >= 0),
    PRIMARY KEY (run_id, file_name)
);

CREATE TABLE IF NOT EXISTS core.instruments (
    instrument_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker text NOT NULL UNIQUE CHECK (ticker = upper(trim(ticker)) AND ticker <> ''),
    yahoo_ticker text,
    security_name text,
    instrument_type text NOT NULL DEFAULT 'unknown' CHECK (instrument_type IN ('stock', 'etf', 'unknown')),
    sector text,
    industry text,
    currency text NOT NULL DEFAULT 'USD' CHECK (length(currency) = 3)
);

CREATE TABLE IF NOT EXISTS core.portfolios (
    portfolio_id text PRIMARY KEY,
    name text NOT NULL,
    portfolio_type text NOT NULL CHECK (portfolio_type IN ('portfolio', 'benchmark'))
);

-- These inputs are fixed holdings. Their actual snapshot date was not supplied.
-- Store it as NULL unless HOLDINGS_AS_OF is explicitly provided; do not invent a date.
CREATE TABLE IF NOT EXISTS core.holdings (
    portfolio_id text NOT NULL REFERENCES core.portfolios(portfolio_id),
    instrument_id bigint NOT NULL REFERENCES core.instruments(instrument_id),
    as_of_date date,
    shares numeric,
    market_value numeric,
    weight numeric NOT NULL CHECK (weight >= 0 AND weight <= 1),
    batch_id uuid NOT NULL REFERENCES ops.pipeline_runs(run_id),
    PRIMARY KEY (portfolio_id, instrument_id)
);

CREATE TABLE IF NOT EXISTS core.daily_prices (
    instrument_id bigint NOT NULL REFERENCES core.instruments(instrument_id),
    trade_date date NOT NULL,
    source text NOT NULL,
    price_type text NOT NULL CHECK (price_type IN ('close', 'adjusted_close')),
    price numeric NOT NULL CHECK (price > 0),
    currency text NOT NULL CHECK (length(currency) = 3),
    retrieved_at timestamptz NOT NULL,
    batch_id uuid NOT NULL REFERENCES ops.pipeline_runs(run_id),
    PRIMARY KEY (instrument_id, trade_date, source, price_type)
);
CREATE INDEX IF NOT EXISTS ix_daily_prices_date ON core.daily_prices(trade_date);

-- The update universe is exactly the ticker set supplied in initial_daily_prices.csv.
-- Retain unpriced holdings separately rather than deleting them from the inputs.
CREATE TABLE IF NOT EXISTS core.price_universe (
    instrument_id bigint PRIMARY KEY REFERENCES core.instruments(instrument_id)
);

-- Each displayed history comes from one complete provider series, never a splice
-- of adjusted price levels from different providers or adjustment vintages.
CREATE TABLE IF NOT EXISTS core.preferred_price_sources (
    instrument_id bigint NOT NULL REFERENCES core.instruments(instrument_id),
    price_type text NOT NULL CHECK (price_type IN ('close', 'adjusted_close')),
    source text NOT NULL,
    PRIMARY KEY (instrument_id, price_type)
);

CREATE TABLE IF NOT EXISTS ops.data_revision (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    revision bigint NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now()
);
INSERT INTO ops.data_revision(singleton) VALUES (true) ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS ops.update_symbol_status (
    run_id uuid NOT NULL REFERENCES ops.pipeline_runs(run_id),
    ticker text NOT NULL,
    provider text NOT NULL,
    status text NOT NULL,
    stats jsonb NOT NULL,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS core.factor_proxies (
    factor_name text PRIMARY KEY CHECK (factor_name IN ('market', 'size', 'value', 'momentum', 'quality')),
    proxy_instrument_id bigint NOT NULL REFERENCES core.instruments(instrument_id)
);

-- One row per instrument/date/source, matching the original CSV's two price columns.
CREATE OR REPLACE VIEW mart.v_daily_prices_wide AS
SELECT p.trade_date, i.ticker, i.yahoo_ticker,
       max(p.price) FILTER (WHERE p.price_type = 'close') AS close,
       max(p.price) FILTER (WHERE p.price_type = 'adjusted_close') AS adj_close,
       p.currency, p.source, max(p.retrieved_at) AS retrieved_at
FROM core.daily_prices p
JOIN core.instruments i USING (instrument_id)
GROUP BY p.trade_date, i.ticker, i.yahoo_ticker, p.currency, p.source;

CREATE OR REPLACE VIEW mart.v_daily_returns AS
SELECT ticker, trade_date, source, price_type, price,
       price / lag(price) OVER (
           PARTITION BY instrument_id, source, price_type ORDER BY trade_date
       ) - 1 AS daily_return
FROM core.daily_prices
JOIN core.instruments USING (instrument_id);

CREATE OR REPLACE VIEW mart.v_sector_weights AS
SELECT h.portfolio_id, coalesce(i.sector, 'Unknown') AS sector, sum(h.weight) AS weight
FROM core.holdings h
JOIN core.instruments i USING (instrument_id)
GROUP BY h.portfolio_id, coalesce(i.sector, 'Unknown');

CREATE OR REPLACE VIEW mart.v_current_prices AS
SELECT i.ticker,p.trade_date,p.price_type,p.price,p.source,p.currency
FROM core.daily_prices p
JOIN core.preferred_price_sources s
  ON (p.instrument_id,p.price_type,p.source)=(s.instrument_id,s.price_type,s.source)
JOIN core.instruments i ON i.instrument_id=p.instrument_id;
