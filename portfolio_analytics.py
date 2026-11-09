"""Date-aligned analytics and Monte Carlo calculations shared by the pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from data_access import DataAccess, DataQualityError


def install_data_refresh():
    """Poll a tiny revision counter; refresh data without adding visible controls."""
    import streamlit as st
    @st.fragment(run_every="12h")
    def poll():
        revision = DataAccess().get_data_revision()
        previous = st.session_state.get("_carlson_data_revision", revision)
        st.session_state["_carlson_data_revision"] = revision
        if revision != previous:
            st.rerun()
    poll()
    return st.session_state["_carlson_data_revision"]


def _prices(frame):
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique:
        raise ValueError("Prices must have a unique DatetimeIndex.")
    if not frame.columns.is_unique:
        raise ValueError("Price ticker columns must be unique.")
    result = frame.sort_index().astype(float)
    values = result.to_numpy()
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Observed prices must be finite and positive.")
    return result


def daily_returns(prices):
    """Leave missing observations missing; never replace them with zero returns."""
    return _prices(prices).pct_change(fill_method=None)


def fit_factor_exposures(asset_prices, factor_prices, min_observations=60):
    """OLS on matching price endpoints, with a shared intercept and five proxies.

    Use the US assets/style ETFs' session calendar before calculating returns, then require
    every proxy's two price endpoints. A UK holiday and its next unmatched
    interval are excluded. Assets with identical availability masks use one multi-
    output least-squares solve. Partial stock histories never get backfilled.
    Coefficients use raw ETF returns; Alpha is a daily intercept, not CAPM alpha.
    """
    assets, factors = _prices(asset_prices), _prices(factor_prices)
    if factors.shape[1] != 5 or min_observations < 6:
        raise ValueError("Five proxy series and at least six observations are required.")
    dates = assets.dropna(how="all").index.union(factors.iloc[:, 1:].dropna(how="all").index).sort_values()
    x_returns = daily_returns(factors.reindex(dates))
    y_returns = daily_returns(assets.reindex(dates))
    x_values = x_returns.to_numpy()
    good_x = np.isfinite(x_values).all(axis=1)
    columns = ["Alpha", "Beta"] + list(factors.columns[1:])
    groups, skipped, counts, windows = {}, [], {}, {}
    for ticker in assets.columns:
        valid = good_x & np.isfinite(y_returns[ticker].to_numpy())
        counts[ticker] = int(valid.sum())
        if valid.sum() < min_observations:
            skipped.append(ticker)
            continue
        windows[ticker] = (dates[valid][0], dates[valid][-1])
        groups.setdefault(valid.tobytes(), (valid, []))[1].append(ticker)
    output = {}
    for valid, tickers in groups.values():
        design = np.column_stack([np.ones(valid.sum()), x_values[valid]])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y_returns.loc[valid, tickers].to_numpy(), rcond=None)
        if rank < design.shape[1]:
            raise ValueError("The factor regression design is rank deficient.")
        output.update({ticker: coefficients[:, idx] for idx, ticker in enumerate(tickers)})
    result = pd.DataFrame.from_dict(output, orient="index", columns=columns)
    result = result.reindex([ticker for ticker in assets.columns if ticker in output])
    result.index.name = "Ticker"
    result.attrs.update(skipped_tickers=skipped, observation_counts=counts, windows=windows)
    return result


def weighted_factor_exposure(exposures, holdings):
    weights = holdings.groupby("Ticker")["Weight (%)"].sum().astype(float) / 100
    if not set(weights.index).issubset(exposures.index):
        raise DataQualityError("Factor results do not cover every portfolio holding.")
    weights /= weights.sum()  # Remove only the input's tiny rounding discrepancy.
    return exposures.reindex(weights.index).mul(weights, axis=0).sum(axis=0)


def select_return_window(portfolio_prices, benchmark_prices, lookback=5):
    """Latest complete endpoints separated by exactly `lookback` US sessions."""
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive number of trading sessions.")
    portfolio_prices, benchmark_prices = _prices(portfolio_prices), _prices(benchmark_prices)
    calendar = portfolio_prices.dropna(how="all").index.intersection(
        benchmark_prices.dropna(how="all").index
    ).sort_values()
    for end_idx in range(len(calendar) - 1, lookback - 1, -1):
        start, end = calendar[end_idx - lookback], calendar[end_idx]
        if (portfolio_prices.loc[[start, end]].notna().all().all()
                and benchmark_prices.loc[[start, end]].notna().all().all()):
            return start, end
    raise DataQualityError("No complete portfolio/benchmark return window is available.")


def calculate_sector_brinson(portfolio_holdings, benchmark_holdings, portfolio_prices, all_prices, lookback=5):
    """Fixed-snapshot Brinson-Fachler attribution in percentage points."""
    portfolio_prices = portfolio_prices.reindex(columns=portfolio_holdings["Ticker"].tolist())
    all_prices = all_prices.reindex(columns=benchmark_holdings["Ticker"].tolist())
    start, end = select_return_window(portfolio_prices, all_prices, lookback)

    def summarize(holdings, prices):
        returns = prices.loc[end] / prices.loc[start] - 1
        frame = holdings[["Ticker", "Sector", "Weight (%)"]].copy()
        frame["weight"] = frame["Weight (%)"] / frame["Weight (%)"].sum()
        frame["weighted_return"] = frame["weight"] * frame["Ticker"].map(returns)
        if frame["weighted_return"].isna().any():
            raise DataQualityError("Brinson requires endpoint prices for every holding.")
        grouped = frame.groupby("Sector")[["weight", "weighted_return"]].sum()
        grouped["return"] = grouped["weighted_return"].div(grouped["weight"]).fillna(0)
        return grouped[["weight", "return"]]

    sectors = summarize(portfolio_holdings, portfolio_prices).merge(
        summarize(benchmark_holdings, all_prices), how="outer", left_index=True, right_index=True,
        suffixes=("_portfolio", "_benchmark")
    ).fillna(0)
    total_benchmark_return = (sectors.weight_benchmark * sectors.return_benchmark).sum()
    active_weight = sectors.weight_portfolio - sectors.weight_benchmark
    active_return = sectors.return_portfolio - sectors.return_benchmark
    result = pd.DataFrame({
        ("portfolio", "weight"): sectors.weight_portfolio * 100,
        ("portfolio", "return"): sectors.return_portfolio * 100,
        ("R2500G", "weight"): sectors.weight_benchmark * 100,
        ("R2500G", "return"): sectors.return_benchmark * 100,
        ("attribution", "allocation effect"): active_weight * (sectors.return_benchmark - total_benchmark_return) * 100,
        ("attribution", "selection effect"): sectors.weight_benchmark * active_return * 100,
        ("attribution", "interaction effect"): active_weight * active_return * 100,
    })
    order = result["attribution"].abs().sum(axis=1).sort_values(ascending=False).index
    result = result.loc[order]
    result.index.name = "sector"
    result.attrs.update(start_date=start, end_date=end, lookback_sessions=lookback)
    return result  # Round for display only, preserving attribution reconciliation.


def aggregate_shares(df_port, ticker_col="Ticker", shares_col="Shares"):
    shares = pd.to_numeric(df_port[shares_col], errors="raise")
    if not np.isfinite(shares).all():
        raise DataQualityError("Shares must be finite.")
    return df_port.assign(**{shares_col: shares}).groupby(ticker_col)[shares_col].sum()


def current_position_values(close_prices, holdings):
    close = _prices(close_prices).dropna(how="any")
    if close.empty:
        raise DataQualityError("No complete closing-price snapshot is available.")
    shares = aggregate_shares(holdings).reindex(close.columns)
    if shares.isna().any():
        raise DataQualityError("A price column has no matching holding/shares.")
    values = close.iloc[-1] * shares
    if not np.isfinite(values).all() or values.sum() <= 0:
        raise DataQualityError("Portfolio market value must be positive.")
    return values


def risk_contributions(adjusted_prices, close_prices, holdings):
    close_snapshot = _prices(close_prices).dropna(how="any")
    if close_snapshot.empty:
        raise DataQualityError("No complete closing-price snapshot is available.")
    returns = daily_returns(adjusted_prices.loc[:close_snapshot.index[-1]]).dropna(how="any")
    if len(returns) < 2:
        raise DataQualityError("Not enough common return history for risk contributions.")
    values = current_position_values(close_prices, holdings).reindex(returns.columns)
    weights = values / values.sum()
    covariance = returns.cov().to_numpy()
    w = weights.to_numpy()
    variance = float(w @ covariance @ w)
    if variance <= 0:
        raise DataQualityError("Portfolio variance must be positive.")
    result = pd.Series(w * (covariance @ w) / variance, index=returns.columns, name="Contribution_Ratio")
    result.index.name = "Ticker"
    return result.sort_values(ascending=False)


def fetch_close_prices(tickers, start, end, interval="1d"):
    """Compatibility wrapper: adjusted prices, requested tickers, inclusive dates."""
    if interval != "1d":
        raise ValueError("The database currently contains daily prices only.")
    return DataAccess().get_prices(tickers, start, end, price_type="adjusted_close")


def load_portfolio_state(file_name=None, cash_pool=0.0):
    """Read holdings from PostgreSQL; the legacy filename argument is recognized only as an identifier."""
    if file_name is not None and str(file_name).replace("\\", "/").rsplit("/", 1)[-1] not in (
        "fund", "portfolio_holdings.csv"
    ):
        raise ValueError("Use portfolio 'fund'; CSV loading is no longer supported.")
    if not np.isfinite(cash_pool) or cash_pool < 0:
        raise ValueError("Cash must be a finite nonnegative amount.")
    holdings = DataAccess().get_holdings("fund")
    return {"df": holdings, "tickers": holdings["Ticker"].tolist(), "cash_pool": float(cash_pool)}


def safe_cholesky(corr, max_tries=5, jitter=1e-10):
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or not np.isfinite(corr).all():
        raise ValueError("Correlation must be a finite square matrix.")
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(corr)
        repaired = (vectors * np.clip(values, jitter, None)) @ vectors.T
        scale = np.sqrt(np.diag(repaired))
        repaired /= np.outer(scale, scale)  # Preserve unit marginal variances.
        return np.linalg.cholesky(repaired)


def simulate_correlated_returns(log_returns, distribution, forecast_days, n_sims=500, seed=None):
    if (not isinstance(forecast_days, (int, np.integer)) or forecast_days < 1
            or not isinstance(n_sims, (int, np.integer)) or n_sims < 2):
        raise ValueError("Use at least one forecast day and two scenarios.")
    values = log_returns.to_numpy(dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Simulation requires finite common return history.")
    mu, sigma = values.mean(axis=0), values.std(axis=0, ddof=1)
    corr = log_returns.corr().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1)
    factor = safe_cholesky(corr)
    rng = np.random.default_rng(seed)
    gaussian = np.einsum("ij,jks->iks", factor,
                         rng.normal(size=(values.shape[1], forecast_days, n_sims)), optimize=True)
    distribution = distribution.lower()
    if distribution == "normal":
        return mu[:, None, None] + sigma[:, None, None] * gaussian
    uniforms = np.clip(norm.cdf(gaussian), np.finfo(float).eps, 1 - np.finfo(float).eps)
    if distribution == "t":
        standardized = t.ppf(uniforms, df=5) / np.sqrt(5 / 3)
        return mu[:, None, None] + sigma[:, None, None] * standardized
    if distribution == "empirical":
        return np.stack([np.quantile(values[:, idx], uniforms[idx]) for idx in range(values.shape[1])])
    raise ValueError("Distribution must be Normal, T, or Empirical.")


def build_portfolio_paths(close_df, df_port, simulated_returns, cash_pool=0.0, cash_daily_rate=0.0):
    initial = _prices(close_df).iloc[-1].to_numpy()
    shares = aggregate_shares(df_port).reindex(close_df.columns)
    if shares.isna().any() or not np.isfinite(initial).all() or simulated_returns.shape[0] != len(initial):
        raise DataQualityError("Simulation assets, starting prices, and shares do not match.")
    num_assets, steps, n_sims = simulated_returns.shape
    paths = np.empty((num_assets, steps + 1, n_sims))
    paths[:, 0, :] = initial[:, None]
    paths[:, 1:, :] = initial[:, None, None] * np.exp(np.cumsum(simulated_returns, axis=1))
    portfolio = np.einsum("its,i->ts", paths, shares.to_numpy(), optimize=True)
    portfolio += cash_pool * np.power(1 + cash_daily_rate, np.arange(steps + 1))[:, None]
    return paths, portfolio


def compute_risk_metrics(portfolio_paths, rf_value=0.03, alpha=0.95, trading_days=252):
    paths = np.asarray(portfolio_paths, dtype=float)
    if (paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] < 2
            or not np.isfinite(paths).all() or not 0 < alpha < 1
            or not np.isfinite(rf_value) or rf_value <= -1 or trading_days <= 0):
        raise ValueError("Invalid simulation paths, confidence level, or annual risk-free rate.")
    initial = float(paths[0].mean())
    if initial <= 0 or not np.allclose(paths[0], initial):
        raise ValueError("All scenarios must share a positive initial portfolio value.")
    horizon = paths.shape[0] - 1
    rf_horizon = (1 + rf_value) ** (horizon / trading_days) - 1
    returns = paths[-1] / initial - 1
    losses = initial - paths[-1]
    var = float(np.quantile(losses, alpha))
    cvar = float(losses[losses >= var].mean())
    excess = returns - rf_horizon
    volatility = float(returns.std(ddof=1))
    downside_deviation = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    return {"VaR": var, "CVaR": cvar, "VaR_pct": var / initial, "CVaR_pct": cvar / initial,
            "sharpe": float(excess.mean() / volatility) if volatility > 0 else np.nan,
            "sortino": float(excess.mean() / downside_deviation) if downside_deviation > 0 else np.nan,
            "horizon_rf": rf_horizon, "horizon_days": horizon, "initial_value": initial}


def plot_portfolio_paths(portfolio_paths):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    days = np.arange(portfolio_paths.shape[0])
    ax.plot(days, portfolio_paths, alpha=0.07)
    ax.plot(days, portfolio_paths.mean(axis=1), linestyle="--", label="Mean")
    lower, upper = np.percentile(portfolio_paths, [5, 95], axis=1)
    ax.fill_between(days, lower, upper, alpha=0.2, label="5-95% CI")
    ax.set(title="Simulated Portfolio Value over Time", xlabel="Steps", ylabel="Portfolio Value (USD)")
    ax.grid(alpha=0.7, linestyle="--")
    ax.legend()
    fig.tight_layout()
    return fig


def run_monte_carlo_simulation(state, distribution, forecast_days=21, start_date=None,
                               end_date=None, rf_value=0.03, n_sims=500, seed=None):
    with DataAccess().session() as db:
        adjusted = db.get_prices(state["tickers"], start_date, end_date, price_type="adjusted_close")
        close = db.get_prices(state["tickers"], start_date, end_date, price_type="close")
    adjusted = adjusted.dropna(how="all")
    # Require every current holding, rather than silently reducing portfolio risk.
    complete = adjusted.dropna(how="any").index.intersection(close.dropna(how="any").index)
    if complete.empty:
        raise DataQualityError("No complete closing-price snapshot for the portfolio.")
    as_of = complete[-1]
    adjusted = adjusted.loc[:as_of]
    log_returns = np.log(adjusted / adjusted.shift(1)).dropna(how="any")
    if len(log_returns) < 20:
        raise DataQualityError("Use a historical window with at least 20 common daily returns.")
    simulated = simulate_correlated_returns(log_returns, distribution, forecast_days, n_sims, seed)
    if not np.isfinite(rf_value) or rf_value <= -1:
        raise ValueError("The annual risk-free rate must exceed -100%.")
    _, paths = build_portfolio_paths(close.loc[[as_of]], state["df"], simulated,
                                     cash_pool=state.get("cash_pool", 0), cash_daily_rate=(1 + rf_value) ** (1 / 252) - 1)
    metrics = compute_risk_metrics(paths, rf_value=rf_value)
    metrics.update(price_as_of=str(as_of.date()), history_start=str(log_returns.index[0].date()),
                   history_end=str(log_returns.index[-1].date()), observations=len(log_returns))
    return plot_portfolio_paths(paths), metrics, []
