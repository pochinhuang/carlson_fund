import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import streamlit as st
from pathlib import Path

def fetch_close_prices(tickers, start, end, interval="1d"):
    """Download adjusted close prices and always return a DataFrame."""
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    close = data["Close"]
    if isinstance(close, pd.Series):
        name = tickers if isinstance(tickers, str) else tickers[0]
        close = close.to_frame(name=name)

    # Drop failed downloads / empty rows
    close = close.dropna(axis=1, how="all").dropna(how="all")
    return close


def aggregate_shares(df_port, ticker_col="Ticker", shares_col="Shares"):
    """Aggregate duplicate ticker rows by summing shares."""
    return (
        df_port[[ticker_col, shares_col]]
        .groupby(ticker_col, as_index=True)[shares_col]
        .sum()
    )


def safe_cholesky(corr, max_tries=5, jitter=1e-10):
    """Compute a stable Cholesky factor from a correlation matrix."""
    corr_work = corr.copy()

    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(corr_work)
        except np.linalg.LinAlgError:
            np.fill_diagonal(
                corr_work,
                np.clip(np.diag(corr_work) + jitter, 1e-12, None)
            )
            jitter *= 10

    vals, vecs = np.linalg.eigh(corr_work)
    vals = np.clip(vals, 1e-10, None)
    corr_pd = vecs @ np.diag(vals) @ vecs.T
    return np.linalg.cholesky(corr_pd)


def load_portfolio_state(file_name, cash_pool=0.0):
    """
    Load cleaned holdings file.
    Expected columns:
    - Ticker
    - Shares
    - Market Value [USD]
    """
    df_port = pd.read_csv(file_name).copy()
    tickers = df_port["Ticker"].dropna().tolist()

    return {
        "df": df_port,
        "tickers": tickers,
        "cash_pool": float(cash_pool),
    }


def simulate_correlated_returns(log_returns, distribution, forecast_days, n_sims=500):
    """Simulate correlated future log returns."""
    mu = log_returns.mean(axis=0).values
    sigma = np.clip(log_returns.std(axis=0).values, 1e-12, None)

    corr_matrix = log_returns.corr().values
    L = safe_cholesky(corr_matrix)

    num_assets = log_returns.shape[1]
    steps = forecast_days

    Z = np.random.normal(size=(num_assets, steps, n_sims))
    X = np.empty_like(Z)

    for s in range(steps):
        X[:, s, :] = L @ Z[:, s, :]

    U = norm.cdf(X)

    distribution = distribution.lower()

    if distribution == "normal":
        simulated_returns = norm.ppf(
            U,
            loc=mu[:, None, None],
            scale=sigma[:, None, None]
        )

    elif distribution == "t":
        df_t = 5
        sim_t = t.ppf(U, df=df_t)
        simulated_returns = (
            mu[:, None, None]
            + sigma[:, None, None] * sim_t / np.sqrt(df_t / (df_t - 2))
        )

    else:  # empirical
        empirical_returns = log_returns.values.T
        simulated_returns = np.zeros_like(U)
        for i in range(num_assets):
            simulated_returns[i] = np.quantile(empirical_returns[i], U[i])

    return simulated_returns


def build_portfolio_paths(close_df, df_port, simulated_returns):
    """Convert simulated returns into price paths and portfolio value paths."""
    num_assets, steps, n_sims = simulated_returns.shape

    s0 = close_df.iloc[-1].values.reshape(-1, 1)

    shares = aggregate_shares(df_port, ticker_col="Ticker", shares_col="Shares")
    shares = shares.reindex(close_df.columns).fillna(0).values.reshape(-1, 1)

    price_paths = np.zeros((num_assets, steps + 1, n_sims))
    price_paths[:, 0, :] = s0

    for step_idx in range(1, steps + 1):
        price_paths[:, step_idx, :] = (
            price_paths[:, step_idx - 1, :] * np.exp(simulated_returns[:, step_idx - 1, :])
        )

    portfolio_paths = (price_paths * shares[..., None]).sum(axis=0)
    return price_paths, portfolio_paths


def compute_risk_metrics(portfolio_paths, rf_value=0.03, alpha=0.95):
    """Compute VaR / CVaR / Sharpe / Sortino from simulated portfolio paths."""
    initial_val = portfolio_paths[0, :].mean()
    final_vals = portfolio_paths[-1, :]

    returns = (final_vals - initial_val) / initial_val
    losses = initial_val - final_vals

    var = np.percentile(losses, alpha * 100)
    tail_losses = losses[losses >= var]
    cvar = tail_losses.mean() if len(tail_losses) > 0 else np.nan

    losses_pct = losses / initial_val
    var_pct = np.percentile(losses_pct, alpha * 100)
    tail_losses_pct = losses_pct[losses_pct >= var_pct]
    cvar_pct = tail_losses_pct.mean() if len(tail_losses_pct) > 0 else np.nan

    excess_returns = returns - rf_value
    sharpe = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else np.nan

    downside = excess_returns[excess_returns < 0]
    sortino = (
        np.mean(excess_returns) / np.std(downside)
        if len(downside) > 0 and np.std(downside) > 0
        else np.nan
    )

    return {
        "VaR": var,
        "CVaR": cvar,
        "VaR_pct": var_pct,
        "CVaR_pct": cvar_pct,
        "sharpe": sharpe,
        "sortino": sortino,
    }


def plot_portfolio_paths(portfolio_paths):
    """Plot simulated portfolio paths."""
    steps = portfolio_paths.shape[0] - 1
    n_sims = portfolio_paths.shape[1]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(n_sims):
        ax.plot(range(steps + 1), portfolio_paths[:, i], alpha=0.07)

    ax.plot(range(steps + 1), portfolio_paths.mean(axis=1), linestyle="--", label="Mean")

    lower = np.percentile(portfolio_paths, 5, axis=1)
    upper = np.percentile(portfolio_paths, 95, axis=1)
    ax.fill_between(range(steps + 1), lower, upper, alpha=0.2, label="5-95% CI")

    ax.set_title("Simulated Portfolio Value over Time")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.grid(alpha=0.7, linestyle="--")
    ax.legend()
    fig.tight_layout()

    return fig


def run_monte_carlo_simulation(
    state,
    distribution,
    forecast_days=21,
    start_date="2024-03-29",
    end_date="2025-03-29",
    rf_value=0.03,
    n_sims=500,
):
    """Run Monte Carlo simulation and return figure + metrics."""
    close_df = fetch_close_prices(state["tickers"], start=start_date, end=end_date)

    if close_df.empty or close_df.shape[1] == 0:
        raise ValueError("No valid price data downloaded for the selected period.")

    valid_tickers = close_df.columns.tolist()
    missing_tickers = [t for t in state["tickers"] if t not in valid_tickers]

    df_port = state["df"][state["df"]["Ticker"].isin(valid_tickers)].copy()
    if df_port.empty:
        raise ValueError("No holdings remain after filtering unavailable tickers.")

    log_returns = np.log(close_df / close_df.shift(1)).dropna()
    if log_returns.empty or len(log_returns) < 2:
        raise ValueError("Not enough return history to run simulation.")

    simulated_returns = simulate_correlated_returns(
        log_returns=log_returns,
        distribution=distribution,
        forecast_days=forecast_days,
        n_sims=n_sims,
    )

    _, portfolio_paths = build_portfolio_paths(close_df, df_port, simulated_returns)
    fig = plot_portfolio_paths(portfolio_paths)
    metrics = compute_risk_metrics(portfolio_paths, rf_value=rf_value)

    return fig, metrics, missing_tickers
