import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import yfinance as yf
# from IPython.display import Image, display, clear_output
from scipy.stats import norm, t
import streamlit as st

def _close_df_from_yf(tickers, start, end):
    """Return Close prices as a DataFrame even when there's a single ticker."""
    data = yf.download(tickers, start=start, end=end, interval="1d", auto_adjust=True, threads=True)
    close = data['Close']
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])
    return close.dropna(how='all')

def _group_shares(df_like, col_ticker='PureTicker', col_shares='Shares'):
    """Aggregate shares per ticker to avoid duplicate index issues."""
    return (df_like[[col_ticker, col_shares]]
            .groupby(col_ticker, as_index=True)[col_shares]
            .sum())

def _safe_cholesky(corr, max_tries=5, jitter=1e-10):
    """Try Cholesky with small diagonal jitter if needed."""
    for k in range(max_tries):
        try:
            return np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            corr = corr.copy()
            np.fill_diagonal(corr, np.clip(np.diag(corr) + jitter, 1e-12, None))
            jitter *= 10
    # last resort: eigval clip
    vals, vecs = np.linalg.eigh(corr)
    vals_clipped = np.clip(vals, 1e-10, None)
    corr_pd = vecs @ np.diag(vals_clipped) @ vecs.T
    return np.linalg.cholesky(corr_pd)

def start_reset(file_name):
    """
    file_name: "Portfolio as of 03-29-2025.csv"
    """
    df = pd.read_csv(file_name)
    df['PureTicker'] = df['Ticker'].str.split(':').str[-1]

    # filter First_American_Funds
    First_American_Funds = ["FAIX.X", "FGVX.X"]

    global df_filtered
    df_filtered = df[~df['PureTicker'].isin(First_American_Funds)].copy()
    
    global cash_pool
    df_faf = df[df['PureTicker'].isin(First_American_Funds)].copy()
    cash_pool = df_faf['Market Value [USD]'].sum()

    global tickers
    tickers = df_filtered['PureTicker'].tolist()


def ivar_multiple(additions, start_date, end_date):
    """
    Compute iVaR (std. dev delta) from adding multiple tickers.

    additions: list of (ticker, shares) tuples
    """
    global df_filtered, cash_pool, tickers

    # Backup original state
    df_backup = df_filtered.copy()
    cash_backup = cash_pool
    tickers_backup = tickers.copy()

    # Compute std dev before
    portfolio = _close_df_from_yf(tickers, start=start_date, end=end_date)

    # aggregate shares per ticker to be safe with duplicates
    shares_series = _group_shares(df_filtered, 'PureTicker', 'Shares')
    shares_series = shares_series.reindex(portfolio.columns).fillna(0)

    position_values = portfolio * shares_series
    position_diff = position_values.diff()

    equity_pool = df_filtered['Market Value [USD]'].sum()
    ratio = equity_pool / (equity_pool + cash_pool) if (equity_pool + cash_pool) != 0 else 0.0

    portfolio_before = position_diff.sum(axis=1)
    std_before = portfolio_before.std() * ratio

    # add new positions
    for ticker, shares in additions:
        price_data = _close_df_from_yf(ticker, start=start_date, end=end_date)
        if price_data.empty:
            print(f"No data for {ticker} in the selected period; skipping.")
            continue
        last_price = float(price_data.iloc[-1].item() if price_data.shape[1] == 1 else price_data[ticker].iloc[-1])
        value = last_price * shares

        if cash_pool < value:
            print(f"NOT ENOUGH CASH for {ticker} ({shares} shares). Needed: ${value:.2f}, Available: ${cash_pool:.2f}")
            continue

        new_row = pd.DataFrame([{
            "Shares": shares,
            "PureTicker": ticker,
            "Market Value [USD]": value,
            "Price [USD]": last_price
        }])
        
        df_filtered = pd.concat([df_filtered, new_row], ignore_index=True)
        # de-duplicate tickers list
        tickers = list(dict.fromkeys(tickers + [ticker]))
        cash_pool -= value

    # std dev after the new names added
    portfolio = _close_df_from_yf(tickers, start=start_date, end=end_date)

    shares_series = _group_shares(df_filtered, 'PureTicker', 'Shares')
    shares_series = shares_series.reindex(portfolio.columns).fillna(0)

    position_values = portfolio * shares_series
    position_diff = position_values.diff()

    equity_pool = df_filtered['Market Value [USD]'].sum()
    ratio = equity_pool / (equity_pool + cash_pool) if (equity_pool + cash_pool) != 0 else 0.0

    portfolio_after = position_diff.sum(axis=1)
    std_after = portfolio_after.std() * ratio

    ivar_value = std_after - std_before

    print("=" * 40)
    print(f"Std Dev BEFORE:     {std_before:.4f}")
    print(f"Std Dev AFTER:      {std_after:.4f}")
    print(f"Incremental VaR:    {ivar_value:.4f}")
    print("=" * 40)

    # Restore original state
    df_filtered = df_backup
    tickers = tickers_backup
    cash_pool = cash_backup


def simulation(distribution, forecast_days=21, start_date="2024-03-29", end_date="2025-03-29", 
               stats_before=None, rf_value=0.03):
    global cash_pool, df_filtered, tickers

    portfolio = _close_df_from_yf(tickers, start=start_date, end=end_date)

    log_returns = np.log(portfolio / portfolio.shift(1)).dropna()
    mu = log_returns.mean(axis=0).values
    sigma = log_returns.std(axis=0).values
    # prevent zero / near-zero volatility issues
    sigma = np.clip(sigma, 1e-12, None)

    corr_matrix = log_returns.corr().values
    L = _safe_cholesky(corr_matrix)

    num_assets = len(tickers)
    N = 500
    steps = forecast_days

    Z = np.random.normal(size=(num_assets, steps, N))
    X = np.empty_like(Z)
    for s in range(steps):
        X[:, s, :] = L @ Z[:, s, :]

    U = norm.cdf(X)

    if distribution == "normal":
        simulated_returns = norm.ppf(U, loc=mu[:, None, None], scale=sigma[:, None, None])
    elif distribution == "t":
        df_t = 5
        sim_t = t.ppf(U, df=df_t)
        simulated_returns = mu[:, None, None] + sigma[:, None, None] * sim_t / np.sqrt(df_t / (df_t - 2))
    else:
        empirical_returns = log_returns.values.T
        simulated_returns = np.zeros_like(U)
        for i in range(num_assets):
            simulated_returns[i] = np.quantile(empirical_returns[i], U[i])

    S0 = portfolio.iloc[-1].values.reshape(-1, 1)
    shares = df_filtered[['PureTicker', 'Shares']].copy()
    # aggregate shares before aligning to columns
    shares_for_weights = _group_shares(shares, 'PureTicker', 'Shares').reindex(portfolio.columns).fillna(0)
    weights = shares_for_weights.values.reshape(-1, 1)

    price_paths = np.zeros((num_assets, steps + 1, N))
    price_paths[:, 0, :] = S0

    for step_idx in range(1, steps + 1):
        price_paths[:, step_idx, :] = price_paths[:, step_idx - 1, :] * np.exp(simulated_returns[:, step_idx - 1, :])

    portfolio_paths = (price_paths * weights[..., None]).sum(axis=0)


    fig, ax = plt.subplots(figsize=(12, 6))

    # 每條模擬路徑
    for i in range(N):
        ax.plot(range(steps + 1), portfolio_paths[:, i], alpha=0.07)

    # 平均路徑
    ax.plot(range(steps + 1), portfolio_paths.mean(axis=1), linestyle='--', label='Mean')

    # 信賴區間
    lower = np.percentile(portfolio_paths, 5, axis=1)
    upper = np.percentile(portfolio_paths, 95, axis=1)
    ax.fill_between(range(steps + 1), lower, upper, alpha=0.2, label='5-95% CI')

    # 標題與格式
    ax.set_title("Simulated Portfolio Value over Time (by Steps)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.grid(alpha=0.7, linestyle='--')
    ax.legend()
    fig.tight_layout()


    # Risk Metrics
    initial_val = portfolio_paths[0, :].mean()
    final_vals = portfolio_paths[-1, :]
    returns = (final_vals - initial_val) / initial_val
    losses = initial_val - final_vals

    alpha = 0.95
    VaR = np.percentile(losses, alpha * 100)
    CVaR = losses[losses >= VaR].mean()
    losses_pct = losses / initial_val
    VaR_pct = np.percentile(losses_pct, alpha * 100)
    CVaR_pct = np.mean(losses_pct[losses_pct >= VaR_pct])

    # sharpe and sortino (note: rf_value is used as-is, not annualized)
    excess_returns = returns - rf_value
    sharpe = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else np.nan
    downside_returns = excess_returns[excess_returns < 0]
    sortino = np.mean(excess_returns) / np.std(downside_returns) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else np.nan


    return fig, VaR, CVaR, VaR_pct, CVaR_pct, sharpe, sortino