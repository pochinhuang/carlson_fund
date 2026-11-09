import numpy as np
import pandas as pd
import pytest

import portfolio_analytics


def synthetic_model():
    rng = np.random.default_rng(812)
    dates = pd.bdate_range("2025-01-01", periods=181)
    x = rng.normal(0.0003, 0.01, (180, 5))
    beta = np.array([[0.8, 0.2], [0.3, -0.1], [-0.2, 0.6], [0.1, 0.4], [0.15, 0.2]])
    alpha = np.array([0.0002, -0.0001])
    y = x @ beta + alpha
    def levels(returns, names):
        return pd.DataFrame(np.vstack([np.full(len(names), 100), 100 * np.cumprod(1 + returns, axis=0)]), index=dates, columns=names)
    return levels(y, ["A", "B"]), levels(x, ["CSUS.L", "SIZE", "VLUE", "MTUM", "QUAL"]), alpha, beta


def test_factor_coefficients_and_weighted_portfolio():
    assets, factors, alpha, beta = synthetic_model()
    result = portfolio_analytics.fit_factor_exposures(assets, factors)
    np.testing.assert_allclose(result.to_numpy(), np.vstack([alpha, beta]).T, atol=1e-12)
    holdings = pd.DataFrame({"Ticker": ["B", "A"], "Weight (%)": [30, 70]})
    weighted = portfolio_analytics.weighted_factor_exposure(result, holdings)
    np.testing.assert_allclose(weighted, np.vstack([alpha, beta]) @ np.array([0.7, 0.3]), atol=1e-12)


def test_calendar_alignment_and_missing_returns():
    assets, factors, _, _ = synthetic_model()
    factors = factors.drop(factors.index[40]).iloc[::-1]
    assets.loc[assets.index[90], "B"] = np.nan
    actual = portfolio_analytics.fit_factor_exposures(assets, factors)
    dates = assets.index
    x = factors.reindex(dates).to_numpy()
    y = assets.loc[dates].to_numpy()
    x = x[1:] / x[:-1] - 1
    y = y[1:] / y[:-1] - 1
    for idx, ticker in enumerate(assets.columns):
        good = np.isfinite(y[:, idx]) & np.isfinite(x).all(axis=1)
        design = np.column_stack([np.ones(good.sum()), x[good]])
        expected = np.linalg.lstsq(design, y[good, idx], rcond=None)[0]
        np.testing.assert_allclose(actual.loc[ticker], expected, atol=1e-12)
        assert actual.attrs["observation_counts"][ticker] == good.sum()
    # Missing price and its following return are both missing, rather than fabricated zeroes.
    missing = assets.index[90]
    assert portfolio_analytics.daily_returns(assets).loc[missing:, "B"].iloc[:2].isna().all()


def test_insufficient_factor_history_is_reported():
    assets, factors, _, _ = synthetic_model()
    assets.loc[assets.index[:-20], "B"] = np.nan
    result = portfolio_analytics.fit_factor_exposures(assets, factors)
    assert list(result.index) == ["A"]
    assert result.attrs["skipped_tickers"] == ["B"]


def test_a_single_assets_missing_day_is_not_collapsed_into_a_longer_return():
    assets, factors, _, _ = synthetic_model()
    assets = assets[["A"]].copy()
    assets.iloc[90, 0] = np.nan
    result = portfolio_analytics.fit_factor_exposures(assets, factors)
    assert result.attrs["observation_counts"]["A"] == 178  # 180 daily returns minus two missing endpoint pairs.


def test_brinson_reconciles_active_return_and_five_sessions():
    dates = pd.bdate_range("2026-01-01", periods=7)
    holdings = pd.DataFrame({"Ticker": ["A", "B"], "Sector": ["X", "Y"], "Weight (%)": [60, 40]})
    benchmark = pd.DataFrame({"Ticker": ["C", "D"], "Sector": ["X", "Y"], "Weight (%)": [20, 80]})
    portfolio_prices = pd.DataFrame({"A": [100] * 6 + [110], "B": [100] * 6 + [105]}, index=dates)
    benchmark_prices = pd.DataFrame({"C": [100] * 6 + [108], "D": [100] * 6 + [102]}, index=dates)
    result = portfolio_analytics.calculate_sector_brinson(holdings, benchmark, portfolio_prices, benchmark_prices, lookback=5)
    assert result.attrs["start_date"] == dates[1]
    assert result.attrs["end_date"] == dates[-1]
    assert np.isclose(result["attribution"].to_numpy().sum(), 4.8)
    benchmark_prices.iloc[-1, 0] = np.nan
    start, end = portfolio_analytics.select_return_window(portfolio_prices, benchmark_prices, lookback=5)
    assert (start, end) == (dates[0], dates[-2])


def test_risk_contributions_reconcile_and_use_raw_close_for_value():
    assets, _, _, _ = synthetic_model()
    raw_close = assets.iloc[[-1]].copy() * [2, 3]
    holdings = pd.DataFrame({"Ticker": ["B", "A"], "Shares": [20, 10]})
    values = portfolio_analytics.current_position_values(raw_close, holdings)
    np.testing.assert_allclose(values, raw_close.iloc[-1] * [10, 20])
    contributions = portfolio_analytics.risk_contributions(assets, raw_close, holdings)
    assert np.isclose(contributions.sum(), 1)
    covariance = portfolio_analytics.daily_returns(assets).dropna().cov().to_numpy()
    weights = (values / values.sum()).to_numpy()
    expected = weights * (covariance @ weights) / (weights @ covariance @ weights)
    np.testing.assert_allclose(contributions.reindex(["A", "B"]), expected)


@pytest.mark.parametrize("distribution", ["Normal", "T", "Empirical"])
def test_monte_carlo_is_finite_and_reproducible(distribution):
    assets, _, _, _ = synthetic_model()
    log_returns = np.log(assets / assets.shift()).dropna()
    result = portfolio_analytics.simulate_correlated_returns(log_returns, distribution, 10, n_sims=300, seed=42)
    repeated = portfolio_analytics.simulate_correlated_returns(log_returns, distribution, 10, n_sims=300, seed=42)
    assert result.shape == (2, 10, 300)
    np.testing.assert_array_equal(result, repeated)
    assert np.isfinite(result).all()


def test_normal_simulation_preserves_marginals_and_correlation():
    rng = np.random.default_rng(11)
    history = rng.multivariate_normal([0.0002, 0.0008], [[0.0001, 0.00006], [0.00006, 0.0002]], size=2000)
    history = pd.DataFrame(history, columns=["A", "B"])
    simulation = portfolio_analytics.simulate_correlated_returns(history, "Normal", 5, n_sims=10000, seed=73).reshape(2, -1)
    np.testing.assert_allclose(simulation.mean(axis=1), history.mean(), atol=0.00015)
    np.testing.assert_allclose(simulation.std(axis=1), history.std(), rtol=0.02)
    np.testing.assert_allclose(np.corrcoef(simulation), history.corr(), atol=0.02)


def test_constant_asset_and_singular_correlation():
    history = pd.DataFrame({"A": [0.01, 0.02, 0.01], "B": [0.0, 0.0, 0.0]})
    simulation = portfolio_analytics.simulate_correlated_returns(history, "Normal", 5, 200, seed=7)
    assert np.all(simulation[1] == 0)
    factor = portfolio_analytics.safe_cholesky(np.ones((2, 2)))
    np.testing.assert_allclose(np.diag(factor @ factor.T), 1, atol=1e-12)


def test_price_paths_shares_and_cash():
    dates = pd.DatetimeIndex(["2026-11-03"])
    close = pd.DataFrame({"A": [100], "B": [50]}, index=dates)
    holdings = pd.DataFrame({"Ticker": ["B", "A", "A"], "Shares": [4, 1, 1]})
    shocks = np.zeros((2, 3, 2))
    paths, portfolio = portfolio_analytics.build_portfolio_paths(close, holdings, shocks, cash_pool=100, cash_daily_rate=0.01)
    np.testing.assert_allclose(paths[0], 100)
    np.testing.assert_allclose(portfolio[:, 0], 400 + 100 * 1.01 ** np.arange(4))


def test_horizon_rf_var_cvar_and_sortino():
    initial = 100
    final = np.array([80, 90, 100, 110], dtype=float)
    paths = np.vstack([np.full(4, initial)] * 21 + [final])
    actual = portfolio_analytics.compute_risk_metrics(paths, rf_value=0.12, alpha=0.75)
    rf = 1.12 ** (21 / 252) - 1
    excess = final / initial - 1 - rf
    losses = initial - final
    assert np.isclose(actual["horizon_rf"], rf)
    assert np.isclose(actual["sharpe"], excess.mean() / (final / initial - 1).std(ddof=1))
    assert np.isclose(actual["sortino"], excess.mean() / np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    assert actual["VaR"] == np.quantile(losses, 0.75)
    assert actual["CVaR"] == 20


def test_simulation_does_not_estimate_beyond_the_valuation_date(monkeypatch):
    from contextlib import contextmanager
    import matplotlib.pyplot as plt
    assets, _, _, _ = synthetic_model()
    close = assets.iloc[:-3].copy() * [2, 3]
    class FakeData:
        @contextmanager
        def session(self):
            yield self
        def get_prices(self, tickers, start, end, price_type):
            return assets.copy() if price_type == "adjusted_close" else close.copy()
    monkeypatch.setattr(portfolio_analytics, "DataAccess", FakeData)
    holdings = pd.DataFrame({"Ticker": ["A", "B"], "Shares": [3, 5]})
    state = {"df": holdings, "tickers": ["A", "B"], "cash_pool": 100}
    fig, metrics, missing = portfolio_analytics.run_monte_carlo_simulation(state, "Normal", forecast_days=5, n_sims=20, seed=8)
    try:
        assert metrics["price_as_of"] == str(close.index[-1].date())
        assert metrics["history_end"] == metrics["price_as_of"]
        assert np.isclose(metrics["initial_value"], (close.iloc[-1] * [3, 5]).sum() + 100)
        assert missing == []
    finally:
        plt.close(fig)
