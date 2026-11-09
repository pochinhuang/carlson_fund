"""Read-only check of the existing database and the application's calculations."""
import argparse
import json

import numpy as np

from data_access import DataAccess, DataAccessError, DatabaseConfig, describe_data
import portfolio_analytics


def verify(config=None):
    data = DataAccess(config).load_analytics_data()
    fund = portfolio_analytics.fit_factor_exposures(data.portfolio_prices, data.factor_prices)
    benchmark = portfolio_analytics.fit_factor_exposures(data.benchmark_prices, data.factor_prices)
    if len(fund) != len(data.holdings) or len(benchmark) != len(data.benchmark_holdings):
        raise ValueError("Some holdings do not have enough common factor history.")
    attribution = portfolio_analytics.calculate_sector_brinson(
        data.holdings, data.benchmark_holdings, data.portfolio_prices, data.benchmark_prices
    )
    portfolio_return = (attribution[("portfolio", "weight")] / 100 * attribution[("portfolio", "return")]).sum()
    benchmark_return = (attribution[("R2500G", "weight")] / 100 * attribution[("R2500G", "return")]).sum()
    total_effect = attribution["attribution"].to_numpy().sum()
    if not np.isclose(total_effect, portfolio_return - benchmark_return, atol=1e-10):
        raise ValueError("Brinson effects do not reconcile.")
    contribution = portfolio_analytics.risk_contributions(data.portfolio_prices, data.close_prices, data.holdings)
    if not np.isclose(contribution.sum(), 1, atol=1e-10):
        raise ValueError("Risk contributions do not sum to 100%.")
    fund_windows = fund.attrs["windows"]
    return {
        **describe_data(data),
        "fund_factor_models": len(fund), "benchmark_factor_models": len(benchmark),
        "fund_factor_observations": min(fund.attrs["observation_counts"].values()),
        "fund_factor_return_start": str(min(x[0] for x in fund_windows.values()).date()),
        "fund_factor_return_end": str(max(x[1] for x in fund_windows.values()).date()),
        "recent_return_start": str(attribution.attrs["start_date"].date()),
        "recent_return_end": str(attribution.attrs["end_date"].date()),
        "lookback_sessions": attribution.attrs["lookback_sessions"],
        "brinson_total_effect_pct_points": float(total_effect),
        "portfolio_minus_benchmark_pct_points": float(portfolio_return - benchmark_return),
        "risk_contribution_sum": float(contribution.sum()),
        "latest_fund_close_date": str(data.close_prices.dropna(how="any").index[-1].date()),
        "raw_benchmark_holdings": data.benchmark_holdings.attrs["raw_count"],
        "benchmark_price_coverage_pct": data.benchmark_holdings.attrs["price_coverage_pct"],
        "excluded_benchmark_count": len(data.benchmark_holdings.attrs["excluded_tickers"]),
        "data_revision": DataAccess(config).get_data_revision(),
        "checks": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", help="Existing PostgreSQL bootstrap .env path")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(DatabaseConfig.from_environment(args.env_file)), indent=2, allow_nan=False))
    except (DataAccessError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
