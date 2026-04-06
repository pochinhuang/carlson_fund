import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Factor Model",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded"
)

factor_tickers = ["SIZE", "VLUE", "MTUM", "QUAL"]
market_ticker = "CSUS.L"
portfolio_data_loc = "data/R2500G_proxy_Holdings_clean.csv"


@st.cache_data(show_spinner=False)
def load_portfolio_data():
    return pd.read_csv(portfolio_data_loc)


@st.cache_data(show_spinner=False)
def load_all_close_prices():
    df_1 = pd.read_csv("data/all_close_prices.csv")
    df_1 = df_1.apply(pd.to_numeric, errors="coerce")
    df_1 = df_1.dropna(axis=1, how="all")
    return df_1


@st.cache_data(show_spinner=False)
def load_factor_and_market_returns():
    factor_data = pd.read_csv("data/etf_factor_data.csv")
    market_data = pd.read_csv("data/market_data.csv")

    factor_data = factor_data.apply(pd.to_numeric, errors="coerce")
    market_data = market_data.apply(pd.to_numeric, errors="coerce")

    factor_returns = factor_data.pct_change().fillna(0)
    market_returns = market_data.ffill().pct_change().fillna(0)

    return factor_returns, market_returns


def compute_sector_factor_exposures(
    df_holdings,
    selected_sector,
    all_close_df,
    factor_returns,
    market_returns
):
    sector_tickers = (
        df_holdings.loc[df_holdings["sector"] == selected_sector, "Ticker"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not sector_tickers:
        return pd.DataFrame()

    valid_tickers = [t for t in sector_tickers if t in all_close_df.columns]

    if not valid_tickers:
        return pd.DataFrame()

    close_df = all_close_df[valid_tickers].copy()
    close_df = close_df.ffill()

    stock_returns = close_df.pct_change().fillna(0)

    valid_cols = [
        c for c in stock_returns.columns
        if not stock_returns[c].isna().all() and stock_returns[c].abs().sum() > 0
    ]
    stock_returns = stock_returns[valid_cols]

    if stock_returns.empty:
        return pd.DataFrame()

    factor_returns = factor_returns.reset_index(drop=True)
    market_returns = market_returns.reset_index(drop=True)
    stock_returns = stock_returns.reset_index(drop=True)

    min_len = min(len(factor_returns), len(market_returns), len(stock_returns))

    factor_returns = factor_returns.iloc[:min_len]
    market_returns = market_returns.iloc[:min_len]
    stock_returns = stock_returns.iloc[:min_len]

    all_returns = pd.concat(
        [factor_returns, market_returns, stock_returns],
        axis=1
    )

    all_returns = all_returns.dropna(subset=factor_tickers).fillna(0)

    factor_columns = [market_ticker] + factor_tickers
    stock_columns = stock_returns.columns.tolist()

    X = all_returns[factor_columns].values
    result_dict = {}

    for ticker in stock_columns:
        y = all_returns[ticker].values
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        result_dict[ticker] = np.concatenate([[model.intercept_], model.coef_])

    exposures = pd.DataFrame(
        result_dict,
        index=["Alpha", "Beta"] + factor_tickers
    ).T

    exposures.index.name = "Ticker"
    return exposures


df_holdings = load_portfolio_data()
all_close_df = load_all_close_prices()
factor_returns, market_returns = load_factor_and_market_returns()

st.subheader("Sector Factor Analysis")

sector_options = sorted(df_holdings["sector"].dropna().unique().tolist())

selected_sector = st.selectbox(
    "Select a Sector:",
    options=sector_options,
    index=0
)

if st.button("Run Factor Analysis"):
    with st.spinner(f"Analyzing sector '{selected_sector}'... please wait"):
        exposures = compute_sector_factor_exposures(
            df_holdings,
            selected_sector,
            all_close_df,
            factor_returns,
            market_returns
        )

    if exposures.empty:
        st.warning("No valid exposure results for this sector.")
    else:
        st.success(f"Done! Sector '{selected_sector}' exposures computed.")
        st.dataframe(exposures)

        factors_to_plot = ["Alpha", "Beta", "SIZE", "VLUE", "MTUM", "QUAL"]

        for factor in factors_to_plot:
            if factor in exposures.columns:
                fig = px.bar(
                    exposures,
                    x=exposures.index,
                    y=factor,
                    title=factor,
                    color_discrete_sequence=["#1ed760"]
                )
                fig.update_traces(hovertemplate="<b>%{x}</b><br>" + f"{factor}: " + "%{y:.4f}<extra></extra>")
                
                fig.update_layout(
                    xaxis_title="Ticker",
                    yaxis_title="Exposure",
                    template="plotly_white",
                    xaxis_tickangle=90,
                    height=400,
                )
                st.plotly_chart(fig, width="stretch")