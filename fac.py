import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from yahooquery import Ticker
from sklearn.linear_model import LinearRegression


factor_tickers = ["SIZE", "VLUE", "MTUM", "QUAL"]
market_ticker = "CSUS.L"
return_history = "1y"
return_interval = "1d"

def load_fundamental_data():
    df = pd.read_csv("R2500G proxy–Holdings.csv").assign(
        Ticker=lambda d: d["Ticker"].str.rsplit(":", n=1).str[-1]
    )

    modules = ["summaryProfile", "quoteType"]

    symbols = df["Ticker"].dropna().unique().tolist()

    datasi = Ticker(symbols, asynchronous=True).get_modules(modules)
    dfsi = pd.DataFrame.from_dict(datasi).T

    dataframes = [pd.json_normalize([x for x in dfsi[module] if isinstance(x, dict)]) for module in modules]
    dfsi = pd.concat(dataframes, axis=1)

    dfsi = dfsi.set_index('symbol')
    dfsi = dfsi.reindex(symbols)

    return dfsi[['industry', 'sector']].dropna()

etf_factor_data = yf.download(factor_tickers, period=return_history,
                              interval=return_interval,
                              progress=True, auto_adjust=True)

etf_factor_returns = etf_factor_data["Close"].pct_change().fillna(0)

market_data = yf.download([market_ticker], period=return_history, interval=return_interval, 
                          progress=True, auto_adjust=True)
# get price returns for market
# note the dangerous ffill here
market_returns = market_data['Close'].ffill().pct_change().fillna(0)


def factors(df, sector):

    ttt = df[df['sector'] == sector].index.tolist()
    port_constituent_data = yf.download(ttt, 
                                        period=return_history, interval=return_interval, 
                                        progress=True, auto_adjust=True)

    # ✅ 填補遺漏值並計算報酬率
    port_constituent_data['Close'] = port_constituent_data['Close'].ffill()
    port_constituent_returns = port_constituent_data['Close'].pct_change().fillna(0)

    # === 移除下載失敗的股票 ===
    invalid_cols = [
        c for c in port_constituent_returns.columns
        if port_constituent_returns[c].isna().all() or (port_constituent_returns[c].abs().sum() == 0)
    ]
    # if invalid_cols:
    #     print(f"⚠️ Skipping invalid tickers ({len(invalid_cols)}): {invalid_cols}")
    port_constituent_returns = port_constituent_returns.drop(columns=invalid_cols)

    # ✅ 整合所有報酬率
    all_returns = pd.concat(
        [etf_factor_returns, market_returns, port_constituent_returns],
        axis=1
    )

    # benchmark 系列通常較長，只保留完整資料區間
    all_returns = all_returns.dropna(subset=factor_tickers)
    all_returns = all_returns.fillna(0)  # 注意：這會填 0 給 NaN 值


    # ✅ 計算每支股票對各因子的暴露 (exposure)
    factor_columns = [market_ticker] + factor_tickers
    X = all_returns[factor_columns].values

    result_dict = {}
    for col in all_returns.columns:
        if col not in factor_columns:
            y = all_returns[col].values
            model = LinearRegression(fit_intercept=True, copy_X=True)
            model.fit(X, y)
            result_dict[col] = np.concatenate([np.array([model.intercept_]), model.coef_])

    exposures = pd.DataFrame(result_dict, index=['Alpha', 'Beta'] + factor_tickers).T
    exposures.index.name = "Ticker"
    return exposures

with st.container():
    if "dfsi" not in st.session_state:
        with st.spinner("🧠 Loading... give me 1 min ⏳"):
            dfsi = load_fundamental_data()
            st.session_state["dfsi"] = dfsi

    else:
        dfsi = st.session_state["dfsi"]



with st.container():
    st.subheader("📊 Sector Factor Analysis")

    # Get unique sector options
    sector_options = dfsi['sector'].dropna().unique().tolist()
    sector_options.sort()

    # Dropdown menu
    selected_sector = st.selectbox(
        "Select a Sector:",
        options=sector_options,
        index=0,
        key="sector_select"
    )
    # Button to trigger analysis
    if st.button("Run Factor Analysis 🚀"):
        with st.spinner(f"Analyzing sector '{selected_sector}'... please wait ⏳"):
            exposures = factors(dfsi, selected_sector)

        st.success(f"✅ Done! Sector '{selected_sector}' exposures computed.")
        st.dataframe(exposures)

        factors_to_plot = ["Alpha", "Beta", "SIZE", "VLUE", "MTUM", "QUAL"]

        # Display five separate charts (or according to actual factor count)
        for factor in factors_to_plot:
            if factor in exposures.columns:
                fig = px.bar(
                    exposures,
                    x=exposures.index,
                    y=factor,
                    title=factor,
                    color_discrete_sequence=["#1f77b4"],
                )
                fig.update_layout(
                    xaxis_title="Ticker",
                    yaxis_title="Exposure",
                    template="plotly_white",
                    xaxis_tickangle=90,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)