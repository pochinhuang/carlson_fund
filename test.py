import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

from yahooquery import Ticker
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title = "Portfolio Dashboard",
    page_icon = "📊",
    layout = "wide",                
    initial_sidebar_state = "expanded" 
)

cash = {"FAIX.X", "FGVX.X"}
etf  = {"IWO", "IWP"}
exclusions = cash | etf
modules = ["summaryProfile", "quoteType"]

# read + organize Ticker
df = (
    pd.read_csv("19-CG001–Holdings.csv")
      .assign(Ticker = lambda d: d["Ticker"].str.rsplit(":", n = 1).str[-1])
)


symbols = [s for s in df["Ticker"].dropna().unique().tolist() if s not in exclusions]

# sector lookup
datasi = Ticker(symbols, asynchronous=True).get_modules(modules)
dfsi = pd.DataFrame.from_dict(datasi, orient="index")

lookup = (
    pd.concat(
        [
            pd.json_normalize([x for x in dfsi[m] if isinstance(x, dict)])
            for m in modules
            if m in dfsi.columns
        ],
        axis=1,
    )
    .set_index("symbol")[["industry", "sector"]]
    .reindex(symbols)
)


kind_map = {**{k: "Cash" for k in cash}, **{k: "ETF" for k in etf}}

df = (
    df.join(lookup, on="Ticker")
      .assign(
          sector=lambda d: np.where(
              d["Ticker"].isin(kind_map), d["Ticker"].map(kind_map), d["sector"]
          )
      )
      .rename(columns={"sector": "Sector", "industry": "Industry"})
)

# 下載（排除現金）
tickers = sorted(df.loc[~df["Ticker"].isin(cash), "Ticker"].dropna().unique().tolist())
data = yf.download(
    tickers,
    start="2025-06-01",
    end="2025-08-01",
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=True,
)

portfolio = data['Close'].copy()

shares = (
    df.loc[~df["Ticker"].isin(cash), ["Ticker", "Shares"]]
      .dropna(subset=["Ticker"])
      .assign(Shares=lambda d: pd.to_numeric(d["Shares"], errors="coerce"))
      .groupby("Ticker", as_index=False)["Shares"].sum() 
      .sort_values("Ticker")
)


shares_series = (
    shares.set_index("Ticker")["Shares"]
          .reindex(portfolio.columns)
          .fillna(0)
)



position_values = portfolio.mul(shares_series, axis=1)
position_diff = position_values.diff().dropna()        # ΔV_i,t
portfolio_diff = position_diff.sum(axis=1)             # ΔV_p,t

cov_mat = position_diff.cov()                          # Cov(ΔV)
cov_with_pf = cov_mat.sum(axis=1)                      # Cov(ΔV_i, Σ_j ΔV_j)
var_p = portfolio_diff.var(ddof=1)                     # Var(ΔV_p)
pct_contrib = (cov_with_pf / var_p).sort_values(ascending = False)  # sum = 1



# define constants for things we can get from yfinance
factor_tickers = ["SIZE", "VLUE", "MTUM", "QUAL"]
market_ticker = "CSUS.L"
return_history = "1y"
return_interval = "1d"

# things we cannot pull from yfinance
# benchmark_data_loc = "Russell 2500 Growth Index (20250220000000000 _ 20210104000000000).csv"
benchmark_ticker = "R25IG"

portfolio_data_loc = "19-CG001–Holdings.csv"

etf_factor_data = yf.download(factor_tickers, period=return_history,
                              interval=return_interval,
                              progress=True, auto_adjust=True)

etf_factor_returns = etf_factor_data["Close"].pct_change().fillna(0)

market_data = yf.download([market_ticker], period=return_history, interval=return_interval, 
                          progress=True, auto_adjust=True)
# get price returns for market
# note the dangerous ffill here
market_returns = market_data['Close'].ffill().pct_change().fillna(0)

# # read and reformat benchmark returns
# benchmark_data = pd.read_csv(benchmark_data_loc, thousands=',')
# # convert Date and Close columns to proper types
# benchmark_data['Date'] = pd.to_datetime(benchmark_data['Date'])
# benchmark_data['Close'] = benchmark_data['Close'].astype(float)
# # housekeeping to align with yfinance data format
# benchmark_data = benchmark_data.rename(columns={'Close':benchmark_ticker}).set_index('Date')
# benchmark_returns = benchmark_data.pct_change().fillna(0)

portfolio_data = pd.read_csv(portfolio_data_loc)
# reformat ticker to remove exchange prefix
portfolio_data['Ticker'] = portfolio_data['Ticker'].apply(lambda x: x[x.find(':')+1:])
# rename weight column so it's easier to work with
portfolio_data = portfolio_data.rename(columns={'Weight (%)':'Weight'})

# mutual fund holdings can't be pulled in yfinance, so drop and note discrepancy
special_cases = ['FAIX.X','FGVX.X']
portfolio_data = portfolio_data[~portfolio_data['Ticker'].isin(special_cases)]

port_constituent_data = yf.download(list(portfolio_data['Ticker'].unique()), 
                                    period=return_history, interval=return_interval, 
                                    progress=True, auto_adjust=True)
# forward fill N/A values
port_constituent_data['Close'] = port_constituent_data['Close'].ffill()
port_constituent_returns = port_constituent_data['Close'].pct_change().fillna(0)

all_returns = pd.concat([etf_factor_returns, market_returns, port_constituent_returns
                        #  , benchmark_returns
                         ], axis=1)
# benchmark series is longer, so only take shortest series
all_returns = all_returns.dropna(subset=factor_tickers)
all_returns = all_returns.fillna(0) # dangerous

# compute portfolio returns
port_individual_returns = all_returns[portfolio_data['Ticker'].unique()].T.sort_index()
port_weights = portfolio_data.set_index('Ticker')['Weight'].sort_index() / 100
port_returns = pd.Series(port_individual_returns.values.T @ port_weights, index=port_individual_returns.columns)
all_returns['PORT'] = port_returns

# compute each column's exposures
factor_columns = [market_ticker] + factor_tickers
X = all_returns[factor_columns].values

result_dict = {}
for col in all_returns.columns:
    if col not in factor_columns:
        y = all_returns[col].values
        model = LinearRegression(fit_intercept=True, copy_X=True)
        model.fit(X, y)
        result_dict[col] = np.concatenate([np.array([model.intercept_]), model.coef_])

exposures = pd.DataFrame(result_dict, index=['Alpha', 'Beta'] + factor_tickers)

# ensure linearity (this should match PORT column above)
port_exposures = exposures[portfolio_data['Ticker'].unique()].T.sort_index()
port_exposures.index.name = 'Ticker'
port_weights = portfolio_data.set_index('Ticker')['Weight'].sort_index() / 100
port_exposure = pd.Series(port_exposures.values.T @ port_weights, index=port_exposures.columns)



metrics = {
    "Alpha (ann.)": f"{port_exposure['Alpha']:.4f}",
    "Beta": f"{port_exposure['Beta']:.4f}",
    "SIZE": f"{port_exposure['SIZE']:.4f}",
    "VALUE": f"{port_exposure['VLUE']:.4f}",
    "MTUM": f"{port_exposure['MTUM']:.4f}",
    "QUAL": f"{port_exposure['QUAL']:.4f}"
}


# ===== Alpha  Beta  SIZE  VALUE  MTUM  QUAL =====
st.markdown("### Factor Exposures")
f1, f2, f3, f4, f5, f6 = st.columns(6, gap="small")
with f1:
    st.metric("Alpha (ann.)", metrics["Alpha (ann.)"])
with f2:
    st.metric("Beta", metrics["Beta"])
with f3:
    st.metric("SIZE", metrics["SIZE"])
with f4:
    st.metric("VALUE", metrics["VALUE"])
with f5:
    st.metric("MTUM", metrics["MTUM"])
with f6:
    st.metric("QUAL", metrics["QUAL"])

st.divider()


st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 1.4rem; }
div[data-testid="stMetricLabel"] { font-weight: 600; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)



col = st.columns((4.5, 2.5), gap = 'medium')

############
# Tree Map #
############

with col[0]:
    fig_tree = px.treemap(df, path = ['Sector', 'Ticker'], values = 'Weight (%)')
    fig_tree.update_traces(marker = dict(cornerradius = 10))

    fig_tree.update_traces(
        hovertemplate="<b>%{label}</b><br>" +
                "Weight: %{value:.2f}%<br>" 
    )

    st.plotly_chart(fig_tree, use_container_width = True)

#############
# Pie Chart #
#############
with col[1]:
    sector_data = df.groupby('Sector')['Weight (%)'].sum().reset_index()

    fig_pie = px.pie(
        sector_data,
        names = 'Sector',
        values = 'Weight (%)',
        hole = 0.4
    )


    fig_pie.update_traces   (
        hovertemplate = "<b>%{label}</b><br>Weight: %{value:.2f}%<br>",
        textinfo = 'percent+label'
                            )           

    fig_pie.update_layout(showlegend = False)

    st.plotly_chart(fig_pie, use_container_width = True)

st.divider()

rank = st.columns((1, 1), gap = 'large')

###################
# Top 10 Holdings #
###################

with rank[0]:
    st.markdown("### Holdings")

    df_sorted = (
        df[['Ticker', 'Weight (%)']]
        .dropna()
        .sort_values('Weight (%)', ascending=False)
        .reset_index(drop=True)
    )


    st.dataframe(
    df_sorted,
    column_order=("Ticker", "Weight (%)"),
    hide_index=True,
    # use_container_width=True,
    width='stretch',
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Weight (%)": st.column_config.ProgressColumn(
            "Weight (%)",
            format="%.2f%%",
            min_value=0.0,
            max_value = float(df_sorted["Weight (%)"].max())
        )
    }
    )

with rank[1]:
    # Euler
    df_contrib = (
        pct_contrib.rename("Contribution_Ratio")
                .to_frame()
                .reset_index()
                .rename(columns={"index": "Ticker"})
    )


    df_contrib["Contribution (%)"] = df_contrib["Contribution_Ratio"] * 100.0


    df_contrib = df_contrib.sort_values("Contribution (%)", ascending=False).reset_index(drop=True)



    st.markdown("### Euler Risk Contribution")
    st.dataframe(
        df_contrib[["Ticker", "Contribution (%)"]],
        column_order=("Ticker", "Contribution (%)"),
        hide_index=True,
        width='stretch',
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Contribution (%)": st.column_config.ProgressColumn(
                "Contribution (%)",
                format="%.2f%%",
                min_value=float(df_contrib["Contribution (%)"].min()),
                max_value=float(df_contrib["Contribution (%)"].max()),
            )
        }
    )

st.divider()

#######################################
############### Heatmap ###############
#######################################


# --- log returns corr ---
close_df = data["Close"]
log_ret = np.log(close_df / close_df.shift(1))
corr_mat = log_ret.corr()

# --- Ticker → Sector  ---
map_df = (
    df.loc[:, ["Ticker", "Sector"]]
      .dropna(subset=["Ticker"])
      .drop_duplicates()
)
map_df = map_df[map_df["Ticker"].isin(corr_mat.columns)]

# --- sort by sector > ticker ---
order = (
    map_df.sort_values(["Sector", "Ticker"])
          .loc[:, "Ticker"].tolist()
)
corr_ord = corr_mat.reindex(index=order, columns=order)

# --- Sector mapping ---
ordered_map = map_df.set_index("Ticker").loc[order].reset_index()
sizes = ordered_map.groupby("Sector")["Ticker"].size().tolist()
labels = ordered_map.groupby("Sector").size().index.tolist()

starts = []
acc = 0
for s in sizes:
    starts.append(acc)
    acc += s
centers_y = [start + size / 2 - 0.5 for start, size in zip(starts, sizes)]

# --- lower triangel without diagonal ---
mask = np.tril(np.ones_like(corr_ord, dtype=bool), k=-1)
corr_lower = corr_ord.where(mask)

# --- actual range ---
zmin = np.nanmin(corr_lower.values)
zmax = np.nanmax(corr_lower.values)

# ---  Heatmap---
heatmap = px.imshow(
    corr_lower,
    color_continuous_scale="RdBu_r",
    zmin=zmin,
    zmax=zmax,
    aspect="equal",
    labels=dict(color="Corr"),
    title="Portfolio Correlation"
)

# --- transparant background ---
heatmap.update_layout(
    width=700, height=780,
    xaxis_title="", yaxis_title="",
    xaxis=dict(tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    margin=dict(l=150, r=20, t=100, b=40),
    coloraxis_colorbar=dict(x=0.8, xanchor="left", xpad=0, thickness=16, len=0.9),
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)" 
)

heatmap.update_xaxes(showgrid=False, zeroline=False)
heatmap.update_yaxes(showgrid=False, zeroline=False)

# --- Sector  label ---
for lab, cy in zip(labels, centers_y):
    heatmap.add_annotation(
        x=-0.25,
        y=cy,
        xref="x domain",
        yref="y",
        text=f"<b>{lab}</b>",
        showarrow=False,
        align="right",
        textangle=0,
        font=dict(size=11, color="white"),
        bgcolor="rgba(0,0,0,0.45)",
        borderpad=1
    )


n = len(corr_ord)
for start, size in zip(starts, sizes):
    end_idx = start + size  

    heatmap.add_shape(
        type="line",
        x0=-0.5, x1=end_idx - 0.5,
        y0=end_idx - 0.5, y1=end_idx - 0.5,
        xref="x", yref="y",
        line=dict(color="rgba(100,100,100,0.5)", width=1.1, dash="dot")
    )

    heatmap.add_shape(
        type="line",
        x0=end_idx - 0.5, x1=end_idx - 0.5,
        y0=end_idx - 0.5, y1=n - 0.5,
        xref="x", yref="y",
        line=dict(color="rgba(100,100,100,0.5)", width=1.1, dash="dot")
    )



# --- Hover ---
heatmap.update_traces(
    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corr = %{z:.3f}<extra></extra>"
)

# --- show on Streamlit ---
st.plotly_chart(heatmap, use_container_width=True)

