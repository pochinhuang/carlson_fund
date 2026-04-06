import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

# from yahooquery import Ticker
from sklearn.linear_model import LinearRegression


st.set_page_config(
    page_title = "Portfolio Dashboard",
    page_icon = ":material/dashboard:",
    layout = "wide",                
    initial_sidebar_state = "expanded" 
)

# read + organize Ticker
df = pd.read_csv("data/19_CG001_Holdings_clean.csv")

tickers = df["Ticker"].tolist()

return_history = "1y"
return_interval = "1d"

data = yf.download(tickers, period=return_history,
                              interval=return_interval,
                              progress=True, auto_adjust=True)


portfolio = data['Close'].copy()

ret = np.round(((portfolio.iloc[-1] / portfolio.iloc[-3]) - 1)*100, 2)
df['Return (%)'] = df['Ticker'].map(ret).fillna(0)

shares = (
    df.loc[:, ["Ticker", "Shares"]]
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


# =========================
# Constants
# =========================
factor_tickers = ["SIZE", "VLUE", "MTUM", "QUAL"]
market_ticker = "CSUS.L"
return_history = "1y"
return_interval = "1d"
portfolio_data_loc = "data/19_CG001_Holdings_clean.csv"

# =========================
# Download factor and market returns
# =========================
etf_factor_data = yf.download(
    factor_tickers,
    period=return_history,
    interval=return_interval,
    progress=True,
    auto_adjust=True
)
etf_factor_returns = etf_factor_data["Close"].pct_change().fillna(0)

market_data = yf.download(
    [market_ticker],
    period=return_history,
    interval=return_interval,
    progress=True,
    auto_adjust=True
)
market_returns = market_data["Close"].ffill().pct_change().fillna(0)

# =========================
# Load portfolio holdings
# =========================
portfolio_data = pd.read_csv(portfolio_data_loc).rename(columns={"Weight (%)": "Weight"})
tickers = portfolio_data["Ticker"].unique()

# =========================
# Constituent returns
# =========================
port_constituent_data = data.copy()
port_constituent_data["Close"] = port_constituent_data["Close"].ffill()
port_constituent_returns = port_constituent_data["Close"].pct_change().fillna(0)

# =========================
# Combine all returns
# =========================
all_returns = pd.concat(
    [etf_factor_returns, market_returns, port_constituent_returns],
    axis=1,
    sort=False
)

all_returns = all_returns.dropna(subset=factor_tickers).fillna(0)

# =========================
# Compute portfolio returns
# =========================
port_weights = portfolio_data.set_index("Ticker")["Weight"].sort_index() / 100
port_individual_returns = all_returns[tickers].T.sort_index()

port_returns = pd.Series(
    port_individual_returns.values.T @ port_weights.values,
    index=port_individual_returns.columns,
    name="PORT"
)

all_returns["PORT"] = port_returns

# =========================
# Factor regression
# =========================
factor_columns = [market_ticker] + factor_tickers
target_columns = [col for col in all_returns.columns if col not in factor_columns]

X = all_returns[factor_columns].values
result_dict = {}

for col in target_columns:
    y = all_returns[col].values
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    result_dict[col] = np.concatenate([[model.intercept_], model.coef_])

exposures = pd.DataFrame(
    result_dict,
    index=["Alpha", "Beta"] + factor_tickers
)

# =========================
# Portfolio factor exposure
# =========================
port_exposures = exposures[tickers].T.sort_index()
port_exposures.index.name = "Ticker"

port_exposure = pd.Series(
    port_exposures.values.T @ port_weights.values,
    index=port_exposures.columns
)

# =========================
# Display metrics
# =========================
metrics = {
    "Alpha": f"{port_exposure['Alpha']:.4f}",
    "Beta": f"{port_exposure['Beta']:.4f}",
    "SIZE": f"{port_exposure['SIZE']:.4f}",
    "VALUE": f"{port_exposure['VLUE']:.4f}",
    "MTUM": f"{port_exposure['MTUM']:.4f}",
    "QUAL": f"{port_exposure['QUAL']:.4f}"
}

st.markdown("### Factor Exposures")
f1, f2, f3, f4, f5, f6 = st.columns(6, gap="small")

with f1:
    st.metric("Alpha", metrics["Alpha"])
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

st.markdown("### Holdings")

fig_tree = px.treemap(
    df,
    path=['Sector', 'Ticker'],
    values='Weight (%)',
    color='Return (%)',
    color_continuous_scale=[
        [0.0, "#ff1f1f"],
        [0.5, "#1a1a1a"],
        [1.0, "#00ff00"]
    ],
    color_continuous_midpoint=0,
    custom_data=['Ticker', 'Return (%)', 'Weight (%)']
)

fig_tree.update_traces(
    marker=dict(cornerradius=0),
    texttemplate="<b>%{label}</b><br>%{customdata[1]:.2f}%",
    textfont=dict(color="white", size=16),
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "Weight: %{customdata[2]:.2f}%<extra></extra>"
    )
)

fig_tree.update_layout(
    margin=dict(t=10, l=10, r=10, b=10),
    coloraxis_showscale=False,
    height=650,
)

st.plotly_chart(fig_tree, width='stretch', config={"displayModeBar": False})

st.divider()


rank = st.columns((1, 1), gap='large')

# =========================
# 1. 個股層級資料整理
# =========================
asset_ret = data['Close'].pct_change().dropna()
vol = asset_ret.std() * np.sqrt(252) * 100

df_contrib = (
    pct_contrib.rename("Contribution_Ratio")
    .mul(100)
    .rename("Contribution (%)")
    .reset_index()
    .rename(columns={"index": "Ticker"})
)

df_plot = df.copy()
df_plot["Volatility (%)"] = df_plot["Ticker"].map(vol)
df_plot = df_plot.merge(df_contrib, on="Ticker", how="left")

df_plot["Volatility (%)"] = df_plot["Volatility (%)"].fillna(0)
df_plot["Contribution (%)"] = df_plot["Contribution (%)"].fillna(0)

# =========================
# 2. sector 層級聚合
# =========================
df_sector = (
    df_plot.groupby("Sector")
    .apply(lambda x: pd.Series({
        "Weight (%)": x["Weight (%)"].sum(),
        "Return (%)": (x["Weight (%)"] * x["Return (%)"]).sum() / x["Weight (%)"].sum()
                      if x["Weight (%)"].sum() != 0 else 0,
        "Volatility (%)": (x["Weight (%)"] * x["Volatility (%)"]).sum() / x["Weight (%)"].sum()
                          if x["Weight (%)"].sum() != 0 else 0,
        "Contribution (%)": x["Contribution (%)"].sum(),
        "Holdings": x["Ticker"].nunique()
    }))
    .reset_index()
)

df_sector["Abs Contribution (%)"] = df_sector["Contribution (%)"].abs()
df_sector["Bubble Size"] = np.sqrt(df_sector["Abs Contribution (%)"])
# =========================
# 3. 左邊：Sector Scatter
# =========================


with rank[0]:
    st.markdown("### Sector Risk Map")
    fig_scatter = px.scatter(
        df_sector,
        x="Weight (%)",
        y="Volatility (%)",
        size="Bubble Size",
        color="Return (%)",
        custom_data=["Sector", "Weight (%)", "Volatility (%)", "Return (%)", "Contribution (%)", "Holdings"],
        color_continuous_scale=[
            [0.0, "#ff1f1f"],
            [0.5, "#1a1a1a"],
            [1.0, "#00ff00"]
        ],
        color_continuous_midpoint=0
    )

    fig_scatter.update_traces(
        marker=dict(
            line=dict(width=1, color="white"),
            opacity=0.9
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Weight: %{customdata[1]:.2f}%<br>"
            "Volatility: %{customdata[2]:.2f}%<br>"
            "Return: %{customdata[3]:.2f}%<br>"
            "Risk Contribution: %{customdata[4]:.2f}%<br>"
            "Holdings: %{customdata[5]}<extra></extra>"
        )
    )

    fig_scatter.update_layout(
        height=500,
        paper_bgcolor="#2a2a2a",
        plot_bgcolor="#2a2a2a",
        font=dict(color="white"),
        coloraxis_showscale=False
    )

    st.plotly_chart(fig_scatter, width="stretch", config={"displayModeBar": False})

# =========================
# 4. 右邊：Euler Risk Contribution
# =========================
with rank[1]:
    df_contrib_show = df_contrib.sort_values("Contribution (%)", ascending=False).reset_index(drop=True)

    st.markdown("### Euler Risk Contribution")
    st.dataframe(
        df_contrib_show[["Ticker", "Contribution (%)"]],
        column_order=("Ticker", "Contribution (%)"),
        hide_index=True,
        width="stretch",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Contribution (%)": st.column_config.ProgressColumn(
                "Contribution (%)",
                format="%.2f%%",
                min_value=float(df_contrib_show["Contribution (%)"].min()),
                max_value=float(df_contrib_show["Contribution (%)"].max()),
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
zabs = np.nanmax(np.abs(corr_lower.values))
zmin, zmax = -zabs, zabs

# ---  Heatmap---
st.markdown("### Portfolio Correlation")

heatmap = px.imshow(
    corr_lower,
    color_continuous_scale="RdBu_r",
    zmin=zmin,
    zmax=zmax,
    aspect="equal",
    labels=dict(color="Corr"),
    # title="Portfolio Correlation"
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
        # bgcolor="rgba(0,0,0,0.45)",
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
# st.plotly_chart(heatmap, use_container_width=True)
st.plotly_chart(heatmap, width='stretch')
