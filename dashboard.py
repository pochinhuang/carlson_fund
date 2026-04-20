import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from html import escape

st.set_page_config(
    page_title="Portfolio Dashboard",
    page_icon=":material/dashboard:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0c0f;
    --surface: #111418;
    --card: #141820;
    --border: #1e2530;
    --text: #e8edf5;
    --muted: #7f8b99;
    --accent: #00e5a0;
    --accent2: #4a9eff;
    --accent3: #ff8a4c;
    --accent4: #c084fc;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

div[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #090b0e 0%, #0d1117 100%);
}

div[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stAppToolbar"],
div[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapsedControl"],
header {
    background: rgba(0,0,0,0) !important;
}

button[kind="header"] {
    background: transparent !important;
    border: none !important;
}

/* container 保留，但弱化 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px;
    padding: 20px 20px 16px 20px;
    margin-bottom: 22px;
    box-shadow: none !important;
}

/* plotly / dataframe 不額外加框 */
div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}

div[data-testid="stDataFrame"] * {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Hero 改得跟 docs 一樣乾淨 */
.main-hero {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    margin-bottom: 24px;
    overflow: visible;
}

.hero-tag {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent);
    font-size: 11px;
    letter-spacing: 2.6px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    color: var(--text);
    font-size: clamp(54px, 8vw, 96px);
    line-height: 0.92;
    letter-spacing: 2px;
    margin: 0;
}

.hero-title span {
    color: var(--accent);
    display: block;
}

.hero-sub {
    margin-top: 16px;
    max-width: 760px;
    color: #9aa7b5;
    font-size: 14px;
    line-height: 1.8;
    font-family: 'IBM Plex Mono', monospace;
}

/* 保留 chip，但更淡 */
.ticker-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}

.ticker-chip {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 10px 14px;
    min-width: 120px;
}

.ticker-chip .k {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.7px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 4px;
}

.ticker-chip .v {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    line-height: 1.1;
    color: var(--text);
    letter-spacing: 0.5px;
    font-weight: 500;
}

/* Section 保留，但弱化 */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.3px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.section-label.green { color: var(--accent); }
.section-label.blue { color: var(--accent2); }
.section-label.orange { color: var(--accent3); }
.section-label.purple { color: var(--accent4); }

.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 42px;
    line-height: 0.98;
    letter-spacing: 1.5px;
    color: var(--text);
    margin-bottom: 10px;
}

.section-title span {
    color: var(--accent);
}

.section-sub {
    color: #95a3b3;
    font-size: 14px;
    line-height: 1.75;
    margin-bottom: 6px;
}

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.12), rgba(0,0,0,0));
    margin: 10px 0 18px 0;
    border-radius: 999px;
}

.mini-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
}

.small-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
}

/* Factor cards 弱化 */
.factor-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(120px, 1fr));
    gap: 14px;
    margin-top: 18px;
}

.factor-card {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 14px 14px 12px 14px;
    min-height: 92px;
    position: relative;
    overflow: hidden;
}

.factor-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 3px;
    height: 100%;
}

.factor-card.alpha::before { background: #00e5a0; }
.factor-card.beta::before  { background: #4a9eff; }
.factor-card.size::before  { background: #c084fc; }
.factor-card.value::before { background: #ff8a4c; }
.factor-card.mtum::before  { background: #ffd166; }
.factor-card.qual::before  { background: #ff5d8f; }

.factor-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.factor-name.alpha { color: #00e5a0; }
.factor-name.beta  { color: #4a9eff; }
.factor-name.size  { color: #c084fc; }
.factor-name.value { color: #ff8a4c; }
.factor-name.mtum  { color: #ffd166; }
.factor-name.qual  { color: #ff5d8f; }

.factor-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    line-height: 1.1;
    letter-spacing: 0.4px;
    color: #ffffff;
    font-weight: 500;
}

/* Heatmap */
.hm-wrapper {
    background: transparent;
    border: none;
    padding: 0;
    overflow-x: auto;
    color: #e8edf5;
    font-family: 'IBM Plex Sans', sans-serif;
}

.hm-legend {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #7f8b99;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

.hm-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
}

.hm-swatch {
    width: 9px;
    height: 9px;
    border-radius: 2px;
    border: 1px solid rgba(255,255,255,0.06);
}

.hm-layout {
    display: grid;
    grid-template-columns: 68px auto;
    grid-template-rows: auto auto;
    gap: 4px 6px;
    min-width: fit-content;
}

.hm-top-left {
    height: 18px;
}

.hm-x-axis {
    display: grid;
    gap: 2px;
    padding-left: 0px;
}

.hm-x-tick {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #9aa7b5;
    transform: rotate(-45deg);
    transform-origin: center;
    white-space: nowrap;
    height: 20px;
    display: flex;
    align-items: flex-end;
    justify-content: center;
}

.hm-left-block {
    position: relative;
}

.hm-y-axis {
    display: grid;
    gap: 2px;
}

.hm-y-tick {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #9aa7b5;
    white-space: nowrap;
}

.hm-matrix-wrap {
    position: relative;
    width: fit-content;
}

.hm-matrix {
    display: grid;
    gap: 2px;
    position: relative;
}

.hm-cell {
    border-radius: 2px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 8px;
    font-weight: 600;
    transition: transform 0.12s ease;
    cursor: default;
    user-select: none;
    line-height: 1;
}

.hm-cell:hover {
    transform: scale(1.04);
}

.hm-cell.very_high {
    background: rgba(0, 229, 160, 0.92);
    color: #000000;
}

.hm-cell.high {
    background: rgba(93, 173, 226, 0.82);
    color: #f4f8fc;
}

.hm-cell.mid {
    background: rgba(90, 98, 112, 0.80);
    color: #f4f8fc;
}

.hm-cell.low {
    background: rgba(255, 138, 76, 0.78);
    color: #ffffff;
}

.hm-cell.very_low {
    background: rgba(220, 53, 69, 0.88);
    color: #ffffff;
}

.hm-cell.empty {
    background: transparent;
    color: transparent;
}

.hm-hline {
    position: absolute;
    left: 0;
    right: 0;
    height: 1px;
    border-top: 1px dotted rgba(255,255,255,0.12);
    pointer-events: none;
}

.hm-vline {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 1px;
    border-left: 1px dotted rgba(255,255,255,0.12);
    pointer-events: none;
}

.hm-note {
    margin-top: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #7f8b99;
}

@media (max-width: 1200px) {
    .factor-grid {
        grid-template-columns: repeat(3, minmax(120px, 1fr));
    }
}

@media (max-width: 700px) {
    .factor-grid {
        grid-template-columns: repeat(2, minmax(120px, 1fr));
    }

    .hero-title {
        font-size: 58px;
    }

    .section-title {
        font-size: 34px;
    }
}

div[data-testid="stDataFrame"] * {
    font-family: 'IBM Plex Mono', monospace !important;
}

.js-plotly-plot * {
    font-family: 'IBM Plex Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Helper
# =========================
def section_header(title_text: str, title_accent: str, subtitle: str):
    st.markdown(
        f"""
        <div class="section-title">{title_text} <span>{title_accent}</span></div>
        <div class="section-sub">{subtitle}</div>
        <div class="section-divider"></div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Read data
# =========================
df = pd.read_csv("data/19_CG001_Holdings_clean.csv")
portfolio = pd.read_csv("data/portfolio.csv")

ret = np.round(((portfolio.iloc[-1] / portfolio.iloc[-3]) - 1) * 100, 2)
df["Return (%)"] = df["Ticker"].map(ret).fillna(0)

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
position_diff = position_values.diff().dropna()
portfolio_diff = position_diff.sum(axis=1)

cov_mat = position_diff.cov()
cov_with_pf = cov_mat.sum(axis=1)
var_p = portfolio_diff.var(ddof=1)
pct_contrib = (cov_with_pf / var_p).sort_values(ascending=False)

# =========================
# Factor model data
# =========================
factor_tickers = ["SIZE", "VLUE", "MTUM", "QUAL"]
market_ticker = "CSUS.L"

etf_factor_data = pd.read_csv("data/etf_factor_data.csv")
etf_factor_returns = etf_factor_data.pct_change().fillna(0)

market_data = pd.read_csv("data/market_data.csv")
market_returns = market_data.ffill().pct_change().fillna(0)

portfolio_data = df.copy().rename(columns={"Weight (%)": "Weight"})
tickers = portfolio_data["Ticker"].unique()

port_constituent_returns = portfolio.copy().ffill().pct_change().fillna(0)

all_returns = pd.concat(
    [etf_factor_returns, market_returns, port_constituent_returns],
    axis=1,
    sort=False
)

all_returns = all_returns.dropna(subset=factor_tickers).fillna(0)

port_weights = portfolio_data.set_index("Ticker")["Weight"].sort_index() / 100
port_individual_returns = all_returns[tickers].T.sort_index()

port_returns = pd.Series(
    port_individual_returns.values.T @ port_weights.values,
    index=port_individual_returns.columns,
    name="PORT"
)

all_returns["PORT"] = port_returns

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

port_exposures = exposures[tickers].T.sort_index()
port_exposures.index.name = "Ticker"

port_exposure = pd.Series(
    port_exposures.values.T @ port_weights.values,
    index=port_exposures.columns
)

metrics = {
    "Alpha": f"{port_exposure['Alpha']:.4f}",
    "Beta": f"{port_exposure['Beta']:.4f}",
    "SIZE": f"{port_exposure['SIZE']:.4f}",
    "VALUE": f"{port_exposure['VLUE']:.4f}",
    "MTUM": f"{port_exposure['MTUM']:.4f}",
    "QUAL": f"{port_exposure['QUAL']:.4f}"
}

# =========================
# Hero
# =========================

st.markdown(f"""
<div class="main-hero">
    <div class="hero-title">PORTFOLIO <span>DASHBOARD</span></div>
    <div class="hero-sub">
        Analytics view of portfolio composition, factor exposure, sector risk, 
        and dependencies across holdings.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Factor Exposures
# =========================
with st.container(border=True):
    section_header(
        "FACTOR",
        "EXPOSURES",
        "Portfolio-level exposure estimates aggregated from constituent regressions against market and style factors."
    )

    st.html(f"""
    <div class="factor-grid">
        <div class="factor-card alpha">
            <div class="factor-name alpha">Alpha</div>
            <div class="factor-value">{metrics["Alpha"]}</div>
        </div>
        <div class="factor-card beta">
            <div class="factor-name beta">Beta</div>
            <div class="factor-value">{metrics["Beta"]}</div>
        </div>
        <div class="factor-card size">
            <div class="factor-name size">SIZE</div>
            <div class="factor-value">{metrics["SIZE"]}</div>
        </div>
        <div class="factor-card value">
            <div class="factor-name value">VALUE</div>
            <div class="factor-value">{metrics["VALUE"]}</div>
        </div>
        <div class="factor-card mtum">
            <div class="factor-name mtum">MTUM</div>
            <div class="factor-value">{metrics["MTUM"]}</div>
        </div>
        <div class="factor-card qual">
            <div class="factor-name qual">QUAL</div>
            <div class="factor-value">{metrics["QUAL"]}</div>
        </div>
    </div>
    <div class="small-note" style="margin-top:14px;">
        Weighted portfolio exposure summary across market, size, value, momentum, and quality factors.
    </div>
    """)

# =========================
# Holdings Treemap
# =========================
with st.container(border=True):
    section_header(
        "HOLDINGS",
        "MAP",
        "Treemap of portfolio weights by sector and ticker, colored by recent return."
    )

    fig_tree = px.treemap(
        df,
        path=["Sector", "Ticker"],
        values="Weight (%)",
        color="Return (%)",
        color_continuous_scale=[
            [0.0, "#ff1f1f"],
            [0.5, "#1a1a1a"],
            [1.0, "#00ff00"]
        ],
        color_continuous_midpoint=0,
        custom_data=["Ticker", "Return (%)", "Weight (%)"]
    )

    fig_tree.update_traces(
        marker=dict(
            cornerradius=0,
            line=dict(width=1, color="rgba(255,255,255,0.08)")
        ),
        texttemplate="<b>%{label}</b><br>%{customdata[1]:.2f}%",
        textfont=dict(
            color="white",
            size=14,
            family="IBM Plex Mono"
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Weight: %{customdata[2]:.2f}%<br>"
            "Return: %{customdata[1]:.2f}%<extra></extra>"
        )
    )

    fig_tree.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        coloraxis_showscale=False,
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white",
            family="IBM Plex Mono",
            size=12
        )
    )

    st.plotly_chart(fig_tree, width="stretch", config={"displayModeBar": False})

# =========================
# Risk Map / Contribution
# =========================
asset_ret = portfolio.pct_change().dropna()
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

with st.container(border=True):
    section_header(
        "RISK",
        "MAP",
        "Sector positioning with volatility, return, and Euler risk contribution concentration shown side by side."
    )

    left_col, right_col = st.columns((1.25, 1), gap="large")

    with left_col:
        st.markdown('<div class="mini-title">Sector Risk Scatter</div>', unsafe_allow_html=True)

        fig_scatter = px.scatter(
            df_sector,
            x="Weight (%)",
            y="Volatility (%)",
            size="Bubble Size",
            color="Return (%)",
            custom_data=["Sector", "Weight (%)", "Volatility (%)", "Return (%)", "Contribution (%)", "Holdings"],
            color_continuous_scale=[
                [0.0, "#ff6b35"],
                [0.5, "#17202a"],
                [1.0, "#00e5a0"]
            ],
            color_continuous_midpoint=0
        )

        fig_scatter.update_traces(
            marker=dict(
                line=dict(width=1, color="rgba(255,255,255,0.75)"),
                opacity=0.92
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
            height=520,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f141b",
            font=dict(color="white", family="IBM Plex Mono", size=11),
            coloraxis_showscale=False,
            xaxis=dict(
                title="Weight (%)",
                title_font=dict(family="IBM Plex Mono", size=11, color="white"),
                tickfont=dict(family="IBM Plex Mono", size=10, color="white"),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False
            ),
            yaxis=dict(
                title="Volatility (%)",
                title_font=dict(family="IBM Plex Mono", size=11, color="white"),
                tickfont=dict(family="IBM Plex Mono", size=10, color="white"),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False
            ),
            margin=dict(t=10, l=10, r=10, b=10)
        )

        st.plotly_chart(fig_scatter, width="stretch", config={"displayModeBar": False})

    with right_col:
        st.markdown('<div class="mini-title">Euler Risk Contribution</div>', unsafe_allow_html=True)

        df_contrib_show = df_contrib.sort_values("Contribution (%)", ascending=False).reset_index(drop=True)

        st.dataframe(
            df_contrib_show[["Ticker", "Contribution (%)"]],
            column_order=("Ticker", "Contribution (%)"),
            hide_index=True,
            width="stretch",
            height=520,
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

# =========================
# Correlation Heatmap
# =========================
CELL = 23
GAP = 2
STEP = CELL + GAP

close_df = portfolio.copy()
log_ret = np.log(close_df / close_df.shift(1))
corr_mat = log_ret.corr()

map_df = (
    df.loc[:, ["Ticker", "Sector"]]
      .dropna(subset=["Ticker"])
      .drop_duplicates()
)
map_df = map_df[map_df["Ticker"].isin(corr_mat.columns)]

order = (
    map_df.sort_values(["Sector", "Ticker"])
          .loc[:, "Ticker"].tolist()
)
corr_ord = corr_mat.reindex(index=order, columns=order)

ordered_map = map_df.set_index("Ticker").loc[order].reset_index()
sizes = ordered_map.groupby("Sector")["Ticker"].size().tolist()

starts = []
acc = 0
for s in sizes:
    starts.append(acc)
    acc += s

mask = np.tril(np.ones_like(corr_ord, dtype=bool), k=-1)
corr_lower = corr_ord.where(mask)

def corr_class(v):
    if pd.isna(v):
        return "empty"
    elif v >= 0.7:
        return "very_high"
    elif v >= 0.3:
        return "high"
    elif v >= 0:
        return "mid"
    elif v >= -0.2:
        return "low"
    else:
        return "very_low"

n = len(order)
matrix_cells = []

for i, row_ticker in enumerate(order):
    for j, col_ticker in enumerate(order):
        v = corr_lower.iloc[i, j]

        if pd.isna(v):
            matrix_cells.append('<div class="hm-cell empty"></div>')
        else:
            cls = corr_class(v)
            tooltip = f"{row_ticker} vs {col_ticker} | Corr = {v:.2f}"
            text = f"{v:.2f}"
            matrix_cells.append(
                f'<div class="hm-cell {cls}" title="{escape(tooltip)}">{text}</div>'
            )

x_ticks_html = "".join(
    [f'<div class="hm-x-tick" style="width:{CELL}px;">{escape(t)}</div>' for t in order]
)

y_ticks_html = "".join(
    [f'<div class="hm-y-tick" style="height:{CELL}px;">{escape(t)}</div>' for t in order]
)

sector_lines_html = ""
for start, size in zip(starts, sizes):
    end_idx = start + size
    if end_idx < n:
        pos = end_idx * STEP - 1
        sector_lines_html += f'<div class="hm-hline" style="top:{pos}px;"></div>'
        sector_lines_html += f'<div class="hm-vline" style="left:{pos}px;"></div>'

heatmap_html = f"""
<div class="hm-wrapper">
    <div class="hm-legend">
        <div class="hm-legend-item">
            <div class="hm-swatch" style="background:rgba(0, 229, 160, 0.92);"></div>
            <span>1.0 ~ 0.7</span>
        </div>
        <div class="hm-legend-item">
            <div class="hm-swatch" style="background:rgba(93, 173, 226, 0.82);"></div>
            <span>0.7 ~ 0.3</span>
        </div>
        <div class="hm-legend-item">
            <div class="hm-swatch" style="background:rgba(90, 98, 112, 0.80);"></div>
            <span>0.3 ~ 0.0</span>
        </div>
        <div class="hm-legend-item">
            <div class="hm-swatch" style="background:rgba(255, 138, 76, 0.78);"></div>
            <span>0.0 ~ -0.2</span>
        </div>
        <div class="hm-legend-item">
            <div class="hm-swatch" style="background:rgba(220, 53, 69, 0.88);"></div>
            <span>&lt; -0.2</span>
        </div>
    </div>

    <div class="hm-layout">
        <div class="hm-top-left"></div>
        <div class="hm-x-axis" style="grid-template-columns: repeat({n}, {CELL}px);">
            {x_ticks_html}
        </div>

        <div class="hm-left-block" style="min-height:{n * STEP}px;">
            <div class="hm-y-axis" style="grid-template-rows: repeat({n}, {CELL}px);">
                {y_ticks_html}
            </div>
        </div>

        <div class="hm-matrix-wrap" style="min-height:{n * STEP}px;">
            <div class="hm-matrix" style="grid-template-columns: repeat({n}, {CELL}px); grid-template-rows: repeat({n}, {CELL}px);">
                {''.join(matrix_cells)}
            </div>
            {sector_lines_html}
        </div>
    </div>

    <div class="hm-note">
        Hover any visible cell to see: Ticker vs Ticker | Corr = value
    </div>
</div>
"""

with st.container(border=True):
    section_header(
        "PORTFOLIO",
        "CORRELATION",
        "Lower triangle heatmap of log-return dependence, grouped by sector to highlight cluster structure."
    )
    st.html(heatmap_html)