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

# =========================
# Global Style
# =========================
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

/* container 改成和 dashboard 一樣偏乾淨、弱化 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px;
    padding: 20px 20px 16px 20px;
    margin-bottom: 22px;
    box-shadow: none !important;
}

/* 控制元件 */
div[data-baseweb="select"] > div {
    background-color: #10151c !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    min-height: 44px !important;
}

div[data-baseweb="select"] * {
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00e5a0 0%, #19c37d 100%);
    color: #04110c;
    border: none;
    border-radius: 8px;
    height: 44px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    filter: brightness(1.05);
}

/* dataframe / plotly 不額外加框 */
div[data-testid="stDataFrame"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}

div[data-testid="stDataFrame"] * {
    font-family: 'IBM Plex Mono', monospace !important;
}

div[data-testid="stPlotlyChart"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}

.js-plotly-plot * {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Hero - 完全改成 dashboard 風格 */
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

/* Section */
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

/* Exposure cards */
.factor-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(120px, 1fr));
    gap: 14px;
    margin-top: 18px;
    margin-bottom: 8px;
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

.small-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
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
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def section_header(
    title_text: str,
    title_accent: str,
    subtitle: str,
    title_color: str = "var(--text)"):


    st.markdown(
        f"""
        <div class="section-title" style="color:{title_color};">{title_text} <span>{title_accent}</span></div>
        <div class="section-sub">{subtitle}</div>
        <div class="section-divider"></div>
        """,
        unsafe_allow_html=True
    )

def style_bar_chart(fig, factor_name: str):
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>" + f"{factor_name}: " + "%{y:.4f}<extra></extra>",
        marker_line_width=0
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f141b",
        font=dict(color="white", family="IBM Plex Mono", size=11),
        xaxis=dict(
            title="Ticker",
            title_font=dict(family="IBM Plex Mono", size=11, color="white"),
            tickfont=dict(family="IBM Plex Mono", size=10, color="white"),
            tickangle=90,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title="Exposure",
            title_font=dict(family="IBM Plex Mono", size=11, color="white"),
            tickfont=dict(family="IBM Plex Mono", size=10, color="white"),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False
        ),
        margin=dict(t=10, l=10, r=10, b=10),
        height=420,
        showlegend=False,
        title=dict(text="")
    )
    return fig

# =========================
# Data loaders
# =========================
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

# =========================
# Core logic
# =========================
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

    close_df = all_close_df[valid_tickers].copy().ffill()
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

    all_returns = pd.concat([factor_returns, market_returns, stock_returns], axis=1)
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

# =========================
# Load data
# =========================
df_holdings = load_portfolio_data()
all_close_df = load_all_close_prices()
factor_returns, market_returns = load_factor_and_market_returns()

sector_options = sorted(df_holdings["sector"].dropna().unique().tolist())

# =========================
# Hero
# =========================
st.markdown(f"""
<div class="main-hero">
    <div class="hero-title">SECTOR FACTOR <span>ANALYSIS</span></div>
    <div class="hero-sub">
        Linear Regression Analysis of Stock Alpha and Factor Sensitivities Within a Selected Sector.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Control panel
# =========================
# with st.container(border=True):
#     section_header(
#         "RUN",
#         "MODEL",
#         "Select a sector and compute ticker-level factor exposures."
#     )

#     col1, col2 = st.columns([2.2, 1], gap="large")

#     with col1:
#         selected_sector = st.selectbox(
#             "Select a Sector",
#             options=sector_options,
#             index=0
#         )

#     with col2:
#         run_analysis = st.button("Run Factor Analysis")

with st.container(border=True):
    section_header(
        "RUN",
        "MODEL",
        "Select a sector and compute factor exposures for individual stocks."
    )

    col1, col2 = st.columns([2.2, 1], gap="large")

    with col1:
        st.markdown("**Select a Sector**")
        selected_sector = st.selectbox(
            "Select a Sector",
            options=sector_options,
            index=0,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        run_analysis = st.button("Run Factor Analysis", use_container_width=True)

# =========================
# Main output
# =========================
if run_analysis:
    with st.spinner(f"Analyzing sector '{selected_sector}'..."):
        exposures = compute_sector_factor_exposures(
            df_holdings,
            selected_sector,
            all_close_df,
            factor_returns,
            market_returns
        )

    if exposures.empty:
        with st.container(border=True):
            section_header(
                "NO VALID",
                "RESULTS",
                f"No valid exposure results were found for sector: {selected_sector}."
            )
            st.warning("No valid exposure results for this sector.")
    else:

        with st.container(border=True):
            section_header(
                "EXPOSURE",
                "TABLE",
                "Full regression output for each ticker in the selected sector."
            )
            st.dataframe(exposures, width='stretch', height=520)



        factor_meta = {
            "Alpha": {
                "subtitle": "Alpha represents the stock's intercept term, capturing average return not explained by the market and style factors.",
                "color": "#00e5a0"
            },
            "Beta": {
                "subtitle": "Beta measures sensitivity to the broad market. A higher beta means the stock tends to move more strongly with market returns.",
                "color": "#4a9eff"
            },
            "SIZE": {
                "subtitle": "SIZE measures exposure to the size factor. Positive exposure indicates behavior more aligned with smaller-cap characteristics.",
                "color": "#c084fc"
            },
            "VLUE": {
                "subtitle": "VLUE measures exposure to the value factor. Positive exposure suggests stronger alignment with value-style stocks.",
                "color": "#ff8a4c"
            },
            "MTUM": {
                "subtitle": "MTUM measures exposure to the momentum factor. Positive exposure suggests stronger comovement with momentum-driven stocks.",
                "color": "#ffd166"
            },
            "QUAL": {
                "subtitle": "QUAL measures exposure to the quality factor. Positive exposure suggests stronger alignment with profitable and financially stable firms.",
                "color": "#ff5d8f"
            }
        }



        factors_to_plot = ["Alpha", "Beta", "SIZE", "VLUE", "MTUM", "QUAL"]

        for factor in factors_to_plot:
            if factor in exposures.columns:
                meta = factor_meta[factor]

                with st.container(border=True):
                    section_header(
                        title_text=factor.upper(),
                        title_accent="",
                        subtitle=meta["subtitle"],
                        title_color=meta["color"]
                    )

                    fig = px.bar(
                        exposures.reset_index(),
                        x="Ticker",
                        y=factor,
                        color_discrete_sequence=[meta["color"]],
                        title=""
                    )

                    fig = style_bar_chart(fig, factor.upper())
                    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})