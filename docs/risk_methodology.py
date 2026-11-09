import streamlit as st

st.set_page_config(
    page_title = "Risk Methodology",
    page_icon = ":material/article:",
    layout = "wide",                
    initial_sidebar_state = "expanded" 
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0c0f;
    --card: #141820;
    --border: #1e2530;
    --text: #e8edf5;
    --muted: #7f8b99;
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

div[data-testid="stMarkdownContainer"] {
    color: var(--text);
}

div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    color: var(--text);
}

div[data-testid="stMarkdownContainer"] code {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #4a9eff !important;
    background-color: rgba(74, 158, 255, 0.12) !important;
    border: 1px solid rgba(74, 158, 255, 0.22) !important;
    border-radius: 4px !important;
    padding: 0.12rem 0.38rem !important;
}
</style>
""", unsafe_allow_html=True)


st.title(
    "Risk Methodology: Monte Carlo Simulation",
    anchor="risk-methodology-monte-carlo-simulation"
)

st.markdown(
    "- [1. Overview](#1-overview)\n"
    "- [2. Data and Settings](#2-data-and-settings)\n"
    "- [3. How the Simulation Works](#3-how-the-simulation-works)\n"
    "- [4. Risk Metrics](#4-risk-metrics)\n"
    "- [5. Reading the Results](#5-reading-the-results)\n"
    "- [6. Key Assumptions](#6-key-assumptions)"
)

# 1. Overview
st.header(
    "1. Overview",
    anchor="1-overview"
)

st.write(
    "This tool estimates how portfolio value could change over a selected forecast period. It uses "
    "historical prices to estimate how stocks behave, then generates **500 possible future portfolio "
    "paths**."
)

st.write(
    "The results show the spread of possible outcomes and summarize risk using **VaR, CVaR, Sharpe "
    "ratio, and Sortino ratio**."
)

# 2. Data and Settings
st.header(
    "2. Data and Settings",
    anchor="2-data-and-settings"
)

st.write(
    "Holdings and prices are read from PostgreSQL. **Adjusted close** is used to calculate historical "
    "returns. **Close multiplied by shares** gives the starting portfolio value."
)

st.table(
    {
        "Setting": [
            "Historical window",
            "Forecast period",
            "Simulated paths",
            "Return distribution",
            "VaR and CVaR level",
            "Annual reference rate",
        ],
        "Current setup": [
            "User selected; initially up to 365 calendar days",
            "1 to 252 trading day steps; default 30",
            "500",
            "Normal, T, or Empirical",
            "95%",
            "3%",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "The calculation requires a complete price snapshot for all holdings and at least **20 matching "
    "daily return observations**. If these requirements are not met, the analysis stops rather than "
    "dropping stocks from the portfolio."
)

st.divider()

# 3. How the Simulation Works
st.header(
    "3. How the Simulation Works",
    anchor="3-how-the-simulation-works"
)

st.subheader(
    "Step 1: Estimate Historical Returns",
    anchor="step-1-estimate-historical-returns"
)

st.write(
    "Daily log returns are calculated from adjusted prices:"
)

st.latex(
    "\n"
    "r_t=\\ln\\left(\\frac{P_t}{P_{t-1}}\\right).\n"
)

st.write(
    "Here, $P_t$ is the adjusted price and $r_t$ is the daily log return. These returns are used to "
    "estimate each stock's **average return, volatility, and correlation with other stocks**."
)

st.subheader(
    "Step 2: Connect Stock Movements",
    anchor="step-2-connect-stock-movements"
)

st.write(
    "Stocks are simulated together because portfolio risk depends on how they move relative to one "
    "another. Cholesky decomposition uses the historical correlation matrix to create related random "
    "shocks:"
)

st.latex(
    "\n"
    "R=LL^{\\mathsf T},\\qquad X=LZ.\n"
)

st.write(
    "$R$ is the correlation matrix. $Z$ contains independent Normal shocks, and $L$ transforms them "
    "into correlated shocks $X$. This gives the simulation a common structure for stock movements."
)

st.subheader(
    "Step 3: Choose a Return Distribution",
    anchor="step-3-choose-a-return-distribution"
)

st.table(
    {
        "Distribution": [
            "Normal",
            "T",
            "Empirical",
        ],
        "How it works": [
            "Uses a bell shaped distribution based on historical average returns and volatility",
            "Allows heavier tails and larger individual shocks; degrees of freedom are fixed at 5",
            "Uses historical return quantiles to reflect the shape of the observed sample",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "All three choices use the same Gaussian dependence model to connect the stocks. Selecting T "
    "changes the individual return distributions."
)

st.subheader(
    "Step 4: Build Portfolio Paths",
    anchor="step-4-build-portfolio-paths"
)

st.write(
    "Each scenario starts from the latest complete Close snapshot within the selected window. "
    "Simulated log returns are compounded into future stock prices:"
)

st.latex(
    "\n"
    "S_{t+1}=S_t e^{r_{t+1}}.\n"
)

st.write(
    "Here, $S$ is the stock price and $r$ is the simulated log return. Portfolio value is then the "
    "sum of each stock's price multiplied by its share count:"
)

st.latex(
    "\n"
    "V=\\sum_i q_i S_i.\n"
)

st.write(
    "$q_i$ is the number of shares in stock $i$, and $S_i$ is its simulated price. Repeating this "
    "process creates 500 portfolio paths over the selected forecast period."
)

st.divider()

# 4. Risk Metrics
st.header(
    "4. Risk Metrics",
    anchor="4-risk-metrics"
)

st.write(
    "At the end of each path, the final portfolio value is compared with the starting value. A "
    "positive loss means the portfolio has fallen in value."
)

st.table(
    {
        "Metric": [
            "VaR at 95%",
            "CVaR at 95%",
            "Sharpe ratio",
            "Sortino ratio",
        ],
        "Meaning": [
            "A loss threshold exceeded by about 5% of the simulated outcomes",
            "The average loss among outcomes at or beyond the VaR threshold",
            "Average excess return relative to total return variability",
            "Average excess return relative to downside variability",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "VaR and CVaR are displayed as percentages of the starting portfolio value. VaR describes a "
    "threshold, not the largest possible loss."
)

st.write(
    "The ratios can be written simply as:"
)

st.latex(
    "\n"
    "\\text{Sharpe}=\\frac{\\text{Average excess return}}{\\text{Standard deviation of returns}}.\n"
)

st.latex(
    "\n"
    "\\text{Sortino}=\\frac{\\text{Average excess return}}{\\text{Downside deviation}}.\n"
)

st.write(
    "Excess return is the return above the reference rate. The annual 3% rate is converted to the "
    "selected forecast period. Downside deviation focuses on returns below that reference."
)

st.write(
    "**Both ratios use the simulated final returns for the selected forecast period and are not "
    "annualized.** If a denominator is zero, the ratio is undefined, even if the current card "
    "displays 0.00."
)

st.divider()

# 5. Reading the Results
st.header(
    "5. Reading the Results",
    anchor="5-reading-the-results"
)

st.write(
    "Choose the historical window, return distribution, and forecast period, then click **Run Monte "
    "Carlo Simulation**."
)

st.markdown(
    "- **Individual lines:** possible future portfolio value paths.\n"
    "- **Dashed line:** the average portfolio value across scenarios.\n"
    "- **Shaded area:** the middle 90% of simulated values at each forecast step, from the 5th to the 95th percentile."
)

st.write(
    "After changing inputs or receiving new database prices, run the simulation again to update the "
    "results."
)

# 6. Key Assumptions
st.header(
    "6. Key Assumptions",
    anchor="6-key-assumptions"
)

st.markdown(
    "- **Fixed holdings:** share counts stay unchanged, with no trading or rebalancing.\n"
    "- **Historical estimates:** average returns, volatility, and dependence remain fixed during the forecast period. Future market conditions may differ.\n"
    "- **Independent future steps:** the model does not explicitly simulate changing volatility or market regimes.\n"
    "- **Simulation noise:** repeated runs can produce different results, especially for tail losses.\n"
    "- **Empirical limits:** simulated daily returns stay within the observed sample range when using Empirical.\n"
    "- **Scope:** trading costs, taxes, and explicit dividend cash flows are not separately modeled."
)
