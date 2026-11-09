import streamlit as st

st.set_page_config(
    page_title = "Factor Methodology",
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
    background-color: rgba(255,255,255,0.06) !important;
}
</style>
""", unsafe_allow_html=True)


st.title(
    "Sector Factor Analysis Documentation",
    anchor="sector-factor-analysis-documentation"
)

st.markdown(
    "- [1. Overview](#1-overview)\n"
    "- [2. Data Inputs](#2-data-inputs)\n"
    "- [3. Application Workflow](#3-application-workflow)\n"
    "- [4. Regression Model](#4-regression-model)\n"
    "- [5. Output Structure](#5-output-structure)\n"
    "- [6. Visualization](#6-visualization)\n"
    "- [7. Practical Interpretation](#7-practical-interpretation)\n"
    "- [8. Limitations](#8-limitations)"
)

# 1. Overview
st.header(
    "1. Overview",
    anchor="1-overview"
)

st.write(
    "This Streamlit application performs **sector factor exposure analysis** for stocks in the "
    "**R2500G benchmark proxy**. The user selects a sector and estimates the factor sensitivities of "
    "stocks within that sector that have sufficient price history."
)

st.write(
    "The workflow combines:"
)

st.markdown(
    "- benchmark proxy holdings and sector classifications,\n"
    "- historical factor ETF returns,\n"
    "- historical market proxy returns,\n"
    "- individual stock return regressions."
)

st.write(
    "For each stock, the application fits a linear regression with one market proxy and four style "
    "ETF proxies. The outputs include a table of estimated exposures and bar charts for each "
    "coefficient."
)

st.divider()

# 2. Data Inputs
st.header(
    "2. Data Inputs",
    anchor="2-data-inputs"
)

st.subheader(
    "2.1 Benchmark Proxy Holdings",
    anchor="21-benchmark-proxy-holdings"
)

st.write(
    "The application reads the benchmark proxy holdings from PostgreSQL. The **Ticker** identifies "
    "each stock, and **Sector** groups stocks into the choices shown in the sector dropdown."
)

st.write(
    "The analytical stock list includes benchmark proxy constituents in the configured price "
    "universe. The original benchmark holdings remain stored, including constituents outside that "
    "priced subset."
)

st.write(
    "The benchmark proxy defines which stocks are analyzed. **CSUS.L provides the market return used "
    "in the regression.** These are separate roles."
)

st.subheader(
    "2.2 Factor and Market Price Data",
    anchor="22-factor-and-market-price-data"
)

st.write(
    "Historical adjusted close prices for the selected stocks and all five proxies are read from "
    "PostgreSQL. The published price histories are maintained by the data pipeline; clicking Run "
    "Factor Analysis uses the stored data."
)

st.table(
    {
        "Role": [
            "Market",
            "Size",
            "Value",
            "Momentum",
            "Quality",
        ],
        "Proxy ticker": [
            "CSUS.L",
            "SIZE",
            "VLUE",
            "MTUM",
            "QUAL",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "The page uses the available database history rather than imposing a fixed one year window. The "
    "usable sample for each stock depends on its overlap with all five proxies."
)

st.subheader(
    "2.3 Daily Returns",
    anchor="23-daily-returns"
)

st.write(
    "Adjusted prices are converted into simple daily returns. For a price series $P_t$:"
)

st.latex(
    "\n"
    "r_t=\\frac{P_t-P_{t-1}}{P_{t-1}}.\n"
)

st.write(
    "The same return definition is used for stocks, the market proxy, and the style ETFs. "
    "Calculations use decimal returns, so a return of 0.01 represents 1%."
)

st.divider()

# 3. Application Workflow
st.header(
    "3. Application Workflow",
    anchor="3-application-workflow"
)

st.write(
    "The user interface and calculation follow this sequence:"
)

st.markdown(
    "1. Read benchmark proxy holdings and obtain the available sectors.\n"
    "2. Select a sector and click **Run Factor Analysis**.\n"
    "3. Retrieve adjusted prices for the selected stocks, CSUS.L, SIZE, VLUE, MTUM, and QUAL.\n"
    "4. Calculate daily returns and align matching return intervals.\n"
    "5. Fit a regression for each stock with at least **60 valid observations** shared with all five proxies.\n"
    "6. Display the coefficient table and factor bar charts."
)

st.write(
    "**Missing prices are not forward filled, and missing returns are not replaced with zero.** Both "
    "price endpoints must be valid for the stock and every proxy. This also handles differences "
    "between US and UK trading calendars, so returns over unmatched intervals are excluded."
)

st.write(
    "Stocks with fewer than 60 valid observations are omitted from the regression output. If no "
    "stocks qualify, the page shows a message indicating that there are no valid results. Missing "
    "required histories or an invalid regression design produce an error."
)

st.divider()

# 4. Regression Model
st.header(
    "4. Regression Model",
    anchor="4-regression-model"
)

st.write(
    "For each stock in the selected sector, the application fits the following linear model:"
)

st.latex(
    "\n"
    "\\begin{aligned}\n"
    "r_{i,t}={}&\\alpha_i+\\beta_i r_{m,t}\n"
    "+\\gamma_{i,\\text{SIZE}}r_{\\text{SIZE},t}\n"
    "+\\gamma_{i,\\text{VLUE}}r_{\\text{VLUE},t}\\\\\n"
    "&+\\gamma_{i,\\text{MTUM}}r_{\\text{MTUM},t}\n"
    "+\\gamma_{i,\\text{QUAL}}r_{\\text{QUAL},t}\n"
    "+\\varepsilon_{i,t}.\n"
    "\\end{aligned}\n"
)

st.write(
    "Here:"
)

st.markdown(
    "- $r_{i,t}$ is the daily return of stock $i$.\n"
    "- $r_{m,t}$ is the daily return of the market proxy, CSUS.L.\n"
    "- $\\alpha_i$ is the intercept.\n"
    "- $\\beta_i$ is the coefficient on the market return.\n"
    "- $\\gamma_{i,\\cdot}$ are the coefficients on the four style ETF returns.\n"
    "- $\\varepsilon_{i,t}$ is the regression residual."
)

st.write(
    "The model is estimated using **ordinary least squares with an intercept**. Each stock has its "
    "own set of coefficients, and holdings weights are not used as regression weights."
)

st.write(
    "The regressions use daily ETF returns without subtracting a risk free rate. **Alpha is a daily "
    "intercept and is not annualized.** It describes the component of daily return not explained by "
    "this particular set of ETF proxies."
)

st.divider()

# 5. Output Structure
st.header(
    "5. Output Structure",
    anchor="5-output-structure"
)

st.write(
    "The results table contains one row for each stock with a valid regression. Its columns are:"
)

st.table(
    {
        "Column": [
            "Alpha",
            "Beta",
            "SIZE",
            "VLUE",
            "MTUM",
            "QUAL",
        ],
        "Interpretation": [
            "Daily regression intercept",
            "Sensitivity to the market proxy, CSUS.L",
            "Sensitivity to the size ETF proxy",
            "Sensitivity to the value ETF proxy",
            "Sensitivity to the momentum ETF proxy",
            "Sensitivity to the quality ETF proxy",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "A positive coefficient indicates movement in the same direction as the corresponding proxy, "
    "after controlling for the other regressors. A negative coefficient indicates the opposite "
    "relationship."
)

st.write(
    "Alpha is expressed in daily return units. For example, an Alpha of 0.001 corresponds to a daily "
    "intercept of 0.1%. The table reports individual stock estimates, rather than a single exposure "
    "for the whole sector."
)

st.divider()

# 6. Visualization
st.header(
    "6. Visualization",
    anchor="6-visualization"
)

st.write(
    "After the exposures are computed, the application generates a bar chart for each coefficient: "
    "**Alpha, Beta, SIZE, VLUE, MTUM, and QUAL**."
)

st.write(
    "Each chart shows:"
)

st.markdown(
    "- **Horizontal axis:** stock ticker.\n"
    "- **Vertical axis:** estimated coefficient."
)

st.write(
    "The charts show how exposures differ across stocks within the selected sector. They help "
    "identify names with higher market sensitivity, stronger relationships with the chosen style "
    "ETFs, or different factor profiles despite sharing a sector classification."
)

st.divider()

# 7. Practical Interpretation
st.header(
    "7. Practical Interpretation",
    anchor="7-practical-interpretation"
)

st.write(
    "The application is useful for comparing factor behavior within a sector. For example, stocks in "
    "the same Technology sector may have different market betas or different sensitivities to "
    "momentum and value ETF returns."
)

st.write(
    "The results can help answer:"
)

st.markdown(
    "- Which stocks in this sector have the strongest market sensitivity?\n"
    "- Which stocks have positive or negative exposure to the selected style proxies?\n"
    "- Does the sector contain similar factor profiles, or distinct groups of stocks?\n"
    "- How do daily intercept estimates differ across names?"
)

st.write(
    "The style coefficients should be interpreted as sensitivities to the selected ETF returns. A "
    "positive loading alone does not establish that a stock has a particular fundamental "
    "characteristic or that its exposure is statistically significant."
)

st.divider()

# 8. Limitations
st.header(
    "8. Limitations",
    anchor="8-limitations"
)

st.subheader(
    "8.1 Estimation Sample",
    anchor="81-estimation-sample"
)

st.write(
    "The estimates depend on the available price history and matching return observations. Stocks can "
    "have different effective sample lengths and date ranges. Shorter samples can produce less stable "
    "coefficients, especially for volatile stocks."
)

st.subheader(
    "8.2 Linear Specification",
    anchor="82-linear-specification"
)

st.write(
    "The model assumes a linear relationship between stock returns and the included proxies. It may "
    "miss nonlinear effects, changing market regimes, or interactions between factors. Strong "
    "correlation between ETF proxies can also make individual coefficients sensitive to the sample."
)

st.subheader(
    "8.3 Data Quality and Coverage",
    anchor="83-data-quality-and-coverage"
)

st.write(
    "Results depend on the quality and availability of the published database prices and benchmark "
    "proxy coverage. Stocks with insufficient common observations are excluded. A successful analysis "
    "therefore describes the stocks with usable data in the selected sector."
)

st.subheader(
    "8.4 Factor Proxy Choice",
    anchor="84-factor-proxy-choice"
)

st.write(
    "The factors are represented by tradable ETFs rather than canonical academic factor portfolios. "
    "The estimated coefficients describe **exposure to the chosen proxies**, which can share market "
    "and style characteristics. They should not automatically be treated as exact academic factor "
    "exposures."
)

st.subheader(
    "8.5 Statistical Diagnostics",
    anchor="85-statistical-diagnostics"
)

st.write(
    "The current page displays coefficient estimates only. It does not report:"
)

st.markdown(
    "- $R^2$,\n"
    "- t statistics,\n"
    "- p values,\n"
    "- standard errors,\n"
    "- confidence intervals."
)

st.write(
    "The output is therefore best used for exploratory exposure analysis and comparison, with "
    "statistical significance left unassessed by the displayed results."
)
