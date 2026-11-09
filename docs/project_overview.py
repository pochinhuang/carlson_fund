import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title = "Project Overview",
    page_icon = ":material/info:",
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

ROOT = Path(__file__).resolve().parent.parent


def show_diagram(file_name, description, figure_caption):
    image_path = ROOT / "imgs" / file_name
    if image_path.is_file():
        st.image(str(image_path), width="stretch", alt=description)
    else:
        st.error(f"Image not found: imgs/{file_name}")
    st.caption(figure_caption)


st.title(
    "Portfolio Analytics Platform: Project Overview",
    anchor="portfolio-analytics-platform-project-overview"
)

st.markdown(
    "- [1. Project Overview](#1-project-overview)\n"
    "- [2. System Architecture](#2-system-architecture)\n"
    "- [3. Data Sources and Data Model](#3-data-sources-and-data-model)\n"
    "- [4. Data Ingestion and Daily Updates](#4-data-ingestion-and-daily-updates)\n"
    "- [5. Calculation Workflow](#5-calculation-workflow)\n"
    "- [6. Validation and Scope](#6-validation-and-scope)"
)

# 1. Project Overview
st.header(
    "1. Project Overview",
    anchor="1-project-overview"
)

st.write(
    "This platform brings portfolio holdings, market data, and quantitative analysis into a shared "
    "environment for monitoring an equity portfolio. It connects portfolio composition with recent "
    "performance, factor exposures, and simulated risk, helping users understand both the sources of "
    "return and the concentration of potential losses."
)

st.write("The application provides three complementary views:")

st.table(
    {
        "Page": [
            "Dashboard",
            "Risk Analytics",
            "Factor Model",
        ],
        "Questions it helps answer": [
            "How is the portfolio allocated, what explains its recent performance relative to the benchmark proxy, and where is risk concentrated?",
            "How might portfolio value evolve over a selected horizon, and how large could losses be under the simulation assumptions?",
            "How do stocks within a selected sector differ in their sensitivity to market and style factors?",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "A central PostgreSQL database connects these views to the same underlying holdings and price "
    "histories. Scheduled market updates keep the analytical inputs current, while shared data "
    "preparation and calculation rules maintain consistency across the application."
)

st.divider()

# 2. System Architecture
st.header(
    "2. System Architecture",
    anchor="2-system-architecture"
)

st.write(
    "Users access the platform through a web browser. Streamlit handles the interface and Python "
    "calculations within the application, while a shared Data Access layer retrieves the required "
    "information from PostgreSQL. A separate background pipeline obtains and validates market data."
)

show_diagram(
    "system_architecture.png",
    "System Architecture",
    "*Figure 1. System components and data flow in the reference AWS deployment.*",
)

st.write(
    "This arrangement separates **using data** from **updating data**. Opening an analytical page "
    "reads stored information rather than starting a new market data download. The background "
    "pipeline can continue updating prices even when no users have the application open."
)

st.write(
    "The shared analytical layer also gives the pages a common foundation. Rules for choosing price "
    "sources, aligning dates, and preparing holdings are applied consistently, so each page can focus "
    "on a different analytical question."
)

st.write(
    "In the reference AWS deployment, the application, PostgreSQL, and scheduler run as separate "
    "Docker Compose services on one EC2 instance. Persistent storage retains the database across "
    "routine container restarts. Market data credentials remain in the server environment."
)

st.divider()

# 3. Data Sources and Data Model
st.header(
    "3. Data Sources and Data Model",
    anchor="3-data-sources-and-data-model"
)

st.write("The initial dataset combines three types of information:")

st.table(
    {
        "Data source": [
            "Portfolio holdings",
            "Benchmark proxy holdings",
            "Historical daily prices",
        ],
        "Contents and analytical role": [
            "Positions, share counts, weights, and classifications used to describe and value the fund",
            "Constituents, weights, and sector classifications used for relative performance and sector comparisons",
            "Close and adjusted close histories used to initialize valuation, return calculations, and the market update universe",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "After initialization, `Alpaca Market Data` supplies prices for US stocks and ETFs in the "
    "configured universe. `yfinance` supplies prices for the market proxy, `CSUS.L`."
)

st.write(
    "The data model connects these inputs through shared instrument records. A security can belong to "
    "both the fund and the benchmark proxy while retaining one instrument definition and its "
    "associated price histories."
)

show_diagram(
    "data_model.png",
    "Data Model",
    "*Figure 2. PostgreSQL entities, relationships, and analytical views.*",
)

st.write(
    "The database separates financial information from the records needed to explain its origin and "
    "update history:"
)

st.table(
    {
        "Schema": [
            "**core**",
            "**raw**",
            "**ops**",
            "**mart**",
        ],
        "Purpose": [
            "Store instruments, portfolio positions, prices, and reference definitions used in analysis",
            "Identify the source files used in each import and preserve their import metadata",
            "Track update outcomes, data publication, and the results for individual instruments",
            "Provide reusable analytical views of the stored data",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "Keeping these responsibilities separate makes results traceable. A price observation retains its "
    "date, provider, price type, and connection to an import or update run. Source selection rules "
    "determine which stored history is used for analysis."
)

st.write(
    "Close and adjusted close serve different purposes. **Close supports valuation with the imported "
    "share counts**, while **adjusted prices support return calculations**. Preserving both avoids "
    "confusing the value of a position with the price history used to measure its returns."
)

st.divider()

# 4. Data Ingestion and Daily Updates
st.header(
    "4. Data Ingestion and Daily Updates",
    anchor="4-data-ingestion-and-daily-updates"
)

st.subheader(
    "4.1 Initial Import",
    anchor="41-initial-import"
)

st.write(
    "The initial import establishes the portfolio, benchmark proxy, instrument definitions, and "
    "historical prices in PostgreSQL. Before publication, the inputs are checked for valid "
    "identifiers, holdings weights, dates, and prices."
)

st.write(
    "Import records identify the source files used and allow unchanged inputs to be recognized. This "
    "provides a reproducible starting point for the application and a way to trace analytical data "
    "back to the supplied snapshots."
)

st.subheader(
    "4.2 Nightly Market Update",
    anchor="42-nightly-market-update"
)

st.write(
    "The market pipeline runs at **22:00 in America/Chicago** and requests data through the previous "
    "calendar date. Trading calendars determine the latest expected session for the US and UK "
    "markets, accounting for weekends and holidays."
)

st.write(
    "Candidate price histories are checked before they become available to the application. The "
    "checks cover valid observations, continuity with previously published dates, and arrival at the "
    "latest expected trading session. Refreshing historical prices also allows earlier provider "
    "adjustments to remain consistent with newer observations."
)

st.write(
    "When a candidate series is incomplete, the application retains its previously selected history. "
    "Other accepted series can still be published, with the update recorded as partial. Execution "
    "records and reports show which instruments were updated and which require attention."
)

st.subheader(
    "4.3 From Published Data to Updated Pages",
    anchor="43-from-published-data-to-updated-pages"
)

st.write("Market downloads and page refresh checks follow separate schedules:")

st.table(
    {
        "Process": [
            "Market data update",
            "Page refresh check",
        ],
        "Reference schedule": [
            "Daily at 22:00, America/Chicago",
            "Every 12 hours while a page session is active",
        ],
        "Purpose": [
            "Retrieve, validate, and publish prices",
            "Detect whether new data have been published",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "Publication advances a database version that the application uses to recognize changes. If the "
    "version is unchanged, the current view remains. If it changes, the page reloads its inputs and "
    "runs again. Opening or reloading a page can also pick up published data before the next periodic "
    "check."
)

st.write(
    "The Dashboard recalculates during page execution. Factor analysis and Monte Carlo calculations "
    "remain controlled by their **Run** buttons. A page refresh check reads the publication status; "
    "it does not contact market data providers."
)

st.divider()

# 5. Calculation Workflow
st.header(
    "5. Calculation Workflow",
    anchor="5-calculation-workflow"
)

st.write(
    "The analytical pages start with shared holdings and preferred price histories, then prepare "
    "inputs for their specific calculations. Dates are aligned to the observations required by each "
    "method. Missing prices remain missing rather than being converted into artificial observations "
    "or zero returns."
)

st.write(
    "Every fund holding is required for portfolio analysis. Benchmark calculations use constituents "
    "with price coverage and normalize their weights to 100%, while the original benchmark holdings "
    "remain stored for reference."
)

show_diagram(
    "calculation_workflow.png",
    "Calculation Workflow",
    "*Figure 3. Data ingestion, daily updates, and analytical calculation workflows.*",
)

st.subheader(
    "5.1 Dashboard",
    anchor="51-dashboard"
)

st.write(
    "The Dashboard combines composition, recent relative performance, and risk. Holdings and sector "
    "displays show how the portfolio is allocated using the imported weights. Recent performance uses "
    "common price endpoints separated by five US trading sessions, allowing the fund and benchmark "
    "proxy to be compared over the same period."
)

st.write(
    "Brinson attribution separates the return difference into **allocation, selection, and "
    "interaction effects**. This helps distinguish the effect of sector positioning from the "
    "performance of stocks within those sectors."
)

st.write(
    "Factor regressions for individual fund holdings are aggregated using portfolio weights to "
    "summarize market and style sensitivities. Risk contributions use valuation weights and "
    "historical covariance to show how each holding contributes to overall portfolio risk. Volatility "
    "and correlation displays add context about variability and how assets move together."
)

st.subheader(
    "5.2 Sector Factor Analysis",
    anchor="52-sector-factor-analysis"
)

st.write(
    "The factor analysis page examines benchmark proxy stocks within a selected sector. Each stock's "
    "adjusted daily returns are regressed on the market proxy **CSUS.L** and four style ETFs: **SIZE, "
    "VLUE, MTUM, and QUAL**. The calculation uses matching return intervals and sufficient shared "
    "observations."
)

st.write(
    "The resulting table and charts allow users to compare sensitivities within a sector. Two stocks "
    "with the same sector classification may respond differently to market movements or exhibit "
    "different style exposures."
)

st.write(
    "The coefficients describe exposure to the selected ETF proxies. **Alpha is the daily regression "
    "intercept**, while the market coefficient measures sensitivity to CSUS.L. Interpretation depends "
    "on the estimation sample and the other proxies included in the model."
)

st.subheader(
    "5.3 Monte Carlo Risk Analysis",
    anchor="53-monte-carlo-risk-analysis"
)

st.write(
    "Monte Carlo analysis explores possible portfolio outcomes over a user selected forecast horizon. "
    "Historical adjusted returns provide estimates of average returns, volatility, and dependence "
    "between holdings. The user chooses Normal, T, or Empirical return distributions."
)

st.write(
    "The simulation generates correlated scenarios, compounds asset returns into price paths, and "
    "aggregates the resulting values using fixed share counts. Starting values come from a complete "
    "Close snapshot. Modeling assets jointly allows the simulation to reflect how their movements "
    "combine at the portfolio level."
)

st.write(
    "The output includes simulated value paths, a mean path, a percentile range, and summary risk "
    "measures. **VaR** describes a loss threshold at the selected confidence level, while **CVaR** "
    "summarizes losses beyond that threshold. Sharpe and Sortino ratios compare simulated excess "
    "returns with total variability and downside variability, respectively."
)

st.write(
    "These measures describe the selected forecast horizon. The reference rate is converted to that "
    "same period, and the displayed Sharpe and Sortino ratios are not annualized. The Risk Management "
    "and Factor Analysis documentation pages provide the mathematical details behind the models."
)

st.divider()

# 6. Validation and Scope
st.header(
    "6. Validation and Scope",
    anchor="6-validation-and-scope"
)

st.subheader(
    "6.1 Validation",
    anchor="61-validation"
)

st.write("Checks are applied during import, market updates, and analytical preparation:")

st.table(
    {
        "Area": [
            "Data quality",
            "Update completeness",
            "Analytical consistency",
            "Traceability",
        ],
        "What is checked": [
            "Holdings weights, identifiers, valid prices, dates, and duplicate observations",
            "Coverage of existing history and the expected latest trading session",
            "Adequate regression data, attribution reconciliation, and risk contribution totals",
            "Source imports, update outcomes, coverage, and published data versions",
        ],
    },
    hide_index=True,
    width="stretch",
)

st.write(
    "Two useful reconciliation checks connect individual results to their totals: Brinson effects sum "
    "to the portfolio return minus the benchmark return, and relative Euler risk contributions sum to "
    "100% within numerical tolerance."
)

st.subheader(
    "6.2 Scope and Interpretation",
    anchor="62-scope-and-interpretation"
)

st.markdown(
    "- **Holdings represent an imported snapshot.** Price updates change valuations and analytical inputs; they do not update trades, share counts, or allocation weights.\n"
    "- **The benchmark is a proxy subject to price coverage.** Results reflect the available constituents and their normalized weights, so coverage matters when interpreting relative performance.\n"
    "- **Calculations can use different effective windows.** Attribution, regression, and simulation require different combinations of valid observations.\n"
    "- **Simulation outcomes depend on model assumptions.** The selected history, return distribution, Gaussian dependence mechanism, and fixed positions affect the results. Simulated outcomes are scenarios rather than forecasts with guaranteed accuracy.\n"
    "- **A published update can be partial.** Some histories may be newer than others. Publication status, coverage, and the effective calculation dates provide the context needed to interpret the latest results."
)
