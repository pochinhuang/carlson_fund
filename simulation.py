import datetime
import math
import streamlit as st
import utils

st.set_page_config(
    page_title="Risk Analytics",
    page_icon=":material/assessment:",
    layout="wide",
    initial_sidebar_state="expanded"
)

PORTFOLIO_FILE = "data/19_CG001_Holdings_clean.csv"
portfolio_state = utils.load_portfolio_state(PORTFOLIO_FILE, cash_pool=0.0)

if "risk_metrics" not in st.session_state:
    st.session_state.risk_metrics = {
        "VaR_pct": 0.0,
        "CVaR_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
    }

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
    --danger: #ff6b6b;
    --warning: #ffd166;
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

/* container 改成 dashboard 那種較淡的弱框 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px;
    padding: 20px 20px 16px 20px;
    margin-bottom: 22px;
    box-shadow: none !important;
}

/* plotly / dataframe */
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

.js-plotly-plot * {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* controls */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInputField"] {
    background-color: #10151c !important;
    color: var(--text) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

div[data-baseweb="select"] * {
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

label,
.stDateInput label,
.stNumberInput label,
.stSelectbox label {
    color: var(--text) !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00e5a0 0%, #19c37d 100%);
    color: #04110c;
    border: none;
    border-radius: 8px;
    min-height: 46px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.4px;
}

.stButton > button:hover {
    filter: brightness(1.05);
}

/* hero - 改成 dashboard 同款 */
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

/* 如果你這頁也有 ticker / stat chips，可以直接沿用 */
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

/* section */
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

.small-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
}

.mini-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
}

/* metric cards 也一起改淡，跟 dashboard factor card 更一致 */
.metric-card {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 14px 14px 16px 14px;
    min-height: 120px;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.metric-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 3px;
    height: 100%;
}

.metric-card.var::before     { background: #ff6b6b; }
.metric-card.cvar::before    { background: #ff8a4c; }
.metric-card.sharpe::before  { background: #4a9eff; }
.metric-card.sortino::before { background: #00e5a0; }

.metric-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.metric-name.var     { color: #ff6b6b; }
.metric-name.cvar    { color: #ff8a4c; }
.metric-name.sharpe  { color: #4a9eff; }
.metric-name.sortino { color: #00e5a0; }

/* 如果你有放數值本體，可直接用這個 class */
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    line-height: 1.1;
    letter-spacing: 0.4px;
    color: #ffffff;
    font-weight: 500;
}

/* loading box */
.loading-box {
    background: transparent;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 18px;
    text-align: center;
}

@media (max-width: 900px) {
    .section-title {
        font-size: 34px;
    }

    .hero-title {
        font-size: 58px;
    }
}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
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

def smooth_number(
    value: float,
    key: str,
    label: str,
    card_class: str,
    label_class: str,
    decimals: int = 0,
    duration_ms: int = 900,
    font_px: int = 30,
    prefix: str = "",
    suffix: str = "",
):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = 0.0

    prev_map = st.session_state.setdefault("_smooth_prev", {})
    version_map = st.session_state.setdefault("_smooth_version", {})

    start_val = prev_map.get(key, value)
    prev_map[key] = value

    version_map[key] = version_map.get(key, 0) + 1
    el_id = f"num-{key}-{version_map[key]}"

    st.html(
        f"""
        <div class="metric-card {card_class}">
            <div class="metric-name {label_class}">{label}</div>
            <div id="{el_id}" style="
                font-family:'IBM Plex Mono', monospace;
                font-size:{font_px}px;
                font-weight:600;
                line-height:1.1;
                color:white;
                margin-top:28px;
            "></div>
        </div>

        <script>
        (() => {{
            const el = document.getElementById("{el_id}");
            if (!el) return;

            const startValue = {float(start_val)};
            const targetValue = {float(value)};
            const decimals = {int(decimals)};
            const duration = {int(duration_ms)};
            const prefix = `{prefix}`;
            const suffix = `{suffix}`;

            const fmt = new Intl.NumberFormat(undefined, {{
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            }});

            const ease = t => 1 - Math.pow(1 - t, 3);

            function render(v) {{
                el.textContent = prefix + fmt.format(v) + suffix;
            }}

            if (startValue === targetValue || duration <= 0) {{
                render(targetValue);
                return;
            }}

            let startTs = null;

            function animate(ts) {{
                if (!startTs) startTs = ts;
                const p = Math.min(1, (ts - startTs) / duration);
                const cur = startValue + (targetValue - startValue) * ease(p);
                render(cur);
                if (p < 1) {{
                    requestAnimationFrame(animate);
                }} else {{
                    render(targetValue);
                }}
            }}

            render(startValue);
            requestAnimationFrame(animate);
        }})();
        </script>
        """,
        width="stretch",
        unsafe_allow_javascript=True,
    )

# =========================
# Hero
# =========================
st.markdown("""
<div class="main-hero">
    <div class="hero-title">RISK <span>ANALYTICS</span></div>
    <div class="hero-sub">
        A simulated view of portfolio downside, tail loss, and portfolio performance relative to risk using
        Monte Carlo simulations with copula dependence. Configure the return distribution, forecast horizon,
        and historical estimation window, then run simulations to update portfolio risk metrics.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Risk Metrics
# =========================
metrics_ph = st.empty()

def render_metrics():
    with metrics_ph.container():
        with st.container(border=True):
            section_header(
                "RISK",
                "METRICS",
                "Current portfolio downside and risk-adjusted performance measures derived from the latest simulation output."
            )

            c1, c2, c3, c4 = st.columns(4, gap="large")

            with c1:
                smooth_number(
                    value=st.session_state.risk_metrics["VaR_pct"] * 100.0,
                    key="var95_pct",
                    label="VAR (95%)",
                    card_class="var",
                    label_class="var",
                    decimals=2,
                    duration_ms=900,
                    font_px=30,
                    suffix="%"
                )

            with c2:
                smooth_number(
                    value=st.session_state.risk_metrics["CVaR_pct"] * 100.0,
                    key="cvar95_pct",
                    label="CVAR (95%)",
                    card_class="cvar",
                    label_class="cvar",
                    decimals=2,
                    duration_ms=900,
                    font_px=30,
                    suffix="%"
                )

            with c3:
                smooth_number(
                    value=st.session_state.risk_metrics["sharpe"],
                    key="sharpe",
                    label="SHARPE RATIO",
                    card_class="sharpe",
                    label_class="sharpe",
                    decimals=2,
                    duration_ms=900,
                    font_px=30
                )

            with c4:
                smooth_number(
                    value=st.session_state.risk_metrics["sortino"],
                    key="sortino",
                    label="SORTINO RATIO",
                    card_class="sortino",
                    label_class="sortino",
                    decimals=2,
                    duration_ms=900,
                    font_px=30
                )

render_metrics()

# =========================
# Controls
# =========================
with st.container(border=True):
    section_header(
        "RUN",
        "SIMULATION",
        "Configure the return distribution, forecast horizon, and estimation window for the Monte Carlo engine."
    )

    row1_col1, row1_col2 = st.columns([1, 1], gap="large")
    with row1_col1:
        distribution = st.selectbox(
            "Select Distribution",
            options=["Normal", "T", "Empirical"],
            index=0,
            help="Choose the distribution model for simulation."
        )
    with row1_col2:
        forecast_days = st.number_input(
            "Forecast Days",
            min_value=1,
            max_value=252,
            value=30,
            step=1,
            help="Number of days to simulate into the future."
        )

    row2_col1, row2_col2 = st.columns([1, 1], gap="large")
    with row2_col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.date(2024, 1, 1),
            help="Select the simulation start date."
        )
    with row2_col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.date(2024, 12, 31),
            help="Select the simulation end date."
        )

    st.markdown(
        '<div class="small-note">The selected date range is used as the historical window for estimating the simulation inputs.</div>',
        unsafe_allow_html=True
    )

    run_sim = st.button("Run Monte Carlo Simulation", type="primary")

# =========================
# Output area
# =========================
chart_placeholder = st.empty()
msg_placeholder = st.empty()
loading_placeholder = st.empty()

if run_sim:
    loading_placeholder.markdown("""
    <div class="loading-box">
        <div class="mini-title">Simulation Running</div>
        <div style="color:#95a3b3; font-size:14px; margin-bottom:14px;">
            Generating scenarios and updating portfolio risk statistics...
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        fig, metrics, missing_tickers = utils.run_monte_carlo_simulation(
            state=portfolio_state,
            distribution=distribution,
            forecast_days=forecast_days,
            start_date=start_date,
            end_date=end_date,
            rf_value=0.03,
            n_sims=500,
        )

        loading_placeholder.empty()

        st.session_state.risk_metrics = {
            "VaR_pct": float(metrics["VaR_pct"]),
            "CVaR_pct": float(metrics["CVaR_pct"]),
            "sharpe": float(metrics["sharpe"]),
            "sortino": float(metrics["sortino"]),
        }

        render_metrics()

        with chart_placeholder.container():
            with st.container(border=True):
                section_header(
                    "MONTE CARLO",
                    "DISTRIBUTION",
                    "Simulated portfolio outcome distribution based on the selected model assumptions and historical estimation window."
                )
                st.pyplot(fig)

        if missing_tickers:
            msg_placeholder.warning(
                "Skipped tickers with no valid data: " + ", ".join(missing_tickers)
            )

    except ValueError as e:
        loading_placeholder.empty()
        chart_placeholder.empty()
        msg_placeholder.error(str(e))