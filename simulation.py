import datetime
import streamlit as st
import utils
import math

st.set_page_config(
    page_title = "Risk Analytics",
    page_icon = ":material/assessment:",
    layout = "wide",                
    initial_sidebar_state = "expanded" 
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


def smooth_number(
    value: float,
    key: str,
    label: str = None,
    decimals: int = 0,
    duration_ms: int = 900,
    font_px: int = 48,
    prefix: str = "",
    suffix: str = "",
):
    # 避免 NaN / inf 搞壞前端
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = 0.0

    prev_map = st.session_state.setdefault("_smooth_prev", {})
    version_map = st.session_state.setdefault("_smooth_version", {})

    start_val = prev_map.get(key, value)
    prev_map[key] = value

    # 每次 render 都遞增版本，強制產生新的 DOM id
    version_map[key] = version_map.get(key, 0) + 1
    el_id = f"num-{key}-{version_map[key]}"

    label_html = (
        f'<div style="font-size:{int(font_px*0.5)}px;opacity:.75;color:white;">{label}</div>'
        if label else ""
    )

    st.html(
        f"""
        <div style="display:flex;gap:8px;align-items:flex-end;">
            {label_html}
            <div id="{el_id}" style="font-size:{font_px}px;font-weight:700;line-height:1;color:white;"></div>
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


metrics_ph = st.empty()


def render_metrics():
    with metrics_ph.container():
        st.markdown("### Risk Metrics")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="medium")

        with c1:
            smooth_number(
                value=st.session_state.risk_metrics["VaR_pct"] * 100.0,
                key="var95_pct",
                label="VaR (95%)",
                decimals=2,
                duration_ms=900,
                font_px=32,
                suffix="%"
            )

        with c2:
            smooth_number(
                value=st.session_state.risk_metrics["CVaR_pct"] * 100.0,
                key="cvar95_pct",
                label="CVaR (95%)",
                decimals=2,
                duration_ms=900,
                font_px=32,
                suffix="%"
            )

        with c3:
            smooth_number(
                value=st.session_state.risk_metrics["sharpe"],
                key="sharpe",
                label="Sharpe Ratio",
                decimals=2,
                duration_ms=900,
                font_px=32
            )

        with c4:
            smooth_number(
                value=st.session_state.risk_metrics["sortino"],
                key="sortino",
                label="Sortino Ratio",
                decimals=2,
                duration_ms=900,
                font_px=32
            )

# def render_metrics():
#     with metrics_ph.container():
#         st.markdown("### Risk Metrics")
#         c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="medium")

#         with c1:
#             st.metric("VaR (95%)", f"{st.session_state.risk_metrics['VaR_pct'] * 100:.2f}%")
#         with c2:
#             st.metric("CVaR (95%)", f"{st.session_state.risk_metrics['CVaR_pct'] * 100:.2f}%")
#         with c3:
#             st.metric("Sharpe Ratio", f"{st.session_state.risk_metrics['sharpe']:.2f}")
#         with c4:
#             st.metric("Sortino Ratio", f"{st.session_state.risk_metrics['sortino']:.2f}")

render_metrics()
st.divider()

st.markdown("### Monte Carlo simulation")

with st.container():
    col1, col2 = st.columns([1, 1])
    with col1:
        distribution = st.selectbox(
            "Select Distribution",
            options=["Normal", "T", "Empirical"],
            index=0,
            help="Choose the distribution model for simulation."
        )
    with col2:
        forecast_days = st.number_input(
            "Forecast Days",
            min_value=1,
            max_value=252,
            value=30,
            step=1,
            help="Number of days to simulate into the future."
        )

    col3, col4 = st.columns([1, 1])
    with col3:
        start_date = st.date_input(
            "Start Date",
            value=datetime.date(2024, 1, 1),
            help="Select the simulation start date."
        )
    with col4:
        end_date = st.date_input(
            "End Date",
            value=datetime.date(2024, 12, 31),
            help="Select the simulation end date."
        )

    st.divider()

run_sim = st.button("Run Simulation", type="primary")

if run_sim:
    gif_placeholder = st.empty()
    chart_placeholder = st.empty()
    msg_placeholder = st.empty()

    gif_placeholder.image("imgs/giphy.gif", caption="Running simulation...", width="content")

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

        gif_placeholder.empty()
        chart_placeholder.pyplot(fig)

        if missing_tickers:
            msg_placeholder.warning(
                "Skipped tickers with no valid data: " + ", ".join(missing_tickers)
            )

        st.session_state.risk_metrics = {
            "VaR_pct": float(metrics["VaR_pct"]),
            "CVaR_pct": float(metrics["CVaR_pct"]),
            "sharpe": float(metrics["sharpe"]),
            "sortino": float(metrics["sortino"]),
        }

        render_metrics()

    except ValueError as e:
        gif_placeholder.empty()
        chart_placeholder.empty()
        msg_placeholder.error(str(e))