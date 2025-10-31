import streamlit as st
from streamlit.components.v1 import html
import utils
import datetime



utils.start_reset("19-CG001–Holdings.csv")


# --- 初始化（用 float，不要用字串） ---
if "risk_metrics" not in st.session_state:
    st.session_state.risk_metrics = {
        "VaR_pct": 0.00,      # 例如 -0.0235 表示 -2.35%
        "CVaR_pct": 0.00,     # 例如 -0.0380 表示 -3.80%
        "sharpe": 0.00,
        "sortino": 0.00,
    }


def smooth_number(value: float, key: str, label: str = None,
                  decimals: int = 0, duration_ms: int = 900,
                  font_px: int = 48, prefix: str = "", suffix: str = ""):
    # 1) 確保 map 存在（這行就足夠，不需要另外的初始化區塊）
    prev_map = st.session_state.setdefault("_smooth_prev", {})

    # 2) 讀先前的值（第一次就用當前值：不動畫）
    start_val = prev_map.get(key, 0.0)
    prev_map[key] = value  # 寫回給下次動畫用

    el_id = f"num-{key}"
    html(f"""
    <div style="display:flex;gap:8px;align-items:flex-end;">
    {f'<div style="font-size:{int(font_px*0.5)}px;opacity:.75;color:white;">{label}</div>' if label else ''}
    <div id="{el_id}" style="font-size:{font_px}px;font-weight:700;line-height:1;color:white;"></div>
    </div>
    <script>
    const el = document.getElementById("{el_id}");
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
    const ease = t => 1 - Math.pow(1 - t, 3);  // easeOutCubic
    let startTs = null;
    function animate(ts) {{
        if (!startTs) startTs = ts;
        const p = Math.min(1, (ts - startTs) / duration);
        const cur = startValue + (targetValue - startValue) * ease(p);
        el.textContent = prefix + fmt.format(cur) + suffix;
        if (p < 1) requestAnimationFrame(animate);
        else el.textContent = prefix + fmt.format(targetValue) + suffix;
    }}
    el.textContent = prefix + fmt.format(startValue) + suffix;
    if (startValue !== targetValue && duration > 0) requestAnimationFrame(animate);
    </script>
    """, height=font_px + 24)


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
                value=float(st.session_state.risk_metrics["sharpe"]),
                key="sharpe",
                label="Sharpe Ratio",
                decimals=2,
                duration_ms=900,
                font_px=32
            )
        with c4:
            smooth_number(
                value=float(st.session_state.risk_metrics["sortino"]),
                key="sortino",
                label="Sortino Ratio",
                decimals=2,
                duration_ms=900,
                font_px=32
            )

# --- 第一次渲染（顯示 0，並把 _smooth_prev 的起點設為 0）---
render_metrics()

st.divider()

st.markdown("### Monte Carlo simulation")


with st.container():
    # --- 第一列：分布 + Forecast days ---
    col1, col2 = st.columns([1, 1])
    with col1:
        distribution = st.selectbox(
            "Select Distribution",
            options = ["Normal", "T", "Empirical"],
            index = 0,
            help = "Choose the distribution model for simulation."
        )
    with col2:
        forecast_days = st.number_input(
            "Forecast Days",
            min_value = 1,
            max_value = 252,
            value = 30,
            step = 1,
            help="Number of days to simulate into the future."
        )

    # --- 第二列：Start / End Date ---
    col3, col4 = st.columns([1, 1])
    with col3:
        start_date = st.date_input(
            "Start Date",
            value = datetime.date(2024, 1, 1),
            help = "Select the simulation start date."
        )
    with col4:
        end_date = st.date_input(
            "End Date",
            value = datetime.date(2024, 12, 31),
            help = "Select the simulation end date."
        )

    st.divider()

# --- 按鈕：執行模擬 ---
run_sim = st.button("🚀 Run Simulation", type="primary")

if run_sim: 

    gif_placeholder = st.empty()
    chart_placeholder = st.empty()

    # loading GIF
    gif_placeholder.image("giphy.gif", caption="Running simulation...", width='content')

    # simulation
    fig, VaR, CVaR, VaR_pct, CVaR_pct, sharpe, sortino = utils.simulation(distribution, forecast_days, start_date, end_date)

    # remove GIF
    gif_placeholder.empty()

    chart_placeholder.pyplot(fig)



    # 更新 session_state（保留為 float）
    st.session_state.risk_metrics = {
        # 這裡用百分比的小數（-0.0235），上方顯示時再乘 100 + 加上 %
        "VaR_pct": float(VaR_pct),
        "CVaR_pct": float(CVaR_pct),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
    }

    render_metrics()