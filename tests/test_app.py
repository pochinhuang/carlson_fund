import base64
import os
from pathlib import Path

import numpy as np
import pytest

from data_access import DataAccess

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not os.environ.get("CARLSON_TEST_DATABASE_URL"), reason="Dedicated imported DB required")


def clean(app):
    assert not app.exception, [x.message for x in app.exception]
    assert not app.error, [x.value for x in app.error]


def test_dashboard_and_sector_controls_and_charts():
    from streamlit.testing.v1 import AppTest
    dashboard = AppTest.from_file(str(ROOT / "portfolio_dashboard.py")).run(timeout=90)
    clean(dashboard)
    assert len(dashboard.get("plotly_chart")) == 3
    assert len(dashboard.dataframe) == 1
    sector = AppTest.from_file(str(ROOT / "factor_analysis.py")).run(timeout=90)
    clean(sector)
    sector.selectbox[0].set_value("Healthcare")
    sector.button[0].click().run(timeout=90)
    clean(sector)
    holdings = DataAccess().get_holdings("r2500g_proxy")
    assert set(sector.dataframe[0].value.index) == set(holdings.loc[holdings.Sector == "Healthcare", "Ticker"])
    assert len(sector.get("plotly_chart")) == 6


def test_simulation_all_original_distribution_options():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "risk_analytics.py")).run(timeout=90)
    clean(app)
    assert app.selectbox[0].options == ["Normal", "T", "Empirical"]
    assert [x.label for x in app.date_input] == ["Start Date", "End Date"]
    for distribution in ["Normal", "T", "Empirical"]:
        app.selectbox[0].set_value(distribution)
        app.button[0].click().run(timeout=90)
        clean(app)
        assert np.isfinite(app.session_state.risk_metrics["VaR_pct"])
        assert np.isfinite(app.session_state.risk_metrics["CVaR_pct"])
    app.date_input[0].set_value(app.date_input[1].value)
    app.button[0].click().run(timeout=90)
    assert not app.exception
    assert "20 common daily returns" in app.error[0].value


def test_original_home_and_documentation_routes_with_fixture_asset():
    from streamlit.testing.v1 import AppTest
    # Only a disposable image fixture tests the file lookup. The original logo
    # was not supplied, and this fixture is NEVER included in the deliverable.
    path = ROOT / "imgs/carlson_funds_enterprise_logo.jpeg"
    existed = path.exists()
    if not existed:
        path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aU1sAAAAASUVORK5CYII="))
    try:
        app = AppTest.from_file(str(ROOT / "home.py")).run(timeout=90)
        clean(app)
        app.switch_page("docs/factor_methodology.py").run(timeout=90)
        clean(app)
        assert len(app.markdown) > 0
        app.switch_page("docs/risk_methodology.py").run(timeout=90)
        clean(app)
    finally:
        if not existed:
            path.unlink(missing_ok=True)
