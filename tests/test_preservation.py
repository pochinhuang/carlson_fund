import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_protected_text_and_supplied_inputs_match_manifest():
    manifest = json.loads((ROOT / "preservation_manifest.json").read_text(encoding="utf-8-sig"))
    for group in ["protected_files", "input_files"]:
        for name, expected in manifest[group].items():
            payload = (ROOT / name).read_bytes()
            if ((group == "protected_files" and manifest.get("protected_text_newlines") == "LF")
                    or (group == "input_files" and manifest.get("input_text_newlines") == "LF")):
                payload = payload.replace(b"\r\n", b"\n")
            assert hashlib.sha256(payload).hexdigest() == expected, name


def test_existing_css_is_unchanged():
    manifest = json.loads((ROOT / "preservation_manifest.json").read_text(encoding="utf-8-sig"))
    for name, expected in manifest["styles"].items():
        actual = []
        for node in ast.walk(ast.parse((ROOT / name).read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "<style>" in node.value:
                actual.append(hashlib.sha256(node.value.encode()).hexdigest())
        assert actual == expected, name


def test_home_keeps_link_and_current_documentation_routes():
    text = (ROOT / "home.py").read_text(encoding="utf-8-sig")
    assert "https://www.linkedin.com/in/brian-huang-239884191" in text
    routes = {
        node.args[0].value
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "Page"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    expected = {
        "docs/project_overview.py", "docs/risk_methodology.py", "docs/factor_methodology.py",
        "portfolio_dashboard.py", "risk_analytics.py", "factor_analysis.py",
    }
    assert expected <= routes, expected - routes
    assert not {"fac_docs.py", "portfolio_docs.py"} & routes
    for route in routes:
        assert (ROOT / route).is_file(), route
