import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_protected_files_and_supplied_inputs_are_byte_identical():
    manifest = json.loads((ROOT / "preservation_manifest.json").read_text())
    for group in ["protected_files", "input_files"]:
        for name, expected in manifest[group].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_existing_css_is_unchanged():
    manifest = json.loads((ROOT / "preservation_manifest.json").read_text())
    for name, expected in manifest["styles"].items():
        actual = []
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and '<style>' in node.value:
                actual.append(hashlib.sha256(node.value.encode()).hexdigest())
        assert actual == expected, name


def test_home_keeps_link_and_all_documentation_routes():
    text = (ROOT / "home.py").read_text()
    assert "https://www.linkedin.com/in/brian-huang-239884191" in text
    assert 'st.Page("fac_docs.py"' in text and 'st.Page("portfolio_docs.py"' in text
