from pathlib import Path
import socket

import start


def test_existing_assets_are_copied_without_touching_project_code(tmp_path, monkeypatch):
    original = tmp_path / "original"
    (original / "imgs").mkdir(parents=True)
    logo = original / "imgs/carlson_funds_enterprise_logo.jpeg"
    logo.write_bytes(b"original-fixture-logo")
    destination = original / "local_test"
    destination.mkdir()
    monkeypatch.setattr(start, "ROOT", destination)
    assert start.copy_original_assets()
    assert (destination / "imgs/carlson_funds_enterprise_logo.jpeg").read_bytes() == logo.read_bytes()
    assert not (destination / "home.py").exists()


def test_env_updates_preserve_key_fields_and_port_conflicts_are_avoided(tmp_path):
    path = tmp_path / ".env"
    path.write_text("ALPACA_API_KEY=dummy\nPOSTGRES_PORT=25432\n")
    start.set_env(path,"POSTGRES_PORT",25433)
    assert start.read_env(path)["ALPACA_API_KEY"] == "dummy"
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1",0))
        port = occupied.getsockname()[1]
        assert start.available_port(port) != port
