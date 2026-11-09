"""Settings shared by the local app and nightly workers. Never print secrets."""
from pathlib import Path
import os
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent


def environment():
    values = dict(dotenv_values(os.environ.get("CARLSON_DB_ENV", ROOT / ".env"), encoding="utf-8-sig"))
    values.update(os.environ)
    return values


def data_dir():
    return Path(environment().get("DATA_DIR") or ROOT / "data")


def reports_dir():
    return Path(environment().get("REPORTS_DIR") or ROOT / "reports")
