"""Build/import/start the complete local stack. Host requirements: Python + Docker."""
import argparse
from pathlib import Path
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def compose(*args, capture=False):
    return subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True,
                          capture_output=capture, text=True)


def read_env(path):
    # The launcher has no third-party Python dependencies.
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def set_env(path, name, value):
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    matching = [idx for idx,line in enumerate(lines) if line.startswith(name + "=")]
    if matching:
        lines[matching[0]] = name + "=" + str(value)
    else:
        lines.append(name + "=" + str(value))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def available_port(start, count=30):
    for port in range(int(start), int(start)+count):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise RuntimeError("No available local port in the configured range; edit .env.")


def copy_original_assets(source=None):
    destination = ROOT / "imgs"
    if (destination / "carlson_funds_enterprise_logo.jpeg").is_file():
        return True
    if source:
        source = Path(source).expanduser().resolve()
        candidates = [source / "imgs", source]
    else:
        candidates = [ROOT.parent / "imgs", ROOT.parent.parent / "imgs"]
    for candidate in candidates:
        if candidate.resolve() == destination.resolve():
            continue
        if (candidate / "carlson_funds_enterprise_logo.jpeg").is_file():
            shutil.copytree(candidate, destination, dirs_exist_ok=True)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-from", help="Your original project folder (copies imgs only)")
    parser.add_argument("--database-only", action="store_true", help="Build and validate the DB without starting UI/scheduler")
    args = parser.parse_args()
    if not shutil.which("docker"):
        parser.exit(1, "Install/open Docker Desktop first, then rerun python start.py.\n")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    compose("version")
    env_path = ROOT / ".env"
    if not env_path.exists():
        shutil.copyfile(ROOT / ".env.example", env_path)
    config = read_env(env_path)
    if not config.get("POSTGRES_PASSWORD"):
        set_env(env_path, "POSTGRES_PASSWORD", secrets.token_hex(24))
    if os.name != "nt":
        env_path.chmod(0o600)
    for service, setting, default in [("db", "POSTGRES_PORT", 25432), ("app", "STREAMLIT_PORT", 8501)]:
        running = compose("ps", "-q", service, capture=True).stdout.strip()
        if not running:
            port = available_port(config.get(setting) or default)
            set_env(env_path, setting, port)
    (ROOT / "reports").mkdir(exist_ok=True)
    have_logo = copy_original_assets(args.assets_from)
    compose("build", "worker")
    compose("up", "-d", "db")
    compose("run", "--rm", "worker", "python", "import_initial_data.py")
    compose("run", "--rm", "worker", "python", "verify_data.py")
    if args.database_only:
        print("Database import and calculation checks passed.")
        return
    if not have_logo:
        parser.exit(2, "Database is ready. To preserve the original home page, copy your original imgs folder here,\n"
                       "or rerun: python start.py --assets-from \"C:\\path\\to\\your\\original\\carlson_funds\"\n")
    compose("up", "-d", "app", "scheduler")
    config = read_env(env_path)
    url = "http://127.0.0.1:" + config["STREAMLIT_PORT"]
    for _ in range(60):
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.5)
    else:
        raise RuntimeError("App is still starting; check docker compose logs app.")
    print("\nReady: " + url)
    print("API keys: edit ALPACA_API_KEY and ALPACA_SECRET_KEY in this folder's .env.")
    print("After key changes: docker compose up -d --force-recreate scheduler")
    print("Manual update: docker compose run --rm worker python update_daily_prices.py")
    print("Nightly updates run while Docker and the scheduler are running.")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
