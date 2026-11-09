"""Run the nightly price job inside Docker, independently of open browser sessions."""
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import psycopg

from data_access import DatabaseConfig
from settings import environment
from update_daily_prices import run, target_date


def build_trigger(env):
    return CronTrigger(hour=int(env.get("UPDATE_HOUR") or 22), minute=int(env.get("UPDATE_MINUTE") or 0),
                       timezone=ZoneInfo(env.get("UPDATE_TIMEZONE") or "America/Chicago"))


def credentials_present(env):
    return bool((env.get("ALPACA_API_KEY") or env.get("APCA_API_KEY_ID"))
                and (env.get("ALPACA_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")))


def main():
    env = environment()
    timezone = ZoneInfo(env.get("UPDATE_TIMEZONE") or "America/Chicago")
    scheduler = BlockingScheduler(timezone=timezone)

    def nightly(day=None, attempt=0):
        if not credentials_present(environment()):
            print("Nightly updater waiting for ALPACA_API_KEY / ALPACA_SECRET_KEY in .env.", flush=True)
            return
        day = day or target_date(timezone_name=str(timezone))
        try:
            result = run(day)
            print(json.dumps({k:v for k,v in result.items() if k != "symbols"}), flush=True)
            retry = result["status"] == "partial"
        except Exception as exc:
            print("Nightly update failed (" + type(exc).__name__ + "); see pipeline audit/logs.", flush=True)
            retry = True
        if retry and attempt < 2:
            scheduler.add_job(nightly, "date", run_date=datetime.now(timezone) + timedelta(minutes=30),
                              kwargs={"day": day, "attempt": attempt+1}, id="retry_"+str(day)+"_"+str(attempt),
                              replace_existing=True, misfire_grace_time=3600)

    scheduler.add_job(nightly, build_trigger(env), id="nightly_prices", coalesce=True,
                      max_instances=1, misfire_grace_time=3600)
    if credentials_present(env):
        with psycopg.connect(**DatabaseConfig.from_environment().connection) as conn:
            previous = conn.execute("SELECT stats->>'requested_date' FROM ops.pipeline_runs WHERE source='nightly_prices' AND status='succeeded' ORDER BY finished_at DESC LIMIT 1").fetchone()
        if previous is None or previous[0] < str(target_date(timezone_name=str(timezone))):
            scheduler.add_job(nightly, "date", run_date=datetime.now(timezone), id="startup_catchup", misfire_grace_time=3600)
    print("Nightly scheduler active: " + str(build_trigger(env)) + " " + str(timezone), flush=True)
    scheduler.start()


if __name__ == "__main__":
    main()
