"""Main entry point: runs the FastAPI dashboard + orchestrator bot loop.

Used for Railway deployment and local development.
"""

import asyncio
import os
import threading

import uvicorn

from config import settings
from src.utils.logger import setup_logging, get_logger
from src.strategy.orchestrator import Orchestrator
from src.web.app import app

log = get_logger(__name__)


def run_bot_loop():
    """Run the orchestrator in a background thread with its own event loop."""
    setup_logging(settings.log_level)
    log.info("bot_thread_starting", cities=settings.active_cities, paper=settings.paper_mode)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    orchestrator = Orchestrator(settings)
    try:
        loop.run_until_complete(
            orchestrator.run_loop(
                market_interval_minutes=settings.market_poll_interval_minutes,
            )
        )
    except Exception as exc:
        log.error("bot_loop_crashed", error=str(exc))
    finally:
        orchestrator.shutdown()
        loop.close()


def main():
    setup_logging(settings.log_level)

    # Optionally start the automated bot loop in a daemon thread
    if settings.bot_auto_start and settings.orchestrator_enabled:
        bot_thread = threading.Thread(target=run_bot_loop, daemon=True, name="bot-loop")
        bot_thread.start()
        log.info("bot_thread_started")
    elif settings.bot_auto_start:
        log.info("orchestrator_disabled", msg="Orchestrator is off (orchestrator_enabled=False). LLM bots handle betting via the web scheduler.")
    else:
        log.info("bot_auto_start_disabled", msg="Dashboard-only mode. Use /opportunities for manual trading.")

    # Run the FastAPI dashboard on the main thread
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    log.info("dashboard_starting", host=host, port=port)

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
