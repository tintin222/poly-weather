"""Shared bot status tracking for the orchestrator.

Used by both the orchestrator (writer) and the dashboard (reader)
to show whether auto-betting is active and what the bot is doing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BotStatus:
    """Current state of the auto-betting bot."""

    running: bool = False
    started_at: float = 0.0
    last_cycle_at: float = 0.0
    last_bet_at: float = 0.0
    cycle_count: int = 0
    bets_placed: int = 0
    last_action: str = ""          # human-readable last action
    errors_recent: int = 0         # errors in last hour
    cities_evaluated: int = 0
    recent_log: list[dict] = field(default_factory=list)  # last N events

    def log_event(self, event_type: str, detail: str) -> None:
        """Append an event to the recent log (keeps last 50)."""
        self.recent_log.append({
            "ts": int(time.time()),
            "type": event_type,
            "detail": detail,
        })
        if len(self.recent_log) > 50:
            self.recent_log = self.recent_log[-50:]

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "started_at": self.started_at,
            "last_cycle_at": self.last_cycle_at,
            "last_bet_at": self.last_bet_at,
            "uptime_seconds": int(time.time() - self.started_at) if self.running else 0,
            "cycle_count": self.cycle_count,
            "bets_placed": self.bets_placed,
            "last_action": self.last_action,
            "errors_recent": self.errors_recent,
            "cities_evaluated": self.cities_evaluated,
            "recent_log": self.recent_log[-20:],
        }


# Global singleton — written by orchestrator thread, read by dashboard
bot_status = BotStatus()
