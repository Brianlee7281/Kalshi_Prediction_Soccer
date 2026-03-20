"""MatchRecorder — saves all live data streams to JSONL for offline replay.

Directory structure (Pattern 7):
    data/recordings/{match_id}/
        odds_api.jsonl      # raw WS messages
        kalshi_ob.jsonl     # orderbook snapshots + deltas
        goalserve.jsonl     # poll responses
        ticks.jsonl         # TickPayload snapshots
        events.jsonl        # detected events
        metadata.json       # match info, start/end times
"""

from __future__ import annotations

import json
import time
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING

from src.common.logging import get_logger

if TYPE_CHECKING:
    from src.common.types import TickPayload

logger = get_logger("recorder")

_FLUSH_EVERY_N = 10
_FLUSH_EVERY_S = 5.0

_STREAM_NAMES = ("odds_api", "kalshi_ob", "goalserve", "ticks", "events", "kalshi_live_data")


class MatchRecorder:
    """Records all live data for a match to JSONL files."""

    def __init__(
        self, match_id: str, base_dir: Path = Path("data/recordings")
    ) -> None:
        self.match_id = match_id
        self.dir = base_dir / match_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self._start_time = time.monotonic()
        self._handles: dict[str, TextIOWrapper] = {}
        self._counts: dict[str, int] = {name: 0 for name in _STREAM_NAMES}
        self._last_flush: float = self._start_time

    def _get_handle(self, stream: str) -> TextIOWrapper:
        """Get or lazily open a file handle for the given stream."""
        handle = self._handles.get(stream)
        if handle is None:
            path = self.dir / f"{stream}.jsonl"
            handle = open(path, "a", encoding="utf-8")  # noqa: SIM115
            self._handles[stream] = handle
        return handle

    def _write(self, stream: str, data: dict) -> None:
        """Write a record with _ts to the given stream."""
        ts = time.monotonic() - self._start_time
        record = {"_ts": ts, **data}
        handle = self._get_handle(stream)
        handle.write(json.dumps(record, default=str) + "\n")
        self._counts[stream] += 1

        # Buffered flush
        total = sum(self._counts.values())
        now = time.monotonic()
        if total % _FLUSH_EVERY_N == 0 or (now - self._last_flush) >= _FLUSH_EVERY_S:
            self._flush_all()
            self._last_flush = now

    def _flush_all(self) -> None:
        """Flush all open file handles."""
        for handle in self._handles.values():
            handle.flush()

    def record_odds_api(self, message: dict) -> None:
        """Append raw Odds-API WS message with _ts."""
        self._write("odds_api", message)

    def record_kalshi_ob(self, data: dict) -> None:
        """Append Kalshi orderbook data with _ts."""
        self._write("kalshi_ob", data)

    def record_goalserve(self, response: dict) -> None:
        """Append Goalserve poll response with _ts."""
        self._write("goalserve", response)

    def record_tick(self, payload: TickPayload) -> None:
        """Append tick snapshot with _ts."""
        self._write("ticks", payload.model_dump())

    def record_event(self, event: dict) -> None:
        """Append detected event with _ts."""
        self._write("events", event)

    def record_kalshi_live_data(self, state_dict: dict) -> None:
        """Append raw Kalshi live_data poll response with _ts."""
        self._write("kalshi_live_data", state_dict)

    def finalize(self) -> None:
        """Write metadata.json, close all file handles."""
        end_time = time.monotonic()
        metadata = {
            "match_id": self.match_id,
            "start_time": self._start_time,
            "end_time": end_time,
            "duration_s": end_time - self._start_time,
            "record_counts": dict(self._counts),
        }
        meta_path = self.dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Flush and close all handles
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self._handles.clear()

        logger.info(
            "recording_finalized",
            match_id=self.match_id,
            counts=self._counts,
        )
