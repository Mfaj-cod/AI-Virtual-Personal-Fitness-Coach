from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class WorkoutEntry:
    timestamp: datetime
    exercise: str
    reps: int
    duration_sec: float
    notes: str = ""


class WorkoutLogger:
    """Persist workout history to disk."""

    def __init__(self, filepath: str = "data/workout_logs.csv") -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: WorkoutEntry) -> None:
        """Add a new entry to the log."""
        df = pd.DataFrame([self._entry_to_record(entry)])
        if self.filepath.exists():
            df.to_csv(self.filepath, mode="a", header=False, index=False)
        else:
            df.to_csv(self.filepath, index=False)

    def load_history(self, limit: Optional[int] = 25) -> pd.DataFrame:
        if not self.filepath.exists():
            return pd.DataFrame(columns=["timestamp", "exercise", "reps", "duration_sec", "notes"])
        df = pd.read_csv(self.filepath, parse_dates=["timestamp"])
        if limit:
            return df.tail(limit)
        return df

    @staticmethod
    def _entry_to_record(entry: WorkoutEntry) -> dict:
        record = asdict(entry)
        record["timestamp"] = entry.timestamp.isoformat()
        return record

