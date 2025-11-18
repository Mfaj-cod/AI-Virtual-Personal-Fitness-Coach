from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .pose_estimator import JointAngle


@dataclass(frozen=True)
class ExerciseConfig:
    """Configuration required to count repetitions for a specific exercise."""

    name: str
    target_joint: str
    down_threshold: float
    up_threshold: float
    min_hold_time: float = 0.0  # seconds spent in contracted state before counting


DEFAULT_EXERCISES: Dict[str, ExerciseConfig] = {
    "Push-Up": ExerciseConfig(
        name="Push-Up",
        target_joint="Left Elbow",
        down_threshold=70.0,
        up_threshold=150.0,
        min_hold_time=0.15,
    ),
    "Squat": ExerciseConfig(
        name="Squat",
        target_joint="Left Knee",
        down_threshold=80.0,
        up_threshold=160.0,
        min_hold_time=0.15,
    ),
    "Bicep Curl (Left)": ExerciseConfig(
        name="Bicep Curl (Left)",
        target_joint="Left Elbow",
        down_threshold=45.0,
        up_threshold=160.0,
        min_hold_time=0.1,
    ),
    "Bicep Curl (Right)": ExerciseConfig(
        name="Bicep Curl (Right)",
        target_joint="Right Elbow",
        down_threshold=45.0,
        up_threshold=160.0,
        min_hold_time=0.1,
    ),
    "Lunge (Left)": ExerciseConfig(
        name="Lunge (Left)",
        target_joint="Left Knee",
        down_threshold=70.0,
        up_threshold=160.0,
        min_hold_time=0.15,
    ),
    "Lunge (Right)": ExerciseConfig(
        name="Lunge (Right)",
        target_joint="Right Knee",
        down_threshold=70.0,
        up_threshold=160.0,
        min_hold_time=0.15,
    ),
}


class RepCounter:
    """Count repetitions based on joint angles."""

    def __init__(self, exercise: ExerciseConfig) -> None:
        self.exercise = exercise
        self.count = 0
        self._stage = "up"
        self._last_down_time: Optional[float] = None

    def reset(self) -> None:
        self.count = 0
        self._stage = "up"
        self._last_down_time = None

    def update(self, joint_angles: Dict[str, JointAngle], timestamp: float) -> int:
        """Update the counter given joint angles at a specific timestamp (seconds)."""
        joint_angle = joint_angles.get(self.exercise.target_joint)
        if not joint_angle:
            return self.count

        angle = joint_angle.value
        if angle <= self.exercise.down_threshold:
            self._stage = "down"
            if self._last_down_time is None:
                self._last_down_time = timestamp

        if angle >= self.exercise.up_threshold and self._stage == "down":
            if self._last_down_time is None or (timestamp - self._last_down_time) >= self.exercise.min_hold_time:
                self.count += 1
            self._stage = "up"
            self._last_down_time = None

        return self.count

    @property
    def stage(self) -> str:
        return self._stage

