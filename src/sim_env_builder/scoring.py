"""Episode scoring from articulation state, with no VLM judge or labeling.

During a rollout we record raw joint positions every step. At episode end
we grade the peaks against the milestones derived from the asset USD
(see `articulation.py`). A failed episode still yields progress metrics
(e.g. the door reached 12 of the 45 required degrees); that partial-progress
readout is the product.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .articulation import REVOLUTE, JointSpec, Milestone

# A joint that returns below this progress fraction (after having reached its
# milestone threshold) counts as "closed again": the second half of
# long-horizon chains like "open the cabinet and then close it".
CLOSE_FRACTION = 0.15

# A joint observed beyond its authored limits by more than this fraction of
# its range indicates a physics fault (impulse through a limit); such an
# episode cannot be trusted and is graded as failed.
LIMIT_VIOLATION_MARGIN = 0.10

# One camera-video frame is written per env step at 30 fps (see tracking.py),
# so video_time_seconds = step_index / VIDEO_FPS. The dashboard relies on this
# to align the milestone timeline with the rollout videos.
VIDEO_FPS = 30


@dataclass
class EpisodeRecord:
    """Accumulates joint-state peaks for one episode."""

    milestones: list[Milestone]
    instruction: str
    episode: int
    # Asset name used to phrase milestone labels ("open the cabinet door"
    # instead of "open the door") so the JSON reads like an instruction.
    asset_name: str = ""
    # Isaac/PhysX reports revolute joint positions in radians; USD limits
    # (and our thresholds) are in degrees.
    revolute_in_radians: bool = True
    _peaks: dict[str, float] = field(default_factory=dict)
    _reached: dict[str, bool] = field(default_factory=dict)
    _closed_after: dict[str, bool] = field(default_factory=dict)
    # Per-step normalized progress per joint, for timeline visualization.
    _history: dict[str, list[float]] = field(default_factory=dict)
    _limit_violations: dict[str, float] = field(default_factory=dict)
    steps: int = 0

    def update(self, joint_positions: dict[str, float]) -> None:
        """Feed current joint positions (name -> scalar) for one sim step."""
        self.steps += 1
        for name, value in joint_positions.items():
            value = float(value)
            spec = self._spec(name)
            if spec is None:
                continue
            if spec.joint_type == REVOLUTE and self.revolute_in_radians:
                value = math.degrees(value)
            # The first observed sim value is the reset state: ground truth
            # for "closed". Specs are shared across episodes, so this fires
            # once per run.
            if spec.rest is None:
                spec.rest = value
            margin = LIMIT_VIOLATION_MARGIN * spec.range
            if value < spec.lower - margin or value > spec.upper + margin:
                worst = self._limit_violations.get(name)
                if worst is None or abs(value) > abs(worst):
                    self._limit_violations[name] = round(value, 4)
            progress = spec.fraction(value)
            prev = self._peaks.get(name)
            # Peak = value at the largest displacement from rest, so joints
            # that open by decreasing grade the same as increasing ones.
            if prev is None or progress > spec.fraction(prev):
                self._peaks[name] = value
            self._history.setdefault(name, []).append(round(progress, 4))
            m = self._milestone(name)
            if m is not None:
                if progress >= m.threshold_fraction:
                    self._reached[name] = True
                elif self._reached.get(name) and progress <= CLOSE_FRACTION:
                    self._closed_after[name] = True

    def _milestone(self, joint_name: str) -> Milestone | None:
        for m in self.milestones:
            if m.joint.name == joint_name:
                return m
        return None

    def _spec(self, joint_name: str) -> JointSpec | None:
        for m in self.milestones:
            if m.joint.name == joint_name:
                return m.joint
        return None

    def result(self, target_milestone: str | None = None, require_close: bool = False) -> dict:
        """Grade the episode.

        `target_milestone`: milestone name that defines episode success
        (e.g. "open the door"). If None, success = any milestone achieved.
        `require_close`: long-horizon mode: success additionally requires the
        target joint to return near closed after reaching its threshold.
        """
        graded = []
        long_horizon = {}
        for m in self.milestones:
            peak = self._peaks.get(m.joint.name)
            if peak is None:
                continue
            g = m.grade(peak)
            if self.asset_name:
                g["milestone"] = m.instruction(self.asset_name)
            g["then_closed"] = bool(self._closed_after.get(m.joint.name, False))
            g.setdefault("threshold_fraction", round(m.threshold_fraction, 4))
            graded.append(g)
            # "…, then close it" only makes sense for parts that open. Buttons
            # (verb "press") spring back by design and are excluded.
            if m.verb in ("open", "pull out", "tilt") and self._reached.get(m.joint.name):
                long_horizon[f"{g['milestone']}, then close it"] = g["then_closed"]

        flat = {g["milestone"]: g["achieved"] for g in graded}
        progress_metrics = {}
        for g in graded:
            for key, value in g.items():
                if key.startswith("max_"):
                    progress_metrics[key] = value
            progress_metrics[f"{g['joint']}_progress"] = g["progress"]

        if target_milestone is not None:
            success = bool(flat.get(target_milestone, False))
        else:
            success = any(flat.values())
        if require_close:
            success = success and bool(long_horizon.get(f"{target_milestone}, then close it"))

        # A limit-violating joint invalidates the episode: the peak came from
        # a physics fault, not from manipulation the asset supports.
        if self._limit_violations:
            success = False

        out = {
            "episode": self.episode,
            "instruction": self.instruction,
            "success": success,
            "target_milestone": target_milestone,
            "require_close": require_close,
            "steps": self.steps,
            "milestones": flat,
            "limit_violations": self._limit_violations,
            "long_horizon": long_horizon,
            "progress": progress_metrics,
            "detail": graded,
        }
        if self._history:
            out["timeseries"] = {"video_fps": VIDEO_FPS, "joints": dict(self._history)}
        return out


def write_episode_json(result: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{result['episode']:03d}_milestones.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def summarize(results: list[dict]) -> str:
    """The pass/fail line printed at the end of a run."""
    n = len(results)
    passed = sum(1 for r in results if r["success"])
    verdict = "PASS" if passed > 0 else "FAIL"
    # Report the best progress metrics even when every episode failed; the
    # partial progress they capture is the signal.
    best = {}
    for r in results:
        for key, value in r["progress"].items():
            if key.startswith("max_") and (key not in best or value > best[key]):
                best[key] = value
    best_str = ", ".join(f"{k}={v}" for k, v in sorted(best.items()))
    return f"[{verdict}] {passed}/{n} episodes succeeded | best: {best_str}"
