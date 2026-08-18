"""Tests for episode scoring from joint state (no Isaac needed).

Milestones are built from inline JointSpecs; nothing here depends on asset
fixtures or USD parsing (covered by test_articulation.py).
"""

import math

from sim_env_builder.articulation import (
    PRISMATIC,
    REVOLUTE,
    JointSpec,
    derive_milestones,
)
from sim_env_builder.scoring import EpisodeRecord, summarize


def _record(episode=0):
    joints = [
        JointSpec(name="door_pivot", joint_type=REVOLUTE, lower=0.0, upper=90.0),
        JointSpec(name="drawer_pivot", joint_type=PRISMATIC, lower=0.0, upper=0.4),
    ]
    return EpisodeRecord(
        milestones=derive_milestones(joints),
        instruction="open the cabinet",
        episode=episode,
        asset_name="cabinet",
    )


def test_episode_success_from_radians():
    rec = _record()
    # Sim reports revolute joints in radians: sweep the door 0 -> 52.3 deg.
    for angle_deg in (0.0, 20.0, 52.3, 40.0):
        rec.update({"door_pivot": math.radians(angle_deg), "drawer_pivot": 0.01})
    result = rec.result(target_milestone="open the cabinet door")
    assert result["success"] is True
    assert result["milestones"]["open the cabinet door"] is True
    assert result["milestones"]["pull out the cabinet drawer"] is False
    assert math.isclose(result["progress"]["max_door_angle_deg"], 52.3, abs_tol=0.01)


def test_zero_success_still_reports_progress():
    rec = _record()
    rec.update({"door_pivot": 0.0})
    rec.update({"door_pivot": math.radians(12.0)})
    result = rec.result(target_milestone="open the cabinet door")
    assert result["success"] is False
    assert math.isclose(result["progress"]["max_door_angle_deg"], 12.0, abs_tol=0.01)
    line = summarize([result])
    assert line.startswith("[FAIL]")
    assert "max_door_angle_deg=12.0" in line


def test_long_horizon_open_then_close():
    rec = _record()
    # Door sweeps 0 -> 60 deg -> back to 5 deg: opened, then closed again.
    for angle_deg in (0.0, 30.0, 60.0, 30.0, 5.0):
        rec.update({"door_pivot": math.radians(angle_deg)})
    result = rec.result(target_milestone="open the cabinet door", require_close=True)
    assert result["success"] is True
    assert result["long_horizon"]["open the cabinet door, then close it"] is True

    # Opened but left open: the plain milestone passes, the sequence fails.
    rec2 = _record()
    for angle_deg in (0.0, 60.0, 55.0):
        rec2.update({"door_pivot": math.radians(angle_deg)})
    result2 = rec2.result(target_milestone="open the cabinet door", require_close=True)
    assert result2["milestones"]["open the cabinet door"] is True
    assert result2["success"] is False


def test_result_includes_timeseries():
    rec = _record()
    sweep = (0.0, 15.0, 30.0, 45.0, 52.3)
    for angle_deg in sweep:
        rec.update({"door_pivot": math.radians(angle_deg), "drawer_pivot": 0.01})
    result = rec.result(target_milestone="open the cabinet door")
    ts = result["timeseries"]
    assert ts["video_fps"] == 30
    assert len(ts["joints"]["door_pivot"]) == len(sweep)
    assert len(ts["joints"]["drawer_pivot"]) == len(sweep)
    # Normalized progress: monotone sweep of a 0-90 deg hinge ends near 0.58.
    assert ts["joints"]["door_pivot"] == sorted(ts["joints"]["door_pivot"])
    assert math.isclose(ts["joints"]["door_pivot"][-1], 52.3 / 90.0, abs_tol=0.001)
    for entry in result["detail"]:
        assert 0.0 <= entry["threshold_fraction"] <= 1.0


def test_result_without_updates_has_no_timeseries():
    result = _record().result(target_milestone="open the cabinet door")
    assert "timeseries" not in result


def test_summarize_pass_line():
    r1 = _record(0)
    r1.update({"door_pivot": 0.0})
    r1.update({"door_pivot": math.radians(50.0)})
    r2 = _record(1)
    r2.update({"door_pivot": math.radians(5.0)})
    results = [
        r1.result(target_milestone="open the cabinet door"),
        r2.result(target_milestone="open the cabinet door"),
    ]
    line = summarize(results)
    assert line.startswith("[PASS] 1/2")


def test_signed_range_joint_graded_from_rest():
    # A slider with limits [-0.025, 0.025] and rest at 0: pushing to either
    # end is 100% progress, and the threshold is half the max excursion.
    joint = JointSpec(name="speed_slide_pivot", joint_type=PRISMATIC, lower=-0.025, upper=0.025)
    (m,) = derive_milestones([joint])
    assert math.isclose(m.threshold, 0.0125)
    rec = EpisodeRecord(milestones=[m], instruction="slide it", episode=0, asset_name="mixer")
    rec.update({"speed_slide_pivot": 0.0})
    rec.update({"speed_slide_pivot": -0.02})
    result = rec.result()
    assert result["milestones"]["pull out the mixer speed slide"] is True
    assert math.isclose(result["progress"]["max_speed_slide_travel_m"], 0.02)


def test_reversed_limits_normalized():
    joint = JointSpec(name="hose_swing_pivot", joint_type=REVOLUTE, lower=25.0, upper=-25.0)
    assert (joint.lower, joint.upper) == (-25.0, 25.0)


def test_limit_violation_invalidates_episode():
    rec = _record()
    rec.update({"door_pivot": 0.0})
    # 340 deg on a 0-90 deg hinge: a physics fault, not manipulation.
    rec.update({"door_pivot": math.radians(340.0)})
    result = rec.result(target_milestone="open the cabinet door")
    assert result["success"] is False
    assert "door_pivot" in result["limit_violations"]
