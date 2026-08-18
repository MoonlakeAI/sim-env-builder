"""Tests for USD articulation parsing and milestone derivation (no Isaac needed).

All tests run against the synthetic fixture in tests/data/ or inline
JointSpecs; none depend on the bundled asset library.
"""

import math
import zipfile
from pathlib import Path

from sim_env_builder.articulation import (
    PRISMATIC,
    REVOLUTE,
    JointSpec,
    derive_milestones,
    match_joint_for_instruction,
    moving_body_paths,
    parse_articulation,
    pick_openable_joint,
)

FIXTURE = Path(__file__).parent / "data" / "cabinet.usda"


def test_parse_fixture():
    joints = {j.name: j for j in parse_articulation(FIXTURE)}
    assert len(joints) == 3
    door = joints["door_pivot"]
    assert door.joint_type == REVOLUTE
    assert (door.lower, door.upper) == (0.0, 90.0)
    drawer = joints["drawer_pivot"]
    assert drawer.joint_type == PRISMATIC
    assert math.isclose(drawer.upper, 0.4, rel_tol=1e-6)


def test_parse_usdz_matches_usda(tmp_path):
    usdz = tmp_path / "cabinet.usdz"
    with zipfile.ZipFile(usdz, "w") as z:
        z.write(FIXTURE, "scene.usda")
    from_zip = {j.name for j in parse_articulation(usdz)}
    from_text = {j.name for j in parse_articulation(FIXTURE)}
    assert from_zip == from_text == {"door_pivot", "drawer_pivot", "button_01_pivot"}


def test_derive_milestones_fixture():
    joints = parse_articulation(FIXTURE)
    milestones = {m.name: m for m in derive_milestones(joints)}
    door = milestones["open the door"]
    assert math.isclose(door.threshold, 45.0)  # 50% of the 0-90 deg excursion
    assert door.instruction("cabinet") == "open the cabinet door"
    assert milestones["pull out the drawer"].verb == "pull out"
    assert milestones["press the button 01"].verb == "press"


def test_parse_joint_frames():
    door = next(j for j in parse_articulation(FIXTURE) if j.name == "door_pivot")
    assert door.axis == "Z"
    assert door.body1.endswith("/door")


def test_pick_openable_prefers_door():
    joints = parse_articulation(FIXTURE)
    assert pick_openable_joint(joints).name == "door_pivot"


def test_pick_openable_prefers_top_lid_when_stacked():
    joints = [
        JointSpec(name="base_lid_hinge_pivot", joint_type=REVOLUTE, lower=0.0, upper=90.0),
        JointSpec(name="mid_lid_hinge_pivot", joint_type=REVOLUTE, lower=0.0, upper=90.0),
        JointSpec(name="top_lid_hinge_pivot", joint_type=REVOLUTE, lower=0.0, upper=90.0),
    ]
    assert pick_openable_joint(joints).name == "top_lid_hinge_pivot"


def test_match_instruction_uses_part_tokens():
    joints = [
        JointSpec(name="base_lid_hinge_pivot", joint_type=REVOLUTE, lower=0.0, upper=90.0),
        JointSpec(name="mid_lid_hinge_pivot", joint_type=REVOLUTE, lower=0.0, upper=90.0),
        JointSpec(name="top_lid_hinge_pivot", joint_type=REVOLUTE, lower=0.0, upper=90.0),
    ]
    assert (
        match_joint_for_instruction(joints, "open the box base lid", "storage_box").name
        == "base_lid_hinge_pivot"
    )


def test_match_instruction_falls_back_to_openable():
    joints = parse_articulation(FIXTURE)
    assert match_joint_for_instruction(joints, "open the cabinet", "cabinet").name == "door_pivot"


def test_moving_body_includes_children():
    joints = [
        JointSpec(
            name="door_pivot",
            joint_type=REVOLUTE,
            lower=0.0,
            upper=90.0,
            body0="/base",
            body1="/door",
        ),
        JointSpec(
            name="button_pivot",
            joint_type=PRISMATIC,
            lower=0.0,
            upper=0.002,
            body0="/door",
            body1="/button",
        ),
    ]
    assert moving_body_paths(joints[0], joints) == ["/door", "/button"]


def test_pick_openable_matches_whole_words_only():
    # "slide" must not match the "lid" pattern: the largest revolute wins
    # over a prismatic whose name merely contains the letters "lid".
    joints = [
        JointSpec(name="speed_slide_pivot", joint_type=PRISMATIC, lower=0.0, upper=0.05),
        JointSpec(name="head_tilt_pivot", joint_type=REVOLUTE, lower=0.0, upper=45.0),
    ]
    assert pick_openable_joint(joints).name == "head_tilt_pivot"
    joints.append(JointSpec(name="lid_pivot", joint_type=REVOLUTE, lower=0.0, upper=30.0))
    assert pick_openable_joint(joints).name == "lid_pivot"


def test_milestone_threshold_override():
    joints = [JointSpec(name="door_pivot", joint_type=REVOLUTE, lower=0.0, upper=90.0)]
    (m,) = derive_milestones(joints, overrides={"door_pivot": 30.0})
    assert math.isclose(m.threshold, 30.0)
    assert m.grade(31.0)["achieved"] is True
    assert m.grade(29.0)["achieved"] is False


def test_grade_reports_progress_on_failure():
    joints = parse_articulation(FIXTURE)
    door = next(m for m in derive_milestones(joints) if m.joint.name == "door_pivot")
    g = door.grade(12.0)
    assert g["achieved"] is False
    assert math.isclose(g["max_door_angle_deg"], 12.0)
    assert math.isclose(g["progress"], 12.0 / 90.0, rel_tol=1e-3)


