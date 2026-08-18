"""Tests for the rollout review dashboard (no sockets, no Isaac)."""

import json

from sim_env_builder.dashboard import build_index, resolve_under

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42fake"

MILESTONE_JSON = {
    "episode": 0,
    "instruction": "open the cabinet",
    "success": False,
    "target_milestone": "open the cabinet door",
    "steps": 3,
    "milestones": {"open the cabinet door": False},
    "progress": {"max_door_angle_deg": 12.0},
    "detail": [
        {
            "milestone": "open the cabinet door",
            "joint": "door_pivot",
            "achieved": False,
            "progress": 0.13,
            "threshold": 45.0,
            "threshold_fraction": 0.5,
        }
    ],
    "timeseries": {"video_fps": 30, "joints": {"door_pivot": [0.0, 0.07, 0.13]}},
}


def _make_run(base, task, stamp, episodes, with_milestones=True):
    run = base / task / stamp
    run.mkdir(parents=True)
    (run / "rl-video-step-0.mp4").write_bytes(FAKE_MP4)
    cams = ["external_camera_rgb", "external_camera_2_rgb", "wrist_camera_rgb"]
    for ep in range(episodes):
        for cam in cams:
            (run / f"robot-cam-env0-{cam}-episode-{ep}.mp4").write_bytes(FAKE_MP4)
    if with_milestones:
        ms_dir = run / "milestones"
        ms_dir.mkdir()
        for ep in range(episodes):
            data = dict(MILESTONE_JSON, episode=ep)
            (ms_dir / f"episode_{ep:03d}_milestones.json").write_text(json.dumps(data))
    return run


def test_build_index_two_runs(tmp_path):
    _make_run(tmp_path, "open-cabinet", "2026-01-01_00-00-00", episodes=2)
    _make_run(tmp_path, "open-printer", "2026-01-02_00-00-00", episodes=1,
              with_milestones=False)
    (tmp_path / "stray.json").write_text("{}")

    index = build_index(tmp_path)
    assert index["root"] == str(tmp_path)
    # Newest run first, regardless of task name.
    assert [r["run_id"] for r in index["runs"]] == [
        "open-printer/2026-01-02_00-00-00",
        "open-cabinet/2026-01-01_00-00-00",
    ]

    dish = index["runs"][1]
    assert dish["task"] == "open-cabinet"
    assert dish["viewport"] == "open-cabinet/2026-01-01_00-00-00/rl-video-step-0.mp4"
    assert [e["episode"] for e in dish["episodes"]] == [0, 1]
    ep0 = dish["episodes"][0]
    prefix = "open-cabinet/2026-01-01_00-00-00/robot-cam-env0-"
    assert ep0["videos"] == {
        "external": prefix + "external_camera_rgb-episode-0.mp4",
        "base": prefix + "external_camera_2_rgb-episode-0.mp4",
        "wrist": prefix + "wrist_camera_rgb-episode-0.mp4",
    }
    assert ep0["milestones"]["instruction"] == "open the cabinet"
    assert ep0["milestones"]["timeseries"]["video_fps"] == 30
    assert ep0["milestones"]["timeseries"]["joints"]["door_pivot"] == [0.0, 0.07, 0.13]
    assert ep0["milestones"]["detail"][0]["threshold_fraction"] == 0.5


def test_build_index_handles_missing_milestones(tmp_path):
    _make_run(tmp_path, "open-printer", "2026-01-02_00-00-00", episodes=1,
              with_milestones=False)
    ep = build_index(tmp_path)["runs"][0]["episodes"][0]
    assert ep["milestones"] is None


def test_build_index_runs_sorted_newest_first(tmp_path):
    _make_run(tmp_path, "open-cabinet", "2026-01-01_00-00-00", episodes=1)
    _make_run(tmp_path, "open-cabinet", "2026-01-03_00-00-00", episodes=1)
    stamps = [r["timestamp"] for r in build_index(tmp_path)["runs"]]
    assert stamps == ["2026-01-03_00-00-00", "2026-01-01_00-00-00"]


def test_build_index_empty_dir(tmp_path):
    assert build_index(tmp_path)["runs"] == []
    assert build_index(tmp_path / "missing")["runs"] == []


def test_resolve_under_blocks_escape(tmp_path):
    run = _make_run(tmp_path, "open-cabinet", "2026-01-01_00-00-00", episodes=1)
    inside = resolve_under(tmp_path, "open-cabinet/2026-01-01_00-00-00/rl-video-step-0.mp4")
    assert inside == run / "rl-video-step-0.mp4"
    (tmp_path.parent / "secret.txt").write_text("x")
    assert resolve_under(tmp_path, "../secret.txt") is None
    assert resolve_under(tmp_path, "/etc/passwd") is None
    assert resolve_under(tmp_path, "open-cabinet") is None
