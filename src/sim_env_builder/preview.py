"""Generate white-background GIFs of each task's intended joint motion.

Renders in Blender (no robot, no HDRI) by driving the USD physics joint
that the task instruction refers to, then stitches frames into a looping GIF.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .articulation import (
    match_joint_for_instruction,
    moving_body_paths,
    parse_articulation,
)
from .paths import REPO_ROOT, ensure_library_extracted
from .tasks import TASK_SPECS

BLENDER_SCRIPT = Path(__file__).with_name("blender_preview.py")
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "media" / "tasks"


def _job_for_task(task_name: str, frames_dir: Path, width: int, height: int, n_frames: int) -> dict:
    spec = TASK_SPECS[task_name]
    usd = ensure_library_extracted(spec.asset)
    joints = parse_articulation(usd)
    joint = match_joint_for_instruction(joints, spec.instruction, spec.asset)
    moving = moving_body_paths(joint, joints)
    if not moving:
        raise ValueError(f"{task_name}: joint {joint.name} has no body1")
    return {
        "task": task_name,
        "usd": str(usd),
        "out_dir": str(frames_dir / task_name),
        "width": width,
        "height": height,
        "frames": n_frames,
        "hold": 4,
        "joint": {
            "name": joint.name,
            "joint_type": joint.joint_type,
            "axis": joint.axis,
            "rest": joint.rest_value,
            "open": joint.open_value,
            "body0": joint.body0,
            "body1": joint.body1,
            "local_pos0": list(joint.local_pos0),
            "local_rot0": list(joint.local_rot0),
            "local_pos1": list(joint.local_pos1),
            "local_rot1": list(joint.local_rot1),
        },
        "moving_bodies": moving,
    }


def _write_gif(frame_dir: Path, dest: Path, duration_ms: int = 70) -> None:
    paths = sorted(frame_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    images = [Image.open(p).convert("RGB") for p in paths]
    dest.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        dest,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for im in images:
        im.close()
    print(f"[preview] {len(paths)} frames -> {dest}")


def generate_task_gifs(
    tasks: list[str] | None = None,
    out_dir: Path | None = None,
    blender: str | None = None,
    width: int = 320,
    height: int = 180,
    n_frames: int = 28,
) -> list[Path]:
    blender_bin = blender or shutil.which("blender")
    if not blender_bin:
        raise SystemExit("blender not found on PATH; install Blender 4.5+ to render previews")
    if not BLENDER_SCRIPT.exists():
        raise SystemExit(f"missing blender script: {BLENDER_SCRIPT}")

    names = tasks or list(TASK_SPECS)
    unknown = [n for n in names if n not in TASK_SPECS]
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")
    dest_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR

    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="sim-env-preview-") as tmp:
        tmp_path = Path(tmp)
        jobs = [_job_for_task(name, tmp_path, width, height, n_frames) for name in names]
        print(f"[preview] {len(jobs)} task(s) via {blender_bin}")
        for job in jobs:
            jobs_path = tmp_path / f"{job['task']}.json"
            jobs_path.write_text(json.dumps([job], indent=2))
            cmd = [
                blender_bin,
                "--background",
                "--python",
                str(BLENDER_SCRIPT),
                "--",
                str(jobs_path),
            ]
            print(f"[preview] rendering {job['task']} ({job['joint']['name']})")
            try:
                subprocess.run(cmd, check=True)
                gif = dest_dir / f"{job['task']}.gif"
                _write_gif(Path(job["out_dir"]), gif)
                print(f"[preview] wrote {gif}")
                written.append(gif)
            except Exception as exc:
                print(f"[preview] FAILED {job['task']}: {exc}")
    return written
