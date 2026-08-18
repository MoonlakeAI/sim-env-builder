"""Drive the Blender renderer as a subprocess.

Blender is optional. The core suite never imports `bpy`. If Blender is missing
or fails, the report leaves the render section empty and the checks continue.
"""

import json
import logging
import os
import pathlib
import shutil
import subprocess

logger = logging.getLogger(__name__)

SCRIPT = pathlib.Path(__file__).with_name("studio_loop.py")
TIMEOUT_SECONDS = 3600


def find_blender(explicit: str | None) -> str | None:
    return explicit or os.environ.get("BLENDER") or shutil.which("blender")


def command(binary: str, asset: str, out: pathlib.Path, mode: str) -> list[str]:
    return [
        binary,
        "-b",
        "--factory-startup",
        "--python",
        str(SCRIPT),
        "--",
        "--input",
        str(asset),
        "--out",
        str(out),
        "--mode",
        mode,
    ]


def render(asset: str, out: pathlib.Path, mode: str, binary: str | None) -> dict:
    """Render an asset, returning the section to place in the report."""
    if mode == "off":
        return {"status": "skipped"}

    found = find_blender(binary)
    if found is None:
        logger.warning("no Blender binary found; skipping render")
        return {"status": "not_run", "reason": "no Blender binary found"}

    out.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command(found, asset, out, mode),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning("render did not run: %s", error)
        return {"status": "failed", "reason": str(error)}
    (out / "blender.log").write_text(result.stdout + result.stderr)

    manifest = out / "manifest.json"
    if result.returncode != 0 or not manifest.exists():
        logger.warning("render failed with code %s", result.returncode)
        return {"status": "failed", "reason": f"exit code {result.returncode}"}

    try:
        recorded = json.loads(manifest.read_text())
        outputs = [f"render/{name}" for name in recorded["outputs"]]
        exploded = recorded["exploded"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        logger.warning("unreadable render manifest: %s", error)
        return {"status": "failed", "reason": "unreadable render manifest"}
    return {"status": "done", "exploded": exploded, "outputs": outputs}
