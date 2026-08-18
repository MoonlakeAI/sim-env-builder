"""Repository paths. Keep this import-safe (no Isaac/pxr imports)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
OUTPUTS_DIR = REPO_ROOT / "outputs"


LIBRARY_DIR = ASSETS_DIR / "library"


def ensure_library_extracted(asset_name: str) -> "Path":
    """Unpack a bundled library .usdz on first use (extracted/ is gitignored)."""
    usd = LIBRARY_DIR / "extracted" / asset_name / "scene.usda"
    if not usd.exists():
        import zipfile  # noqa: PLC0415

        with zipfile.ZipFile(LIBRARY_DIR / f"{asset_name}.usdz") as z:
            z.extractall(usd.parent)
    deactivate_embedded_lights(usd)
    return usd


def disable_usd_instancing(usd_path: "str | Path") -> str:
    """Write a sibling copy with every instanceable prim made regular.

    The DROID robot USD marks most Robotiq gripper meshes (and the stand)
    instanceable. With Fabric physics, instanced visual prims never receive
    transform updates in the offscreen camera render path, so they draw frozen
    at their authored pose: the moving gripper appears to detach and only the
    few non-instanced meshes (fingertips) follow the arm in recorded videos.
    De-instancing turns them into ordinary meshes that update like the rest of
    the robot.

    The source file is left untouched (upstream regenerates it on every boot)
    and the copy is written via tempfile + atomic replace, so parallel rollout
    processes sharing the cache directory never observe a half-written robot.
    Returns the path of the de-instanced copy.
    """
    import os  # noqa: PLC0415

    from pxr import Usd  # noqa: PLC0415

    src = Path(usd_path)
    out = src.with_name(f"{src.stem}_noinstance{src.suffix}")
    stage = Usd.Stage.Open(str(src))
    while instances := [p for p in stage.Traverse() if p.IsInstance()]:
        for prim in instances:
            prim.SetInstanceable(False)
    tmp = src.with_name(f".{out.stem}.{os.getpid()}{src.suffix}")
    stage.GetRootLayer().Export(str(tmp))
    os.replace(tmp, out)
    return str(out)


def deactivate_embedded_lights(usd_path: "Path") -> bool:
    """Deactivate light prims that some library exports embed (idempotent).

    A few packages ship a studio preview rig with their own DomeLight; RTX
    supports a single dome light per stage, so the extra one turns the HDRI
    background black. Scene lighting belongs to the environment, not the
    asset, so all embedded lights are deactivated in place. Returns whether
    the file was modified.
    """
    from pxr import Usd  # noqa: PLC0415

    stage = Usd.Stage.Open(str(usd_path))
    lights = [p for p in stage.Traverse() if p.GetTypeName().endswith("Light")]
    for prim in lights:
        prim.SetActive(False)
    if lights:
        stage.Save()
    return bool(lights)
