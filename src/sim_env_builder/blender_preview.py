"""Blender-side renderer for articulation preview GIFs.

Invoked by `preview.py` as:

    blender --background --python blender_preview.py -- jobs.json

Do not import sim_env_builder here; this file runs under Blender's Python.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

AXIS_VEC = {
    "X": Vector((1.0, 0.0, 0.0)),
    "Y": Vector((0.0, 1.0, 0.0)),
    "Z": Vector((0.0, 0.0, 1.0)),
}


def _argv_after_double_dash() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def _quat(wxyz) -> Quaternion:
    w, x, y, z = wxyz
    return Quaternion((w, x, y, z))


def _frame_matrix(pos, rot_wxyz) -> Matrix:
    mat = _quat(rot_wxyz).to_matrix().to_4x4()
    mat.translation = Vector(pos)
    return mat


def _prim_name(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def _name_variants(prim_path: str) -> list[str]:
    last = _prim_name(prim_path)
    dotted = re.sub(r"_(\d+)$", r".\1", last)
    return [last, dotted]


def find_object(prim_path: str):
    variants = set(_name_variants(prim_path))
    matches = []
    for obj in bpy.data.objects:
        if obj.name in variants:
            matches.append(obj)
    if matches:
        return sorted(matches, key=lambda o: len(o.name), reverse=True)[0]
    suffix = [obj for obj in bpy.data.objects if any(obj.name.endswith(v) for v in variants)]
    if suffix:
        return sorted(suffix, key=lambda o: len(o.name), reverse=True)[0]
    raise LookupError(
        f"no Blender object for prim {prim_path!r}; "
        f"objects={sorted(o.name for o in bpy.data.objects)[:40]}"
    )


def reset_scene() -> None:
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    if bpy.context.selected_objects:
        bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.curves,
        bpy.data.armatures,
    ):
        for item in list(collection):
            collection.remove(item)


def import_usd(path: str) -> None:
    kwargs = dict(
        filepath=path,
        import_cameras=False,
        import_lights=False,
        create_world_material=False,
        import_usd_preview=True,
        import_materials=True,
        relative_path=False,
        set_frame_range=False,
    )
    result = bpy.ops.wm.usd_import(**kwargs)
    if result != {"FINISHED"}:
        raise RuntimeError(f"USD import failed for {path}: {result}")


VIEW_DIR = Vector((0.45, -1.0, 0.35)).normalized()
LENS_MM = 50.0
SENSOR_MM = 36.0
# Fraction of the frame the motion AABB may occupy (rest of the image is margin).
FRAME_FILL = 0.86


def _is_framing_mesh(obj) -> bool:
    if obj.type != "MESH" or not obj.visible_get():
        return False
    # Authored extended, so it inflates the AABB and shrinks the rest of the asset.
    return "telescopic" not in obj.name.lower()


def mesh_aabb() -> tuple[Vector, Vector]:
    mins = Vector((math.inf, math.inf, math.inf))
    maxs = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in bpy.data.objects:
        if not _is_framing_mesh(obj):
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, world.x)
            mins.y = min(mins.y, world.y)
            mins.z = min(mins.z, world.z)
            maxs.x = max(maxs.x, world.x)
            maxs.y = max(maxs.y, world.y)
            maxs.z = max(maxs.z, world.z)
            found = True
    if not found:
        return Vector((-0.5, -0.5, -0.5)), Vector((0.5, 0.5, 0.5))
    return mins, maxs


def union_aabb(
    first: tuple[Vector, Vector], second: tuple[Vector, Vector]
) -> tuple[Vector, Vector]:
    (amin, amax), (bmin, bmax) = first, second
    return (
        Vector((min(amin.x, bmin.x), min(amin.y, bmin.y), min(amin.z, bmin.z))),
        Vector((max(amax.x, bmax.x), max(amax.y, bmax.y), max(amax.z, bmax.z))),
    )


def aabb_corners(mins: Vector, maxs: Vector):
    for x in (mins.x, maxs.x):
        for y in (mins.y, maxs.y):
            for z in (mins.z, maxs.z):
                yield Vector((x, y, z))


def aabb_center(mins: Vector, maxs: Vector) -> Vector:
    return (mins + maxs) * 0.5


def place_camera(cam, look_at: Vector, corners: list, aspect: float) -> float:
    """Push the camera back along VIEW_DIR until every corner of the motion
    AABB sits inside the frame. Look-at is the rest-pose center so the asset
    stays visually centered instead of drifting when a lid swings up."""
    tan_x = (SENSOR_MM * 0.5) / LENS_MM
    tan_y = (SENSOR_MM / aspect * 0.5) / LENS_MM
    cam.rotation_euler = (-VIEW_DIR).to_track_quat("-Z", "Y").to_euler()
    lo, hi = 0.05, 80.0
    for _ in range(22):
        mid = (lo + hi) * 0.5
        cam.location = look_at + VIEW_DIR * mid
        bpy.context.view_layer.update()
        inv = cam.matrix_world.inverted()
        fits = True
        for corner in corners:
            loc = inv @ corner
            depth = -loc.z
            if depth < 1e-4:
                fits = False
                break
            if abs(loc.x) > FRAME_FILL * tan_x * depth:
                fits = False
                break
            if abs(loc.y) > FRAME_FILL * tan_y * depth:
                fits = False
                break
        if fits:
            hi = mid
        else:
            lo = mid
    cam.location = look_at + VIEW_DIR * hi
    bpy.context.view_layer.update()
    return hi


def setup_studio(
    look_at: Vector,
    union_mins: Vector,
    union_maxs: Vector,
    width: int,
    height: int,
) -> None:
    scene = bpy.context.scene
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    else:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 32
        if hasattr(scene.eevee, "use_raytracing"):
            scene.eevee.use_raytracing = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.dither_intensity = 0.0
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 0.45
    out = nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])
    scene.world = world

    aspect = width / max(height, 1)
    size = max((union_maxs - union_mins).length, 0.1)
    corners = list(aabb_corners(union_mins, union_maxs))

    cam_data = bpy.data.cameras.new("preview")
    cam_data.lens = LENS_MM
    cam_data.sensor_width = SENSOR_MM
    cam_data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("preview", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    dist = place_camera(cam, look_at, corners, aspect)
    cam_data.clip_start = max(dist * 0.02, 0.01)
    cam_data.clip_end = max(dist * 8.0, 10.0)

    def add_light(name, energy, location):
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.size = max(0.4, size * 0.25)
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (look_at - location).to_track_quat("-Z", "Y").to_euler()

    add_light("key", energy=25, location=look_at + Vector((size * 0.7, -size * 1.0, size * 1.1)))
    add_light("fill", energy=8, location=look_at + Vector((-size * 0.9, -size * 0.5, size * 0.5)))
    add_light("rim", energy=10, location=look_at + Vector((size * 0.2, size * 0.8, size * 0.7)))


def parent_keep_world(child, parent) -> None:
    mw = child.matrix_world.copy()
    child.parent = parent
    child.matrix_world = mw


def attach_joint(job: dict):
    joint = job["joint"]
    body0 = find_object(joint["body0"])
    j0 = body0.matrix_world @ _frame_matrix(joint["local_pos0"], joint["local_rot0"])
    empty = bpy.data.objects.new(f"drive_{joint['name']}", None)
    empty.empty_display_type = "PLAIN_AXES"
    bpy.context.scene.collection.objects.link(empty)
    empty.matrix_world = j0
    for path in job["moving_bodies"]:
        parent_keep_world(find_object(path), empty)
    return empty, j0, joint


def apply_joint(empty, rest_matrix: Matrix, joint: dict, u: float) -> None:
    rest = float(joint["rest"])
    target = float(joint["open"])
    value = rest + (target - rest) * u
    delta = value - rest
    axis = AXIS_VEC[joint["axis"]]
    if joint["joint_type"] == "revolute":
        motion = Quaternion(axis, math.radians(delta)).to_matrix().to_4x4()
    else:
        motion = Matrix.Translation(axis * delta)
    empty.matrix_world = rest_matrix @ motion
    bpy.context.view_layer.update()


def ease(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def motion_samples(n_frames: int, hold: int) -> list[float]:
    n_move = max((n_frames - hold) // 2, 1)
    opening = [ease(i / max(n_move - 1, 1)) for i in range(n_move)]
    held = [1.0] * hold
    closing = list(reversed(opening))
    samples = opening + held + closing
    while len(samples) < n_frames:
        samples.append(0.0)
    return samples[:n_frames]


def motion_bounds(empty, rest_matrix, joint):
    apply_joint(empty, rest_matrix, joint, 0.0)
    rest = mesh_aabb()
    bounds = rest
    for u in (0.5, 1.0):
        apply_joint(empty, rest_matrix, joint, u)
        bounds = union_aabb(bounds, mesh_aabb())
    apply_joint(empty, rest_matrix, joint, 0.0)
    return rest, bounds


def render_job(job: dict) -> None:
    reset_scene()
    import_usd(job["usd"])
    empty, rest_matrix, joint = attach_joint(job)
    bpy.context.view_layer.update()
    _rest, union = motion_bounds(empty, rest_matrix, joint)
    setup_studio(aabb_center(*union), union[0], union[1], job["width"], job["height"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = motion_samples(job["frames"], job.get("hold", 4))
    scene = bpy.context.scene
    for i, u in enumerate(samples):
        apply_joint(empty, rest_matrix, joint, u)
        scene.render.filepath = str(out_dir / f"frame_{i:03d}")
        bpy.ops.render.render(write_still=True)


def main() -> None:
    args = _argv_after_double_dash()
    if not args:
        raise SystemExit("usage: blender --background --python blender_preview.py -- jobs.json")
    jobs = json.loads(Path(args[0]).read_text())
    for job in jobs:
        print(f"[blender-preview] {job['task']}  joint={job['joint']['name']}")
        render_job(job)


if __name__ == "__main__":
    main()
