"""Studio turntable + explode-loop renderer for part-segmented GLB/USD assets.

This module runs in headless Blender 5.x. It builds a product-photography set
(off-white cyclorama and four soft area lights) for a GLB or USD with separate
mesh objects. It renders two preview stills or a looping animation.

The animation is one continuous turntable (two full revolutions, linear) with an
eased explode envelope layered on top:

    assembled spin -> explode outward -> exploded spin -> implode -> assembled

The spin completes whole revolutions, and the parts start and end seated. The
last frame meets the first without a visible seam.

Two output modes:
    --mode preview   two stills (assembled + fully exploded) for dialing look
    --mode loop      PNG frame sequence + H.264 loop.mp4 (ffmpeg)

A model whose geometry is one fused mesh cannot explode; it renders as a plain
turntable and the manifest records `exploded: false`.

Usage:
    blender -b --factory-startup --python studio_loop.py -- \
        --input asset.usdz --out OUTDIR [--mode preview|loop] [flags]
"""
import argparse
import colorsys
import math
import os
import sys

import bpy
import mathutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asset_io


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["preview", "loop"], default="preview")
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--res", type=int, default=1080)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--explode", type=float, default=0.6,
                   help="radial explode scale (fraction of each part's offset)")
    p.add_argument("--frames", type=int, default=180, help="total loop frames")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cam_az", type=float, default=35.0)
    p.add_argument("--cam_el", type=float, default=18.0)
    p.add_argument("--view_transform", default="AgX")
    p.add_argument("--wireframe", action="store_true",
                   help="overlay a wireframe on the shaded model")
    p.add_argument("--wire_thickness", type=float, default=0.0009,
                   help="wire thickness as a fraction of model radius")
    p.add_argument("--wire_value", type=float, default=0.85,
                   help="wire brightness 0..1")
    p.add_argument("--wire_decimate", type=float, default=1.0,
                   help="collapse ratio for the wire mesh (1.0 = no decimate)")
    p.add_argument("--wire_emission", type=float, default=0.35,
                   help="wire emission strength (0 = pure diffuse, no bloom)")
    p.add_argument("--light_scale", type=float, default=1.0,
                   help="scale all key/fill/rim/top powers (lower for dark assets)")
    p.add_argument("--exposure", type=float, default=0.0,
                   help="view-transform exposure in stops")
    p.add_argument("--bg_emit", type=float, default=1.0,
                   help="backdrop brightness to camera (emits no scene light)")
    p.add_argument("--highlight_part", default="",
                   help="debug: force parts whose name contains this string to an"
                        " opaque magenta material")
    p.add_argument("--color_parts", action="store_true",
                   help="debug: give every part its own flat hue, to read the"
                        " model's part decomposition and where each part travels"
                        " under explode")
    return p.parse_args(argv)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def flatten_meshes(imported: list) -> list:
    """Return the renderable meshes with world transforms baked in.

    Drops node-empties and helper geometry (see asset_io), leaving independently
    transformable part meshes for the explode animation.

    Order matters. Bake the kept meshes' world transforms (CLEAR_KEEP_TRANSFORM)
    *before* deleting the empties. In glTF and USD, a parent node often carries a
    part's placement. Deleting the parent first discards that translation.
    """
    keep = [o for o in imported if o.type == "MESH" and not asset_io.is_helper(o.name)]
    drop = [o for o in imported if o not in keep]
    bpy.ops.object.select_all(action="DESELECT")
    for m in keep:
        m.select_set(True)
    if keep:
        bpy.context.view_layer.objects.active = keep[0]
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    for o in drop:
        bpy.data.objects.remove(o, do_unlink=True)
    return keep


def world_bbox(objs: list):
    lo = mathutils.Vector((math.inf,) * 3)
    hi = mathutils.Vector((-math.inf,) * 3)
    for o in objs:
        for corner in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(corner)
            lo = mathutils.Vector(map(min, lo, w))
            hi = mathutils.Vector(map(max, hi, w))
    return lo, hi


def obj_world_center(o) -> mathutils.Vector:
    lo, hi = world_bbox([o])
    return (lo + hi) * 0.5


def build_backdrop_material(name, value=0.9, cam_emit=1.0):
    """Backdrop that reads bright off-white to the camera while emitting no light
    into the scene.

    A Light Path 'Is Camera Ray' node gates the emission. Camera rays see a lit
    off-white sweep, while shading rays see only the diffuse base. The base still
    catches the contact shadow and contributes some bounce.

    This separates background brightness from subject exposure. Some assets have
    near-black materials (~0.01). A conventionally emissive backdrop bright
    enough to look white lifts a dark subject to grey. Set subject exposure with
    --light_scale and background brightness with --bg_emit.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    add = nt.nodes.new("ShaderNodeAddShader")
    diff = nt.nodes.new("ShaderNodeBsdfDiffuse")
    emis = nt.nodes.new("ShaderNodeEmission")
    lp = nt.nodes.new("ShaderNodeLightPath")
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = cam_emit
    col = (value, value, value * 0.99, 1.0)
    diff.inputs["Color"].default_value = col
    emis.inputs["Color"].default_value = col
    nt.links.new(lp.outputs["Is Camera Ray"], mul.inputs[0])
    nt.links.new(mul.outputs[0], emis.inputs["Strength"])
    nt.links.new(emis.outputs[0], add.inputs[0])
    nt.links.new(diff.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    return mat


def build_cyclorama(center, radius, mat, az_deg, floor_z) -> None:
    """Build a cyclorama: a floor that sweeps up into a wall via a rounded fillet.

    The floor extends toward -Y, and the wall rises at +Y. Rotation by the camera
    azimuth keeps the floor facing the camera and the wall behind the subject at
    any --cam_az. Without this rotation, the wall can occlude the model.
    """
    import bmesh

    floor = radius * 10
    wall = radius * 8
    fillet = radius * 2.5
    width = radius * 10
    segs = 20
    # Y-Z profile: floor toward -Y (camera) -> rounded corner -> wall up at +Y
    profile = [(-floor, 0.0)]
    for i in range(segs + 1):
        a = math.radians(-90 + 90 * i / segs)  # -90 -> 0
        profile.append((fillet * math.cos(a), fillet + fillet * math.sin(a)))
    profile.append((fillet, wall))

    me = bpy.data.meshes.new("cyc")
    bm = bmesh.new()
    left = [bm.verts.new((-width, y, z)) for (y, z) in profile]
    right = [bm.verts.new((width, y, z)) for (y, z) in profile]
    for i in range(len(profile) - 1):
        bm.faces.new((left[i], left[i + 1], right[i + 1], right[i]))
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("cyclorama", me)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.location = mathutils.Vector((center.x, center.y, floor_z))
    obj.rotation_euler = (0.0, 0.0, math.radians(az_deg))
    for poly in obj.data.polygons:
        poly.use_smooth = True


def add_area(name, loc, target, size, power, color=(1, 1, 1)):
    light = bpy.data.lights.new(name, "AREA")
    light.size = size
    light.energy = power
    light.color = color
    ob = bpy.data.objects.new(name, light)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    d = (mathutils.Vector(target) - mathutils.Vector(loc))
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


def setup_studio(center, radius, az_deg, floor_z, light_scale=1.0, bg_emit=0.12):
    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.9, 0.9, 0.89, 1.0)
    bg.inputs["Strength"].default_value = 0.15
    bpy.context.scene.world = world

    build_cyclorama(center, radius,
                    build_backdrop_material("backdrop", cam_emit=bg_emit),
                    az_deg, floor_z)

    c = mathutils.Vector(center)
    r = radius
    s = light_scale
    # key (front-left, high), fill (front-right, softer), rim (back), top
    add_area("key", c + mathutils.Vector((-2.4 * r, 2.2 * r, 3.0 * r)), c,
             size=3.5 * r, power=560 * r * r * s)
    add_area("fill", c + mathutils.Vector((2.8 * r, 1.6 * r, 1.4 * r)), c,
             size=4.5 * r, power=210 * r * r * s)
    add_area("rim", c + mathutils.Vector((0.5 * r, -3.0 * r, 2.6 * r)), c,
             size=2.5 * r, power=480 * r * r * s, color=(1.0, 0.98, 0.95))
    add_area("top", c + mathutils.Vector((0.0, 0.0, 4.0 * r)), c,
             size=6.0 * r, power=280 * r * r * s)


def setup_camera(center, radius, az_deg, el_deg, res):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 85
    cam_data.sensor_width = 36
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)

    az, el = math.radians(az_deg), math.radians(el_deg)
    # `radius` is the exploded bounding-sphere radius; 1.11 leaves ~10% margin
    hfov = 2 * math.atan((cam_data.sensor_width / 2) / cam_data.lens)
    dist = (radius / math.tan(hfov / 2)) * 1.11
    dirv = mathutils.Vector((math.cos(el) * math.sin(az),
                             -math.cos(el) * math.cos(az),
                             math.sin(el)))
    cam.location = mathutils.Vector(center) + dirv * dist
    look = (mathutils.Vector(center) - cam.location)
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def add_wireframe(meshes, thickness, value, decimate=1.0, emission=0.35):
    """Overlay a wireframe on the shaded model ("wireframe-on-shaded").

    Adds wire geometry with the Wireframe modifier and retains the original shaded
    faces (use_replace=False). This works identically in EEVEE and Cycles, unlike
    viewport-only overlays.

    Two settings that decide whether the result is readable:

    * use_even_offset is OFF. Even Thickness scales the offset by 1/sin(angle)
      between adjacent faces, which diverges at sharp or degenerate edges and
      throws long spikes off clean hard-surface meshes.
    * `decimate` < 1.0 collapses the mesh before wiring it. Dense
      photogrammetry-style meshes (10^5-10^6 verts) have so many edges that a
      full-resolution wireframe fills in to a solid mass. Leave at 1.0 to show
      true topology. Lower it only to make a dense mesh legible, since the wires
      then no longer depict the real edges.
    """
    wire = bpy.data.materials.new("wire")
    wire.use_nodes = True
    bsdf = wire.node_tree.nodes["Principled BSDF"]
    col = (value, value, value, 1.0)
    bsdf.inputs["Base Color"].default_value = col
    try:
        bsdf.inputs["Emission Color"].default_value = col
        bsdf.inputs["Emission Strength"].default_value = emission
    except KeyError:
        pass
    for m in meshes:
        m.data.materials.append(wire)
        idx = len(m.data.materials) - 1
        if decimate < 1.0:
            dec = m.modifiers.new("dec", "DECIMATE")
            dec.decimate_type = "COLLAPSE"
            dec.ratio = decimate
        mod = m.modifiers.new("wire", "WIREFRAME")
        mod.thickness = thickness
        mod.use_replace = False        # keep the shaded original faces
        mod.material_offset = idx      # wire skin uses the appended material
        # Even Thickness offsets by 1/sin(angle) between faces, causing large
        # artifacts at sharp or acute edges on clean CAD meshes. Keep it off.
        mod.use_even_offset = False
        mod.use_boundary = True


def color_parts(meshes):
    """Repaint each part a distinct flat hue.

    A shaded render cannot distinguish neighbouring parts that share a material.
    This function assigns hues by index. A golden-ratio step separates adjacent
    indices, reducing the chance that neighbouring parts receive similar colours.
    """
    for i, m in enumerate(meshes):
        hue = (i * 0.61803398875) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.62, 0.9)
        mat = bpy.data.materials.new(f"partcol_{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.55
        m.data.materials.clear()
        m.data.materials.append(mat)


def setup_turntable(meshes, pivot):
    empty = bpy.data.objects.new("turntable", None)
    bpy.context.collection.objects.link(empty)
    empty.location = pivot
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = empty
    empty.select_set(True)
    bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
    return empty


def separate_concentric(meshes, offsets, radius, explode):
    """Slide concentric parts apart so a nested assembly actually opens.

    The radial rule scales a part's offset by its distance from the explode
    origin. Parts that share a centre receive small offsets in nearly the same
    direction and travel as a block.

    Concentric parts are detected by centre proximity relative to their own size,
    not to the model: two shells 5 mm apart on a 40 mm product are concentric, the
    same two on a 4 m product are not.

    The function separates each group along its thinnest axis. It steps by each
    member's extent on that axis, largest member first. Co-centred parts usually
    follow this axis: a lid over a bowl, plier plates side by side, or the shells
    of a chair back. Moving them by their thickness separates them while limiting
    growth of the exploded bounding box. Vertical stacking works for squat
    appliances but makes tall assemblies too large for useful framing.
    """
    names = [m.name for m in meshes]
    centre = {m.name: obj_world_center(m) for m in meshes}
    extent = {}
    for m in meshes:
        lo, hi = world_bbox([m])
        extent[m.name] = hi - lo

    parent = {n: n for n in names}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            span = min(extent[a].length, extent[b].length)
            if (centre[a] - centre[b]).length < 0.2 * span:
                parent[find(a)] = find(b)

    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)

    for members in groups.values():
        if len(members) < 2:
            continue
        glo, ghi = world_bbox([m for m in meshes if m.name in members])
        axis = min(range(3), key=lambda k: (ghi - glo)[k])
        order = sorted(members, key=lambda n: -extent[n].length)
        slide = 0.0
        for n in order:
            step = mathutils.Vector((0.0, 0.0, 0.0))
            step[axis] = slide * explode
            offsets[n] = offsets[n] + step
            slide += extent[n][axis]
        print(f"[studio] concentric group of {len(members)} on {'XYZ'[axis]}: "
              f"{', '.join(order)}")


def part_offsets(meshes, origin, explode):
    """Per-mesh world-space explode offset, radiating from `origin`.

    Each part moves along the vector from `origin` to its own centre, scaled by
    `explode`, so the assembly opens up while roughly preserving its layout.

    `origin` sits near the model's base rather than its centre (see main()): parts
    then fan upward and outward instead of downward, which keeps low parts from
    driving through the floor at full explode.

    Parts whose centre is nearly on the vertical axis have no meaningful radial
    direction, so they are fanned out on an index-derived bearing instead of
    being left in place.

    A radial offset does not open every assembly. See
    separate_concentric(), which runs afterwards.
    """
    lo, hi = world_bbox(meshes)
    radius = max(hi - lo) * 0.5
    offsets = {}
    for i, m in enumerate(meshes):
        d = obj_world_center(m) - origin
        if mathutils.Vector((d.x, d.y)).length < 0.12 * radius:  # central: nudge out
            ang = 2 * math.pi * i / max(1, len(meshes))
            d = mathutils.Vector((math.cos(ang) * 0.3 * radius,
                                  math.sin(ang) * 0.3 * radius, d.z))
        offsets[m.name] = d * explode
    separate_concentric(meshes, offsets, radius, explode)
    return offsets


def exploded_bounds(meshes, offsets, pivot):
    """World bbox min/max when exploded, and the bounding-sphere radius about `pivot`.

    Measure the radius about the camera's aim point, the assembled centre, rather
    than the exploded bounding-box centre. Asymmetric displacement moves the
    exploded centre away from the aim point. Measuring from that centre
    underreports the extent and clips the farthest parts.
    """
    lo = mathutils.Vector((math.inf,) * 3)
    hi = mathutils.Vector((-math.inf,) * 3)
    for m in meshes:
        o = offsets[m.name]
        for corner in m.bound_box:
            w = (m.matrix_world @ mathutils.Vector(corner)) + o
            lo = mathutils.Vector(map(min, lo, w))
            hi = mathutils.Vector(map(max, hi, w))
    r = 0.0
    for m in meshes:
        o = offsets[m.name]
        for corner in m.bound_box:
            w = (m.matrix_world @ mathutils.Vector(corner)) + o
            r = max(r, (w - pivot).length)
    return lo, hi, r


def fcurves_of(obj):
    """F-curves for an object across old and 5.x slotted-action APIs."""
    ad = obj.animation_data
    if not ad or not ad.action:
        return []
    act = ad.action
    fcs = getattr(act, "fcurves", None)
    if fcs is not None:
        return list(fcs)
    out = []
    for layer in getattr(act, "layers", []):
        for strip in getattr(layer, "strips", []):
            for cb in getattr(strip, "channelbags", []):
                out.extend(cb.fcurves)
    return out


def key_loc(m, frame, loc):
    # Default Bezier interpolation eases the explode and implode.
    m.location = loc
    m.keyframe_insert("location", frame=frame)


def animate_loop(empty, meshes, offsets, F):
    """Key the loop: constant spin + eased explode envelope.

    Spin: 720 degrees with LINEAR interpolation, keyed at frames 1 and F+1. Whole
    revolutions and linear timing avoid a position or speed jump at the wrap
    point. Keying one frame past the last rendered frame avoids a duplicate at
    the seam.

    Explode: Bezier-interpolated position keys at fractions of F, holding assembled
    at both ends so the envelope also wraps cleanly.
    """
    rest = {m.name: m.location.copy() for m in meshes}
    empty.rotation_euler = (0, 0, 0)
    empty.keyframe_insert("rotation_euler", frame=1)
    empty.rotation_euler = (0, 0, 4 * math.pi)
    empty.keyframe_insert("rotation_euler", frame=F + 1)
    for fc in fcurves_of(empty):
        if "rotation" in fc.data_path:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

    # Envelope timing as fractions of the loop: hold assembled to a0, open out by
    # a1, hold exploded to b0, close back by b1, hold assembled to the end.
    a0, a1, b0, b1 = 0.20, 0.34, 0.62, 0.78
    for m in meshes:
        r = rest[m.name]
        e = r + offsets[m.name]
        key_loc(m, 1, r)
        key_loc(m, 1 + a0 * F, r)
        key_loc(m, 1 + a1 * F, e)
        key_loc(m, 1 + b0 * F, e)
        key_loc(m, 1 + b1 * F, r)
        key_loc(m, F + 1, r)


def set_explode_static(meshes, offsets, factor):
    for m in meshes:
        m.location = m.location + offsets[m.name] * factor
    bpy.context.view_layer.update()


def setup_render(engine, res, samples, fps, view_transform, exposure=0.0):
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.fps = fps
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = view_transform
        scene.view_settings.exposure = exposure
    except Exception as exc:  # noqa: BLE001, settings names vary by Blender version
        print("[studio] view transform skipped:", exc)
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "METAL"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = True
            scene.cycles.device = "GPU"
        except Exception as exc:  # noqa: BLE001, settings names vary by Blender version
            print("[studio] GPU setup failed, using CPU:", exc)
    else:  # EEVEE Next, with soft ray-traced shadows for a studio look
        try:
            scene.eevee.taa_render_samples = samples
            scene.eevee.use_raytracing = True
        except Exception as exc:  # noqa: BLE001, settings names vary by Blender version
            print("[studio] eevee settings skipped:", exc)


def main():
    args = parse_args()
    reset_scene()
    imported = asset_io.import_asset(args.input)
    meshes = flatten_meshes(imported)
    if not meshes:
        raise SystemExit(f"no mesh objects in {args.input}")
    # Render models at native scale. Derive light power and distance, backdrop
    # dimensions, camera distance, and wire thickness from the measured radius.
    # Individual rescaling displaces parts because exporters use inconsistent
    # part origins.
    lo, hi = world_bbox(meshes)
    pivot = (lo + hi) * 0.5
    assembled_radius = max(hi - lo) * 0.5
    height = hi.z - lo.z
    explode_origin = mathutils.Vector((pivot.x, pivot.y, lo.z + 0.22 * height))
    offsets = part_offsets(meshes, explode_origin, args.explode)
    # Fit the frame and floor to the exploded extent so the widest frame fits and
    # no part sinks through the floor. The assembled phase appears slightly smaller.
    elo, _, exploded_radius = exploded_bounds(meshes, offsets, pivot)
    floor_z = elo.z - 0.02 * assembled_radius

    setup_studio(pivot, assembled_radius, args.cam_az, floor_z,
                 args.light_scale, args.bg_emit)
    setup_camera(pivot, exploded_radius, args.cam_az, args.cam_el, args.res)
    setup_render(args.engine, args.res, args.samples, args.fps,
                 args.view_transform, args.exposure)

    if args.highlight_part:
        hot = bpy.data.materials.new("highlight")
        hot.use_nodes = True
        hot.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 1, 1)
        hits = 0
        for m in meshes:
            if args.highlight_part.lower() in m.name.lower():
                m.data.materials.clear()
                m.data.materials.append(hot)
                hits += 1
        print(f"[studio] highlighted {hits} part(s) matching '{args.highlight_part}'")

    if args.color_parts:
        color_parts(meshes)
        print(f"[studio] coloured {len(meshes)} part(s)")

    if args.wireframe:
        add_wireframe(meshes, args.wire_thickness * assembled_radius,
                      args.wire_value, args.wire_decimate, args.wire_emission)

    if args.mode == "preview":
        # assembled still
        bpy.context.scene.render.filepath = f"{args.out}/preview_assembled.png"
        bpy.ops.render.render(write_still=True)
        # exploded still
        set_explode_static(meshes, offsets, 1.0)
        bpy.context.scene.render.filepath = f"{args.out}/preview_exploded.png"
        bpy.ops.render.render(write_still=True)
        write_manifest(args.out, meshes, ["preview_assembled.png", "preview_exploded.png"])
        print("[studio] preview stills written to", args.out)
        return

    # loop mode
    empty = setup_turntable(meshes, pivot)
    animate_loop(empty, meshes, offsets, args.frames)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = args.frames
    # PNG frames are resumable. Skip frames already on disk so an interrupted
    # render (e.g. machine sleep) continues instead of restarting.
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = f"{args.out}/frames/f_"
    scene.render.use_overwrite = False
    scene.render.use_placeholder = False
    bpy.ops.render.render(animation=True)
    # Encode the MP4 from existing frames without rendering again.
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(args.fps), "-start_number", "1",
         "-i", f"{args.out}/frames/f_%04d.png", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
         f"{args.out}/loop.mp4"],
        check=False,
    )
    write_manifest(args.out, meshes, ["loop.mp4"])
    print("[studio] loop written to", args.out)


def write_manifest(out: str, meshes: list, outputs: list) -> None:
    """Record what was rendered, for the caller to fold into its report."""
    import json
    manifest = {"parts": len(meshes), "exploded": len(meshes) > 1, "outputs": outputs}
    with open(f"{out}/manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)


main()
