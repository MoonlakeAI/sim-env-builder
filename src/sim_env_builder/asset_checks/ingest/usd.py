"""Load a USD stage into the normalized asset model."""

import io
import logging

import numpy as np
import PIL.Image
from pxr import Ar, Gf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel

from sim_env_builder.asset_checks import geometry
from sim_env_builder.asset_checks.ingest import model

logger = logging.getLogger(__name__)

STATIC_LINK = model.STATIC_LINK

# Map each UsdPreviewSurface input to the slot used by material checks.
TEXTURE_SLOTS = {
    "diffuseColor": "basecolor",
    "normal": "normal",
    "displacement": "displacement",
    "roughness": "roughness",
    "metallic": "metallic",
    "occlusion": "occlusion",
    "emissiveColor": "emissive",
    "opacity": "opacity",
}

DERIVED_SHAPES = {
    "convexHull": "convex_hull",
    "boundingCube": "primitive",
    "boundingSphere": "primitive",
}


def load(path: str) -> model.AssetModel:
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise ValueError(f"could not open USD stage: {path}")
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    materials: dict[str, model.Material] = {}
    parts: list[model.Part] = []
    proxies: list[model.Proxy] = []
    link_prims: dict[str, Usd.Prim] = {}
    shape_mass: dict[str, float] = {}

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            link_prims[str(prim.GetPath())] = prim
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        if not prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        link = _owning_link(prim)
        purpose = UsdGeom.Imageable(prim).ComputePurpose()
        mass = _authored(UsdPhysics.MassAPI(prim).GetMassAttr())
        if mass is not None and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            shape_mass[link] = shape_mass.get(link, 0.0) + mass

        if prim.IsA(UsdGeom.Mesh):
            transform = np.array(cache.GetLocalToWorldTransform(prim)).T
            part = _read_mesh(prim, transform, scale, link, materials, stage)
            if part is None:
                continue
            renders = purpose != UsdGeom.Tokens.proxy
            if renders:
                parts.append(part)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                proxies.append(
                    _to_proxy(prim, part.vertices, part.triangles, link, renders)
                )
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            proxies.append(_bounds_proxy(prim, cache, scale, link))

    links = _build_links(parts, proxies, link_prims, shape_mass, cache, scale)
    joints = _read_joints(stage, cache, scale)
    return model.AssetModel(
        path=path,
        format="usd",
        parts=parts,
        proxies=proxies,
        materials=materials,
        joints=joints,
        links=links,
        meters_per_unit=scale,
        bbox_diag=model.bounding_diagonal(parts),
    )


def _owning_link(prim: Usd.Prim) -> str:
    """Nearest ancestor carrying a rigid body, or the static link."""
    node = prim
    while node and not node.IsPseudoRoot():
        if node.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(node.GetPath())
        node = node.GetParent()
    return STATIC_LINK


def _read_mesh(
    prim: Usd.Prim,
    transform: np.ndarray,
    scale: float,
    link: str,
    materials: dict[str, model.Material],
    stage: Usd.Stage,
) -> model.Part | None:
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts:
        return None

    vertices = np.asarray(points, dtype=np.float64)
    vertices = (vertices @ transform[:3, :3].T + transform[:3, 3]) * scale
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)

    slots = _fan_slots(counts)
    triangles = indices[slots]
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        triangles = triangles[:, ::-1]
        slots = slots[:, ::-1]

    material = _read_binding(prim, materials, stage)
    return model.Part(
        name=str(prim.GetPath()),
        link=link,
        vertices=vertices,
        triangles=triangles,
        face_counts=counts,
        face_indices=indices,
        normals=_read_normals(mesh, len(vertices), transform),
        uvs=_read_uvs(mesh, slots, triangles),
        material=material,
        skin=_read_skin(prim, len(vertices)),
    )


def _fan_slots(counts: np.ndarray) -> np.ndarray:
    """Fan-triangulate polygons, returning indices into the face-vertex array."""
    per_face = np.maximum(counts - 2, 0)
    face_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tri_starts = np.concatenate([[0], np.cumsum(per_face)[:-1]])
    face_of_tri = np.repeat(np.arange(len(counts)), per_face)
    within = np.arange(per_face.sum()) - np.repeat(tri_starts, per_face)
    base = face_starts[face_of_tri]
    return np.stack([base, base + within + 1, base + within + 2], axis=1)


def _read_normals(
    mesh: UsdGeom.Mesh, count: int, transform: np.ndarray
) -> np.ndarray | None:
    values = mesh.GetNormalsAttr().Get()
    if values is None or len(values) != count:
        return None
    return _transform_normals(np.asarray(values), transform)


def _transform_normals(normals: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform normals by the inverse transpose.

    The inverse transpose differs from the linear matrix under non-uniform scale.
    """
    linear = transform[:3, :3]
    try:
        inverse = np.linalg.inv(linear)
    except np.linalg.LinAlgError:
        inverse = linear.T  # degenerate transform; flagged elsewhere
    return geometry.normalize(normals @ inverse)


def _read_uvs(
    mesh: UsdGeom.Mesh, slots: np.ndarray, triangles: np.ndarray
) -> np.ndarray | None:
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
    if not primvar or primvar.Get() is None:
        return None
    values = np.asarray(primvar.Get(), dtype=np.float64)

    # An indexed primvar stores a table of unique values plus one index per
    # element; resolve it to the flat per-element array first.
    indices = primvar.GetIndices()
    if indices:
        table = np.asarray(indices)
        if table.max(initial=0) >= len(values):
            return None
        values = values[table]

    interpolation = primvar.GetInterpolation()
    if interpolation == UsdGeom.Tokens.faceVarying:
        corners = slots
    elif interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        corners = triangles
    else:
        return None  # uniform/constant primvars carry no per-corner mapping
    if corners.max(initial=0) >= len(values):
        return None
    return values[corners]


def _read_skin(prim: Usd.Prim, count: int) -> model.Skin | None:
    binding = UsdSkel.BindingAPI(prim)
    indices = binding.GetJointIndicesPrimvar()
    weights = binding.GetJointWeightsPrimvar()
    if not indices or indices.Get() is None or weights.Get() is None:
        return None
    influences = indices.GetElementSize() or 1
    joints = binding.GetJointsAttr().Get() or []
    return model.Skin(
        joints=[str(j) for j in joints],
        indices=np.asarray(indices.Get()).reshape(count, influences),
        weights=np.asarray(weights.Get(), dtype=np.float64).reshape(count, influences),
    )


def _read_binding(
    prim: Usd.Prim, materials: dict[str, model.Material], stage: Usd.Stage
) -> str | None:
    """Bound material name, falling back to the first bound GeomSubset."""
    names = []
    for source in [prim] + list(prim.GetChildren()):
        bound = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial()[0]
        if bound:
            name = str(bound.GetPath())
            if name not in materials:
                materials[name] = _read_material(bound, stage)
            names.append(name)
    return names[0] if names else None


def _read_material(bound: UsdShade.Material, stage: Usd.Stage) -> model.Material:
    surface = bound.ComputeSurfaceSource()[0]
    params: dict[str, float | tuple] = {}
    textures: dict[str, model.Texture] = {}
    shader_id = None

    if surface:
        shader_id = surface.GetShaderId()
        for shader_input in surface.GetInputs():
            slot = TEXTURE_SLOTS.get(shader_input.GetBaseName())
            source = shader_input.GetConnectedSource()
            if source and slot:
                texture = _read_texture(UsdShade.Shader(source[0].GetPrim()), stage)
                if texture:
                    textures[slot] = texture
            elif not source and shader_input.Get() is not None:
                params[shader_input.GetBaseName()] = _scalar(shader_input.Get())
    return model.Material(
        name=str(bound.GetPath()), shader=shader_id, params=params, textures=textures
    )


def _read_texture(shader: UsdShade.Shader, stage: Usd.Stage) -> model.Texture | None:
    asset = shader.GetInput("file")
    if not asset or asset.Get() is None:
        return None
    path = asset.Get().resolvedPath or asset.Get().path
    return model.Texture(path=path, image=_open_image(path))


def _open_image(path: str) -> PIL.Image.Image | None:
    try:
        asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(path))
        data = asset.GetBuffer() if asset else None
        image = PIL.Image.open(io.BytesIO(data) if data else path)
        image.load()
        return image
    except Exception:  # noqa: BLE001, any decode failure means unresolved
        logger.warning("unreadable texture: %s", path)
        return None


def _to_proxy(
    prim: Usd.Prim,
    vertices: np.ndarray,
    triangles: np.ndarray,
    link: str,
    is_render_mesh: bool,
) -> model.Proxy:
    """Build the simulator's collision shape."""
    approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
    source = DERIVED_SHAPES.get(
        approximation, "render_mesh" if is_render_mesh else "authored_mesh"
    )

    if approximation == "convexHull":
        vertices, triangles = geometry.convex_hull(vertices)
    elif approximation == "boundingCube":
        vertices, triangles = geometry.box_mesh(
            np.stack([vertices.min(axis=0), vertices.max(axis=0)])
        )
    elif approximation == "boundingSphere":
        center = vertices.mean(axis=0)
        radius = float(np.linalg.norm(vertices - center, axis=1).max())
        vertices, triangles = geometry.sphere_mesh(center, radius)

    return _make_proxy(prim, source, vertices, triangles, link)


def _bounds_proxy(
    prim: Usd.Prim, cache: UsdGeom.XformCache, scale: float, link: str
) -> model.Proxy:
    """Approximate non-mesh collision shapes by their bounds."""
    bounds = UsdGeom.Imageable(prim).ComputeWorldBound(
        Usd.TimeCode.Default(), UsdGeom.Tokens.default_
    )
    box = bounds.ComputeAlignedRange()
    corners = np.array([list(box.GetMin()), list(box.GetMax())]) * scale
    vertices, triangles = geometry.box_mesh(corners)
    return _make_proxy(prim, "primitive", vertices, triangles, link)


def _make_proxy(
    prim: Usd.Prim,
    source: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    link: str,
) -> model.Proxy:
    physics = _physics_material(prim)
    return model.Proxy(
        name=str(prim.GetPath()),
        link=link,
        source=source,
        vertices=vertices,
        triangles=triangles,
        static_friction=physics.get("staticFriction"),
        dynamic_friction=physics.get("dynamicFriction"),
        restitution=physics.get("restitution"),
    )


def _physics_material(prim: Usd.Prim) -> dict[str, float]:
    bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
    if not bound or not bound.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        return {}
    api = UsdPhysics.MaterialAPI(bound.GetPrim())
    values = {
        "staticFriction": api.GetStaticFrictionAttr(),
        "dynamicFriction": api.GetDynamicFrictionAttr(),
        "restitution": api.GetRestitutionAttr(),
    }
    return {k: a.Get() for k, a in values.items() if a and a.HasAuthoredValue()}


def _build_links(
    parts: list[model.Part],
    proxies: list[model.Proxy],
    link_prims: dict[str, Usd.Prim],
    shape_mass: dict[str, float],
    cache: UsdGeom.XformCache,
    scale: float,
) -> dict[str, model.Link]:
    names = set(link_prims) | {p.link for p in parts} | {p.link for p in proxies}
    links = {}
    for name in sorted(n for n in names if n):
        prim = link_prims.get(name)
        transform = np.eye(4)
        mass = density = None
        if prim is not None:
            transform = np.array(cache.GetLocalToWorldTransform(prim)).T
            transform[:3, 3] *= scale
            api = UsdPhysics.MassAPI(prim)
            mass = _authored(api.GetMassAttr())
            density = _authored(api.GetDensityAttr())
        links[name] = model.Link(
            name=name,
            parts=[i for i, p in enumerate(parts) if p.link == name],
            proxies=[i for i, p in enumerate(proxies) if p.link == name],
            # USD accumulates mass authored on a body's collision shapes.
            mass=mass if mass is not None else shape_mass.get(name),
            density=density,
            transform=transform,
        )
    return links


def _read_joints(
    stage: Usd.Stage, cache: UsdGeom.XformCache, scale: float
) -> list[model.Joint]:
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joints.append(_read_joint(prim, cache, scale))
    return joints


def _read_joint(
    prim: Usd.Prim, cache: UsdGeom.XformCache, scale: float
) -> model.Joint:
    joint = UsdPhysics.Joint(prim)
    bodies = [_first_target(joint.GetBody0Rel()), _first_target(joint.GetBody1Rel())]
    frames = [
        _joint_frame(prim, stage_body, index, cache, scale)
        for index, stage_body in enumerate(bodies)
    ]
    joint_type = _joint_type(prim)

    # Omitting either body relationship attaches the joint to the world.
    # The moving body is whichever one is present.
    parent, child = (1, 0) if bodies[1] is None and bodies[0] is not None else (0, 1)

    axis_local = _AXES.get(prim.GetAttribute("physics:axis").Get(), _AXES["X"])
    axis = frames[parent][:3, :3] @ axis_local
    anchor = frames[parent][:3, 3]
    lower, upper = _joint_limits(prim, joint_type, scale)

    return model.Joint(
        name=str(prim.GetPath()),
        joint_type=joint_type,
        parent_link=bodies[parent],
        child_link=bodies[child] or STATIC_LINK,
        axis=axis,
        anchor=anchor,
        lower=lower,
        upper=upper,
        rest_value=_rest_value([frames[parent], frames[child]], axis, joint_type),
        drive=_read_drive(prim),
    )


_AXES = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}

_JOINT_TYPES = {
    "PhysicsRevoluteJoint": "revolute",
    "PhysicsPrismaticJoint": "prismatic",
    "PhysicsSphericalJoint": "spherical",
    "PhysicsFixedJoint": "fixed",
}


def _joint_type(prim: Usd.Prim) -> str:
    return _JOINT_TYPES.get(str(prim.GetTypeName()), "unknown")


def _joint_frame(
    prim: Usd.Prim,
    body: str | None,
    index: int,
    cache: UsdGeom.XformCache,
    scale: float,
) -> np.ndarray:
    """Joint frame `index` expressed in world space."""
    body_transform = np.eye(4)
    if body:
        body_prim = prim.GetStage().GetPrimAtPath(body)
        if body_prim:
            body_transform = np.array(cache.GetLocalToWorldTransform(body_prim)).T

    position = prim.GetAttribute(f"physics:localPos{index}").Get() or Gf.Vec3f(0)
    rotation = prim.GetAttribute(f"physics:localRot{index}").Get() or Gf.Quatf(1)
    local = np.eye(4)
    local[:3, :3] = np.array(Gf.Matrix3d(Gf.Rotation(rotation))).T
    local[:3, 3] = np.asarray(position, dtype=np.float64)

    frame = body_transform @ local
    frame[:3, 3] *= scale
    return frame


def _joint_limits(
    prim: Usd.Prim, joint_type: str, scale: float
) -> tuple[float | None, float | None]:
    lower = _authored(prim.GetAttribute("physics:lowerLimit"))
    upper = _authored(prim.GetAttribute("physics:upperLimit"))
    if joint_type == "revolute":
        factor = np.pi / 180.0
    elif joint_type == "prismatic":
        factor = scale
    else:
        return lower, upper
    return (
        None if lower is None else lower * factor,
        None if upper is None else upper * factor,
    )


def _rest_value(
    frames: list[np.ndarray], axis: np.ndarray, joint_type: str
) -> float | None:
    """Joint coordinate implied by the offset between the two joint frames."""
    if joint_type == "prismatic":
        return float((frames[1][:3, 3] - frames[0][:3, 3]) @ geometry.normalize(axis))
    if joint_type != "revolute":
        return None

    unit = geometry.normalize(axis)
    reference = frames[0][:3, 0] - unit * (frames[0][:3, 0] @ unit)
    current = frames[1][:3, 0] - unit * (frames[1][:3, 0] @ unit)
    if np.linalg.norm(reference) < 1e-9 or np.linalg.norm(current) < 1e-9:
        reference = frames[0][:3, 1] - unit * (frames[0][:3, 1] @ unit)
        current = frames[1][:3, 1] - unit * (frames[1][:3, 1] @ unit)
    reference, current = geometry.normalize(reference), geometry.normalize(current)
    return float(
        np.arctan2(np.cross(reference, current) @ unit, reference @ current)
    )


def _read_drive(prim: Usd.Prim) -> dict[str, float] | None:
    prefixes = [
        s.split(":", 1)[1]
        for s in prim.GetAppliedSchemas()
        if s.startswith("PhysicsDriveAPI:")
    ]
    if not prefixes:
        return None
    drive = {}
    for name in ("stiffness", "damping", "maxForce"):
        attribute = prim.GetAttribute(f"drive:{prefixes[0]}:physics:{name}")
        drive[name] = _authored(attribute)
    return drive


def _first_target(relationship) -> str | None:
    targets = relationship.GetTargets() if relationship else []
    return str(targets[0]) if targets else None


def _authored(attribute) -> float | None:
    if not attribute or not attribute.HasAuthoredValue():
        return None
    return float(attribute.Get())


def _scalar(value):
    if isinstance(value, (Gf.Vec3f, Gf.Vec3d, Gf.Vec4f)):
        return tuple(float(v) for v in value)
    return float(value)
