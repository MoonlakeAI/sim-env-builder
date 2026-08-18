"""Write real USD and glTF files for the loader tests."""

import base64
import json
import pathlib
import struct

import numpy as np
import PIL.Image
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import factories

COMPONENT_FLOAT = 5126
COMPONENT_USHORT = 5123


def hinged_stage(path: pathlib.Path, meters_per_unit: float = 1.0) -> str:
    """A two-link revolute door with colliders, mass and a reversed limit pair."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.Xform.Define(stage, "/asset")

    base = _rigid_body(stage, "/asset/base")
    _mesh(stage, "/asset/base/geom", offset=(0, 0, 0), mass=2.0, collider=True)
    door = _rigid_body(stage, "/asset/door")
    _mesh(stage, "/asset/door/geom", offset=(1.0, 0, 0), mass=1.0, collider=True)

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/asset/hinge")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([door.GetPath()])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(90.0)
    joint.CreateUpperLimitAttr(-90.0)
    joint.GetPrim().CreateAttribute(
        "physics:localPos0", Sdf.ValueTypeNames.Point3f
    ).Set(Gf.Vec3f(1.0, 0.0, 0.0))
    joint.GetPrim().CreateAttribute(
        "physics:localPos1", Sdf.ValueTypeNames.Point3f
    ).Set(Gf.Vec3f(0.0, 0.0, 0.0))

    stage.GetRootLayer().Save()
    return str(path)


def _rigid_body(stage: Usd.Stage, path: str) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return prim


def _mesh(
    stage: Usd.Stage,
    path: str,
    offset=(0.0, 0.0, 0.0),
    mass: float | None = None,
    collider: bool = False,
    purpose: str | None = None,
    approximation: str | None = None,
    left_handed: bool = False,
) -> UsdGeom.Mesh:
    points, _, counts, indices = factories.cube_arrays(offset=offset)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([int(c) for c in counts])
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in indices])
    if left_handed:
        mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    if purpose:
        mesh.CreatePurposeAttr(purpose)

    corners = np.tile([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], (len(counts), 1))
    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set([Gf.Vec2f(*uv) for uv in corners])

    if collider:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        if approximation:
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
                approximation
            )
    if mass is not None:
        UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(mass)
    return mesh


def stage_with_purposes(path: pathlib.Path) -> str:
    """A render mesh alongside a proxy mesh and a guide mesh."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _rigid_body(stage, "/asset")
    _mesh(stage, "/asset/render", mass=1.0)
    _mesh(stage, "/asset/proxy", purpose="proxy", collider=True, offset=(0.1, 0, 0))
    _mesh(stage, "/asset/guide", purpose="guide", offset=(5, 0, 0))
    stage.GetRootLayer().Save()
    return str(path)


def stage_with_convex_hull(path: pathlib.Path) -> str:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _rigid_body(stage, "/asset")
    _mesh(stage, "/asset/render", mass=1.0, collider=True, approximation="convexHull")
    stage.GetRootLayer().Save()
    return str(path)


def stage_with_material(path: pathlib.Path, texture: pathlib.Path) -> str:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _rigid_body(stage, "/asset")
    mesh = _mesh(stage, "/asset/render", mass=1.0)

    material = UsdShade.Material.Define(stage, "/asset/mat")
    shader = UsdShade.Shader.Define(stage, "/asset/mat/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    sampler = UsdShade.Shader.Define(stage, "/asset/mat/albedo")
    sampler.CreateIdAttr("UsdUVTexture")
    sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return str(path)


def stage_left_handed(path: pathlib.Path) -> str:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _rigid_body(stage, "/asset")
    _mesh(stage, "/asset/render", mass=1.0, left_handed=True)
    stage.GetRootLayer().Save()
    return str(path)


def two_part_glb(path: pathlib.Path) -> str:
    """Two boxes as separate scene nodes, one placed by its node transform."""
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(1, 1, 1)), node_name="body")
    scene.add_geometry(
        trimesh.creation.box(extents=(0.5, 0.5, 0.5)),
        node_name="knob",
        transform=trimesh.transformations.translation_matrix([2.0, 0, 0]),
    )
    path.write_bytes(scene.export(file_type="glb"))
    return str(path)


def glb_with_collision_name(path: pathlib.Path) -> str:
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), node_name="body")
    scene.add_geometry(
        trimesh.creation.box(extents=(1.2, 1.2, 1.2)), node_name="body_collision"
    )
    path.write_bytes(scene.export(file_type="glb"))
    return str(path)


def textured_quad_gltf(path: pathlib.Path, texture_name: str) -> str:
    """A quad whose UVs and quadrant texture pin the coordinate convention.

    The raw specification places uv (0, 0) at the image's top-left corner. The
    texture uses red top-left, green top-right, blue bottom-left, and white
    bottom-right.
    """
    image = np.zeros((64, 64, 3), np.uint8)
    image[:32, :32] = (255, 0, 0)
    image[:32, 32:] = (0, 255, 0)
    image[32:, :32] = (0, 0, 255)
    image[32:, 32:] = (255, 255, 255)
    PIL.Image.fromarray(image).save(path.parent / texture_name)

    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    indices = np.array([0, 1, 2, 1, 3, 2], np.uint16)

    blobs = [positions.tobytes(), uvs.tobytes(), indices.tobytes()]
    buffer, views, offset = b"", [], 0
    for blob in blobs:
        padding = b"\x00" * (-len(blob) % 4)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        buffer += blob + padding
        offset += len(blob) + len(padding)

    document = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [{"mesh": 0, "name": "quad"}],
        "meshes": [
            {
                "name": "quad_mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {"name": "quadrants", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
        "textures": [{"source": 0}],
        "images": [{"uri": texture_name}],
        "accessors": [
            _accessor(0, COMPONENT_FLOAT, len(positions), "VEC3", positions),
            _accessor(1, COMPONENT_FLOAT, len(uvs), "VEC2"),
            _accessor(2, COMPONENT_USHORT, len(indices), "SCALAR"),
        ],
        "bufferViews": views,
        "buffers": [
            {
                "byteLength": len(buffer),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(buffer).decode(),
            }
        ],
    }
    path.write_text(json.dumps(document))
    return str(path)


def skinned_gltf(
    path: pathlib.Path, weights: np.ndarray, normalized: bool = False
) -> str:
    """A minimal skinned glTF, written directly to exercise accessor decoding.

    `normalized` stores weights as fixed-point unsigned shorts. The specification
    allows this representation, and readers must divide the values back to [0, 1].
    """
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32
    )
    joints = np.zeros((len(positions), 4), dtype=np.uint16)
    indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint16)
    padded = np.zeros((len(positions), 4), dtype=np.float32)
    padded[:, : weights.shape[1]] = weights

    if normalized:
        stored = np.round(padded * 65535).astype(np.uint16)
        weight_accessor = {"componentType": COMPONENT_USHORT, "normalized": True}
    else:
        stored = padded
        weight_accessor = {"componentType": COMPONENT_FLOAT}

    blobs = [positions.tobytes(), joints.tobytes(), stored.tobytes(), indices.tobytes()]
    buffer, views, offset = b"", [], 0
    for blob in blobs:
        padding = b"\x00" * (-len(blob) % 4)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        buffer += blob + padding
        offset += len(blob) + len(padding)

    document = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "scene": 0,
        "nodes": [
            {"mesh": 0, "skin": 0, "name": "skinned"},
            {"name": "root_joint"},
        ],
        "meshes": [
            {
                "name": "skinned_mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "indices": 3,
                    }
                ],
            }
        ],
        "skins": [{"joints": [1]}],
        "accessors": [
            _accessor(0, COMPONENT_FLOAT, len(positions), "VEC3", positions),
            _accessor(1, COMPONENT_USHORT, len(joints), "VEC4"),
            _accessor(2, weight_accessor["componentType"], len(padded), "VEC4")
            | ({"normalized": True} if normalized else {}),
            _accessor(3, COMPONENT_USHORT, len(indices), "SCALAR"),
        ],
        "bufferViews": views,
        "buffers": [
            {
                "byteLength": len(buffer),
                "uri": "data:application/octet-stream;base64,"
                + base64.b64encode(buffer).decode(),
            }
        ],
    }
    path.write_text(json.dumps(document))
    return str(path)


def _accessor(view: int, component: int, count: int, kind: str, values=None) -> dict:
    accessor = {
        "bufferView": view,
        "componentType": component,
        "count": count,
        "type": kind,
    }
    if values is not None:
        accessor["min"] = values.min(axis=0).tolist()
        accessor["max"] = values.max(axis=0).tolist()
    return accessor


def glb_from_gltf(source: pathlib.Path, target: pathlib.Path) -> str:
    """Repackage a .gltf as .glb to exercise both container paths."""
    document = json.loads(source.read_text())
    payload = json.dumps(document).encode()
    payload += b" " * (-len(payload) % 4)
    chunk = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    target.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 12 + len(chunk)) + chunk
    )
    return str(target)


def world_anchored_stage(path: pathlib.Path, omit: str) -> str:
    """One rigid body hinged to the world, with body0 or body1 left empty."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = _rigid_body(stage, "/asset")
    _mesh(stage, "/asset/geom", mass=1.0, collider=True)

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/asset/hinge")
    if omit == "body0":
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
    else:
        joint.CreateBody0Rel().SetTargets([body.GetPath()])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-45.0)
    joint.CreateUpperLimitAttr(45.0)
    stage.GetRootLayer().Save()
    return str(path)


def scaled_normals_stage(path: pathlib.Path) -> str:
    """A mesh with authored normals under a non-uniform scale."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/asset")
    root.AddScaleOp().Set(Gf.Vec3f(2.0, 1.0, 1.0))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

    mesh = _mesh(stage, "/asset/geom", mass=1.0)
    slanted = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    mesh.CreateNormalsAttr([Gf.Vec3f(*slanted)] * 8)
    stage.GetRootLayer().Save()
    return str(path)


def indexed_uv_stage(path: pathlib.Path) -> str:
    """A quad whose vertex-interpolated UVs go through an index table."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    _rigid_body(stage, "/asset")
    mesh = UsdGeom.Mesh.Define(stage, "/asset/geom")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    # Two unique values; the index table maps the four points onto them.
    primvar.Set([Gf.Vec2f(0.25, 0.25), Gf.Vec2f(0.75, 0.75)])
    primvar.SetIndices([0, 1, 0, 1])
    stage.GetRootLayer().Save()
    return str(path)
