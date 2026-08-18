"""Builders for the minimal assets the check tests run against."""

import dataclasses

import numpy as np
import PIL.Image
import trimesh

from sim_env_builder.asset_checks import config
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.ingest import model

CUBE_POINTS = np.array(
    [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ],
    dtype=np.float64,
)

# Outward-facing quads, counter-clockwise seen from outside.
CUBE_QUADS = np.array(
    [
        [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
)


def cube_arrays(scale: float = 1.0, offset=(0.0, 0.0, 0.0)):
    """Vertices, triangles, and the source quad topology."""
    points = CUBE_POINTS * scale + np.asarray(offset, dtype=np.float64)
    triangles = np.concatenate([CUBE_QUADS[:, [0, 1, 2]], CUBE_QUADS[:, [0, 2, 3]]])
    counts = np.full(len(CUBE_QUADS), 4)
    return points, triangles, counts, CUBE_QUADS.ravel()


def part(
    name: str = "part",
    scale: float = 1.0,
    offset=(0.0, 0.0, 0.0),
    link: str = "link",
    polygons: bool = True,
    uvs=None,
    material: str | None = None,
    skin: model.Skin | None = None,
    vertices=None,
    triangles=None,
) -> model.Part:
    points, faces, counts, indices = cube_arrays(scale, offset)
    if vertices is not None:
        points = np.asarray(vertices, dtype=np.float64)
    if triangles is not None:
        faces = np.asarray(triangles, dtype=np.int64)
        counts, indices = None, None
    return model.Part(
        name=name,
        link=link,
        vertices=points,
        triangles=faces,
        face_counts=counts if polygons else None,
        face_indices=indices if polygons else None,
        normals=None,
        uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
        material=material,
        skin=skin,
    )


def proxy(
    name: str = "proxy",
    link: str = "link",
    source: str = "convex_hull",
    scale: float = 1.0,
    offset=(0.0, 0.0, 0.0),
    friction: float | None = 0.5,
    restitution: float | None = 0.1,
    vertices=None,
    triangles=None,
) -> model.Proxy:
    points, faces, _, _ = cube_arrays(scale, offset)
    return model.Proxy(
        name=name,
        link=link,
        source=source,
        vertices=points if vertices is None else np.asarray(vertices, dtype=np.float64),
        triangles=faces if triangles is None else np.asarray(triangles, dtype=np.int64),
        static_friction=friction,
        dynamic_friction=friction,
        restitution=restitution,
    )


def joint(
    name: str = "joint",
    joint_type: str = "revolute",
    parent_link: str | None = "link",
    child_link: str = "child",
    axis=(0.0, 0.0, 1.0),
    anchor=(0.0, 0.0, 0.0),
    lower: float | None = -1.0,
    upper: float | None = 1.0,
    rest_value: float | None = 0.0,
    drive: dict | None = None,
) -> model.Joint:
    return model.Joint(
        name=name,
        joint_type=joint_type,
        parent_link=parent_link,
        child_link=child_link,
        axis=np.asarray(axis, dtype=np.float64),
        anchor=np.asarray(anchor, dtype=np.float64),
        lower=lower,
        upper=upper,
        rest_value=rest_value,
        drive=drive,
    )


def texture(colour=(0.5, 0.5, 0.5), size=(512, 512), image=True) -> model.Texture:
    if not image:
        return model.Texture(path="missing.png", image=None)
    pixels = np.tile(
        (np.asarray(colour) * 255).astype(np.uint8), (size[1], size[0], 1)
    )
    return model.Texture(path="texture.png", image=PIL.Image.fromarray(pixels))


def gradient_texture(size=(64, 64)) -> model.Texture:
    """A texture whose brightness ramps along V, as baked lighting would."""
    ramp = np.linspace(20, 235, size[1], dtype=np.uint8)
    pixels = np.repeat(np.repeat(ramp[:, None], size[0], axis=1)[:, :, None], 3, axis=2)
    return model.Texture(path="ramp.png", image=PIL.Image.fromarray(pixels))


def material(
    name: str = "material",
    shader: str = "UsdPreviewSurface",
    textures: dict | None = None,
    params: dict | None = None,
) -> model.Material:
    return model.Material(
        name=name,
        shader=shader,
        params=params if params is not None else {"roughness": 0.5},
        textures=textures or {},
    )


def link(
    name: str = "link",
    parts=(),
    proxies=(),
    mass: float | None = 1.0,
    density: float | None = None,
    transform=None,
) -> model.Link:
    return model.Link(
        name=name,
        parts=list(parts),
        proxies=list(proxies),
        mass=mass,
        density=density,
        transform=np.eye(4) if transform is None else np.asarray(transform),
    )


def asset(
    parts=(),
    proxies=(),
    joints=(),
    materials=None,
    links=None,
    asset_format: str = "usd",
    mass: float | None = 1.0,
) -> model.AssetModel:
    """Assemble a model, deriving links from part and proxy membership."""
    parts, proxies, joints = list(parts), list(proxies), list(joints)
    if links is None:
        names = [p.link for p in parts] + [p.link for p in proxies]
        names += [j.child_link for j in joints]
        names += [j.parent_link for j in joints if j.parent_link]
        links = {}
        for name in dict.fromkeys(n for n in names if n):
            links[name] = link(
                name,
                [i for i, p in enumerate(parts) if p.link == name],
                [i for i, p in enumerate(proxies) if p.link == name],
                mass=mass,
            )
    return model.AssetModel(
        path="test-asset",
        format=asset_format,
        parts=parts,
        proxies=proxies,
        materials=materials or {},
        joints=joints,
        links=links,
        meters_per_unit=1.0,
        bbox_diag=model.bounding_diagonal(parts) if parts else 1.0,
    )


def context(target: model.AssetModel, **overrides) -> registry.Context:
    thresholds = dataclasses.replace(config.Thresholds(), **overrides)
    return registry.Context(target, thresholds)


def unwrapped_cube_uvs(margin: float = 0.02, scale: float = 0.3) -> np.ndarray:
    """Per-triangle-corner UVs laying each cube quad in its own square shell."""
    squares = []
    for row in range(2):
        for column in range(3):
            origin = np.array(
                [column * (scale + margin) + margin, row * (scale + margin) + margin]
            )
            squares.append(
                origin
                + np.array([[0, 0], [scale, 0], [scale, scale], [0, scale]])
            )
    quads = np.stack(squares)
    return np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def sphere_part(subdivisions: int = 2, **kwargs) -> model.Part:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    return part(
        vertices=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.faces),
        polygons=False,
        **kwargs,
    )
