"""Normalized asset representation shared by every check.

Loaders populate fields supported by each format. Unsupported fields remain
`None`, and checks that require them report `not_applicable`. All geometry uses
meters in world space at the authored rest pose.
"""

import dataclasses
import functools

import numpy as np
import PIL.Image

# Link name for geometry with no rigid body: static scenery, or every part of
# a format that cannot express rigid bodies.
STATIC_LINK = "<static>"


@dataclasses.dataclass
class Texture:
    path: str
    image: PIL.Image.Image | None

    @property
    def size(self) -> tuple[int, int] | None:
        return None if self.image is None else self.image.size

    @functools.cached_property
    def content_hash(self) -> str:
        if self.image is None:
            return f"missing:{self.path}"
        return str(hash(self.image.tobytes()))


@dataclasses.dataclass
class Skin:
    joints: list[str]
    indices: np.ndarray  # (n_verts, k) joint index per influence
    weights: np.ndarray  # (n_verts, k)


@dataclasses.dataclass
class Material:
    name: str
    shader: str | None
    params: dict[str, float | tuple]
    textures: dict[str, Texture]  # slot name -> texture

    @property
    def has_detail_map(self) -> bool:
        return any(slot in self.textures for slot in ("normal", "bump", "displacement"))

    def fingerprint(self) -> str:
        params = tuple(sorted((k, _round(v)) for k, v in self.params.items()))
        textures = tuple(sorted((k, t.content_hash) for k, t in self.textures.items()))
        return repr((self.shader, params, textures))


@dataclasses.dataclass
class Part:
    """One render mesh.

    `triangles` is always present. `face_counts` and `face_indices` carry the
    pre-triangulation polygons when the format preserves them. `uvs` uses one
    index per triangle corner, avoiding a separate corner mapping.
    """

    name: str
    link: str | None
    vertices: np.ndarray  # (n, 3)
    triangles: np.ndarray  # (m, 3)
    face_counts: np.ndarray | None  # (n_faces,) vertices per polygon
    face_indices: np.ndarray | None  # flat vertex indices, grouped by polygon
    normals: np.ndarray | None  # (n, 3) per vertex
    uvs: np.ndarray | None  # (m, 3, 2) per triangle corner
    material: str | None
    skin: Skin | None


@dataclasses.dataclass
class Proxy:
    """A collision shape, as the simulator will see it."""

    name: str
    link: str | None
    # "render_mesh" reuses the full-resolution render geometry; the others are
    # dedicated collision shapes.
    source: str  # "render_mesh" | "authored_mesh" | "convex_hull" | "primitive"
    vertices: np.ndarray
    triangles: np.ndarray
    static_friction: float | None
    dynamic_friction: float | None
    restitution: float | None

    @property
    def has_friction(self) -> bool:
        return self.static_friction is not None and self.dynamic_friction is not None


@dataclasses.dataclass
class Joint:
    """A physics joint, with its axis resolved into world space at rest.

    `lower`/`upper`/`rest_value` are radians for revolute joints and meters for
    prismatic ones. `None` limits mean the joint is unbounded.
    """

    name: str
    joint_type: str  # "revolute" | "prismatic" | "spherical" | "fixed" | "unknown"
    parent_link: str | None  # None anchors the joint to the world
    child_link: str
    axis: np.ndarray | None
    anchor: np.ndarray | None
    lower: float | None
    upper: float | None
    rest_value: float | None
    drive: dict[str, float] | None


@dataclasses.dataclass
class Link:
    name: str
    parts: list[int]
    proxies: list[int]
    mass: float | None
    density: float | None
    transform: np.ndarray  # 4x4 world transform at rest


@dataclasses.dataclass
class AssetModel:
    path: str
    format: str  # "usd" | "gltf"
    parts: list[Part]
    proxies: list[Proxy]
    materials: dict[str, Material]
    joints: list[Joint]
    links: dict[str, Link]
    meters_per_unit: float
    bbox_diag: float

    @property
    def has_polygons(self) -> bool:
        return any(p.face_counts is not None for p in self.parts)

    @property
    def moving_joints(self) -> list[Joint]:
        return [j for j in self.joints if j.joint_type not in ("fixed", "unknown")]

    @property
    def is_articulated(self) -> bool:
        return bool(self.moving_joints)

    @property
    def has_skin(self) -> bool:
        return any(p.skin is not None for p in self.parts)

    @property
    def triangle_count(self) -> int:
        return sum(len(p.triangles) for p in self.parts)

    def parts_of(self, link: str) -> list[Part]:
        return [self.parts[i] for i in self.links[link].parts]


def _round(value):
    if isinstance(value, (tuple, list)):
        return tuple(round(float(v), 4) for v in value)
    return round(float(value), 4)


def bounding_diagonal(parts: list[Part]) -> float:
    populated = [p.vertices for p in parts if len(p.vertices)]
    if not populated:
        return 1.0
    points = np.concatenate(populated)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) or 1.0
