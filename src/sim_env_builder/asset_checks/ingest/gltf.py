"""Load a glTF or GLB file into the normalized asset model.

glTF carries no physics or rigid-articulation data. The loader returns no joints
and finds proxies only through collision-mesh names.
"""

import base64
import json
import logging
import pathlib
import re
import struct

import numpy as np
import trimesh

from sim_env_builder.asset_checks.ingest import model, usd

logger = logging.getLogger(__name__)

STATIC_LINK = model.STATIC_LINK
COLLISION_NAME = re.compile(r"^ucx_|_(collision|col)$", re.IGNORECASE)

# Map each trimesh PBRMaterial attribute to the slot used by material checks.
TEXTURE_SLOTS = {
    "baseColorTexture": "basecolor",
    "normalTexture": "normal",
    "metallicRoughnessTexture": "roughness",
    "emissiveTexture": "emissive",
    "occlusionTexture": "occlusion",
}

PARAM_ATTRIBUTES = ("baseColorFactor", "metallicFactor", "roughnessFactor")

_COMPONENTS = {
    5120: np.dtype(np.int8),
    5121: np.dtype(np.uint8),
    5122: np.dtype(np.int16),
    5123: np.dtype(np.uint16),
    5125: np.dtype(np.uint32),
    5126: np.dtype(np.float32),
}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path: str) -> model.AssetModel:
    scene = trimesh.load(path, process=False, force="scene")
    document = _Document(path)

    materials: dict[str, model.Material] = {}
    parts: list[model.Part] = []
    proxies: list[model.Proxy] = []

    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node)
        geometry = scene.geometry[geometry_name]
        vertices = np.asarray(geometry.vertices) @ transform[:3, :3].T + transform[:3, 3]
        triangles = np.asarray(geometry.faces, dtype=np.int64)

        if COLLISION_NAME.search(node) or COLLISION_NAME.search(geometry_name):
            proxies.append(_proxy(node, vertices, triangles))
            continue
        parts.append(
            model.Part(
                name=node,
                link=STATIC_LINK,
                vertices=vertices,
                triangles=triangles,
                face_counts=None,
                face_indices=None,
                normals=_normals(geometry, transform),
                uvs=_uvs(geometry, triangles),
                material=_material(geometry, materials, document),
                skin=document.skin(geometry_name, len(vertices)),
            )
        )

    links = {
        STATIC_LINK: model.Link(
            name=STATIC_LINK,
            parts=list(range(len(parts))),
            proxies=list(range(len(proxies))),
            mass=None,
            density=None,
            transform=np.eye(4),
        )
    }
    return model.AssetModel(
        path=path,
        format="gltf",
        parts=parts,
        proxies=proxies,
        materials=materials,
        joints=[],
        links=links,
        meters_per_unit=1.0,
        bbox_diag=model.bounding_diagonal(parts),
    )


def _normals(geometry: trimesh.Trimesh, transform: np.ndarray) -> np.ndarray | None:
    normals = geometry.vertex_normals
    if normals is None or len(normals) != len(geometry.vertices):
        return None
    return usd._transform_normals(np.asarray(normals), transform)


def _uvs(geometry: trimesh.Trimesh, triangles: np.ndarray) -> np.ndarray | None:
    uv = getattr(geometry.visual, "uv", None)
    if uv is None or len(uv) != len(geometry.vertices):
        return None
    return np.asarray(uv, dtype=np.float64)[triangles]


def _proxy(name: str, vertices: np.ndarray, triangles: np.ndarray) -> model.Proxy:
    return model.Proxy(
        name=name,
        link=STATIC_LINK,
        source="authored_mesh",
        vertices=vertices,
        triangles=triangles,
        static_friction=None,
        dynamic_friction=None,
        restitution=None,
    )


def _material(
    geometry: trimesh.Trimesh,
    materials: dict[str, model.Material],
    document: "_Document",
) -> str | None:
    source = getattr(geometry.visual, "material", None)
    if source is None:
        return None

    name = getattr(source, "name", "") or f"material_{len(materials)}"
    if name in materials:
        return name

    textures = {}
    for attribute, slot in TEXTURE_SLOTS.items():
        image = getattr(source, attribute, None)
        if image is not None:
            textures[slot] = model.Texture(path=f"{name}:{slot}", image=image)
    for slot, uri in document.missing_textures(name).items():
        textures.setdefault(slot, model.Texture(path=uri, image=None))

    params = {
        attribute: getattr(source, attribute)
        for attribute in PARAM_ATTRIBUTES
        if getattr(source, attribute, None) is not None
    }
    materials[name] = model.Material(
        name=name,
        shader="pbrMetallicRoughness",
        params={k: _scalar(v) for k, v in params.items()},
        textures=textures,
    )
    return name


def _scalar(value):
    array = np.atleast_1d(np.asarray(value, dtype=np.float64))
    return tuple(float(v) for v in array) if array.size > 1 else float(array[0])


class _Document:
    """Provide raw glTF JSON for skins and textures that trimesh does not expose."""

    def __init__(self, path: str):
        self.directory = pathlib.Path(path).parent
        self.json, self.binary = _read_container(pathlib.Path(path))
        self._skins: dict[str, model.Skin] | None = None

    def skin(self, mesh_name: str, count: int) -> model.Skin | None:
        if self._skins is None:
            self._skins = self._read_skins()
        skin = self._skins.get(mesh_name)
        return skin if skin is not None and len(skin.weights) == count else None

    def _read_skins(self) -> dict[str, model.Skin]:
        if not self.json.get("skins"):
            return {}
        nodes = self.json.get("nodes", [])
        joints = [
            nodes[i].get("name", f"joint_{i}")
            for i in self.json["skins"][0].get("joints", [])
        ]

        skins = {}
        for index, mesh in enumerate(self.json.get("meshes", [])):
            blocks = [
                (
                    self._accessor(p["attributes"]["JOINTS_0"]),
                    self._accessor(p["attributes"]["WEIGHTS_0"]),
                )
                for p in mesh.get("primitives", [])
                if "JOINTS_0" in p.get("attributes", {})
            ]
            if not blocks:
                continue
            name = mesh.get("name", f"mesh_{index}")
            skins[name] = model.Skin(
                joints=joints,
                indices=np.concatenate([b[0] for b in blocks]),
                weights=np.concatenate([b[1] for b in blocks]).astype(np.float64),
            )
        return skins

    def _accessor(self, index: int) -> np.ndarray:
        accessor = self.json["accessors"][index]
        view = self.json["bufferViews"][accessor["bufferView"]]
        dtype = _COMPONENTS[accessor["componentType"]]
        width = _WIDTHS[accessor["type"]]
        element = width * dtype.itemsize
        stride = view.get("byteStride") or element
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)

        buffer = self._buffer(view.get("buffer", 0))
        count = accessor["count"]
        raw = np.frombuffer(buffer, np.uint8, count=(count - 1) * stride + element, offset=start)
        rows = np.lib.stride_tricks.as_strided(
            raw, shape=(count, element), strides=(stride, 1)
        )
        values = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
        # Normalized integer accessors are fixed-point fractions of [0, 1];
        # WEIGHTS_0 in particular may be stored this way.
        if accessor.get("normalized") and dtype.kind == "u":
            return values.astype(np.float64) / np.iinfo(dtype).max
        return values

    def _buffer(self, index: int) -> bytes:
        uri = self.json["buffers"][index].get("uri")
        if uri is None:
            return self.binary
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        return (self.directory / uri).read_bytes()

    def missing_textures(self, material_name: str) -> dict[str, str]:
        """Texture slots that declare an image file the loader cannot resolve."""
        for material in self.json.get("materials", []):
            if material.get("name") != material_name:
                continue
            return {
                slot: uri
                for slot, index in _texture_indices(material).items()
                if (uri := self._unresolved_image(index)) is not None
            }
        return {}

    def _unresolved_image(self, texture_index: int) -> str | None:
        texture = self.json.get("textures", [])[texture_index]
        image = self.json.get("images", [])[texture.get("source", 0)]
        uri = image.get("uri")
        if uri is None or uri.startswith("data:"):
            return None
        return None if (self.directory / uri).exists() else uri


def _texture_indices(material: dict) -> dict[str, int]:
    pbr = material.get("pbrMetallicRoughness", {})
    slots = {
        "basecolor": pbr.get("baseColorTexture"),
        "roughness": pbr.get("metallicRoughnessTexture"),
        "normal": material.get("normalTexture"),
        "emissive": material.get("emissiveTexture"),
        "occlusion": material.get("occlusionTexture"),
    }
    return {slot: ref["index"] for slot, ref in slots.items() if ref}


def _read_container(path: pathlib.Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if data[:4] != b"glTF":
        return json.loads(data), b""

    document, binary = {}, b""
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(chunk)
        else:
            binary = chunk
        offset += 8 + length + (-length % 4)
    return document, binary
