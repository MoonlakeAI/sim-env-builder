"""UV layout quality: distortion, packing and seam placement.

Strict by design. The checks report mirrored shells and deliberate
out-of-unit-square tiling as failures despite their legitimate uses.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from sim_env_builder.asset_checks import geometry
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.checks.registry import Result, advisory, check, verdict

UNTEXTURED = "no material binds a base-colour texture"
NO_SAMPLING = "no material samples an image texture"


def _mapped(context: registry.Context) -> tuple[list, list[str]]:
    """Texture-sampling parts carrying UVs, and those that should but do not.

    A part with no bound image texture never reads its UVs. The checks neither
    judge nor require this inert data.
    """
    usable, missing = [], []
    for part, surface in zip(context.asset.parts, context.surfaces):
        if not _samples_texture(context, part.material):
            continue
        if part.uvs is None:
            missing.append(part.name)
        elif len(part.triangles):
            usable.append((part, surface))
    return usable, missing


def _samples_texture(context: registry.Context, material_name: str | None) -> bool:
    material = context.asset.materials.get(material_name)
    return bool(material and material.textures)


def _resolution(context: registry.Context, material_name: str | None) -> int:
    """Side of the texture an atlas addresses, or the configured default."""
    material = context.asset.materials.get(material_name)
    texture = material.textures.get("basecolor") if material else None
    if texture and texture.size:
        return max(texture.size)
    return context.thresholds.raster_resolution


def _textured(context: registry.Context) -> bool:
    return any(_binds_atlas(context, name) for name in context.asset.materials)


def _binds_atlas(context: registry.Context, material_name: str | None) -> bool:
    """Whether a material addresses a per-asset atlas rather than a detail map.

    A base-colour texture belongs to one asset, and each part must own a region.
    Atlas packing therefore matters. Normal-only or roughness-only maps usually
    provide tileable surface detail shared across parts, where UV overlap and
    repetition are intentional.
    """
    material = context.asset.materials.get(material_name)
    return bool(material and "basecolor" in material.textures)


def _rasterized(context: registry.Context) -> list[tuple[np.ndarray, np.ndarray]]:
    """Coverage counts and shell labels per textured atlas, rasterized once.

    Overlap, padding, and utilization read the same buffers. Rasterizing a large
    atlas is this section's most expensive step. Skip atlases with no bound image
    texture because their UV layout addresses nothing.
    """
    if "uv_rasters" not in context.memo:
        context.memo["uv_rasters"] = [
            geometry.rasterize_uv(uvs, resolution, labels)
            for material, resolution, uvs, labels in _atlases(context)
            if _binds_atlas(context, material)
        ]
    return context.memo["uv_rasters"]


def _atlas_parts(context: registry.Context) -> tuple[list, list[str]]:
    """Parts whose material binds a base-colour atlas, split by UV presence."""
    usable, missing = [], []
    for part, surface in zip(context.asset.parts, context.surfaces):
        if not _binds_atlas(context, part.material):
            continue
        if part.uvs is None:
            missing.append(part.name)
        elif len(part.triangles):
            usable.append((part, surface))
    return usable, missing


def _atlases(context: registry.Context) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """UV triangles grouped by the atlas they address, with shell labels.

    Pack and measure together parts that share a material and texture. Parts with
    different materials use separate atlases.
    """
    groups: dict[str | None, list] = {}
    for part, surface in _mapped(context)[0]:
        groups.setdefault(part.material, []).append((part, surface))

    atlases = []
    for material, members in groups.items():
        labels, offset = [], 0
        for part, surface in members:
            shells = _shell_labels(part, surface)
            labels.append(shells + offset)
            offset += int(shells.max()) + 1
        atlases.append(
            (
                material,
                _resolution(context, material),
                np.concatenate([p.uvs for p, _ in members]),
                np.concatenate(labels),
            )
        )
    return atlases


def _uv_areas(uvs: np.ndarray) -> np.ndarray:
    edge_a = uvs[:, 1] - uvs[:, 0]
    edge_b = uvs[:, 2] - uvs[:, 0]
    return 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])


@check("uv.stretch")
def stretch(context: registry.Context) -> Result:
    """Conformal and area distortion of the 3D-to-UV map, per triangle."""
    usable, missing = _mapped(context)
    if not usable and not missing:
        return registry.not_applicable({}, NO_SAMPLING)
    conformal, area = [], []
    for part, surface in usable:
        singular = _singular_values(part.uvs, surface)
        valid = singular[:, 1] > 0
        if not valid.any():
            continue
        conformal.append(singular[valid, 0] / singular[valid, 1])
        scale = np.sqrt(_uv_areas(part.uvs).sum() / max(surface.areas.sum(), 1e-30))
        area.append(singular[valid].prod(axis=1) / max(scale**2, 1e-30))

    if not conformal:
        return verdict(False, {"parts_without_uv": len(missing)}, "no UVs to measure")

    p95 = float(np.percentile(np.concatenate(conformal), 95))
    return verdict(
        p95 <= context.thresholds.uv_conformal_p95 and not missing,
        {
            "conformal_p95": p95,
            "area_distortion_p95": float(np.percentile(np.concatenate(area), 95)),
            "parts_without_uv": len(missing),
        },
        f"conformal distortion {p95:.2f} at p95 across {len(usable)} part(s)",
    )


def _singular_values(uvs: np.ndarray, surface: geometry.Surface) -> np.ndarray:
    """Singular values of each triangle's UV-to-surface Jacobian."""
    corners = surface.corners
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    basis_x = geometry.normalize(edge_a)
    normal = geometry.normalize(np.cross(edge_a, edge_b))
    basis_y = np.cross(normal, basis_x)

    flattened = np.stack(
        [
            np.stack([(edge_a * basis_x).sum(1), (edge_a * basis_y).sum(1)], axis=1),
            np.stack([(edge_b * basis_x).sum(1), (edge_b * basis_y).sum(1)], axis=1),
        ],
        axis=2,
    )
    texture = np.stack([uvs[:, 1] - uvs[:, 0], uvs[:, 2] - uvs[:, 0]], axis=2)

    determinant = np.linalg.det(flattened)
    usable = np.abs(determinant) > 1e-30
    jacobian = np.zeros_like(texture)
    jacobian[usable] = texture[usable] @ np.linalg.inv(flattened[usable])
    return np.linalg.svd(jacobian, compute_uv=False)


@check("uv.degenerate")
def degenerate(context: registry.Context) -> Result:
    """Triangles with real surface area collapsed to nothing in UV space."""
    usable, missing = _mapped(context)
    if not usable and not missing:
        return registry.not_applicable({}, NO_SAMPLING)
    total = sum(
        int(((_uv_areas(p.uvs) <= 1e-12) & (s.areas > 0)).sum()) for p, s in usable
    )
    return verdict(
        total == 0 and not missing,
        {"degenerate_uv_triangles": total, "parts_without_uv": len(missing)},
        f"{total} triangle(s) collapse in UV space",
    )


@check("uv.shells")
def shells(context: registry.Context) -> Result:
    """How far the unwrap shatters the surface into islands.

    Island count measures segmentation independently of mesh tessellation. It
    distinguishes deliberate unwraps from automatic fragmentation.
    """
    usable, missing = _mapped(context)
    if not usable and not missing:
        return registry.not_applicable({}, NO_SAMPLING)
    counts = [len(np.unique(_shell_labels(p, s))) for p, s in usable]
    worst = max(counts, default=0)
    return verdict(
        worst <= context.thresholds.uv_shells_per_part,
        {
            "shells_total": int(sum(counts)),
            "shells_max_per_part": int(worst),
            "parts_without_uv": len(missing),
        },
        f"a part is split into {worst} UV shells",
    )


def _shell_labels(part, surface: geometry.Surface) -> np.ndarray:
    """Connected-component label per triangle, cut wherever UVs do not agree."""
    adjacency = geometry.shared_edges(surface)
    count = len(surface.triangles)
    if not len(adjacency):
        return np.arange(count)

    joined = np.ones(len(adjacency), dtype=bool)
    for k in (0, 1):
        left = part.uvs[adjacency.left, adjacency.left_slots[:, k]]
        right = part.uvs[adjacency.right, adjacency.right_slots[:, k]]
        joined &= np.isclose(left, right, atol=1e-6).all(axis=1)

    graph = scipy.sparse.coo_matrix(
        (np.ones(joined.sum()), (adjacency.left[joined], adjacency.right[joined])),
        shape=(count, count),
    )
    return scipy.sparse.csgraph.connected_components(graph, directed=False)[1]


@check("uv.texel_density")
def texel_density(context: registry.Context) -> Result:
    """Spread of texels per unit surface area, across atlas-bound parts."""
    if not _textured(context):
        return registry.not_applicable({}, UNTEXTURED)
    usable, missing = _atlas_parts(context)
    densities, weights = [], []
    for part, surface in usable:
        valid = surface.areas > 0
        if not valid.any():
            continue
        ratio = np.sqrt(_uv_areas(part.uvs)[valid] / surface.areas[valid])
        densities.append(ratio * _resolution(context, part.material))
        weights.append(surface.areas[valid])

    if not densities:
        return verdict(False, {"parts_without_uv": len(missing)}, "no UVs to measure")

    values = np.concatenate(densities)
    weight = np.concatenate(weights)
    mean = np.average(values, weights=weight)
    spread = np.sqrt(np.average((values - mean) ** 2, weights=weight))
    variation = float(spread / mean) if mean else 0.0
    return verdict(
        variation <= context.thresholds.uv_texel_density_cv and not missing,
        {
            "texels_per_unit_mean": float(mean),
            "coefficient_of_variation": variation,
            "parts_without_uv": len(missing),
        },
        f"texel density varies by {variation:.2f} of its mean",
    )


@check("uv.out_of_bounds")
def out_of_bounds(context: registry.Context) -> Result:
    """Atlas UV coordinates outside the unit square.

    Only base-colour atlases require the unit square; tiled detail maps run
    past it on purpose.
    """
    if not _textured(context):
        return registry.not_applicable({}, UNTEXTURED)
    usable, missing = _atlas_parts(context)
    total = sum(
        int(((p.uvs < 0.0) | (p.uvs > 1.0)).any(axis=2).sum()) for p, _ in usable
    )
    return verdict(
        total == 0 and not missing,
        {"out_of_bounds_corners": total, "parts_without_uv": len(missing)},
        f"{total} UV corner(s) fall outside the unit square",
    )


@check("uv.utilization")
def utilization(context: registry.Context) -> Result:
    """Share of each atlas the layout actually covers."""
    if not _textured(context):
        return registry.not_applicable({}, UNTEXTURED)
    missing = _atlas_parts(context)[1]
    covered = [float((counts > 0).mean()) for counts, _ in _rasterized(context)]
    if not covered:
        return verdict(False, {"parts_without_uv": len(missing)}, "no UVs to measure")

    worst = min(covered)
    return advisory(
        worst >= context.thresholds.uv_utilization_warn,
        {
            "atlas_utilization_min": worst,
            "atlas_utilization_mean": float(np.mean(covered)),
            "atlases": len(covered),
            "parts_without_uv": len(missing),
        },
        f"an atlas is only {worst:.1%} covered",
    )


@check("uv.seams_on_sharp_edges")
def seams_on_sharp_edges(context: registry.Context) -> Result:
    """Share of UV seams that follow a sharp crease rather than smooth surface."""
    usable, missing = _mapped(context)
    sharp = seams = 0
    limit = np.deg2rad(context.thresholds.uv_sharp_edge_deg)
    for part, surface in usable:
        adjacency = geometry.shared_edges(surface)
        if not len(adjacency):
            continue
        cut = np.zeros(len(adjacency), dtype=bool)
        for k in (0, 1):
            left = part.uvs[adjacency.left, adjacency.left_slots[:, k]]
            right = part.uvs[adjacency.right, adjacency.right_slots[:, k]]
            cut |= ~np.isclose(left, right, atol=1e-6).all(axis=1)

        dihedral = surface.mesh.face_adjacency_angles
        seams += int(cut.sum())
        sharp += int((dihedral[cut] > limit).sum())

    return registry.info(
        {
            "seam_edges": seams,
            "seams_on_sharp_edges": sharp,
            "sharp_seam_ratio": sharp / seams if seams else 0.0,
            "parts_without_uv": len(missing),
        }
    )
