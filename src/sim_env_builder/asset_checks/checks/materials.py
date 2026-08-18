"""Material and texture quality, including a heuristic for baked-in lighting."""

import collections

import numpy as np
import skimage.color
import trimesh
import trimesh.sample

from sim_env_builder.asset_checks import geometry
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.checks.registry import Result, check, verdict

PBR_SHADERS = ("UsdPreviewSurface", "pbrMetallicRoughness")
LUMA = np.array([0.2126, 0.7152, 0.0722])


@check("materials.bound")
def bound(context: registry.Context) -> Result:
    """Every render part has a material binding."""
    unbound = [p.name for p in context.asset.parts if p.material is None]
    return verdict(
        not unbound,
        {
            "materials": len(context.asset.materials),
            "unbound_parts": len(unbound),
            "examples": sorted(unbound)[:10],
        },
        f"{len(unbound)} part(s) have no material bound",
    )


@check("materials.pbr")
def pbr(context: registry.Context) -> Result:
    """Materials use a physically based surface model."""
    materials = context.asset.materials
    if not materials:
        return verdict(False, {"materials": 0}, "asset has no materials")

    other = [m.name for m in materials.values() if m.shader not in PBR_SHADERS]
    return verdict(
        not other,
        {
            "materials": len(materials),
            "non_pbr_materials": len(other),
            "examples": sorted(other)[:10],
        },
        f"{len(other)} material(s) do not use a PBR surface",
    )


@check("materials.detail_maps")
def detail_maps(context: registry.Context) -> Result:
    """Share of materials carrying a normal, bump or displacement map."""
    materials = list(context.asset.materials.values())
    if not materials:
        return verdict(False, {"materials": 0}, "asset has no materials")

    detailed = sum(m.has_detail_map for m in materials)
    fraction = detailed / len(materials)
    return verdict(
        fraction >= context.thresholds.detail_map_fraction,
        {
            "materials": len(materials),
            "with_detail_map": detailed,
            "detail_map_fraction": fraction,
        },
        f"only {fraction:.0%} of materials carry a surface-detail map",
    )


@check("materials.duplicates")
def duplicates(context: registry.Context) -> Result:
    """Distinct materials whose parameters and textures are identical."""
    groups = collections.defaultdict(list)
    for material in context.asset.materials.values():
        groups[material.fingerprint()].append(material.name)

    repeated = [sorted(names) for names in groups.values() if len(names) > 1]
    return verdict(
        not repeated,
        {
            "materials": len(context.asset.materials),
            "duplicate_groups": len(repeated),
            "examples": sorted(repeated)[:5],
        },
        f"{len(repeated)} group(s) of materials are interchangeable",
    )


@check("materials.missing_textures")
def missing_textures(context: registry.Context) -> Result:
    """Texture references that do not resolve to a readable image."""
    broken = [
        f"{material.name}:{slot}"
        for material in context.asset.materials.values()
        for slot, texture in material.textures.items()
        if texture.image is None
    ]
    return verdict(
        not broken,
        {"missing_textures": len(broken), "examples": sorted(broken)[:10]},
        f"{len(broken)} texture reference(s) do not resolve",
    )


@check("materials.resolution")
def resolution(context: registry.Context) -> Result:
    """Texture detail relative to physical size, and absolute texture size."""
    limits = context.thresholds
    oversized = [
        f"{material.name}:{slot}"
        for material in context.asset.materials.values()
        for slot, texture in material.textures.items()
        if texture.size and max(texture.size) > limits.texture_max_side
    ]

    densities, weights = [], []
    for part, surface in zip(context.asset.parts, context.surfaces):
        if part.uvs is None or not len(surface.triangles):
            continue
        valid = surface.areas > 0
        if not valid.any():
            continue
        side = _texture_side(context, part.material)
        ratio = np.sqrt(_uv_area(part.uvs)[valid] / surface.areas[valid]) * side
        densities.append(ratio / 100.0)
        weights.append(surface.areas[valid])

    if not densities:
        return registry.not_applicable({}, "no textured geometry to measure")

    median = float(
        np.quantile(
            np.concatenate(densities),
            0.5,
            weights=np.concatenate(weights),
            method="inverted_cdf",
        )
    )
    in_range = limits.texels_per_cm_min <= median <= limits.texels_per_cm_max
    return verdict(
        in_range and not oversized,
        {
            "texels_per_cm_median": median,
            "oversized_textures": len(oversized),
            "examples": sorted(oversized)[:10],
        },
        f"{median:.1f} texels/cm with {len(oversized)} oversized texture(s)",
    )


def _texture_side(context: registry.Context, material_name: str | None) -> int:
    material = context.asset.materials.get(material_name)
    texture = material.textures.get("basecolor") if material else None
    if texture and texture.size:
        return max(texture.size)
    return context.thresholds.raster_resolution


def _uv_area(uvs: np.ndarray) -> np.ndarray:
    edge_a = uvs[:, 1] - uvs[:, 0]
    edge_b = uvs[:, 2] - uvs[:, 0]
    return 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])


@check("materials.baked_lighting")
def baked_lighting(context: registry.Context) -> Result:
    """Signs that artists painted shading into base colour instead of leaving it
    to the renderer.

    Two independent signals contribute: correlation between base-colour luminance
    and geometric accessibility, and a large-scale linear brightness ramp across
    a UV shell. Neither signal is conclusive, so the result remains heuristic.
    """
    textured = [
        (part, surface)
        for part, surface in zip(context.asset.parts, context.surfaces)
        if part.uvs is not None
        and _basecolor(context, part.material) is not None
        and len(surface.triangles)
    ]
    if not textured:
        return registry.not_applicable({}, "no base-colour textures to analyse")

    budget = max(context.thresholds.shading_samples // len(textured), 32)
    correlation, ramps = [], []
    for index, (part, surface) in enumerate(textured):
        points, normals, coords = _sample_surface(surface, part.uvs, budget, index)
        colours = _sample_texture(_basecolor(context, part.material), coords)
        luminance = colours @ LUMA
        if luminance.std() < 1e-6:
            continue

        openness = 1.0 - geometry.occlusion(
            context.combined_mesh,
            points,
            normals,
            context.thresholds.occlusion_directions,
        )
        if openness.std() > 1e-6:
            correlation.append(float(np.corrcoef(openness, luminance)[0, 1]))
        ramps.append(_ramp_strength(coords, luminance))

    if not correlation and not ramps:
        return registry.not_applicable({}, "base colour has no measurable variation")

    worst = max(correlation, default=0.0)
    ramp = float(np.mean(ramps)) if ramps else 0.0
    limits = context.thresholds
    return Result(
        registry.FAIL
        if worst > limits.baked_lighting_correlation or ramp > limits.baked_lighting_gradient
        else registry.PASS,
        {
            "accessibility_luminance_r": worst,
            "shell_gradient": ramp,
            "parts_analysed": len(textured),
        },
        f"base colour tracks accessibility (r={worst:.2f}) "
        f"with a {ramp:.2f} luminance ramp",
    )


def _ramp_strength(coords: np.ndarray, luminance: np.ndarray) -> float:
    """Amplitude of the best linear brightness ramp in UV, over mean luminance."""
    design = np.column_stack([coords, np.ones(len(coords))])
    fitted = design @ np.linalg.lstsq(design, luminance, rcond=None)[0]
    mean = float(luminance.mean())
    return float(np.ptp(fitted) / mean) if mean > 1e-6 else 0.0


@check("materials.seam_discontinuity")
def seam_discontinuity(context: registry.Context) -> Result:
    """Colour and normal mismatch where a UV seam splits the surface."""
    colour_gaps, normal_gaps = [], []
    for part, surface in zip(context.asset.parts, context.surfaces):
        if part.uvs is None or not len(surface.triangles):
            continue
        material = context.asset.materials.get(part.material)
        if material is None:
            continue

        adjacency = geometry.shared_edges(surface)
        left, right = _seam_midpoints(part, adjacency)
        if not len(left):
            continue

        basecolor = material.textures.get("basecolor")
        if basecolor is not None and basecolor.image is not None:
            colour_gaps.append(_delta_e(basecolor, left, right))
        normal = material.textures.get("normal")
        if normal is not None and normal.image is not None:
            normal_gaps.append(_normal_angle(normal, left, right))

    if not colour_gaps and not normal_gaps:
        return registry.not_applicable({}, "no textured seams to compare")

    limits = context.thresholds
    colour_p95 = _percentile(colour_gaps)
    normal_p95 = _percentile(normal_gaps)
    return verdict(
        colour_p95 <= limits.seam_delta_e_p95 and normal_p95 <= limits.seam_normal_deg_p95,
        {"delta_e_p95": colour_p95, "normal_angle_p95_deg": normal_p95},
        f"seams differ by dE {colour_p95:.1f} and {normal_p95:.1f} degrees at p95",
    )


def _seam_midpoints(part, adjacency: geometry.SharedEdges) -> tuple[np.ndarray, np.ndarray]:
    """UV midpoint of each seam edge, as seen from either side."""
    if not len(adjacency):
        return np.zeros((0, 2)), np.zeros((0, 2))
    left = np.stack(
        [part.uvs[adjacency.left, adjacency.left_slots[:, k]] for k in (0, 1)]
    )
    right = np.stack(
        [part.uvs[adjacency.right, adjacency.right_slots[:, k]] for k in (0, 1)]
    )
    cut = ~np.isclose(left, right, atol=1e-6).all(axis=2).all(axis=0)
    return left[:, cut].mean(axis=0), right[:, cut].mean(axis=0)


def _percentile(groups: list[np.ndarray]) -> float:
    return float(np.percentile(np.concatenate(groups), 95)) if groups else 0.0


def _delta_e(texture, left, right) -> np.ndarray:
    a = skimage.color.rgb2lab(_sample_texture(texture, left))
    b = skimage.color.rgb2lab(_sample_texture(texture, right))
    return skimage.color.deltaE_ciede2000(a, b)


def _normal_angle(texture, left, right) -> np.ndarray:
    a = geometry.normalize(_sample_texture(texture, left) * 2.0 - 1.0)
    b = geometry.normalize(_sample_texture(texture, right) * 2.0 - 1.0)
    return np.rad2deg(np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0)))


def _basecolor(context: registry.Context, material_name: str | None):
    material = context.asset.materials.get(material_name)
    if material is None:
        return None
    texture = material.textures.get("basecolor")
    return texture if texture is not None and texture.image is not None else None


def _sample_surface(
    surface: geometry.Surface, uvs: np.ndarray, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted surface points, with their normals and interpolated UVs."""
    points, faces = trimesh.sample.sample_surface(surface.mesh, count, seed=seed)
    barycentric = trimesh.triangles.points_to_barycentric(
        surface.corners[faces], points
    )
    coords = (barycentric[:, :, None] * uvs[faces]).sum(axis=1)
    return np.asarray(points), surface.face_normals[faces], coords


def _sample_texture(texture, coords: np.ndarray) -> np.ndarray:
    """Nearest-texel lookup.

    `Part.uvs` uses the image's lower-left corner as its origin. USD authors `st`
    this way, and the glTF loader converts the format's top-left origin during
    import. Image row 0 is the top, so use `1 - v`.
    """
    image = np.asarray(texture.image.convert("RGB"), dtype=np.float64) / 255.0
    height, width = image.shape[:2]
    x = np.clip((coords[:, 0] % 1.0 * width).astype(int), 0, width - 1)
    # Wrap before flipping. Repeat wrapping maps v and v + 1 to the same texel,
    # and v = 0 to the bottom row.
    y = np.clip(((1.0 - coords[:, 1] % 1.0) * height).astype(int), 0, height - 1)
    return image[y, x]
