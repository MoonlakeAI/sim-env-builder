"""Render-geometry quality: topology, degeneracy, orientation and density."""

import numpy as np
import trimesh

from sim_env_builder.asset_checks import geometry
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.checks.registry import Result, advisory, check, verdict


@check("mesh.quad_ratio", requires="polygons")
def quad_ratio(context: registry.Context) -> Result:
    """Share of polygons that are quads, across the whole asset."""
    counts = np.concatenate([p.face_counts for p in context.asset.parts])
    ratio = float((counts == 4).mean())
    return advisory(
        ratio >= context.thresholds.quad_ratio_warn,
        {"quad_ratio": ratio, "polygons": len(counts)},
        f"quad share {ratio:.2f} is below {context.thresholds.quad_ratio_warn}",
    )


@check("mesh.valence", requires="polygons")
def valence(context: registry.Context) -> Result:
    """Share of interior polygon vertices with four incident edges."""
    matching = total = 0
    for part, surface in zip(context.asset.parts, context.surfaces):
        welded = surface.weld_map[part.face_indices]
        edges, uses = geometry.polygon_edges(part.face_counts, welded)
        incident = np.bincount(edges.ravel(), minlength=len(surface.welded_vertices))
        interior = np.ones(len(surface.welded_vertices), dtype=bool)
        interior[edges[uses == 1].ravel()] = False
        matching += int((incident[interior] == 4).sum())
        total += int(interior.sum())
    return registry.info(
        {"valence_four_ratio": matching / total if total else 0.0, "vertices": total}
    )


@check("mesh.ngons", requires="polygons")
def ngons(context: registry.Context) -> Result:
    """Polygons with more than four vertices."""
    counts = np.concatenate([p.face_counts for p in context.asset.parts])
    total = int((counts > 4).sum())
    return verdict(
        total == 0,
        {"ngons": total, "polygons": len(counts), "max_sides": int(counts.max())},
        f"{total} polygon(s) have more than four sides",
    )


@check("mesh.degenerate_faces")
def degenerate_faces(context: registry.Context) -> Result:
    """Sliver and needle triangles, by corner angle and aspect ratio."""
    limits = context.thresholds
    min_angle, max_angle, worst_aspect, bad = np.pi, 0.0, 1.0, 0
    for surface in context.surfaces:
        if not len(surface.triangles):
            continue
        angles = surface.corner_angles
        aspect = _aspect_ratio(surface)
        min_angle = min(min_angle, float(angles.min()))
        max_angle = max(max_angle, float(angles.max()))
        worst_aspect = min(worst_aspect, float(aspect.min()))
        bad += int(
            (
                (angles.min(axis=1) < np.deg2rad(limits.min_triangle_angle_deg))
                | (aspect < limits.min_triangle_aspect)
            ).sum()
        )
    return verdict(
        bad == 0,
        {
            "degenerate_triangles": bad,
            "min_angle_deg": float(np.rad2deg(min_angle)),
            "max_angle_deg": float(np.rad2deg(max_angle)),
            "min_aspect": worst_aspect,
        },
        f"{bad} triangle(s) are slivers or needles",
    )


def _aspect_ratio(surface: geometry.Surface) -> np.ndarray:
    """Twice the inradius over the circumradius; 1 for an equilateral triangle."""
    corners = surface.corners
    sides = np.stack(
        [
            np.linalg.norm(corners[:, (i + 1) % 3] - corners[:, (i + 2) % 3], axis=1)
            for i in range(3)
        ],
        axis=1,
    )
    perimeter = sides.sum(axis=1)
    product = sides.prod(axis=1)
    # 2r/R with r = 2A/perimeter and R = abc/4A.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 16.0 * surface.areas**2 / (perimeter * product)
    return np.nan_to_num(ratio, nan=0.0, posinf=0.0)


@check("mesh.zero_area_faces")
def zero_area_faces(context: registry.Context) -> Result:
    """Triangles with no measurable area."""
    limit = context.thresholds.zero_area_rel * context.asset.bbox_diag**2
    total = sum(int((s.areas < limit).sum()) for s in context.surfaces)
    return verdict(
        total == 0,
        {"zero_area_triangles": total, "area_threshold": limit},
        f"{total} triangle(s) have effectively zero area",
    )


@check("mesh.duplicate_vertices")
def duplicate_vertices(context: registry.Context) -> Result:
    """Coincident vertices that also agree in normal and UV, so carry no seam."""
    total = 0
    for part in context.asset.parts:
        if not len(part.vertices):
            continue
        features = [np.round(part.vertices / context.weld_tol)]
        if part.normals is not None:
            features.append(np.round(part.normals, 4))
        if part.uvs is not None:
            features.append(np.round(_vertex_uvs(part), 4))
        stacked = np.concatenate(features, axis=1)
        total += len(stacked) - len(np.unique(stacked, axis=0))
    return verdict(
        total == 0,
        {"duplicate_vertices": total},
        f"{total} vertex/vertices duplicate another with the same normal and UV",
    )


def _vertex_uvs(part) -> np.ndarray:
    """Mean UV per vertex; seam vertices differ and so are not duplicates."""
    summed = np.zeros((len(part.vertices), 2))
    counts = np.zeros(len(part.vertices))
    np.add.at(summed, part.triangles.ravel(), part.uvs.reshape(-1, 2))
    np.add.at(counts, part.triangles.ravel(), 1.0)
    return summed / np.where(counts == 0, 1.0, counts)[:, None]


@check("mesh.loose_vertices")
def loose_vertices(context: registry.Context) -> Result:
    """Count vertices that no triangle references."""
    total = 0
    for part in context.asset.parts:
        used = np.zeros(len(part.vertices), dtype=bool)
        used[part.triangles.ravel()] = True
        total += int((~used).sum())
    return verdict(
        total == 0, {"loose_vertices": total}, f"{total} vertex/vertices are unused"
    )


@check("mesh.non_manifold")
def non_manifold(context: registry.Context) -> Result:
    """Edges shared by more than two triangles, and bow-tie vertices."""
    edges = sum(len(s.nonmanifold_edges) for s in context.surfaces)
    bowties = sum(len(geometry.bowtie_vertices(s)) for s in context.surfaces)
    return verdict(
        edges == 0 and bowties == 0,
        {"nonmanifold_edges": edges, "bowtie_vertices": bowties},
        f"{edges} non-manifold edge(s), {bowties} bow-tie vertex/vertices",
    )


@check("mesh.watertight")
def watertight(context: registry.Context) -> Result:
    """Every connected component is closed."""
    components = open_components = 0
    for surface in context.surfaces:
        if not len(surface.triangles):
            continue
        labels = surface.vertex_components
        components += int(labels.max()) + 1
        open_components += len(np.unique(labels[surface.boundary_edges.ravel()]))
    return verdict(
        open_components == 0,
        {"components": components, "open_components": open_components},
        f"{open_components} of {components} component(s) are not closed",
    )


@check("mesh.normals_outward")
def normals_outward(context: registry.Context) -> Result:
    """Closed components enclose positive volume and wind consistently."""
    inward = checked = 0
    tangled = []
    for part, surface in zip(context.asset.parts, context.surfaces):
        if not len(surface.triangles):
            continue
        if not surface.mesh.is_winding_consistent:
            tangled.append(part.name)
        labels = surface.vertex_components
        open_labels = set(np.unique(labels[surface.boundary_edges.ravel()]).tolist())
        for label in range(int(labels.max()) + 1):
            if label in open_labels:
                continue
            selected = surface.welded_triangles[surface.triangle_components == label]
            checked += 1
            inward += geometry.signed_volume(surface.welded_vertices, selected) < 0
    return verdict(
        inward == 0 and not tangled,
        {
            "closed_components": checked,
            "inward_components": int(inward),
            "parts_wound_inconsistently": len(tangled),
            "examples": sorted(tangled)[:10],
        },
        f"{inward} component(s) face inward, "
        f"{len(tangled)} part(s) wound inconsistently",
    )


@check("mesh.self_intersection")
def self_intersection(context: registry.Context) -> Result:
    """Triangles within one part that pass through each other."""
    limit = context.thresholds.self_intersection_max_triangles
    if context.asset.triangle_count > limit:
        return registry.not_applicable(
            {"triangles": context.asset.triangle_count},
            f"mesh exceeds the {limit} triangle analysis cap",
        )

    total = 0
    for surface in context.surfaces:
        pairs = geometry.candidate_pairs(surface)
        if not len(pairs):
            continue
        corners = surface.mesh.triangles
        total += int(
            geometry.triangles_intersect(
                corners[pairs[:, 0]], corners[pairs[:, 1]]
            ).sum()
        )
    return verdict(
        total == 0,
        {"intersecting_pairs": total},
        f"{total} triangle pair(s) intersect within a part",
    )


@check("mesh.rest_penetration")
def rest_penetration(context: registry.Context) -> Result:
    """How deeply separate parts overlap in the authored rest pose."""
    limit = context.rel(context.thresholds.rest_penetration_rel)
    depth, culprit = 0.0, None
    for i, j in _overlapping_pairs([s.mesh.bounds for s in context.surfaces]):
        measured, _ = geometry.penetration_depth(
            context.surfaces[i].mesh,
            context.surfaces[j].mesh,
            context.thresholds.penetration_samples,
        )
        if measured > depth:
            depth = measured
            culprit = [context.asset.parts[i].name, context.asset.parts[j].name]
    return verdict(
        depth < limit,
        {"max_depth": depth, "depth_limit": limit, "parts": culprit},
        f"parts overlap by {depth:.4g} (limit {limit:.4g})",
    )


def _overlapping_pairs(bounds: list[np.ndarray]) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(len(bounds))
        for j in range(i + 1, len(bounds))
        if (bounds[i][0] <= bounds[j][1]).all() and (bounds[j][0] <= bounds[i][1]).all()
    ]


@check("mesh.floaters")
def floaters(context: registry.Context) -> Result:
    """Components sitting clear of every other piece of geometry.

    Surface-to-surface distance excludes touching parts, such as a knob seated
    against a body or a link resting on its neighbour.
    """
    components = _components(context)
    if len(components) < 2:
        return Result(registry.PASS, {"components": len(components), "floaters": 0})

    limit = context.rel(context.thresholds.floater_gap_rel)
    budget = max(context.thresholds.surface_samples // len(components), 1)

    detached, worst = [], 0.0
    for index, (name, mesh) in enumerate(components):
        gap = _nearest_component(mesh, components, index, limit, budget)
        worst = max(worst, gap)
        if gap > limit:
            detached.append(name)
    return verdict(
        not detached,
        {
            "components": len(components),
            "floaters": len(detached),
            "max_gap": worst,
            "gap_limit": limit,
            "examples": sorted(detached)[:10],
        },
        f"{len(detached)} component(s) are detached from the rest of the asset",
    )


def _nearest_component(mesh, components, index: int, limit: float, budget: int) -> float:
    """Distance from one component's surface to the closest other component.

    Bounding boxes farther apart than `limit` cannot contain the nearest surface.
    Only the remaining candidates need an exact surface query. If none remain,
    the function returns the box distance, a lower bound already over the limit.

    The function samples candidates in both directions. Sampling only a large
    part can miss its contact patch with a small part.
    """
    bounded = [
        (_box_gap(mesh.bounds, other.bounds), other)
        for position, (_, other) in enumerate(components)
        if position != index
    ]
    nearby = [other for gap, other in bounded if gap <= limit]
    if not nearby:
        return min(gap for gap, _ in bounded)

    return min(
        min(
            _surface_gap(mesh, other, budget),
            _surface_gap(other, mesh, budget),
        )
        for other in nearby
    )


def _surface_gap(sampled, target, budget: int) -> float:
    points = geometry.surface_points(sampled, budget)
    return float(trimesh.proximity.closest_point(target, points)[1].min())


def _box_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0.0)))


def _components(context: registry.Context) -> list[tuple[str, trimesh.Trimesh]]:
    """One mesh per connected component of the asset."""
    components = []
    for part, surface in zip(context.asset.parts, context.surfaces):
        if not len(surface.triangles):
            continue
        labels = surface.triangle_components
        for label in np.unique(labels):
            components.append(
                (part.name, surface.mesh.submesh([labels == label], append=True))
            )
    return components


@check("mesh.spikes")
def spikes(context: registry.Context) -> Result:
    """Vertices whose surrounding surface pinches into a needle."""
    limit = context.thresholds.spike_defect_rad
    worst, total = 0.0, 0
    for surface in context.surfaces:
        if not len(surface.triangles):
            continue
        defect = geometry.interior_defects(surface)
        if not len(defect):
            continue
        worst = max(worst, float(defect.max()))
        total += int((defect > limit).sum())
    return verdict(
        total == 0,
        {"spike_vertices": total, "max_angle_defect": worst, "defect_limit": limit},
        f"{total} vertex/vertices form a spike",
    )


@check("mesh.poly_budget")
def poly_budget(context: registry.Context) -> Result:
    """Total render triangles across the asset."""
    total = context.asset.triangle_count
    return verdict(
        total <= context.thresholds.poly_budget,
        {"triangles": total, "budget": context.thresholds.poly_budget},
        f"{total} triangles exceeds the budget of {context.thresholds.poly_budget}",
    )


@check("mesh.density")
def density(context: registry.Context) -> Result:
    """Edge length relative to local curvature radius, as a distribution."""
    ratios = []
    for surface in context.surfaces:
        if len(surface.triangles) < 4:
            continue
        ratios.append(
            geometry.median_edge_length(surface) * geometry.mean_curvature(surface)
        )
    if not ratios:
        return registry.not_applicable({}, "asset has no geometry")

    values = np.concatenate(ratios)
    percentiles = np.percentile(values, [5, 50, 95])
    return registry.info(
        {
            "edge_over_curvature_radius_p5": float(percentiles[0]),
            "edge_over_curvature_radius_p50": float(percentiles[1]),
            "edge_over_curvature_radius_p95": float(percentiles[2]),
        }
    )
