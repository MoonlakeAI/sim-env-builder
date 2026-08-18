"""Collision-proxy quality.

`sim_ready.colliders_present` asks whether the asset declares collision. This
section checks for dedicated proxy geometry and measures how well it represents
the render mesh.
"""

import numpy as np
import trimesh

from sim_env_builder.asset_checks import geometry
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.checks.registry import Result, check, verdict

DEDICATED = ("authored_mesh", "convex_hull", "primitive")


@check("proxy.present")
def present(context: registry.Context) -> Result:
    """Every link with render geometry has a collider distinct from that mesh."""
    proxies = context.asset.proxies
    missing = [
        name
        for name, link in context.asset.links.items()
        if link.parts
        and not any(proxies[i].source in DEDICATED for i in link.proxies)
    ]
    return verdict(
        not missing,
        {
            "proxies": len(proxies),
            "reused_render_meshes": sum(p.source == "render_mesh" for p in proxies),
            "links_without_proxy": len(missing),
            "examples": sorted(missing)[:10],
        },
        f"{len(missing)} link(s) collide against render geometry with no proxy",
    )


@check("proxy.poly_budget", requires="physics_schema")
def poly_budget(context: registry.Context) -> Result:
    """Proxies stay small, both absolutely and relative to the render mesh."""
    proxies = context.asset.proxies
    if not proxies:
        return registry.not_applicable({}, "asset has no colliders")

    limits = context.thresholds
    largest = max(len(p.triangles) for p in proxies)
    render = context.asset.triangle_count
    ratio = sum(len(p.triangles) for p in proxies) / render if render else 0.0
    over = [p.name for p in proxies if len(p.triangles) > limits.proxy_triangle_budget]
    return verdict(
        not over and ratio <= limits.proxy_render_ratio,
        {
            "max_proxy_triangles": largest,
            "proxy_render_ratio": ratio,
            "proxies_over_budget": len(over),
            "examples": sorted(over)[:10],
        },
        f"{len(over)} proxy/proxies over {limits.proxy_triangle_budget} triangles, "
        f"proxy/render ratio {ratio:.2f}",
    )


@check("proxy.watertight", requires="physics_schema")
def watertight(context: registry.Context) -> Result:
    """Every proxy component is closed."""
    surfaces = context.proxy_surfaces
    if not surfaces:
        return registry.not_applicable({}, "asset has no colliders")

    leaking = [
        context.asset.proxies[i].name
        for i, surface in enumerate(surfaces)
        if len(surface.boundary_edges)
    ]
    return verdict(
        not leaking,
        {
            "proxies": len(surfaces),
            "open_proxies": len(leaking),
            "examples": sorted(leaking)[:10],
        },
        f"{len(leaking)} proxy/proxies are not closed",
    )


@check("proxy.surface_distance", requires="physics_schema")
def surface_distance(context: registry.Context) -> Result:
    """How far each link's collision surface sits from its render surface."""
    limits = context.thresholds
    distances, protrusion = [], 0.0
    for link in context.asset.links.values():
        if not link.parts or not link.proxies:
            continue
        render = _combine([context.surfaces[i].mesh for i in link.parts])
        collision = _combine([context.proxy_surfaces[i].mesh for i in link.proxies])
        if not len(render.faces) or not len(collision.faces):
            continue

        outward = _signed_gap(render, collision, limits.surface_samples)
        distances.append(np.abs(outward))
        distances.append(np.abs(_signed_gap(collision, render, limits.surface_samples)))
        protrusion = max(protrusion, float(outward.max(initial=0.0)))

    if not distances:
        return registry.not_applicable({}, "no link has both render and proxy geometry")

    p95 = float(np.percentile(np.concatenate(distances), 95))
    return verdict(
        p95 < context.rel(limits.proxy_surface_p95_rel)
        and protrusion < context.rel(limits.proxy_penetration_rel),
        {
            "p95_distance": p95,
            "max_protrusion": protrusion,
            "p95_limit": context.rel(limits.proxy_surface_p95_rel),
            "protrusion_limit": context.rel(limits.proxy_penetration_rel),
        },
        f"proxy deviates from the render surface by {p95:.4g} at p95, "
        f"protruding up to {protrusion:.4g}",
    )


def _combine(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.util.concatenate(meshes or [trimesh.Trimesh()])


def _signed_gap(
    reference: trimesh.Trimesh, sampled: trimesh.Trimesh, count: int
) -> np.ndarray:
    """Distance from `sampled`'s surface to `reference`, positive when outside.

    Sampling includes face interiors: a coarse face can bridge far from the
    reference while all of its vertices sit on it.
    """
    points = geometry.surface_points(sampled, count)
    return -trimesh.proximity.signed_distance(reference, points)
