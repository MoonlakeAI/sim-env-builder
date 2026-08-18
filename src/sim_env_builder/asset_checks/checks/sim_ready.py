"""Physics authoring: colliders, mass, friction and joint parameters.

Absence is the finding here rather than a reason to skip: a package that carries
no physics data is not sim-ready, whatever its format.
"""

import numpy as np

from sim_env_builder.asset_checks import geometry
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.checks.registry import check, verdict
from sim_env_builder.asset_checks.ingest import model


def _renderable_links(context: registry.Context) -> list[str]:
    return [name for name, link in context.asset.links.items() if link.parts]


def _dynamic_links(context: registry.Context) -> list[str]:
    """Renderable links that are rigid bodies rather than static scenery."""
    return [n for n in _renderable_links(context) if n != model.STATIC_LINK]


def _names(values: list[str], limit: int = 10) -> list[str]:
    return sorted(values)[:limit]


@check("sim_ready.colliders_present")
def colliders_present(context: registry.Context) -> registry.Result:
    """Every link with render geometry carries at least one collision shape."""
    links = context.asset.links
    missing = [name for name in _renderable_links(context) if not links[name].proxies]
    return verdict(
        not missing,
        {
            "links_with_geometry": len(_renderable_links(context)),
            "links_without_collider": len(missing),
            "examples": _names(missing),
        },
        f"{len(missing)} link(s) have render geometry but no collider",
    )


@check("sim_ready.mass_authored")
def mass_authored(context: registry.Context) -> registry.Result:
    """Every dynamic link has an authored mass or density.

    Static scenery carries no mass, so it is exempt; an asset with no dynamic
    links at all has authored no rigid bodies, which is itself the finding.
    """
    links = context.asset.links
    dynamic = _dynamic_links(context)
    if not dynamic:
        return verdict(
            False, {"dynamic_links": 0}, "asset has no dynamic rigid bodies"
        )
    missing = [
        name
        for name in dynamic
        if links[name].mass is None and links[name].density is None
    ]
    return verdict(
        not missing,
        {
            "dynamic_links": len(dynamic),
            "links_without_mass": len(missing),
            "examples": _names(missing),
        },
        f"{len(missing)} link(s) have neither mass nor density",
    )


@check("sim_ready.mass_plausible", requires="physics_schema")
def mass_plausible(context: registry.Context) -> registry.Result:
    """Mass divided by render-mesh volume falls in a physical density range."""
    limits = context.thresholds
    densities = {}
    for name, link in context.asset.links.items():
        volume = sum(
            abs(geometry.signed_volume(s.welded_vertices, s.welded_triangles))
            for s in context.surfaces_of(name)
        )
        if link.mass is None or volume <= 0.0:
            continue
        densities[name] = link.mass / volume

    if not densities:
        return registry.not_applicable({}, "no link has both mass and volume")

    outside = {
        name: value
        for name, value in densities.items()
        if not limits.density_min_kg_m3 <= value <= limits.density_max_kg_m3
    }
    values = np.fromiter(densities.values(), dtype=float)
    return verdict(
        not outside,
        {
            "links_measured": len(densities),
            "density_min_kg_m3": float(values.min()),
            "density_max_kg_m3": float(values.max()),
            "links_outside_range": len(outside),
            "examples": _names(list(outside)),
        },
        f"{len(outside)} link(s) imply an implausible density",
    )


@check("sim_ready.physics_material_authored")
def physics_material_authored(context: registry.Context) -> registry.Result:
    """Every collider has static and dynamic friction plus restitution."""
    proxies = context.asset.proxies
    if not proxies:
        return verdict(False, {"colliders": 0}, "asset has no colliders")

    missing = [p.name for p in proxies if not p.has_friction or p.restitution is None]
    return verdict(
        not missing,
        {
            "colliders": len(proxies),
            "colliders_without_material": len(missing),
            "examples": _names(missing),
        },
        f"{len(missing)} collider(s) lack friction or restitution",
    )


@check("sim_ready.joint_params_authored", requires="joints")
def joint_params_authored(context: registry.Context) -> registry.Result:
    """Joints that need limits have them, and any drive is fully parameterized.

    A revolute joint with no limits lets a wheel, caster, or swivel turn freely.
    Count unlimited revolute joints without failing them. A prismatic joint with
    no limits slides without end, unlike any real mechanism.
    """
    unbounded_travel, free_turning, partial_drives = [], [], []
    for joint in context.asset.moving_joints:
        if joint.lower is None or joint.upper is None:
            target = free_turning if joint.joint_type == "revolute" else unbounded_travel
            target.append(joint.name)
        if joint.drive is not None and any(v is None for v in joint.drive.values()):
            partial_drives.append(joint.name)

    return verdict(
        not unbounded_travel and not partial_drives,
        {
            "moving_joints": len(context.asset.moving_joints),
            "joints_without_limits": len(unbounded_travel),
            "free_turning_joints": len(free_turning),
            "joints_with_partial_drive": len(partial_drives),
            "examples": _names(unbounded_travel + partial_drives),
        },
        f"{len(unbounded_travel)} sliding joint(s) unlimited, "
        f"{len(partial_drives)} drive(s) incompletely parameterized",
    )
