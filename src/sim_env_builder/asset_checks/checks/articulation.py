"""Articulation structure: joint graph, limits, transforms and skin weights."""

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from sim_env_builder.asset_checks import geometry
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.checks.registry import Result, check, verdict
from sim_env_builder.asset_checks.ingest import model


@check("articulation.articulated")
def articulated(context: registry.Context) -> Result:
    """Whether the asset declares any moving joint. Reported, never failed."""
    joints = context.asset.joints
    return registry.info(
        {
            "articulated": context.asset.is_articulated,
            "joints": len(joints),
            "moving_joints": len(context.asset.moving_joints),
            "links": len(context.asset.links),
        }
    )


@check("articulation.acyclic", requires="joints")
def acyclic(context: registry.Context) -> Result:
    """The link graph is a tree rather than a loop."""
    index, matrix = _link_graph(context)
    cycles = matrix.shape[0] - scipy.sparse.csgraph.connected_components(matrix)[0]
    edges = int(matrix.nnz)
    return verdict(
        edges <= cycles,
        {"links": len(index), "joint_edges": edges, "independent_loops": edges - cycles},
        f"{edges - cycles} loop(s) in the link graph",
    )


def _link_graph(context: registry.Context):
    index = {name: i for i, name in enumerate(context.asset.links)}
    rows, cols = [], []
    for joint in context.asset.joints:
        if joint.parent_link in index and joint.child_link in index:
            rows.append(index[joint.parent_link])
            cols.append(index[joint.child_link])
    matrix = scipy.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(index), len(index))
    )
    return index, matrix


@check("articulation.single_parent", requires="joints")
def single_parent(context: registry.Context) -> Result:
    """Each link is driven by at most one joint, and one link is the root.

    A joint whose parent is the world anchors its child rather than demoting
    it, so a world-anchored link still counts as the root of its tree.
    """
    parents: dict[str, int] = {}
    anchored: set[str] = set()
    for joint in context.asset.joints:
        parents[joint.child_link] = parents.get(joint.child_link, 0) + 1
        if joint.parent_link:
            anchored.add(joint.child_link)

    shared = [name for name, count in parents.items() if count > 1]
    roots = [
        name
        for name in context.asset.links
        if name not in anchored and name != model.STATIC_LINK
    ]
    return verdict(
        not shared and len(roots) == 1,
        {
            "links_with_multiple_parents": len(shared),
            "roots": len(roots),
            "examples": sorted(shared)[:10],
        },
        f"{len(shared)} link(s) have several parents, {len(roots)} root(s) found",
    )


@check("articulation.limits_ordered", requires="joints")
def limits_ordered(context: registry.Context) -> Result:
    """Lower limits sit below upper limits."""
    reversed_limits = [
        joint.name
        for joint in context.asset.moving_joints
        if joint.lower is not None and joint.upper is not None and joint.lower >= joint.upper
    ]
    return verdict(
        not reversed_limits,
        {
            "reversed_limits": len(reversed_limits),
            "examples": sorted(reversed_limits)[:10],
        },
        f"{len(reversed_limits)} joint(s) have a lower limit at or above the upper",
    )


@check("articulation.rest_in_limits", requires="joints")
def rest_in_limits(context: registry.Context) -> Result:
    """The authored rest pose lies inside every joint's travel."""
    outside = []
    for joint in context.asset.moving_joints:
        if joint.rest_value is None or joint.lower is None or joint.upper is None:
            continue
        if not joint.lower - 1e-6 <= joint.rest_value <= joint.upper + 1e-6:
            outside.append(joint.name)
    return verdict(
        not outside,
        {"joints_outside_limits": len(outside), "examples": sorted(outside)[:10]},
        f"{len(outside)} joint(s) rest outside their own limits",
    )


@check("articulation.joints_control_geometry", requires="joints")
def joints_control_geometry(context: registry.Context) -> Result:
    """Every joint moves a link that actually has render geometry."""
    empty = [
        joint.name
        for joint in context.asset.moving_joints
        if not _subtree_parts(context, joint.child_link)
    ]
    return verdict(
        not empty,
        {"joints_without_geometry": len(empty), "examples": sorted(empty)[:10]},
        f"{len(empty)} joint(s) drive no geometry",
    )


def _subtree_parts(context: registry.Context, root: str) -> list[int]:
    """Part indices belonging to a link and everything below it."""
    return [i for name in context.subtree(root) for i in context.asset.links[name].parts]


@check("articulation.links_connected", requires="joints")
def links_connected(context: registry.Context) -> Result:
    """No link with geometry is left out of the joint graph."""
    attached = {j.parent_link for j in context.asset.joints}
    attached |= {j.child_link for j in context.asset.joints}
    orphans = [
        name
        for name, link in context.asset.links.items()
        if link.parts and name not in attached and name != model.STATIC_LINK
    ]
    return verdict(
        not orphans,
        {"orphan_links": len(orphans), "examples": sorted(orphans)[:10]},
        f"{len(orphans)} link(s) with geometry take part in no joint",
    )


@check("articulation.transforms_invertible", requires="joints")
def transforms_invertible(context: registry.Context) -> Result:
    """Link transforms and joint frames are usable.

    A link transform that collapses a dimension has no inverse. A zero-length or
    non-finite joint axis cannot define a rotation. Neither can be posed.
    """
    limit = context.thresholds.transform_det_min
    smallest, singular = np.inf, []
    for name, link in context.asset.links.items():
        determinant = abs(float(np.linalg.det(link.transform[:3, :3])))
        smallest = min(smallest, determinant)
        if determinant <= limit:
            singular.append(name)

    unusable = [
        joint.name
        for joint in context.asset.moving_joints
        if not _usable_frame(joint)
    ]
    return verdict(
        not singular and not unusable,
        {
            "min_determinant": float(smallest) if np.isfinite(smallest) else None,
            "singular_links": len(singular),
            "unusable_joint_frames": len(unusable),
            "examples": sorted(singular + unusable)[:10],
        },
        f"{len(singular)} link transform(s) are not invertible, "
        f"{len(unusable)} joint frame(s) cannot be posed",
    )


def _usable_frame(joint) -> bool:
    if joint.axis is None or joint.anchor is None:
        return False
    finite = np.isfinite(joint.axis).all() and np.isfinite(joint.anchor).all()
    return bool(finite and np.linalg.norm(joint.axis) > 0.0)


@check("articulation.no_bridging_geometry", requires="joints")
def no_bridging_geometry(context: registry.Context) -> Result:
    """Each link's surface closes on its own rather than through a neighbour.

    A link that closes only after welding to its neighbour shares a continuous
    surface across the joint. That surface tears when the joint moves. Flush
    links remain closed individually and are not bridged.
    """
    bridged = []
    for name, link in context.asset.links.items():
        if not link.parts or not _open_edges(context, [name]):
            continue
        neighbours = _neighbours(context, name)
        if neighbours and not _open_edges(context, [name] + neighbours):
            bridged.append(name)

    return verdict(
        not bridged,
        {"bridged_links": len(bridged), "examples": sorted(bridged)[:10]},
        f"{len(bridged)} link(s) rely on a neighbour to close their surface",
    )


def _open_edges(context: registry.Context, links: list[str]) -> int:
    """Boundary edges of the given links' geometry, welded together."""
    parts = [i for name in links for i in context.asset.links[name].parts]
    if not parts:
        return 0
    vertices, triangles, offset = [], [], 0
    for index in parts:
        part = context.asset.parts[index]
        vertices.append(part.vertices)
        triangles.append(part.triangles + offset)
        offset += len(part.vertices)
    merged = geometry.Surface(
        np.concatenate(vertices), np.concatenate(triangles), context.weld_tol
    )
    return len(merged.boundary_edges)


def _neighbours(context: registry.Context, name: str) -> list[str]:
    joined = []
    for joint in context.asset.joints:
        if joint.child_link == name and joint.parent_link:
            joined.append(joint.parent_link)
        elif joint.parent_link == name:
            joined.append(joint.child_link)
    return [n for n in joined if n in context.asset.links]


@check("articulation.skin_weights", requires="skin")
def skin_weights(context: registry.Context) -> Result:
    """Skin weights are non-negative, normalized, and locally supported."""
    tolerance = context.thresholds.skin_weight_sum_tol
    negative = unnormalized = unweighted = 0
    spreads = []
    for part, surface in zip(context.asset.parts, context.surfaces):
        if part.skin is None:
            continue
        weights = part.skin.weights
        negative += int((weights < 0).sum())
        totals = weights.sum(axis=1)
        unnormalized += int((np.abs(totals - 1.0) > tolerance).sum())
        unweighted += int((totals <= 0).sum())
        spreads.append(_influence_spread(part))

    spread = float(np.mean(spreads)) if spreads else 0.0
    return verdict(
        negative == 0 and unnormalized == 0 and unweighted == 0,
        {
            "negative_weights": negative,
            "unnormalized_vertices": unnormalized,
            "unweighted_vertices": unweighted,
            "influence_spread": spread,
        },
        f"{negative} negative, {unnormalized} unnormalized, "
        f"{unweighted} unweighted vertex/vertices",
    )


def _influence_spread(part) -> float:
    """Mean extent of each joint's influence, relative to the part's size.

    Weight columns represent influence slots, not joints. The same joint may
    appear in different slots on different vertices, so this function groups
    vertices by the joint index referenced by each weight.
    """
    skin = part.skin
    extent = float(np.linalg.norm(np.ptp(part.vertices, axis=0))) or 1.0
    active = skin.weights > 0.05
    spreads = []
    for joint in np.unique(skin.indices[active]):
        rows = ((skin.indices == joint) & active).any(axis=1)
        influenced = part.vertices[rows]
        if len(influenced) > 1:
            spreads.append(float(np.linalg.norm(np.ptp(influenced, axis=0))) / extent)
    return float(np.mean(spreads)) if spreads else 0.0
