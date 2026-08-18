"""Check contract, registration and execution."""

import dataclasses
import functools
import logging
from collections.abc import Callable

import trimesh

from sim_env_builder.asset_checks import config, geometry
from sim_env_builder.asset_checks.ingest import model

logger = logging.getLogger(__name__)

PASS = "pass"
FAIL = "fail"
NOT_APPLICABLE = "not_applicable"
# Measure and report these values without scoring them. No defensible threshold
# separates a defect, so the values inform rather than judge.
INFO = "info"

# Capabilities a check can require, with the reason reported when absent.
CAPABILITIES = {
    "polygons": (
        lambda asset: asset.has_polygons,
        "format does not preserve polygon topology",
    ),
    "joints": (lambda asset: asset.is_articulated, "asset has no articulation"),
    "skin": (lambda asset: asset.has_skin, "asset has no skinning"),
    "physics_schema": (
        lambda asset: asset.format == "usd",
        "format carries no physics schema",
    ),
}


@dataclasses.dataclass
class Result:
    """A check's verdict, before the runner attaches its identifier."""

    status: str
    metrics: dict
    message: str | None = None


@dataclasses.dataclass
class CheckResult:
    check_id: str
    status: str
    metrics: dict
    message: str | None = None


@dataclasses.dataclass(frozen=True)
class Check:
    check_id: str
    requires: str | None
    run: Callable[["Context"], Result]


# Report order. Registration order is preserved within each section, but the
# sections themselves are ordered here rather than by module import order, which
# an import sorter is free to rearrange.
SECTIONS = ("sim_ready", "mesh", "proxy", "uv", "materials", "articulation")

REGISTRY: list[Check] = []


def ordered() -> list[Check]:
    """Every registered check, in report order."""
    unknown = {c.check_id.split(".")[0] for c in REGISTRY} - set(SECTIONS)
    if unknown:
        raise ValueError(f"checks registered outside SECTIONS: {sorted(unknown)}")
    return sorted(REGISTRY, key=lambda c: SECTIONS.index(c.check_id.split(".")[0]))


def check(check_id: str, requires: str | None = None):
    """Register a check. `requires` names a capability the asset must have."""
    if requires is not None and requires not in CAPABILITIES:
        raise ValueError(f"unknown capability: {requires}")

    def decorate(function: Callable[["Context"], Result]) -> Callable:
        REGISTRY.append(Check(check_id, requires, function))
        return function

    return decorate


def verdict(ok: bool, metrics: dict, message: str | None = None) -> Result:
    """Return pass or fail, with a message only on failure."""
    return Result(PASS if ok else FAIL, metrics, None if ok else message)


def advisory(ok: bool, metrics: dict, message: str) -> Result:
    """Informational; the message flags a value outside the advisory range."""
    return Result(INFO, metrics, None if ok else message)


def info(metrics: dict) -> Result:
    return Result(INFO, metrics)


def not_applicable(metrics: dict, message: str) -> Result:
    return Result(NOT_APPLICABLE, metrics, message)


class Context:
    """One asset plus the derived data checks share."""

    def __init__(self, asset: model.AssetModel, thresholds: config.Thresholds):
        self.asset = asset
        self.thresholds = thresholds
        # Scratch space for derived data more than one check needs.
        self.memo: dict = {}

    def rel(self, fraction: float) -> float:
        """Scale a bounding-box-relative tolerance into model units."""
        return fraction * self.asset.bbox_diag

    @functools.cached_property
    def weld_tol(self) -> float:
        return self.rel(self.thresholds.duplicate_vertex_rel)

    @functools.cached_property
    def surfaces(self) -> list[geometry.Surface]:
        return [
            geometry.Surface(p.vertices, p.triangles, self.weld_tol)
            for p in self.asset.parts
        ]

    @functools.cached_property
    def proxy_surfaces(self) -> list[geometry.Surface]:
        return [
            geometry.Surface(p.vertices, p.triangles, self.weld_tol)
            for p in self.asset.proxies
        ]

    @functools.cached_property
    def combined_mesh(self):
        """Every render mesh in one trimesh, for rays that must see the whole asset."""
        return trimesh.util.concatenate(
            [s.mesh for s in self.surfaces] or [trimesh.Trimesh()]
        )

    def surfaces_of(self, link: str) -> list[geometry.Surface]:
        return [self.surfaces[i] for i in self.asset.links[link].parts]

    @functools.cached_property
    def link_children(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {}
        for joint in self.asset.joints:
            children.setdefault(joint.parent_link, []).append(joint.child_link)
        return children

    def subtree(self, root: str) -> list[str]:
        """`root` and every link below it in the joint graph."""
        collected, stack, seen = [], [root], set()
        while stack:
            name = stack.pop()
            if name in seen or name not in self.asset.links:
                continue
            seen.add(name)
            collected.append(name)
            stack += self.link_children.get(name, [])
        return collected


def run_all(context: Context) -> list[CheckResult]:
    """Run every registered check, in report order."""
    return [_run(entry, context) for entry in ordered()]


def _run(entry: Check, context: Context) -> CheckResult:
    if entry.requires:
        available, reason = CAPABILITIES[entry.requires]
        if not available(context.asset):
            return CheckResult(entry.check_id, NOT_APPLICABLE, {}, reason)
    logger.debug("running %s", entry.check_id)
    result = entry.run(context)
    return CheckResult(
        entry.check_id, result.status, result.metrics, result.message
    )
