"""Report assembly and JSON serialization."""

import collections
import json
import pathlib

import numpy as np

from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.ingest import model

SCHEMA_VERSION = 1
STATUSES = (registry.PASS, registry.FAIL, registry.NOT_APPLICABLE, registry.INFO)


def build(
    asset: model.AssetModel,
    results: list[registry.CheckResult],
    render: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "asset": _describe(asset),
        "checks": [_entry(result) for result in results],
        "sections": _sections(results),
        "render": render,
    }


def _describe(asset: model.AssetModel) -> dict:
    return {
        "path": asset.path,
        "format": asset.format,
        "parts": len(asset.parts),
        "links": len(asset.links),
        "triangles": asset.triangle_count,
        "materials": len(asset.materials),
        "proxies": len(asset.proxies),
        "joints": len(asset.joints),
        "articulated": asset.is_articulated,
        "bbox_diag_m": asset.bbox_diag,
    }


def _entry(result: registry.CheckResult) -> dict:
    entry = {
        "id": result.check_id,
        "status": result.status,
        "metrics": result.metrics,
    }
    if result.message:
        entry["message"] = result.message
    return entry


def _sections(results: list[registry.CheckResult]) -> dict:
    sections: dict[str, dict] = {}
    for result in results:
        name = result.check_id.split(".", 1)[0]
        counts = sections.setdefault(name, dict.fromkeys(STATUSES, 0))
        counts[result.status] += 1
    return sections


def write(path: pathlib.Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, default=_encode) + "\n")


def _encode(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, collections.abc.Set):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def summarize(report: dict) -> str:
    """One line per section, for terminal output."""
    lines = []
    for name, counts in report["sections"].items():
        lines.append(
            f"{name:14s} {counts[registry.PASS]:3d} pass  "
            f"{counts[registry.FAIL]:3d} fail  "
            f"{counts[registry.NOT_APPLICABLE]:3d} n/a  "
            f"{counts[registry.INFO]:3d} info"
        )
    return "\n".join(lines)
