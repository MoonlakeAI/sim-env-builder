"""Asset loading, dispatched by file extension."""

import pathlib

from sim_env_builder.asset_checks.ingest import gltf, model, usd

USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
GLTF_SUFFIXES = {".glb", ".gltf"}


def load(path: str) -> model.AssetModel:
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in USD_SUFFIXES:
        return usd.load(path)
    if suffix in GLTF_SUFFIXES:
        return gltf.load(path)
    raise ValueError(
        f"unsupported format '{suffix}'; expected one of "
        f"{sorted(USD_SUFFIXES | GLTF_SUFFIXES)}"
    )
