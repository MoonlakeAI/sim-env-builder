"""Asset import inside Blender.

Supports glTF (`.glb`, `.gltf`) and USD (`.usd`, `.usda`, `.usdc`, `.usdz`).
"""
import os
import re

import bpy

# Source scenes sometimes include studio geometry, such as a large
# "aesthetic_ground" plane. These objects dominate the bounding box used for
# framing, explode radius, and floor height. Drop matching names during import.
_HELPER = re.compile(r"ground|floor|backdrop|cyclorama|aesthetic", re.IGNORECASE)

_USD_EXT = {".usd", ".usda", ".usdc", ".usdz"}


def is_helper(name: str) -> bool:
    return bool(_HELPER.search(name))


def import_asset(path: str) -> list:
    """Import a model and return the objects it added to the scene.

    Simulator-oriented USD assets can include collision proxies, guide prims,
    cameras, and lights. Exclude them because proxy hulls cover the render mesh,
    and imported lights interfere with the renderer's studio rig.
    """
    before = set(bpy.data.objects)
    if os.path.splitext(path)[1].lower() in _USD_EXT:
        bpy.ops.wm.usd_import(
            filepath=path,
            import_render=True,
            import_proxy=False,
            import_guide=False,
            import_cameras=False,
            import_lights=False,
            import_visible_only=True,
        )
    else:
        bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]
