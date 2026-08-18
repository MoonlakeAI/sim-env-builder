"""Every threshold and sampling parameter used by the check suite."""

import dataclasses
import json
import math
import pathlib


@dataclasses.dataclass(frozen=True)
class Thresholds:
    # sim_ready
    density_min_kg_m3: float = 10.0
    density_max_kg_m3: float = 30000.0

    # mesh
    quad_ratio_warn: float = 0.5
    min_triangle_angle_deg: float = 0.5
    min_triangle_aspect: float = 0.01
    zero_area_rel: float = 1e-10
    duplicate_vertex_rel: float = 1e-6
    rest_penetration_rel: float = 0.001
    floater_gap_rel: float = 0.02
    # A cube corner has an angle defect of pi/2 and must pass; a needle tip
    # approaches 2*pi. 1.7*pi admits every plausible hard-surface corner.
    spike_defect_rad: float = 1.7 * math.pi
    poly_budget: int = 500_000

    # proxy
    proxy_triangle_budget: int = 5_000
    proxy_render_ratio: float = 0.25
    proxy_surface_p95_rel: float = 0.01
    proxy_penetration_rel: float = 0.02

    # uv
    uv_conformal_p95: float = 2.0
    uv_shells_per_part: int = 1200
    uv_texel_density_cv: float = 0.25
    uv_utilization_warn: float = 0.3
    uv_sharp_edge_deg: float = 30.0

    # materials
    detail_map_fraction: float = 0.5
    texels_per_cm_min: float = 1.0
    texels_per_cm_max: float = 200.0
    texture_max_side: int = 4096
    baked_lighting_correlation: float = 0.5
    # Fraction of mean luminance explained by a linear ramp across a UV shell.
    baked_lighting_gradient: float = 0.3
    seam_delta_e_p95: float = 10.0
    seam_normal_deg_p95: float = 25.0

    # articulation
    transform_det_min: float = 1e-9
    skin_weight_sum_tol: float = 1e-3

    # sampling
    raster_resolution: int = 1024
    surface_samples: int = 20_000
    penetration_samples: int = 5_000
    occlusion_directions: int = 64
    shading_samples: int = 4_000
    # Broad-phase self-intersection is linear in triangles but with a large
    # constant; past this size the check reports not_applicable instead.
    self_intersection_max_triangles: int = 1_000_000


def load(path: pathlib.Path | None) -> Thresholds:
    """Return defaults, overriding keys present in a JSON file."""
    if path is None:
        return Thresholds()
    overrides = json.loads(path.read_text())
    known = {f.name for f in dataclasses.fields(Thresholds)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown threshold keys: {sorted(unknown)}")
    return dataclasses.replace(Thresholds(), **overrides)
