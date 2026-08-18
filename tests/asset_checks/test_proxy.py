"""Collision-proxy checks."""

import numpy as np

import factories
from sim_env_builder.asset_checks.checks import proxy
from sim_env_builder.asset_checks.checks.registry import FAIL, NOT_APPLICABLE, PASS


def test_present_passes_with_a_dedicated_proxy():
    asset = factories.asset([factories.part()], [factories.proxy()])
    assert proxy.present(factories.context(asset)).status == PASS


def test_present_fails_when_collision_reuses_the_render_mesh():
    asset = factories.asset(
        [factories.part()], [factories.proxy(source="render_mesh")]
    )
    result = proxy.present(factories.context(asset))
    assert result.status == FAIL
    assert result.metrics["reused_render_meshes"] == 1


def test_present_fails_on_gltf_without_colliders():
    asset = factories.asset([factories.part()], asset_format="gltf")
    assert proxy.present(factories.context(asset)).status == FAIL


def test_poly_budget_passes_when_the_proxy_is_much_lighter():
    asset = factories.asset([factories.sphere_part(subdivisions=3)], [factories.proxy()])
    assert proxy.poly_budget(factories.context(asset)).status == PASS


def test_poly_budget_fails_when_the_proxy_is_as_heavy_as_the_render_mesh():
    asset = factories.asset([factories.part()], [factories.proxy()])
    result = proxy.poly_budget(factories.context(asset))
    assert result.status == FAIL and result.metrics["proxy_render_ratio"] == 1.0


def test_poly_budget_fails_over_the_triangle_cap():
    asset = factories.asset([factories.sphere_part(subdivisions=3)], [factories.proxy()])
    result = proxy.poly_budget(factories.context(asset, proxy_triangle_budget=4))
    assert result.status == FAIL and result.metrics["proxies_over_budget"] == 1


def test_watertight_passes_closed_and_fails_open():
    asset = factories.asset([factories.part()], [factories.proxy()])
    assert proxy.watertight(factories.context(asset)).status == PASS

    points, triangles, _, _ = factories.cube_arrays()
    leaky = factories.asset(
        [factories.part()],
        [factories.proxy(vertices=points, triangles=triangles[:-1])],
    )
    result = proxy.watertight(factories.context(leaky))
    assert result.status == FAIL and result.metrics["open_proxies"] == 1


def test_surface_distance_passes_on_a_matching_proxy():
    asset = factories.asset([factories.part()], [factories.proxy()])
    result = proxy.surface_distance(factories.context(asset))
    assert result.status == PASS and result.metrics["max_protrusion"] <= 0.0


def test_surface_distance_fails_when_the_proxy_protrudes():
    swollen = factories.proxy(scale=2.0, offset=(-0.5, -0.5, -0.5))
    asset = factories.asset([factories.part()], [swollen])
    result = proxy.surface_distance(factories.context(asset))
    assert result.status == FAIL and result.metrics["max_protrusion"] > 0.0


def test_surface_distance_not_applicable_without_both_surfaces():
    asset = factories.asset([factories.part()])
    result = proxy.surface_distance(factories.context(asset))
    assert result.status == NOT_APPLICABLE


def test_signed_gap_samples_face_interiors():
    """Three proxy vertices on the render surface must not hide a face whose
    interior sits well away from it."""
    import trimesh

    from sim_env_builder.asset_checks.checks.proxy import _signed_gap

    render = trimesh.creation.box(extents=(1, 1, 1))
    # A triangle whose corners touch three faces of the box; its interior
    # points, centroid included, sit off the surface.
    bridge = trimesh.Trimesh(
        vertices=[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
        faces=[[0, 1, 2]],
        process=False,
    )
    distances = np.abs(_signed_gap(render, bridge, 100))
    assert distances.max() > 0.05
    assert np.sort(distances)[:3].max() < 1e-9
