"""Capability gating and the runner's report contract."""

import factories
from sim_env_builder.asset_checks.checks import registry
from sim_env_builder.asset_checks.checks.registry import FAIL, INFO, NOT_APPLICABLE, PASS

PHYSICS_ONLY = {
    "sim_ready.mass_plausible",
    "proxy.poly_budget",
    "proxy.watertight",
    "proxy.surface_distance",
}
ABSENCE_IS_FAILURE = {
    "sim_ready.colliders_present",
    "sim_ready.mass_authored",
    "sim_ready.physics_material_authored",
    "proxy.present",
}


def _statuses(asset):
    return {r.check_id: r.status for r in registry.run_all(factories.context(asset))}


def test_gltf_skips_physics_schema_checks_but_fails_on_absent_colliders():
    asset = factories.asset(
        [factories.part(uvs=factories.unwrapped_cube_uvs())],
        asset_format="gltf",
        mass=None,
    )
    statuses = _statuses(asset)
    for check_id in PHYSICS_ONLY:
        assert statuses[check_id] == NOT_APPLICABLE, check_id
    for check_id in ABSENCE_IS_FAILURE:
        assert statuses[check_id] == FAIL, check_id


def test_gltf_skips_polygon_topology_checks():
    asset = factories.asset([factories.part(polygons=False)], asset_format="gltf")
    statuses = _statuses(asset)
    for check_id in ("mesh.quad_ratio", "mesh.valence", "mesh.ngons"):
        assert statuses[check_id] == NOT_APPLICABLE, check_id


def test_unarticulated_asset_reports_articulated_false_and_skips_the_rest():
    statuses = _statuses(factories.asset([factories.part()]))
    assert statuses["articulation.articulated"] == INFO
    for check_id, status in statuses.items():
        if (
            check_id.startswith("articulation.")
            and check_id != "articulation.articulated"
        ):
            assert status == NOT_APPLICABLE, check_id


def test_every_check_returns_a_known_status():
    asset = factories.asset(
        [factories.part(uvs=factories.unwrapped_cube_uvs())], [factories.proxy()]
    )
    results = registry.run_all(factories.context(asset))
    assert len(results) == len(registry.REGISTRY)
    assert all(r.status in (PASS, FAIL, NOT_APPLICABLE, INFO) for r in results)
    assert all(r.metrics is not None for r in results)


def test_check_ids_are_unique_and_namespaced():
    ids = [entry.check_id for entry in registry.REGISTRY]
    assert len(ids) == len(set(ids))
    assert all("." in check_id for check_id in ids)
