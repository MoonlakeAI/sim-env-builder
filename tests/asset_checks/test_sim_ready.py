"""Physics authoring checks, including the absence-is-failure behaviour on glTF."""

import factories
from sim_env_builder.asset_checks.checks import sim_ready
from sim_env_builder.asset_checks.checks.registry import FAIL, NOT_APPLICABLE, PASS


def _authored():
    return factories.asset([factories.part()], [factories.proxy()])


def test_colliders_present_passes_when_every_link_has_one():
    assert sim_ready.colliders_present(factories.context(_authored())).status == PASS


def test_colliders_present_fails_without_colliders():
    result = sim_ready.colliders_present(
        factories.context(factories.asset([factories.part()]))
    )
    assert result.status == FAIL and result.metrics["links_without_collider"] == 1


def test_mass_authored_passes_and_fails():
    assert sim_ready.mass_authored(factories.context(_authored())).status == PASS

    without = factories.asset([factories.part()], [factories.proxy()], mass=None)
    result = sim_ready.mass_authored(factories.context(without))
    assert result.status == FAIL and result.metrics["links_without_mass"] == 1


def test_mass_plausible_accepts_a_realistic_density():
    """A one-metre cube at 1 kg implies 1 kg/m3, below the floor."""
    light = factories.asset([factories.part()], [factories.proxy()], mass=1.0)
    assert sim_ready.mass_plausible(factories.context(light)).status == FAIL

    plausible = factories.asset([factories.part()], [factories.proxy()], mass=800.0)
    result = sim_ready.mass_plausible(factories.context(plausible))
    assert result.status == PASS
    assert result.metrics["density_max_kg_m3"] == 800.0


def test_mass_plausible_not_applicable_without_any_mass():
    massless = factories.asset([factories.part()], [factories.proxy()], mass=None)
    result = sim_ready.mass_plausible(factories.context(massless))
    assert result.status == NOT_APPLICABLE


def test_physics_material_authored():
    assert (
        sim_ready.physics_material_authored(factories.context(_authored())).status
        == PASS
    )

    bare = factories.asset(
        [factories.part()], [factories.proxy(friction=None, restitution=None)]
    )
    result = sim_ready.physics_material_authored(factories.context(bare))
    assert result.status == FAIL and result.metrics["colliders_without_material"] == 1


def test_physics_material_fails_when_there_are_no_colliders():
    result = sim_ready.physics_material_authored(
        factories.context(factories.asset([factories.part()]))
    )
    assert result.status == FAIL and result.metrics["colliders"] == 0


def test_joint_params_authored_requires_limits_and_complete_drives():
    complete = factories.asset(
        [factories.part()],
        joints=[factories.joint(drive={"stiffness": 1.0, "damping": 0.1, "maxForce": 5.0})],
    )
    assert sim_ready.joint_params_authored(factories.context(complete)).status == PASS

    sliding = factories.asset(
        [factories.part()],
        joints=[factories.joint(joint_type="prismatic", lower=None, upper=None)],
    )
    result = sim_ready.joint_params_authored(factories.context(sliding))
    assert result.status == FAIL and result.metrics["joints_without_limits"] == 1

    partial = factories.asset(
        [factories.part()],
        joints=[factories.joint(drive={"stiffness": 1.0, "damping": None, "maxForce": 5.0})],
    )
    result = sim_ready.joint_params_authored(factories.context(partial))
    assert result.status == FAIL and result.metrics["joints_with_partial_drive"] == 1


def test_free_turning_revolute_joints_are_counted_but_not_failed():
    """Authors leave a caster or wheel unlimited on purpose."""
    swivel = factories.asset(
        [factories.part()],
        joints=[factories.joint(joint_type="revolute", lower=None, upper=None)],
    )
    result = sim_ready.joint_params_authored(factories.context(swivel))
    assert result.status == PASS
    assert result.metrics["free_turning_joints"] == 1
    assert result.metrics["joints_without_limits"] == 0


def test_mass_authored_exempts_static_scenery():
    from sim_env_builder.asset_checks.ingest import model

    asset = factories.asset(
        [
            factories.part(name="body", link="body"),
            factories.part(name="decal", link=model.STATIC_LINK, offset=(3, 0, 0)),
        ]
    )
    asset.links[model.STATIC_LINK].mass = None
    assert sim_ready.mass_authored(factories.context(asset)).status == PASS


def test_mass_authored_fails_when_nothing_is_a_rigid_body():
    from sim_env_builder.asset_checks.ingest import model

    asset = factories.asset([factories.part(link=model.STATIC_LINK)], mass=None)
    result = sim_ready.mass_authored(factories.context(asset))
    assert result.status == FAIL and result.metrics["dynamic_links"] == 0
