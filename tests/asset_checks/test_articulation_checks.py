"""Articulation-structure checks."""

import numpy as np
import pytest

import factories
from sim_env_builder.asset_checks.checks import articulation
from sim_env_builder.asset_checks.checks.registry import FAIL, INFO, PASS
from sim_env_builder.asset_checks.ingest import model


def hinged():
    """Two links joined by one revolute joint, geometry on both."""
    return factories.asset(
        [
            factories.part(name="base", link="base"),
            factories.part(name="door", link="door", offset=(1.0, 0, 0)),
        ],
        joints=[factories.joint(parent_link="base", child_link="door")],
    )


def test_articulated_reports_the_joint_count():
    result = articulation.articulated(factories.context(hinged()))
    assert result.status == INFO
    assert result.metrics["articulated"] is True
    assert result.metrics["moving_joints"] == 1


def test_articulated_reports_false_for_static_geometry():
    result = articulation.articulated(
        factories.context(factories.asset([factories.part()]))
    )
    assert result.metrics["articulated"] is False


def test_acyclic_passes_on_a_chain_and_fails_on_a_loop():
    assert articulation.acyclic(factories.context(hinged())).status == PASS

    looped = factories.asset(
        [
            factories.part(name="a", link="a"),
            factories.part(name="b", link="b", offset=(1, 0, 0)),
        ],
        joints=[
            factories.joint(name="ab", parent_link="a", child_link="b"),
            factories.joint(name="ba", parent_link="b", child_link="a"),
        ],
    )
    result = articulation.acyclic(factories.context(looped))
    assert result.status == FAIL and result.metrics["independent_loops"] >= 1


def test_single_parent_fails_when_two_joints_drive_one_link():
    assert articulation.single_parent(factories.context(hinged())).status == PASS

    shared = factories.asset(
        [
            factories.part(name="a", link="a"),
            factories.part(name="b", link="b", offset=(1, 0, 0)),
            factories.part(name="c", link="c", offset=(2, 0, 0)),
        ],
        joints=[
            factories.joint(name="ac", parent_link="a", child_link="c"),
            factories.joint(name="bc", parent_link="b", child_link="c"),
        ],
    )
    result = articulation.single_parent(factories.context(shared))
    assert result.status == FAIL and result.metrics["links_with_multiple_parents"] == 1


def test_limits_ordered_fails_when_reversed():
    assert articulation.limits_ordered(factories.context(hinged())).status == PASS

    backwards = factories.asset(
        [factories.part(link="base"), factories.part(name="d", link="door")],
        joints=[factories.joint(parent_link="base", child_link="door", lower=1.0, upper=-1.0)],
    )
    result = articulation.limits_ordered(factories.context(backwards))
    assert result.status == FAIL and result.metrics["reversed_limits"] == 1


def test_rest_in_limits_fails_when_the_rest_pose_is_outside():
    assert articulation.rest_in_limits(factories.context(hinged())).status == PASS

    parked = factories.asset(
        [factories.part(link="base"), factories.part(name="d", link="door")],
        joints=[
            factories.joint(
                parent_link="base", child_link="door", lower=0.0, upper=1.0, rest_value=2.0
            )
        ],
    )
    result = articulation.rest_in_limits(factories.context(parked))
    assert result.status == FAIL and result.metrics["joints_outside_limits"] == 1


def test_joints_control_geometry_fails_on_an_empty_child():
    assert (
        articulation.joints_control_geometry(factories.context(hinged())).status == PASS
    )

    hollow = factories.asset(
        [factories.part(name="base", link="base")],
        joints=[factories.joint(parent_link="base", child_link="empty")],
    )
    result = articulation.joints_control_geometry(factories.context(hollow))
    assert result.status == FAIL and result.metrics["joints_without_geometry"] == 1


def test_links_connected_fails_on_an_orphan_link():
    assert articulation.links_connected(factories.context(hinged())).status == PASS

    orphaned = factories.asset(
        [
            factories.part(name="base", link="base"),
            factories.part(name="door", link="door", offset=(1, 0, 0)),
            factories.part(name="loose", link="loose", offset=(3, 0, 0)),
        ],
        joints=[factories.joint(parent_link="base", child_link="door")],
    )
    result = articulation.links_connected(factories.context(orphaned))
    assert result.status == FAIL and result.metrics["orphan_links"] == 1


def test_transforms_invertible_fails_on_a_flattened_link():
    assert (
        articulation.transforms_invertible(factories.context(hinged())).status == PASS
    )

    collapsed = np.eye(4)
    collapsed[2, 2] = 0.0
    asset = hinged()
    asset.links["door"].transform = collapsed
    result = articulation.transforms_invertible(factories.context(asset))
    assert result.status == FAIL and result.metrics["singular_links"] == 1


def test_no_bridging_geometry_allows_links_sitting_flush():
    """Two closed shells touching face to face are not bridged."""
    assert (
        articulation.no_bridging_geometry(factories.context(hinged())).status == PASS
    )


def test_no_bridging_geometry_fails_on_one_surface_split_across_links():
    points, triangles, _, _ = factories.cube_arrays()
    split = factories.asset(
        [
            factories.part(
                name="shell", link="base", vertices=points, triangles=triangles[:-2]
            ),
            factories.part(
                name="lid", link="door", vertices=points, triangles=triangles[-2:]
            ),
        ],
        joints=[factories.joint(parent_link="base", child_link="door")],
    )
    result = articulation.no_bridging_geometry(factories.context(split))
    assert result.status == FAIL and result.metrics["bridged_links"] >= 1


def skinned_part(weights):
    part = factories.part()
    part.skin = model.Skin(
        joints=["a", "b"],
        indices=np.zeros((len(part.vertices), 2), dtype=int),
        weights=np.asarray(weights, dtype=float),
    )
    return part


def test_skin_weights_pass_when_normalized():
    weights = np.tile([0.6, 0.4], (8, 1))
    result = articulation.skin_weights(
        factories.context(factories.asset([skinned_part(weights)]))
    )
    assert result.status == PASS


def test_skin_weights_fail_on_negative_unnormalized_and_unweighted_vertices():
    weights = np.tile([0.6, 0.4], (8, 1))
    weights[0] = [-0.1, 1.1]
    weights[1] = [0.2, 0.2]
    weights[2] = [0.0, 0.0]
    result = articulation.skin_weights(
        factories.context(factories.asset([skinned_part(weights)]))
    )
    assert result.status == FAIL
    assert result.metrics["negative_weights"] == 1
    assert result.metrics["unnormalized_vertices"] == 2
    assert result.metrics["unweighted_vertices"] == 1


def test_transforms_invertible_fails_on_a_degenerate_joint_axis():
    asset = hinged()
    asset.joints[0].axis = np.array([np.nan, 0.0, 0.0])
    result = articulation.transforms_invertible(factories.context(asset))
    assert result.status == FAIL and result.metrics["unusable_joint_frames"] == 1


def test_transforms_invertible_fails_on_a_zero_length_axis():
    asset = hinged()
    asset.joints[0].axis = np.zeros(3)
    result = articulation.transforms_invertible(factories.context(asset))
    assert result.status == FAIL and result.metrics["unusable_joint_frames"] == 1


def test_single_parent_treats_a_world_anchored_link_as_the_root():
    anchored = factories.asset(
        [factories.part(name="lid", link="lid")],
        joints=[factories.joint(parent_link=None, child_link="lid")],
    )
    result = articulation.single_parent(factories.context(anchored))
    assert result.status == PASS and result.metrics["roots"] == 1


def test_static_scenery_is_neither_an_orphan_nor_a_second_root():
    from sim_env_builder.asset_checks.ingest import model

    asset = factories.asset(
        [
            factories.part(name="base", link="base"),
            factories.part(name="door", link="door", offset=(1, 0, 0)),
            factories.part(name="decal", link=model.STATIC_LINK, offset=(3, 0, 0)),
        ],
        joints=[factories.joint(parent_link="base", child_link="door")],
    )
    context = factories.context(asset)
    assert articulation.links_connected(context).status == PASS
    result = articulation.single_parent(context)
    assert result.status == PASS and result.metrics["roots"] == 1


def test_influence_spread_groups_by_joint_not_by_slot():
    """The same joint spread across two influence slots is one influence."""
    part = factories.part()
    # Joint 0 drives all eight vertices, but through slot 0 for the first
    # half and slot 1 for the second. Slot-wise grouping would see two small
    # groups; joint-wise grouping sees one full-extent influence.
    indices = np.zeros((8, 2), dtype=int)
    indices[4:, 0] = 1
    indices[4:, 1] = 0
    weights = np.zeros((8, 2))
    weights[:4, 0] = 1.0
    weights[4:, 1] = 1.0
    part.skin = model.Skin(joints=["a", "b"], indices=indices, weights=weights)

    spread = articulation._influence_spread(part)
    # Joint 0 spans the whole cube (spread 1); joint 1 has no active weights.
    assert spread == pytest.approx(1.0)
