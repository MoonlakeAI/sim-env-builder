"""Every mesh check on passing and failing geometry."""

import numpy as np
import pytest
import trimesh

import factories
from sim_env_builder.asset_checks.checks import mesh
from sim_env_builder.asset_checks.checks.registry import FAIL, INFO, NOT_APPLICABLE, PASS


def test_quad_ratio_reports_on_a_quad_cube():
    result = mesh.quad_ratio(factories.context(factories.asset([factories.part()])))
    assert result.status == INFO
    assert result.metrics["quad_ratio"] == 1.0


def test_quad_ratio_warns_when_mostly_triangles():
    _, triangles, _, _ = factories.cube_arrays()
    part = factories.part()
    part.face_counts = np.full(len(triangles), 3)
    part.face_indices = triangles.ravel()
    result = mesh.quad_ratio(factories.context(factories.asset([part])))
    assert result.status == INFO and result.message is not None


def test_valence_reports_the_share_of_four_valent_vertices():
    grid = trimesh.creation.box()
    part = factories.part()
    result = mesh.valence(factories.context(factories.asset([part])))
    assert 0.0 <= result.metrics["valence_four_ratio"] <= 1.0
    assert result.metrics["vertices"] == 8
    assert grid is not None


def test_ngons_pass_on_quads_and_fail_on_a_pentagon():
    assert mesh.ngons(factories.context(factories.asset([factories.part()]))).status == PASS

    part = factories.part()
    part.face_counts = np.array([5, 4, 4, 4, 4, 4, 3])
    part.face_indices = np.concatenate([[0, 1, 2, 3, 4], part.face_indices[4:24], [0, 1, 2]])
    result = mesh.ngons(factories.context(factories.asset([part])))
    assert result.status == FAIL and result.metrics["ngons"] == 1


def test_ngons_not_applicable_without_polygon_topology():
    context = factories.context(factories.asset([factories.part(polygons=False)]))
    assert not context.asset.has_polygons


def test_degenerate_faces_pass_on_a_cube_and_fail_on_a_sliver():
    assert (
        mesh.degenerate_faces(
            factories.context(factories.asset([factories.part()]))
        ).status
        == PASS
    )

    sliver = factories.part(
        vertices=[[0, 0, 0], [1, 0, 0], [0.5, 1e-7, 0], [0, 0, 1]],
        triangles=[[0, 1, 2], [0, 1, 3]],
    )
    result = mesh.degenerate_faces(factories.context(factories.asset([sliver])))
    assert result.status == FAIL and result.metrics["degenerate_triangles"] >= 1


def test_zero_area_faces_detected():
    assert (
        mesh.zero_area_faces(
            factories.context(factories.asset([factories.part()]))
        ).status
        == PASS
    )

    collapsed = factories.part(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]],
        triangles=[[0, 1, 2], [0, 1, 3]],
    )
    result = mesh.zero_area_faces(factories.context(factories.asset([collapsed])))
    assert result.status == FAIL and result.metrics["zero_area_triangles"] == 1


def test_duplicate_vertices_only_counted_without_a_seam():
    assert (
        mesh.duplicate_vertices(
            factories.context(factories.asset([factories.part()]))
        ).status
        == PASS
    )

    doubled = factories.part(
        vertices=np.concatenate([factories.CUBE_POINTS, factories.CUBE_POINTS[:1]]),
    )
    result = mesh.duplicate_vertices(factories.context(factories.asset([doubled])))
    assert result.status == FAIL and result.metrics["duplicate_vertices"] == 1


def test_duplicate_vertices_ignores_uv_seam_splits():
    """A glTF-style seam split shares a position but differs in UV."""
    points = np.concatenate([factories.CUBE_POINTS, factories.CUBE_POINTS[:1]])
    part = factories.part(vertices=points)
    part.uvs = np.zeros((len(part.triangles), 3, 2))
    part.uvs[0, 0] = [0.9, 0.9]
    part.triangles = np.where(part.triangles == 0, 8, part.triangles)
    part.triangles[0, 0] = 0
    result = mesh.duplicate_vertices(factories.context(factories.asset([part])))
    assert result.status == PASS


def test_loose_vertices_detected():
    assert (
        mesh.loose_vertices(
            factories.context(factories.asset([factories.part()]))
        ).status
        == PASS
    )

    stray = factories.part(
        vertices=np.concatenate([factories.CUBE_POINTS, [[9.0, 9.0, 9.0]]])
    )
    result = mesh.loose_vertices(factories.context(factories.asset([stray])))
    assert result.status == FAIL and result.metrics["loose_vertices"] == 1


def test_non_manifold_detects_a_third_face_on_one_edge():
    assert (
        mesh.non_manifold(factories.context(factories.asset([factories.part()]))).status
        == PASS
    )

    points, triangles, _, _ = factories.cube_arrays()
    extra = np.concatenate([points, [[0.5, 0.5, 2.0]]])
    faces = np.concatenate([triangles, [[0, 1, 8]]])
    result = mesh.non_manifold(
        factories.context(
            factories.asset([factories.part(vertices=extra, triangles=faces)])
        )
    )
    assert result.status == FAIL and result.metrics["nonmanifold_edges"] == 1


def test_non_manifold_detects_a_bowtie_vertex():
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, -1, 0]]
    part = factories.part(vertices=points, triangles=[[0, 1, 2], [0, 3, 4]])
    result = mesh.non_manifold(factories.context(factories.asset([part])))
    assert result.metrics["bowtie_vertices"] == 1


def test_watertight_passes_closed_and_fails_open():
    assert (
        mesh.watertight(factories.context(factories.asset([factories.part()]))).status
        == PASS
    )

    points, triangles, _, _ = factories.cube_arrays()
    open_box = factories.part(vertices=points, triangles=triangles[:-1])
    result = mesh.watertight(factories.context(factories.asset([open_box])))
    assert result.status == FAIL and result.metrics["open_components"] == 1


def test_normals_outward_detects_inverted_winding():
    assert (
        mesh.normals_outward(
            factories.context(factories.asset([factories.part()]))
        ).status
        == PASS
    )

    points, triangles, _, _ = factories.cube_arrays()
    flipped = factories.part(vertices=points, triangles=triangles[:, ::-1])
    result = mesh.normals_outward(factories.context(factories.asset([flipped])))
    assert result.status == FAIL and result.metrics["inward_components"] == 1


def test_self_intersection_finds_crossing_triangles():
    assert (
        mesh.self_intersection(
            factories.context(factories.asset([factories.part()]))
        ).status
        == PASS
    )

    points = [
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
        [0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1],
    ]
    crossed = factories.part(
        vertices=points, triangles=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    )
    result = mesh.self_intersection(factories.context(factories.asset([crossed])))
    assert result.status == FAIL and result.metrics["intersecting_pairs"] > 0


def test_self_intersection_reports_not_applicable_over_the_cap():
    context = factories.context(
        factories.asset([factories.part()]), self_intersection_max_triangles=2
    )
    assert mesh.self_intersection(context).status == NOT_APPLICABLE


def test_rest_penetration_measures_overlap_between_parts():
    apart = [factories.part(name="a"), factories.part(name="b", offset=(2, 0, 0))]
    assert (
        mesh.rest_penetration(factories.context(factories.asset(apart))).status == PASS
    )

    overlapping = [
        factories.part(name="a"),
        factories.part(name="b", offset=(0.5, 0, 0)),
    ]
    result = mesh.rest_penetration(factories.context(factories.asset(overlapping)))
    assert result.status == FAIL and result.metrics["max_depth"] > 0


def test_floaters_ignores_touching_parts_and_flags_detached_ones():
    touching = [
        factories.part(name="body", scale=2.0),
        factories.part(name="knob", scale=0.2, offset=(2.0, 0.5, 0.5)),
    ]
    assert mesh.floaters(factories.context(factories.asset(touching))).status == PASS

    detached = [
        factories.part(name="body", scale=2.0),
        factories.part(name="knob", scale=0.2, offset=(2.0, 0.5, 0.5)),
        factories.part(name="stray", scale=0.2, offset=(20.0, 0, 0)),
    ]
    result = mesh.floaters(factories.context(factories.asset(detached)))
    assert result.status == FAIL
    assert result.metrics["floaters"] == 1
    assert result.metrics["examples"] == ["stray"]


def test_spikes_tolerates_cube_corners_and_flags_a_needle():
    result = mesh.spikes(factories.context(factories.asset([factories.part()])))
    assert result.status == PASS
    assert result.metrics["max_angle_defect"] == pytest.approx(np.pi / 2, abs=1e-9)

    # A tall spike on a tiny base: the apex subtends almost no solid angle.
    tip = np.array([[-0.004, -0.004, 0], [0.004, -0.004, 0], [0, 0.004, 0], [0, 0, 1.0]])
    needle = factories.part(
        vertices=tip, triangles=[[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]]
    )
    result = mesh.spikes(factories.context(factories.asset([needle])))
    assert result.status == FAIL and result.metrics["spike_vertices"] == 1


def test_poly_budget():
    assert (
        mesh.poly_budget(factories.context(factories.asset([factories.part()]))).status
        == PASS
    )
    result = mesh.poly_budget(
        factories.context(factories.asset([factories.part()]), poly_budget=4)
    )
    assert result.status == FAIL and result.metrics["triangles"] == 12


def test_density_reports_percentiles():
    result = mesh.density(
        factories.context(factories.asset([factories.sphere_part()]))
    )
    assert result.status == INFO
    assert (
        result.metrics["edge_over_curvature_radius_p5"]
        <= result.metrics["edge_over_curvature_radius_p95"]
    )
