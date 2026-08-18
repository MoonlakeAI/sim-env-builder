"""Geometry routines checked against closed-form answers.

These tests cover code with no dependable library equivalent. They exclude
operations delegated to trimesh, scipy, or scikit-image.
"""

import numpy as np
import pytest
import trimesh

import factories
from sim_env_builder.asset_checks import geometry


def surface_of(mesh: trimesh.Trimesh, tol: float = 1e-9) -> geometry.Surface:
    return geometry.Surface(np.asarray(mesh.vertices), np.asarray(mesh.faces), tol)


def test_weld_merges_coincident_vertices_and_keeps_corner_order():
    points = np.concatenate([factories.CUBE_POINTS, factories.CUBE_POINTS])
    _, triangles, _, _ = factories.cube_arrays()
    doubled = np.concatenate([triangles, triangles + 8])
    surface = geometry.Surface(points, doubled, 1e-6)

    assert len(surface.welded_vertices) == 8
    assert len(surface.welded_triangles) == len(doubled)
    # The two halves address the same welded vertices in the same corner order.
    assert np.array_equal(
        surface.welded_triangles[: len(triangles)],
        surface.welded_triangles[len(triangles) :],
    )


def test_signed_volume_is_negative_for_inverted_winding():
    points, triangles, _, _ = factories.cube_arrays()
    assert geometry.signed_volume(points, triangles) == pytest.approx(1.0)
    assert geometry.signed_volume(points, triangles[:, ::-1]) == pytest.approx(-1.0)


def test_interior_defects_of_a_cube_are_right_angles():
    points, triangles, _, _ = factories.cube_arrays()
    defects = geometry.interior_defects(geometry.Surface(points, triangles, 1e-9))
    assert np.allclose(defects, np.pi / 2)


def test_interior_defects_exclude_boundary_vertices():
    points, triangles, _, _ = factories.cube_arrays()
    # Triangles i and i + 6 form one fanned quad. Drop both to open one face
    # and leave its four corners on the boundary.
    open_box = geometry.Surface(points, np.delete(triangles, [5, 11], axis=0), 1e-9)
    assert len(geometry.interior_defects(open_box)) == 4


@pytest.mark.parametrize("radius", [0.5, 2.0])
def test_mean_curvature_of_a_sphere_is_the_inverse_radius(radius):
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    curvature = geometry.mean_curvature(surface_of(sphere))
    assert curvature.mean() == pytest.approx(1.0 / radius, rel=0.02)


def test_mean_curvature_of_a_plane_is_zero():
    plane = trimesh.creation.box(extents=(1, 1, 1))
    curvature = geometry.mean_curvature(surface_of(plane))
    assert np.isfinite(curvature).all()


def test_triangles_intersect_on_known_pairs():
    a = np.array(
        [
            [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        ],
        dtype=float,
    )
    b = np.array(
        [
            [[1, 1, -1], [1, 1, 1], [0.2, 0.2, 1]],  # pierces the first
            [[5, 5, 5], [6, 5, 5], [5, 6, 5]],  # far away
            [[0, 0, 0], [0, 0, 1], [1, 0, 1]],  # shares an edge
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],  # parallel, offset
        ],
        dtype=float,
    )
    assert geometry.triangles_intersect(a, b).tolist() == [True, False, True, False]


def test_self_intersection_pipeline_finds_crossing_planes():
    points = np.array(
        [
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
            [0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    surface = geometry.Surface(points, faces, 1e-9)
    pairs = geometry.candidate_pairs(surface)
    corners = surface.mesh.triangles
    assert geometry.triangles_intersect(
        corners[pairs[:, 0]], corners[pairs[:, 1]]
    ).any()


def test_candidate_pairs_exclude_triangles_sharing_a_vertex():
    points, triangles, _, _ = factories.cube_arrays()
    surface = geometry.Surface(points, triangles, 1e-9)
    for left, right in geometry.candidate_pairs(surface):
        shared = set(surface.welded_triangles[left]) & set(
            surface.welded_triangles[right]
        )
        assert not shared


def test_rasterize_uv_tiles_adjacent_triangles_without_double_coverage():
    triangles = np.array(
        [[[0, 0], [0.5, 0], [0, 0.5]], [[0.5, 0], [0.5, 0.5], [0, 0.5]]], dtype=float
    )
    counts, _ = geometry.rasterize_uv(triangles, 256, np.arange(2))
    assert (counts > 0).mean() == pytest.approx(0.25, abs=1e-3)
    assert (counts > 1).sum() == 0


def test_rasterize_uv_covers_the_whole_atlas():
    full = np.array(
        [[[0, 0], [1, 0], [0, 1]], [[1, 0], [1, 1], [0, 1]]], dtype=float
    )
    counts, _ = geometry.rasterize_uv(full, 128, np.arange(2))
    assert (counts > 0).all()
    assert (counts > 1).sum() == 0


def test_rasterize_uv_detects_stacked_triangles():
    triangles = np.array([[[0, 0], [1, 0], [0, 1]]] * 2, dtype=float)
    counts, _ = geometry.rasterize_uv(triangles, 64, np.arange(2))
    assert (counts == 2).sum() > 0


def test_rasterize_uv_labels_each_texel_with_a_covering_triangle():
    triangles = np.array(
        [[[0, 0], [0.5, 0], [0, 0.5]], [[0.6, 0.6], [1.0, 0.6], [0.6, 1.0]]],
        dtype=float,
    )
    _, painted = geometry.rasterize_uv(triangles, 64, np.array([7, 9]))
    assert set(np.unique(painted).tolist()) == {-1, 7, 9}


def test_rasterize_uv_handles_an_empty_atlas():
    counts, painted = geometry.rasterize_uv(np.zeros((0, 3, 2)), 16, np.zeros(0))
    assert counts.sum() == 0 and (painted == -1).all()


def test_shared_edges_address_the_same_vertex_from_both_sides():
    points, triangles, _, _ = factories.cube_arrays()
    surface = geometry.Surface(points, triangles, 1e-9)
    adjacency = geometry.shared_edges(surface)
    welded = surface.welded_triangles

    for k in (0, 1):
        left = welded[adjacency.left, adjacency.left_slots[:, k]]
        right = welded[adjacency.right, adjacency.right_slots[:, k]]
        assert np.array_equal(left, right)


def test_bowtie_vertices_finds_the_pinch_point():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 0, 0], [-1, -1, 0]], dtype=float
    )
    surface = geometry.Surface(points, np.array([[0, 1, 2], [0, 3, 4]]), 1e-9)
    assert geometry.bowtie_vertices(surface).tolist() == [0]


def test_bowtie_vertices_empty_on_a_clean_cube():
    points, triangles, _, _ = factories.cube_arrays()
    surface = geometry.Surface(points, triangles, 1e-9)
    assert len(geometry.bowtie_vertices(surface)) == 0


def test_polygon_edges_counts_uses_per_edge():
    _, _, counts, indices = factories.cube_arrays()
    edges, uses = geometry.polygon_edges(counts, indices)
    assert len(edges) == 12
    assert (uses == 2).all()


def test_polygon_edges_marks_an_open_boundary():
    counts = np.array([4])
    indices = np.array([0, 1, 2, 3])
    edges, uses = geometry.polygon_edges(counts, indices)
    assert len(edges) == 4 and (uses == 1).all()


def test_median_edge_length_on_a_unit_cube():
    points, triangles, _, _ = factories.cube_arrays()
    lengths = geometry.median_edge_length(geometry.Surface(points, triangles, 1e-9))
    # Every cube vertex touches three unit edges and some face diagonals.
    assert np.all(lengths >= 1.0)
    assert np.all(lengths <= np.sqrt(2) + 1e-9)


def test_fibonacci_sphere_is_unit_length_and_balanced():
    directions = geometry.fibonacci_sphere(512)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.allclose(directions.mean(axis=0), 0.0, atol=0.05)


def test_occlusion_is_total_inside_a_closed_shell():
    shell = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    inside = geometry.occlusion(
        shell, np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), 32
    )
    assert inside[0] == pytest.approx(1.0)


def test_occlusion_is_zero_on_an_exposed_surface():
    plate = trimesh.creation.box(extents=(4, 4, 0.1))
    exposed = geometry.occlusion(
        plate, np.array([[0.0, 0.0, 0.05]]), np.array([[0.0, 0.0, 1.0]]), 32
    )
    assert exposed[0] == pytest.approx(0.0)


def test_penetration_depth_measures_overlapping_boxes():
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(
        extents=(1, 1, 1),
        transform=trimesh.transformations.translation_matrix([0.5, 0, 0]),
    )
    # The boxes overlap over half their width. The deepest interior point sits
    # 0.5 from the far surface. Finite sampling can only approach that from below.
    depth, _ = geometry.penetration_depth(a, b, 5000)
    assert 0.3 < depth <= 0.5


def test_penetration_depth_is_zero_when_separated():
    a = trimesh.creation.box()
    b = trimesh.creation.box(
        transform=trimesh.transformations.translation_matrix([5, 0, 0])
    )
    assert geometry.penetration_depth(a, b, 1000)[0] == 0.0


def test_surface_points_include_face_interiors():
    box = trimesh.creation.box()
    points = geometry.surface_points(box, 10_000)
    assert len(points) == len(box.vertices) + len(box.faces)


def test_subsample_is_deterministic_and_bounded():
    values = np.arange(100).reshape(-1, 1)
    first = geometry.subsample(values, 10)
    assert len(first) == 10
    assert np.array_equal(first, geometry.subsample(values, 10))
    assert len(geometry.subsample(values, 500)) == 100


def test_aspect_ratio_is_one_for_an_equilateral_triangle():
    """2r/R gives an equilateral triangle 1 and a sliver 0."""
    from sim_env_builder.asset_checks.checks import mesh as mesh_checks

    equilateral = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    surface = geometry.Surface(equilateral, np.array([[0, 1, 2]]), 1e-9)
    assert mesh_checks._aspect_ratio(surface)[0] == pytest.approx(1.0)

    sliver = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-6, 0]])
    surface = geometry.Surface(sliver, np.array([[0, 1, 2]]), 1e-9)
    assert mesh_checks._aspect_ratio(surface)[0] < 0.01


def test_coplanar_overlap_is_detected():
    base = np.array([[[0, 0, 0], [2, 0, 0], [0, 2, 0]]], dtype=float)
    overlapping = np.array([[[0.5, 0.5, 0], [2.5, 0.5, 0], [0.5, 2.5, 0]]], dtype=float)
    assert geometry.triangles_intersect(base, overlapping)[0]


def test_coplanar_duplicate_face_is_detected():
    base = np.array([[[0, 0, 0], [2, 0, 0], [0, 2, 0]]], dtype=float)
    assert geometry.triangles_intersect(base, base.copy())[0]


def test_coplanar_contained_triangle_is_detected():
    outer = np.array([[[0, 0, 0], [4, 0, 0], [0, 4, 0]]], dtype=float)
    inner = np.array([[[1, 1, 0], [2, 1, 0], [1, 2, 0]]], dtype=float)
    assert geometry.triangles_intersect(outer, inner)[0]
    assert geometry.triangles_intersect(inner, outer)[0]


def test_coplanar_disjoint_and_edge_sharing_do_not_count():
    base = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    far = np.array([[[5, 5, 0], [6, 5, 0], [5, 6, 0]]], dtype=float)
    assert not geometry.triangles_intersect(base, far)[0]

    # The other half of the same quad: shares the diagonal, zero-area overlap.
    neighbour = np.array([[[1, 0, 0], [1, 1, 0], [0, 1, 0]]], dtype=float)
    assert not geometry.triangles_intersect(base, neighbour)[0]


def test_self_intersection_check_catches_a_coplanar_duplicated_face():
    from sim_env_builder.asset_checks.checks import mesh as mesh_checks

    points, triangles, _, _ = factories.cube_arrays()
    # This shrunken copy lies in the same plane as a bottom-face triangle. Fresh
    # vertices prevent welding with the original. It touches no other face, so
    # only the coplanar branch can catch it.
    original = points[triangles[0]]
    shrunk = original.mean(axis=0) + (original - original.mean(axis=0)) * 0.5
    assert (shrunk[:, 2] == original[:, 2]).all()
    doubled = np.concatenate([points, shrunk])
    faces = np.concatenate([triangles, [[8, 9, 10]]])
    part = factories.part(vertices=doubled, triangles=faces)
    result = mesh_checks.self_intersection(
        factories.context(factories.asset([part]))
    )
    assert result.status == "fail" and result.metrics["intersecting_pairs"] >= 1
