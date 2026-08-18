"""Hand-written numerics validated against independent references.

Each routine lacks a dependable library equivalent. These tests use closed-form
values or an independent oracle.
"""

import numpy as np
import pytest
import skimage.color
import trimesh

import factories
from sim_env_builder.asset_checks import geometry
from sim_env_builder.asset_checks.checks import mesh as mesh_checks
from sim_env_builder.asset_checks.checks import uv as uv_checks
from sim_env_builder.asset_checks.ingest import usd


def test_ciede2000_reference_pair():
    """Sharma, Wu & Dalal (2005) test pair number 1: dE00 = 2.0425."""
    a = np.array([[50.0, 2.6772, -79.7751]])
    b = np.array([[50.0, 0.0, -82.7485]])
    assert skimage.color.deltaE_ciede2000(a, b)[0] == pytest.approx(2.0425, abs=1e-4)


def test_aspect_ratio_closed_forms():
    """2r/R: equilateral = 1; a 3-4-5 right triangle has r = 1, R = 2.5."""
    equilateral = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    surface = geometry.Surface(equilateral, np.array([[0, 1, 2]]), 1e-9)
    assert mesh_checks._aspect_ratio(surface)[0] == pytest.approx(1.0)

    right = np.array([[0, 0, 0], [4, 0, 0], [0, 3, 0]])
    surface = geometry.Surface(right, np.array([[0, 1, 2]]), 1e-9)
    assert mesh_checks._aspect_ratio(surface)[0] == pytest.approx(0.8)


def test_uv_jacobian_singular_values_on_analytic_maps():
    """A k-times-scaled unwrap has singular values (k, k); squashing one axis
    by k makes the conformal ratio k."""
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    surface = geometry.Surface(triangle, np.array([[0, 1, 2]]), 1e-9)

    identity = np.array([[[0, 0], [1, 0], [0, 1]]], dtype=float)
    singular = uv_checks._singular_values(identity, surface)[0]
    assert singular == pytest.approx([1.0, 1.0])

    doubled = identity * 2.0
    singular = uv_checks._singular_values(doubled, surface)[0]
    assert singular == pytest.approx([2.0, 2.0])

    squashed = identity.copy()
    squashed[:, :, 1] *= 1.0 / 3.0
    singular = uv_checks._singular_values(squashed, surface)[0]
    assert singular[0] / singular[1] == pytest.approx(3.0)


def test_revolute_rest_value_recovers_a_known_rotation():
    """Joint frames differing by a rotation of theta about the axis must read
    back as a rest value of theta."""
    axis = np.array([0.0, 0.0, 1.0])
    for theta in (0.3, -1.2, 2.9):
        frame0 = np.eye(4)
        frame1 = np.eye(4)
        frame1[:3, :3] = trimesh.transformations.rotation_matrix(theta, axis)[:3, :3]
        value = usd._rest_value([frame0, frame1], axis, "revolute")
        assert value == pytest.approx(theta)


def test_prismatic_rest_value_is_the_offset_along_the_axis():
    axis = np.array([1.0, 0.0, 0.0])
    frame0, frame1 = np.eye(4), np.eye(4)
    frame1[:3, 3] = [0.4, 9.0, -2.0]
    assert usd._rest_value([frame0, frame1], axis, "prismatic") == pytest.approx(0.4)


def _oracle_intersects(a: np.ndarray, b: np.ndarray) -> bool:
    """Independent tri-tri oracle: two non-coplanar triangles intersect exactly
    when an edge of one passes through the other. Edge-versus-triangle is
    delegated to trimesh's ray engine."""
    for source, target in ((a, b), (b, a)):
        target_mesh = trimesh.Trimesh(
            vertices=target, faces=[[0, 1, 2]], process=False
        )
        for i in range(3):
            origin = source[i]
            direction = source[(i + 1) % 3] - source[i]
            length = np.linalg.norm(direction)
            locations, _, _ = target_mesh.ray.intersects_location(
                [origin], [direction / length]
            )
            if len(locations) and (
                np.linalg.norm(locations - origin, axis=1) <= length + 1e-12
            ).any():
                return True
    return False


def test_triangle_intersection_agrees_with_the_ray_oracle():
    generator = np.random.default_rng(7)
    disagreements = 0
    checked = 0
    for _ in range(400):
        a = generator.normal(size=(3, 3))
        b = generator.normal(size=(3, 3)) * 0.8 + generator.normal(size=3) * 0.5
        mine = bool(
            geometry.triangles_intersect(a[None, ...], b[None, ...])[0]
        )
        oracle = _oracle_intersects(a, b)
        checked += 1
        disagreements += mine != oracle
    assert checked == 400
    assert disagreements == 0


def test_rasterizer_partitions_a_fan_triangulation():
    """A triangulated convex polygon covers texels exactly once."""
    generator = np.random.default_rng(3)
    for _ in range(20):
        angles = np.sort(generator.uniform(0, 2 * np.pi, 8))
        radius = generator.uniform(0.2, 0.45)
        centre = generator.uniform(0.45, 0.55, 2)
        ring = centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        fan = np.stack(
            [np.stack([ring[0], ring[i], ring[i + 1]]) for i in range(1, 7)]
        )
        counts, _ = geometry.rasterize_uv(fan, 128, np.arange(len(fan)))
        assert (counts > 1).sum() == 0
        covered = (counts > 0).sum() / 128.0**2
        area = uv_checks._uv_areas(fan).sum()
        assert covered == pytest.approx(area, rel=0.08)


def test_angle_defect_matches_gauss_bonnet_on_a_closed_mesh():
    """Sum of angle defects of any closed genus-0 mesh is 4*pi."""
    for shape in (
        trimesh.creation.icosphere(subdivisions=2),
        trimesh.creation.box(),
        trimesh.creation.cylinder(radius=0.5, height=2.0, sections=16),
    ):
        surface = geometry.Surface(
            np.asarray(shape.vertices), np.asarray(shape.faces), 1e-9
        )
        assert geometry.interior_defects(surface).sum() == pytest.approx(
            4 * np.pi, rel=1e-6
        )


def test_fan_triangulation_preserves_polygon_area():
    """Fanned convex polygons keep their shoelace area."""
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(7)], axis=1)
    counts = np.array([7])
    indices = np.arange(7)
    slots = usd._fan_slots(counts)
    corners = ring[indices[slots]]
    fanned = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    ).sum()
    shoelace = 0.5 * abs(
        np.sum(
            ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1]
        )
    )
    assert fanned == pytest.approx(shoelace)


def test_delta_e_operates_in_lab_space():
    """Red versus green is near 86 dE00; a loose band distinguishes Lab from RGB."""
    red = factories.texture(colour=(1.0, 0.0, 0.0), size=(8, 8))
    green = factories.texture(colour=(0.0, 1.0, 0.0), size=(8, 8))
    from sim_env_builder.asset_checks.checks import materials

    point = np.array([[0.5, 0.5]])
    a = skimage.color.rgb2lab(materials._sample_texture(red, point))
    b = skimage.color.rgb2lab(materials._sample_texture(green, point))
    assert 80.0 < float(skimage.color.deltaE_ciede2000(a, b)[0]) < 95.0
