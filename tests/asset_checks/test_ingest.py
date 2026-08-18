"""Loader behaviour on real USD and glTF files written by the tests."""

import numpy as np
import PIL.Image
import pytest

import authoring
from sim_env_builder.asset_checks import ingest
from sim_env_builder.asset_checks.ingest import gltf, usd


def test_unsupported_extension_is_rejected():
    with pytest.raises(ValueError, match="unsupported format"):
        ingest.load("asset.obj")


def test_usd_hinge_round_trips(tmp_path):
    asset = ingest.load(authoring.hinged_stage(tmp_path / "hinge.usda"))
    assert asset.format == "usd"
    assert len(asset.parts) == 2
    assert len(asset.proxies) == 2
    assert asset.is_articulated

    joint = asset.joints[0]
    assert joint.joint_type == "revolute"
    assert joint.parent_link.endswith("/base")
    assert joint.child_link.endswith("/door")
    assert np.allclose(joint.axis, [0, 0, 1])
    assert np.allclose(joint.anchor, [1.0, 0.0, 0.0])
    # Authored in degrees, reported in radians, and deliberately reversed.
    assert joint.lower == pytest.approx(np.pi / 2)
    assert joint.upper == pytest.approx(-np.pi / 2)
    assert joint.rest_value == pytest.approx(0.0, abs=1e-9)


def test_usd_mass_on_collision_shapes_accumulates_to_the_link(tmp_path):
    asset = ingest.load(authoring.hinged_stage(tmp_path / "hinge.usda"))
    masses = sorted(link.mass for link in asset.links.values())
    assert masses == [1.0, 2.0]


def test_usd_stage_units_are_converted_to_metres(tmp_path):
    centimetres = ingest.load(
        authoring.hinged_stage(tmp_path / "cm.usda", meters_per_unit=0.01)
    )
    assert centimetres.parts[0].vertices.max() == pytest.approx(0.01)
    assert centimetres.joints[0].anchor[0] == pytest.approx(0.01)


def test_usd_polygons_are_preserved_and_fanned(tmp_path):
    asset = ingest.load(authoring.hinged_stage(tmp_path / "hinge.usda"))
    part = asset.parts[0]
    assert part.face_counts.tolist() == [4] * 6
    assert len(part.triangles) == 12
    assert asset.has_polygons
    assert part.uvs.shape == (12, 3, 2)


def test_usd_purposes_split_render_from_proxy_and_drop_guides(tmp_path):
    asset = ingest.load(authoring.stage_with_purposes(tmp_path / "purpose.usda"))
    assert [p.name.split("/")[-1] for p in asset.parts] == ["render"]
    assert [p.name.split("/")[-1] for p in asset.proxies] == ["proxy"]


def test_usd_convex_hull_approximation_builds_the_derived_shape(tmp_path):
    asset = ingest.load(authoring.stage_with_convex_hull(tmp_path / "hull.usda"))
    assert asset.proxies[0].source == "convex_hull"


def test_usd_collider_on_a_render_mesh_is_marked_as_reused(tmp_path):
    asset = ingest.load(authoring.hinged_stage(tmp_path / "hinge.usda"))
    assert {p.source for p in asset.proxies} == {"render_mesh"}


def test_usd_left_handed_orientation_flips_winding(tmp_path):
    right = ingest.load(authoring.stage_with_purposes(tmp_path / "right.usda"))
    left = ingest.load(authoring.stage_left_handed(tmp_path / "left.usda"))
    assert np.array_equal(left.parts[0].triangles, right.parts[0].triangles[:, ::-1])


def test_usd_material_and_texture_are_read(tmp_path):
    texture = tmp_path / "albedo.png"
    PIL.Image.new("RGB", (8, 8), (200, 100, 50)).save(texture)
    asset = ingest.load(
        authoring.stage_with_material(tmp_path / "mat.usda", texture)
    )
    material = next(iter(asset.materials.values()))
    assert material.shader == "UsdPreviewSurface"
    assert material.params["roughness"] == pytest.approx(0.4)
    assert material.textures["basecolor"].size == (8, 8)
    assert asset.parts[0].material == material.name


def test_usd_missing_texture_loads_as_unresolved(tmp_path):
    asset = ingest.load(
        authoring.stage_with_material(tmp_path / "mat.usda", tmp_path / "absent.png")
    )
    material = next(iter(asset.materials.values()))
    assert material.textures["basecolor"].image is None


def test_gltf_applies_node_transforms(tmp_path):
    asset = ingest.load(authoring.two_part_glb(tmp_path / "two.glb"))
    assert asset.format == "gltf"
    assert len(asset.parts) == 2
    assert not asset.has_polygons
    assert not asset.is_articulated

    centres = sorted(float(p.vertices.mean(axis=0)[0]) for p in asset.parts)
    assert centres[0] == pytest.approx(0.0, abs=1e-6)
    assert centres[1] == pytest.approx(2.0, abs=1e-6)


def test_gltf_collision_named_meshes_become_proxies(tmp_path):
    asset = ingest.load(authoring.glb_with_collision_name(tmp_path / "col.glb"))
    assert [p.name for p in asset.parts] == ["body"]
    assert [p.name for p in asset.proxies] == ["body_collision"]


def test_gltf_skin_weights_are_decoded(tmp_path):
    weights = np.array(
        [[1.0, 0.0], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]], dtype=np.float32
    )
    asset = ingest.load(authoring.skinned_gltf(tmp_path / "skin.gltf", weights))
    skin = asset.parts[0].skin
    assert skin is not None
    assert asset.has_skin
    assert skin.weights.shape == (4, 4)
    assert np.allclose(skin.weights[:, :2], weights)
    assert skin.joints == ["root_joint"]


def test_glb_and_gltf_containers_agree(tmp_path):
    weights = np.array([[1.0, 0.0]] * 4, dtype=np.float32)
    source = tmp_path / "skin.gltf"
    authoring.skinned_gltf(source, weights)
    packed = ingest.load(authoring.glb_from_gltf(source, tmp_path / "skin.glb"))
    assert packed.parts[0].skin is not None


def test_loaders_expose_the_same_static_link_name():
    assert usd.STATIC_LINK == gltf.STATIC_LINK


def test_gltf_normalized_weights_are_divided_back_to_unit_range(tmp_path):
    weights = np.array(
        [[1.0, 0.0], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]], dtype=np.float32
    )
    asset = ingest.load(
        authoring.skinned_gltf(tmp_path / "norm.gltf", weights, normalized=True)
    )
    skin = asset.parts[0].skin
    assert skin.weights.max() <= 1.0
    assert np.allclose(skin.weights[:, :2], weights, atol=1e-4)


def test_gltf_uv_origin_matches_the_specification(tmp_path):
    """uv (0,0) is the image's top-left corner; the loader converts to the
    model's lower-left-origin convention, so the sampler must recover the
    spec colours end to end."""
    from sim_env_builder.asset_checks.checks import materials as material_checks

    asset = ingest.load(
        authoring.textured_quad_gltf(tmp_path / "quad.gltf", "quadrants.png")
    )
    part = asset.parts[0]
    texture = asset.materials[part.material].textures["basecolor"]

    # File uv (0,0) belongs to the vertex at position (0,0,0): triangle 0,
    # corner 0. Sampling the loader's uv for that corner must return red.
    by_position = {
        tuple(part.vertices[part.triangles[t, c]][:2]): part.uvs[t, c]
        for t in range(len(part.triangles))
        for c in range(3)
    }
    expectations = {
        (0.0, 0.0): (1.0, 0.0, 0.0),  # file uv (0,0) -> top-left -> red
        (1.0, 0.0): (0.0, 1.0, 0.0),  # file uv (1,0) -> top-right -> green
        (0.0, 1.0): (0.0, 0.0, 1.0),  # file uv (0,1) -> bottom-left -> blue
        (1.0, 1.0): (1.0, 1.0, 1.0),  # file uv (1,1) -> bottom-right -> white
    }
    for position, expected in expectations.items():
        loader_uv = by_position[position]
        # Nudge toward the quadrant interior to stay off the colour boundary.
        interior = loader_uv + (0.5 - loader_uv) * 0.2
        sampled = material_checks._sample_texture(texture, interior[None, :])[0]
        assert tuple(sampled) == expected, (position, loader_uv, sampled)


def test_world_anchored_joint_moves_the_present_body(tmp_path):
    """The joint drives the real body regardless of the omitted relationship."""
    for omit in ("body0", "body1"):
        asset = ingest.load(
            authoring.world_anchored_stage(tmp_path / f"{omit}.usda", omit)
        )
        joint = asset.joints[0]
        assert joint.parent_link is None, omit
        assert joint.child_link.endswith("/asset"), omit


def test_usd_normals_use_the_inverse_transpose(tmp_path):
    """Under a scale of (2,1,1), a local normal along (1,1,0) must transform
    proportionally to (0.5,1,0), not (2,1,0)."""
    asset = ingest.load(authoring.scaled_normals_stage(tmp_path / "scaled.usda"))
    expected = np.array([0.5, 1.0, 0.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(asset.parts[0].normals[0], expected, atol=1e-6)


def test_gltf_normals_stay_true_under_nonuniform_node_scale(tmp_path):
    import trimesh

    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.icosphere(subdivisions=2),
        node_name="ball",
        transform=np.diag([2.0, 1.0, 1.0, 1.0]),
    )
    path = tmp_path / "scaled.glb"
    path.write_bytes(scene.export(file_type="glb"))

    asset = ingest.load(str(path))
    part = asset.parts[0]
    truth = trimesh.Trimesh(part.vertices, part.triangles).vertex_normals
    agreement = np.abs((part.normals * truth).sum(axis=1))
    assert np.percentile(agreement, 5) > 0.99


def test_usd_indexed_vertex_uvs_resolve_through_the_index_table(tmp_path):
    asset = ingest.load(authoring.indexed_uv_stage(tmp_path / "indexed.usda"))
    part = asset.parts[0]
    # Point 0 maps to table entry 0, point 1 to entry 1, and so on.
    per_point = {0: (0.25, 0.25), 1: (0.75, 0.75), 2: (0.25, 0.25), 3: (0.75, 0.75)}
    for t in range(len(part.triangles)):
        for c in range(3):
            assert tuple(part.uvs[t, c]) == per_point[part.triangles[t, c]]
