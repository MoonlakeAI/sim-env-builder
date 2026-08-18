"""Material and texture checks, including the baked-lighting heuristic."""

import factories
from sim_env_builder.asset_checks.checks import materials
from sim_env_builder.asset_checks.checks.registry import FAIL, NOT_APPLICABLE, PASS


def textured_asset(texture=None, shader="UsdPreviewSurface", **material_kwargs):
    slots = {"basecolor": texture or factories.texture(), "normal": factories.texture()}
    material = factories.material(shader=shader, textures=slots, **material_kwargs)
    part = factories.part(uvs=factories.unwrapped_cube_uvs(), material=material.name)
    return factories.asset([part], materials={material.name: material})


def test_bound_passes_when_every_part_has_a_material():
    assert materials.bound(factories.context(textured_asset())).status == PASS


def test_bound_fails_on_an_unbound_part():
    result = materials.bound(
        factories.context(factories.asset([factories.part(material=None)]))
    )
    assert result.status == FAIL and result.metrics["unbound_parts"] == 1


def test_pbr_passes_for_preview_surface_and_fails_otherwise():
    assert materials.pbr(factories.context(textured_asset())).status == PASS

    result = materials.pbr(factories.context(textured_asset(shader="Phong")))
    assert result.status == FAIL and result.metrics["non_pbr_materials"] == 1


def test_pbr_fails_when_there_are_no_materials():
    result = materials.pbr(factories.context(factories.asset([factories.part()])))
    assert result.status == FAIL


def test_detail_maps_pass_with_a_normal_map_and_fail_without():
    assert materials.detail_maps(factories.context(textured_asset())).status == PASS

    flat = factories.material(textures={"basecolor": factories.texture()})
    asset = factories.asset(
        [factories.part(material=flat.name)], materials={flat.name: flat}
    )
    result = materials.detail_maps(factories.context(asset))
    assert result.status == FAIL and result.metrics["detail_map_fraction"] == 0.0


def test_duplicates_detects_interchangeable_materials():
    assert materials.duplicates(factories.context(textured_asset())).status == PASS

    twins = {
        name: factories.material(name=name, textures={"basecolor": factories.texture()})
        for name in ("first", "second")
    }
    asset = factories.asset([factories.part(material="first")], materials=twins)
    result = materials.duplicates(factories.context(asset))
    assert result.status == FAIL and result.metrics["duplicate_groups"] == 1


def test_missing_textures_detected():
    assert materials.missing_textures(factories.context(textured_asset())).status == PASS

    broken = factories.material(
        textures={"basecolor": factories.texture(image=False)}
    )
    asset = factories.asset(
        [factories.part(material=broken.name)], materials={broken.name: broken}
    )
    result = materials.missing_textures(factories.context(asset))
    assert result.status == FAIL and result.metrics["missing_textures"] == 1


def test_resolution_passes_in_range_and_fails_when_oversized():
    result = materials.resolution(factories.context(textured_asset()))
    assert result.status == PASS
    assert result.metrics["texels_per_cm_median"] > 0

    huge = factories.texture(size=(8192, 8192))
    result = materials.resolution(factories.context(textured_asset(texture=huge)))
    assert result.status == FAIL and result.metrics["oversized_textures"] == 1


def test_resolution_fails_when_texel_density_is_too_coarse():
    context = factories.context(textured_asset(), texels_per_cm_max=0.001)
    assert materials.resolution(context).status == FAIL


def test_resolution_not_applicable_without_uvs():
    material = factories.material(textures={"basecolor": factories.texture()})
    asset = factories.asset(
        [factories.part(material=material.name)], materials={material.name: material}
    )
    assert materials.resolution(factories.context(asset)).status == NOT_APPLICABLE


def test_baked_lighting_accepts_flat_albedo():
    result = materials.baked_lighting(factories.context(textured_asset()))
    assert result.status in (PASS, NOT_APPLICABLE)


def test_baked_lighting_flags_a_brightness_ramp():
    ramped = textured_asset(texture=factories.gradient_texture())
    result = materials.baked_lighting(factories.context(ramped))
    assert result.status == FAIL
    assert result.metrics["shell_gradient"] > 0.3


def test_baked_lighting_not_applicable_without_base_colour():
    material = factories.material(textures={"normal": factories.texture()})
    asset = factories.asset(
        [factories.part(uvs=factories.unwrapped_cube_uvs(), material=material.name)],
        materials={material.name: material},
    )
    result = materials.baked_lighting(factories.context(asset))
    assert result.status == NOT_APPLICABLE


def test_seam_discontinuity_passes_on_a_uniform_texture():
    result = materials.seam_discontinuity(factories.context(textured_asset()))
    assert result.status == PASS
    assert result.metrics["delta_e_p95"] == 0.0


def test_seam_discontinuity_fails_across_a_strong_gradient():
    ramped = textured_asset(texture=factories.gradient_texture())
    result = materials.seam_discontinuity(factories.context(ramped))
    assert result.status == FAIL and result.metrics["delta_e_p95"] > 10.0


def test_seam_discontinuity_not_applicable_without_textures():
    asset = factories.asset([factories.part(uvs=factories.unwrapped_cube_uvs())])
    result = materials.seam_discontinuity(factories.context(asset))
    assert result.status == NOT_APPLICABLE


def test_sample_texture_wraps_before_flipping():
    """v = 0 is the bottom row, and integer tiling wraps back to it."""
    import numpy as np
    import PIL.Image

    from sim_env_builder.asset_checks.checks.materials import _sample_texture
    from sim_env_builder.asset_checks.ingest import model

    image = np.zeros((4, 4, 3), np.uint8)
    image[0, :] = (255, 0, 0)  # top row red
    image[3, :] = (0, 0, 255)  # bottom row blue
    texture = model.Texture(path="t", image=PIL.Image.fromarray(image))

    def sample(v):
        return tuple(_sample_texture(texture, np.array([[0.5, v]]))[0])

    blue, red = (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    assert sample(0.0) == blue
    assert sample(0.001) == blue
    assert sample(0.999) == red
    assert sample(2.0) == blue  # tiled v wraps to the bottom row
