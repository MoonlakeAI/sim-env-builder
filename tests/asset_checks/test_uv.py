"""UV layout checks on a cleanly unwrapped cube and deliberate faults."""

import numpy as np

import factories
from sim_env_builder.asset_checks.checks import uv
from sim_env_builder.asset_checks.checks.registry import FAIL, INFO, NOT_APPLICABLE, PASS


def textured(uvs, **kwargs):
    """One part with the given UVs, bound to a material that has a texture."""
    material = factories.material(textures={"basecolor": factories.texture()})
    part = factories.part(uvs=uvs, material=material.name, **kwargs)
    return factories.asset([part], materials={material.name: material})


def unwrapped(**kwargs):
    return textured(factories.unwrapped_cube_uvs(), **kwargs)


def test_stretch_passes_on_an_even_unwrap():
    result = uv.stretch(factories.context(unwrapped()))
    assert result.status == PASS
    assert result.metrics["conformal_p95"] < 2.0


def test_stretch_fails_when_shells_are_squashed():
    squashed = factories.unwrapped_cube_uvs()
    squashed[:, :, 1] *= 0.05
    result = uv.stretch(factories.context(textured(squashed)))
    assert result.status == FAIL and result.metrics["conformal_p95"] > 2.0


def test_stretch_fails_when_a_textured_part_has_no_uvs():
    material = factories.material(textures={"basecolor": factories.texture()})
    mixed = factories.asset(
        [
            factories.part(
                name="mapped",
                uvs=factories.unwrapped_cube_uvs(),
                material=material.name,
            ),
            factories.part(name="bare", offset=(3, 0, 0), material=material.name),
        ],
        materials={material.name: material},
    )
    result = uv.stretch(factories.context(mixed))
    assert result.status == FAIL and result.metrics["parts_without_uv"] == 1


def test_degenerate_uv_triangles_detected():
    assert uv.degenerate(factories.context(unwrapped())).status == PASS

    collapsed = factories.unwrapped_cube_uvs()
    collapsed[0] = collapsed[0][0]
    result = uv.degenerate(factories.context(textured(collapsed)))
    assert result.status == FAIL and result.metrics["degenerate_uv_triangles"] == 1


def test_shells_counts_islands_and_fails_when_fragmented():
    result = uv.shells(factories.context(unwrapped()))
    assert result.status == PASS and result.metrics["shells_max_per_part"] == 6

    fragmented = uv.shells(factories.context(unwrapped(), uv_shells_per_part=2))
    assert fragmented.status == FAIL


def test_texel_density_passes_even_and_fails_uneven():
    assert uv.texel_density(factories.context(unwrapped())).status == PASS

    uneven = factories.unwrapped_cube_uvs()
    uneven[:2] *= 0.1
    result = uv.texel_density(factories.context(textured(uneven)))
    assert result.status == FAIL





def test_out_of_bounds_detected():
    assert uv.out_of_bounds(factories.context(unwrapped())).status == PASS

    tiled = factories.unwrapped_cube_uvs() * 3.0
    result = uv.out_of_bounds(factories.context(textured(tiled)))
    assert result.status == FAIL and result.metrics["out_of_bounds_corners"] > 0


def test_utilization_reports_coverage_and_warns_when_sparse():
    result = uv.utilization(factories.context(unwrapped()))
    assert result.status == INFO
    assert 0.0 < result.metrics["atlas_utilization_min"] <= 1.0

    tiny = factories.unwrapped_cube_uvs(margin=0.01, scale=0.05)
    sparse = uv.utilization(factories.context(textured(tiny)))
    assert sparse.status == INFO and sparse.message is not None


def test_seams_on_sharp_edges_reports_the_share_on_creases():
    result = uv.seams_on_sharp_edges(factories.context(unwrapped()))
    assert result.status == INFO
    # Every cube quad is its own shell, so every seam sits on a 90 degree edge.
    assert result.metrics["seam_edges"] > 0
    assert result.metrics["sharp_seam_ratio"] == 1.0


def test_uv_origin_convention_does_not_change_shell_count():
    flipped = factories.unwrapped_cube_uvs()
    flipped[:, :, 1] = 1.0 - flipped[:, :, 1]
    counts = [
        uv.shells(factories.context(textured(coords))).metrics["shells_total"]
        for coords in (factories.unwrapped_cube_uvs(), flipped)
    ]
    assert counts[0] == counts[1] == 6
    assert np.isfinite(counts).all()


def _procedural(**kwargs):
    """A material with no image textures, as procedural shading produces."""
    material = factories.material(textures={})
    part = factories.part(
        uvs=factories.unwrapped_cube_uvs(**kwargs), material=material.name
    )
    return factories.asset([part], materials={material.name: material})


def test_packing_checks_skip_materials_with_no_image_texture():
    """Packing checks skip UVs that address no texture."""
    overlapping = factories.unwrapped_cube_uvs()
    overlapping[1] = overlapping[0]
    material = factories.material(textures={})
    part = factories.part(uvs=overlapping, material=material.name)
    asset = factories.asset([part], materials={material.name: material})

    context = factories.context(asset)
    for check in (uv.utilization, uv.texel_density):
        assert check(context).status == NOT_APPLICABLE, check.__name__


def test_packing_checks_still_run_when_a_texture_is_bound():
    material = factories.material(textures={"basecolor": factories.texture()})
    part = factories.part(
        uvs=factories.unwrapped_cube_uvs(), material=material.name
    )
    asset = factories.asset([part], materials={material.name: material})
    assert uv.texel_density(factories.context(asset)).status == PASS


def test_shape_checks_skip_parts_that_sample_no_texture():
    """UVs a material never reads are inert data, not a quality surface."""
    from sim_env_builder.asset_checks.checks.registry import NOT_APPLICABLE as NA

    context = factories.context(_procedural())
    for check in (uv.stretch, uv.degenerate, uv.shells, uv.out_of_bounds):
        assert check(context).status == NA, check.__name__


def test_shape_checks_run_on_detail_only_materials():
    """Normal and roughness maps sample UVs, so the checks judge the unwrap."""
    material = factories.material(textures={"normal": factories.texture()})
    squashed = factories.unwrapped_cube_uvs()
    squashed[:, :, 1] *= 0.05
    part = factories.part(uvs=squashed, material=material.name)
    asset = factories.asset([part], materials={material.name: material})
    assert uv.stretch(factories.context(asset)).status == FAIL


def test_out_of_bounds_allows_tiling_on_detail_materials():
    material = factories.material(textures={"normal": factories.texture()})
    tiled = factories.unwrapped_cube_uvs() * 3.0
    part = factories.part(uvs=tiled, material=material.name)
    asset = factories.asset([part], materials={material.name: material})
    from sim_env_builder.asset_checks.checks.registry import NOT_APPLICABLE as NA

    assert uv.out_of_bounds(factories.context(asset)).status == NA


def test_packing_checks_skip_detail_only_materials():
    """Normal and roughness maps provide tileable detail with intentional overlap."""
    material = factories.material(
        textures={"normal": factories.texture(), "roughness": factories.texture()}
    )
    overlapping = factories.unwrapped_cube_uvs()
    overlapping[1] = overlapping[0]
    part = factories.part(uvs=overlapping, material=material.name)
    asset = factories.asset([part], materials={material.name: material})

    context = factories.context(asset)
    for check in (uv.utilization, uv.texel_density):
        assert check(context).status == NOT_APPLICABLE, check.__name__


def test_texel_density_ignores_parts_without_an_atlas():
    """A detail-only part with a wild unwrap must not drag down a clean atlas."""
    atlas = factories.material(
        name="atlas", textures={"basecolor": factories.texture()}
    )
    detail = factories.material(name="detail", textures={"normal": factories.texture()})
    wild = factories.unwrapped_cube_uvs()
    wild[:3] *= 0.02
    asset = factories.asset(
        [
            factories.part(
                name="skinned",
                uvs=factories.unwrapped_cube_uvs(),
                material=atlas.name,
            ),
            factories.part(
                name="trim", offset=(3, 0, 0), uvs=wild, material=detail.name
            ),
        ],
        materials={atlas.name: atlas, detail.name: detail},
    )
    result = uv.texel_density(factories.context(asset))
    assert result.status == PASS


def test_packing_checks_only_require_uvs_on_atlas_parts():
    """A detail-only part with no UVs at all is not a packing failure."""
    atlas = factories.material(
        name="atlas", textures={"basecolor": factories.texture()}
    )
    detail = factories.material(name="detail", textures={"normal": factories.texture()})
    asset = factories.asset(
        [
            factories.part(
                name="skinned",
                uvs=factories.unwrapped_cube_uvs(),
                material=atlas.name,
            ),
            factories.part(name="bare", offset=(3, 0, 0), material=detail.name),
        ],
        materials={atlas.name: atlas, detail.name: detail},
    )
    context = factories.context(asset)
    assert uv.texel_density(context).status == PASS
