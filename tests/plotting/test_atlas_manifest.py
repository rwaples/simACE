"""Sanity checks on the atlas manifest registry."""

from simace.plotting.atlas_manifest import (
    MODEL_SECTION,
    PHENOTYPE_ATLAS,
    VALIDATION_ATLAS,
    PlotEntry,
    SectionBreak,
    build_phenotype_atlas,
    effective_size_basenames,
    phenotype_basenames,
    validation_basenames,
)
from simace.plotting.plot_phenotype import PHENOTYPE_RENDERERS

# Includes ``consanguineous_matings`` at index 3 — the entry that
# ``workflow/common.py`` pre-PR8 was missing despite it being rendered.
EXPECTED_VALIDATION_BASENAMES = [
    "family_size",
    "twin_rate",
    "half_sib_proportions",
    "consanguineous_matings",
    "variance_components",
    "correlations_A",
    "correlations_phenotype",
    "heritability_estimates",
    "cross_trait_correlations",
    "summary_bias",
    "runtime",
    "memory",
]

EXPECTED_EFFECTIVE_SIZE_BASENAMES = [
    "effective_size.estimators",
    "effective_size.by_generation",
    "effective_size.drift",
    "effective_size.family_size_variance",
]


def test_every_phenotype_basename_has_exactly_one_renderer():
    """The renderer registry and the manifest must declare the same basenames.

    This is the drift gate that replaced the frozen phenotype basename list:
    adding a plot requires both a PlotEntry in PHENOTYPE_ATLAS and a
    PlotRenderSpec in PHENOTYPE_RENDERERS, or this fails. Render order is
    irrelevant to outputs, so parity is asserted on the *set*, not the order.
    """
    reg = [spec.basename for spec in PHENOTYPE_RENDERERS]
    assert sorted(reg) == sorted(phenotype_basenames())
    assert len(reg) == len(set(reg))


def test_validation_basenames_match_frozen_list():
    assert validation_basenames() == EXPECTED_VALIDATION_BASENAMES


def test_effective_size_basenames_match_frozen_list():
    assert effective_size_basenames() == EXPECTED_EFFECTIVE_SIZE_BASENAMES


def test_phenotype_basenames_are_unique():
    names = phenotype_basenames()
    assert len(names) == len(set(names))


def test_validation_basenames_are_unique():
    names = validation_basenames()
    assert len(names) == len(set(names))


def test_effective_size_basenames_are_unique():
    names = effective_size_basenames()
    assert len(names) == len(set(names))


def test_no_basename_collision_across_atlases():
    p = set(phenotype_basenames())
    v = set(validation_basenames())
    e = set(effective_size_basenames())
    assert p.isdisjoint(v)
    assert p.isdisjoint(e)
    assert v.isdisjoint(e)


def test_model_section_appears_at_most_once():
    occurrences = sum(1 for item in PHENOTYPE_ATLAS if item is MODEL_SECTION)
    assert occurrences == 1


def test_section_breaks_not_at_atlas_endpoints():
    """Section breaks shouldn't be the very first or last item (no orphan dividers)."""
    for atlas in (PHENOTYPE_ATLAS, VALIDATION_ATLAS):
        if not atlas:
            continue
        assert isinstance(atlas[0], PlotEntry), f"first item is a section break in {atlas[0]}"
        assert isinstance(atlas[-1], PlotEntry), f"last item is a section break in {atlas[-1]}"


def test_build_phenotype_atlas_no_params_omits_model_section():
    items = build_phenotype_atlas(None)
    assert all(item is not MODEL_SECTION for item in items)
    # Other section breaks survive
    section_titles = [it.title for it in items if isinstance(it, SectionBreak)]
    assert "Age of Onset & Censoring" in section_titles
    assert "Within-Trait Correlations" in section_titles
    assert "Cross-Trait Correlations" in section_titles
    assert "<MODEL>" not in section_titles


def test_build_phenotype_atlas_adult_resolves_model_section():
    params = {
        "phenotype_model1": "adult",
        "phenotype_params1": {"method": "ltm", "cip_x0": 50.0, "cip_k": 0.2, "prevalence": 0.10},
        "phenotype_model2": "adult",
        "phenotype_params2": {"method": "ltm", "cip_x0": 50.0, "cip_k": 0.2, "prevalence": 0.20},
        "beta1": 1.0,
        "beta2": 1.0,
    }
    items = build_phenotype_atlas(params)
    section_breaks = [it for it in items if isinstance(it, SectionBreak)]
    titles = [s.title for s in section_breaks]
    # Some adult-flavored title should now be present (e.g. "ADuLT LTM" or
    # similar from get_model_family).
    assert any("ADuLT" in t or "adult" in t.lower() for t in titles), titles
    # The placeholder sentinel never leaks into the rendered atlas.
    assert "<MODEL>" not in titles


def test_build_phenotype_atlas_frailty_resolves_model_section_with_no_equations():
    params = {
        "phenotype_model1": "frailty",
        "phenotype_params1": {"distribution": "weibull", "scale": 100.0, "rho": 2.0},
        "phenotype_model2": "frailty",
        "phenotype_params2": {"distribution": "weibull", "scale": 100.0, "rho": 2.0},
        "beta1": 1.0,
        "beta2": 1.0,
    }
    items = build_phenotype_atlas(params)
    section_breaks = [it for it in items if isinstance(it, SectionBreak)]
    # The model section is present (resolved against frailty params).
    assert any("Frailty" in s.title or "Weibull" in s.title for s in section_breaks)


def test_plot_entries_have_no_figure_prefix_in_title():
    """The ``Figure N:`` prefix is derived at render time. Stored titles
    must not contain it (otherwise figure numbers would double-prefix)."""
    import re

    figure_prefix = re.compile(r"^Figure\s+\d+:")
    for atlas in (PHENOTYPE_ATLAS, VALIDATION_ATLAS):
        for item in atlas:
            if isinstance(item, PlotEntry):
                assert not figure_prefix.match(item.title), (
                    f"{item.basename}: stored title {item.title!r} starts with 'Figure N:' "
                    f"— this should be derived at render time."
                )
