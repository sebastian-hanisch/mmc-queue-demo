"""Presets: Vollständigkeit, gültige Werte, Permalink-Konstanten."""

import pytest

import mmc_constants as C
import mmc_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER) and len(C.PRESETS) == 4
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for preset in C.PRESETS.values():
        assert C.C_MIN <= preset["c"] <= C.C_MAX and C.RHO_PCT_MIN <= preset["rho_pct"] <= C.RHO_PCT_MAX
        assert preset["n"] in C.N_OPTIONS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(preset[key])


def test_preset_names_state_the_values_they_set():
    for name, preset in C.PRESETS.items():
        if "Spuren" in name and "(" in name and "c =" not in name:
            assert str(preset["c"]) in name
        if "ρ = " in name:
            assert f"ρ = {preset['rho_pct']} %" in name
    assert C.PRESETS["Eine Spur (c = 1)"]["c"] == 1 and C.PRESETS["Großes Gate (16 Spuren)"]["c"] == 16


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Normalfall (4 Spuren, ρ = 90 %)"]
    assert (p["c"], p["rho_pct"], p["n"], p["seed"]) == (C.DEFAULT_C, C.DEFAULT_RHO_PCT, C.DEFAULT_N, C.DEFAULT_SEED)


def test_bounds_and_url_params():
    assert P.bounds("c_slider") == (C.C_MIN, C.C_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


@pytest.mark.parametrize("value,expected", [(1000, 1000), (1400, 1000), (3000, 2000), (3600, 5000), (3500, 2000),
                                            (10**6, 50000), (1, 1000)])
def test_n_snaps_to_the_nearest_option(value, expected):
    assert P.snap_n(value) == expected


def test_formatters():
    assert C.fmt_int(10000) == "10.000" and C.fmt_int(950) == "950"
    assert C.fmt_pct(0.86) == "86 %" and C.fmt_pct(0.5) == "50 %" and C.fmt_pct(0.0) == "0 %"


def test_constants_are_consistent():
    assert C.RHO_PCT_MAX < 100 and C.GRID_N == C.N_OPTIONS and set(C.GRID_C) <= set(range(C.C_MIN, C.C_MAX + 1))
    assert set(C.GRID_RHO_PCT) <= set(range(C.RHO_PCT_MIN, C.RHO_PCT_MAX + 1))
    assert C.DEFAULT_STAFFING_LOAD in C.STAFFING_LOADS and C.DEFAULT_STAFFING_TARGET in C.STAFFING_TARGETS
