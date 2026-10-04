"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Randwerte, Würfel-Knopf, Permalink-Grenzen, Abschnitte, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mmc_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_formula_and_simulation():
    at = _run()
    _ok(at)
    assert _metric(at, "Auslastung je Spur ρ") == "90 %"
    assert _metric(at, "Anteil der Lkw, die warten müssen (Formel)") == "79 %"
    assert _metric(at, "Wartezeit in der Schlange (Formel)") == "5.9 min"
    assert _metric(at, "Little's Gesetz im Lauf: L gegen λ·W") == "stimmt"
    assert any("Ein einzelner Lauf ist keine Messung der Formel" in i.value for i in at.info)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["c_slider"] == p["c"] and at.session_state["rho_slider"] == p["rho_pct"]
    assert at.metric


@pytest.mark.parametrize("kw", [dict(c_slider=C.C_MIN), dict(c_slider=C.C_MAX), dict(rho_slider=C.RHO_PCT_MIN),
                                 dict(rho_slider=C.RHO_PCT_MAX), dict(n_select=C.N_OPTIONS[0]),
                                 dict(n_select=C.N_OPTIONS[-1], c_slider=16, rho_slider=97),
                                 dict(c_slider=1, rho_slider=97, n_select=1000),
                                 dict(staffing_load=200, staffing_target=0.5), dict(staffing_load=2, staffing_target=5),
                                 dict(cov_n_select=50000)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_one_lane_equals_the_mm1_values_of_stueck_1():
    at = _run(c_slider=1)
    _ok(at)
    assert _metric(at, "Wartezeit in der Schlange (Formel)") == "27.0 min"
    assert _metric(at, "Anteil der Lkw, die warten müssen (Formel)") == "90 %"
    assert any("nichts zu poolen" in i.value for i in at.info)


def test_more_lanes_change_the_main_metric_and_the_pooling_factor():
    one, four = _run(c_slider=1), _run(c_slider=4)
    assert _metric(one, "Wartezeit in der Schlange (Formel)") != _metric(four, "Wartezeit in der Schlange (Formel)")
    assert _metric(four, "Pooling-Faktor (Formel)") == "4.6×"
    assert any("4.6-Fache" in s.value for s in four.success)


def test_dice_button_changes_the_seed_and_the_result():
    at = _run()
    old_seed, old = at.session_state["seed_input"], _metric(at, "Wartezeit in der Schlange (simuliert)")
    next(b for b in at.button if b.label == "🎲 Neuen Lauf würfeln").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old_seed and _metric(at, "Wartezeit in der Schlange (simuliert)") != old


def test_window_slider_exists_for_long_runs():
    at = _run()
    sl = next(s for s in at.slider if s.key == "window_start_h")
    assert sl.min == 0 and sl.max > 100
    at.slider(key="window_start_h").set_value(sl.max).run()
    _ok(at)


def test_staffing_section_hand_values():
    at = _run(staffing_load=10, staffing_target=1)
    _ok(at)
    assert _metric(at, "Nötige Spuren") == "12" and _metric(at, "Auslastung je Spur dabei") == "83 %"
    assert any("13 Spuren" in i.value and "eine Spur mehr als nötig" in i.value for i in at.info)
    small = _run(staffing_load=2, staffing_target=1)
    assert _metric(small, "Nötige Spuren") == "4" and any("zu wenig" in i.value for i in small.info)


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["c"] = "99"
    at.query_params["rho"] = "9999"
    at.query_params["n"] = "3000"
    at.run()
    _ok(at)
    assert at.session_state["c_slider"] == C.C_MAX and at.session_state["rho_slider"] == C.RHO_PCT_MAX
    assert at.session_state["n_select"] == 2000


def test_permalink_ignores_garbage():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["c"] = "viele"
    at.query_params["seed"] = "x"
    at.run()
    _ok(at)
    assert at.session_state["c_slider"] == C.DEFAULT_C and at.session_state["seed_input"] == C.DEFAULT_SEED


def test_charts_sections_and_limits_table_are_present():
    at = _run()
    _ok(at)
    assert len(at.get("plotly_chart")) == 8
    headers = [s.value for s in at.subheader]
    assert any("Pooling" in h for h in headers) and any("Wie viele Spuren" in h for h in headers)
    assert any("Intervall-Problem" in h for h in headers) and any("Wo die Annahmen enden" in h for h in headers)
    table = next(m.value for m in at.markdown if "Wer setzt an" in m.value)
    for name in ("Kingman", "Erlang B", "Erlang A", "Halfin-Whitt", "zeitvariable Ankünfte", "Power-of-d", "Prioritätsklassen"):
        assert name in table
    assert "geplant" not in table      # die Linie wird erst vollständig veröffentlicht, kein Status-Zusatz


def test_seed_control_uses_the_portfolio_wording():
    at = _run()
    assert [n.label for n in at.number_input] == ["Zufalls-Seed"]


def test_related_demos_are_linked_and_footer_is_present():
    at = _run()
    text = " ".join(c.value for c in at.caption)
    for name in ("ems-demo", "mm1-queue-demo", "output-analysis-demo"):
        assert name in text
    assert "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in text


def test_no_sentence_wide_comma_replacement_in_the_app_source():
    """Regressionsschutz: `.replace(",", ".")` auf einem ganzen (verketteten) Satz macht aus Kommas im Fließtext Punkte; Tausender
    nur über `fmt_int`."""
    source = Path(APP).read_text(encoding="utf-8")
    assert '.replace(",", ".")' not in source.replace('f"{n:,}".replace(",", ".")', "")
