"""JEDE Zahl aus README und App-Texten wird hier nachgerechnet: Formelwerte exakt, Messreihen-Zahlen aus der
vorgerechneten Datei (400 Läufe je Zelle, Standardfehler einer Abdeckung höchstens 2.5 Prozentpunkte), daher mit Marge
und nie auf einen einzelnen verrauschten Wert gepinnt."""

import pytest

import mmc_constants as C
import mmc_evaluation as E
import mmc_formulas as F
from mmc_simulation import MU, rates

PRE = E.load_precomputed()


def metrics(c, rho_pct):
    lam, mu = rates(c, rho_pct)
    return F.stationary_metrics(c, lam, mu)


def factor(c, rho_pct):
    lam, mu = rates(c, rho_pct)
    return F.pooling_factor(c, lam, mu)


def cell(c, rho, n):
    return E.coverage_cell(PRE, c, rho, n)["variants"]


def test_preset_help_numbers():
    """PRESET_HELP: 4 Spuren ρ 90 %: 79 %, 5.9 min, getrennt 27 min (4.6-fach); eine Spur 27 min; 16 Spuren 1.1 min und 24-fach;
    8 Spuren ρ 50 %: 5.9 %, 0.04 min, getrennt 3 min (68-fach)."""
    m4 = metrics(4, 90)
    assert m4["p_wait"] == pytest.approx(0.788, abs=0.0005) and m4["Wq"] == pytest.approx(5.91, abs=0.005)
    assert F.separate_wq(4, *rates(4, 90)) == pytest.approx(27.0) and factor(4, 90) == pytest.approx(4.57, abs=0.005)
    assert metrics(1, 90)["Wq"] == pytest.approx(27.0)
    assert metrics(16, 90)["Wq"] == pytest.approx(1.11, abs=0.005) and factor(16, 90) == pytest.approx(24.35, abs=0.005)
    m8 = metrics(8, 50)
    assert m8["p_wait"] == pytest.approx(0.059, abs=0.0005) and m8["Wq"] == pytest.approx(0.04, abs=0.005)
    assert F.separate_wq(8, *rates(8, 50)) == pytest.approx(3.0) and factor(8, 50) == pytest.approx(67.75, abs=0.05)


def test_pooling_factors_quoted_in_readme():
    """README: ρ = 90 %: c = 2: 2.1, 4: 4.6, 8: 10.3, 16: 24.4; ρ = 50 %: c = 2: 3.0, 4: 11.5, 8: 68; ρ = 95 %: c = 16: 19.5."""
    for c, expected in ((2, 2.11), (4, 4.57), (8, 10.26), (16, 24.35)):
        assert factor(c, 90) == pytest.approx(expected, abs=0.01)
    assert factor(2, 50) == pytest.approx(3.0) and factor(4, 50) == pytest.approx(11.5, abs=0.05)
    assert factor(16, 95) == pytest.approx(19.48, abs=0.01) and factor(1, 90) == pytest.approx(1.0)


def test_erlang_c_values_quoted_in_readme():
    """README: 4 Spuren: ρ = 80 %: 60 % warten, 2.2 min; 16 Spuren ρ = 95 %: 78 % warten, 2.9 min."""
    assert metrics(4, 80)["p_wait"] == pytest.approx(0.596, abs=0.0005) and metrics(4, 80)["Wq"] == pytest.approx(2.24, abs=0.005)
    assert metrics(16, 95)["p_wait"] == pytest.approx(0.780, abs=0.0005) and metrics(16, 95)["Wq"] == pytest.approx(2.93, abs=0.005)


def test_staffing_table_quoted_in_readme():
    """README (Ziel 1 min): a = 2 -> 4 Spuren (50 %), 5 -> 7 (71 %), 10 -> 12 (83 %), 20 -> 22 (91 %), 50 -> 53 (94 %),
    100 -> 103 (97 %), 200 -> 203 (98 %); die 80-%-Faustregel: 3 (zu wenig, 1.3 min), 7, 13, 25, 63, 125, 250."""
    expected = {2: (4, 0.50), 5: (7, 0.71), 10: (12, 0.83), 20: (22, 0.91), 50: (53, 0.94), 100: (103, 0.97), 200: (203, 0.985)}
    for a, (c, rho) in expected.items():
        row = E.staffing_row(a, 1)
        assert row["c"] == c and row["rho"] == pytest.approx(rho, abs=0.006), a
    rule = {a: E.staffing_row(a, 1)["c_rule"] for a in expected}
    assert rule == {2: 3, 5: 7, 10: 13, 20: 25, 50: 63, 100: 125, 200: 250}
    assert E.staffing_row(2, 1)["wq_rule"] == pytest.approx(1.33, abs=0.01)
    assert E.staffing_row(5, 1)["wq_rule"] <= 1.0


def test_extra_lanes_over_the_offered_load_grow_slowly():
    """README: Bei Ziel 1 min braucht a = 100 nur 3 Spuren mehr als das Angebot, a = 200 ebenfalls nur 3, a = 2 aber 2 mehr."""
    assert E.staffing_row(100, 1)["c"] - 100 == 3 and E.staffing_row(200, 1)["c"] - 200 == 3
    assert E.staffing_row(2, 1)["c"] - 2 == 2


def test_naive_interval_never_covers_more_than_about_half_in_any_cell():
    """README: in keiner der 120 Zellen (5 Spurzahlen × 4 Auslastungen × 6 Lauflängen) häufiger als 51 %."""
    assert len(PRE["coverage"]) == 120
    assert max(x["variants"]["naive"]["cover"] for x in PRE["coverage"]) < 0.56


def test_naive_coverage_at_90_percent_utilisation_is_the_same_for_every_lane_count():
    """README: bei ρ = 90 % und 10 000 Lkw trifft das naive Intervall für c = 1, 2, 4, 8, 16 in 7–11 % der Fälle."""
    covers = [cell(c, 90, 10000)["naive"]["cover"] for c in C.GRID_C]
    assert min(covers) > 0.04 and max(covers) < 0.14


def test_batch_means_coverage_quoted_in_readme():
    """README: Batch Means (20) bei ρ = 90 %, 10 000 Lkw: 81–84 % für alle Spurzahlen; bei ρ = 95 %, 50 000 Lkw: 83–87 %;
    5 Batches sind in 117 von 120 Zellen mindestens so gut wie 20; bei 50 000 Lkw und ρ ≤ 90 % trifft 5 Batches in ≥ 91 %."""
    c90 = [cell(c, 90, 10000)["batch20"]["cover"] for c in C.GRID_C]
    assert min(c90) > 0.77 and max(c90) < 0.88
    c95 = [cell(c, 95, 50000)["batch20"]["cover"] for c in C.GRID_C]
    assert min(c95) > 0.79 and max(c95) < 0.90
    ge = sum(1 for x in PRE["coverage"] if x["variants"]["batch5"]["cover"] >= x["variants"]["batch20"]["cover"] - 0.02)
    assert ge >= 112
    assert min(cell(c, rho, 50000)["batch5"]["cover"] for c in C.GRID_C for rho in (50, 80, 90)) > 0.87


def test_run_to_run_spread_does_not_shrink_with_more_lanes():
    """README: Streuung eines Laufs (rel. Std. der Wartezeit) bei ρ = 90 %, 10 000 Lkw: 21 % (c = 1), 21 %, 21 %, 24 % (c = 8),
    31 % (c = 16); sie wächst mit der Spurzahl, schrumpft nie."""
    spread = [cell(c, 90, 10000)["naive"]["rel_std"] for c in C.GRID_C]
    assert spread[0] == pytest.approx(0.208, abs=0.05) and spread[-1] == pytest.approx(0.308, abs=0.06)
    assert spread[-1] > spread[0] and min(spread) > 0.15
    assert round(100 * spread[-1]) == 31


def test_start_bias_at_short_high_utilisation_runs():
    """README: ρ = 95 %, 1 000 Lkw: Mittelwert 27–33 % unter der Formel für alle Spurzahlen; bei 50 000 Lkw im Rauschen."""
    for c in C.GRID_C:
        assert -40 < cell(c, 95, 1000)["naive"]["bias_pct"] < -20
        assert abs(cell(c, 95, 50000)["naive"]["bias_pct"]) < 4
