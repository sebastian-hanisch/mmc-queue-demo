"""Auswertung: Mini-Instanz von Hand, Pooling-Vergleich, Personalbedarf, kleine Abdeckungsstudie, Vollständigkeit der
vorgerechneten Datei."""

import pytest

import mmc_constants as C
import mmc_evaluation as E
import mmc_formulas as F
import mmc_simulation as S
from conftest import ScriptedRng


@pytest.fixture
def mini(mini_streams):
    gap, svc = mini_streams
    return S.simulate(2, 1.0, 1.0, 4, seed=0, record=True, gap_rng=gap, svc_rng=svc)


def test_little_check_on_the_mini_instance(mini):
    l_hat, lw, gap = E.little_check(mini)
    assert l_hat == pytest.approx(9.5 / 5) and lw == pytest.approx(9.5 / 5) and gap < 1e-12


def test_state_distribution_on_the_mini_instance(mini):
    shares, rest = E.state_distribution(mini, max_state=4)
    assert shares == pytest.approx([0.2, 0.2, 0.2, 0.3, 0.1]) and rest == pytest.approx(0.0, abs=1e-12)
    shares2, rest2 = E.state_distribution(mini, max_state=2)
    assert rest2 == pytest.approx(0.4) and sum(shares2) + rest2 == pytest.approx(1.0)


def test_wait_tail_and_survival_curve_by_hand(mini):
    """Wartezeiten 0 / 0 / 1.5 / 1."""
    assert E.wait_tail_fractions(mini.waits, thresholds=(0.5, 1.2, 2.0)) == pytest.approx({0.5: 0.5, 1.2: 0.25, 2.0: 0.0})
    ts, surv = E.wait_survival_curve(mini.waits, t_max=2.0, n_points=5)
    assert ts == pytest.approx([0, 0.5, 1.0, 1.5, 2.0])
    assert surv == pytest.approx([0.5, 0.5, 0.25, 0.0, 0.0])


def test_window_steps_on_the_mini_trajectory(mini):
    """Treppe (t, n): (0,0) (1,1) (1.5,2) (2,3) (3,4) (3.5,3) (4,2) (4.5,1) (5,0)."""
    assert E.window_steps(mini.trajectory, 2.0, 4.0) == [(2.0, 3), (3.0, 4), (3.5, 3), (4.0, 2), (4.0, 2)]
    assert E.window_steps(mini.trajectory, 0.0, 1.2)[0] == (0.0, 0)


def test_run_live_records_the_trajectory():
    sim = E.run_live(4, 90, 500, 3)
    assert sim.c == 4 and sim.trajectory[0] == (0.0, 0) and len(sim.trajectory) == 2 * 500 + 1


def test_compare_pooling_formulas_and_branch():
    out = E.compare_pooling(4, 80, 20000, 3)
    assert out["formula_factor"] == pytest.approx(F.pooling_factor(4, *S.rates(4, 80)), rel=1e-12)
    assert out["sim_separate"] > out["sim_shared"] * 2                      # Zweig: die getrennte Variante läuft wirklich
    assert out["sim_share_waiting_separate"] > out["sim_share_waiting_shared"]


def test_compare_pooling_with_one_lane_has_no_effect():
    out = E.compare_pooling(1, 80, 5000, 3)
    assert out["formula_factor"] == pytest.approx(1.0) and out["sim_shared"] == pytest.approx(out["sim_separate"])


def test_staffing_rows_by_hand_and_rule_of_thumb():
    """a = 2, Ziel 1 min: 4 Spuren (ρ = 50 %, Wq = 0.26 min); die 80-%-Faustregel gibt nur 3 Spuren und verfehlt das Ziel."""
    row = E.staffing_row(2, 1)
    assert row["c"] == 4 and row["rho"] == pytest.approx(0.5) and row["wq"] == pytest.approx(0.26, abs=0.005)
    assert row["c_rule"] == 3 and row["wq_rule"] > 1
    big = E.staffing_row(100, 1)
    assert big["c"] == 103 and big["c_rule"] == 125 and big["wq_rule"] < 0.01


def test_staffing_needs_more_lanes_for_a_tighter_target():
    for a in C.STAFFING_LOADS:
        needed = [E.staffing_row(a, t)["c"] for t in sorted(C.STAFFING_TARGETS)]
        assert needed == sorted(needed, reverse=True) and needed[0] >= needed[-1]


def test_coverage_study_small_cell_structure_and_ordering():
    cell = E.coverage_study(4, 90, 1000, 60, seed_base=11)
    v = cell["variants"]
    assert set(v) == {"naive", "batch5", "batch20"} and cell["reps"] == 60
    assert cell["truth"] == pytest.approx(F.stationary_metrics(4, *S.rates(4, 90))["Wq"])
    for entry in v.values():
        assert 0.0 <= entry["cover"] <= 1.0 and entry["rel_half"] > 0 and entry["rel_std"] > 0 and entry["mean_wq"] > 0
    assert v["naive"]["cover"] < v["batch5"]["cover"] - 0.3
    assert v["naive"]["rel_half"] < v["batch20"]["rel_half"]


def test_nearest_ties_go_to_the_smaller_value():
    assert E.nearest(C.GRID_RHO_PCT, 65) == 50 and E.nearest(C.GRID_RHO_PCT, 66) == 80 and E.nearest(C.GRID_RHO_PCT, 97) == 95
    assert E.nearest(C.GRID_C, 3) == 2 and E.nearest(C.GRID_C, 12) == 8 and E.nearest(C.GRID_C, 13) == 16


def test_precomputed_file_is_complete():
    pre = E.load_precomputed()
    assert {(x["c"], x["rho_pct"], x["n"]) for x in pre["coverage"]} == {
        (c, r, n) for c in C.GRID_C for r in C.GRID_RHO_PCT for n in C.GRID_N}
    assert all(x["reps"] == pre["grid_reps"] == C.GRID_REPS for x in pre["coverage"])
    for x in pre["coverage"]:
        assert set(x["variants"]) == {"naive", "batch5", "batch20"}
        assert x["truth"] == pytest.approx(F.stationary_metrics(x["c"], *S.rates(x["c"], x["rho_pct"]))["Wq"])
