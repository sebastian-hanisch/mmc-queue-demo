"""Simulation: Generator, Drei-Lkw-Instanz an zwei Spuren von Hand, Ereignissimulation gegen Kiefer-Wolfowitz-Rekursion,
Little's Gesetz als Pfadidentität, getrennte Schlangen."""

import pytest

import mmc_formulas as F
import mmc_simulation as S
from conftest import ScriptedRng


def test_splitmix64_matches_the_reference_sequence():
    rng = S.SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_streams_are_distinct_and_rates_follow_utilisation_per_lane():
    g, s, r = S.streams(7)
    assert len({g.state, s.state, r.state}) == 3
    lam, mu = S.rates(4, 90)
    assert mu == pytest.approx(1 / 3) and lam == pytest.approx(0.9 * 4 / 3)


def test_mini_instance_by_hand(mini_streams):
    """Von Hand (siehe conftest): Wartezeiten 0 / 0 / 1.5 / 1, Verweilzeiten 3 / 2 / 2.5 / 2, Ende 5, ∫N dt = 9.5, ∫Nq dt = 2.5,
    beschäftigte Spuren ∫ = 7 (Spur A 4 min, Spur B 3 min), Zeit je Zustand {0: 1, 1: 1, 2: 1, 3: 1.5, 4: 0.5}."""
    gap, svc = mini_streams
    r = S.simulate(2, 1.0, 1.0, 4, seed=0, record=True, gap_rng=gap, svc_rng=svc)
    assert r.waits == pytest.approx([0.0, 0.0, 1.5, 1.0])
    assert r.sojourns == pytest.approx([3.0, 2.0, 2.5, 2.0])
    assert r.end_time == pytest.approx(5.0)
    assert r.area_in_system == pytest.approx(9.5) and r.area_in_queue == pytest.approx(2.5)
    assert r.busy_integral == pytest.approx(7.0) and r.utilisation == pytest.approx(0.7)
    assert r.time_in_state == pytest.approx({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.5, 4: 0.5})
    assert [n for _, n in r.trajectory] == [0, 1, 2, 3, 4, 3, 2, 1, 0]
    assert r.share_waiting == pytest.approx(0.5) and r.mean_in_system == pytest.approx(9.5 / 5)


def test_mini_instance_kiefer_wolfowitz_gives_the_same_waits(mini_streams):
    gap, svc = mini_streams
    assert S.kw_waits(2, 1.0, 1.0, 4, gap, svc) == pytest.approx([0.0, 0.0, 1.5, 1.0])


def test_mini_instance_separate_queues_by_hand():
    """Zuteilung zu Spur 0 / 1 / 0 / 0 (Zufallszahlen 0.1 / 0.9 / 0.2 / 0.3): Spur 0 bedient Lkw 1 (1-4), Lkw 3 wartet bis 4
    (Wartezeit 2, Abgang 5), Lkw 4 wartet bis 5 (Wartezeit 2); Spur 1 bedient Lkw 2 sofort -> Wartezeiten 0 / 0 / 2 / 2."""
    gap, svc = ScriptedRng(exp_values=[1, 0.5, 0.5, 1]), ScriptedRng(exp_values=[3, 2, 1, 1])
    route = ScriptedRng(uniform_values=[0.1, 0.9, 0.2, 0.3])
    assert S.separate_waits(2, 1.0, 1.0, 4, gap, svc, route) == pytest.approx([0.0, 0.0, 2.0, 2.0])


@pytest.mark.parametrize("c,rho", [(1, 90), (2, 80), (4, 90), (8, 95), (16, 70)])
@pytest.mark.parametrize("seed", [1, 2])
def test_event_simulation_and_kiefer_wolfowitz_agree_customer_by_customer(c, rho, seed):
    lam, mu = S.rates(c, rho)
    sim = S.simulate(c, lam, mu, 3000, seed)
    g, s, _ = S.streams(seed)
    kw = S.kw_waits(c, lam, mu, 3000, g, s)
    assert max(abs(a - b) for a, b in zip(sim.waits, kw)) < 1e-6


def test_one_lane_matches_the_lindley_recursion():
    """Mit c = 1 reduziert sich Kiefer-Wolfowitz auf W' = max(0, W + S − A)."""
    lam, mu = S.rates(1, 90)
    g, s, _ = S.streams(5)
    kw = S.kw_waits(1, lam, mu, 2000, g, s)
    g, s, _ = S.streams(5)
    w, prev, out = 0.0, 0.0, []
    for k in range(2000):
        gap = g.expovariate(lam)
        if k > 0:
            w = max(0.0, w + prev - gap)
        out.append(w)
        prev = s.expovariate(mu)
    assert max(abs(a - b) for a, b in zip(kw, out)) < 1e-9


@pytest.mark.parametrize("c,rho", [(1, 90), (3, 80), (8, 97)])
def test_littles_law_holds_exactly_on_every_path(c, rho):
    """Das System endet leer: ∫N dt = Σ Verweilzeiten (jeder Kunde trägt seine Verweilzeit bei)."""
    lam, mu = S.rates(c, rho)
    sim = S.simulate(c, lam, mu, 4000, 7)
    assert sim.area_in_system == pytest.approx(sum(sim.sojourns), rel=1e-9)
    assert sim.mean_in_system == pytest.approx(sim.arrival_rate * sim.mean_sojourn, rel=1e-9)


def test_invariants_of_a_long_run():
    lam, mu = S.rates(4, 85)
    sim = S.simulate(4, lam, mu, 4000, 3)
    assert min(sim.waits) >= 0.0 and all(s >= w for s, w in zip(sim.sojourns, sim.waits))
    assert sum(sim.time_in_state.values()) == pytest.approx(sim.end_time)
    assert max(sim.time_in_state) >= 4 and 0 < sim.utilisation < 1
    assert sim.share_waiting == pytest.approx(sum(1 for w in sim.waits if w > 0) / 4000)


def test_same_seed_same_result_different_seed_different_result():
    lam, mu = S.rates(3, 80)
    a, b, c_ = S.simulate(3, lam, mu, 500, 11), S.simulate(3, lam, mu, 500, 11), S.simulate(3, lam, mu, 500, 12)
    assert a.waits == b.waits and a.waits != c_.waits


def test_long_run_matches_the_erlang_c_formula():
    """c = 4, ρ = 80 %: ein langer Lauf (200 000 Lkw) liegt innerhalb weniger Prozent an der Formel."""
    lam, mu = S.rates(4, 80)
    g, s, _ = S.streams(1)
    waits = S.kw_waits(4, lam, mu, 200_000, g, s)
    assert sum(waits) / len(waits) == pytest.approx(F.stationary_metrics(4, lam, mu)["Wq"], rel=0.1)
    assert sum(1 for w in waits if w > 0) / len(waits) == pytest.approx(F.erlang_c(4, lam / mu), abs=0.02)


def test_separate_queues_wait_much_longer_than_one_shared_queue():
    """Zweig-Test: die getrennte Variante weicht von der gemeinsamen ab (Pooling-Effekt, hier Faktor 5 laut Formel)."""
    lam, mu = S.rates(4, 80)
    shared, separate = [], []
    for seed in range(1, 6):
        g, s, r = S.streams(seed)
        shared.append(sum(S.kw_waits(4, lam, mu, 20000, g, s)) / 20000)
        g, s, r = S.streams(seed)
        separate.append(sum(S.separate_waits(4, lam, mu, 20000, g, s, r)) / 20000)
    assert sum(separate) > 3 * sum(shared)
