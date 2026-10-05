"""Orakel mit anderem Rechenweg: (1) Erlang B/C in exakter Bruchrechnung (Summenformel, auch für c ≈ 200), (2) P(Wq > t) als Erlang-Mischung über die abgeschnittene Kette,
(3) Spurbedarf per Brute Force, (4) die Ereignissimulation und die getrennten Schlangen gegen eine Kunde-für-Kunde-Rechnung (frühester freier Server per Listensuche, kein Heap, keine Ereignisliste)."""

import math
import random
from fractions import Fraction

import numpy as np
import pytest

import mmc_evaluation as E
import mmc_formulas as F
import mmc_simulation as S


def _ctmc(c, lam, mu, k_max):
    n = k_max + 1
    q = np.zeros((n, n))
    for i in range(n - 1):
        q[i, i + 1] = lam
        q[i + 1, i] = min(i + 1, c) * mu
    np.fill_diagonal(q, -q.sum(axis=1))
    a = np.vstack([q.T[:-1], np.ones(n)])
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(a, b)


@pytest.mark.parametrize("c,a", [(1, Fraction(1, 3)), (7, Fraction(13, 2)), (50, Fraction(45)), (100, Fraction(97)), (203, Fraction(200))])
def test_erlang_b_and_c_equal_exact_rational_arithmetic(c, a):
    b = a ** c / math.factorial(c) / sum(a ** j / math.factorial(j) for j in range(c + 1))
    assert F.erlang_b(c, float(a)) == pytest.approx(float(b), rel=1e-10)
    if a < c:
        rho = a / c
        assert F.erlang_c(c, float(a)) == pytest.approx(float(b / (1 - rho * (1 - b))), rel=1e-10)


def test_wait_tail_equals_the_erlang_mixture_over_the_chain():
    """Wer n ≥ c Lkw im System findet, wartet Erlang(n − c + 1, cμ)-verteilt: P(Wq > t) = Σ_{n≥c} π_n · P(Poisson(cμt) ≤ n − c)."""
    rng = random.Random(2)
    for _ in range(12):
        c, mu, rho = rng.randint(1, 12), rng.uniform(0.2, 2.0), rng.uniform(0.2, 0.85)
        lam = rho * c * mu
        pi = _ctmc(c, lam, mu, c + max(130, int(40 / (1 - rho))))
        assert pi[-5:].sum() < 1e-12
        for t in (0.0, 1.0 / mu, 4.0 / mu):
            x = c * mu * t
            cum = np.cumsum([math.exp(-x) * x ** k / math.factorial(k) for k in range(120)])
            surv = sum(pi[n] * cum[n - c] for n in range(c, c + 120)) + pi[c + 120:].sum()
            assert F.prob_wait_exceeds(c, lam, mu, t) == pytest.approx(surv, abs=1e-8)


def test_min_servers_equals_a_brute_force_search():
    mu = 1 / 3

    def wq(c, a):
        b = 1.0
        for k in range(1, c + 1):
            b = a * b / (k + a * b)
        rho = a / c
        return b / (1 - rho * (1 - b)) / (c * mu - a * mu)

    for a in (0.5, 1.5, 2, 3.7, 5, 10, 12.3, 20, 50, 77.7, 100, 200):
        for target in (0.1, 0.5, 1, 2, 5):
            c = F.min_servers(a, mu, target)
            assert c > a and wq(c, a) <= target and (c - 1 <= a or wq(c - 1, a) > target)
            assert E.staffing_row(a, target)["c"] == c


def test_rule_of_thumb_is_the_exact_ceiling_of_the_ratio():
    rng = random.Random(3)
    for _ in range(400):
        a = Fraction(rng.randint(1, 4000), rng.choice([1, 2, 5, 10, 100]))
        rho = rng.choice([Fraction(7, 10), Fraction(4, 5), Fraction(9, 10), Fraction(3, 4), Fraction(3, 5), Fraction(1, 2)])
        assert F.rule_of_thumb_servers(float(a), float(rho)) == math.ceil(a / rho)


def _naive_shared(c, lam, mu, n, seed):
    gap, svc, _ = S.streams(seed)
    arr, t = [], 0.0
    for _ in range(n):
        t += gap.expovariate(lam)
        arr.append(t)
    service = [svc.expovariate(mu) for _ in range(n)]
    free, start, dep = [0.0] * c, [], []
    for k in range(n):
        i = min(range(c), key=lambda j: free[j])
        s = max(arr[k], free[i])
        free[i] = s + service[k]
        start.append(s)
        dep.append(free[i])
    return arr, start, dep


def test_event_simulation_equals_a_customer_by_customer_recursion():
    rng = random.Random(7)
    for it in range(40):
        c, n, mu = rng.randint(1, 6), rng.randint(1, 40), rng.uniform(0.2, 2.0)
        lam, seed = rng.uniform(0.1, 1.3) * c * mu, rng.randint(0, 10 ** 6)
        sim = S.simulate(c, lam, mu, n, seed, record=True)
        arr, start, dep = _naive_shared(c, lam, mu, n, seed)
        assert sim.waits == pytest.approx([s - a for s, a in zip(start, arr)], abs=1e-9)
        assert sim.end_time == pytest.approx(max(dep))
        events = sorted([(a, 1) for a in arr] + [(d, -1) for d in dep])
        cur, t0, area, area_q, busy, spent = 0, 0.0, 0.0, 0.0, 0.0, {}
        for t, delta in events:
            dt = t - t0
            area += cur * dt
            area_q += max(cur - c, 0) * dt
            busy += min(cur, c) * dt
            spent[cur] = spent.get(cur, 0.0) + dt
            cur, t0 = cur + delta, t
        assert (sim.area_in_system, sim.area_in_queue, sim.busy_integral) == pytest.approx((area, area_q, busy), rel=1e-9, abs=1e-9)
        for state, value in spent.items():
            assert sim.time_in_state.get(state, 0.0) == pytest.approx(value, abs=1e-9)
        g, s, _ = S.streams(seed)
        assert S.kw_waits(c, lam, mu, n, g, s) == pytest.approx(sim.waits, abs=1e-9)


def test_separate_queues_equal_a_customer_by_customer_recursion():
    rng = random.Random(11)
    for _ in range(30):
        c, n, mu = rng.randint(1, 6), rng.randint(1, 40), rng.uniform(0.2, 2.0)
        lam, seed = rng.uniform(0.1, 1.3) * c * mu, rng.randint(0, 10 ** 6)
        g, s, r = S.streams(seed)
        got = S.separate_waits(c, lam, mu, n, g, s, r)
        g, s, r = S.streams(seed)
        arr, t = [], 0.0
        for _ in range(n):
            t += g.expovariate(lam)
            arr.append(t)
        lanes = [min(int(r.uniform() * c), c - 1) for _ in range(n)]
        service = [s.expovariate(mu) for _ in range(n)]
        free, want = [0.0] * c, []
        for k in range(n):
            st = max(arr[k], free[lanes[k]])
            want.append(st - arr[k])
            free[lanes[k]] = st + service[k]
        assert got == pytest.approx(want, abs=1e-9)
