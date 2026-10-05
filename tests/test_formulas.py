"""Formeln gegen Handrechnung und gegen eine UNABHÄNGIGE Referenz: das lineare Gleichungssystem der abgeschnittenen
Geburts-Sterbe-Kette mit Sterberate min(n, c)·μ (kein Gebrauch der geschlossenen Erlang-C-Formel)."""

import math

import numpy as np
import pytest

import mmc_formulas as F


def _ctmc_stationary(c, lam, mu, k_max=600):
    """Stationäre Verteilung der auf 0..k_max abgeschnittenen Kette: π Q = 0, Σπ = 1 (lineares Gleichungssystem)."""
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


@pytest.mark.parametrize("c,rho", [(1, 0.8), (2, 0.5), (3, 0.9), (4, 0.7), (8, 0.9), (12, 0.95)])
def test_erlang_c_formulas_match_the_ctmc_reference(c, rho):
    mu = 1.0
    lam = rho * c * mu
    pi = _ctmc_stationary(c, lam, mu)
    states = np.arange(len(pi))
    m = F.stationary_metrics(c, lam, mu)
    assert pi[c:].sum() == pytest.approx(m["p_wait"], rel=1e-6)                         # P(warten) = P(N ≥ c)
    assert (pi * states).sum() == pytest.approx(m["L"], rel=1e-6)
    assert (pi[c:] * (states[c:] - c)).sum() == pytest.approx(m["Lq"], rel=1e-6)
    for n in (0, 1, c, c + 1, c + 5):
        assert pi[n] == pytest.approx(F.p_n(c, lam, mu, n), rel=1e-6)
    assert m["L"] == pytest.approx(lam * m["W"], rel=1e-12)                              # Little (Gegenprobe)
    assert m["Lq"] == pytest.approx(lam * m["Wq"], rel=1e-12)


def test_waiting_time_tail_matches_the_ctmc_reference_at_t_zero_and_by_hand():
    """P(Wq > 0) = C; P(Wq > t) = C·e^{−(cμ−λ)t}."""
    c, lam, mu = 3, 2.4, 1.0
    assert F.prob_wait_exceeds(c, lam, mu, 0.0) == pytest.approx(F.erlang_c(c, lam / mu))
    assert F.prob_wait_exceeds(c, lam, mu, 2.0) == pytest.approx(F.erlang_c(c, lam / mu) * math.exp(-0.6 * 2.0))


def test_hand_values_two_servers():
    """B(1, 1) = 1/2, B(2, 1) = 0.5/(2 + 0.5) = 0.2, C(2, 1) = 0.2/(1 − 0.5·0.8) = 1/3; mit μ = 1, λ = 1: Wq = (1/3)/(2 − 1)."""
    assert F.erlang_b(1, 1.0) == pytest.approx(0.5) and F.erlang_b(2, 1.0) == pytest.approx(0.2)
    assert F.erlang_c(2, 1.0) == pytest.approx(1 / 3)
    m = F.stationary_metrics(2, 1.0, 1.0)
    assert m["Wq"] == pytest.approx(1 / 3) and m["L"] == pytest.approx(1.0 / 3 + 1.0) and m["W"] == pytest.approx(4 / 3)


@pytest.mark.parametrize("c", range(1, 21))
def test_erlang_c_matches_the_textbook_sum_formula(c):
    a = 0.85 * c
    rho = a / c
    head = sum(a ** k / math.factorial(k) for k in range(c))
    tail = a ** c / (math.factorial(c) * (1 - rho))
    assert F.erlang_c(c, a) == pytest.approx(tail / (head + tail), rel=1e-9)


def test_one_server_reduces_to_mm1():
    lam, mu = 0.3, 1 / 3
    m = F.stationary_metrics(1, lam, mu)
    assert m["p_wait"] == pytest.approx(0.9) and m["Wq"] == pytest.approx(0.9 / (mu - lam)) and m["L"] == pytest.approx(9.0)


def test_overload_has_no_stationary_distribution():
    for fn in (lambda: F.erlang_c(2, 2.0), lambda: F.stationary_metrics(2, 2.5, 1.0), lambda: F.p_n(3, 3.0, 1.0, 1),
               lambda: F.separate_wq(2, 2.0, 1.0)):
        with pytest.raises(ValueError):
            fn()


def test_probabilities_sum_to_one():
    c, lam, mu = 5, 4.2, 1.0
    assert sum(F.p_n(c, lam, mu, n) for n in range(0, 800)) == pytest.approx(1.0, rel=1e-9)


def test_separate_queues_and_pooling_factor_by_hand():
    """c = 2, λ = μ = 1: je Spur λ/2 = 0.5, ρ = 0.5, Wq = 0.5/(1 − 0.5) = 1; gemeinsam Wq = 1/3 -> Faktor 3 (Tabelle der Vorab-Messreihe)."""
    assert F.separate_wq(2, 1.0, 1.0) == pytest.approx(1.0)
    assert F.pooling_factor(2, 1.0, 1.0) == pytest.approx(3.0)
    assert F.pooling_factor(1, 0.5, 1.0) == pytest.approx(1.0)


def test_pooling_factor_grows_with_the_number_of_servers_at_fixed_utilisation():
    for rho in (0.5, 0.8, 0.95):
        factors = [F.pooling_factor(c, rho * c, 1.0) for c in range(1, 17)]
        assert factors[0] == pytest.approx(1.0) and all(b > a for a, b in zip(factors, factors[1:]))


def test_min_servers_is_minimal_and_feasible():
    mu = 1 / 3
    for a in (2, 5, 10, 20, 50):
        c = F.min_servers(a, mu, 1.0)
        assert F.stationary_metrics(c, a * mu, mu)["Wq"] <= 1.0
        assert c - 1 <= a or F.stationary_metrics(c - 1, a * mu, mu)["Wq"] > 1.0


def test_rule_of_thumb_by_hand():
    assert F.rule_of_thumb_servers(10) == 13 and F.rule_of_thumb_servers(2) == 3 and F.rule_of_thumb_servers(100) == 125


def test_rule_of_thumb_does_not_round_up_an_exactly_integer_ratio():
    """Exakte Bruchrechnung als Orakel: 2,1 / 0,7 = 3 genau, in Gleitkomma 3,0000000000000004 (früher 4 Spuren)."""
    from fractions import Fraction

    cases = [(2.1, 0.7), (4.2, 0.7), (6.0, 0.6), (9.0, 0.9), (3.0, 0.75), (10, 0.8), (8, 0.8), (5.6, 0.8)]
    for a, rho in cases:
        exact = math.ceil(Fraction(str(a)) / Fraction(str(rho)))
        assert F.rule_of_thumb_servers(a, rho) == exact, (a, rho)
    assert F.rule_of_thumb_servers(2.1, 0.7) == 3 and F.rule_of_thumb_servers(2.1000001, 0.7) == 4


def test_offered_load_and_utilisation():
    assert F.offered_load(0.9, 0.3) == pytest.approx(3.0) and F.utilisation(4, 0.9, 0.3) == pytest.approx(0.75)
