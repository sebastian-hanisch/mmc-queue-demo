"""Geschlossene Formeln der M/M/c-Schlange (Poisson-Ankünfte, exponentielle Abfertigung je Spur, c gleich schnelle Spuren,
EINE gemeinsame FIFO-Schlange, unbegrenzt). Zeiten in Minuten, Raten je Minute. Nur gültig für ρ = λ/(cμ) < 1."""

import math


def offered_load(lam, mu):
    """Angebot a = λ/μ: im Mittel so viele Spuren sind gleichzeitig mit Abfertigen beschäftigt (in Erlang)."""
    return lam / mu


def utilisation(c, lam, mu):
    """Auslastung je Spur ρ = λ/(cμ)."""
    return lam / (c * mu)


def erlang_b(c, a):
    """Erlang-B-Verlustwahrscheinlichkeit B(c, a) über die stabile Rekursion B_k = a·B_{k-1}/(k + a·B_{k-1}). Hier nur
    Zwischenschritt für Erlang C."""
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c, a):
    """Erlang C: Wahrscheinlichkeit, dass ein ankommender Lkw warten muss (alle c Spuren belegt), C = B/(1 − ρ(1 − B))
    mit ρ = a/c < 1."""
    rho = a / c
    if rho >= 1:
        raise ValueError("ρ ≥ 1: keine stationäre Verteilung")
    b = erlang_b(c, a)
    return b / (1 - rho * (1 - b))


def stationary_metrics(c, lam, mu):
    """Gleichgewichtskennzahlen: Wahrscheinlichkeit zu warten (Erlang C), mittlere Wartezeit Wq = C/(cμ−λ), Verweilzeit
    W = Wq + 1/μ, mittlere Schlangenlänge Lq = λ·Wq und mittlere Zahl im System L = Lq + a (Little's Gesetz)."""
    a = lam / mu
    p_wait = erlang_c(c, a)
    wq = p_wait / (c * mu - lam)
    return {"c": c, "rho": a / c, "a": a, "p_wait": p_wait, "Wq": wq, "W": wq + 1 / mu, "Lq": lam * wq, "L": lam * wq + a}


def p_n(c, lam, mu, n):
    """P(N = n): Verteilung der Zahl im System (Geburts-Sterbe-Kette mit Sterberate min(n, c)·μ)."""
    a = lam / mu
    rho = a / c
    if rho >= 1:
        raise ValueError("ρ ≥ 1: keine stationäre Verteilung")
    head = sum(a ** k / math.factorial(k) for k in range(c))
    tail = a ** c / (math.factorial(c) * (1 - rho))
    p0 = 1.0 / (head + tail)
    if n <= c:
        return p0 * a ** n / math.factorial(n)
    return p0 * a ** c / math.factorial(c) * rho ** (n - c)       # p_c · ρ^(n−c), ohne riesige Zwischenzahlen


def prob_wait_exceeds(c, lam, mu, t):
    """P(Wq > t) = C · e^{−(cμ−λ)t}: Anteil der Lkw mit Wartezeit über t Minuten (t ≥ 0)."""
    return erlang_c(c, lam / mu) * math.exp(-(c * mu - lam) * t)


def separate_wq(c, lam, mu):
    """Mittlere Wartezeit, wenn jede der c Spuren ihre EIGENE Schlange hat und ankommende Lkw zufällig verteilt werden:
    c unabhängige M/M/1-Schlangen mit Rate λ/c, Wq = ρ/(μ − λ/c)."""
    lam_lane = lam / c
    if lam_lane >= mu:
        raise ValueError("ρ ≥ 1: keine stationäre Verteilung")
    return (lam_lane / mu) / (mu - lam_lane)


def pooling_factor(c, lam, mu):
    """Wie viel mal länger ist die mittlere Wartezeit bei c getrennten Schlangen als bei einer gemeinsamen?"""
    return separate_wq(c, lam, mu) / stationary_metrics(c, lam, mu)["Wq"]


def min_servers(a, mu, target_wq):
    """Kleinste Spurzahl c > a, bei der die mittlere Wartezeit höchstens `target_wq` Minuten beträgt (Angebot a)."""
    c = int(a) + 1
    while stationary_metrics(c, a * mu, mu)["Wq"] > target_wq:
        c += 1
    return c


def rule_of_thumb_servers(a, max_rho=0.8):
    """Faustregel: so viele Spuren, dass die Auslastung höchstens `max_rho` beträgt."""
    return math.ceil(a / max_rho)
