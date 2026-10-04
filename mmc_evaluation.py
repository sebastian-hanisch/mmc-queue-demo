"""Auswertung der Simulation gegen die Formeln: Kennzahlen, Verteilungen, Little-Gegenprobe, gemeinsam gegen getrennte
Schlangen, Personalbedarf und die Abdeckungsstudie der Intervalle je Spurzahl. Die teure Messreihe (Hunderte Läufe je
Zelle) steht vorgerechnet in `precomputed_sweep.json` (Generator: generate_precomputed.py, Laden: `load_precomputed`)."""

import json
import statistics
from pathlib import Path

import mmc_constants as C
import mmc_estimators as E
import mmc_formulas as F
from mmc_simulation import MU, kw_waits, rates, separate_waits, simulate, streams

PRECOMPUTED_PATH = Path(__file__).resolve().parent / "precomputed_sweep.json"


def run_live(c, rho_pct, n_customers, seed):
    lam, mu = rates(c, rho_pct)
    return simulate(c, lam, mu, n_customers, seed, record=True)


def little_check(sim):
    """Little's Gesetz auf dem simulierten Pfad: (L aus ∫N dt / T, λ̂ · Ŵ, relative Abweichung). Gilt exakt, weil jeder
    Lauf mit leerem System endet - jeder Kunde trägt genau seine Verweilzeit zur Fläche unter N(t) bei."""
    l_hat = sim.mean_in_system
    lw = sim.arrival_rate * sim.mean_sojourn
    return l_hat, lw, abs(l_hat - lw) / l_hat if l_hat else 0.0


def state_distribution(sim, max_state=C.MAX_STATE_SHOWN):
    """Anteil der Zeit mit genau n im System (n = 0 … max_state); der Rest darüber wird zusammengefasst."""
    total = sum(sim.time_in_state.values())
    shares = [sim.time_in_state.get(n, 0.0) / total for n in range(max_state + 1)]
    return shares, max(0.0, 1.0 - sum(shares))


def wait_tail_fractions(waits, thresholds=C.TAIL_MINUTES):
    """Anteil der Lkw mit Wartezeit über t Minuten, für jedes t in `thresholds`."""
    n = len(waits)
    return {t: sum(1 for w in waits if w > t) / n for t in thresholds}


def wait_survival_curve(waits, t_max, n_points=60):
    """P(Wq > t) aus den simulierten Wartezeiten an gleichmäßigen Stützstellen 0 … t_max: ([t], [Anteil])."""
    ordered = sorted(waits)
    n = len(ordered)
    ts = [t_max * k / (n_points - 1) for k in range(n_points)]
    out, idx = [], 0
    for t in ts:
        while idx < n and ordered[idx] <= t:
            idx += 1
        out.append((n - idx) / n)
    return ts, out


def window_steps(trajectory, start_min, end_min):
    """Treppenkurve N(t) im Fenster [start_min, end_min] für `line_shape='hv'`: links mit dem Wert zum Fensterbeginn,
    rechts mit dem letzten Wert im Fenster abgeschlossen."""
    n_start = 0
    pts = []
    for t, n in trajectory:
        if t <= start_min:
            n_start = n
        elif t <= end_min:
            pts.append((t, n))
        else:
            break
    out = [(start_min, n_start)] + pts
    out.append((end_min, out[-1][1]))
    return out


def compare_pooling(c, rho_pct, n_customers, seed):
    """Dasselbe Gate zweimal mit denselben Ankunfts- und Bedienzeiten-Strömen: EINE gemeinsame Schlange gegen c getrennte
    Schlangen mit zufälliger Zuteilung. Rückgabe: Formelwerte, simulierte Mittel und der Pooling-Faktor (getrennt/gemeinsam)."""
    lam, mu = rates(c, rho_pct)
    g, s, r = streams(seed)
    shared = kw_waits(c, lam, mu, n_customers, g, s)
    g, s, r = streams(seed)
    separate = separate_waits(c, lam, mu, n_customers, g, s, r)
    f_shared = F.stationary_metrics(c, lam, mu)["Wq"]
    f_sep = F.separate_wq(c, lam, mu)
    return {"formula_shared": f_shared, "formula_separate": f_sep, "formula_factor": f_sep / f_shared,
            "sim_shared": statistics.fmean(shared), "sim_separate": statistics.fmean(separate),
            "sim_share_waiting_shared": sum(1 for w in shared if w > 0) / n_customers,
            "sim_share_waiting_separate": sum(1 for w in separate if w > 0) / n_customers}


def staffing_row(a, target_wq):
    """Kleinste Spurzahl für die Ziel-Wartezeit bei Angebot a, mit erreichter Auslastung und Erlang-C-Wahrscheinlichkeit,
    dazu die Faustregel „höchstens 80 % Auslastung“ und ihre mittlere Wartezeit."""
    c = F.min_servers(a, MU, target_wq)
    m = F.stationary_metrics(c, a * MU, MU)
    c_rule = F.rule_of_thumb_servers(a, C.RULE_OF_THUMB_MAX_RHO)
    wq_rule = F.stationary_metrics(c_rule, a * MU, MU)["Wq"] if c_rule > a else None
    return {"a": a, "target_wq": target_wq, "c": c, "rho": m["rho"], "p_wait": m["p_wait"], "wq": m["Wq"],
            "c_rule": c_rule, "wq_rule": wq_rule}


def _summarise(point_estimates, covers_flags, half_widths, truth):
    mean_est = statistics.fmean(point_estimates)
    return {"cover": sum(covers_flags) / len(covers_flags), "bias_pct": 100.0 * (mean_est - truth) / truth,
            "rel_std": statistics.stdev(point_estimates) / truth, "rel_half": statistics.fmean(half_widths) / truth,
            "mean_wq": mean_est}


def coverage_study(c, rho_pct, n, reps, seed_base):
    """Über `reps` unabhängige Läufe (Kiefer-Wolfowitz, leerer Start): wie oft enthält das 95-%-Intervall den wahren Wert
    (Erlang C), wie verzerrt und wie breit ist es, für das naive Intervall und Batch Means mit 5 und 20 Batches."""
    lam, mu = rates(c, rho_pct)
    truth = F.stationary_metrics(c, lam, mu)["Wq"]
    acc = {}

    def add(name, interval):
        mean, half = interval
        acc.setdefault(name, ([], [], []))
        acc[name][0].append(mean)
        acc[name][1].append(E.covers(truth, mean, half))
        acc[name][2].append(half)

    for r in range(reps):
        g, s, _ = streams(seed_base + 97 * r)
        x = kw_waits(c, lam, mu, n, g, s)
        add("naive", E.naive_interval(x))
        add("batch5", E.batch_means_interval(x, 5))
        add("batch20", E.batch_means_interval(x, C.BATCHES))
    return {"c": c, "rho_pct": rho_pct, "n": n, "reps": reps, "truth": truth,
            "variants": {name: _summarise(*vals, truth) for name, vals in acc.items()}}


def load_precomputed():
    with open(PRECOMPUTED_PATH, encoding="utf-8") as f:
        return json.load(f)


def nearest(grid, value):
    """Nächster Wert der Messreihen-Achse (bei Gleichstand der kleinere)."""
    return min(grid, key=lambda g: (abs(g - value), g))


def coverage_cell(precomputed, c, rho_pct, n):
    """Zelle der Abdeckungs-Tabelle für (gemessene Spurzahl, gemessene Auslastung, Lauflänge)."""
    return next(x for x in precomputed["coverage"] if x["c"] == c and x["rho_pct"] == rho_pct and x["n"] == n)
