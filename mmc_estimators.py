"""Intervallschätzer für die Messreihen (Kopie der benötigten Einheiten aus output-analysis-demo, Stück 2 - bewusst ohne
Import zwischen Repos). Zweiseitige Intervalle mit Nennniveau 95 %: Rückgabe (Mittelwert, Halbbreite)."""

import math
import statistics

# Quantil 0.975 der t-Verteilung (Standardtabelle, Freiheitsgrade 1..30; darüber leicht konservativ)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
         21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
         40: 2.021, 60: 2.000, 120: 1.980}


def t_quantile(df):
    """Quantil 0.975 der t-Verteilung mit `df` Freiheitsgraden (df ≥ 1)."""
    if df < 1:
        raise ValueError("Freiheitsgrade müssen ≥ 1 sein")
    if df <= 30:
        return _T975[df]
    for edge in (120, 60, 40):
        if df >= edge:
            return _T975[edge]
    return _T975[30]


def _interval_from_independent(values):
    m = len(values)
    return statistics.fmean(values), t_quantile(m - 1) * statistics.stdev(values) / math.sqrt(m)


def naive_interval(x):
    """Intervall, das alle Wartezeiten als unabhängig behandelt: s/√n. Zu schmal bei Autokorrelation."""
    return _interval_from_independent(x)


def batch_means(x, n_batches):
    """Mittelwerte von `n_batches` gleich langen, aufeinanderfolgenden Blöcken; ein Rest am Ende wird verworfen."""
    size = len(x) // n_batches
    if size < 1:
        raise ValueError("weniger Werte als Batches")
    return [sum(x[i * size:(i + 1) * size]) / size for i in range(n_batches)]


def batch_means_interval(x, n_batches):
    """Intervall aus den Batch-Mittelwerten (fast unabhängig, fast normalverteilt bei langen Batches)."""
    return _interval_from_independent(batch_means(x, n_batches))


def covers(truth, mean, half_width):
    """Ob das Intervall mean ± half_width den wahren Wert enthält."""
    return abs(mean - truth) <= half_width
