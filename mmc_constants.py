"""Konstanten der M/M/c-Demo: Regler, Voreinstellungen, Messreihen-Parameter. Zeiten in Minuten."""


def fmt_int(n):
    """Ganzzahl mit Punkt als Tausendertrenner (10000 -> 10.000)."""
    return f"{n:,}".replace(",", ".")


def fmt_pct(x):
    """Anteil als ganze Prozent mit Leerzeichen (0.86 -> "86 %")."""
    return f"{x:.0%}".replace("%", " %")


C_MIN, C_MAX, DEFAULT_C = 1, 16, 4                           # Spuren
RHO_PCT_MIN, RHO_PCT_MAX, DEFAULT_RHO_PCT = 10, 97, 90      # Auslastung JE SPUR in Prozent (nur Gleichgewichtsfälle)
N_OPTIONS = (1000, 2000, 5000, 10000, 20000, 50000)         # Lkw je Lauf
DEFAULT_N = 10000
SEED_MAX = 999999
DEFAULT_SEED = 35

WINDOW_HOURS = 4                  # Fensterbreite der Treppenkurve N(t)
WINDOW_STEP_HOURS = 1
TAIL_MINUTES = (1, 5, 15, 30)     # Schwellen für "Anteil der Lkw mit Wartezeit über ..."
MAX_STATE_SHOWN = 60              # Balken der Verteilung der Zahl im System (darüber zusammengefasst)
BATCHES = 20                      # Batches für das Intervall der Messreihen

# Personalbedarf (nur Formeln)
STAFFING_LOADS = (2, 5, 10, 20, 50, 100, 200)       # Angebot a = λ/μ in Erlang
STAFFING_TARGETS = (0.5, 1, 2, 5)                   # Ziel-Wartezeit (Minuten)
DEFAULT_STAFFING_LOAD, DEFAULT_STAFFING_TARGET = 10, 1
RULE_OF_THUMB_MAX_RHO = 0.8                         # Faustregel "höchstens 80 % Auslastung"

# Vorgerechnete Messreihen (generate_precomputed.py)
GRID_C = (1, 2, 4, 8, 16)
GRID_RHO_PCT = (50, 80, 90, 95)
GRID_N = N_OPTIONS
GRID_REPS = 400
SWEEP_C = GRID_C
POOLING_RHO_PCT = (50, 80, 90, 95)

PRESET_ORDER = ("Normalfall (4 Spuren, ρ = 90 %)", "Eine Spur (c = 1)", "Großes Gate (16 Spuren)",
                "Entspannt (8 Spuren, ρ = 50 %)")


def _preset(c=DEFAULT_C, rho_pct=DEFAULT_RHO_PCT, n=DEFAULT_N):
    return {"c": c, "rho_pct": rho_pct, "n": n, "seed": DEFAULT_SEED}


PRESETS = {
    "Normalfall (4 Spuren, ρ = 90 %)": _preset(),
    "Eine Spur (c = 1)": _preset(c=1),
    "Großes Gate (16 Spuren)": _preset(c=16),
    "Entspannt (8 Spuren, ρ = 50 %)": _preset(c=8, rho_pct=50),
}
# Formelwerte bei 3 min Abfertigung je Spur (exakt, tests/test_claims.py rechnet sie nach)
PRESET_HELP = {
    "Normalfall (4 Spuren, ρ = 90 %)": "4 Spuren bei ρ = 90 %: Die Formel sagt 79 % der Lkw müssen warten, im Mittel 5.9 min. Mit vier getrennten Schlangen wären es 27 min, also das 4.6-Fache.",
    "Eine Spur (c = 1)": "Eine Spur bei ρ = 90 %: genau die M/M/1-Schlange aus Stück 1, mittlere Wartezeit 27 min. Mit nur einer Spur gibt es keinen Pooling-Effekt.",
    "Großes Gate (16 Spuren)": "16 Spuren bei ρ = 90 %: im Mittel nur 1.1 min Wartezeit statt 27 min bei einer Spur, getrennte Schlangen wären 24-mal so lang.",
    "Entspannt (8 Spuren, ρ = 50 %)": "8 Spuren bei ρ = 50 %: nur 5.9 % der Lkw müssen warten, im Mittel 0.04 min. Getrennte Schlangen: 3 min, das 68-Fache.",
}
