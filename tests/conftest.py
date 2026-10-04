import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ScriptedRng:
    """Liefert vorgegebene Werte statt Zufall (Mini-Instanzen von Hand gerechnet). `expovariate(rate)` ignoriert die
    Rate und gibt der Reihe nach die Werte zurück, `uniform()` ebenso aus einer eigenen Liste."""

    def __init__(self, exp_values=(), uniform_values=()):
        self.exp_values, self.uniform_values = list(exp_values), list(uniform_values)
        self.n_exp = self.n_uniform = 0

    def expovariate(self, rate):
        v = self.exp_values[self.n_exp]
        self.n_exp += 1
        return v

    def uniform(self):
        v = self.uniform_values[self.n_uniform]
        self.n_uniform += 1
        return v


@pytest.fixture
def mini_streams():
    """Vier Lkw an zwei Spuren: Zwischenankünfte 1 / 0.5 / 0.5 / 1 (Ankünfte bei 1, 1.5, 2, 3), Bedienzeiten 3 / 2 / 1 / 1.
    Von Hand: Lkw 1 startet bei 1 (Abgang 4), Lkw 2 bei 1.5 (Abgang 3.5), Lkw 3 wartet bis 3.5 (Abgang 4.5), Lkw 4 wartet bis 4
    (Abgang 5): Wartezeiten 0 / 0 / 1.5 / 1, Verweilzeiten 3 / 2 / 2.5 / 2, Ende bei 5."""
    return ScriptedRng(exp_values=[1, 0.5, 0.5, 1]), ScriptedRng(exp_values=[3, 2, 1, 1])
