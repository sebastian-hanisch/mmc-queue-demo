"""Rechnet die teure Messreihe vor (Build-Zeit, nicht in der App): `python generate_precomputed.py [Prozesse]` schreibt
`precomputed_sweep.json`.

  coverage  Spurzahl × Auslastung × Lauflänge: Abdeckung des 95-%-Intervalls, Verzerrung, Streuung, Breite und Mittel der
            Schätzwerte (400 Läufe je Zelle) für naives Intervall und Batch Means"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mmc_constants as C
from mmc_evaluation import PRECOMPUTED_PATH, coverage_study


def _task(args):
    c, rho, n_idx, n = args
    return coverage_study(c, rho, n, C.GRID_REPS, seed_base=c * 100_000_000 + rho * 1_000_000 + n_idx * 100_000)


def main(workers):
    t0 = time.time()
    jobs = [(c, r, i, n) for c in C.GRID_C for r in C.GRID_RHO_PCT for i, n in enumerate(C.GRID_N)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        coverage = list(ex.map(_task, sorted(jobs, key=lambda j: -j[3])))
    coverage.sort(key=lambda x: (x["c"], x["rho_pct"], x["n"]))
    out = {"grid_reps": C.GRID_REPS, "coverage": coverage}
    Path(PRECOMPUTED_PATH).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"fertig in {time.time() - t0:.0f} s -> {PRECOMPUTED_PATH}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else min(6, os.cpu_count() or 1))
