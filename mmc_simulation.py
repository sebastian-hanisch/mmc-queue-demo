"""Simulation einer Schlange mit c gleich schnellen Spuren und EINER gemeinsamen FIFO-Schlange (M/M/c), dazu als
unabhängige zweite Rechnung die Kiefer-Wolfowitz-Rekursion und das Gegenstück mit c getrennten Schlangen.

Aufbau nach Einheiten (je Ereignistyp ein Handler, kein versteckter Zustand):
  - `handle_arrival`, `handle_departure`: Zustandsübergänge der Ereignissimulation
  - `simulate`: Ereignisschleife (Komposition, keine eigene Regel)
  - `kw_waits`: Wartezeiten ohne Ereignisliste (Server-Freizeiten in einem Heap), aus denselben Zufallszahlen
  - `separate_waits`: c getrennte Schlangen, jeder ankommende Lkw wird zufällig einer Spur zugeteilt

Zufall nur über übergebene `SplitMix64`-Generatoren (reine Ganzzahl-Arithmetik, Portfolio-Konvention): Ankünfte und
Bedienzeiten haben je einen EIGENEN Strom, die Zuteilung zu Spuren einen dritten. Dadurch liefern Ereignissimulation und
Kiefer-Wolfowitz-Rekursion Zahl für Zahl dieselben Wartezeiten."""

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

_MASK = (1 << 64) - 1
ARRIVAL, DEPARTURE = 0, 1
MEAN_SERVICE_MIN = 3.0          # mittlere Abfertigungsdauer je Spur (Minuten) in allen Läufen der Demo
MU = 1.0 / MEAN_SERVICE_MIN


class SplitMix64:
    """Kleiner, gut gemischter 64-Bit-Zufallsgenerator (Vigna); reine Ganzzahl-Arithmetik."""

    def __init__(self, seed):
        self.state = seed & _MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self):
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def expovariate(self, rate):
        """Exponentiell mit Mittel 1/rate (Inversion; 1 − u liegt in (0, 1], der Logarithmus ist endlich)."""
        return -math.log(1.0 - self.uniform()) / rate


def rates(c, rho_pct):
    """(λ, μ) je Minute: Auslastung `rho_pct` (%) JE SPUR bei c Spuren und 3 min mittlerer Abfertigung, also
    λ = ρ·c·μ."""
    return rho_pct / 100.0 * c * MU, MU


def streams(seed):
    """Die drei Zufallsströme eines Laufs: Ankünfte, Bedienzeiten, Spurzuteilung."""
    return SplitMix64(seed), SplitMix64(seed + 7_777_777), SplitMix64(seed + 15_555_555)


@dataclass
class SimResult:
    c: int
    n_customers: int
    end_time: float                  # Zeitpunkt des letzten Abgangs (System dann leer)
    waits: list                      # Wartezeit in der Schlange je Kunde (Reihenfolge der Ankunft)
    sojourns: list                   # Verweilzeit je Kunde (Warten + Bedienung)
    area_in_system: float            # ∫ N(t) dt über [0, end_time]
    area_in_queue: float             # ∫ Nq(t) dt, Nq = max(N − c, 0)
    busy_integral: float             # ∫ (Zahl beschäftigter Spuren) dt
    time_in_state: dict              # n -> Zeit, in der genau n im System waren
    trajectory: list = field(default_factory=list)   # [(t, N nach dem Ereignis)], nur mit `record=True`

    @property
    def mean_in_system(self):
        return self.area_in_system / self.end_time

    @property
    def mean_wait(self):
        return sum(self.waits) / self.n_customers

    @property
    def mean_sojourn(self):
        return sum(self.sojourns) / self.n_customers

    @property
    def utilisation(self):
        """Mittlere Auslastung je Spur."""
        return self.busy_integral / (self.c * self.end_time)

    @property
    def arrival_rate(self):
        return self.n_customers / self.end_time

    @property
    def share_waiting(self):
        """Anteil der Lkw mit Wartezeit über 0."""
        return sum(1 for w in self.waits if w > 0) / self.n_customers


class _State:
    __slots__ = ("c", "t", "n", "queue", "busy", "events", "seq", "n_customers", "lam", "mu", "area_n", "area_q",
                 "busy_integral", "time_in_state", "waits", "sojourns", "arrival_time", "service_time", "trajectory",
                 "record")


def _advance_clock(s, t_new):
    """Zeit auf t_new vorstellen und die Flächen unter N(t), Nq(t), den beschäftigten Spuren und die Zeit je Zustand
    fortschreiben."""
    dt = t_new - s.t
    s.area_n += s.n * dt
    s.area_q += max(s.n - s.c, 0) * dt
    s.busy_integral += s.busy * dt
    s.time_in_state[s.n] = s.time_in_state.get(s.n, 0.0) + dt
    s.t = t_new


def _start_service(s, cid):
    """Kunde cid beginnt die Bedienung jetzt auf einer freien Spur: Wartezeit festhalten, Abgang einplanen."""
    s.waits[cid] = s.t - s.arrival_time[cid]
    s.busy += 1
    s.seq += 1
    heapq.heappush(s.events, (s.t + s.service_time[cid], s.seq, DEPARTURE, cid))


def handle_arrival(s, cid, gap_rng, svc_rng):
    """Ein Lkw kommt an. Bedienzeit wird beim Eintreffen gezogen (eigener Strom), danach die nächste Ankunft eingeplant
    - solange noch Kunden kommen sollen. Ist eine Spur frei, beginnt die Abfertigung sofort, sonst reiht er sich ein."""
    s.arrival_time[cid] = s.t
    s.service_time[cid] = svc_rng.expovariate(s.mu)
    s.n += 1
    if s.busy < s.c:
        _start_service(s, cid)
    else:
        s.queue.append(cid)
    if cid + 1 < s.n_customers:
        s.seq += 1
        heapq.heappush(s.events, (s.t + gap_rng.expovariate(s.lam), s.seq, ARRIVAL, cid + 1))


def handle_departure(s, cid):
    """Ein Lkw ist abgefertigt und gibt seine Spur frei. Der nächste in der Schlange (FIFO) rückt sofort nach."""
    s.sojourns[cid] = s.t - s.arrival_time[cid]
    s.n -= 1
    s.busy -= 1
    if s.queue:
        _start_service(s, s.queue.popleft())


def simulate(c, lam, mu, n_customers, seed, record=False, gap_rng=None, svc_rng=None):
    """Simuliert `n_customers` Ankünfte (Rate lam) an c Spuren mit Bedienrate mu je Spur und läuft, bis alle abgefertigt
    sind. Start mit leerem System. `record=True` schreibt die Treppenkurve N(t) mit. `gap_rng`/`svc_rng` ersetzen die
    Ströme aus `seed` (für Tests mit vorgegebenen Zahlen)."""
    if gap_rng is None or svc_rng is None:
        gap_rng, svc_rng, _ = streams(seed)
    s = _State()
    s.c, s.t, s.n, s.queue, s.busy = c, 0.0, 0, deque(), 0
    s.events, s.seq, s.n_customers, s.lam, s.mu = [], 0, n_customers, lam, mu
    s.area_n = s.area_q = s.busy_integral = 0.0
    s.time_in_state = {}
    s.waits, s.sojourns = [0.0] * n_customers, [0.0] * n_customers
    s.arrival_time, s.service_time = [0.0] * n_customers, [0.0] * n_customers
    s.trajectory, s.record = [(0.0, 0)] if record else [], record
    heapq.heappush(s.events, (gap_rng.expovariate(lam), 0, ARRIVAL, 0))
    while s.events:
        t, _, kind, cid = heapq.heappop(s.events)
        _advance_clock(s, t)
        if kind == ARRIVAL:
            handle_arrival(s, cid, gap_rng, svc_rng)
        else:
            handle_departure(s, cid)
        if s.record:
            s.trajectory.append((t, s.n))
    return SimResult(c, n_customers, s.t, s.waits, s.sojourns, s.area_n, s.area_q, s.busy_integral, s.time_in_state,
                     s.trajectory)


def kw_waits(c, lam, mu, n_customers, gap_rng, svc_rng):
    """Wartezeiten per Kiefer-Wolfowitz-Rekursion (FIFO, c Server): ein ankommender Lkw nimmt die Spur, die am
    frühesten frei wird; Wartezeit = max(0, früheste Freizeit − Ankunftszeit). Ohne Ereignisliste, aus denselben
    Zufallszahlen wie `simulate`."""
    free = [0.0] * c
    heapq.heapify(free)
    t, out = 0.0, [0.0] * n_customers
    for k in range(n_customers):
        t += gap_rng.expovariate(lam)
        earliest = heapq.heappop(free)
        start = t if t > earliest else earliest
        out[k] = start - t
        heapq.heappush(free, start + svc_rng.expovariate(mu))
    return out


def separate_waits(c, lam, mu, n_customers, gap_rng, svc_rng, route_rng):
    """Wartezeiten bei c GETRENNTEN Schlangen: jeder ankommende Lkw wird zufällig (gleichverteilt) einer Spur zugeteilt
    und wartet nur dort; jede Spur ist eine eigene FIFO-Schlange mit einem Server."""
    lane_free = [0.0] * c
    t, out = 0.0, [0.0] * n_customers
    for k in range(n_customers):
        t += gap_rng.expovariate(lam)
        lane = min(int(route_rng.uniform() * c), c - 1)
        start = t if t > lane_free[lane] else lane_free[lane]
        out[k] = start - t
        lane_free[lane] = start + svc_rng.expovariate(mu)
    return out
