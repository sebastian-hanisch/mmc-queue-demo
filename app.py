"""M/M/c - mehrere Spuren, eine Schlange - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Konzepte-Linie "Warteschlangentheorie und Simulation": Das Terminal-Gate aus Stück 1 bekommt c
gleich schnelle Spuren mit EINER gemeinsamen Schlange (Erlang C). Die Demo zeigt, was das Zusammenlegen der Schlange
bringt (Pooling), wie viele Spuren man für eine Ziel-Wartezeit braucht und dass das Intervall-Problem aus Stück 2 mit mehr
Spuren nicht verschwindet. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mmc_constants as C
import mmc_formulas as F
from mmc_evaluation import (coverage_cell, compare_pooling, little_check, load_precomputed, nearest, run_live,
                            staffing_row, state_distribution, wait_survival_curve, wait_tail_fractions, window_steps)
from mmc_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                         sync_query_params)
from mmc_simulation import rates
from mmc_visualization import (build_coverage_chart, build_pooling_factor_chart, build_shared_vs_separate, build_state_distribution,
                               build_staffing_chart, build_sweep, build_trajectory, build_wait_survival)

st.set_page_config(page_title="M/M/c-Warteschlange – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _precomputed():
    return load_precomputed()


@st.cache_data(show_spinner=False)
def _run(c, rho_pct, n, seed):
    return run_live(c, rho_pct, n, seed)


@st.cache_data(show_spinner=False)
def _pooling(c, rho_pct, n, seed):
    return compare_pooling(c, rho_pct, n, seed)


def _min(x):
    return f"{x:.2f} min" if x < 1 else f"{x:.1f} min"


st.title("🛣️ M/M/c: mehrere Spuren, eine Schlange")
st.markdown(
    """
Das **Terminal-Gate** aus Stück 1 bekommt **c gleich schnelle Spuren**; wartende Lkw stehen in **einer gemeinsamen
Schlange** (FIFO) und rücken auf die nächste freie Spur nach. Das ist die **M/M/c-Schlange**, ihre Warteformel heißt
**Erlang C**. Die Kernaussage: **Eine gemeinsame Schlange ist viel besser als c getrennte**, und der Gewinn wächst mit
der Spurzahl (**Pooling**). Außerdem darf ein großes Gate bei gleichem Service **heißer gefahren** werden als ein kleines.
Weiter unten: wie viele Spuren eine Ziel-Wartezeit verlangt, und ob die Intervall-Probleme aus Stück 2 bei mehr Spuren
verschwinden.
"""
)
st.caption(
    "Drittes Stück der Linie „Warteschlangentheorie und Simulation“, aufbauend auf "
    "[mm1-queue-demo](https://sebastianhanisch-mm1-queue-demo.streamlit.app/) (eine Spur, c = 1) und "
    "[output-analysis-demo](https://sebastianhanisch-output-analysis-demo.streamlit.app/) (Intervalle für Simulationsläufe). "
    "Jedes Folgestück hebt eine der Annahmen unter „Wo die Annahmen enden“ auf."
)

with st.expander("So funktioniert die M/M/c-Schlange", expanded=True):
    st.markdown(
        """
- **Angebot** a = λ/μ: so viele Spuren sind im Mittel gleichzeitig mit Abfertigen beschäftigt. **Auslastung je Spur**
  ρ = a/c. Nur bei **ρ < 100 %** stellt sich ein Gleichgewicht ein.
- **Erlang C** C(c, a): Wahrscheinlichkeit, dass ein ankommender Lkw warten muss (alle c Spuren belegt). Daraus
  **Wq = C/(cμ − λ)**; ein wartender Lkw wartet im Mittel 1/(cμ − λ), die Spuren arbeiten gemeinsam gegen die Schlange.
- **Pooling:** Teilt man dieselbe Last in c getrennte Schlangen mit je einer Spur, kann ein Lkw hinter einem langsamen
  stehen, während nebenan eine Spur frei ist. Die gemeinsame Schlange verhindert das.
- **Simulation:** Ereignis für Ereignis vom leeren Gate aus, mit c Spuren; dasselbe Prinzip wie in Stück 1.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{name}", width="stretch", on_click=apply_preset, args=(name,),
                  help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    c = st.slider("Zahl der Spuren c", *bounds("c_slider"), key="c_slider",
                  help="Gleich schnelle Spuren mit einer gemeinsamen Schlange. Bei 1 ist es die M/M/1-Schlange aus Stück 1.")
    rho_pct = st.slider("Auslastung ρ je Spur", *bounds("rho_slider"), key="rho_slider", format="%d %%",
                        help="Anteil der Zeit, in dem eine Spur im Mittel belegt ist (Ankunftsrate geteilt durch die "
                             "Abfertigungsrate aller Spuren). Nur Gleichgewichtsfälle (unter 100 %).")
    n = st.select_slider("Simulierte Lkw je Lauf", options=C.N_OPTIONS, key="n_select",
                         help="Länge des Simulationslaufs. Die mittlere Abfertigungsdauer je Spur ist fest 3 min; "
                              "sie verschiebt nur die Zeitachse.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1],
                           step=1, key="seed_input", help="Bestimmt alle Zufallszahlen des Laufs.")
    st.button("🎲 Neuen Lauf würfeln", on_click=randomize_seed)

c, rho_pct, n, seed = int(c), int(rho_pct), int(n), int(seed)
sync_query_params({"c_slider": c, "rho_slider": rho_pct, "n_select": n, "seed_input": seed})

lam, mu = rates(c, rho_pct)
formula = F.stationary_metrics(c, lam, mu)
pre = _precomputed()
grid_c, grid_rho = nearest(C.GRID_C, c), nearest(C.GRID_RHO_PCT, rho_pct)

with st.spinner("Simuliere das Gate …"):
    sim = _run(c, rho_pct, n, seed)

st.markdown("---")
st.markdown("## 🛣️ Das Gate in Zahlen")
st.caption(
    f"{c} Spuren, je μ = {mu * 60:.0f} Lkw/h (3 min Abfertigung), Ankunftsrate λ = {lam * 60:.1f} Lkw/h, "
    f"Angebot a = {formula['a']:.2f}, {C.fmt_int(n)} simulierte Lkw."
)
l_hat, lw_hat, little_gap = little_check(sim)
c1, c2, c3 = st.columns(3)
c1.metric("Auslastung je Spur ρ", f"{rho_pct} %")
c2.metric("Anteil der Lkw, die warten müssen (Formel)", C.fmt_pct(formula["p_wait"]))
c3.metric("Anteil der Lkw, die warten müssen (simuliert)", C.fmt_pct(sim.share_waiting),
          delta=f"{100 * (sim.share_waiting - formula['p_wait']):+.1f} Prozentpunkte gegen Formel", delta_color="off")
c4, c5, c6 = st.columns(3)
c4.metric("Wartezeit in der Schlange (Formel)", _min(formula["Wq"]))
c5.metric("Wartezeit in der Schlange (simuliert)", _min(sim.mean_wait),
          delta=f"{(sim.mean_wait - formula['Wq']) / formula['Wq']:+.1%} gegen Formel", delta_color="off")
c6.metric("Little's Gesetz im Lauf: L gegen λ·W", "stimmt" if little_gap < 1e-6 else f"Abweichung {little_gap:.2%}",
          help=f"L = {l_hat:.4f} (Fläche unter der Kurve geteilt durch die Laufzeit), λ·W = {lw_hat:.4f} "
               "(Ankunftsrate mal mittlere Verweilzeit). Auf jedem Lauf gleich, weil das Gate am Ende leer ist.")

cell = coverage_cell(pre, grid_c, grid_rho, n)["variants"]
st.info(
    f"**Ein einzelner Lauf ist keine Messung der Formel.** Bei {C.fmt_int(n)} Lkw weicht die simulierte Wartezeit in der nächsten "
    f"gemessenen Zelle ({grid_c} Spuren, ρ = {grid_rho} %) typischerweise um etwa **±{100 * cell['naive']['rel_std']:.0f} %** "
    f"von der Formel ab ({pre['grid_reps']} Läufe). Auch mit mehr Spuren bleibt das so (siehe unten)."
)

st.markdown("### Die Schlange über der Zeit")
total_min = sim.end_time
window_min = C.WINDOW_HOURS * 60
max_start_h = int((total_min - window_min) // 60)
if max_start_h >= 1:
    start_h = st.slider("Fenster ab Stunde", 0, max_start_h, key="window_start_h", step=C.WINDOW_STEP_HOURS,
                        help=f"Zeigt {C.WINDOW_HOURS} Stunden des Laufs; 0 = vom leeren Gate aus.")
else:
    start_h = 0
start_min = start_h * 60
end_min = min(total_min, start_min + window_min)
st.plotly_chart(build_trajectory(window_steps(sim.trajectory, start_min, end_min), start_min, end_min, c, formula["L"]),
                width="stretch", key=f"traj_{c}_{rho_pct}_{n}_{seed}_{start_h}")
st.caption(
    "Unterhalb der gepunkteten Linie ist mindestens eine Spur frei, kein Lkw wartet; darüber steht die Differenz in der "
    "gemeinsamen Schlange."
)

st.markdown("### Mittelwerte täuschen: die Verteilung")
col_a, col_b = st.columns(2)
shares, rest = state_distribution(sim)
with col_a:
    st.markdown("**Wie viele Lkw stehen gleichzeitig im System?**")
    st.plotly_chart(build_state_distribution(shares, rest, c, lam, mu), width="stretch",
                    key=f"dist_{c}_{rho_pct}_{n}_{seed}")
    if rest > 0.0005:
        st.caption(f"Weitere {rest:.1%} der Zeit stehen mehr als {C.MAX_STATE_SHOWN} Lkw im System (nicht gezeigt).")
with col_b:
    st.markdown("**Wie viele Lkw warten länger als t Minuten?**")
    t_max = max(2.0, min(8.0 * formula["Wq"] + 3.0 / (c * mu - lam), max(sim.waits)))
    ts, surv = wait_survival_curve(sim.waits, t_max)
    st.plotly_chart(build_wait_survival(ts, surv, c, lam, mu), width="stretch", key=f"surv_{c}_{rho_pct}_{n}_{seed}")
tails = wait_tail_fractions(sim.waits)
tail_cols = st.columns(len(C.TAIL_MINUTES))
for col, minutes in zip(tail_cols, C.TAIL_MINUTES):
    col.metric(f"Anteil mit Wartezeit über {minutes} min", C.fmt_pct(tails[minutes]),
               delta=f"Formel {C.fmt_pct(F.prob_wait_exceeds(c, lam, mu, minutes))}", delta_color="off", delta_arrow="off")
st.caption(
    f"Im Mittel wartet ein Lkw {_min(formula['Wq'])}, aber **{C.fmt_pct(formula['p_wait'])} aller Lkw müssen überhaupt "
    "warten**, und wer warten muss, wartet im Mittel "
    f"{_min(1 / (c * mu - lam))} (Erlang C: die Wartezeit der Wartenden ist exponentiell)."
)

st.markdown("---")
st.subheader("📐 Gemeinsam oder getrennt? Der Pooling-Effekt")
pool = _pooling(c, rho_pct, n, seed)
p1, p2, p5 = st.columns(3)
p3, p4, _ = st.columns(3)
p1.metric("Gemeinsam (Formel)", _min(pool["formula_shared"]))
p2.metric("Gemeinsam (simuliert)", _min(pool["sim_shared"]))
p5.metric("Pooling-Faktor (Formel)", f"{pool['formula_factor']:.1f}×",
          help="Wie viel mal länger die mittlere Wartezeit bei getrennten Schlangen ist (zufällige Zuteilung der Lkw).")
p3.metric("Getrennt (Formel)", _min(pool["formula_separate"]),
          help=f"{c} getrennte Schlangen, jede mit einer Spur, Lkw zufällig zugeteilt.")
p4.metric("Getrennt (simuliert)", _min(pool["sim_separate"]))
st.plotly_chart(build_shared_vs_separate(c, rho_pct), width="stretch", key=f"shared_sep_{c}_{rho_pct}")
if c == 1:
    st.info("Bei einer Spur gibt es nichts zu poolen: gemeinsam und getrennt sind dasselbe (Faktor 1).")
else:
    st.success(
        f"Bei {c} Spuren und ρ = {rho_pct} % warten Lkw mit **einer** gemeinsamen Schlange im Mittel {_min(pool['formula_shared'])}, "
        f"mit {c} getrennten {_min(pool['formula_separate'])}: das **{pool['formula_factor']:.1f}-Fache**. Derselbe Lkw-Strom, "
        "dieselben Spuren, nur die Schlange ist anders gebaut."
    )
st.markdown("**Wie wächst der Gewinn mit der Spurzahl?**")
st.plotly_chart(build_pooling_factor_chart(c), width="stretch", key=f"pooling_factor_{c}")
st.caption(
    "Formelwerte (Wartezeit bei c getrennten Schlangen geteilt durch die Wartezeit bei einer gemeinsamen). Der Gewinn "
    "wächst mit der Spurzahl und ist bei niedriger Auslastung am größten."
)

st.markdown("---")
st.subheader("📐 Wartezeit über der Auslastung für verschiedene Spurzahlen")
st.plotly_chart(build_sweep(pre), width="stretch", key="sweep")
st.caption(
    f"Linien: Formel (Erlang C); Punkte: Mittel aus {pre['grid_reps']} Läufen à {C.fmt_int(max(C.GRID_N))} Lkw. Mit mehr Spuren "
    "bleibt die Wartezeit bei gleicher Auslastung deutlich kürzer; die Kurven steigen erst ganz nahe 100 % steil."
)

st.markdown("---")
st.subheader("🔬 Wie viele Spuren braucht ein Gate?")
st.markdown(
    "Gegeben das **Angebot** a (so viele Lkw werden im Mittel gleichzeitig abgefertigt) und eine **Ziel-Wartezeit**: "
    "die kleinste Spurzahl, die sie einhält, mit der Auslastung, die dabei herauskommt."
)
s1, s2 = st.columns(2)
with s1:
    load_a = st.select_slider("Angebot a", options=C.STAFFING_LOADS, value=C.DEFAULT_STAFFING_LOAD, key="staffing_load",
                              help="Ankunftsrate geteilt durch die Abfertigungsrate einer Spur (a = 10 heißt: im Mittel "
                                   "sind 10 Spuren beschäftigt).")
with s2:
    target = st.select_slider("Ziel: mittlere Wartezeit höchstens", options=C.STAFFING_TARGETS,
                              value=C.DEFAULT_STAFFING_TARGET, format_func=lambda v: f"{v} min", key="staffing_target")
row = staffing_row(load_a, target)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Nötige Spuren", row["c"])
m2.metric("Auslastung je Spur dabei", C.fmt_pct(row["rho"]))
m3.metric("Anteil, der warten muss", C.fmt_pct(row["p_wait"]))
m4.metric("Mittlere Wartezeit", _min(row["wq"]))
all_rows = [staffing_row(a, target) for a in C.STAFFING_LOADS]
st.plotly_chart(build_staffing_chart(all_rows, load_a), width="stretch", key=f"staffing_{load_a}_{target}")
rule_text = (f"Die Faustregel „höchstens {int(C.RULE_OF_THUMB_MAX_RHO * 100)} % Auslastung“ ergäbe {row['c_rule']} Spuren "
             f"und eine mittlere Wartezeit von {_min(row['wq_rule'])}")
if row["c_rule"] < row["c"]:
    verdict = f"{rule_text}: **zu wenig**, das Ziel von {target} min würde verfehlt."
elif row["c_rule"] > row["c"]:
    extra = row["c_rule"] - row["c"]
    verdict = f"{rule_text}: **{'eine Spur' if extra == 1 else f'{extra} Spuren'} mehr als nötig**."
else:
    verdict = f"{rule_text}: **genau das Nötige**."
st.info(
    f"Für das Angebot a = {load_a} und höchstens {target} min Wartezeit genügen **{row['c']} Spuren** bei {C.fmt_pct(row['rho'])} "
    f"Auslastung. {verdict} Große Gates dürfen bei gleichem Service heißer laufen als kleine: das ist der Skaleneffekt, "
    "den die Wurzel-Personalregel des Folgestücks genau beschreibt."
)

st.markdown("---")
st.subheader("🔬 Verschwindet das Intervall-Problem bei mehr Spuren?")
st.markdown(
    "In Stück 2 traf das naive 95-%-Intervall den wahren Wert nur selten. Bei mehr Spuren ändern sich Zeitskala und "
    "Wartezeiten; gemessen über 400 Läufe je Zelle, ob das Problem bleibt."
)
cov_n = st.select_slider("Lauflänge der Messreihe", options=C.GRID_N, value=n, key="cov_n_select",
                         help="Lkw je Lauf in der vorgerechneten Messreihe.")
st.plotly_chart(build_coverage_chart(pre, grid_rho, int(cov_n)), width="stretch", key=f"coverage_{grid_rho}_{cov_n}")
cc = coverage_cell(pre, grid_c, grid_rho, int(cov_n))["variants"]
st.info(
    f"Bei {grid_c} Spuren, ρ = {grid_rho} % (nächste gemessene Zelle) und {C.fmt_int(int(cov_n))} Lkw je Lauf enthält das nominale "
    f"95-%-Intervall den wahren Wert beim **naiven Intervall** in {C.fmt_pct(cc['naive']['cover'])} der Fälle, bei "
    f"**Batch Means (20)** in {C.fmt_pct(cc['batch20']['cover'])} und bei **Batch Means (5)** in {C.fmt_pct(cc['batch5']['cover'])}. "
    "Mehr Spuren ändern daran nichts Grundsätzliches: Wartezeiten aufeinanderfolgender Lkw bleiben verwandt."
)
st.caption(
    f"Je Punkt {pre['grid_reps']} unabhängige Läufe; der Standardfehler einer Abdeckung beträgt höchstens etwa 2.5 "
    "Prozentpunkte. Alle Läufe starten leer."
)

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Abfertigungsdauer exponentiell** | Die Wartezeit hängt von der Streuung der Dauer ab; Erlang C gilt dann nicht mehr. | **M/G/1, Kingman-Näherung** (Folgestück) |
| **Unbegrenzte Schlange** | Stellplätze sind knapp: wer bei voller Zufahrt ankommt, geht verloren. Das Gegenstück ohne Warten heißt Erlang B; `ems_demo` prüft sich an dieser Formel. | **M/M/c/c (Erlang B)** (Folgestück) |
| **Unendliche Geduld** | Niemand dreht um. Mit Abwanderung bleibt auch bei Überlast ein Gleichgewicht. | **Erlang A** (Folgestück) |
| **Konstante Ankunftsrate** | Echte Gates haben Morgenspitzen; die Gleichgewichtsformel mit dem Tagesmittel unterschätzt die Spitze. | **Wurzel-Personalregel (Halfin-Whitt)**, **zeitvariable Ankünfte** (Folgestücke) |
| **Eine gemeinsame Schlange, kein Kunde wählt** | Wählen Kunden selbst eine Spur (kürzeste Schlange), liegt das Ergebnis zwischen „getrennt“ und „gemeinsam“. Hier gibt es nur die zufällige Zuteilung als Gegenbeispiel. | **Power-of-d-Choices** (Folgestück) |
| **Alle Lkw gleich wichtig** | Eilige Lkw brauchen Vorfahrt; das verschiebt die Wartezeit zwischen den Klassen. | **Prioritätsklassen** (Folgestück) |
| **Alle Spuren gleich schnell** | Mit unterschiedlich schnellen Spuren gibt es keine einfache Formel mehr. | kein Folgestück |
"""
)
st.caption(
    "Verwandt im Portfolio: die Rettungsdienst-Demo [ems-demo](https://sebastianhanisch-ems-demo.streamlit.app/) rechnet mit "
    "einer Markov-Kette über mehrere Server (Hypercube Queueing Model) und prüft sich an der Erlang-B-Formel, also am "
    "Verlustsystem; hier ist es das Wartesystem. Außerdem "
    "[mm1-queue-demo](https://sebastianhanisch-mm1-queue-demo.streamlit.app/) (Stück 1) und "
    "[output-analysis-demo](https://sebastianhanisch-output-analysis-demo.streamlit.app/) (Stück 2)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell** (Kendall-Notation M/M/c): Poisson-Ankünfte mit Rate $\lambda$, exponentielle Abfertigung mit Rate $\mu$ je Spur,
$c$ Spuren, eine FIFO-Schlange. Angebot $a = \lambda/\mu$, Auslastung je Spur $\rho = a/c < 1$.

**Gleichgewicht.** Die Zahl der Lkw im System ist eine Geburts-Sterbe-Kette mit Geburtsrate $\lambda$ und Sterberate
$\min(n, c)\,\mu$. Mit $p_0 = \Bigl[\sum_{k<c} \frac{a^k}{k!} + \frac{a^c}{c!\,(1-\rho)}\Bigr]^{-1}$ gilt
$p_n = p_0\,\frac{a^n}{n!}$ für $n \le c$ und $p_n = p_0\,\frac{a^n}{c!\,c^{\,n-c}}$ für $n > c$.

**Erlang C.** Die Wahrscheinlichkeit zu warten (PASTA) ist
$$C(c, a) = \sum_{n \ge c} p_n = \frac{a^c/(c!\,(1-\rho))}{\sum_{k<c} a^k/k! + a^c/(c!\,(1-\rho))}.$$
Numerisch stabil über die Erlang-B-Rekursion $B_k = \frac{a B_{k-1}}{k + a B_{k-1}}$ und $C = \frac{B_c}{1 - \rho(1 - B_c)}$.
Daraus $W_q = \frac{C(c,a)}{c\mu - \lambda}$, $L_q = \lambda W_q$, $W = W_q + 1/\mu$, $L = L_q + a$ (Little's Gesetz) und
$P(W_q > t) = C\,e^{-(c\mu-\lambda)t}$.

**Getrennte Schlangen.** Teilt man den Poisson-Strom zufällig auf $c$ Spuren auf, entstehen $c$ unabhängige M/M/1-Schlangen
mit Rate $\lambda/c$: $W_q^{\text{getrennt}} = \frac{\rho}{\mu - \lambda/c}$. Der **Pooling-Faktor** ist
$W_q^{\text{getrennt}}/W_q^{\text{gemeinsam}}$.

**Spurbedarf.** Die kleinste ganze Zahl $c > a$ mit $W_q(c, a) \le$ Ziel; sie wächst ungefähr wie $a$ plus ein Vielfaches von
$\sqrt a$ (das Thema der Wurzel-Personalregel).

**Simulation.** Ereignisliste mit Ankünften und Abgängen für $c$ Spuren; zur Gegenprobe die **Kiefer-Wolfowitz-Rekursion**: ein
ankommender Lkw nimmt die Spur, die am frühesten frei wird, $W = \max(0,\ \text{früheste Freizeit} - \text{Ankunft})$.
Ankünfte, Bedienzeiten und die Spurzuteilung haben getrennte Zufallsströme (SplitMix64); beide Rechnungen liefern Kunde für
Kunde dieselben Wartezeiten.

Implementiert in `mmc_formulas.py` (Erlang C, Verteilungen, Spurbedarf), `mmc_simulation.py` (Ereignissimulation,
Kiefer-Wolfowitz, getrennte Schlangen), `mmc_evaluation.py` (Kennzahlen, Verteilungen, Pooling, Abdeckungsstudie),
`generate_precomputed.py` (vorgerechnete Messreihe).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html))."
)
