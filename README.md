# M/M/c – mehrere Spuren, eine Schlange (Streamlit-Demo)

Interaktive Demo zur **M/M/c-Schlange** und ihrer Warteformel **Erlang C**, am Terminal-Gate aus
[mm1-queue-demo](https://github.com/sebastian-hanisch/mm1-queue-demo), jetzt mit c gleich schnellen Spuren und einer
gemeinsamen Schlange. **Drittes Stück der Konzepte-Linie „Warteschlangentheorie und Simulation“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein
wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Mit c = 1 ist es genau die Schlange aus Stück 1. Neu sind zwei Fragen: **Was bringt es, die Schlange zusammenzulegen?**
(Pooling) und **Wie viele Spuren braucht ein Gate für eine Ziel-Wartezeit?** Dazu die Gegenprobe, ob die Intervall-Probleme
aus [output-analysis-demo](https://github.com/sebastian-hanisch/output-analysis-demo) bei mehr Spuren verschwinden.

## Kernfrage

Eine gemeinsame Schlange gegen c getrennte Schlangen bei derselben Last und denselben Spuren: um wie viel kürzer wartet ein
Lkw, und wie groß muss ein Gate sein, damit es bei hoher Auslastung noch gut funktioniert?

## Modell und Methodik

- **Modell:** Poisson-Ankünfte (Rate λ), exponentielle Abfertigung (Rate μ je Spur, 3 min Mittel), c Spuren, eine FIFO-Schlange,
  unbegrenzt. Angebot a = λ/μ, Auslastung je Spur ρ = a/c; Gleichgewicht nur bei ρ < 100 %. Regler: Spuren (1–16), Auslastung
  (10–97 %), Lauflänge (1 000–50 000 Lkw), Zufalls-Seed. Voreinstellung: 4 Spuren, ρ = 90 %, 10 000 Lkw.
- **Formeln** (`mmc_formulas.py`): Erlang C (über die stabile Erlang-B-Rekursion), Wq = C/(cμ − λ), die Verteilung der Zahl im
  System, P(Wq > t) = C·e^(−(cμ−λ)t), c getrennte M/M/1-Schlangen (Wq = ρ/(μ − λ/c)), Spurbedarf. Unabhängige Referenz im Test:
  das lineare Gleichungssystem der abgeschnittenen Geburts-Sterbe-Kette mit Sterberate min(n, c)·μ.
- **Simulation** (`mmc_simulation.py`): Ereignisliste mit c Spuren, je Ereignistyp ein Handler; Zufall aus SplitMix64 mit
  getrennten Strömen für Ankünfte, Bedienzeiten und die Spurzuteilung. Zur Gegenprobe die **Kiefer-Wolfowitz-Rekursion**
  (FIFO, c Server, ohne Ereignisliste): beide liefern Kunde für Kunde dieselben Wartezeiten.
- **Getrennte Schlangen:** jeder Lkw wird zufällig einer Spur zugeteilt (gleichverteilt) und wartet nur dort.
- **Vorgerechnete Messreihe** (`generate_precomputed.py` → `precomputed_sweep.json`, rund sechs Minuten parallel): 5 Spurzahlen ×
  4 Auslastungen × 6 Lauflängen, je 400 Läufe, mit naivem Intervall und Batch Means (5 und 20 Batches). Live läuft nur der
  gewählte Einzellauf und der Pooling-Vergleich.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; Zeiten bei 3 min Abfertigung je Spur. Die Messreihe hat 400 Läufe je Zelle
(Standardfehler einer Abdeckung höchstens 2.5 Prozentpunkte).

| Frage | Befund |
|---|---|
| Was sagt Erlang C? | 4 Spuren, ρ = 80 %: 60 % der Lkw warten, im Mittel 2.2 min. 4 Spuren, ρ = 90 %: **79 % warten, 5.9 min**. 16 Spuren, ρ = 95 %: 78 % warten, 2.9 min. |
| Was bringt die gemeinsame Schlange? | Wartezeit bei c getrennten gegen eine gemeinsame Schlange, ρ = 90 %: c = 2 **2.1×**, c = 4 **4.6×**, c = 8 **10×**, c = 16 **24×**. Bei ρ = 50 %: c = 2 3×, c = 4 11.5×, c = 8 **68×**. Mit einer Spur gibt es nichts zu poolen (Faktor 1). |
| Wie viele Spuren braucht ein Gate (Ziel: im Mittel höchstens 1 min Wartezeit)? | Angebot a = 2: **4 Spuren** (Auslastung 50 %), a = 5: 7 (71 %), a = 10: 12 (83 %), a = 20: 22 (91 %), a = 50: 53 (94 %), a = 100: **103** (97 %), a = 200: 203 (98 %). Große Gates brauchen nur wenige Spuren mehr als das Angebot (a = 100: 3, a = 200: 3, a = 2: 2). |
| Und die Faustregel „höchstens 80 % Auslastung“? | Sie ist **für kleine Gates zu knapp**: a = 2 ergibt 3 Spuren und 1.3 min Wartezeit (Ziel 1 min verfehlt); a = 5 trifft genau (7 Spuren). Für große Gates ist sie **zu großzügig**: a = 10: 13 statt 12, a = 20: 25 statt 22, a = 50: 63 statt 53, a = 100: 125 statt 103. |
| Trifft das naive 95-%-Intervall bei mehr Spuren? | **Nein, in keiner der 120 Zellen** (5 Spurzahlen × 4 Auslastungen × 6 Lauflängen) häufiger als in 51 % der Fälle. Bei ρ = 90 % und 10 000 Lkw für c = 1, 2, 4, 8, 16 in 8–11 % der Fälle. |
| Und Batch Means? | Bei ρ = 90 % und 10 000 Lkw mit 20 Batches 81–84 % für alle Spurzahlen; bei ρ = 95 % und 50 000 Lkw 83–87 %. **5 Batches sind in 117 von 120 Zellen mindestens so gut wie 20**; bei 50 000 Lkw und ρ ≤ 90 % trifft das Intervall mit 5 Batches in mindestens 91 % der Fälle. |
| Schrumpft die Streuung eines Laufs mit mehr Spuren? | **Nein, sie wächst**: relative Standardabweichung der Wartezeit bei ρ = 90 % und 10 000 Lkw 21 % (c = 1), 21 %, 21 %, 24 % (c = 8), **31 %** (c = 16). |
| Startverzerrung bei kurzen Läufen? | ρ = 95 %, 1 000 Lkw: Mittelwert für alle Spurzahlen 27–33 % unter der Formel; bei 50 000 Lkw im Rauschen. |

## Befunde und Korrekturen gegenüber dem Plan

- **„Streuung unabhängig von c“ (Vorab-Messreihe) war zu grob.** Die Vorab-Messreihe (c = 1 bis 8, ρ = 80 und 90 %, 300 Läufe)
  zeigte 20–25 % und sah unabhängig von c aus. Die vollständige Messreihe bis c = 16 zeigt: **sie wächst leicht mit c**
  (ρ = 90 %: 21 % bis 31 %) und bei ρ = 50 % deutlich (5 % bei c = 1, 51 % bei c = 16, weil dort fast kein Lkw mehr wartet und der
  Mittelwert aus wenigen Ereignissen besteht).
- **Zusätzlich gefunden: Weniger, längere Batches** sind auch hier besser (siehe Befund oben), wie in output-analysis-demo.
- **Zwei Fehler im Bau, im Test gefangen:** `p_n` lief für große n über (`int too large to convert to float`), jetzt über
  p_c·ρ^(n−c); und ein Satz-Komma wurde von einem Tausender-`replace` in einen Punkt verwandelt, jetzt gesichert durch einen
  Quelltext-Test.

## Ehrliche Grenzen

- Die Auslastung gilt je Spur, bei gleichem Angebot ändert sich also mit c die Zahl der Spuren und nicht die Last je Spur;
  der Vergleich „eine gemeinsame gegen c getrennte“ hält die Gesamtkapazität gleich.
- „Getrennte Schlangen“ heißt hier **zufällige Zuteilung**. Wählen Kunden die kürzeste Schlange, liegt das Ergebnis zwischen
  beiden Fällen; das ist nicht Gegenstand dieses Stücks.
- Alle Spuren sind gleich schnell, und die Schlange ist FIFO ohne Prioritäten.
- Der Spurbedarf rechnet mit der **mittleren** Wartezeit; Ziele wie „95 % der Lkw warten höchstens 5 min“ führen zu anderen
  Spurzahlen (nicht untersucht).
- Die Messreihe läuft bei vier Auslastungen (50, 80, 90, 95 %) und fünf Spurzahlen (1, 2, 4, 8, 16); die App zeigt für andere
  Werte die nächste gemessene Zelle und sagt das.
- Die Live-Ansicht rechnet **einen** Lauf; Intervalle über viele Läufe stehen nur in den vorgerechneten Messreihen.

## Verwandte Demos im Portfolio

- [`mm1-queue-demo`](https://github.com/sebastian-hanisch/mm1-queue-demo) (Stück 1): der Fall c = 1.
- [`output-analysis-demo`](https://github.com/sebastian-hanisch/output-analysis-demo) (Stück 2): Intervalle für Simulationsläufe, deren
  Probleme hier bei mehr Spuren geprüft werden.
- [`ems_demo`](https://github.com/sebastian-hanisch/ems-demo): Rettungsdienst-Standortplanung mit dem **Hypercube Queueing
  Model**, einer Markov-Kette über mehrere Server; ihr Korrektheitstest ist die **Erlang-B**-Formel des Verlustsystems. Hier ist
  es das Wartesystem (Erlang C); Erlang B behandelt ein Folgestück der Linie.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Abfertigungsdauer exponentiell | M/G/1, Kingman-Näherung |
| Unbegrenzte Schlange | M/M/c/c (Erlang B) |
| Unendliche Geduld | Erlang A |
| Konstante Ankunftsrate | Wurzel-Personalregel (Halfin-Whitt), zeitvariable Ankünfte |
| Eine gemeinsame Schlange, kein Kunde wählt | Power-of-d-Choices |
| Alle Lkw gleich wichtig | Prioritätsklassen |

Kein Folgestück: unterschiedlich schnelle Spuren.

## Tests

121 Tests, rund 15 s: Erlang C gegen die abgeschnittene Geburts-Sterbe-Kette und gegen die Lehrbuchsumme (c = 1 bis 20),
Handwerte (c = 2, a = 1: C = 1/3), Simulation gegen eine von Hand gerechnete Vier-Lkw-Instanz an zwei Spuren (Wartezeiten,
∫N dt, ∫Nq dt, beschäftigte Spuren, Zeit je Zustand, Treppenkurve, getrennte Schlangen), Ereignissimulation gegen
Kiefer-Wolfowitz Kunde für Kunde (c = 1 bis 16), Little's Gesetz als Pfadidentität, Spurbedarf und Faustregel von Hand,
Vollständigkeit der vorgerechneten Datei, Presets/Permalink, AppTest-Rauchtests und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `mmc_formulas.py` | Erlang C, Verteilungen, getrennte Schlangen, Spurbedarf |
| `mmc_simulation.py` | Generator, Ereignissimulation mit Handlern, Kiefer-Wolfowitz, getrennte Schlangen |
| `mmc_estimators.py` | naives Intervall, Batch Means (Kopie aus output-analysis-demo) |
| `mmc_evaluation.py` | Kennzahlen, Verteilungen, Pooling-Vergleich, Personalbedarf, Abdeckungsstudie |
| `generate_precomputed.py` | rechnet die Messreihe vor → `precomputed_sweep.json` |
| `mmc_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mmc_presets.py`, `mmc_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Erlang, A. K. (1917): Løsning af nogle Problemer fra Sandsynlighedsregningen af Betydning for de automatiske Telefoncentraler.
  *Elektroteknikeren* (Kopenhagen); enthält die Formeln für Verlust und Wartezeit.
- Kiefer, J., Wolfowitz, J. (1955): On the theory of queues with many servers. *Transactions of the American Mathematical Society*
  78, 1–18.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Messreihe neu rechnen:
`python generate_precomputed.py`.

Gebaut mit Streamlit und Plotly.
