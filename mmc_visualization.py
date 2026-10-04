"""Plotly-Abbildungen der M/M/c-Demo: Treppenkurve mit Spurlinie, Verteilung der Zahl im System, Wartezeit-Überschreitung,
gemeinsam gegen getrennt, Wartezeit über der Auslastung je Spurzahl, Pooling-Faktor, Personalbedarf, Abdeckung der
Intervalle. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import math

import plotly.graph_objects as go

import mmc_constants as C
import mmc_formulas as F
from mmc_simulation import MU

SIM_COLOR = "#4c78a8"
FORMULA_COLOR = "#f58518"
SHARED_COLOR = "#54a24b"
SEPARATE_COLOR = "#e45756"
C_COLORS = {1: "#9ecae1", 2: "#6baed6", 4: "#3182bd", 8: "#08519c", 16: "#08306b"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25, top=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend=dict(orientation="h", y=legend_y),
                      plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_trajectory(steps, start_min, end_min, c, formula_l=None):
    """Treppenkurve N(t) im Fenster (Minuten); waagerecht die Spurzahl c (darüber warten Lkw) und, falls vorhanden, der
    Formelwert L."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[t / 60 for t, _ in steps], y=[n for _, n in steps], mode="lines", line_shape="hv",
                             line=dict(color=SIM_COLOR, width=2), name="Lkw im System (simuliert)",
                             hovertemplate="%{x:.2f} h: %{y} Lkw<extra></extra>"))
    x0, x1 = start_min / 60, end_min / 60
    fig.add_trace(go.Scatter(x=[x0, x1], y=[c, c], mode="lines", line=dict(color=SEPARATE_COLOR, width=2, dash="dot"),
                             name=f"alle {c} Spuren belegt (darüber warten Lkw)"))
    if formula_l is not None:
        fig.add_trace(go.Scatter(x=[x0, x1], y=[formula_l, formula_l], mode="lines",
                                 line=dict(color=FORMULA_COLOR, width=2, dash="dash"), name=f"Formel L = {formula_l:.1f}"))
    fig.update_xaxes(title_text="Zeit seit Start (Stunden)", range=[x0, x1])
    fig.update_yaxes(title_text="Lkw im System", rangemode="tozero")
    return _base(fig, 320)


def build_state_distribution(shares, rest, c, lam, mu):
    """Balken: Zeitanteil mit genau n Lkw im System; Punkte: Formel (Geburts-Sterbe-Kette); senkrecht die Spurzahl c."""
    ns = list(range(len(shares)))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ns, y=shares, marker_color=SIM_COLOR, name="simuliert (Zeitanteil)",
                         hovertemplate="n = %{x}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ns, y=[F.p_n(c, lam, mu, n) for n in ns], mode="markers",
                             marker=dict(color=FORMULA_COLOR, size=7, symbol="diamond"), name="Formel"))
    fig.add_vline(x=c + 0.5, line=dict(color=SEPARATE_COLOR, width=1.5, dash="dot"),
                  annotation_text="ab hier warten Lkw", annotation_position="top")
    fig.update_xaxes(title_text="Zahl der Lkw im System n")
    fig.update_yaxes(title_text="Anteil der Zeit", tickformat=".0%")
    return _base(fig, 320, top=30)


def build_wait_survival(ts, sim_surv, c, lam, mu):
    """Anteil der Lkw mit Wartezeit über t (logarithmische Achse; Nullwerte werden ausgelassen)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=[v if v > 0 else None for v in sim_surv], mode="lines",
                             line=dict(color=SIM_COLOR, width=2.5), name="simuliert"))
    fig.add_trace(go.Scatter(x=ts, y=[F.prob_wait_exceeds(c, lam, mu, t) for t in ts], mode="lines",
                             line=dict(color=FORMULA_COLOR, width=2, dash="dash"), name="Formel C·e^(−(cμ−λ)t)"))
    fig.update_xaxes(title_text="Wartezeit t (Minuten)")
    fig.update_yaxes(title_text="Anteil der Lkw mit Wartezeit > t", type="log", tickformat=".1%")
    return _base(fig, 320)


def build_shared_vs_separate(c, rho_pct):
    """Mittlere Wartezeit über der Auslastung für die gewählte Spurzahl: eine gemeinsame Schlange gegen c getrennte
    (Formeln); die senkrechte Linie markiert die gewählte Auslastung."""
    xs = list(range(5, 99))
    shared = [F.stationary_metrics(c, x / 100 * c * MU, MU)["Wq"] for x in xs]
    separate = [F.separate_wq(c, x / 100 * c * MU, MU) for x in xs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=shared, mode="lines", line=dict(color=SHARED_COLOR, width=2.5),
                             name="eine gemeinsame Schlange (Erlang C)"))
    fig.add_trace(go.Scatter(x=xs, y=separate, mode="lines", line=dict(color=SEPARATE_COLOR, width=2.5),
                             name=f"{c} getrennte Schlangen (M/M/1 je Spur)"))
    if rho_pct < 99:
        fig.add_vline(x=rho_pct, line=dict(color="#888", width=1, dash="dot"))
    ticks = [0.01, 0.1, 1, 10, 100]
    fig.update_xaxes(title_text="Auslastung je Spur ρ (%)", range=[0, 100])
    fig.update_yaxes(title_text="Mittlere Wartezeit (Minuten, log)", type="log", range=[-2.2, 2.8], tickmode="array",
                     tickvals=ticks, ticktext=["0.01", "0.1", "1", "10", "100"])
    return _base(fig, 340)


def build_pooling_factor_chart(current_c):
    """Pooling-Faktor (Wartezeit getrennt ÷ gemeinsam) über der Spurzahl, je gemessener Auslastung eine Linie (Formeln)."""
    palette = {50: "#9ecae1", 80: "#4c78a8", 90: "#f58518", 95: "#e45756"}
    cs = list(range(2, C.C_MAX + 1))
    fig = go.Figure()
    for rho in C.POOLING_RHO_PCT:
        fig.add_trace(go.Scatter(x=cs, y=[F.pooling_factor(c, rho / 100 * c * MU, MU) for c in cs], mode="lines+markers",
                                 line=dict(color=palette[rho], width=2.5), name=f"ρ = {rho} %",
                                 hovertemplate="c = %{x}: Faktor %{y:.1f}<extra>" + f"ρ = {rho} %</extra>"))
    if current_c >= 2:
        fig.add_vline(x=current_c, line=dict(color="#888", width=1, dash="dot"))
    ticks = [1, 2, 5, 10, 20, 50, 100, 1000]
    fig.update_xaxes(title_text="Zahl der Spuren c", tickmode="array", tickvals=[2, 4, 6, 8, 10, 12, 14, 16])
    fig.update_yaxes(title_text="So viel länger warten Lkw bei getrennten Schlangen (log)", type="log", tickmode="array",
                     tickvals=ticks, ticktext=[str(t) for t in ticks])
    return _base(fig, 340)


def build_sweep(precomputed):
    """Wartezeit über der Auslastung für verschiedene Spurzahlen: Formelkurven (Erlang C) und simulierte Mittel aus den
    vorgerechneten Läufen (50 000 Lkw)."""
    n_max = max(C.GRID_N)
    xs = list(range(10, 98))
    fig = go.Figure()
    for c in C.SWEEP_C:
        color = C_COLORS[c]
        fig.add_trace(go.Scatter(x=xs, y=[F.stationary_metrics(c, x / 100 * c * MU, MU)["Wq"] for x in xs], mode="lines",
                                 line=dict(color=color, width=2.5), name=f"c = {c}", legendgroup=str(c)))
        pts = sorted((x for x in precomputed["coverage"] if x["c"] == c and x["n"] == n_max), key=lambda x: x["rho_pct"])
        fig.add_trace(go.Scatter(x=[p["rho_pct"] for p in pts], y=[p["variants"]["batch20"]["mean_wq"] for p in pts],
                                 mode="markers", marker=dict(color=color, size=8, line=dict(color="white", width=1)),
                                 name=f"c = {c} simuliert", legendgroup=str(c), showlegend=False,
                                 hovertemplate="ρ = %{x} %: %{y:.2f} min<extra>" + f"c = {c}</extra>"))
    ticks = [0.01, 0.1, 1, 10, 100]
    fig.update_xaxes(title_text="Auslastung je Spur ρ (%)", range=[0, 100])
    fig.update_yaxes(title_text="Mittlere Wartezeit (Minuten, log)", type="log", range=[-2.2, 2.8], tickmode="array",
                     tickvals=ticks, ticktext=["0.01", "0.1", "1", "10", "100"])
    return _base(fig, 380)


def build_staffing_chart(rows, current_a):
    """Erreichbare Auslastung bei Ziel-Wartezeit (Linie) und bei der Faustregel „höchstens 80 %“ (Linie), über dem
    Angebot a; der gewählte Wert ist markiert."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["a"] for r in rows], y=[r["rho"] for r in rows], mode="lines+markers",
                             line=dict(color=SHARED_COLOR, width=2.5), name="nötige Spuren für die Ziel-Wartezeit",
                             hovertemplate="a = %{x}: Auslastung %{y:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["a"] for r in rows], y=[r["a"] / r["c_rule"] for r in rows], mode="lines+markers",
                             line=dict(color=SEPARATE_COLOR, width=2, dash="dash"),
                             name=f"Faustregel „höchstens {int(C.RULE_OF_THUMB_MAX_RHO * 100)} %“",
                             hovertemplate="a = %{x}: Auslastung %{y:.0%}<extra></extra>"))
    fig.add_vline(x=current_a, line=dict(color="#888", width=1, dash="dot"))
    fig.update_xaxes(title_text="Angebot a = λ/μ (Erlang, log)", type="log", tickmode="array",
                     tickvals=list(C.STAFFING_LOADS), ticktext=[str(a) for a in C.STAFFING_LOADS])
    fig.update_yaxes(title_text="Auslastung je Spur", tickformat=".0%", range=[0, 1.02])
    return _base(fig, 340)


def build_coverage_chart(precomputed, rho_pct, n):
    """Abdeckung des nominalen 95-%-Intervalls (naiv, Batch Means) über der Spurzahl für eine gemessene Auslastung."""
    fig = go.Figure()
    for method, label, color in (("naive", "naives Intervall", SEPARATE_COLOR), ("batch5", "Batch Means, 5 Batches", "#6baed6"),
                                 ("batch20", "Batch Means, 20 Batches", "#08519c")):
        cells = sorted((x for x in precomputed["coverage"] if x["rho_pct"] == rho_pct and x["n"] == n),
                       key=lambda x: x["c"])
        fig.add_trace(go.Scatter(x=[x["c"] for x in cells], y=[x["variants"][method]["cover"] for x in cells],
                                 mode="lines+markers", line=dict(color=color, width=2.5), name=label,
                                 hovertemplate="c = %{x}: %{y:.0%}<extra>" + label + "</extra>"))
    fig.add_hline(y=0.95, line=dict(color="#888", width=1, dash="dot"), annotation_text="nominal 95 %",
                  annotation_position="top left")
    fig.update_xaxes(title_text="Zahl der Spuren c (log)", type="log", tickmode="array", tickvals=list(C.GRID_C),
                     ticktext=[str(c) for c in C.GRID_C])
    fig.update_yaxes(title_text="Anteil der Intervalle, die den wahren Wert enthalten", tickformat=".0%", range=[0, 1.02])
    return _base(fig, 340)
