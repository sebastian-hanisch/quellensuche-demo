"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import qsu_constants as C
from qsu_evaluation import de, pct

D_COLORS = {2: "#c4623b", 3: "#2F6B65", 4: "#6b5b95", 5: "#9d755d", 6: "#4c78a8"}
MAX_MARKS = 12   # Schätzer mit mehr gleich guten Kandidaten werden nicht einzeln markiert


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, bottom=10):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=bottom), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _pct_axis(fig, title):
    fig.update_yaxes(title_text=title, ticksuffix=" %", rangemode="tozero")
    return fig


def build_network(an, height=460):
    """Netz mit festen Positionen: grau = gesund, orange = befallen und ausgewertet, blass = befallen, aber nicht beobachtet bzw. nicht in der ausgewerteten Gruppe, rot umrandet = wahre Quelle,
    dazu je Schätzer Markierungen an seinen Kandidaten (Rumor Centrality Raute, Distanz Quadrat, Jordan Dreieck)."""
    inst = an.inst
    n = inst.n
    size = 5 if n > 200 else 8
    u, v = _edges(inst.adj)
    xs, ys = [], []
    for a, b in zip(u, v):
        xs += [inst.pos[a][0], inst.pos[b][0], None]
        ys += [inst.pos[a][1], inst.pos[b][1], None]
    infected = set(an.order)
    used = set(an.nodes)
    groups = (("gesund", [i for i in range(n) if i not in infected], C.COLOR_HEALTHY), ("befallen, nicht ausgewertet", [i for i in an.order if i not in used], "rgba(224,164,88,0.35)"),
              ("befallen, ausgewertet", [i for i in an.order if i in used], C.COLOR_INFECTED))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_EDGE, width=0.7), hoverinfo="skip", showlegend=False))
    for name, idx, color in groups:
        if idx:
            fig.add_trace(go.Scatter(x=[inst.pos[i][0] for i in idx], y=[inst.pos[i][1] for i in idx], mode="markers", name=name, marker=dict(size=size, color=color, line=dict(color="rgba(40,40,40,0.45)", width=0.4)),
                                     hovertext=[f"Betrieb {i}: {len(inst.adj[i])} Partner" for i in idx], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[inst.pos[an.source][0]], y=[inst.pos[an.source][1]], mode="markers", name="wahre Quelle", marker=dict(size=size + 12, color="rgba(0,0,0,0)", line=dict(color=C.COLOR_SOURCE, width=3)),
                             hovertext=[f"Quelle: Betrieb {an.source}"], hoverinfo="text"))
    for e in (C.E_RC, C.E_DC, C.E_J):
        cand = an.estimate_nodes(e)
        if cand and len(cand) <= MAX_MARKS:
            fig.add_trace(go.Scatter(x=[inst.pos[i][0] for i in cand], y=[inst.pos[i][1] for i in cand], mode="markers", name=f"Schätzung: {e}",
                                     marker=dict(size=size + 6, symbol=C.ESTIMATOR_SYMBOLS[e] + "-open", color=C.ESTIMATOR_COLORS[e], line=dict(width=2.5)),
                                     hovertext=[f"{e}: Betrieb {i}" for i in cand], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[-0.04, 1.04], y=[-0.04, 1.04], mode="markers", marker=dict(size=1, color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, autorange=True)
    fig.update_yaxes(visible=False, autorange=True, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.02), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _edges(adj):
    out_u, out_v = [], []
    for a in range(len(adj)):
        for b in adj[a]:
            if a < b:
                out_u.append(a)
                out_v.append(b)
    return out_u, out_v


def build_run_distances(an):
    """Mittlerer Abstand der Schätzungen zur wahren Quelle in diesem Lauf (Balken); ein Treffer hat Abstand 0."""
    names = list(C.ESTIMATORS)
    ys = [an.dists[e] for e in names]
    fig = go.Figure(go.Bar(x=names, y=ys, marker_color=[C.ESTIMATOR_COLORS[e] for e in names], text=[de(y, 1) for y in ys], textposition="outside"))
    fig.update_yaxes(title_text="Abstand zur Quelle (Schritte)", rangemode="tozero")
    return _base(fig, height=300, bottom=20)


def build_tree_hits(rows):
    """Trefferquote der Rumor Centrality über die Zahl der Befallenen je Verzweigung d; gestrichelt: Zufall (1 / Zahl der Befallenen)."""
    fig = go.Figure()
    ns = sorted({r["ni"] for r in rows})
    for d in sorted({r["d"] for r in rows}):
        ys = [100.0 * next(r for r in rows if r["d"] == d and r["ni"] == ni)["cells"][C.E_RC][0] for ni in ns]
        fig.add_trace(go.Scatter(x=ns, y=ys, mode="lines+markers", name=f"d = {d}" + (" (Linie)" if d == 2 else ""), line=dict(color=D_COLORS.get(d, C.COLOR_C), width=3)))
    fig.add_trace(go.Scatter(x=ns, y=[100.0 / ni for ni in ns], mode="lines", name="Zufall", line=dict(color="rgba(120,120,120,0.9)", dash="dash")))
    fig.update_xaxes(title_text="Zahl der Befallenen", type="log", tickvals=ns, ticktext=[str(x) for x in ns])
    _pct_axis(fig, "Trefferquote der Rumor Centrality")
    return _base(fig, height=360, bottom=50)


def _estimator_bars(xs, cells, idx, title, fmt, digits):
    fig = go.Figure()
    for e in C.ESTIMATORS:
        ys = [(100.0 if idx == 0 else 1.0) * c[e][idx] for c in cells]
        fig.add_trace(go.Bar(x=xs, y=ys, name=e, marker_color=C.ESTIMATOR_COLORS[e], text=[de(y, digits) for y in ys], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-20)
    if idx == 0:
        _pct_axis(fig, title)
    else:
        fig.update_yaxes(title_text=title, rangemode="tozero")
    return _base(fig, height=380, bottom=110)


def build_network_hits(rows, frac):
    mine = [r for r in rows if r["frac"] == frac]
    return _estimator_bars([r["typ"] for r in mine], [r["cells"] for r in mine], 0, f"Trefferquote bei {pct(frac)} % befallen", de, 1)


def build_network_distances(rows, frac):
    mine = [r for r in rows if r["frac"] == frac]
    return _estimator_bars([r["typ"] for r in mine], [r["cells"] for r in mine], 1, f"Abstand zur Quelle bei {pct(frac)} % befallen", de, 1)


def build_sizes(rows):
    """Trefferquote und mittlerer Abstand über die Zahl der Befallenen, je Schätzer."""
    fig = go.Figure()
    xs = [r["ni"] for r in rows]
    for e in C.ESTIMATORS:
        fig.add_trace(go.Scatter(x=xs, y=[100.0 * r["cells"][e][0] for r in rows], mode="lines+markers", name=e, line=dict(color=C.ESTIMATOR_COLORS[e], width=3)))
    fig.update_xaxes(title_text="Zahl der Befallenen", tickvals=xs)
    _pct_axis(fig, "Trefferquote")
    return _base(fig, height=320, bottom=50)


def build_size_distances(rows):
    fig = go.Figure()
    xs = [r["ni"] for r in rows]
    for e in C.ESTIMATORS:
        fig.add_trace(go.Scatter(x=xs, y=[r["cells"][e][1] for r in rows], mode="lines+markers", name=e, line=dict(color=C.ESTIMATOR_COLORS[e], width=3)))
    fig.update_xaxes(title_text="Zahl der Befallenen", tickvals=xs)
    fig.update_yaxes(title_text="Abstand zur Quelle (Schritte)", rangemode="tozero")
    return _base(fig, height=320, bottom=50)


def build_observation(rows):
    """Trefferquote je Schätzer bei vollständiger und unvollständiger Beobachtung."""
    xs = [f"{pct(r['share'])} % beobachtet" for r in rows]
    return _estimator_bars(xs, [r["cells"] for r in rows], 0, "Trefferquote", de, 1)


def build_exact(rows):
    """Exakte Trefferwahrscheinlichkeit von Bayes-Optimum und Schätzern je Zahl Befallener (kleine Netze)."""
    fig = go.Figure()
    xs = [f"{r['ni']} Befallene" for r in rows]
    series = (("Bayes-Optimum", "Bayes", C.COLOR_B),) + tuple((e, e, C.ESTIMATOR_COLORS[e]) for e in C.ESTIMATORS)
    for label, key, color in series:
        ys = [100.0 * r["hits"][key] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=ys, name=label, marker_color=color, text=[de(y, 1) for y in ys], textposition="outside"))
    fig.update_layout(barmode="group")
    _pct_axis(fig, "Trefferwahrscheinlichkeit (exakt)")
    return _base(fig, height=340, bottom=60)
