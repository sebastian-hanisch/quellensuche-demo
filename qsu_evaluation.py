"""Kennzahlen, Messreihen und Verdikte der Demo „Quellensuche“. Verdikte sind reine Funktionen (Art, Text); jede Zahl im Text kommt aus der Rechnung.

Protokoll: SI als Eden-Wachstum (jede Kante zwischen Befallenen und Gesunden hat dieselbe Rate; in jedem Schritt springt der Befall über eine zufällige Randkante). Beobachtet wird die Menge der Befallenen
(bei unvollständiger Beobachtung eine zufällige Teilmenge; ausgewertet wird dann ihre größte zusammenhängende Gruppe). Schätzer: Rumor Centrality, Distanzzentralität, Jordan-Zentrum, Zufall; ein Lauf
liefert je Schätzer den anteiligen Treffer und den mittleren Abstand der Kandidaten zur Quelle. Jede Messreihe mittelt über Läufe auf festen Netzen (`LIVE_SEEDS` in der App, `SWEEP_SEEDS` für die
README-Zahlen); jeder Lauf hat seinen eigenen Zufallsstrom (Seed, Netz, Lauf)."""

import math
from dataclasses import dataclass

import numpy as np

import qsu_algorithm as A
import qsu_constants as C
from qsu_scenario import Instance, layout_force, make_instance

NAN = float("nan")


def de(x, digits=1) -> str:
    """Dezimalzahl mit Komma (deutsche Texte); eine nicht definierte Zahl erscheint als Gedankenstrich."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x, digits=0) -> str:
    return de(100.0 * x, digits)


def _rng(*parts):
    return np.random.default_rng([int(p) for p in parts])


# --- Ein Lauf -------------------------------------------------------------------------------------------------------------------------------------


def evaluate_order(adj, source, order, obs, rng):
    """Beobachtung, Schätzer und Maße zu einer Befallsreihenfolge: Rückgabe (beobachtete Betriebe, Knoten der größten zusammenhängenden Gruppe, Teilgraph, Kandidatenmengen, Abstände von der Quelle)."""
    seen = A.observe(order, obs, rng)
    comp = A.components(A.induced(adj, seen))[0]
    nodes = [seen[i] for i in comp]
    sub = A.induced(adj, nodes)
    return seen, nodes, sub, A.estimate_sets(sub), A.bfs_distances(adj, source)


def trial(adj, source, ni, obs, rng):
    """Ein Lauf auf einem Netz: je Schätzer (Treffer, mittlerer Abstand), dazu der Anteil der beobachteten Betriebe in der ausgewerteten Gruppe."""
    order = A.grow(adj, source, ni, rng)
    seen, nodes, sub, sets, dist = evaluate_order(adj, source, order, obs, rng)
    return {"cells": {e: (A.credit(source, nodes, sets[e]), A.mean_distance(dist, nodes, sets[e])) for e in C.ESTIMATORS}, "connected": len(nodes) / len(seen), "infected": len(order)}


def tree_trial(d, ni, obs, rng):
    """Ein Lauf auf dem befallenen Baum einer unendlichen d-regulären Struktur (Quelle = Knoten 0)."""
    adj, order = A.grow_regular_tree(d, ni, rng)
    seen, nodes, sub, sets, dist = evaluate_order(adj, 0, order, obs, rng)
    return {"cells": {e: (A.credit(0, nodes, sets[e]), A.mean_distance(dist, nodes, sets[e])) for e in C.ESTIMATORS}, "connected": len(nodes) / len(seen), "infected": len(order)}


def _mean_cells(trials):
    return {e: (float(np.mean([t["cells"][e][0] for t in trials])), float(np.mean([t["cells"][e][1] for t in trials]))) for e in C.ESTIMATORS}


# --- Hauptanalyse -----------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    netztyp: str = C.T_RANDOM
    n: int = C.DEFAULT_N
    c: int = C.DEFAULT_C
    seed: int = C.DEFAULT_SEED
    d: int = C.DEFAULT_D
    ni: int = C.DEFAULT_NI
    obs: int = C.DEFAULT_OBS


@dataclass(frozen=True, eq=False)
class Analysis:
    inst: Instance
    settings: Settings
    source: int
    order: tuple            # alle Befallenen in der Reihenfolge des Befalls
    seen: tuple             # beobachtete Betriebe
    nodes: tuple            # Betriebe der ausgewerteten (größten zusammenhängenden) Gruppe
    sets: dict              # Schätzer -> Kandidaten (lokale Nummern in `nodes`)
    dist: dict              # Abstände von der Quelle im Gesamtnetz
    hits: dict              # Schätzer -> anteiliger Treffer
    dists: dict             # Schätzer -> mittlerer Abstand der Kandidaten zur Quelle
    ranks: dict             # Schätzer -> Rang der wahren Quelle (1 = bester Wert), nur wenn sie in der Gruppe liegt
    values: dict            # Schätzer -> Wert je Betrieb der Gruppe (ln R, Abstandssumme, Exzentrizität)

    @property
    def ni(self) -> int:
        return len(self.order)

    def estimate_nodes(self, estimator) -> list:
        return [self.nodes[i] for i in self.sets[estimator]]


def analyse(settings: Settings) -> Analysis:
    rng = _rng(settings.seed, 1)
    if settings.netztyp == C.T_TREE:
        adj, order = A.grow_regular_tree(settings.d, settings.ni, rng)
        inst = Instance(settings.netztyp, adj, layout_force(adj, int(settings.seed) % 1_000_003 + 19))
        source = 0
    else:
        inst = make_instance(settings.netztyp, settings.n, settings.c, settings.seed, layout=True)
        adj = inst.adj
        source = int(rng.integers(inst.n))
        order = A.grow(adj, source, settings.ni, rng)
    seen, nodes, sub, sets, dist = evaluate_order(adj, source, order, settings.obs / 100.0, rng)
    rc, dc, ecc = A.scores(sub)
    values = {C.E_RC: rc, C.E_DC: dc, C.E_J: ecc}
    pos = {v: i for i, v in enumerate(nodes)}
    ranks = {}
    if source in pos:
        ranks = {C.E_RC: A.source_rank(rc, pos[source]), C.E_DC: A.source_rank(dc, pos[source], -1), C.E_J: A.source_rank(ecc, pos[source], -1)}
    hits = {e: A.credit(source, nodes, sets[e]) for e in C.ESTIMATORS}
    dists = {e: A.mean_distance(dist, nodes, sets[e]) for e in C.ESTIMATORS}
    return Analysis(inst, settings, source, tuple(order), tuple(seen), tuple(nodes), sets, dist, hits, dists, ranks, values)


def verdict_main(an: Analysis) -> tuple[str, str]:
    """Verdikt dieses Laufs: wer hat die Quelle getroffen, und wie weit liegen die Schätzungen daneben?"""
    if an.ni == 1:
        return "info", "Es ist nur ein Betrieb befallen: er ist die Quelle."
    head = f"{an.ni} Betriebe sind befallen" + (f", {len(an.seen)} davon beobachtet, ausgewertet wird die größte zusammenhängende Gruppe mit {len(an.nodes)}." if len(an.seen) < an.ni else ".")
    parts = []
    for e in (C.E_RC, C.E_DC, C.E_J):
        k = len(an.sets[e])
        hit = an.hits[e]
        verb = "trifft die Quelle" if hit >= 1.0 else (f"hat die Quelle unter {k} gleich guten Kandidaten (Treffer {de(hit, 2)})" if hit > 0 else "verfehlt die Quelle")
        parts.append(f"{e} {verb}, Abstand {de(an.dists[e], 1)}" + (f", Rang {an.ranks[e]}" if e in an.ranks else ""))
    kind = "success" if any(an.hits[e] > 0 for e in (C.E_RC, C.E_DC, C.E_J)) else "info"
    tail = "" if an.source in an.nodes else " Die Quelle gehört nicht zur ausgewerteten Gruppe."
    return kind, head + " " + "; ".join(parts) + f". Zufall käme im Mittel auf {de(1.0 / len(an.nodes), 3)} Treffer und Abstand {de(an.dists[C.E_RAND], 1)}." + tail


# --- E1: reguläre Bäume -------------------------------------------------------------------------------------------------------------------------------


def tree_rows(ds=C.TREE_DS, ns=C.TREE_NS, runs=C.TREE_RUNS, first_seed=100000) -> list[dict]:
    """Je Verzweigung d und Zahl Befallener n: mittlere Trefferquote und mittlerer Abstand je Schätzer über `runs` Läufe auf befallenen regulären Bäumen."""
    rows = []
    for d in ds:
        for ni in ns:
            trials = [tree_trial(d, ni, 1.0, _rng(first_seed, d, ni, r)) for r in range(runs)]
            rows.append({"d": d, "ni": ni, "cells": _mean_cells(trials), "runs": runs})
    return rows


def tree_cell(rows, d, ni):
    return next(r for r in rows if r["d"] == d and r["ni"] == ni)["cells"]


def verdict_trees(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E1: auf der Linie fällt die Trefferquote, ab d = 3 bleibt sie; Rumor Centrality und Distanz sind auf Bäumen dasselbe Maß."""
    ds = sorted({r["d"] for r in rows})
    ns = sorted({r["ni"] for r in rows})
    if not rows or len(ns) < 2:
        return "warning", "Keine Läufe."
    lo, hi = ns[0], ns[-1]
    same = all(r["cells"][C.E_RC][0] == r["cells"][C.E_DC][0] for r in rows)
    parts = []
    for d in ds:
        a, b = tree_cell(rows, d, lo)[C.E_RC][0], tree_cell(rows, d, hi)[C.E_RC][0]
        parts.append(f"d = {d}{' (Linie)' if d == 2 else ''}: {de(a, 3)} bei {lo}, {de(b, 3)} bei {hi} Befallenen")
    return "info", ("Trefferquote der Rumor Centrality auf befallenen Bäumen: " + "; ".join(parts) + ". "
                    + ("Rumor Centrality und Distanzzentralität treffen in jeder Zelle gleich oft: auf Bäumen sind es dieselben Kandidaten." if same else "Rumor Centrality und Distanzzentralität unterscheiden sich in mindestens einer Zelle."))


# --- E2: Netztypen ------------------------------------------------------------------------------------------------------------------------------------


def network_rows(n, c, fracs=C.NET_FRACS, seeds=C.LIVE_SEEDS, runs=C.NET_RUNS, types=C.GENERATED_TYPES) -> list[dict]:
    """Je Netztyp und befallenem Anteil: mittlere Trefferquote und mittlerer Abstand je Schätzer über die festen Netze und `runs` Läufe je Netz (Quelle zufällig)."""
    rows = []
    for typ in types:
        for frac in fracs:
            trials = []
            for seed in seeds:
                adj = make_instance(typ, n, c, seed, layout=False).adj
                ni = max(2, int(round(frac * len(adj))))
                for r in range(runs):
                    rng = _rng(seed, int(round(frac * 100)), r)
                    trials.append(trial(adj, int(rng.integers(len(adj))), ni, 1.0, rng))
            rows.append({"typ": typ, "frac": frac, "cells": _mean_cells(trials), "runs": len(trials)})
    return rows


def verdict_networks(rows: list[dict], frac) -> tuple[str, str]:
    """Verdikt zu E2: wo trifft die Quelle über Zufall, wo schlägt die Distanz die Rumor Centrality, wo bleibt nur der Abstand?"""
    mine = [r for r in rows if r["frac"] == frac]
    if not mine:
        return "warning", "Keine Messung für diesen Anteil."
    best = max(mine, key=lambda r: r["cells"][C.E_RC][0])
    near_random = [r["typ"] for r in mine if max(r["cells"][e][0] for e in (C.E_RC, C.E_DC, C.E_J)) < 1.5 * r["cells"][C.E_RAND][0]]
    dist_better = [r["typ"] for r in mine if r["cells"][C.E_DC][0] > 1.2 * r["cells"][C.E_RC][0] and r["cells"][C.E_DC][0] - r["cells"][C.E_RC][0] > 0.01]
    text = (f"Bei {pct(frac)} % befallenen Betrieben trifft die Rumor Centrality am häufigsten in „{best['typ']}“ ({pct(best['cells'][C.E_RC][0], 1)} % gegen {pct(best['cells'][C.E_RAND][0], 1)} % bei Zufall, mittlerer Abstand "
            f"{de(best['cells'][C.E_RC][1], 2)} gegen {de(best['cells'][C.E_RAND][1], 2)}). ")
    text += ("Kaum über Zufall (unter dem 1,5-Fachen) liegt kein Schätzer in: " + ", ".join(f"„{t}“" for t in near_random) + ". ") if near_random else "In jedem Netztyp liegt der beste Schätzer über dem 1,5-Fachen von Zufall. "
    text += ("Die Distanzzentralität trifft mindestens 20 % öfter als die Rumor Centrality in: " + ", ".join(f"„{t}“" for t in dist_better) + ".") if dist_better else "Die Rumor Centrality ist in keinem Netztyp deutlich schlechter als die Distanzzentralität."
    return "info", text


# --- E3: Ausbreitungsgröße ---------------------------------------------------------------------------------------------------------------------------


def size_rows(typ, n, c, sizes=C.SIZES, seeds=C.LIVE_SEEDS, runs=C.NET_RUNS) -> list[dict]:
    """Im Netztyp `typ`: mittlere Trefferquote und Abstand je Schätzer über die Zahl der Befallenen (Läufe wie in E2; Größen über der Netzgröße entfallen)."""
    rows = []
    for ni in sizes:
        if typ == C.T_TREE or ni >= n:
            continue
        trials = []
        for seed in seeds:
            adj = make_instance(typ, n, c, seed, layout=False).adj
            for r in range(runs):
                rng = _rng(seed, ni, r, 3)
                trials.append(trial(adj, int(rng.integers(len(adj))), ni, 1.0, rng))
        rows.append({"typ": typ, "ni": ni, "cells": _mean_cells(trials), "runs": len(trials)})
    return rows


def verdict_sizes(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E3: wie ändern sich Trefferquote und Abstand, wenn mehr Betriebe befallen sind?"""
    if len(rows) < 2:
        return "warning", "Zu wenige Messpunkte."
    a, b = rows[0], rows[-1]
    ra, rb = a["cells"][C.E_RC], b["cells"][C.E_RC]
    trend = "fällt" if rb[0] < ra[0] else "steigt"
    return "info", (f"In „{a['typ']}“ {trend} die Trefferquote der Rumor Centrality von {pct(ra[0], 1)} % bei {a['ni']} Befallenen auf {pct(rb[0], 1)} % bei {b['ni']}; der mittlere Abstand zur Quelle geht von {de(ra[1], 2)} auf {de(rb[1], 2)}. "
                    f"Zufall trifft {pct(a['cells'][C.E_RAND][0], 1)} % bzw. {pct(b['cells'][C.E_RAND][0], 1)} %.")


# --- E4: Unvollständigkeit -----------------------------------------------------------------------------------------------------------------------------


def observation_rows(typ, n, c, shares=C.OBS_SHARES, frac=0.3, seeds=C.LIVE_SEEDS, runs=C.NET_RUNS) -> list[dict]:
    """Im Netztyp `typ` bei befallenem Anteil `frac`: Trefferquote und Abstand je Schätzer, wenn nur ein Anteil der Befallenen beobachtet wird (größte zusammenhängende Gruppe der Beobachteten), und der Anteil der Beobachteten in dieser Gruppe."""
    rows = []
    for share in shares:
        trials = []
        for seed in seeds:
            adj = make_instance(typ, n, c, seed, layout=False).adj
            ni = max(3, int(round(frac * len(adj))))
            for r in range(runs):
                rng = _rng(seed, int(round(100 * share)), r, 4)
                trials.append(trial(adj, int(rng.integers(len(adj))), ni, share, rng))
        rows.append({"typ": typ, "share": share, "cells": _mean_cells(trials), "connected": float(np.mean([t["connected"] for t in trials])), "runs": len(trials)})
    return rows


def verdict_observation(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E4: was bleibt von der Quellensuche, wenn nur ein Teil der Befallenen bekannt ist?"""
    if len(rows) < 2:
        return "warning", "Zu wenige Messpunkte."
    parts = [f"bei {pct(r['share'])} % beobachtet: Treffer {pct(r['cells'][C.E_RC][0], 1)} %, Abstand {de(r['cells'][C.E_RC][1], 2)}, im Mittel {pct(r['connected'])} % der Beobachteten in der ausgewerteten Gruppe" for r in rows]
    return "info", f"Rumor Centrality in „{rows[0]['typ']}“: " + "; ".join(parts) + ". Das Verfahren erwartet eine zusammenhängende befallene Menge; fehlende Betriebe lassen sie zerfallen."


# --- E5: exakt auf kleinen Netzen -----------------------------------------------------------------------------------------------------------------------


def small_network(seed, n=C.EXACT_N, m=C.EXACT_M):
    """Verbundenes Zufallsnetz mit n Knoten und m verschiedenen Kanten (Ablehnungsverfahren). Rückgabe (n, Kantenliste)."""
    rng = np.random.default_rng(int(seed))
    while True:
        pairs = set()
        while len(pairs) < m:
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            pairs.add((a, b))
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in pairs:
            parent[find(a)] = find(b)
        if len({find(i) for i in range(n)}) == 1:
            return n, sorted(pairs)


def exact_rows(nets=C.EXACT_NETS, nis=C.EXACT_NIS, first_seed=100000) -> list[dict]:
    """Auf kleinen Netzen: exakte Trefferwahrscheinlichkeit des Bayes-Schätzers und der vier Schätzer je Zahl Befallener (Mittel über die Netze)."""
    rows = []
    for ni in nis:
        per = []
        for i in range(nets):
            n, edges = small_network(first_seed + i)
            adj = tuple(tuple(sorted(w for a, b in edges for u, w in ((a, b), (b, a)) if u == v)) for v in range(n))
            per.append(A.exact_hit_probabilities(adj, ni))
        keys = ("Bayes",) + C.ESTIMATORS
        rows.append({"ni": ni, "nets": nets, "hits": {k: float(np.mean([p[k] for p in per])) for k in keys}, "bayes_wins": sum(1 for p in per if all(p["Bayes"] >= p[e] - 1e-12 for e in C.ESTIMATORS))})
    return rows


def verdict_exact(rows: list[dict]) -> tuple[str, str]:
    """Verdikt zu E5: wie weit liegt die beste mögliche Trefferquote (Bayes) über den Schätzern und über Zufall?"""
    if not rows:
        return "warning", "Keine Läufe."
    parts = []
    for r in rows:
        h = r["hits"]
        parts.append(f"{r['ni']} Befallene: Bayes-Optimum {pct(h['Bayes'], 1)} %, Rumor Centrality {pct(h[C.E_RC], 1)} %, Distanz {pct(h[C.E_DC], 1)} %, Zufall {pct(h[C.E_RAND], 1)} %")
    return "info", f"Auf {rows[0]['nets']} kleinen Netzen ({C.EXACT_N} Betriebe, {C.EXACT_M} Partnerschaften), exakt gerechnet: " + "; ".join(parts) + ". Selbst das Bayes-Optimum verfehlt die Quelle meist."
