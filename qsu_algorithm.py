"""Kern: Wachstum eines Befalls (SI, Eden-Wachstum), Schätzer der Quelle aus der Menge der Befallenen (Rumor Centrality, Distanzzentralität, Jordan-Zentrum), Trefferquote und Abstand,
exakte Wahrscheinlichkeiten auf kleinen Netzen (Teilmengen-Rechnung) und der Bayes-Schätzer als Obergrenze.

Modell: ein Befall beginnt in einem Betrieb (Quelle) und wächst in Schritten: jede Kante zwischen einem befallenen und einem gesunden Betrieb hat dieselbe Rate, in jedem Schritt springt der Befall also über
eine zufällig gewählte Randkante (SI: wer befallen ist, bleibt es). Beobachtet wird nur die Menge der Befallenen. Rumor Centrality (Shah und Zaman 2011): für einen Baum mit n Knoten und einen Kandidaten v ist
R(v) = n! / Π_u T_u^v, mit T_u^v der Größe des Teilbaums von u bei Wurzel v; das ist die Zahl der Reihenfolgen, in denen der Befall von v aus genau diese Menge erreichen kann.

Netze sind Nachbarlisten `adj` (Tupel von sortierten Tupeln, Knoten 0 bis n - 1, ungerichtet). Zufall nur über `numpy.random.default_rng(Seed)`."""

import itertools
import math

import numpy as np

TOL = 1e-9   # relative Toleranz für „gleich gut“: ln R wird über viele Umwurzelungen summiert, mathematisch gleiche Werte unterscheiden sich um ~1e-11 (ein Runden auf feste Stellen trennt sie sonst zufällig)


# --- Wachstum --------------------------------------------------------------------------------------------------------------------------------------


def grow(adj, source, n_inf, rng):
    """Befallsreihenfolge im Netz: ab `source` springt der Befall in jedem Schritt über eine zufällig gewählte Randkante, bis `n_inf` Betriebe befallen sind oder kein Rand mehr da ist.
    Rückgabe die Betriebe in der Reihenfolge ihres Befalls (erster Eintrag = Quelle)."""
    n = len(adj)
    if not 0 <= source < n:
        raise ValueError("Quelle außerhalb des Netzes")
    if n_inf < 1:
        raise ValueError("mindestens ein befallener Betrieb")
    infected = {source}
    order = [source]
    front = [(source, w) for w in adj[source]]
    while len(order) < n_inf and front:
        i = int(rng.integers(len(front)))
        _, w = front[i]
        front[i] = front[-1]
        front.pop()
        if w in infected:
            continue
        infected.add(w)
        order.append(w)
        front.extend((w, x) for x in adj[w] if x not in infected)
    return order


def grow_regular_tree(d, n_inf, rng):
    """Befallener Baum einer unendlichen d-regulären Struktur (jeder Betrieb hat d Partner): Quelle ist Knoten 0, die Knoten sind nach der Befallsreihenfolge nummeriert.
    Rückgabe (Nachbarlisten des befallenen Baums, Reihenfolge = 0 bis n_inf - 1)."""
    if d < 2:
        raise ValueError("Verzweigung d mindestens 2")
    if n_inf < 1:
        raise ValueError("mindestens ein befallener Betrieb")
    nb = [[]]
    slots = [0] * d
    while len(nb) < n_inf:
        i = int(rng.integers(len(slots)))
        p = slots[i]
        slots[i] = slots[-1]
        slots.pop()
        c = len(nb)
        nb.append([p])
        nb[p].append(c)
        slots.extend([c] * (d - 1))
    return tuple(tuple(sorted(a)) for a in nb), list(range(n_inf))


def induced(adj, nodes):
    """Teilgraph der Betriebe `nodes` (Liste, Reihenfolge bestimmt die neue Nummerierung): Nachbarlisten in den neuen Nummern."""
    idx = {v: i for i, v in enumerate(nodes)}
    return tuple(tuple(sorted(idx[w] for w in adj[v] if w in idx)) for v in nodes)


def observe(order, share, rng):
    """Beobachtete Betriebe: bei share = 1 alle Befallenen, sonst eine zufällige Teilmenge dieses Anteils (mindestens zwei Betriebe), in der Reihenfolge des Befalls."""
    if not 0 < share <= 1:
        raise ValueError("Anteil muss in (0, 1] liegen")
    if share >= 1:
        return list(order)
    m = max(2, int(round(share * len(order)))) if len(order) >= 2 else len(order)
    pick = sorted(int(i) for i in rng.choice(len(order), size=m, replace=False))
    return [order[i] for i in pick]


def components(sub):
    """Zusammenhangskomponenten als Listen von Knoten, absteigend nach Größe (Gleichstand: kleinster Knoten)."""
    n = len(sub)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        comp = [s]
        seen[s] = True
        for x in comp:
            for w in sub[x]:
                if not seen[w]:
                    seen[w] = True
                    comp.append(w)
        comps.append(sorted(comp))
    return sorted(comps, key=lambda c: (-len(c), c[0]))


def bfs_distances(adj, s):
    """Abstände von s zu allen erreichbaren Knoten (Wörterbuch)."""
    dist = {s: 0}
    queue = [s]
    for x in queue:
        for y in adj[x]:
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    return dist


def is_tree(sub):
    """Zusammenhängend und ohne Kreis (Kantenzahl = Knotenzahl - 1)."""
    n = len(sub)
    if n == 0:
        return False
    m = sum(len(a) for a in sub) // 2
    return m == n - 1 and len(components(sub)[0]) == n


# --- Schätzer ----------------------------------------------------------------------------------------------------------------------------------------


def bfs_tree(sub, root):
    """BFS-Baum ab `root` (nur erreichbare Knoten): (Besuchsreihenfolge, Elternknoten, Abstände) als Wörterbuch bzw. Listen der Länge n."""
    n = len(sub)
    parent = [-1] * n
    dist = [-1] * n
    dist[root] = 0
    order = [root]
    for x in order:
        for w in sub[x]:
            if dist[w] < 0:
                dist[w] = dist[x] + 1
                parent[w] = x
                order.append(w)
    return order, parent, dist


def _subtree_sizes(order, parent):
    size = {x: 1 for x in order}
    for x in reversed(order[1:]):
        size[parent[x]] += size[x]
    return size


def rumor_centrality_log(sub):
    """ln R(v) für jeden Knoten v eines zusammenhängenden Teilgraphen: ln n! - Σ_u ln T_u^v, mit den Teilbaumgrößen des BFS-Baums mit Wurzel v. Auf Bäumen ist das der ML-Wert, sonst eine Näherung."""
    n = len(sub)
    lf = math.lgamma(n + 1)
    out = []
    for v in range(n):
        order, parent, _ = bfs_tree(sub, v)
        if len(order) < n:
            raise ValueError("Teilgraph nicht zusammenhängend")
        size = _subtree_sizes(order, parent)
        out.append(lf - sum(math.log(s) for s in size.values()))
    return out


def distance_sums(sub):
    """Summe der Abstände von v zu allen anderen Knoten, für jeden Knoten v eines zusammenhängenden Teilgraphen."""
    n = len(sub)
    out = []
    for v in range(n):
        order, _, dist = bfs_tree(sub, v)
        if len(order) < n:
            raise ValueError("Teilgraph nicht zusammenhängend")
        out.append(sum(dist))
    return out


def eccentricities(sub):
    """Größter Abstand von v zu einem anderen Knoten, für jeden Knoten v eines zusammenhängenden Teilgraphen."""
    n = len(sub)
    out = []
    for v in range(n):
        order, _, dist = bfs_tree(sub, v)
        if len(order) < n:
            raise ValueError("Teilgraph nicht zusammenhängend")
        out.append(max(dist))
    return out


def tree_scores(sub):
    """Rumor Centrality (ln), Abstandssummen und Exzentrizitäten eines Baums in O(n): Umwurzeln statt je Wurzel neu zu rechnen.
    Von Wurzel 0 auf Kind c: ln R(c) = ln R(p) + ln T_c - ln (n - T_c); D(c) = D(p) + n - 2 T_c; die Exzentrizität ist das Maximum der Abstände zu den beiden Enden eines längsten Wegs."""
    n = len(sub)
    order, parent, dist0 = bfs_tree(sub, 0)
    if len(order) < n or sum(len(a) for a in sub) // 2 != n - 1:
        raise ValueError("kein Baum")
    size = _subtree_sizes(order, parent)
    rc = [0.0] * n
    dc = [0] * n
    rc[0] = math.lgamma(n + 1) - sum(math.log(s) for s in size.values())
    dc[0] = sum(dist0)
    for x in order[1:]:
        p = parent[x]
        rc[x] = rc[p] + math.log(size[x]) - math.log(n - size[x])
        dc[x] = dc[p] + n - 2 * size[x]
    a = max(range(n), key=lambda v: (dist0[v], -v))
    _, _, da = bfs_tree(sub, a)
    b = max(range(n), key=lambda v: (da[v], -v))
    _, _, db = bfs_tree(sub, b)
    ecc = [max(da[v], db[v]) for v in range(n)]
    return rc, dc, ecc


def general_scores(sub):
    """Wie `rumor_centrality_log`, `distance_sums` und `eccentricities` zusammen, mit einem BFS je Knoten statt dreien."""
    n = len(sub)
    lf = math.lgamma(n + 1)
    rc, dc, ecc = [], [], []
    for v in range(n):
        order, parent, dist = bfs_tree(sub, v)
        if len(order) < n:
            raise ValueError("Teilgraph nicht zusammenhängend")
        size = _subtree_sizes(order, parent)
        rc.append(lf - sum(math.log(s) for s in size.values()))
        dc.append(sum(dist))
        ecc.append(max(dist))
    return rc, dc, ecc


def scores(sub):
    """(ln Rumor Centrality, Abstandssumme, Exzentrizität) je Knoten; Bäume über das schnelle Umwurzeln, sonst je Knoten ein BFS."""
    if len(sub) == 1:
        return [0.0], [0], [0]
    if is_tree(sub):
        return tree_scores(sub)
    return general_scores(sub)


def argmax_set(values, sign=1):
    """Knoten mit dem größten Wert von sign · values; Werte innerhalb der relativen Toleranz TOL des Maximums gelten als gleich gut."""
    vals = [sign * float(v) for v in values]
    best = max(vals)
    margin = TOL * max(1.0, abs(best))
    return [i for i, v in enumerate(vals) if best - v <= margin]


def estimate_sets(sub):
    """Kandidatenmengen je Schätzer (lokale Knotennummern): Rumor Centrality (größtes R), Distanzzentralität (kleinste Summe), Jordan-Zentrum (kleinste Exzentrizität), Zufall (alle)."""
    rc, dc, ecc = scores(sub)
    return {"Rumor Centrality": argmax_set(rc), "Distanzzentralität": argmax_set(dc, -1), "Jordan-Zentrum": argmax_set(ecc, -1), "Zufall": list(range(len(sub)))}


def credit(source, nodes, candidates):
    """Anteiliger Treffer: 1 / Zahl der Kandidaten, wenn die Quelle unter ihnen ist, sonst 0. `candidates` sind lokale Nummern in `nodes`."""
    if not candidates:
        return 0.0
    return 1.0 / len(candidates) if any(nodes[i] == source for i in candidates) else 0.0


def mean_distance(dist, nodes, candidates):
    """Mittlerer Abstand der Kandidaten zur Quelle (`dist`: Abstände von der Quelle im Gesamtnetz)."""
    if not candidates:
        return float("nan")
    return float(sum(dist[nodes[i]] for i in candidates) / len(candidates))


def source_rank(values, source_index, sign=1):
    """Rang der wahren Quelle: 1 + Zahl der Knoten mit strikt besserem Wert (um mehr als die Toleranz TOL); 1 heißt, sie hat den besten Wert (auch bei Gleichstand)."""
    mine = sign * float(values[source_index])
    margin = TOL * max(1.0, abs(mine))
    return 1 + sum(1 for v in values if sign * float(v) - mine > margin)


# --- Exakte Rechnung auf kleinen Netzen --------------------------------------------------------------------------------------------------------------


def exact_set_probabilities(adj, source, n_inf):
    """Wahrscheinlichkeit jeder Menge von n_inf Betrieben (als Bitmaske), nach n_inf - 1 Schritten ab `source` genau diese Menge befallen zu haben (Teilmengen-Rechnung, n höchstens 12)."""
    n = len(adj)
    if n > 12:
        raise ValueError("nur für höchstens 12 Betriebe")
    cur = {1 << source: 1.0}
    for _ in range(n_inf - 1):
        nxt = {}
        for mask, p in cur.items():
            cnt = {}
            for u in range(n):
                if mask >> u & 1:
                    for w in adj[u]:
                        if not mask >> w & 1:
                            cnt[w] = cnt.get(w, 0) + 1
            total = sum(cnt.values())
            for w, c in cnt.items():
                m2 = mask | (1 << w)
                nxt[m2] = nxt.get(m2, 0.0) + p * c / total
        cur = nxt
    return cur


def exact_hit_probabilities(adj, n_inf):
    """Exakte Trefferwahrscheinlichkeit (Quelle gleichverteilt) für den Bayes-Schätzer (Quelle mit der größten Wahrscheinlichkeit der beobachteten Menge, Gleichstand anteilig) und die vier Schätzer.
    Rückgabe Wörterbuch Name -> Wahrscheinlichkeit; der Bayes-Wert ist die Obergrenze für jeden Schätzer."""
    n = len(adj)
    f = [exact_set_probabilities(adj, v, n_inf) for v in range(n)]
    masks = set()
    for d in f:
        masks |= set(d)
    out = {"Bayes": 0.0, "Rumor Centrality": 0.0, "Distanzzentralität": 0.0, "Jordan-Zentrum": 0.0, "Zufall": 0.0}
    for mask in masks:
        joint = {v: f[v].get(mask, 0.0) / n for v in range(n)}
        best = max(joint.values())
        out["Bayes"] += best
        nodes = [i for i in range(n) if mask >> i & 1]
        sets = estimate_sets(induced(adj, nodes))
        for name, cand in sets.items():
            out[name] += sum(joint[nodes[i]] for i in cand) / len(cand)
    return out


def count_orders(sub, root):
    """Zahl der Reihenfolgen, in denen ein Befall ab `root` alle Knoten des Teilgraphen erreicht, so dass jede Teilfolge zusammenhängend ist (Aufzählung aller Permutationen, höchstens 9 Knoten): das Orakel für R(root)."""
    n = len(sub)
    if n > 9:
        raise ValueError("nur für höchstens 9 Knoten")
    others = [x for x in range(n) if x != root]
    total = 0
    for perm in itertools.permutations(others):
        seen = {root}
        ok = True
        for x in perm:
            if not any(y in seen for y in sub[x]):
                ok = False
                break
            seen.add(x)
        total += ok
    return total
