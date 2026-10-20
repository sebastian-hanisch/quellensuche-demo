"""Orakel-Test quellensuche-demo: das Verfahren (Rumor Centrality, Abstände, Wachstum, exakte Wahrscheinlichkeiten) gegen andere Rechenwege.

* R(v) = n! / Π T_u^v stimmt für jede Wurzel auf 60 Zufallsbäumen mit 3 bis 8 Knoten mit der **Zählung aller Infektionsreihenfolgen** (Permutationen mit zusammenhängenden Teilfolgen) überein.
* Das Umwurzeln in O(n) stimmt mit der Neuberechnung je Wurzel überein; auf Bäumen haben Rumor Centrality und Distanzzentralität dieselbe Kandidatenmenge (100 Bäume).
* Abstandssummen und Exzentrizitäten stimmen mit `networkx` überein (30 Netze).
* Die exakten Wahrscheinlichkeiten der befallenen Mengen (Teilmengen-Rechnung) stimmen mit der Häufigkeit aus vielen Wachstumsläufen überein, summieren sich zu 1 und sind für den Bayes-Schätzer die Obergrenze jedes Schätzers; Zufall trifft mit 1 / n_I.
* Die Trefferwahrscheinlichkeit der Rumor Centrality aus der exakten Rechnung stimmt mit der simulierten Trefferquote überein."""

import math

import numpy as np
import pytest

import qsu_algorithm as A
from conftest import make_adj, random_tree


def test_rumor_centrality_counts_the_infection_orders_on_random_trees():
    rng = np.random.default_rng(1)
    for _ in range(60):
        n = int(rng.integers(3, 9))
        tree = random_tree(rng, n)
        rc = A.rumor_centrality_log(tree)
        for v in range(n):
            assert math.exp(rc[v]) == pytest.approx(A.count_orders(tree, v), rel=1e-9), (n, v)


def test_rerooting_matches_the_per_root_computation_and_rc_equals_distance_on_trees():
    rng = np.random.default_rng(2)
    for _ in range(100):
        n = int(rng.integers(2, 40))
        tree = random_tree(rng, n)
        rc, dc, ecc = A.tree_scores(tree)
        assert rc == pytest.approx(A.rumor_centrality_log(tree)) and dc == A.distance_sums(tree) and ecc == A.eccentricities(tree)
        sets = A.estimate_sets(tree)
        assert sets["Rumor Centrality"] == sets["Distanzzentralität"]                      # auf Bäumen dasselbe Maß


def test_distance_sums_and_eccentricities_against_networkx():
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(3)
    for t in range(30):
        n = int(rng.integers(4, 25))
        pairs = {(i, int(rng.integers(i))) for i in range(1, n)}                           # zusammenhängend: Baum plus Zusatzkanten
        for _ in range(int(rng.integers(0, n))):
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            pairs.add((a, b))
        sub = make_adj(n, list(pairs))
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(pairs)
        lengths = dict(nx.all_pairs_shortest_path_length(g))
        assert A.distance_sums(sub) == [sum(lengths[v].values()) for v in range(n)], t
        assert A.eccentricities(sub) == [max(lengths[v].values()) for v in range(n)], t


def test_exact_set_probabilities_against_growth_frequencies(small_edges):
    n, edges = small_edges
    adj = make_adj(n, edges)
    rng = np.random.default_rng(4)
    for source, n_inf in ((0, 4), (3, 5), (6, 3)):
        exact = A.exact_set_probabilities(adj, source, n_inf)
        assert sum(exact.values()) == pytest.approx(1.0)
        runs = 30000
        counts = {}
        for _ in range(runs):
            mask = sum(1 << v for v in A.grow(adj, source, n_inf, rng))
            counts[mask] = counts.get(mask, 0) + 1
        for mask in set(exact) | set(counts):
            p = exact.get(mask, 0.0)
            se = (p * (1 - p) / runs) ** 0.5
            assert counts.get(mask, 0) / runs == pytest.approx(p, abs=4 * se + 1e-3), (source, n_inf, bin(mask))


def test_bayes_is_an_upper_bound_and_random_hits_one_over_n_inf(small_edges):
    n, edges = small_edges
    adj = make_adj(n, edges)
    for n_inf in (3, 4, 5):
        out = A.exact_hit_probabilities(adj, n_inf)
        assert out["Zufall"] == pytest.approx(1.0 / n_inf)
        for name in ("Rumor Centrality", "Distanzzentralität", "Jordan-Zentrum", "Zufall"):
            assert out[name] <= out["Bayes"] + 1e-12, (n_inf, name)
        assert out["Bayes"] > out["Zufall"]


def test_exact_hit_probability_of_rumor_centrality_against_simulation(small_edges):
    n, edges = small_edges
    adj = make_adj(n, edges)
    n_inf = 5
    exact = A.exact_hit_probabilities(adj, n_inf)
    rng = np.random.default_rng(6)
    runs = 20000
    hits = {name: 0.0 for name in ("Rumor Centrality", "Distanzzentralität", "Zufall")}
    for _ in range(runs):
        source = int(rng.integers(n))
        order = A.grow(adj, source, n_inf, rng)
        sets = A.estimate_sets(A.induced(adj, order))
        for name in hits:
            hits[name] += A.credit(source, order, sets[name])
    for name, total in hits.items():
        assert total / runs == pytest.approx(exact[name], abs=0.012), name


def test_the_regular_line_has_a_hit_rate_that_falls_with_size():
    rng = np.random.default_rng(7)
    rates = []
    for n_inf in (10, 40, 160):
        hits = 0.0
        for _ in range(300):
            sub, order = A.grow_regular_tree(2, n_inf, rng)
            hits += A.credit(0, order, A.estimate_sets(sub)["Rumor Centrality"])
        rates.append(hits / 300)
    assert rates[0] > rates[1] > rates[2] and rates[2] < 0.12                                # Satz nach Shah und Zaman: auf der Linie geht die Trefferquote gegen 0


def test_rumor_centrality_and_distance_agree_on_large_trees_despite_rounding_noise():
    rng = np.random.default_rng(11)
    for _ in range(300):
        n = int(rng.integers(60, 260))
        tree = random_tree(rng, n)
        sets = A.estimate_sets(tree)
        assert sets["Rumor Centrality"] == sets["Distanzzentralität"], n
    sub, _ = A.grow_regular_tree(3, 100, np.random.default_rng(5))
    assert A.estimate_sets(sub)["Rumor Centrality"] == A.estimate_sets(sub)["Distanzzentralität"]


def test_ties_on_regular_trees_are_not_split_by_rounding_noise(monkeypatch):
    """Regression: bei zwei gleich guten Zentren unterscheiden sich ln R um ~1e-11; ohne Toleranz (TOL = 0) weichen etwa 1 von 75 befallenen 3-regulären Bäumen mit 100 Knoten ab."""
    rng = np.random.default_rng(3)
    trees = [A.grow_regular_tree(3, 100, rng)[0] for _ in range(600)]
    assert all(A.estimate_sets(t)["Rumor Centrality"] == A.estimate_sets(t)["Distanzzentralität"] for t in trees)
    monkeypatch.setattr(A, "TOL", 0.0)
    assert any(A.estimate_sets(t)["Rumor Centrality"] != A.estimate_sets(t)["Distanzzentralität"] for t in trees)      # ohne Toleranz tritt der Fehler auf: der Test trennt
