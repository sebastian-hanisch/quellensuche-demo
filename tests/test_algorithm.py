"""Einheiten des Kerns an Mini-Instanzen mit von Hand gerechneten Werten (DEMO-PLAYBOOK §1b)."""

import math
import warnings

import numpy as np
import pytest

import qsu_algorithm as A
from conftest import make_adj


def _rng(seed=1):
    return np.random.default_rng(seed)


# --- Wachstum -----------------------------------------------------------------------------------------------------------------------------------


def test_growth_returns_n_inf_distinct_connected_nodes_starting_at_the_source(small_edges):
    n, edges = small_edges
    adj = make_adj(n, edges)
    for seed in range(30):
        order = A.grow(adj, 3, 6, _rng(seed))
        assert order[0] == 3 and len(order) == len(set(order)) == 6
        for i, v in enumerate(order[1:], 1):
            assert any(w in order[:i] for w in adj[v])                                       # jeder neue Betrieb hat einen schon befallenen Partner


def test_growth_stops_when_the_component_is_exhausted_and_is_deterministic():
    adj = make_adj(5, [(0, 1), (1, 2), (3, 4)])
    assert sorted(A.grow(adj, 0, 10, _rng())) == [0, 1, 2]
    assert A.grow(adj, 0, 3, _rng(4)) == A.grow(adj, 0, 3, _rng(4))
    assert A.grow(adj, 3, 1, _rng()) == [3]


def test_growth_on_a_path_from_the_end_is_forced():
    adj = make_adj(4, [(0, 1), (1, 2), (2, 3)])
    assert A.grow(adj, 0, 4, _rng(7)) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        A.grow(adj, 9, 2, _rng())
    with pytest.raises(ValueError):
        A.grow(adj, 0, 0, _rng())


def test_first_step_of_the_growth_picks_each_edge_with_equal_probability():
    adj = make_adj(4, [(0, 1), (0, 2), (0, 3)])                                              # Stern: drei gleich wahrscheinliche erste Schritte
    counts = np.zeros(4)
    rng = _rng(2)
    for _ in range(6000):
        counts[A.grow(adj, 0, 2, rng)[1]] += 1
    assert counts[1:] / 6000 == pytest.approx([1 / 3] * 3, abs=0.03)


def test_regular_tree_has_the_right_degrees_and_root_zero():
    for d in (2, 3, 4):
        sub, order = A.grow_regular_tree(d, 40, _rng(d))
        assert order == list(range(40)) and A.is_tree(sub)
        degs = [len(a) for a in sub]
        assert max(degs) <= d and len(sub[0]) <= d
    line, _ = A.grow_regular_tree(2, 30, _rng(5))
    assert max(len(a) for a in line) <= 2                                                    # Linie
    with pytest.raises(ValueError):
        A.grow_regular_tree(1, 5, _rng())
    assert A.grow_regular_tree(3, 1, _rng())[0] == ((),)


def test_induced_subgraph_renumbers_and_observe_keeps_a_share():
    adj = make_adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    assert A.induced(adj, [4, 2, 3]) == ((2,), (2,), (0, 1))                                  # 4 -> 0, 2 -> 1, 3 -> 2: Kanten 3-4 und 2-3
    order = [7, 3, 9, 1, 5, 8, 2, 6, 0, 4]
    assert A.observe(order, 1.0, _rng()) == order
    seen = A.observe(order, 0.5, _rng(3))
    assert len(seen) == 5 and set(seen) <= set(order) and [order.index(x) for x in seen] == sorted(order.index(x) for x in seen)
    assert len(A.observe(order, 0.01, _rng())) == 2
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            A.observe(order, bad, _rng())


def test_components_sorted_by_size_and_is_tree():
    sub = make_adj(7, [(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)])
    assert A.components(sub) == [[0, 1, 2], [3, 4, 5], [6]]                                   # Gleichstand: kleinster Knoten zuerst
    assert not A.is_tree(sub) and A.is_tree(make_adj(3, [(0, 1), (1, 2)])) and not A.is_tree(make_adj(3, [(0, 1), (1, 2), (0, 2)])) and not A.is_tree(())
    assert A.bfs_distances(sub, 0) == {0: 0, 1: 1, 2: 2}


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------------


def test_rumor_centrality_hand_values(path3, star5, line5):
    assert [round(math.exp(x), 9) for x in A.rumor_centrality_log(path3)] == pytest.approx([1, 2, 1])
    assert [round(math.exp(x), 9) for x in A.rumor_centrality_log(star5)] == pytest.approx([24, 6, 6, 6, 6])
    assert [round(math.exp(x), 9) for x in A.rumor_centrality_log(line5)] == pytest.approx([1, 4, 6, 4, 1])


def test_tree_scores_agree_with_the_general_computation(path3, star5, line5):
    for sub in (path3, star5, line5):
        rc, dc, ecc = A.tree_scores(sub)
        assert rc == pytest.approx(A.rumor_centrality_log(sub)) and dc == A.distance_sums(sub) and ecc == A.eccentricities(sub)
    assert A.tree_scores(line5)[1] == [10, 7, 6, 7, 10] and A.tree_scores(line5)[2] == [4, 3, 2, 3, 4]
    with pytest.raises(ValueError):
        A.tree_scores(make_adj(3, [(0, 1), (1, 2), (0, 2)]))


def test_distance_sums_and_eccentricities_on_a_triangle_and_a_square(triangle):
    assert A.distance_sums(triangle) == [2, 2, 2] and A.eccentricities(triangle) == [1, 1, 1]
    square = make_adj(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert A.distance_sums(square) == [4, 4, 4, 4] and A.eccentricities(square) == [2, 2, 2, 2]
    with pytest.raises(ValueError):
        A.rumor_centrality_log(make_adj(3, [(0, 1)]))
    with pytest.raises(ValueError):
        A.distance_sums(make_adj(3, [(0, 1)]))
    with pytest.raises(ValueError):
        A.eccentricities(make_adj(3, [(0, 1)]))


def test_scores_of_a_single_node():
    assert A.scores(((),)) == ([0.0], [0], [0])


def test_estimate_sets_on_the_line_name_the_middle(line5):
    sets = A.estimate_sets(line5)
    assert sets["Rumor Centrality"] == [2] and sets["Distanzzentralität"] == [2] and sets["Jordan-Zentrum"] == [2] and sets["Zufall"] == [0, 1, 2, 3, 4]
    even = make_adj(4, [(0, 1), (1, 2), (2, 3)])
    assert A.estimate_sets(even)["Rumor Centrality"] == [1, 2] and A.estimate_sets(even)["Jordan-Zentrum"] == [1, 2]


def test_argmax_set_rounds_noise_and_respects_the_sign():
    assert A.argmax_set([1.0, 1.0 + 1e-14, 0.5]) == [0, 1] and A.argmax_set([3, 1, 1], sign=-1) == [1, 2]


def test_credit_distance_and_rank():
    nodes = [10, 11, 12]
    assert A.credit(11, nodes, [1]) == 1.0 and A.credit(11, nodes, [0, 1, 2]) == pytest.approx(1 / 3) and A.credit(11, nodes, [0, 2]) == 0.0 and A.credit(11, nodes, []) == 0.0
    dist = {10: 3, 11: 0, 12: 1}
    assert A.mean_distance(dist, nodes, [0, 2]) == 2.0 and math.isnan(A.mean_distance(dist, nodes, []))
    assert A.source_rank([5.0, 3.0, 4.0], 2) == 2 and A.source_rank([5.0, 3.0, 4.0], 1, sign=-1) == 1 and A.source_rank([2.0, 2.0, 1.0], 1) == 1


def test_components_after_observation_pick_the_largest_group():
    adj = make_adj(6, [(0, 1), (1, 2), (3, 4)])
    sub = A.induced(adj, [0, 1, 2, 3])
    assert A.components(sub)[0] == [0, 1, 2]


# --- Exakte Rechnung -----------------------------------------------------------------------------------------------------------------------------


def test_exact_set_probabilities_by_hand(path3):
    # ab Knoten 1 mit zwei Befallenen: {0,1} und {1,2} je 1/2; ab Knoten 0: nur {0,1}
    assert A.exact_set_probabilities(path3, 1, 2) == {0b011: 0.5, 0b110: 0.5}
    assert A.exact_set_probabilities(path3, 0, 2) == {0b011: 1.0}
    assert A.exact_set_probabilities(path3, 1, 3) == {0b111: 1.0}
    with pytest.raises(ValueError):
        A.exact_set_probabilities(make_adj(13, [(i, i + 1) for i in range(12)]), 0, 3)


def test_exact_set_probabilities_sum_to_one_and_follow_the_edge_rule(small_edges):
    n, edges = small_edges
    adj = make_adj(n, edges)
    for v in (0, 4):
        probs = A.exact_set_probabilities(adj, v, 4)
        assert sum(probs.values()) == pytest.approx(1.0) and all(m >> v & 1 and bin(m).count("1") == 4 for m in probs)
    # zwei Befallene ab Knoten 4 (Partner 3 und 5): je 1/2
    two = A.exact_set_probabilities(adj, 4, 2)
    assert two == {(1 << 4) | (1 << 3): 0.5, (1 << 4) | (1 << 5): 0.5}


def test_exact_hit_probabilities_on_the_path(path3):
    # zwei Befallene auf dem Pfad: Menge {0,1}: Quelle 0 (Wkt. 1/3 · 1) oder 1 (1/3 · 1/2); Bayes wählt 0 → 1/3; analog {1,2}; Menge {1,2} ab 2 oder 1
    out = A.exact_hit_probabilities(path3, 2)
    assert out["Bayes"] == pytest.approx(1 / 3 + 1 / 3)                                      # je Menge die wahrscheinlichere Quelle: 1/3 + 1/3
    assert out["Zufall"] == pytest.approx(0.5)                                               # jeder Befallene mit 1/2
    for name in ("Rumor Centrality", "Distanzzentralität", "Jordan-Zentrum"):
        assert out[name] <= out["Bayes"] + 1e-12 and out[name] == pytest.approx(0.5)         # beide Befallene sind gleichberechtigt: Gleichstand, anteilig 1/2 von (1/3 + 1/6)·2


def test_count_orders_hand_values(path3, star5, line5):
    assert [A.count_orders(path3, r) for r in range(3)] == [1, 2, 1]
    assert [A.count_orders(star5, r) for r in range(5)] == [24, 6, 6, 6, 6]
    assert [A.count_orders(line5, r) for r in range(5)] == [1, 4, 6, 4, 1]
    with pytest.raises(ValueError):
        A.count_orders(make_adj(10, [(i, i + 1) for i in range(9)]), 0)


def test_no_warnings_in_the_core(line5):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A.estimate_sets(line5)
        A.exact_hit_probabilities(line5, 3)
        A.grow(line5, 2, 4, _rng())


def test_general_scores_agree_with_the_separate_functions(triangle):
    square_with_chord = make_adj(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    for sub in (triangle, square_with_chord):
        rc, dc, ecc = A.general_scores(sub)
        assert rc == pytest.approx(A.rumor_centrality_log(sub)) and dc == A.distance_sums(sub) and ecc == A.eccentricities(sub)
        assert A.scores(sub)[1] == dc
    with pytest.raises(ValueError):
        A.general_scores(make_adj(3, [(0, 1)]))


# --- Lücken aus der Mutationsprüfung ---------------------------------------------------------------------------------------------------------


def test_count_orders_accepts_nine_nodes_and_exact_probabilities_accept_twelve():
    assert A.count_orders(make_adj(9, [(i, i + 1) for i in range(8)]), 0) == 1                 # Linie ab dem Rand: eine Reihenfolge
    assert A.count_orders(make_adj(9, [(i, i + 1) for i in range(8)]), 4) == 70                # 8! / (4! 4!)
    twelve = make_adj(12, [(i, i + 1) for i in range(11)])
    assert sum(A.exact_set_probabilities(twelve, 5, 3).values()) == pytest.approx(1.0)


def test_exact_probabilities_count_every_edge_into_the_infected_set():
    # Kanten 0-1, 0-2, 1-2, 1-3, Quelle 0: nach {0,1} hat Betrieb 2 zwei Kanten in die Menge (0-2, 1-2), Betrieb 3 eine (1-3): 2/3 gegen 1/3
    adj = make_adj(4, [(0, 1), (0, 2), (1, 2), (1, 3)])
    probs = A.exact_set_probabilities(adj, 0, 3)
    assert probs == {0b0111: pytest.approx(5 / 6), 0b1011: pytest.approx(1 / 6)}


def test_growth_rejects_sources_at_the_border_of_the_range():
    adj = make_adj(4, [(0, 1), (1, 2), (2, 3)])
    for bad in (4, -1):
        with pytest.raises(ValueError):
            A.grow(adj, bad, 2, _rng())
    assert A.grow(adj, 3, 2, _rng()) == [3, 2] and A.grow(adj, 0, 2, _rng()) == [0, 1]


def test_regular_tree_validation_and_sorted_adjacency():
    with pytest.raises(ValueError):
        A.grow_regular_tree(1, 1, _rng())                                                        # d = 1 ist keine Struktur (auch ohne Wachstumsschritt)
    with pytest.raises(ValueError):
        A.grow_regular_tree(3, 0, _rng())
    sub, _ = A.grow_regular_tree(4, 60, _rng(3))
    assert all(list(a) == sorted(a) for a in sub)


def test_observe_of_a_single_infected_node_keeps_it():
    assert A.observe([5], 0.5, _rng()) == [5] and A.observe([5, 7], 0.5, _rng()) == [5, 7] and A.observe([5, 7, 9], 0.1, _rng()) != []
    assert len(A.observe([5, 7, 9], 0.1, _rng())) == 2
