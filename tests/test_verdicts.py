"""Verdikte und Messhilfen an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import dataclasses
import math

import pytest

import qsu_algorithm as A
import qsu_constants as C
import qsu_evaluation as ev

NAN = float("nan")


def _an(hits, dists, ranks=None, ni=30, seen=None, nodes=None, source_in=True, sets=None):
    """Analyse mit handgesetzten Treffern und Abständen auf einem echten kleinen Lauf."""
    real = ev.analyse(ev.Settings(C.T_RANDOM, C.N_MIN, 4, 3, 3, 12, 100))
    nodes = tuple(range(40)) if nodes is None else nodes
    order = tuple(range(ni)) if ni != real.ni else real.order
    full = {e: 0.0 for e in C.ESTIMATORS}
    full.update(hits)
    dd = {e: 1.0 for e in C.ESTIMATORS}
    dd.update(dists)
    all_sets = {e: [0] for e in C.ESTIMATORS}
    all_sets[C.E_RAND] = list(range(len(nodes)))
    all_sets.update(sets or {})
    return dataclasses.replace(real, source=nodes[0] if source_in else 10**6, order=order, seen=tuple(range(ni if seen is None else seen)), nodes=nodes, sets=all_sets, hits=full, dists=dd,
                               ranks={} if ranks is None else ranks)


def test_de_pct_helpers():
    assert ev.de(1.26, 1) == "1,3" and ev.de(0.9487, 3) == "0,949" and ev.de(NAN, 2) == "–" and ev.de(None) == "–" and ev.pct(0.1234, 1) == "12,3" and ev.pct(0.5) == "50"


def test_main_verdict_all_miss_is_info_and_names_distance_rank_and_random():
    an = _an({}, {C.E_RC: 2.0, C.E_DC: 3.0, C.E_J: 4.5, C.E_RAND: 2.4}, ranks={C.E_RC: 5, C.E_DC: 7, C.E_J: 6})
    kind, text = ev.verdict_main(an)
    assert kind == "info"
    assert text == ("30 Betriebe sind befallen. Rumor Centrality verfehlt die Quelle, Abstand 2,0, Rang 5; Distanzzentralität verfehlt die Quelle, Abstand 3,0, Rang 7; "
                    "Jordan-Zentrum verfehlt die Quelle, Abstand 4,5, Rang 6. Zufall käme im Mittel auf 0,025 Treffer und Abstand 2,4.")


def test_main_verdict_hit_is_success_and_ties_are_named():
    an = _an({C.E_DC: 1.0, C.E_J: 0.25}, {C.E_DC: 0.0, C.E_J: 1.5}, ranks={C.E_RC: 2, C.E_DC: 1, C.E_J: 1}, sets={C.E_J: [0, 1, 2, 3]})
    kind, text = ev.verdict_main(an)
    assert kind == "success" and "Distanzzentralität trifft die Quelle, Abstand 0,0, Rang 1" in text
    assert "Jordan-Zentrum hat die Quelle unter 4 gleich guten Kandidaten (Treffer 0,25), Abstand 1,5, Rang 1" in text and "Rumor Centrality verfehlt die Quelle" in text


def test_main_verdict_with_incomplete_observation_and_a_lost_source():
    an = _an({}, {}, ni=30, seen=24, nodes=tuple(range(20)), source_in=False)
    kind, text = ev.verdict_main(an)
    assert kind == "info" and text.startswith("30 Betriebe sind befallen, 24 davon beobachtet, ausgewertet wird die größte zusammenhängende Gruppe mit 20.")
    assert text.endswith("Die Quelle gehört nicht zur ausgewerteten Gruppe.") and "Rang" not in text and "Zufall käme im Mittel auf 0,050 Treffer" in text


def test_main_verdict_single_infected():
    an = dataclasses.replace(_an({}, {}), order=(5,))
    assert ev.verdict_main(an) == ("info", "Es ist nur ein Betrieb befallen: er ist die Quelle.")


# --- Messhilfen und Läufe ---------------------------------------------------------------------------------------------------------------------------


def test_trial_returns_credit_and_distance_for_every_estimator_and_is_reproducible():
    adj = ev.make_instance(C.T_RANDOM, 80, 6, 2, layout=False).adj
    t1 = ev.trial(adj, 5, 20, 1.0, ev._rng(1, 2, 3))
    assert t1 == ev.trial(adj, 5, 20, 1.0, ev._rng(1, 2, 3)) and set(t1["cells"]) == set(C.ESTIMATORS) and t1["connected"] == 1.0 and t1["infected"] == 20
    assert all(0.0 <= h <= 1.0 and d >= 0.0 for h, d in t1["cells"].values())
    partial = ev.trial(adj, 5, 20, 0.5, ev._rng(1, 2, 4))
    assert 0.0 < partial["connected"] <= 1.0


def test_tree_trial_on_a_line_of_three_is_decided_by_hand():
    # Linie mit 3 Befallenen ab Quelle 0: der erste Schritt geht nach links oder rechts (je 1/2); im zweiten Schritt hat die Menge zwei Randkanten (je 1/2): nach außen derselben Seite
    # (Quelle am Rand, Menge ist die Linie) oder auf die andere Seite (Quelle in der Mitte). Die Mitte ist der einzige Kandidat: Treffer mit Wahrscheinlichkeit 1/2.
    hits = [ev.tree_trial(2, 3, 1.0, ev._rng(9, r))["cells"][C.E_RC][0] for r in range(4000)]
    assert sum(hits) / len(hits) == pytest.approx(0.5, abs=0.03)


def test_mean_cells_averages_hits_and_distances():
    trials = [{"cells": {e: (1.0, 2.0) for e in C.ESTIMATORS}}, {"cells": {e: (0.0, 4.0) for e in C.ESTIMATORS}}]
    assert ev._mean_cells(trials)[C.E_RC] == (0.5, 3.0)


# --- Verdikte der Experimente -----------------------------------------------------------------------------------------------------------------------


def _tree_row(d, ni, rc, dc=None, j=0.0, rand=0.0):
    return {"d": d, "ni": ni, "cells": {C.E_RC: (rc, 1.0), C.E_DC: (rc if dc is None else dc, 1.0), C.E_J: (j, 1.0), C.E_RAND: (rand, 1.0)}, "runs": 100}


def test_verdict_trees_names_line_and_branching_and_detects_equal_estimators():
    rows = [_tree_row(2, 10, 0.25), _tree_row(2, 200, 0.05), _tree_row(3, 10, 0.4), _tree_row(3, 200, 0.25)]
    kind, text = ev.verdict_trees(rows)
    assert kind == "info"
    assert text == ("Trefferquote der Rumor Centrality auf befallenen Bäumen: d = 2 (Linie): 0,250 bei 10, 0,050 bei 200 Befallenen; d = 3: 0,400 bei 10, 0,250 bei 200 Befallenen. "
                    "Rumor Centrality und Distanzzentralität treffen in jeder Zelle gleich oft: auf Bäumen sind es dieselben Kandidaten.")
    rows[1] = _tree_row(2, 200, 0.05, dc=0.07)
    assert "unterscheiden sich in mindestens einer Zelle" in ev.verdict_trees(rows)[1]
    assert ev.verdict_trees([])[0] == "warning" and ev.verdict_trees(rows[:1])[0] == "warning"
    assert ev.tree_cell(rows, 3, 10)[C.E_RC][0] == 0.4


def _net_row(typ, frac, rc, dc, j, rand, d_rc=1.5, d_rand=3.0):
    return {"typ": typ, "frac": frac, "cells": {C.E_RC: (rc, d_rc), C.E_DC: (dc, 1.4), C.E_J: (j, 1.6), C.E_RAND: (rand, d_rand)}, "runs": 100}


def test_verdict_networks_finds_best_near_random_and_distance_wins():
    rows = [_net_row("A", 0.3, 0.05, 0.05, 0.04, 0.01), _net_row("B", 0.3, 0.015, 0.012, 0.01, 0.011), _net_row("C", 0.3, 0.02, 0.08, 0.06, 0.01), _net_row("D", 0.3, 0.025, 0.027, 0.01, 0.01)]
    kind, text = ev.verdict_networks(rows, 0.3)
    assert kind == "info"
    assert text.startswith("Bei 30 % befallenen Betrieben trifft die Rumor Centrality am häufigsten in „A“ (5,0 % gegen 1,0 % bei Zufall, mittlerer Abstand 1,50 gegen 3,00). ")
    assert "Kaum über Zufall (unter dem 1,5-Fachen) liegt kein Schätzer in: „B“. " in text                      # B: bester 0,015 < 1,5 · 0,011; D: 0,027 > 1,5 · 0,01
    assert text.endswith("Die Distanzzentralität trifft mindestens 20 % öfter als die Rumor Centrality in: „C“.")
    clean = ev.verdict_networks([_net_row("A", 0.1, 0.05, 0.05, 0.05, 0.01)], 0.1)[1]
    assert "In jedem Netztyp liegt der beste Schätzer über dem 1,5-Fachen von Zufall." in clean and "in keinem Netztyp deutlich schlechter" in clean
    assert ev.verdict_networks(rows, 0.1)[0] == "warning"


def _size_row(ni, rc, rand, dist, typ="Zufallsnetz"):
    return {"typ": typ, "ni": ni, "cells": {C.E_RC: (rc, dist), C.E_DC: (rc, dist), C.E_J: (rc, dist), C.E_RAND: (rand, 3.0)}, "runs": 100}


def test_verdict_sizes_reports_the_trend():
    kind, text = ev.verdict_sizes([_size_row(10, 0.2, 0.1, 1.2), _size_row(40, 0.08, 0.03, 2.0), _size_row(150, 0.01, 0.007, 2.8)])
    assert kind == "info" and text == ("In „Zufallsnetz“ fällt die Trefferquote der Rumor Centrality von 20,0 % bei 10 Befallenen auf 1,0 % bei 150; der mittlere Abstand zur Quelle geht von 1,20 auf 2,80. Zufall trifft 10,0 % bzw. 0,7 %.")
    assert " steigt " in ev.verdict_sizes([_size_row(10, 0.1, 0.1, 1.0), _size_row(20, 0.2, 0.05, 1.0)])[1]
    assert ev.verdict_sizes([_size_row(10, 0.1, 0.1, 1.0)])[0] == "warning"


def test_verdict_observation_by_hand():
    rows = [{"typ": "X", "share": 1.0, "cells": {C.E_RC: (0.11, 2.16)}, "connected": 1.0, "runs": 100}, {"typ": "X", "share": 0.5, "cells": {C.E_RC: (0.02, 2.46)}, "connected": 0.66, "runs": 100}]
    kind, text = ev.verdict_observation(rows)
    assert kind == "info" and text == ("Rumor Centrality in „X“: bei 100 % beobachtet: Treffer 11,0 %, Abstand 2,16, im Mittel 100 % der Beobachteten in der ausgewerteten Gruppe; "
                                       "bei 50 % beobachtet: Treffer 2,0 %, Abstand 2,46, im Mittel 66 % der Beobachteten in der ausgewerteten Gruppe. Das Verfahren erwartet eine zusammenhängende befallene Menge; fehlende Betriebe lassen sie zerfallen.")
    assert ev.verdict_observation(rows[:1])[0] == "warning"


def test_verdict_exact_by_hand_and_bayes_is_the_upper_bound_in_the_rows():
    rows = [{"ni": 4, "nets": 8, "hits": {"Bayes": 0.435, "Rumor Centrality": 0.293, "Distanzzentralität": 0.293, "Jordan-Zentrum": 0.25, "Zufall": 0.25}, "bayes_wins": 8}]
    kind, text = ev.verdict_exact(rows)
    assert kind == "info" and text == ("Auf 8 kleinen Netzen (9 Betriebe, 12 Partnerschaften), exakt gerechnet: 4 Befallene: Bayes-Optimum 43,5 %, Rumor Centrality 29,3 %, Distanz 29,3 %, Zufall 25,0 %. "
                                       "Selbst das Bayes-Optimum verfehlt die Quelle meist.")
    assert ev.verdict_exact([])[0] == "warning"
    real = ev.exact_rows(nets=2, nis=(3,))
    assert real[0]["bayes_wins"] == 2 and real[0]["hits"]["Bayes"] >= max(real[0]["hits"][e] for e in C.ESTIMATORS) - 1e-12


def test_exact_rows_hit_probabilities_for_zufall_are_one_over_ni():
    for row in ev.exact_rows(nets=2, nis=(3, 4)):
        assert row["hits"]["Zufall"] == pytest.approx(1.0 / row["ni"])


def test_rank_and_hits_in_the_real_analysis_agree():
    an = ev.analyse(ev.Settings(C.T_RING, 80, 6, 5, 3, 20, 100))
    for e in (C.E_RC, C.E_DC, C.E_J):
        assert (an.ranks[e] == 1) == (an.hits[e] > 0) and (an.dists[e] == 0.0) == (an.hits[e] == 1.0)
    assert math.isclose(an.hits[C.E_RAND], 1.0 / len(an.nodes)) or an.hits[C.E_RAND] == 0.0
    assert set(an.estimate_nodes(C.E_RC)) <= set(an.nodes) and A.credit(an.source, list(an.nodes), an.sets[C.E_RC]) == an.hits[C.E_RC]
