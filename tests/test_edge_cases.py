"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste und größte Regler, alle Netztypen, Warnungen als Fehler, kleinste Netze in allen Experimenten, Abbildungen."""

import warnings

import numpy as np
import pytest

import qsu_algorithm as A
import qsu_constants as C
import qsu_evaluation as ev
import qsu_visualization as V
from conftest import make_adj


@pytest.mark.parametrize("typ", C.ALL_TYPES)
def test_every_network_type_at_the_slider_corners_runs_without_warnings(typ):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for n, c, d, ni, obs in ((C.N_MIN, C.C_MIN, C.D_MIN, C.NI_MIN, C.OBS_MIN), (C.N_MIN, C.C_MAX, C.D_MAX, C.NI_MAX, C.OBS_MAX), (C.N_MAX, C.C_MAX, C.D_MAX, C.NI_MAX, C.OBS_MIN), (C.N_MIN, 4, 3, C.NI_MAX, 75)):
            an = ev.analyse(ev.Settings(typ, n, c, 3, d, ni, obs))
            assert 1 <= an.ni <= max(ni, 1) and an.order[0] == an.source and set(an.nodes) <= set(an.seen) <= set(an.order)
            assert all(0.0 <= an.hits[e] <= 1.0 for e in C.ESTIMATORS) and all(an.dists[e] >= 0 for e in C.ESTIMATORS) and ev.verdict_main(an)[0] in ("success", "info", "warning")


def test_growth_in_a_network_smaller_than_the_request_stops_at_the_component():
    an = ev.analyse(ev.Settings(C.T_RANDOM, C.N_MIN, C.C_MIN, 3, 3, C.NI_MAX, 100))                # n = 60 < 150 Befallene verlangt
    assert an.ni <= C.N_MIN and an.ni >= 2


def test_observation_with_two_nodes_and_a_disconnected_pick_is_still_evaluated():
    adj = make_adj(6, [(0, 1), (1, 2), (3, 4), (4, 5), (2, 3)])
    seen = A.observe([0, 1, 2, 3, 4, 5], 0.34, np.random.default_rng(1))
    assert len(seen) == 2
    comp = A.components(A.induced(adj, seen))[0]
    sub = A.induced(adj, [seen[i] for i in comp])
    assert A.estimate_sets(sub)["Rumor Centrality"]                                               # auch eine Gruppe aus einem Betrieb liefert einen Kandidaten


def test_a_group_of_one_node_has_rank_and_credit():
    sets = A.estimate_sets(((),))
    assert sets["Rumor Centrality"] == [0] and A.credit(5, [5], sets["Rumor Centrality"]) == 1.0 and A.source_rank([0.0], 0) == 1


def test_the_line_of_two_and_the_pair_are_symmetric():
    sub = make_adj(2, [(0, 1)])
    sets = A.estimate_sets(sub)
    assert sets["Rumor Centrality"] == [0, 1] and sets["Distanzzentralität"] == [0, 1] and A.credit(0, [0, 1], sets["Rumor Centrality"]) == 0.5


def test_experiments_run_on_the_smallest_networks_without_warnings():
    seeds = C.LIVE_SEEDS[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = ev.tree_rows(ds=(2, 6), ns=(5, 20), runs=5)
        w = ev.network_rows(C.N_MIN, C.C_MIN, fracs=(0.1, 0.3), seeds=seeds, runs=3, types=(C.T_BA, C.T_GRID))
        s = ev.size_rows(C.T_RING, C.N_MIN, 4, sizes=(5, 20, 59, 100), seeds=seeds, runs=3)
        o = ev.observation_rows(C.T_RANDOM, C.N_MIN, 4, shares=(1.0, 0.5), seeds=seeds, runs=3)
        e = ev.exact_rows(nets=2, nis=(3, 4))
    assert len(t) == 4 and len(w) == 4 and [r["ni"] for r in s] == [5, 20, 59] and len(o) == 2 and len(e) == 2
    for verdict in (ev.verdict_trees(t), ev.verdict_networks(w, 0.3), ev.verdict_sizes(s), ev.verdict_observation(o), ev.verdict_exact(e)):
        assert verdict[0] in ("success", "info", "warning") and verdict[1]


def test_size_rows_skip_the_tree_type_and_sizes_at_or_above_the_network_size():
    assert ev.size_rows(C.T_TREE, 100, 6, runs=2) == []
    assert [r["ni"] for r in ev.size_rows(C.T_RANDOM, 60, 4, sizes=(10, 60, 80), seeds=C.LIVE_SEEDS[:1], runs=2)] == [10]


def test_figures_are_built_for_the_analysis_and_the_experiment_rows():
    seeds = C.LIVE_SEEDS[:2]
    t = ev.tree_rows(ds=(2, 3), ns=(10, 20), runs=4)
    w = ev.network_rows(C.N_MIN, 4, fracs=C.NET_FRACS, seeds=seeds, runs=2, types=(C.T_RANDOM, C.T_RING))
    s = ev.size_rows(C.T_RANDOM, C.N_MIN, 4, sizes=(5, 20), seeds=seeds, runs=2)
    o = ev.observation_rows(C.T_RANDOM, C.N_MIN, 4, shares=(1.0, 0.5), seeds=seeds, runs=2)
    e = ev.exact_rows(nets=2, nis=(3, 4))
    for typ in (C.T_RING, C.T_TREE):
        an = ev.analyse(ev.Settings(typ, 80, 6, 3, 3, 25, 80))
        for fig in (V.build_network(an), V.build_run_distances(an)):
            assert fig.data and all(ax.fixedrange for ax in (fig.layout.xaxis, fig.layout.yaxis))
    for fig in (V.build_tree_hits(t), V.build_network_hits(w, 0.1), V.build_network_distances(w, 0.3), V.build_sizes(s), V.build_size_distances(s), V.build_observation(o), V.build_exact(e)):
        assert fig.data and all(ax.fixedrange for ax in (fig.layout.xaxis, fig.layout.yaxis))


def test_the_network_figure_marks_the_source_and_the_estimates():
    an = ev.analyse(ev.Settings(C.T_RING, 100, 6, 35, 3, 40, 100))
    fig = V.build_network(an)
    names = [tr.name for tr in fig.data if tr.name]
    assert "wahre Quelle" in names and "befallen, ausgewertet" in names and any(n.startswith("Schätzung: ") for n in names)
    infected = next(tr for tr in fig.data if tr.name == "befallen, ausgewertet")
    assert len(infected.x) == len(an.nodes)
