"""Netze: gültige Netze, Determinismus (auch über Prozesse), Positionen; kleine Netze und befallene reguläre Bäume."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import qsu_algorithm as A
import qsu_constants as C
import qsu_evaluation as ev
import qsu_scenario as S

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("typ", C.GENERATED_TYPES)
@pytest.mark.parametrize("n,c", [(C.N_MIN, C.C_MIN), (C.N_MIN, C.C_MAX), (C.DEFAULT_N, C.DEFAULT_C), (C.N_MAX, C.C_MAX)])
def test_every_corner_gives_a_valid_network(typ, n, c):
    inst = S.make_instance(typ, n, c, 35)
    assert inst.n >= 2 and len(inst.pos) == inst.n and inst.m >= 1
    for x, y in inst.pos:
        assert math.isfinite(x) and math.isfinite(y) and -1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9
    for a, nb in enumerate(inst.adj):
        assert list(nb) == sorted(set(nb)) and a not in nb
        for b in nb:
            assert a in inst.adj[b]


def test_instances_are_deterministic_per_seed_and_differ_between_seeds():
    a = S.make_instance(C.T_COMMUNITIES, 80, 6, 5)
    assert a.adj == S.make_instance(C.T_COMMUNITIES, 80, 6, 5).adj and a.pos == S.make_instance(C.T_COMMUNITIES, 80, 6, 5).pos
    assert S.make_instance(C.T_COMMUNITIES, 80, 6, 6, layout=False).adj != a.adj


def test_network_and_run_are_independent_of_the_hash_seed_of_the_process():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import hashlib, qsu_scenario as S, qsu_constants as C, qsu_evaluation as E;"
            "h = hashlib.sha256();"
            "[h.update(repr(S.make_instance(t, 60, 5, 7, layout=False).adj).encode()) for t in C.GENERATED_TYPES];"
            "[h.update(repr(E.analyse(E.Settings(t, 60, 5, 7, 3, 20, 80)).order).encode()) for t in C.ALL_TYPES];"
            "print(h.hexdigest())")
    outs = []
    for hashseed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        outs.append(subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code, str(ROOT)], capture_output=True, text=True, env=env, check=True).stdout.strip())
    assert outs[0] == outs[1] and len(outs[0]) == 64


def test_layout_false_skips_the_force_layout_and_unknown_type_raises():
    inst = S.make_instance(C.T_RANDOM, 60, 4, 1, layout=False)
    assert set(inst.pos) == {(0.5, 0.5)}
    with pytest.raises(ValueError):
        S.make_instance("gibt es nicht", 60, 4, 1)
    with pytest.raises(ValueError):
        S.make_instance(C.T_TREE, 60, 4, 1)                                                  # der reguläre Baum entsteht mit dem Befall, nicht als Netz


def test_the_regular_tree_analysis_has_the_right_shape():
    for d in (2, 3, 5):
        an = ev.analyse(ev.Settings(C.T_TREE, 100, 6, 4, d, 40, 100))
        assert an.inst.n == 40 and an.source == 0 and an.order == tuple(range(40)) and A.is_tree(an.inst.adj)
        assert max(len(a) for a in an.inst.adj) <= d and len(an.inst.pos) == 40
    line = ev.analyse(ev.Settings(C.T_TREE, 100, 6, 4, 2, 40, 100))
    assert max(len(a) for a in line.inst.adj) == 2 and sum(1 for a in line.inst.adj if len(a) == 1) == 2          # Linie: zwei Enden


def test_small_network_is_connected_has_the_requested_size_and_is_deterministic():
    for seed in range(100000, 100010):
        n, edges = ev.small_network(seed)
        assert n == C.EXACT_N and len(edges) == C.EXACT_M == len(set(edges)) and all(a < b for a, b in edges)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for a, b in edges:
            parent[find(a)] = find(b)
        assert len({find(i) for i in range(n)}) == 1
        assert ev.small_network(seed) == (n, edges)
    assert ev.small_network(100000) != ev.small_network(100001)
