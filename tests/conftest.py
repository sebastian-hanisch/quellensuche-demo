"""Gemeinsame Fixtures und Hilfen (DEMO-PLAYBOOK §1b): Mini-Instanzen, an denen jede Einheit von Hand geprüft wird."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_adj(n, pairs):
    """Nachbarlisten (Tupel von sortierten Tupeln) aus einer Kantenliste."""
    sets = [set() for _ in range(n)]
    for a, b in pairs:
        if a != b:
            sets[a].add(b)
            sets[b].add(a)
    return tuple(tuple(sorted(s)) for s in sets)


def random_tree(rng, n):
    """Zufälliger Baum mit n Knoten (jeder neue Knoten hängt an einem zufälligen früheren)."""
    return make_adj(n, [(i, int(rng.integers(i))) for i in range(1, n)])


@pytest.fixture
def path3():
    """Pfad 0 - 1 - 2. R(0) = 3! / (3 · 2 · 1) = 1, R(1) = 3! / (3 · 1 · 1) = 2, R(2) = 1."""
    return make_adj(3, [(0, 1), (1, 2)])


@pytest.fixture
def star5():
    """Stern: Mitte 0, Blätter 1 bis 4. R(Mitte) = 5! / (5 · 1 · 1 · 1 · 1) = 24, R(Blatt) = 5! / (5 · 4 · 1 · 1 · 1) = 6."""
    return make_adj(5, [(0, 1), (0, 2), (0, 3), (0, 4)])


@pytest.fixture
def line5():
    """Linie 0 - 1 - 2 - 3 - 4. R(Mitte) = 5! / (5 · 2 · 1 · 2 · 1) = 6, R(Rand) = 5! / (5 · 4 · 3 · 2 · 1) = 1, R(Knoten 1) = 5! / (5 · 1 · 3 · 2 · 1) = 4."""
    return make_adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)])


@pytest.fixture
def triangle():
    """Dreieck: alle drei Quellen sind gleich wahrscheinlich."""
    return make_adj(3, [(0, 1), (1, 2), (0, 2)])


@pytest.fixture
def small_edges():
    """Kleines verbundenes Netz mit 9 Knoten und 12 Kanten (Ring mit drei Sehnen) für die exakte Rechnung."""
    return 9, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (0, 8), (1, 5), (2, 6), (3, 7)]
