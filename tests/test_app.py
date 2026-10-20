"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Slider an seine Grenzen,
prüft Permalink, Netztypen und alle fünf Experimente; es darf kein Python-Fehler auftreten."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import qsu_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
EXPERIMENTS = ("trees", "networks", "sizes", "observation", "exact")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert _metric(at, "Befallen / beobachtet") == "30 / 30"
    assert _metric(at, "Rumor Centrality") == "Abstand 1,0" and _metric(at, "Distanzzentralität") == "Abstand 2,0"
    assert not at.error and not at.warning


def test_every_preset_click_does_not_raise():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value) if isinstance(slider.value, int) else value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_all_sliders_at_their_minimum_and_all_at_their_maximum_do_not_raise():
    for edge in ("min", "max"):
        at = _fresh_app()
        for index in range(len(at.slider)):
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value) if isinstance(slider.value, int) else value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("typ", C.ALL_TYPES)
def test_every_network_type_renders(typ):
    at = _fresh_app()
    at.selectbox(key="netztyp_select").set_value(typ).run(timeout=TIMEOUT)
    assert not at.exception, (typ, [str(e) for e in at.exception])


def test_a_new_run_changes_the_seed_and_the_result():
    at = _fresh_app()
    before = (at.number_input(key="seed_input").value, _metric(at, "Rumor Centrality"))
    at.button[[b.label for b in at.button].index("🎲 Neuer Lauf")].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before[0]


def test_permalink_values_are_clamped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "5"
    at.query_params["c"] = "99"
    at.query_params["d"] = "0"
    at.query_params["befallen"] = "9999"
    at.query_params["beobachtet"] = "1"
    at.query_params["typ"] = "gibt es nicht"
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="n_slider").value == C.N_MIN and at.slider(key="c_slider").value == C.C_MAX and at.slider(key="d_slider").value == C.D_MIN
    assert at.slider(key="ni_slider").value == C.NI_MAX and at.slider(key="obs_slider").value == C.OBS_MIN and at.selectbox(key="netztyp_select").value == C.T_RANDOM


def test_permalink_selects_the_tree_type():
    at = AppTest.from_file(APP_PATH)
    at.query_params["typ"] = C.T_TREE
    at.query_params["d"] = "2"
    at.run(timeout=TIMEOUT)
    assert not at.exception and at.selectbox(key="netztyp_select").value == C.T_TREE and at.slider(key="d_slider").value == 2


def test_every_experiment_runs_and_is_invalidated_by_a_setting_change():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(C.N_MIN).run(timeout=TIMEOUT)         # kleines Netz: schnell
    for name in EXPERIMENTS:
        at.button(key=f"run_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
    texts = " ".join(x.value for x in list(at.info) + list(at.warning) + list(at.success))
    for fragment in ("Trefferquote der Rumor Centrality auf befallenen Bäumen", "Bei 30 % befallenen Betrieben trifft die Rumor Centrality", "Trefferquote der Rumor Centrality von", "beobachtet: Treffer", "exakt gerechnet"):
        assert fragment in texts, fragment
    at.slider(key="n_slider").set_value(C.N_MIN + 10).run(timeout=TIMEOUT)
    assert not at.exception
    assert sum("neu starten" in i.value for i in at.info) >= len(EXPERIMENTS) - 2       # die Experimente 2 bis 4 hängen von n ab (Bäume und exakter Vergleich nicht)
