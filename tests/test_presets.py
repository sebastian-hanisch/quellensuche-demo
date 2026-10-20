"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset zeigt, wofür es benannt ist."""

import qsu_algorithm as A
import qsu_constants as C
import qsu_evaluation as ev
import qsu_presets as P


def _settings(name):
    return ev.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert p["netztyp"] in C.ALL_TYPES and C.N_MIN <= p["n"] <= C.N_MAX and C.C_MIN <= p["c"] <= C.C_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert C.D_MIN <= p["d"] <= C.D_MAX and C.NI_MIN <= p["ni"] <= C.NI_MAX and C.OBS_MIN <= p["obs"] <= C.OBS_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[C.DEFAULT_PRESET_NAME]) == ev.Settings()
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET_NAME


def test_setting_specs_cover_every_preset_key_and_defaults_match():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == C.PRESETS[C.DEFAULT_PRESET_NAME][key], key


def test_every_preset_differs_from_the_default_in_something_besides_the_network_type():
    default = C.PRESETS[C.DEFAULT_PRESET_NAME]
    for name, p in C.PRESETS.items():
        if name != C.DEFAULT_PRESET_NAME:
            assert any(p[k] != default[k] for k in p if k != "netztyp"), name


def test_each_preset_shows_what_its_name_says():
    an = {name: ev.analyse(_settings(name)) for name in C.PRESETS}
    line = an["Linie: der Satz"]
    assert line.settings.d == 2 and max(len(a) for a in line.inst.adj) == 2 and line.ni == 100
    tree3 = an["Baum mit drei Partnern: es geht"]
    assert max(len(a) for a in tree3.inst.adj) == 3 and tree3.settings.d == 3
    ba = an["Skalenfrei: Treffer wie Zufall, Abstand besser"]
    assert ba.inst.typ == C.T_BA and all(ba.hits[e] == 0.0 for e in (C.E_RC, C.E_DC, C.E_J)) and ba.dists[C.E_RC] < ba.dists[C.E_RAND]
    ring = an["Ring: Distanz schlägt Rumor Centrality"]
    assert ring.hits[C.E_DC] == 1.0 and ring.hits[C.E_RC] == 0.0
    full = an["Fast alles befallen: mehr hilft nicht"]
    assert full.ni >= 0.75 * full.inst.n and full.hits[C.E_RC] < 0.1


def test_every_preset_runs_and_the_verdict_is_a_known_kind():
    for name in C.PRESETS:
        an = ev.analyse(_settings(name))
        assert an.inst.n >= 2 and an.ni >= 2 and an.order[0] == an.source and len(set(an.order)) == an.ni
        assert ev.verdict_main(an)[0] in ("success", "info", "warning"), name
        assert A.credit(an.source, list(an.nodes), an.sets[C.E_RAND]) == (1.0 / len(an.nodes) if an.source in an.nodes else 0.0)
