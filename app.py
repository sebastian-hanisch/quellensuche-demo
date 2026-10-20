"""Quellensuche: wo hat der Befall angefangen? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Reihe „Netzwerkanalyse“: das inverse Problem zu den Ausbreitungs-Stücken. Aus der Menge der befallenen Betriebe wird die Quelle geschätzt (Rumor Centrality, Distanzzentralität,
Jordan-Zentrum, Zufall); die Demo zeigt, wo das gelingt und wo nicht (Linie, schwankender Grad, mehr Befallene, fehlende Beobachtung) und wie gut es überhaupt sein könnte (exaktes Optimum auf kleinen Netzen).
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import qsu_constants as C
from qsu_evaluation import (
    Settings,
    analyse,
    de,
    exact_rows,
    network_rows,
    observation_rows,
    pct,
    size_rows,
    tree_rows,
    verdict_exact,
    verdict_main,
    verdict_networks,
    verdict_observation,
    verdict_sizes,
    verdict_trees,
)
from qsu_layout import dunkles_design, kopf_css
from qsu_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from qsu_visualization import (
    build_exact,
    build_network,
    build_network_distances,
    build_network_hits,
    build_observation,
    build_run_distances,
    build_size_distances,
    build_sizes,
    build_tree_hits,
)

st.set_page_config(page_title="Quellensuche – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _trees():
    return tree_rows()


@st.cache_data(show_spinner=False)
def _networks(n, c):
    return network_rows(n, c)


@st.cache_data(show_spinner=False)
def _sizes(typ, n, c):
    return size_rows(typ, n, c)


@st.cache_data(show_spinner=False)
def _observation(typ, n, c):
    return observation_rows(typ, n, c)


@st.cache_data(show_spinner=False)
def _exact():
    return exact_rows()


def experiment(name, label, key, compute, show, hint):
    """Teure Zusatzansicht erst auf Klick; das Ergebnis bleibt nur für genau diese Einstellungen stehen (Session State)."""
    state_key = f"exp_{name}"
    if st.button(label, key=f"run_{name}", help=hint):
        with st.spinner("Rechne …"):
            result = compute()
        st.session_state[state_key] = (key, result)
    state = st.session_state.get(state_key)
    if state is None:
        st.caption(f"Noch nicht gerechnet. {hint}")
        return
    if state[0] != key:
        st.info("Die Einstellungen haben sich seit der letzten Rechnung geändert. Bitte neu starten.")
        return
    show(state[1])


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in qsu_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔎 Quellensuche: wo hat der Befall angefangen?")
st.markdown(
    """
**Achtes Stück** der Reihe „Netzwerkanalyse“, im Ausbreitungs-Arm das **inverse Problem** zu den Stücken 5 bis 7: dort war die Quelle gegeben und der Verlauf gesucht, hier ist der Endzustand gegeben und die **Quelle** gesucht. In einem Netz von Subunternehmern und Lieferanten ist ein **Befall** aufgetreten (Schadsoftware, ein Qualitätsproblem). Sie sehen, welche Betriebe befallen sind, aber nicht, wo es angefangen hat. Die Demo stellt die **Rumor Centrality** (Shah und Zaman 2011) der **Distanzzentralität**, dem **Jordan-Zentrum** und dem **Zufallstipp** gegenüber; die [Kaskaden-Demo](https://sebastianhanisch-kaskaden-demo.streamlit.app/) rechnet die Ausbreitung in Vorwärtsrichtung.

Die Kernfrage: **Wo hat es angefangen, und wo geht es nicht mehr?** Der Hook ist ein **Grenzenbefund**, kein Verfahrenserfolg: auf der Linie versagt der Schätzer, in Netzen mit stark schwankendem Grad trifft er die Quelle kaum über Zufall, und **mehr Befallene helfen nicht**. Die Ergebnisse sind **Modellbefunde**, keine Aussagen über die Wirklichkeit.

Weiter unten: fünf Experimente (**🔬**), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Quellensuche", expanded=True):
    st.markdown(
        """
1. **Befall (SI):** er beginnt in einem Betrieb, der **Quelle**. In jedem Schritt springt er über eine **zufällig gewählte Randkante** (eine Partnerschaft zwischen einem befallenen und einem gesunden Betrieb) auf einen weiteren Betrieb; wer befallen ist, bleibt es. Das geht, bis die eingestellte Zahl Betriebe befallen ist.
2. **Beobachtung:** Sie sehen die **Menge** der Befallenen, nicht die Reihenfolge. Bei unvollständiger Beobachtung fehlen zufällige Betriebe; ausgewertet wird die größte zusammenhängende Gruppe der Beobachteten.
3. **Schätzer** nennen je einen oder mehrere gleich gute Kandidaten für die Quelle. **Treffer:** 1, wenn der Kandidat die Quelle ist; bei *k* gleich guten Kandidaten zählt 1/*k*. **Abstand:** der mittlere Abstand der Kandidaten zur Quelle im ganzen Netz (ein Treffer hat Abstand 0).
4. **Rang** der wahren Quelle: 1 + Zahl der Betriebe mit strikt besserem Wert beim jeweiligen Schätzer.
        """
    )
    st.markdown(
        """
| Schätzer | Was er nennt | Aufwand |
|---|---|---|
| **Rumor Centrality** | den Betrieb, von dem aus die meisten Reihenfolgen die beobachtete Menge erzeugen (auf Bäumen der Maximum-Likelihood-Schätzer; auf anderen Netzen mit dem BFS-Baum je Kandidat genähert) | ein BFS je Kandidat |
| **Distanzzentralität** | den Betrieb mit der kleinsten Summe der Abstände zu allen Befallenen (auf Bäumen dasselbe wie die Rumor Centrality) | ein BFS je Kandidat |
| **Jordan-Zentrum** | den Betrieb mit dem kleinsten größten Abstand zu einem Befallenen | ein BFS je Kandidat |
| **Zufall** | einen beliebigen Befallenen | keiner |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um einen Lauf zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz**")
    typ = st.selectbox("Netztyp", C.ALL_TYPES, key="netztyp_select", help="Welches Netz erzeugt wird. Der reguläre Baum wächst mit dem Befall; mit Verzweigung 2 ist er eine Linie.")
    n = st.slider("Betriebe n", *bounds("n_slider"), key="n_slider", help="Zahl der Betriebe (beim Gitter auf ein Quadrat gerundet; beim regulären Baum ohne Wirkung).")
    c = st.slider("Partner je Betrieb c", *bounds("c_slider"), key="c_slider", help="Was c bedeutet, hängt vom Netztyp ab (siehe darunter).")
    st.caption(C.C_MEANING[typ])
    d = st.slider("Verzweigung d (regulärer Baum)", *bounds("d_slider"), key="d_slider", help="Nur beim regulären Baum: jeder Betrieb hat d Partner; d = 2 ist eine Linie.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt das Netz, die Quelle und den Verlauf des Befalls.")
    st.button("🎲 Neuer Lauf", width="stretch", on_click=randomize_seed)
    st.markdown("**Befall und Beobachtung**")
    ni = st.slider("Befallene Betriebe", *bounds("ni_slider"), key="ni_slider", help="Wie viele Betriebe befallen sind, wenn Sie hinsehen.")
    obs = st.slider("Beobachtet (%)", *bounds("obs_slider"), key="obs_slider", help="Anteil der Befallenen, die Sie kennen; die übrigen fehlen. Ausgewertet wird die größte zusammenhängende Gruppe der Beobachteten.")

sync_query_params({"netztyp_select": typ, "n_slider": int(n), "c_slider": int(c), "seed_input": int(seed), "d_slider": int(d), "ni_slider": int(ni), "obs_slider": int(obs)})

settings = Settings(typ, int(n), int(c), int(seed), int(d), int(ni), int(obs))
with st.spinner("Lasse den Befall wachsen und schätze die Quelle …"):
    an = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wo ist die Quelle, und wo liegen die Schätzungen?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Befallen / beobachtet", f"{an.ni} / {len(an.seen)}", delta=f"ausgewertet: {len(an.nodes)}", delta_color="off")
for col, e in zip((m2, m3, m4), (C.E_RC, C.E_DC, C.E_J)):
    col.metric(e, f"Abstand {de(an.dists[e], 1)}", delta=("Treffer" if an.hits[e] >= 1.0 else f"Treffer {de(an.hits[e], 2)}") if an.hits[e] > 0 else "kein Treffer", delta_color="off")
st.markdown("**Das Netz im Lauf** (rot umrandet: die wahre Quelle; Raute, Quadrat, Dreieck: die Schätzungen der Rumor Centrality, der Distanzzentralität und des Jordan-Zentrums)")
st.plotly_chart(build_network(an), width="stretch", key="net_chart")
st.markdown("**Abstand der Schätzungen zur Quelle** (in diesem Lauf)")
st.plotly_chart(build_run_distances(an), width="stretch", key="run_chart")
if an.ranks:
    st.caption("Rang der wahren Quelle (1 = bester Wert): " + ", ".join(f"{e} {r}" for e, r in an.ranks.items()) + f"; von {len(an.nodes)} Betrieben der ausgewerteten Gruppe.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wo hat es angefangen?")
kind, text = verdict_main(an)
getattr(st, kind)(text)
st.caption("Ein einzelner Lauf ist ein Zufallsereignis; die Trefferquoten über viele Läufe zeigen die Experimente.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

n_exp, c_exp = settings.n, settings.c
exp_typ = typ if typ in C.GENERATED_TYPES else C.T_RANDOM
st.subheader("🔬 Experiment 1: Reguläre Bäume und der Satz zur Linie")
st.caption(f"Trefferquote der Rumor Centrality auf befallenen regulären Bäumen mit Verzweigung *d* = {', '.join(str(x) for x in C.TREE_DS)} bei {', '.join(str(x) for x in C.TREE_NS)} Befallenen; {C.TREE_RUNS} Läufe je Zelle.")


def show_trees(rows):
    st.plotly_chart(build_tree_hits(rows), width="stretch", key="tree_chart")
    kind_, text_ = verdict_trees(rows)
    getattr(st, kind_)(text_)


experiment("trees", "Bäume messen", "fest", _trees, show_trees, "Rechnet 20 Zellen mit je 150 Läufen auf Bäumen, einige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 2: Wo trifft die Quelle über Zufall?")
st.caption(f"Sechs Netztypen, befallen {pct(C.NET_FRACS[0])} % und {pct(C.NET_FRACS[1])} % der Betriebe, Quelle zufällig; {C.NET_RUNS} Läufe je festem Netz ({len(C.LIVE_SEEDS)} Netze je Typ), n = {n_exp}, c = {c_exp}.")


def show_networks(rows):
    for frac in C.NET_FRACS:
        st.markdown(f"**Trefferquote bei {pct(frac)} % befallen**")
        st.plotly_chart(build_network_hits(rows, frac), width="stretch", key=f"net_hits_{int(100 * frac)}")
    st.markdown(f"**Abstand zur Quelle bei {pct(C.NET_FRACS[1])} % befallen**")
    st.plotly_chart(build_network_distances(rows, C.NET_FRACS[1]), width="stretch", key="net_dist_chart")
    for frac in C.NET_FRACS:
        kind_, text_ = verdict_networks(rows, frac)
        getattr(st, kind_)(text_)


experiment("networks", "Netztypen vergleichen", (n_exp, c_exp), lambda: _networks(n_exp, c_exp), show_networks, "Rechnet 12 Zellen mit je 100 Läufen, etwa 10 Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 3: Helfen mehr Befallene?")
st.caption(f"Im Netztyp „{exp_typ}“ (bei „Regulärer Baum“ das Zufallsnetz): Trefferquote und Abstand über die Zahl der Befallenen {', '.join(str(x) for x in C.SIZES)}; {C.NET_RUNS} Läufe je festem Netz, n = {n_exp}, c = {c_exp}.")


def show_sizes(rows):
    st.plotly_chart(build_sizes(rows), width="stretch", key="size_chart")
    st.plotly_chart(build_size_distances(rows), width="stretch", key="size_dist_chart")
    kind_, text_ = verdict_sizes(rows)
    getattr(st, kind_)(text_)


experiment("sizes", "Größen vergleichen", (exp_typ, n_exp, c_exp), lambda: _sizes(exp_typ, n_exp, c_exp), show_sizes, "Rechnet fünf Größen mit je 100 Läufen, wenige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 4: Was, wenn nicht alle Befallenen bekannt sind?")
st.caption(f"Im Netztyp „{exp_typ}“ bei 30 % befallen: beobachtet werden {', '.join(pct(x) for x in C.OBS_SHARES)} % der Befallenen, ausgewertet wird die größte zusammenhängende Gruppe der Beobachteten; n = {n_exp}, c = {c_exp}.")


def show_observation(rows):
    st.plotly_chart(build_observation(rows), width="stretch", key="obs_chart")
    kind_, text_ = verdict_observation(rows)
    getattr(st, kind_)(text_)


experiment("observation", "Beobachtung testen", (exp_typ, n_exp, c_exp), lambda: _observation(exp_typ, n_exp, c_exp), show_observation, "Rechnet drei Anteile mit je 100 Läufen, wenige Sekunden.")

st.markdown("---")

st.subheader("🔬 Experiment 5: Wie gut könnte eine Schätzung überhaupt sein?")
st.caption(f"{C.EXACT_NETS} kleine verbundene Zufallsnetze ({C.EXACT_N} Betriebe, {C.EXACT_M} Partnerschaften) mit **exakten** Wahrscheinlichkeiten (Teilmengen-Rechnung über alle Befallsmengen); das **Bayes-Optimum** nennt zu jeder beobachteten Menge die Quelle mit der größten Wahrscheinlichkeit und ist damit die Obergrenze jedes Schätzers.")


def show_exact(rows):
    st.plotly_chart(build_exact(rows), width="stretch", key="exact_chart")
    kind_, text_ = verdict_exact(rows)
    getattr(st, kind_)(text_)


experiment("exact", "Exakt rechnen", "fest", _exact, show_exact, "Rechnet acht Netze und alle Befallsmengen, wenige Sekunden.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **SI mit gleicher Rate je Randkante** | Jede Partnerschaft überträgt gleich schnell, der Befall springt nacheinander. Mit unterschiedlichen Raten, Genesung (SIR) oder echten Zeiten ändern sich Wahrscheinlichkeiten und Schätzer; die Rumor Centrality ist nur für gleiche Raten auf bestimmten Bäumen der Maximum-Likelihood-Schätzer. | – |
| **Vollständige Beobachtung der Menge** | Fehlt ein Teil der Befallenen, zerfällt die Menge; das Verfahren ist dafür nicht gebaut (Experiment 4 zeigt das als Grenze). Beobachter mit Zeitstempeln (Pinto, Thiran, Vetterli 2012) sind ein anderes Verfahren. | – |
| **Ein einzelner Ursprung** | Zwei Quellen in verschiedenen Teilen des Netzes verwirren alle Schätzer. | – |
| **Bekanntes Netz** | Die Schätzer kennen alle Partnerschaften der Betriebe. | – |
| **BFS-Baum auf Netzen mit Kreisen** | Nur auf Bäumen zählt die Rumor Centrality alle Reihenfolgen genau; auf anderen Netzen ist sie eine Näherung, und die Demo findet sie dort nicht besser als die Distanzzentralität. | – |
| **Eden-Wachstum zählt Schritte, nicht Zeit** | Die Reihenfolge der Sprünge ist die einer Rate-Dynamik; absolute Zeiten gehen nicht ein. | Stück 13 (zeitliche Netze) |
| **Stichproben** | Trefferquoten sind Mittel über 100 bis 150 Läufe je Zelle und vier feste Netze je Typ; die Streuung zwischen den Netzen ist nicht ausgewiesen. Exakt gerechnet ist nur das kleine Netz mit 9 Betrieben. | – |
| **Modellbefund** | Die Demo zeigt Verhalten im Modell, nicht die Wirkung in der Wirklichkeit. Wer den Ursprung kennt, kann gezielt eindämmen. | Stück 7 (Eindämmung) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Wachstum.** Zu einem Zeitpunkt seien die Betriebe der Menge $S$ befallen, und $E(S)$ sei die Menge der Kanten zwischen $S$ und dem Rest. Der Befall springt über eine gleichverteilt gewählte Kante aus $E(S)$; ein Betrieb $w \notin S$ wird mit Wahrscheinlichkeit

$$P(w \mid S) = \frac{|\{(u, w) \in E(S)\}|}{|E(S)|}$$

der nächste. Die Wahrscheinlichkeit, nach $k - 1$ Schritten ab der Quelle $v$ genau die Menge $S$ befallen zu haben, ist die Summe über alle zulässigen Reihenfolgen $\sigma$ von $v$ aus des Produkts dieser Schrittwahrscheinlichkeiten:

$$P(S \mid v) = \sum_{\sigma} \prod_{t=1}^{k-1} P(\sigma_{t+1} \mid \{\sigma_1, \dots, \sigma_t\})$$

**Bayes-Schätzer.** Bei gleichverteilter Quelle ist die Quelle mit der größten Wahrscheinlichkeit der beobachteten Menge die beste mögliche Antwort:

$$\hat v = \arg\max_v \, P(S \mid v)$$

Die Demo rechnet das exakt für kleine Netze (Experiment 5).

**Rumor Centrality** (Shah und Zaman 2011). Für einen Baum mit $n$ Knoten und einen Kandidaten $v$ sei $T_u^v$ die Größe des Teilbaums von $u$ bei Wurzel $v$. Dann ist

$$R(v) = \frac{n!}{\prod_{u} T_u^v}$$

die Zahl der Reihenfolgen, in denen der Befall von $v$ aus die Menge erreichen kann. Auf regulären Bäumen haben alle Reihenfolgen dieselbe Wahrscheinlichkeit, $R$ ist dort der Maximum-Likelihood-Schätzer. Beim Umwurzeln von $p$ auf einen Nachbarn $c$ gilt

$$R(c) = R(p)\,\frac{T_c^p}{n - T_c^p}$$

**Distanzzentralität.** $\hat v = \arg\min_v \sum_u d(v, u)$; auf Bäumen ist das dieselbe Rangfolge wie bei $R$.

**Literatur.**
- Shah, D., Zaman, T. (2011): *Rumors in a Network: Who's the Culprit?* IEEE Transactions on Information Theory 57(8), 5163–5181.
- Pinto, P. C., Thiran, P., Vetterli, M. (2012): *Locating the Source of Diffusion in Large-Scale Networks.* Physical Review Letters (Ausblick: Beobachter mit Zeitstempeln).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
