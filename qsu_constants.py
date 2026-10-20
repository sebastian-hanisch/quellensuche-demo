"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Netztypen -----------------------------------------------------------------------------------------------------------------------------------
T_RANDOM = "Zufallsnetz"
T_POWER = "Potenzgesetz-Netz"
T_BA = "Skalenfreies Netz (BA)"
T_COMMUNITIES = "Verbünde"
T_RING = "Ring mit Abkürzungen"
T_GRID = "Gitter"
T_TREE = "Regulärer Baum"
GENERATED_TYPES = (T_RANDOM, T_POWER, T_BA, T_COMMUNITIES, T_RING, T_GRID)
ALL_TYPES = GENERATED_TYPES + (T_TREE,)
BLOCKS = 6

C_MEANING = {
    T_RANDOM: "mittlerer Grad des Zufallsnetzes",
    T_POWER: "mittlerer Grad (Gradfolge mit Potenzgesetz-Exponent 2,5, Konfigurationsmodell)",
    T_BA: "zwei Kanten je neuem Knoten, also m = c / 2 gerundet (jeder Knoten hat mindestens m Nachbarn)",
    T_COMMUNITIES: "mittlerer Grad; 80 % der Kanten bleiben im eigenen Verbund (6 Verbünde)",
    T_RING: "Nachbarn auf dem Ring (c gerade gerundet), dazu Abkürzungen mit 5 % je Ringkante",
    T_GRID: "Nachbarschaft: unter 6 vier Nachbarn, 6 bis 7 mit einer Diagonale, ab 8 mit beiden Diagonalen",
    T_TREE: "wirkt nicht: im regulären Baum hat jeder Betrieb d Partner (Verzweigung d), das Netz wächst mit dem Befall",
}

N_MIN, N_MAX, DEFAULT_N = 60, 250, 150
C_MIN, C_MAX, DEFAULT_C = 3, 12, 6
SEED_MAX, DEFAULT_SEED = 999999, 35
D_MIN, D_MAX, DEFAULT_D = 2, 6, 3                 # Verzweigung des regulären Baums (2 = Linie)
NI_MIN, NI_MAX, DEFAULT_NI = 5, 150, 30           # Zahl der befallenen Betriebe
OBS_MIN, OBS_MAX, DEFAULT_OBS = 50, 100, 100      # beobachteter Anteil der Befallenen in Prozent
SWEEP_SEEDS = tuple(range(100000, 100008))        # feste Netze für alle Sweeps: nie ein einzelner Seed als Befund
LIVE_SEEDS = SWEEP_SEEDS[:4]                      # so viele feste Netze je Zelle rechnet die App live (die README-Zahlen nutzen alle acht)

# --- Schätzer ------------------------------------------------------------------------------------------------------------------------------------
E_RC, E_DC, E_J, E_RAND = "Rumor Centrality", "Distanzzentralität", "Jordan-Zentrum", "Zufall"
ESTIMATORS = (E_RC, E_DC, E_J, E_RAND)
ESTIMATOR_HELP = {
    E_RC: "Zählt für jeden Kandidaten die möglichen Reihenfolgen, in denen der Befall von ihm aus die beobachtete Menge erreichen kann (Shah und Zaman 2011); auf Bäumen der Maximum-Likelihood-Schätzer.",
    E_DC: "Der Betrieb mit der kleinsten Summe der Abstände zu allen befallenen Betrieben.",
    E_J: "Der Betrieb mit dem kleinsten größten Abstand zu einem befallenen Betrieb (Jordan-Zentrum).",
    E_RAND: "Ein beliebiger befallener Betrieb.",
}

# --- Messungen -----------------------------------------------------------------------------------------------------------------------------------
TREE_DS = (2, 3, 4, 5)
TREE_NS = (10, 20, 50, 100, 200)
TREE_RUNS = 150                                   # Läufe je Zelle in Experiment 1 (live)
NET_FRACS = (0.10, 0.30)                          # befallener Anteil in Experiment 2
NET_RUNS = 25                                     # Läufe je Netz und Zelle (live: je festes Netz)
SIZES = (10, 20, 40, 80, 150)                     # Zahl der Befallenen in Experiment 3
OBS_SHARES = (1.0, 0.8, 0.5)                      # beobachteter Anteil in Experiment 4
EXACT_N, EXACT_M = 9, 12
EXACT_NETS, EXACT_NIS = 8, (4, 5, 6)

# --- Darstellung ---------------------------------------------------------------------------------------------------------------------------------
COLOR_EDGE = "rgba(120,120,120,0.30)"
COLOR_SOURCE = "#d62728"
COLOR_INFECTED = "#e0a458"
COLOR_HEALTHY = "#b8b8b8"
COLOR_A = "#2F6B65"
COLOR_B = "#c4623b"
COLOR_C = "#6b5b95"
ESTIMATOR_COLORS = {E_RC: "#2F6B65", E_DC: "#6b5b95", E_J: "#c4623b", E_RAND: "#8c8c8c"}
ESTIMATOR_SYMBOLS = {E_RC: "diamond", E_DC: "square", E_J: "triangle-up", E_RAND: "x"}

PRESETS = {
    "Zufallsnetz: die Quelle liegt in der Mitte (Voreinstellung)": {"netztyp": T_RANDOM, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "d": DEFAULT_D, "ni": DEFAULT_NI, "obs": DEFAULT_OBS},
    "Linie: der Satz": {"netztyp": T_TREE, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "d": 2, "ni": 100, "obs": DEFAULT_OBS},
    "Baum mit drei Partnern: es geht": {"netztyp": T_TREE, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "d": 3, "ni": 100, "obs": DEFAULT_OBS},
    "Skalenfrei: Treffer wie Zufall, Abstand besser": {"netztyp": T_BA, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "d": DEFAULT_D, "ni": 40, "obs": DEFAULT_OBS},
    "Ring: Distanz schlägt Rumor Centrality": {"netztyp": T_RING, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "d": DEFAULT_D, "ni": 50, "obs": DEFAULT_OBS},
    "Fast alles befallen: mehr hilft nicht": {"netztyp": T_RANDOM, "n": DEFAULT_N, "c": DEFAULT_C, "seed": DEFAULT_SEED, "d": DEFAULT_D, "ni": 120, "obs": DEFAULT_OBS},
}
PRESET_HELP = {
    "Zufallsnetz: die Quelle liegt in der Mitte (Voreinstellung)": "Ein Befall mit 30 Betrieben im Zufallsnetz: die Schätzer suchen die Mitte der befallenen Gruppe.",
    "Linie: der Satz": "Auf der Linie hat der Befall nur zwei Richtungen: die Quelle lässt sich kaum finden, und mit mehr Befallenen wird es schlechter.",
    "Baum mit drei Partnern: es geht": "Wächst der Befall schneller als eine Linie, bleibt die Chance auf einen Treffer bei einer festen Größe.",
    "Skalenfrei: Treffer wie Zufall, Abstand besser": "Im Netz mit Superspreadern trifft kein Schätzer die Quelle öfter als Zufall, liegt aber näher an ihr.",
    "Ring: Distanz schlägt Rumor Centrality": "Auf nicht baumartigen Netzen ist die Rumor Centrality (mit BFS-Baum) nicht besser als die einfache Distanzzentralität.",
    "Fast alles befallen: mehr hilft nicht": "Je mehr Betriebe befallen sind, desto weniger verrät ihre Menge über den Anfang.",
}
DEFAULT_PRESET_NAME = "Zufallsnetz: die Quelle liegt in der Mitte (Voreinstellung)"
