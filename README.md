# Quellensuche: wo hat der Befall angefangen? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-quellensuche-demo.streamlit.app/)** -->

Achtes Stück der **Netzwerkanalyse-Reihe** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, im Ausbreitungs-Arm das **inverse Problem** zu den Stücken 5 bis 7: dort war die Quelle gegeben und der Verlauf gesucht, hier ist der Endzustand gegeben und die Quelle gesucht (Zusammenfluss mit Stück 7: wer den Ursprung kennt, kann gezielt eindämmen). Die [Demo zu Kaskaden](https://sebastianhanisch-kaskaden-demo.streamlit.app/) rechnet die Ausbreitung in Vorwärtsrichtung, die [Zentralitäts-Demo](https://sebastianhanisch-centrality-demo.streamlit.app/) kennt die Maße, nicht aber ihre Verwendung als Schätzer des Ursprungs.

In einem Netz von Subunternehmern und Lieferanten ist ein Befall aufgetreten. Er beginnt in einem Betrieb (der **Quelle**) und wächst als **SI**: in jedem Schritt springt er über eine zufällig gewählte Randkante (Partnerschaft zwischen einem befallenen und einem gesunden Betrieb), wer befallen ist, bleibt es (Eden-Wachstum). Beobachtet wird die **Menge** der Befallenen. Die Demo stellt die **Rumor Centrality** (Shah und Zaman 2011) der **Distanzzentralität**, dem **Jordan-Zentrum** und dem **Zufallstipp** gegenüber und zeigt, **wo die Quellensuche an ihre Grenze stößt**.

## Kernfrage

**Wo hat es angefangen, und wo geht es nicht mehr?** Der Hook ist ein **Grenzenbefund**: auf der Linie versagt der Schätzer (der Satz von Shah und Zaman), in Netzen mit stark schwankendem Grad trifft die Rumor Centrality die Quelle nicht über Zufall (das Jordan-Zentrum im skalenfreien Netz bei 10 % etwa 1,6-fach), aber der Schätzer liegt näher an ihr, und **mehr Befallene verschlechtern die Treffer** (bei 250 Knoten).

## Befunde und Korrekturen gegenüber dem Plan

- **Auf Bäumen sind Rumor Centrality und Distanzzentralität dasselbe Maß** (gleiche Kandidaten in jeder Zelle der Messung; die Vermutung im Gesamtplan, sie nacheinander gegeneinander antreten zu lassen, entfällt dort). Die Rumor Centrality zählt genau die Reihenfolgen, in denen der Befall von einem Kandidaten aus die Menge erreichen kann (Zählung aller Reihenfolgen auf 60 Zufallsbäumen bestätigt).
- **Der Satz zur Linie stimmt:** auf der Linie fällt die Trefferquote der Rumor Centrality von 25,7 % bei 10 auf 6,2 % bei 200 Befallenen (etwa wie 1/√*n*), bei Verzweigung *d* = 3 von 32,2 % auf 23,3 %, bei *d* = 4 und 5 bleibt sie bei 27,3 % bzw. 28,1 %: ab *d* = 3 bleibt sie bei etwa einem Viertel bis einem Drittel.
- **Rumor Centrality ist auf Netzen mit Kreisen nicht besser als die Distanzzentralität** (Plan-Vermutung nach der Simulation bei Shah und Zaman nicht bestätigt): bei 30 % Befallenen trifft die Distanz im Zufallsnetz 6,3 % gegen 4,9 %, im Gitter 5,5 % gegen 3,0 %, im Ring mit Abkürzungen **7,9 % gegen 2,6 %** (BFS-Baum je Kandidat als Näherung).
- **Quelle in heavy-tailed Netzen:** die Trefferquote liegt bei Zufall oder darunter (Potenzgesetz-Netz bei 10 % befallen: Rumor Centrality **0,4 %** gegen 4,0 % bei Zufall, weil sie Hubs wählt), der **Abstand** zur Quelle ist aber deutlich kleiner als bei Zufall (skalenfrei 1,77 gegen 2,33).
- **Mehr Befallene (bei 250 Knoten) verbessern die Trefferquote nicht, sie sinkt:** im Zufallsnetz von 26,5 % bei 10 auf 1,0 % bei 150 Befallenen (Abstand 1,07 → 2,68), im Ring von 21,1 % auf 0,1 % (Abstand 0,92 → 4,22). Der Plan fragte „ab welcher Größe ist die Quelle erkennbar“; die Richtung war falsch.
- **Auch das Bayes-Optimum verfehlt die Quelle meist:** auf kleinen Netzen (9 Betriebe) trifft es mit 4 bis 6 Befallenen 43,5 % bis 30,8 %, die Rumor Centrality 29,3 % bis 19,7 %, Zufall 25,0 % bis 16,7 %.
- **Fehlende Beobachtungen zerlegen die Menge:** bei 80 % bzw. 50 % beobachteten Befallenen bleiben im Zufallsnetz 96 % bzw. 66 % der Beobachteten in der größten zusammenhängenden Gruppe. Die Quelle liegt oft nicht in der ausgewerteten Gruppe (bei 50 % Beobachtung in 41 % der Läufe); die Trefferquote fällt dadurch von 6,2 % auf 2,6 % bzw. 3,2 %.
- **Rundungsfehler bei Gleichständen:** das Runden auf feste Stellen trennt zwei mathematisch gleiche Zentren (ln *R* weicht um etwa 1e-11 ab) und ließ Rumor Centrality und Distanz auf 3-regulären Bäumen scheinbar abweichen (40 von 3 000 Bäumen); jetzt gilt eine relative Toleranz (Regressionstest).

## Modell

SI als Eden-Wachstum: in jedem Schritt springt der Befall über eine gleichverteilt gewählte Randkante. Beobachtet wird die Menge der Befallenen (bei unvollständiger Beobachtung eine zufällige Teilmenge, ausgewertet wird ihre größte zusammenhängende Gruppe). **Rumor Centrality:** *R*(*v*) = *n*! / Π<sub>*u*</sub> *T*<sub>*u*</sub><sup>*v*</sup> mit den Teilbaumgrößen bei Wurzel *v* (im Logarithmus, auf Bäumen über Umwurzeln in O(*n*), sonst mit dem BFS-Baum je Kandidat); **Distanzzentralität:** kleinste Abstandssumme; **Jordan-Zentrum:** kleinste Exzentrizität; **Zufall:** ein beliebiger Befallener. **Treffer:** 1/*k* bei *k* gleich guten Kandidaten, wenn die Quelle darunter ist. **Abstand:** mittlerer Abstand der Kandidaten zur Quelle im ganzen Netz. **Bayes-Optimum** (nur kleine Netze): die Quelle mit der größten Wahrscheinlichkeit der beobachteten Menge, exakt durch Teilmengen-Rechnung.

## Methodik

| Baustein | Inhalt |
|---|---|
| **Netze** | Generatoren aus `k-core-demo` (Stück 1) als Kopie: Zufallsnetz, Potenzgesetz-Netz, skalenfrei (BA), Verbünde, Ring mit Abkürzungen, Gitter; dazu der befallene Baum einer unendlichen *d*-regulären Struktur (*d* = 2 ist die Linie) |
| **Experiment 1** | reguläre Bäume: Trefferquote über die Zahl der Befallenen (10 bis 200) für *d* = 2 bis 5 |
| **Experiment 2** | sechs Netztypen, 10 % und 30 % befallen: Trefferquote und Abstand je Schätzer |
| **Experiment 3** | Trefferquote und Abstand über die Zahl der Befallenen (10 bis 150) im gewählten Netztyp |
| **Experiment 4** | beobachteter Anteil 100, 80, 50 % |
| **Experiment 5** | exakt auf 8 kleinen Netzen (9 Betriebe, 12 Partnerschaften) mit 4, 5, 6 Befallenen: Bayes-Optimum gegen die Schätzer |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, *n* = 250, *c* = 6, je Netz 100 Läufe (800 Läufe je Zelle; Bäume 1 000 Läufe je Zelle), Quelle je Lauf zufällig; Trefferquote in Prozent, Abstand in Schritten.

| Frage | Befund |
|---|---|
| **Stimmen die Bausteine?** | *R*(*v*) stimmt auf 60 Zufallsbäumen mit 3 bis 8 Knoten für jede Wurzel mit der **Zählung aller Infektionsreihenfolgen** überein; das Umwurzeln mit der Neuberechnung je Wurzel und auf 100 Bäumen mit der Gleichheit der Kandidaten von Rumor Centrality und Distanz; Abstandssummen und Exzentrizitäten mit `networkx` (30 Netze); die exakten Wahrscheinlichkeiten der befallenen Mengen (Teilmengen-Rechnung) mit der Häufigkeit aus 30 000 Wachstumsläufen; die Trefferwahrscheinlichkeit der Rumor Centrality aus der exakten Rechnung mit der Simulation (20 000 Läufe, Band 0,012); Bayes ≥ jeder Schätzer, Zufall = 1/*n*<sub>I</sub> exakt (`tests/test_oracle_quellensuche.py`) |
| **Satz zur Linie** | Tabelle zu Experiment 1 |
| **Wo trifft die Quelle über Zufall?** | Tabellen zu Experiment 2: bei 10 % befallen im Zufallsnetz 15,1 % (Rumor Centrality) bzw. 17,5 % (Distanz) gegen 4,1 % bei Zufall, in Verbünden 16,1 % bzw. 16,8 % gegen 4,0 %; im Potenzgesetz-Netz und im skalenfreien Netz kein Schätzer über dem 1,5-Fachen von Zufall bei 30 % |
| **Helfen mehr Befallene?** | Nein, bei 250 Knoten sinkt die Trefferquote (Tabelle zu Experiment 3) |
| **Was bei fehlenden Beobachtungen?** | Tabelle zu Experiment 4 |
| **Wie gut kann eine Schätzung sein?** | Tabelle zu Experiment 5 |
| **Standardlauf (Zufallsnetz, n = 150, Seed 35)** | 450 Kanten; 30 befallen; Treffer Rumor Centrality 0,00, Distanz 0,00, Jordan 0,00; Abstand 1,00 / 2,00 / 2,00, Zufall 2,43; Rang der Quelle 3 / 5 / 4 |
| **Preset Linie** | 99 Kanten; 100 befallen; Treffer Rumor Centrality 0,00, Distanz 0,00, Jordan 0,00; Abstand 2,50 / 2,50 / 2,50, Zufall 25,06; Rang der Quelle 5 / 5 / 5 |
| **Preset Baum mit drei Partnern** | 99 Kanten; 100 befallen; Treffer Rumor Centrality 0,00, Distanz 0,00, Jordan 0,00; Abstand 1,00 / 1,00 / 1,00, Zufall 5,45; Rang der Quelle 3 / 3 / 2 |
| **Preset Skalenfrei** | 444 Kanten; 40 befallen; Treffer Rumor Centrality 0,00, Distanz 0,00, Jordan 0,00; Abstand 2,00 / 2,00 / 2,00, Zufall 2,38; Rang der Quelle 37 / 26 / 27 |
| **Preset Ring** | 471 Kanten; 50 befallen; Treffer Rumor Centrality 0,00, Distanz 1,00, Jordan 1,00; Abstand 2,00 / 0,00 / 0,00, Zufall 3,02; Rang der Quelle 5 / 1 / 1 |
| **Preset Fast alles befallen** | 450 Kanten; 120 befallen; Treffer Rumor Centrality 0,00, Distanz 0,00, Jordan 0,02; Abstand 2,00 / 2,00 / 2,85, Zufall 2,96; Rang der Quelle 53 / 76 / 1 |

**Experiment 1: Trefferquote der Rumor Centrality (= Distanzzentralität) auf befallenen regulären Bäumen** (Prozent; 1 000 Läufe je Zelle):

| Verzweigung | 10 Befallene | 20 Befallene | 50 Befallene | 100 Befallene | 200 Befallene |
|---|---|---|---|---|---|
| *d* = 2 (Linie) | 25,7 | 18,2 | 10,5 | 9,2 | 6,2 |
| *d* = 3 | 32,2 | 28,5 | 25,9 | 26,2 | 23,3 |
| *d* = 4 | 32,7 | 30,1 | 26,9 | 27,2 | 27,3 |
| *d* = 5 | 34,3 | 28,7 | 29,4 | 29,3 | 28,1 |
| Zufall | 10,0 | 5,0 | 2,0 | 1,0 | 0,5 |

Das Jordan-Zentrum trifft bei Verzweigung ab 3 seltener (Prozent):

| Jordan-Zentrum | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| *d* = 3 | 30,5 | 24,7 | 23,7 | 22,1 | 18,8 |
| *d* = 4 | 30,8 | 27,4 | 23,4 | 20,8 | 20,5 |
| *d* = 5 | 31,0 | 26,2 | 26,0 | 23,4 | 20,3 |

**Experiment 2: Trefferquote bei 10 % befallenen Betrieben** (Prozent):

| Netztyp | Rumor Centrality | Distanz | Jordan | Zufall |
|---|---|---|---|---|
| Zufallsnetz | 15,1 | 17,5 | 14,5 | 4,1 |
| Potenzgesetz-Netz | 0,4 | 0,7 | 3,6 | 4,0 |
| Skalenfreies Netz (BA) | 2,4 | 3,7 | 6,6 | 4,0 |
| Verbünde | 16,1 | 16,8 | 14,0 | 4,0 |
| Ring mit Abkürzungen | 10,0 | 13,6 | 11,8 | 4,0 |
| Gitter | 9,3 | 12,7 | 8,8 | 3,8 |

**Trefferquote bei 30 % befallenen Betrieben** (Prozent):

| Netztyp | Rumor Centrality | Distanz | Jordan | Zufall |
|---|---|---|---|---|
| Zufallsnetz | 4,9 | 6,3 | 3,8 | 1,7 |
| Potenzgesetz-Netz | 0,1 | 0,3 | 1,0 | 1,3 |
| Skalenfreies Netz (BA) | 0,3 | 1,5 | 1,6 | 1,3 |
| Verbünde | 4,0 | 6,0 | 5,6 | 1,3 |
| Ring mit Abkürzungen | 2,6 | 7,9 | 6,3 | 1,3 |
| Gitter | 3,0 | 5,5 | 4,0 | 1,3 |

**Mittlerer Abstand der Kandidaten zur Quelle bei 30 % befallenen Betrieben** (Schritte):

| Netztyp | Rumor Centrality | Distanz | Jordan | Zufall |
|---|---|---|---|---|
| Zufallsnetz | 2,28 | 2,07 | 2,36 | 2,93 |
| Potenzgesetz-Netz | 1,74 | 1,63 | 1,92 | 2,56 |
| Skalenfreies Netz (BA) | 2,05 | 1,86 | 2,08 | 2,66 |
| Verbünde | 2,19 | 1,86 | 2,22 | 2,98 |
| Ring mit Abkürzungen | 2,95 | 2,08 | 2,20 | 3,91 |
| Gitter | 3,06 | 2,45 | 2,68 | 4,38 |

**Experiment 3: Trefferquote und Abstand über die Zahl der Befallenen** (Prozent bzw. Schritte):

*Zufallsnetz*:

| Befallene | Rumor Centrality Treffer | Abstand | Distanz Treffer | Abstand | Zufall Treffer | Abstand |
|---|---|---|---|---|---|---|
| 10 | 26,5 | 1,07 | 26,9 | 1,05 | 10,6 | 1,88 |
| 20 | 18,1 | 1,40 | 20,5 | 1,32 | 5,5 | 2,34 |
| 40 | 12,8 | 1,70 | 13,8 | 1,55 | 2,9 | 2,69 |
| 80 | 5,6 | 2,27 | 8,0 | 2,03 | 1,7 | 2,91 |
| 150 | 1,0 | 2,68 | 2,8 | 2,45 | 0,9 | 3,10 |

*Skalenfreies Netz (BA)*:

| Befallene | Rumor Centrality Treffer | Abstand | Distanz Treffer | Abstand | Zufall Treffer | Abstand |
|---|---|---|---|---|---|---|
| 10 | 11,5 | 1,30 | 10,8 | 1,29 | 10,0 | 1,88 |
| 20 | 4,0 | 1,64 | 5,2 | 1,54 | 5,0 | 2,22 |
| 40 | 1,1 | 1,90 | 2,2 | 1,75 | 2,5 | 2,48 |
| 80 | 0,0 | 2,04 | 0,5 | 1,88 | 1,3 | 2,68 |
| 150 | 0,0 | 2,18 | 0,4 | 1,93 | 0,7 | 2,84 |

*Ring mit Abkürzungen*:

| Befallene | Rumor Centrality Treffer | Abstand | Distanz Treffer | Abstand | Zufall Treffer | Abstand |
|---|---|---|---|---|---|---|
| 10 | 21,1 | 0,92 | 21,7 | 0,90 | 10,0 | 1,50 |
| 20 | 9,6 | 1,31 | 13,3 | 1,17 | 5,0 | 2,21 |
| 40 | 5,1 | 2,02 | 9,5 | 1,55 | 2,5 | 3,10 |
| 80 | 1,6 | 3,02 | 5,8 | 2,11 | 1,3 | 3,98 |
| 150 | 0,1 | 4,22 | 3,7 | 3,20 | 0,7 | 4,76 |

**Experiment 4: unvollständige Beobachtung** (ausgewertet wird die größte zusammenhängende Gruppe der Beobachteten):

*Zufallsnetz* (30 % befallen):

| Beobachtet | in der Gruppe | Rumor Centrality Treffer | Abstand | Distanz Treffer | Abstand |
|---|---|---|---|---|---|
| 100 % | 100 % | 6,2 | 2,19 | 7,5 | 1,94 |
| 80 % | 96 % | 2,6 | 2,43 | 5,8 | 2,19 |
| 50 % | 66 % | 3,2 | 2,47 | 3,7 | 2,46 |

*Gitter* (30 % befallen):

| Beobachtet | in der Gruppe | Rumor Centrality Treffer | Abstand | Distanz Treffer | Abstand |
|---|---|---|---|---|---|
| 100 % | 100 % | 4,1 | 3,01 | 6,0 | 2,45 |
| 80 % | 98 % | 2,6 | 2,95 | 4,5 | 2,54 |
| 50 % | 58 % | 2,7 | 3,21 | 3,1 | 3,16 |

**Experiment 5: exakte Trefferwahrscheinlichkeit auf kleinen Netzen** (Prozent; Mittel über 8 Netze; das Bayes-Optimum ist in jedem Netz mindestens so gut wie jeder Schätzer):

| Befallene | Bayes-Optimum | Rumor Centrality | Distanz | Jordan | Zufall |
|---|---|---|---|---|---|
| 4 | 43,5 | 29,3 | 29,3 | 29,3 | 25,0 |
| 5 | 37,0 | 23,8 | 23,8 | 22,3 | 20,0 |
| 6 | 30,8 | 19,7 | 19,4 | 18,3 | 16,7 |

### Hypothesen des Einzelplans

| | Hypothese | Ergebnis |
|---|---|---|
| H1 | *R*(*v*) zählt die Infektionsreihenfolgen ab *v* | ✅ 60 Zufallsbäume, jede Wurzel, 0 Abweichungen |
| H2 | Auf Bäumen sind Rumor Centrality und Distanzzentralität dasselbe Maß | ✅ gleiche Kandidaten in jeder Zelle (20 Zellen mit je 1 000 Läufen); Voraussetzung ist die Toleranz bei Gleichständen |
| H3 | Satz nach Shah und Zaman: auf der Linie geht die Trefferquote gegen 0, bei schnellerem Wachstum bleibt sie positiv | ✅ Linie 25,7 % → 6,2 %, *d* ≥ 3 bleibt bei 23,3 % bis 28,1 % bei 200 Befallenen |
| H4 | Rumor Centrality schlägt Distanzzentralität auf allgemeinen Netzen | ❌ bei 30 % befallen in allen sechs Netztypen nicht besser, im Ring 2,6 % gegen 7,9 % |
| H5 | In Netzen mit stark schwankendem Grad ist die Quelle erkennbar | ❌ Rumor Centrality nicht über Zufall (Potenzgesetz 0,4 % gegen 4,0 %; das Jordan-Zentrum im skalenfreien Netz bei 10 % 1,6-fach), aber Abstand kleiner (1,58 gegen 2,23) |
| H6 | Der Bayes-Schätzer liegt weit über den Heuristiken | ⚠ 43,5 % gegen 29,3 % (Rumor Centrality) und 25,0 % (Zufall) bei 4 Befallenen; nur 9 Betriebe |
| H7 | Mit mehr Befallenen steigt die Trefferquote | ❌ sie fällt bei 250 Knoten (Zufallsnetz 26,5 % → 1,0 %) |
| H8 | Fehlende Beobachtungen verschlechtern die Schätzung | ✅ die Menge zerfällt (66 % bzw. 58 % der Beobachteten bei 50 % in der größten Gruppe), die Trefferquote fällt (auch weil die Quelle oft nicht in der ausgewerteten Gruppe liegt); das Maß ist dafür nicht gebaut |

## Ehrliche Grenzen

- **SI mit gleicher Rate je Randkante, vollständige Beobachtung der Menge, ein einzelner Ursprung, bekanntes Netz.** Mit Genesung (SIR), verschiedenen Raten oder mehreren Quellen sind Wahrscheinlichkeiten und Schätzer andere; die Rumor Centrality ist nur für gleiche Raten auf bestimmten Bäumen der Maximum-Likelihood-Schätzer.
- **BFS-Baum auf Netzen mit Kreisen** ist eine Näherung der Rumor Centrality; die Demo findet sie dort nicht besser als die Distanz. Ob eine andere Näherung (alle Spannbäume, Belief Propagation) mehr leistet, ist nicht gemessen.
- **Eden-Wachstum zählt Schritte, nicht Zeit;** Beobachter mit Zeitstempeln (Pinto, Thiran, Vetterli 2012) sind ein anderes Verfahren und nur als Ausblick genannt.
- **Stichproben:** Mittel über 800 bzw. 1 000 Läufe je Zelle und acht feste Netze; die Streuung zwischen den Netzen ist nicht ausgewiesen. Exakt gerechnet ist nur das Netz mit 9 Betrieben (8 Netze).
- **Unvollständige Beobachtung** ist als Grenzenversuch gebaut (größte zusammenhängende Gruppe), nicht als Verfahren für fehlende Daten.
- **Preset-Läufe** sind einzelne Läufe (ein Zufallsereignis); die Aussagen der Presets stützen die Messreihen darüber.
- **Literatur:** Shah und Zaman, IEEE Transactions on Information Theory 57(8), 2011, S. 5163–5181, nachgeschlagen (Modell SI; die arXiv-Fassung rechnet mit einer SIR-Variante); Pinto, Thiran, Vetterli (PRL 2012) nur als Ausblick, Artikelnummer nicht geprüft; die Arbeiten selbst sind nicht vollständig gelesen.

## Verifikation

- **Kern:** Handwerte auf Mini-Instanzen (`tests/conftest.py`, `tests/test_algorithm.py`): Pfad (*R* = 1, 2, 1), Stern (24 und 6), Linie (1, 4, 6, 4, 1), Dreieck, Zählung aller Reihenfolgen, Wachstum (gültige Pfade, erster Schritt gleichverteilt), Beobachtung, Komponenten, Gleichstandsanteile, exakte Mengenwahrscheinlichkeiten (Pfad: je 1/2).
- **Orakel:** `tests/test_oracle_quellensuche.py` – *R*(*v*) gegen die Zählung, Umwurzeln gegen Neuberechnung, Rumor Centrality = Distanz auf Bäumen (auch groß und regulär: Regressionstest zur Rundungstoleranz), `networkx`-Abstände, exakte Wahrscheinlichkeiten gegen Wachstumsläufe, Bayes als Obergrenze, simulierte Trefferquote gegen die exakte, Linie fällt mit der Größe.
- **Netze und Läufe:** gültig in allen Ecken der Regler, deterministisch je Seed (auch über Prozesse mit verschiedenem Hash-Seed), der befallene reguläre Baum hat die richtigen Grade (`tests/test_scenario.py`).
- **Verdikte:** jede Schwelle und jede Zahl im Text an von Hand gesetzten Werten (`tests/test_verdicts.py`), Randfälle (Regler-Ecken für alle Netztypen, Gruppen aus einem oder zwei Betrieben, kleinste Netze in allen Experimenten, alle Abbildungen) in `tests/test_edge_cases.py`.
- **Befunde:** jede Zahl dieser README in `tests/test_claims.py`.
- **Mutationsprüfung des Kerns:** (`qsu_algorithm.py`, 152 Stellen, ohne die langsamen Aussagen-Tests in der Suite der Überlebenden): 18 Stellen sind durch wortgleiche Zeilen in Nachbar-Demos gesichert, 123 Mutanten gefangen (nach Ergänzung von Tests), 10 als äquivalent festgehalten (Toleranz-Faktor, Skalierung aller Kantenzahlen mit demselben Faktor, `share >= 1` gegen `> 1` bei einer Auswahl aller Betriebe, Gleichheit genau an der Toleranz, Sonderfall eines einzelnen Betriebs); ein Mutant (Abstand im BFS um 1 verkleinert) läuft in eine Endlosschleife und wird durch das Hängen der Suite erkannt. Zustand in `tests/mutation_state.json`.
- **Presets:** vollständig, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung, jedes Preset zeigt, wofür es benannt ist.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jeder Netztyp, neuer Lauf, Permalink, alle fünf Experimente) und Footer-Test.
- **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf (`tests/test_layout.py`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `qsu_algorithm.py` | Kern: Wachstum, Schätzer (mit Umwurzeln auf Bäumen), Treffer und Abstand, exakte Mengenwahrscheinlichkeiten, Bayes-Schätzer, Zählung der Reihenfolgen (Orakel) |
| `qsu_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `qsu_evaluation.py` | Läufe, Analyse, Verdikt-Logik, Messreihen über feste Netze |
| `qsu_layout.py` | Fester Kopf (Stylesheet) |
| `qsu_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `qsu_scenario.py` | Netze: Generatoren (Kopie aus `k-core-demo`), Anordnung |
| `qsu_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx, scipy (nur für die Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Instanzen und Handwerte, Orakel, Netze, Verdikte, Randfälle, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe, Befunde |

## Bewusst nicht umgesetzt

- **SIR, verschiedene Raten, mehrere Quellen:** eigene Fragen; als möglicher Ausbau (gelb).
- **Beobachter mit Zeitstempeln** (Pinto, Thiran, Vetterli 2012) und **Belief Propagation** auf Netzen mit Kreisen: nur als Ausblick.
- **Zeitliche Netze:** Stück 13.
- **Netze über 250 Betriebe live:** die Rumor Centrality braucht auf Netzen mit Kreisen ein BFS je Kandidat; die Dauer wächst mit der Zahl der Befallenen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
