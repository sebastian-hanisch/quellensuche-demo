"""Jede Zahl aus der README wird hier über dieselben Auswertungsfunktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Netze der Messreihen: die acht festen Netze `SWEEP_SEEDS` je Zelle, n = 250, c = 6, je Netz 100 Läufe (800 Läufe je Zelle; Bäume: 1 000 Läufe je Zelle), Quelle je Lauf zufällig, SI als Eden-Wachstum; Trefferquote
(Gleichstände anteilig) und mittlerer Abstand der Kandidaten zur Quelle. Zufall läuft über seedbare Generatoren (ein Strom je Lauf) und ist plattformstabil; Trefferquoten stehen auf drei, Abstände auf zwei
Nachkommastellen, die Bänder 0,0006 und 0,006 fangen die Rundung."""

from functools import lru_cache

import pytest

import qsu_algorithm as A
import qsu_constants as C
import qsu_evaluation as ev

SW = C.SWEEP_SEEDS
RUNS, TREE_RUNS = 100, 1000
RC, DC, J, RAND = C.E_RC, C.E_DC, C.E_J, C.E_RAND
EXPECT = {'meta': {'runs': 100, 'tree_runs': 1000},
 'E1': [{'d': 2, 'ni': 10, 'cells': {'Rumor Centrality': [0.257, 1.21], 'Distanzzentralität': [0.257, 1.21], 'Jordan-Zentrum': [0.257, 1.21], 'Zufall': [0.1, 2.7]}},
        {'d': 2, 'ni': 20, 'cells': {'Rumor Centrality': [0.182, 1.7], 'Distanzzentralität': [0.182, 1.7], 'Jordan-Zentrum': [0.182, 1.7], 'Zufall': [0.05, 5.21]}},
        {'d': 2, 'ni': 50, 'cells': {'Rumor Centrality': [0.105, 2.85], 'Distanzzentralität': [0.105, 2.85], 'Jordan-Zentrum': [0.105, 2.85], 'Zufall': [0.02, 12.74]}},
        {'d': 2, 'ni': 100, 'cells': {'Rumor Centrality': [0.092, 3.93], 'Distanzzentralität': [0.092, 3.93], 'Jordan-Zentrum': [0.092, 3.93], 'Zufall': [0.01, 25.24]}},
        {'d': 2, 'ni': 200, 'cells': {'Rumor Centrality': [0.062, 5.67], 'Distanzzentralität': [0.062, 5.67], 'Jordan-Zentrum': [0.062, 5.67], 'Zufall': [0.005, 50.25]}},
        {'d': 3, 'ni': 10, 'cells': {'Rumor Centrality': [0.322, 0.93], 'Distanzzentralität': [0.322, 0.93], 'Jordan-Zentrum': [0.305, 0.96], 'Zufall': [0.1, 2.13]}},
        {'d': 3, 'ni': 20, 'cells': {'Rumor Centrality': [0.285, 1.08], 'Distanzzentralität': [0.285, 1.08], 'Jordan-Zentrum': [0.247, 1.2], 'Zufall': [0.05, 3.15]}},
        {'d': 3, 'ni': 50, 'cells': {'Rumor Centrality': [0.259, 1.2], 'Distanzzentralität': [0.259, 1.2], 'Jordan-Zentrum': [0.237, 1.31], 'Zufall': [0.02, 4.65]}},
        {'d': 3, 'ni': 100, 'cells': {'Rumor Centrality': [0.262, 1.22], 'Distanzzentralität': [0.262, 1.22], 'Jordan-Zentrum': [0.221, 1.4], 'Zufall': [0.01, 5.89]}},
        {'d': 3, 'ni': 200, 'cells': {'Rumor Centrality': [0.233, 1.27], 'Distanzzentralität': [0.233, 1.27], 'Jordan-Zentrum': [0.188, 1.55], 'Zufall': [0.005, 7.22]}},
        {'d': 4, 'ni': 10, 'cells': {'Rumor Centrality': [0.327, 0.91], 'Distanzzentralität': [0.327, 0.91], 'Jordan-Zentrum': [0.308, 0.94], 'Zufall': [0.1, 2.05]}},
        {'d': 4, 'ni': 20, 'cells': {'Rumor Centrality': [0.301, 1.0], 'Distanzzentralität': [0.301, 1.0], 'Jordan-Zentrum': [0.274, 1.1], 'Zufall': [0.05, 2.89]}},
        {'d': 4, 'ni': 50, 'cells': {'Rumor Centrality': [0.269, 1.09], 'Distanzzentralität': [0.269, 1.09], 'Jordan-Zentrum': [0.234, 1.27], 'Zufall': [0.02, 4.12]}},
        {'d': 4, 'ni': 100, 'cells': {'Rumor Centrality': [0.272, 1.1], 'Distanzzentralität': [0.272, 1.1], 'Jordan-Zentrum': [0.208, 1.34], 'Zufall': [0.01, 5.08]}},
        {'d': 4, 'ni': 200, 'cells': {'Rumor Centrality': [0.273, 1.14], 'Distanzzentralität': [0.273, 1.14], 'Jordan-Zentrum': [0.205, 1.41], 'Zufall': [0.005, 6.11]}},
        {'d': 5, 'ni': 10, 'cells': {'Rumor Centrality': [0.343, 0.88], 'Distanzzentralität': [0.343, 0.88], 'Jordan-Zentrum': [0.31, 0.93], 'Zufall': [0.1, 2.02]}},
        {'d': 5, 'ni': 20, 'cells': {'Rumor Centrality': [0.287, 1.02], 'Distanzzentralität': [0.287, 1.02], 'Jordan-Zentrum': [0.262, 1.13], 'Zufall': [0.05, 2.83]}},
        {'d': 5, 'ni': 50, 'cells': {'Rumor Centrality': [0.294, 1.01], 'Distanzzentralität': [0.294, 1.01], 'Jordan-Zentrum': [0.26, 1.16], 'Zufall': [0.02, 3.93]}},
        {'d': 5, 'ni': 100, 'cells': {'Rumor Centrality': [0.293, 1.11], 'Distanzzentralität': [0.293, 1.11], 'Jordan-Zentrum': [0.234, 1.3], 'Zufall': [0.01, 4.82]}},
        {'d': 5, 'ni': 200, 'cells': {'Rumor Centrality': [0.281, 1.08], 'Distanzzentralität': [0.281, 1.08], 'Jordan-Zentrum': [0.203, 1.36], 'Zufall': [0.005, 5.7]}}],
 'V': {'E1': 'Trefferquote der Rumor Centrality auf befallenen Bäumen: d = 2 (Linie): 0,257 bei 10, 0,062 bei 200 Befallenen; d = 3: 0,322 bei 10, 0,233 bei 200 '
             'Befallenen; d = 4: 0,327 bei 10, 0,273 bei 200 Befallenen; d = 5: 0,343 bei 10, 0,281 bei 200 Befallenen. Rumor Centrality und Distanzzentralität treffen '
             'in jeder Zelle gleich oft: auf Bäumen sind es dieselben Kandidaten.',
       'E2 0.1': 'Bei 10 % befallenen Betrieben trifft die Rumor Centrality am häufigsten in „Verbünde“ (16,1 % gegen 4,0 % bei Zufall, mittlerer Abstand 1,43 gegen '
                 '2,41). Kaum über Zufall (unter dem 1,5-Fachen) liegt kein Schätzer in: „Potenzgesetz-Netz“. Die Distanzzentralität trifft mindestens 20 % öfter als '
                 'die Rumor Centrality in: „Skalenfreies Netz (BA)“, „Ring mit Abkürzungen“, „Gitter“.',
       'E2 0.3': 'Bei 30 % befallenen Betrieben trifft die Rumor Centrality am häufigsten in „Zufallsnetz“ (4,9 % gegen 1,7 % bei Zufall, mittlerer Abstand 2,28 gegen '
                 '2,93). Kaum über Zufall (unter dem 1,5-Fachen) liegt kein Schätzer in: „Potenzgesetz-Netz“, „Skalenfreies Netz (BA)“. Die Distanzzentralität trifft '
                 'mindestens 20 % öfter als die Rumor Centrality in: „Zufallsnetz“, „Skalenfreies Netz (BA)“, „Verbünde“, „Ring mit Abkürzungen“, „Gitter“.',
       'E3 Zufallsnetz': 'In „Zufallsnetz“ fällt die Trefferquote der Rumor Centrality von 26,5 % bei 10 Befallenen auf 1,0 % bei 150; der mittlere Abstand zur Quelle '
                         'geht von 1,07 auf 2,68. Zufall trifft 10,6 % bzw. 0,9 %.',
       'E3 Skalenfreies Netz (BA)': 'In „Skalenfreies Netz (BA)“ fällt die Trefferquote der Rumor Centrality von 11,5 % bei 10 Befallenen auf 0,0 % bei 150; der '
                                    'mittlere Abstand zur Quelle geht von 1,30 auf 2,18. Zufall trifft 10,0 % bzw. 0,7 %.',
       'E3 Ring mit Abkürzungen': 'In „Ring mit Abkürzungen“ fällt die Trefferquote der Rumor Centrality von 21,1 % bei 10 Befallenen auf 0,1 % bei 150; der mittlere '
                                  'Abstand zur Quelle geht von 0,92 auf 4,22. Zufall trifft 10,0 % bzw. 0,7 %.',
       'E4 Zufallsnetz': 'Rumor Centrality in „Zufallsnetz“: bei 100 % beobachtet: Treffer 6,2 %, Abstand 2,19, im Mittel 100 % der Beobachteten in der ausgewerteten '
                         'Gruppe; bei 80 % beobachtet: Treffer 2,6 %, Abstand 2,43, im Mittel 96 % der Beobachteten in der ausgewerteten Gruppe; bei 50 % beobachtet: '
                         'Treffer 3,2 %, Abstand 2,47, im Mittel 66 % der Beobachteten in der ausgewerteten Gruppe. Das Verfahren erwartet eine zusammenhängende '
                         'befallene Menge; fehlende Betriebe lassen sie zerfallen.',
       'E4 Gitter': 'Rumor Centrality in „Gitter“: bei 100 % beobachtet: Treffer 4,1 %, Abstand 3,01, im Mittel 100 % der Beobachteten in der ausgewerteten Gruppe; bei '
                    '80 % beobachtet: Treffer 2,6 %, Abstand 2,95, im Mittel 98 % der Beobachteten in der ausgewerteten Gruppe; bei 50 % beobachtet: Treffer 2,7 %, '
                    'Abstand 3,21, im Mittel 58 % der Beobachteten in der ausgewerteten Gruppe. Das Verfahren erwartet eine zusammenhängende befallene Menge; fehlende '
                    'Betriebe lassen sie zerfallen.',
       'E5': 'Auf 8 kleinen Netzen (9 Betriebe, 12 Partnerschaften), exakt gerechnet: 4 Befallene: Bayes-Optimum 43,5 %, Rumor Centrality 29,3 %, Distanz 29,3 %, Zufall '
             '25,0 %; 5 Befallene: Bayes-Optimum 37,0 %, Rumor Centrality 23,8 %, Distanz 23,8 %, Zufall 20,0 %; 6 Befallene: Bayes-Optimum 30,8 %, Rumor Centrality '
             '19,7 %, Distanz 19,4 %, Zufall 16,7 %. Selbst das Bayes-Optimum verfehlt die Quelle meist.'},
 'E2': [{'typ': 'Zufallsnetz',
         'frac': 0.1,
         'cells': {'Rumor Centrality': [0.151, 1.51], 'Distanzzentralität': [0.175, 1.4], 'Jordan-Zentrum': [0.145, 1.58], 'Zufall': [0.041, 2.48]}},
        {'typ': 'Zufallsnetz',
         'frac': 0.3,
         'cells': {'Rumor Centrality': [0.049, 2.28], 'Distanzzentralität': [0.063, 2.07], 'Jordan-Zentrum': [0.038, 2.36], 'Zufall': [0.017, 2.93]}},
        {'typ': 'Potenzgesetz-Netz',
         'frac': 0.1,
         'cells': {'Rumor Centrality': [0.004, 1.58], 'Distanzzentralität': [0.007, 1.54], 'Jordan-Zentrum': [0.036, 1.58], 'Zufall': [0.04, 2.23]}},
        {'typ': 'Potenzgesetz-Netz',
         'frac': 0.3,
         'cells': {'Rumor Centrality': [0.001, 1.74], 'Distanzzentralität': [0.003, 1.63], 'Jordan-Zentrum': [0.01, 1.92], 'Zufall': [0.013, 2.56]}},
        {'typ': 'Skalenfreies Netz (BA)',
         'frac': 0.1,
         'cells': {'Rumor Centrality': [0.024, 1.77], 'Distanzzentralität': [0.037, 1.64], 'Jordan-Zentrum': [0.066, 1.72], 'Zufall': [0.04, 2.33]}},
        {'typ': 'Skalenfreies Netz (BA)',
         'frac': 0.3,
         'cells': {'Rumor Centrality': [0.003, 2.05], 'Distanzzentralität': [0.015, 1.86], 'Jordan-Zentrum': [0.016, 2.08], 'Zufall': [0.013, 2.66]}},
        {'typ': 'Verbünde',
         'frac': 0.1,
         'cells': {'Rumor Centrality': [0.161, 1.43], 'Distanzzentralität': [0.168, 1.3], 'Jordan-Zentrum': [0.14, 1.53], 'Zufall': [0.04, 2.41]}},
        {'typ': 'Verbünde',
         'frac': 0.3,
         'cells': {'Rumor Centrality': [0.04, 2.19], 'Distanzzentralität': [0.06, 1.86], 'Jordan-Zentrum': [0.056, 2.22], 'Zufall': [0.013, 2.98]}},
        {'typ': 'Ring mit Abkürzungen',
         'frac': 0.1,
         'cells': {'Rumor Centrality': [0.1, 1.47], 'Distanzzentralität': [0.136, 1.19], 'Jordan-Zentrum': [0.118, 1.36], 'Zufall': [0.04, 2.47]}},
        {'typ': 'Ring mit Abkürzungen',
         'frac': 0.3,
         'cells': {'Rumor Centrality': [0.026, 2.95], 'Distanzzentralität': [0.079, 2.08], 'Jordan-Zentrum': [0.063, 2.2], 'Zufall': [0.013, 3.91]}},
        {'typ': 'Gitter',
         'frac': 0.1,
         'cells': {'Rumor Centrality': [0.093, 1.58], 'Distanzzentralität': [0.127, 1.37], 'Jordan-Zentrum': [0.088, 1.61], 'Zufall': [0.038, 2.56]}},
        {'typ': 'Gitter',
         'frac': 0.3,
         'cells': {'Rumor Centrality': [0.03, 3.06], 'Distanzzentralität': [0.055, 2.45], 'Jordan-Zentrum': [0.04, 2.68], 'Zufall': [0.013, 4.38]}}],
 'E3': {'Zufallsnetz': [{'ni': 10,
                         'cells': {'Rumor Centrality': [0.265, 1.07], 'Distanzzentralität': [0.269, 1.05], 'Jordan-Zentrum': [0.255, 1.13], 'Zufall': [0.106, 1.88]}},
                        {'ni': 20,
                         'cells': {'Rumor Centrality': [0.181, 1.4], 'Distanzzentralität': [0.205, 1.32], 'Jordan-Zentrum': [0.19, 1.44], 'Zufall': [0.055, 2.34]}},
                        {'ni': 40,
                         'cells': {'Rumor Centrality': [0.128, 1.7], 'Distanzzentralität': [0.138, 1.55], 'Jordan-Zentrum': [0.105, 1.86], 'Zufall': [0.029, 2.69]}},
                        {'ni': 80,
                         'cells': {'Rumor Centrality': [0.056, 2.27], 'Distanzzentralität': [0.08, 2.03], 'Jordan-Zentrum': [0.047, 2.33], 'Zufall': [0.017, 2.91]}},
                        {'ni': 150,
                         'cells': {'Rumor Centrality': [0.01, 2.68], 'Distanzzentralität': [0.028, 2.45], 'Jordan-Zentrum': [0.018, 2.67], 'Zufall': [0.009, 3.1]}}],
        'Skalenfreies Netz (BA)': [{'ni': 10,
                                    'cells': {'Rumor Centrality': [0.115, 1.3],
                                              'Distanzzentralität': [0.108, 1.29],
                                              'Jordan-Zentrum': [0.155, 1.28],
                                              'Zufall': [0.1, 1.88]}},
                                   {'ni': 20,
                                    'cells': {'Rumor Centrality': [0.04, 1.64],
                                              'Distanzzentralität': [0.052, 1.54],
                                              'Jordan-Zentrum': [0.071, 1.63],
                                              'Zufall': [0.05, 2.22]}},
                                   {'ni': 40,
                                    'cells': {'Rumor Centrality': [0.011, 1.9],
                                              'Distanzzentralität': [0.022, 1.75],
                                              'Jordan-Zentrum': [0.027, 1.93],
                                              'Zufall': [0.025, 2.48]}},
                                   {'ni': 80,
                                    'cells': {'Rumor Centrality': [0.0, 2.04],
                                              'Distanzzentralität': [0.005, 1.88],
                                              'Jordan-Zentrum': [0.006, 2.11],
                                              'Zufall': [0.013, 2.68]}},
                                   {'ni': 150,
                                    'cells': {'Rumor Centrality': [0.0, 2.18],
                                              'Distanzzentralität': [0.004, 1.93],
                                              'Jordan-Zentrum': [0.006, 2.13],
                                              'Zufall': [0.007, 2.84]}}],
        'Ring mit Abkürzungen': [{'ni': 10,
                                  'cells': {'Rumor Centrality': [0.211, 0.92],
                                            'Distanzzentralität': [0.217, 0.9],
                                            'Jordan-Zentrum': [0.176, 1.01],
                                            'Zufall': [0.1, 1.5]}},
                                 {'ni': 20,
                                  'cells': {'Rumor Centrality': [0.096, 1.31],
                                            'Distanzzentralität': [0.133, 1.17],
                                            'Jordan-Zentrum': [0.117, 1.27],
                                            'Zufall': [0.05, 2.21]}},
                                 {'ni': 40,
                                  'cells': {'Rumor Centrality': [0.051, 2.02],
                                            'Distanzzentralität': [0.095, 1.55],
                                            'Jordan-Zentrum': [0.091, 1.64],
                                            'Zufall': [0.025, 3.1]}},
                                 {'ni': 80,
                                  'cells': {'Rumor Centrality': [0.016, 3.02],
                                            'Distanzzentralität': [0.058, 2.11],
                                            'Jordan-Zentrum': [0.058, 2.24],
                                            'Zufall': [0.013, 3.98]}},
                                 {'ni': 150,
                                  'cells': {'Rumor Centrality': [0.001, 4.22],
                                            'Distanzzentralität': [0.037, 3.2],
                                            'Jordan-Zentrum': [0.022, 3.49],
                                            'Zufall': [0.007, 4.76]}}]},
 'E4': {'Zufallsnetz': [{'share': 1.0,
                         'cells': {'Rumor Centrality': [0.062, 2.19], 'Distanzzentralität': [0.075, 1.94], 'Jordan-Zentrum': [0.042, 2.33], 'Zufall': [0.017, 2.89]},
                         'connected': 1.0},
                        {'share': 0.8,
                         'cells': {'Rumor Centrality': [0.026, 2.43], 'Distanzzentralität': [0.058, 2.19], 'Jordan-Zentrum': [0.037, 2.5], 'Zufall': [0.016, 2.9]},
                         'connected': 0.955},
                        {'share': 0.5,
                         'cells': {'Rumor Centrality': [0.032, 2.47], 'Distanzzentralität': [0.037, 2.46], 'Jordan-Zentrum': [0.03, 2.57], 'Zufall': [0.019, 2.83]},
                         'connected': 0.665}],
        'Gitter': [{'share': 1.0,
                    'cells': {'Rumor Centrality': [0.041, 3.01], 'Distanzzentralität': [0.06, 2.45], 'Jordan-Zentrum': [0.047, 2.67], 'Zufall': [0.013, 4.38]},
                    'connected': 1.0},
                   {'share': 0.8,
                    'cells': {'Rumor Centrality': [0.026, 2.95], 'Distanzzentralität': [0.045, 2.54], 'Jordan-Zentrum': [0.045, 2.7], 'Zufall': [0.014, 4.32]},
                    'connected': 0.977},
                   {'share': 0.5,
                    'cells': {'Rumor Centrality': [0.027, 3.21], 'Distanzzentralität': [0.031, 3.16], 'Jordan-Zentrum': [0.026, 3.17], 'Zufall': [0.017, 3.92]},
                    'connected': 0.578}]},
 'E5': [{'ni': 4, 'hits': {'Bayes': 0.435, 'Rumor Centrality': 0.293, 'Distanzzentralität': 0.293, 'Jordan-Zentrum': 0.293, 'Zufall': 0.25}, 'bayes_wins': 8},
        {'ni': 5, 'hits': {'Bayes': 0.37, 'Rumor Centrality': 0.238, 'Distanzzentralität': 0.238, 'Jordan-Zentrum': 0.223, 'Zufall': 0.2}, 'bayes_wins': 8},
        {'ni': 6, 'hits': {'Bayes': 0.308, 'Rumor Centrality': 0.197, 'Distanzzentralität': 0.194, 'Jordan-Zentrum': 0.183, 'Zufall': 0.167}, 'bayes_wins': 8}],
 'presets': {'Zufallsnetz: die Quelle liegt in der Mitte (Voreinstellung)': {'n': 150,
                                                                             'm': 450,
                                                                             'ni': 30,
                                                                             'obs': 30,
                                                                             'nodes': 30,
                                                                             'source': 51,
                                                                             'hits': {'Rumor Centrality': 0.0,
                                                                                      'Distanzzentralität': 0.0,
                                                                                      'Jordan-Zentrum': 0.0,
                                                                                      'Zufall': 0.033},
                                                                             'dists': {'Rumor Centrality': 1.0,
                                                                                       'Distanzzentralität': 2.0,
                                                                                       'Jordan-Zentrum': 2.0,
                                                                                       'Zufall': 2.43},
                                                                             'ranks': {'Rumor Centrality': 3, 'Distanzzentralität': 5, 'Jordan-Zentrum': 4},
                                                                             'kind': 'info',
                                                                             'verdict': '30 Betriebe sind befallen. Rumor Centrality verfehlt die Quelle, Abstand 1,0, '
                                                                                        'Rang 3; Distanzzentralität verfehlt die Quelle, Abstand 2,0, Rang 5; '
                                                                                        'Jordan-Zentrum verfehlt die Quelle, Abstand 2,0, Rang 4. Zufall käme im Mittel '
                                                                                        'auf 0,033 Treffer und Abstand 2,4.'},
             'Linie: der Satz': {'n': 100,
                                 'm': 99,
                                 'ni': 100,
                                 'obs': 100,
                                 'nodes': 100,
                                 'source': 0,
                                 'hits': {'Rumor Centrality': 0.0, 'Distanzzentralität': 0.0, 'Jordan-Zentrum': 0.0, 'Zufall': 0.01},
                                 'dists': {'Rumor Centrality': 2.5, 'Distanzzentralität': 2.5, 'Jordan-Zentrum': 2.5, 'Zufall': 25.06},
                                 'ranks': {'Rumor Centrality': 5, 'Distanzzentralität': 5, 'Jordan-Zentrum': 5},
                                 'kind': 'info',
                                 'verdict': '100 Betriebe sind befallen. Rumor Centrality verfehlt die Quelle, Abstand 2,5, Rang 5; Distanzzentralität verfehlt die '
                                            'Quelle, Abstand 2,5, Rang 5; Jordan-Zentrum verfehlt die Quelle, Abstand 2,5, Rang 5. Zufall käme im Mittel auf 0,010 '
                                            'Treffer und Abstand 25,1.'},
             'Baum mit drei Partnern: es geht': {'n': 100,
                                                 'm': 99,
                                                 'ni': 100,
                                                 'obs': 100,
                                                 'nodes': 100,
                                                 'source': 0,
                                                 'hits': {'Rumor Centrality': 0.0, 'Distanzzentralität': 0.0, 'Jordan-Zentrum': 0.0, 'Zufall': 0.01},
                                                 'dists': {'Rumor Centrality': 1.0, 'Distanzzentralität': 1.0, 'Jordan-Zentrum': 1.0, 'Zufall': 5.45},
                                                 'ranks': {'Rumor Centrality': 3, 'Distanzzentralität': 3, 'Jordan-Zentrum': 2},
                                                 'kind': 'info',
                                                 'verdict': '100 Betriebe sind befallen. Rumor Centrality verfehlt die Quelle, Abstand 1,0, Rang 3; Distanzzentralität '
                                                            'verfehlt die Quelle, Abstand 1,0, Rang 3; Jordan-Zentrum verfehlt die Quelle, Abstand 1,0, Rang 2. Zufall '
                                                            'käme im Mittel auf 0,010 Treffer und Abstand 5,5.'},
             'Skalenfrei: Treffer wie Zufall, Abstand besser': {'n': 150,
                                                                'm': 444,
                                                                'ni': 40,
                                                                'obs': 40,
                                                                'nodes': 40,
                                                                'source': 51,
                                                                'hits': {'Rumor Centrality': 0.0, 'Distanzzentralität': 0.0, 'Jordan-Zentrum': 0.0, 'Zufall': 0.025},
                                                                'dists': {'Rumor Centrality': 2.0, 'Distanzzentralität': 2.0, 'Jordan-Zentrum': 2.0, 'Zufall': 2.38},
                                                                'ranks': {'Rumor Centrality': 37, 'Distanzzentralität': 26, 'Jordan-Zentrum': 27},
                                                                'kind': 'info',
                                                                'verdict': '40 Betriebe sind befallen. Rumor Centrality verfehlt die Quelle, Abstand 2,0, Rang 37; '
                                                                           'Distanzzentralität verfehlt die Quelle, Abstand 2,0, Rang 26; Jordan-Zentrum verfehlt die '
                                                                           'Quelle, Abstand 2,0, Rang 27. Zufall käme im Mittel auf 0,025 Treffer und Abstand 2,4.'},
             'Ring: Distanz schlägt Rumor Centrality': {'n': 150,
                                                        'm': 471,
                                                        'ni': 50,
                                                        'obs': 50,
                                                        'nodes': 50,
                                                        'source': 51,
                                                        'hits': {'Rumor Centrality': 0.0, 'Distanzzentralität': 1.0, 'Jordan-Zentrum': 1.0, 'Zufall': 0.02},
                                                        'dists': {'Rumor Centrality': 2.0, 'Distanzzentralität': 0.0, 'Jordan-Zentrum': 0.0, 'Zufall': 3.02},
                                                        'ranks': {'Rumor Centrality': 5, 'Distanzzentralität': 1, 'Jordan-Zentrum': 1},
                                                        'kind': 'success',
                                                        'verdict': '50 Betriebe sind befallen. Rumor Centrality verfehlt die Quelle, Abstand 2,0, Rang 5; '
                                                                   'Distanzzentralität trifft die Quelle, Abstand 0,0, Rang 1; Jordan-Zentrum trifft die Quelle, Abstand '
                                                                   '0,0, Rang 1. Zufall käme im Mittel auf 0,020 Treffer und Abstand 3,0.'},
             'Fast alles befallen: mehr hilft nicht': {'n': 150,
                                                       'm': 450,
                                                       'ni': 120,
                                                       'obs': 120,
                                                       'nodes': 120,
                                                       'source': 51,
                                                       'hits': {'Rumor Centrality': 0.0, 'Distanzzentralität': 0.0, 'Jordan-Zentrum': 0.017, 'Zufall': 0.008},
                                                       'dists': {'Rumor Centrality': 2.0, 'Distanzzentralität': 2.0, 'Jordan-Zentrum': 2.85, 'Zufall': 2.96},
                                                       'ranks': {'Rumor Centrality': 53, 'Distanzzentralität': 76, 'Jordan-Zentrum': 1},
                                                       'kind': 'success',
                                                       'verdict': '120 Betriebe sind befallen. Rumor Centrality verfehlt die Quelle, Abstand 2,0, Rang 53; '
                                                                  'Distanzzentralität verfehlt die Quelle, Abstand 2,0, Rang 76; Jordan-Zentrum hat die Quelle unter 59 '
                                                                  'gleich guten Kandidaten (Treffer 0,02), Abstand 2,8, Rang 1. Zufall käme im Mittel auf 0,008 Treffer '
                                                                  'und Abstand 3,0.'}}}


@lru_cache(maxsize=None)
def _e1():
    return ev.tree_rows(runs=TREE_RUNS)


@lru_cache(maxsize=None)
def _e2():
    return ev.network_rows(250, 6, seeds=SW, runs=RUNS)


@lru_cache(maxsize=None)
def _e3(typ):
    return ev.size_rows(typ, 250, 6, seeds=SW, runs=RUNS)


@lru_cache(maxsize=None)
def _e4(typ):
    return ev.observation_rows(typ, 250, 6, seeds=SW, runs=RUNS)


@lru_cache(maxsize=None)
def _e5():
    return ev.exact_rows()


def _hit(x):
    return pytest.approx(x, abs=0.0006)


def _dist(x):
    return pytest.approx(x, abs=0.006)


def _check_cells(got, expected):
    for e, (hit, dist) in expected.items():
        assert got[e][0] == _hit(hit) and got[e][1] == _dist(dist), e


# --- E1: reguläre Bäume ------------------------------------------------------------------------------------------------------------------------------


def test_hit_rates_and_distances_on_regular_trees():
    rows = _e1()
    for exp in EXPECT["E1"]:
        _check_cells(ev.tree_cell(rows, exp["d"], exp["ni"]), exp["cells"])
        assert ev.tree_cell(rows, exp["d"], exp["ni"])[RAND][0] == pytest.approx(1.0 / exp["ni"])


def test_the_line_fails_and_branching_trees_keep_a_constant_hit_rate():
    rows = _e1()
    line = [ev.tree_cell(rows, 2, ni)[RC][0] for ni in C.TREE_NS]
    assert line == sorted(line, reverse=True) and line[-1] < 0.07 and line[0] > 0.25
    for d in (3, 4, 5):
        assert ev.tree_cell(rows, d, 200)[RC][0] > 0.18 and ev.tree_cell(rows, d, 50)[RC][0] > 0.25 - 0.02 * (d == 3)
    assert ev.tree_cell(rows, 2, 200)[RC][0] * 3 < ev.tree_cell(rows, 3, 200)[RC][0]


def test_rumor_centrality_equals_distance_centrality_on_trees_and_the_jordan_center_is_weaker_with_branching():
    rows = _e1()
    assert all(r["cells"][RC][0] == r["cells"][DC][0] and r["cells"][RC][1] == r["cells"][DC][1] for r in rows)
    for d in (3, 4, 5):
        assert ev.tree_cell(rows, d, 200)[J][0] < 0.85 * ev.tree_cell(rows, d, 200)[RC][0]
    assert all(ev.tree_cell(rows, 2, ni)[J][0] == ev.tree_cell(rows, 2, ni)[RC][0] for ni in C.TREE_NS)       # auf der Linie liegen alle Maße gleichauf
    assert ev.verdict_trees(rows)[1] == EXPECT["V"]["E1"]


# --- E2: Netztypen -------------------------------------------------------------------------------------------------------------------------------


def test_hit_rates_and_distances_per_network_type_and_infected_share():
    rows = {(r["typ"], r["frac"]): r for r in _e2()}
    for exp in EXPECT["E2"]:
        _check_cells(rows[(exp["typ"], exp["frac"])]["cells"], exp["cells"])
    for frac in C.NET_FRACS:
        assert ev.verdict_networks(_e2(), frac)[1] == EXPECT["V"][f"E2 {frac}"]


def test_in_heavy_tailed_networks_the_hit_rate_is_no_better_than_random_but_the_distance_is_smaller():
    rows = {(r["typ"], r["frac"]): r["cells"] for r in _e2()}
    for typ in (C.T_POWER, C.T_BA):
        cells = rows[(typ, 0.3)]
        assert max(cells[e][0] for e in (RC, DC, J)) < 1.5 * cells[RAND][0]
        assert all(cells[e][1] < 0.85 * cells[RAND][1] for e in (RC, DC, J))
    assert rows[(C.T_POWER, 0.1)][RC][0] < 0.2 * rows[(C.T_POWER, 0.1)][RAND][0]                                 # die Rumor Centrality liegt im Potenzgesetz-Netz weit unter Zufall (sie wählt Hubs)


def test_the_distance_centrality_is_at_least_as_good_as_the_rumor_centrality_in_every_network_type_at_thirty_percent():
    rows = {(r["typ"], r["frac"]): r["cells"] for r in _e2()}
    for typ in C.GENERATED_TYPES:
        assert rows[(typ, 0.3)][DC][0] >= rows[(typ, 0.3)][RC][0] and rows[(typ, 0.3)][DC][1] <= rows[(typ, 0.3)][RC][1] + 1e-9, typ
    assert rows[(C.T_RING, 0.3)][DC][0] > 2.5 * rows[(C.T_RING, 0.3)][RC][0]


def test_every_estimator_beats_random_in_distance_in_every_network_type():
    for r in _e2():
        assert all(r["cells"][e][1] < r["cells"][RAND][1] for e in (RC, DC, J)), (r["typ"], r["frac"])


# --- E3: Ausbreitungsgröße -------------------------------------------------------------------------------------------------------------------------


def test_more_infected_do_not_help():
    for typ, exp in EXPECT["E3"].items():
        rows = _e3(typ)
        assert [r["ni"] for r in rows] == [e["ni"] for e in exp]
        for r, e in zip(rows, exp):
            _check_cells(r["cells"], e["cells"])
        hits = [r["cells"][RC][0] for r in rows]
        dists = [r["cells"][RC][1] for r in rows]
        assert hits == sorted(hits, reverse=True) and dists == sorted(dists)
        assert ev.verdict_sizes(rows)[1] == EXPECT["V"]["E3 " + typ]


# --- E4: Unvollständigkeit -------------------------------------------------------------------------------------------------------------------------


def test_incomplete_observation_breaks_the_infected_set_apart():
    for typ, exp in EXPECT["E4"].items():
        rows = _e4(typ)
        for r, e in zip(rows, exp):
            assert r["share"] == e["share"] and r["connected"] == pytest.approx(e["connected"], abs=0.0006)
            _check_cells(r["cells"], e["cells"])
        assert rows[0]["connected"] == 1.0 and rows[1]["connected"] < 1.0 and rows[2]["connected"] < 0.7
        assert rows[0]["cells"][RC][0] > rows[1]["cells"][RC][0]
        assert ev.verdict_observation(rows)[1] == EXPECT["V"]["E4 " + typ]


# --- E5: exakt auf kleinen Netzen ----------------------------------------------------------------------------------------------------------------------


def test_exact_hit_probabilities_on_small_networks():
    rows = _e5()
    for r, e in zip(rows, EXPECT["E5"]):
        assert r["ni"] == e["ni"] and r["bayes_wins"] == e["bayes_wins"] == 8
        for k, v in e["hits"].items():
            assert r["hits"][k] == _hit(v), (r["ni"], k)
        assert r["hits"]["Zufall"] == pytest.approx(1.0 / r["ni"]) and r["hits"]["Bayes"] > r["hits"][RC] > r["hits"]["Zufall"]
    assert ev.verdict_exact(rows)[1] == EXPECT["V"]["E5"]


# --- Presets ---------------------------------------------------------------------------------------------------------------------------------------


def test_presets_numbers():
    for name, exp in EXPECT["presets"].items():
        an = ev.analyse(ev.Settings(**C.PRESETS[name]))
        assert (an.inst.n, an.inst.m, an.ni, len(an.seen), len(an.nodes), an.source) == (exp["n"], exp["m"], exp["ni"], exp["obs"], exp["nodes"], exp["source"]), name
        for e in C.ESTIMATORS:
            assert an.hits[e] == _hit(exp["hits"][e]) and an.dists[e] == _dist(exp["dists"][e]), (name, e)
        assert an.ranks == exp["ranks"] and ev.verdict_main(an) == (exp["kind"], exp["verdict"]), name


def test_the_shortest_run_helpers_agree_with_the_oracle_identity():
    sub, _ = A.grow_regular_tree(3, 12, ev._rng(1, 1))
    rc, dc, ecc = A.scores(sub)
    assert A.argmax_set(rc) == A.argmax_set(dc, -1)
