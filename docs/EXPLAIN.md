# Was ich zu jeder Abbildung sage (3 Sätze + die ehrliche Grenze)

Abbildungen: `docs/slides/figs/` (PNG + SVG), erzeugt mit `python docs/slides/make_figures.py`. Alle Zahlen stammen aus
Skripten, die gelaufen sind (MacBook Air M1, clang 22.1.8, -O3, feste Seeds). KI-unterstützt erstellt (Claude Code).

## Abb. 5 – Architektur (zuerst zeigen)
1. Das Modell wird in Python trainiert und als Gewichte exportiert; eine eigene C++-Engine rechnet es Probe für Probe nach.
2. Hinter der Engine sitzt ein einfacher, deterministischer Sicherheitswächter, den das Netz nicht überstimmen kann.
3. Jede Stufe hat einen automatischen Test: Parität mit Python, keine Speicherallokation, Wächter-Regeln, Fehlerinjektion.
- **Grenze:** Das ist eine Demonstration der Methode, keine Zertifizierung nach IEC 62304.

## Abb. 2 – Zeit pro Probe
1. Bei 1 kHz hat jede Probe 1 ms Budget; Engine und Wächter brauchen im Median 78 µs.
2. Jede zehnte Probe rechnet zusätzlich das große Fensternetz mit, deshalb die Spitzen bei etwa 270 µs.
3. Von einer Million Proben lagen 7 über 1 ms – das kommt vom Betriebssystem, nicht vom Algorithmus, der pro Probe konstante Arbeit macht.
- **Grenze:** Ein Laptop ist nicht das Zielsystem; das gemessene Maximum ist keine garantierte Obergrenze (kein WCET).

## Abb. 3 – Fehlerinjektion
1. Ich störe echte Testdaten gezielt – Rauschen, Störimpulse, Ausfälle, eingefrorenes Signal, falsche Abtastrate – und vergleiche das Netz allein mit Engine plus Wächter.
2. „Gefährlich“ heißt: ein gemeldetes Ereignis, das auf dem ungestörten Signal physiologisch unmöglich ist.
3. Mit Wächter gibt es in 16 von 24 getesteten Fällen null gefährliche Ausgaben (in 3 davon war auch das Netz allein bei null; die Abbildung zeigt 13 Fälle), etwa beim eingefrorenen Signal (77 → 0) oder einem hängenden Netz (243 → 0).
- **Grenze:** Bei mäßigem Rauschen, schwachen Störimpulsen, ×10-Skalierung und 500-Hz-Daten mit 1-kHz-Zeitstempeln bleibt der Wächter wirkungslos oder nur teilweise wirksam – das zeige ich bewusst, und die Schwellen wurden danach nicht nachjustiert.

## Abb. 4 – Beispiel eingefrorenes Signal
1. Oben friert das Eingangssignal 200 ms lang ein, wie bei einem hängenden Kameratreiber.
2. Der Wächter erkennt nach 50 ms, dass beide Achsen exakt konstant sind, und schaltet auf SAFE: keine Ausgabe statt einer Vermutung.
3. Zurück geht es nur stufenweise, SAFE → DEGRADED → NORMAL, jeweils nach 0,5 s ohne Auffälligkeit.
- **Grenze:** Sicherheit kostet hier Empfindlichkeit – die echte Sakkade bei 780 ms wird in dieser Zeit ebenfalls nicht gemeldet.

## Abb. 1 – Python vs. C++
1. Dasselbe kausale Netz läuft in PyTorch (float32 und float64) und in meiner C++-Engine, auf 205 Sequenzen mit je 1000 Proben, echte Daten und Stresseingaben.
2. Die C++-Ausgabe weicht höchstens 6·10⁻⁶ von der float64-Referenz ab – so viel wie PyTorch in float32 selbst (3,5·10⁻⁶); kein einziges Label ist verschieden.
3. Ein automatischer Test prüft das bei jedem Build mit einer Toleranz von 5·10⁻⁵, abgeleitet aus der Messung, und schlägt bei einem falschen Modell sofort fehl.
- **Grenze:** Bitgleichheit gilt nur für denselben Compiler mit denselben Optionen; ein Vergleich Schicht für Schicht fehlt noch.
