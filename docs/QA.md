# Wahrscheinliche Fragen / likely questions

Kurze Antworten, Deutsch zuerst, dann Englisch. Zahlen: siehe `docs/safety/hazards.md`, `docs/EXPLAIN.md`.

**1. Warum ein Wächter, wenn das Netz gut ist? / Why a guard if the network is good?**
DE: Ein Netz ist nur auf Daten geprüft, die es kennt; bei eingefrorenem Signal meldete es allein 77 unmögliche Ereignisse, mit Wächter 0.
EN: A network is only validated on data like its training data; on a frozen signal it alone emitted 77 impossible events, with the guard 0.

**2. Warum kann das Netz den Wächter nicht überstimmen? / Why can the network not override the guard?**
DE: Der Wächter liegt hinter der Engine und prüft jede Ausgabe mit einfachen Regeln; im Zustand SAFE ist die Ausgabe immer „ungültig“, egal was das Netz sagt.
EN: The guard sits after the engine and checks every output with simple rules; in SAFE the output is always "invalid", whatever the network says.

**3. Was heißt „kein Heap“ und warum ist das wichtig? / What does "zero heap" mean and why does it matter?**
DE: Nach dem Start reserviert die Engine keinen Speicher mehr – gemessen: 0 Allokationen in 100 000 Proben; so gibt es keine Speicherfehler und keine unvorhersehbaren Wartezeiten durch den Allocator.
EN: After start-up the engine never allocates memory – measured: 0 allocations in 100,000 samples – so no out-of-memory failure and no allocator pauses at run time.

**4. float32 oder float64? / float32 or float64?**
DE: Gewichte und Netz rechnen in float32 wie in PyTorch; Positionen und Zeit in double. Die Parität mit Python liegt bei 1e-4 auf Wahrscheinlichkeiten; ein schichtweiser float64-Vergleich (Schritt 2) ist noch offen.
EN: Weights and network run in float32, as in PyTorch; positions and time in double. Parity with Python is 1e-4 on probabilities; a layer-by-layer float64 comparison (Step 2) is still open.

**5. Ist das echtzeitfähig? / Is it real-time?**
DE: Median 78 µs pro Probe bei 1 ms Budget, aber 7 von einer Million Proben über 1 ms auf dem Laptop; eine Garantie braucht ein Echtzeit-Betriebssystem und eine Messung auf dem Zielsystem.
EN: Median 78 µs per sample against a 1 ms budget, but 7 in a million samples over 1 ms on a laptop; a guarantee needs a real-time OS and measurement on the target.

**6. Was passiert, wenn die Berechnung zu spät kommt? / What if a result is late?**
DE: Der Aufrufer misst die Zeit und gibt sie dem Wächter; eine verspätete Probe wird als ungültig ausgegeben, wiederholte Verspätungen führen zu DEGRADED oder SAFE.
EN: The caller measures the time and passes it in; a late sample is output as invalid, repeated lateness leads to DEGRADED or SAFE.

**7. Was ist SOUP? / What is SOUP?**
DE: Fremdsoftware unbekannter Herkunft, hier ONNX Runtime; deshalb gibt es eine eigene, kleine C++-Engine, deren Verhalten wir vollständig testen können.
EN: Software of unknown provenance, here ONNX Runtime; that is why a small own C++ engine exists whose behaviour we can test completely.

**8. Wo versagt der Wächter? / Where does the guard fail?**
DE: Bei mäßigem Rauschen, schwachen Störimpulsen, ×10-Skalierung und 500-Hz-Daten mit 1-kHz-Zeitstempeln; die Schwellen habe ich danach bewusst nicht an die Testdaten angepasst.
EN: Moderate noise, small interference bursts, ×10 scaling, and 500 Hz data with 1 kHz timestamps; I deliberately did not retune the thresholds on the test data.

**9. Was kostet der Wächter? / What does the guard cost?**
DE: Empfindlichkeit: auf sauberen Daten sinkt der Anteil erkannter Sakkaden von 0,94 auf 0,89, und nach einem Ausfall bleibt er bis zu 1 s vorsichtig.
EN: Sensitivity: on clean data the share of detected saccades drops from 0.94 to 0.89, and after a fault it stays cautious for up to 1 s.

**10. Was wurde NICHT gemacht? / What was NOT done?**
DE: Keine Zertifizierung, kein WCET, keine Bit-Gleichheit über Maschinen, kein schichtweiser Python/C++-Vergleich, keine CI-Pipeline (Stand: siehe ROADMAP), keine klinische Bewertung.
EN: No certification, no WCET, no bit-equality across machines, no layer-by-layer Python/C++ comparison, no CI pipeline yet (see ROADMAP), no clinical evaluation.
