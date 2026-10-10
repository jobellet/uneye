# data/ — benchmark datasets (CSV, one row per trial)
`dataset1..4/<dsN>_<rate>hz_{X,Y,Labels}_set{A,B}.csv` (positions in degrees, labels 1 = saccade; set A = train, set B = test; dataset4 also `Subject_nb_set{A,B}.csv`, 10 subjects), `Synthetic/`, `Andersson/` (5 classes, GPL-3.0, NOT in git).
Rates: d1, d2, d4 1 kHz; d3, Andersson 500 Hz. Load through `foundation/data.py` (handles rates, Andersson classes, the dataset-4 set B row mismatch); do not parse the CSVs by hand. Unlabeled extra recordings are in `../archive/` (gitignored; `docs/ARCHIVE_DATA.md`).
