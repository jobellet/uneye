# Web labeler: file formats (trials of different durations)

`index.html` + `app.js` + `formats.js` (2026-10-23). Open a file with "Open...", choose the sampling rate and the units in the dialog (a file that declares `fs` overrides the rate).

**Read:** `.csv` / `.txt` (two columns x y, or alternating rows X, Y per trial: the rows may have different lengths), `.mat` (MATLAB v5, compressed or not; v7.3 = HDF5),
`.npz` (numpy), `.h5` / `.hdf5`. Layouts, in this order (names are not case sensitive):

| layout | |
|---|---|
| `trials` | a cell array (MAT) / datasets `trial_0`, `trial_1`, ... (npz, h5) of (T x 2) or (2 x T) matrices [x y] |
| `X`, `Y` | two cell arrays of vectors, or two 2-D arrays trials x time (rows padded with NaN: the padding is cut) |
| `xy` + `trial_offset` (or `trial_lengths`) | (sum of T, 2): trial k = rows `trial_offset[k]` to `trial_offset[k+1]` - what the page writes |
| `eye` + `trial_offset` | the eye traces of the mocol export (`mocol_aMUA_eye_sessionNN.h5`); the huge `amua` dataset is not read |
| `x`, `y` | one trial |
| optional | `names` (one string per trial, shown next to the trial number), `fs` (Hz), `labels` (same layout as the positions: int8, 0 fixation, 1 saccade, 2 blink, -1 not reviewed; a cell array in MAT) |

**Write:** the "Export" format selector: CSV (L.csv, X.csv, Y.csv as before) or one file `unEye_labeled.npz` / `.mat` / `.h5` with the raw positions, `labels`, `names`, `fs`
(+ `uneye_config.json` always). A file written by the page can be opened again: the labels come back (reviewed ranges, events).

HDF5 uses h5wasm 0.10.3 (5 MB), fetched from cdn.jsdelivr.net the first time it is needed. Python objects (pickles, `object` arrays) cannot be read: save plain numeric arrays.

**Test:** `H5WASM=file:///.../node_modules/h5wasm/dist/node/hdf5_hl.js node test_formats.js FIXTURE_DIR OUT_DIR` (fixtures: `mocol_saccades/make_combined_formats.py`); checked in Chrome (headless) with the
213 mocol trials of 3.5-7.3 s: every format loads, exports write npz / mat / h5, the labeled h5 is read back with its labels; the written files were read back with numpy, scipy and h5py.
