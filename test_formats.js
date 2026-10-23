// node test_formats.js FIXTURE_DIR [OUT_DIR]
// FIXTURE_DIR holds all_trials.{csv,mat,npz,h5} and all_trials_compressed.mat (made by mocol_saccades/make_combined_formats.py, or any file set of the same
// 213 trials). Reads every format with formats.js, checks the trials against each other, writes the labeled files (npz, mat, h5) to OUT_DIR, reads them
// back, and leaves them there for a check from Python (mocol_saccades/check_written_formats.py).
const fs = require('fs'), path = require('path');
const F = require('./formats.js');
const dir = process.argv[2], outDir = process.argv[3] || fs.mkdtempSync('/tmp/uneye_formats_');
if (!dir) { console.log('usage: node test_formats.js FIXTURE_DIR [OUT_DIR]'); process.exit(2); }
const buf = f => { const b = fs.readFileSync(path.join(dir, f)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
let failed = 0;
const ok = (c, msg) => { console.log((c ? 'ok   ' : 'FAIL ') + msg); if (!c) failed++; };

function csvTrials(text) {
  const rows = text.trim().split(/\r?\n/).map(l => l.split(',').map(Number)), out = [];
  for (let i = 0; i + 1 < rows.length; i += 2) out.push({ x: rows[i], y: rows[i + 1] });
  return out;
}
const same = (a, b, tol = 2e-5) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > tol) return false; return true; };

(async () => {
  const ref = csvTrials(fs.readFileSync(path.join(dir, 'all_trials.csv'), 'utf8'));
  ok(ref.length === 213, 'csv: ' + ref.length + ' trials of different lengths ' + Math.min(...ref.map(t => t.x.length)) + '-' + Math.max(...ref.map(t => t.x.length)));
  // HDF5 needs h5wasm: npm install h5wasm somewhere and give the path of dist/node/hdf5_hl.js in H5WASM (or install it next to this file)
  const hm = await import(process.env.H5WASM || 'h5wasm/node'); const h5mod = hm.default && hm.default.ready ? hm.default : hm;
  const results = {};
  for (const name of ['all_trials.npz', 'all_trials.mat', 'all_trials_compressed.mat', 'all_trials.h5']) {
    let r;
    try { r = await F.readTrialsFile(name, buf(name), h5mod); } catch (e) { ok(false, name + ': ' + e.message); continue; }
    results[name] = r;
    let bad = 0; r.trials.forEach((t, k) => { if (!same(t.x, ref[k].x) || !same(t.y, ref[k].y)) bad++; });
    ok(r.trials.length === ref.length && bad === 0, `${name}: ${r.trials.length} trials, every x and y equal to the csv (${bad} different), fs ${r.fs}, names ${r.trials[0].name} ... ${r.trials[r.trials.length - 1].name}`);
    ok(r.trials.every(t => t.valid.every(v => v === 1)), name + ': no invalid sample');
  }
  // the mocol export (eye + trial_offset), if a file is given
  if (process.env.MOCOL_H5) {
    const b = fs.readFileSync(process.env.MOCOL_H5), r = await F.readTrialsFile('s.h5', b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), h5mod);
    ok(r.trials.length > 100, `mocol export ${path.basename(process.env.MOCOL_H5)}: ${r.trials.length} trials, first length ${r.trials[0].x.length}`);
  }
  // writers: labels with a pattern (0 / 1 / 2, -1 unreviewed), then read back
  const src = results['all_trials.npz'].trials;
  const labels = src.map((t, k) => Int8Array.from({ length: t.x.length }, (_, i) => (k % 5 === 0 ? -1 : (i % 97 < 5 ? 1 : (i % 997 < 30 ? 2 : 0)))));
  const names = src.map(t => t.name);
  fs.mkdirSync(outDir, { recursive: true });
  const outs = { 'labeled.npz': F.writeNpzFile(src, labels, names, 1000), 'labeled.mat': F.writeMatFile(src, labels, names, 1000), 'labeled.h5': await F.writeH5File(src, labels, names, 1000, h5mod) };
  for (const [n, bytes] of Object.entries(outs)) {
    fs.writeFileSync(path.join(outDir, n), bytes);
    const back = await F.readTrialsFile(n, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), h5mod);
    let bad = 0, badL = 0; back.trials.forEach((t, k) => { if (!same(t.x, src[k].x, 1e-4) || !same(t.y, src[k].y, 1e-4)) bad++; if (!back.labels || !same(back.labels[k], labels[k], 0)) badL++; });
    ok(back.trials.length === src.length && bad === 0 && badL === 0 && back.trials[7].name === names[7] && back.fs === 1000, `${n}: written ${(bytes.length / 2 ** 20).toFixed(1)} MiB, read back: ${back.trials.length} trials, positions equal, labels equal (${badL} different), names, fs`);
  }
  console.log(failed ? failed + ' check(s) FAILED' : 'all checks passed', '- written files in', outDir);
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
