// Tests of import.js (text import: delimiter / header detection, ARFF, EyeLink .asc, time -> rate, resampling, trial splitting).
// Always runs on the trimmed real files in test_data/import (taken from public datasets: GazeBase, GazeBaseVR, GazeCom, 360EM, NUIG, EMTeC);
// when the full datasets folder exists (UNEYE_DATASETS, default ~/Desktop/Desktop/eye_tracking_datasets/datasets) it also reads whole files.
// Usage: node test_import.js
const fs = require('fs'), path = require('path'), os = require('os'), cp = require('child_process');
const I = require('./import.js');

let failed = 0;
const ok = (c, msg) => { if (c) console.log('ok   ' + msg); else { failed++; console.log('FAIL ' + msg); } };
const near = (a, b, tol) => Math.abs(a - b) <= tol;
const FIX = path.join(__dirname, 'test_data', 'import');
const read = (f) => fs.readFileSync(path.join(FIX, f), 'utf8');

async function load(f) {
  const t = await I.parseString(read(f), f);
  const stats = I.describeColumns(t);
  return { t, stats, s: I.suggest(t, stats), name: (i) => (i >= 0 ? t.cols[i].name : null) };
}
function mapping(t, s, extra) {
  const fs_ = s.time && s.time.rate ? s.time.rate : 1000;
  return Object.assign({ x: s.x, y: s.y, t: s.t, unit: s.time ? s.time.unit : 'samples', fs: fs_, valid: s.valid, split: { mode: 'none' }, chunkSec: 0, resample: s.time ? !s.time.regular : false }, extra || {});
}
const count = (a, f) => { let n = 0; for (let i = 0; i < a.length; i++) if (f(a[i])) n++; return n; };

(async () => {
  // ------------------------------------------------------------ real files, trimmed
  {
    const r = await load('gazebase_hss.csv');
    ok(r.t.kind === 'delimited' && r.t.info.header && r.t.info.delimiter === ',', 'GazeBase: comma separated with a header');
    ok(r.name(r.s.x) === 'x' && r.name(r.s.y) === 'y', 'GazeBase: X, Y found (not the targets xT, yT)');
    ok(r.s.valid && r.name(r.s.valid.col) === 'val' && r.s.valid.op === '=' && r.s.valid.val === 0, 'GazeBase: validity flag "val = 0" proposed');
    ok(r.s.time && r.s.time.unit === 'samples', 'GazeBase: "n" is a sample index, the rate is left to the user');
    const b = I.buildTrials(r.t, mapping(r.t, r.s, { fs: 1000 }));
    ok(b.trials.length === 1 && b.trials[0].x.length === r.t.nRows - 0 || b.trials.length === 1, 'GazeBase: one trial');
    const nBad = count(r.t.cols[r.t.cols.findIndex(c => c.name === 'val')].data, v => v !== 0);
    ok(count(b.trials[0].valid, v => v === 0) >= nBad, 'GazeBase: samples with val != 0 are invalid (' + nBad + ')');
  }
  {
    const r = await load('gazebasevr.csv');
    ok(r.name(r.s.x) === 'x' && r.name(r.s.y) === 'y', 'GazeBaseVR: cyclopean x, y preferred over lx / rx');
    ok(r.t.cols.some(c => c.name === 'x (mean L,R)'), 'GazeBaseVR: a mean-of-eyes column is offered');
    ok(r.name(r.s.t) === 'n' && r.s.time.unit === 'ms' && r.s.time.rate === 250 && r.s.time.regular, 'GazeBaseVR: "n" read as milliseconds, 250 Hz, regular');
    const b = I.buildTrials(r.t, mapping(r.t, r.s));
    ok(b.trials.length === 1 && b.trials[0].x.length === r.t.nRows && !b.trials[0].resampled, 'GazeBaseVR: kept sample by sample');
  }
  {
    const r = await load('gazecom.arff');
    ok(r.t.kind === 'arff' && r.t.cols.length === 7, 'GazeCom: ARFF read (7 attributes)');
    ok(r.name(r.s.x) === 'x' && r.s.time.unit === 'us' && r.s.time.rate === 250, 'GazeCom: time in microseconds, 250 Hz');
    ok(r.s.time.regular === false, 'GazeCom: jittered / missing time stamps are flagged as irregular');
    ok(r.s.valid && r.name(r.s.valid.col) === 'confidence', 'GazeCom: confidence column proposed as validity');
    const b = I.buildTrials(r.t, mapping(r.t, r.s));
    const tr = b.trials[0], T = r.t.cols[r.s.t].data;
    const expected = Math.floor((T[r.t.nRows - 1] - T[0]) * 1e-6 * 250 + 1e-9) + 1;
    ok(tr.resampled && tr.x.length === expected, 'GazeCom: resampled onto the 250 Hz grid (' + tr.x.length + ' samples)');
  }
  {
    const r = await load('360em.arff');
    ok(r.s.time.unit === 'us' && r.s.time.rate === 120 && r.s.time.regular, '360EM: ms-rounded stamps in us still give 120 Hz, regular');
    ok(r.name(r.s.x) === 'x' && r.name(r.s.y) === 'y', '360EM: X, Y found (not the head angles)');
    const lab = r.t.cols.find(c => c.name === 'handlabeller_1_pl');
    ok(lab && lab.labels && lab.labels.includes('saccade'), '360EM: nominal attribute kept with its labels');
  }
  {
    const r = await load('nuig_tablet.csv');
    ok(r.name(r.s.t) === 'TIM REL' && r.s.time.unit === 'ms' && r.s.time.rate === 30, 'NUIG: "TIM REL" in ms, about 30 Hz');
    ok(r.name(r.s.x) !== null && !/GT/i.test(r.name(r.s.x)) && !/GT/i.test(r.name(r.s.y)), 'NUIG: ground-truth target columns (GTX, GTY) not proposed as the gaze: ' + r.name(r.s.x) + ', ' + r.name(r.s.y));
  }
  {
    const r = await load('emtec_samples.csv');
    ok(r.t.info.delimiter === '\t', 'EMTeC csv: tab separated');
    ok(r.name(r.s.x) === 'x' && r.name(r.s.t) === 'time' && r.s.time.rate === 2000, 'EMTeC csv: duplicated millisecond stamps give 2000 Hz');
    ok(r.name(r.s.trial) === 'TRIAL_ID', 'EMTeC csv: trial column TRIAL_ID proposed');
    const b = I.buildTrials(r.t, mapping(r.t, r.s, { split: { mode: 'column', col: r.s.trial } }));
    ok(b.trials.length === 2 && b.trials[0].x.length === 1500 && b.trials[1].x.length === 1500, 'EMTeC csv: split into the 2 trials (1500 + 1500 samples)');
    const g = I.buildTrials(r.t, mapping(r.t, r.s, { split: { mode: 'gap', gapSec: 1 } }));
    ok(g.trials.length >= 1, 'EMTeC csv: splitting at time gaps works too (' + g.trials.length + ')');
    const c = I.buildTrials(r.t, mapping(r.t, r.s, { split: { mode: 'column', col: r.s.trial }, chunkSec: 0.5 }));
    ok(c.trials.length > 2 && c.trials.every(t => t.x.length <= 1.25 * 1000 + 1), 'EMTeC csv: 0.5 s chunks of at most 1250 samples (' + c.trials.length + ' parts)');
  }
  {
    const r = await load('emtec.asc');
    ok(r.t.kind === 'asc' && r.t.info.rate === 2000 && r.t.info.eyes === 1 && r.t.info.blocks === 3, 'EyeLink asc: 3 recording blocks, one eye, 2000 Hz');
    ok(r.t.cols.map(c => c.name).join() === 'time,x,y,pupil,block', 'EyeLink asc: columns time, x, y, pupil, block');
    ok(r.name(r.s.x) === 'x' && r.name(r.s.trial) === 'block' && r.s.time.rate === 2000, 'EyeLink asc: block proposed as the trial column');
    const nan = count(r.t.cols[1].data, Number.isNaN);
    ok(nan > 0 && r.t.nRows === 4500, 'EyeLink asc: "." samples read as NaN (' + nan + ' of ' + r.t.nRows + ')');
    const b = I.buildTrials(r.t, mapping(r.t, r.s, { split: { mode: 'column', col: r.s.trial } }));
    ok(b.trials.length === 3 && b.trials.every(t => t.x.length === 1500), 'EyeLink asc: 3 trials of 1500 samples');
    ok(count(b.trials[0].valid, v => v === 0) === count(r.t.cols[1].data.subarray(0, 1500), Number.isNaN), 'EyeLink asc: blink samples are invalid');
    ok(r.t.info.messages.length > 0 && r.t.info.events.fixations >= 0, 'EyeLink asc: messages listed');
  }

  // ------------------------------------------------------------ plain files keep the old fast path
  {
    const two = fs.readFileSync(path.join(__dirname, 'test_data', 'sample_two_column_data.txt'), 'utf8');
    const one = fs.readFileSync(path.join(__dirname, 'test_data', 'sample_one_column_data.txt'), 'utf8');
    ok(I.sniffHead(two, 'a.txt').plain && I.sniffHead(one, 'b.txt').plain, 'plain numeric 1- and 2-column files are left to the old reader');
    ok(!I.sniffHead(read('gazebase_hss.csv'), 'x.csv').plain && !I.sniffHead(read('emtec.asc'), 'x.asc').plain && !I.sniffHead(read('gazecom.arff'), 'x.arff').plain, 'files with a header, ARFF and .asc go to the explorer');
  }

  // ------------------------------------------------------------ synthetic edge cases
  {
    const txt = 'Zeit;PosX;PosY\n0,000;1,5;2,5\n0,004;1,6;2,4\n0,008;1,7;2,3\n0,012;1,8;2,2\n';
    const t = await I.parseString(txt, 'de.csv');
    ok(t.info.delimiter === ';' && t.info.decimalComma && t.cols[1].data[1] === 1.6, 'semicolon + decimal comma');
    const s = I.suggest(t);
    ok(t.cols[s.x].name === 'PosX' && t.cols[s.y].name === 'PosY' && t.cols[s.t].name === 'Zeit', 'names PosX / PosY / Zeit recognised');
    ok(s.time.unit === 's' && s.time.rate === 250, 'seconds with a 4 ms step = 250 Hz');
  }
  {
    const lines = ['# recorded by a test', 'timestamp_us  gaze_x  gaze_y'];
    for (let i = 0; i < 300; i++) lines.push((i * 8333) + '  ' + (i * 0.1).toFixed(3) + '  ' + Math.sin(i / 20).toFixed(3));
    const t = await I.parseString(lines.join('\r\n'), 'ws.txt');
    ok(t.info.delimiter === 'ws' && t.info.header && t.cols.length === 3, 'whitespace separated, comment line skipped, CRLF');
    const s = I.suggest(t);
    ok(s.time.unit === 'us' && s.time.rate === 120, 'name hint "us" + 8.333 ms step = 120 Hz');
  }
  {
    // irregular time stamps: dropped frames become invalid gaps, the rest is interpolated exactly
    const n = 400, t = [], x = [], y = [];
    for (let i = 0; i < n; i++) { if (i >= 100 && i < 110) continue; t.push(i * 10 + (i % 3 === 0 ? 2 : 0)); x.push(i * 10 * 0.01); y.push(5); }
    const txt = 'time_ms,x,y\n' + t.map((v, i) => v + ',' + x[i] + ',' + y[i]).join('\n');
    const tb = await I.parseString(txt, 'irr.csv');
    const s = I.suggest(tb);
    ok(s.time.unit === 'ms' && s.time.rate === 100, 'jittered 10 ms stamps: 100 Hz');
    const b = I.buildTrials(tb, mapping(tb, s, { resample: true }));
    const tr = b.trials[0];
    ok(tr.resampled && near(tr.x[50], 5.0, 0.2), 'resampled signal follows the ramp (x(0.5 s) = ' + tr.x[50].toFixed(3) + ')');
    ok(Number.isNaN(tr.x[105]) && tr.valid[105] === 0 && tr.valid[50] === 1, 'a 100 ms hole in the recording stays invalid');
  }
  {
    // the same file read in 37-character chunks gives the same table (carry-over of partial lines, sniffing across chunks)
    const txt = read('gazebase_hss.csv').slice(0, 20000);
    const chunks = (function* () { for (let i = 0; i < txt.length; i += 37) yield txt.slice(i, i + 37); })();
    const a = await I.parseString(txt, 'a.csv'), b = await I.parseSource(chunks, 'a.csv', txt.slice(0, 500));
    ok(a.nRows === b.nRows && a.cols.every((c, j) => c.data.every((v, i) => Object.is(v, b.cols[j].data[i]))), 'chunked reading equals whole-string reading (' + a.nRows + ' rows)');
  }
  {
    const hs = I.degPerPixel(1920, 52.7, 60);
    ok(near(hs, 0.0247, 0.0002), 'degrees per pixel from screen width and distance: ' + hs.toFixed(4));
    const tr = [{ x: Float32Array.from([0, 1, 2, 3]), y: new Float32Array(4), valid: Uint8Array.from([1, 1, 1, 1]) }];
    const big = []; for (let i = 0; i < 200; i++) big.push(i);
    ok(near(I.peakSpeed([{ x: Float32Array.from(big), y: new Float32Array(200), valid: new Uint8Array(200).fill(1) }], 100, 1), 100, 1e-6), 'peak speed in deg/s');
  }

  // ------------------------------------------------------------ whole files from the datasets folder
  const D = process.env.UNEYE_DATASETS || path.join(os.homedir(), 'Desktop', 'Desktop', 'eye_tracking_datasets', 'datasets');
  if (fs.existsSync(D)) {
    const fromPath = (p) => fs.createReadStream(p, { encoding: 'utf8', highWaterMark: 1 << 20 });
    const walk = (dir, ext, out, lim) => { for (const e of fs.readdirSync(dir, { withFileTypes: true })) { if (out.length >= lim) return; const p = path.join(dir, e.name); if (e.isDirectory()) walk(p, ext, out, lim); else if (p.endsWith(ext)) out.push(p); } return out; };
    const sweep = async (label, files, check) => {
      let good = 0, bad = [];
      for (const f of files) {
        try { const t = await I.parseSource(fromPath(f), path.basename(f), fs.readFileSync(f, 'utf8').slice(0, 8192)); const st = I.describeColumns(t); const s = I.suggest(t, st); if (check(t, s)) good++; else bad.push(path.basename(f)); }
        catch (e) { bad.push(path.basename(f) + ': ' + e.message); }
      }
      ok(files.length > 0 && bad.length === 0, label + ': ' + good + '/' + files.length + ' files read with X, Y and a plausible time axis' + (bad.length ? ' (bad: ' + bad.slice(0, 3).join('; ') + ')' : ''));
    };
    const pick = (arr, n) => arr.filter((_, i) => i % Math.max(1, Math.floor(arr.length / n)) === 0).slice(0, n);
    const xy = (t, s) => s.x >= 0 && s.y >= 0 && t.cols[s.x].name.toLowerCase() === 'x';
    try {
      const g = path.join(D, 'GazeCom'); if (fs.existsSync(g)) await sweep('GazeCom (whole files)', pick(walk(g, '.arff', [], 2000), 6), (t, s) => xy(t, s) && s.time && s.time.unit === 'us' && s.time.rate === 250);
      const e3 = path.join(D, '360EM_gin'); if (fs.existsSync(e3)) await sweep('360EM (whole files)', pick(walk(path.join(e3, 'ground_truth'), '.arff', [], 400), 6), (t, s) => xy(t, s) && s.time && s.time.rate >= 115 && s.time.rate <= 125);
      const vr = path.join(D, 'gazebasevr', 'data'); if (fs.existsSync(vr)) await sweep('GazeBaseVR (whole files)', pick(fs.readdirSync(vr).filter(f => f.endsWith('.csv')).sort().map(f => path.join(vr, f)), 6), (t, s) => xy(t, s) && s.time && s.time.unit === 'ms' && s.time.rate === 250);
      const nu = path.join(D, 'NUIG_EyeGaze01'); if (fs.existsSync(nu)) await sweep('NUIG (whole files)', pick(walk(nu, '.csv', [], 800), 6), (t, s) => s.x >= 0 && s.y >= 0 && s.time && s.time.unit === 'ms' && s.time.rate >= 28 && s.time.rate <= 62);
      const gb = path.join(D, 'GazeBase');
      if (fs.existsSync(gb)) {
        const zips = pick(walk(gb, '.zip', [], 900).filter(z => !/Stimuli/.test(z)), 3); let good = 0, n = 0;
        for (const z of zips) {
          const names = cp.execFileSync('unzip', ['-Z1', z]).toString().split('\n').filter(f => f.endsWith('.csv'));
          for (const nm of names.filter((_, i) => i % 3 === 0)) {
            n++; const txt = cp.execFileSync('unzip', ['-p', z, nm], { maxBuffer: 1 << 28 }).toString();
            const t = await I.parseString(txt, nm), s = I.suggest(t);
            if (t.cols[s.x].name === 'x' && t.cols[s.y].name === 'y' && s.valid && t.cols[s.valid.col].name === 'val') good++;
          }
        }
        ok(n > 0 && good === n, 'GazeBase (whole files from zips): ' + good + '/' + n + ' tasks read, X, Y and validity found');
      }
      const em = path.join(D, 'EMTeC', 'data', 'subject_level_data', 'ET_111');
      if (fs.existsSync(path.join(em, 'ET_111.asc'))) {
        const t0 = Date.now();
        const t = await I.parseSource(fromPath(path.join(em, 'ET_111.asc')), 'ET_111.asc', '** CONVERTED FROM');
        const s = I.suggest(t), sec = (Date.now() - t0) / 1000;
        ok(t.info.blocks === 86 && t.info.rate === 2000 && t.nRows > 1.9e6, 'EMTeC ET_111.asc (70 MB): ' + t.nRows + ' samples in ' + t.info.blocks + ' blocks, read in ' + sec.toFixed(1) + ' s');
        const b = I.buildTrials(t, mapping(t, s, { split: { mode: 'column', col: s.trial } }));
        ok(b.trials.length === 86 && b.trials.every(x => x.x.length > 1000), 'EMTeC ET_111.asc: 86 trials built, first named "' + b.trials[0].name + '"');
        ok(/^trial /.test(b.trials[0].name), 'EMTeC ET_111.asc: trials carry the TRIALID of the recording');
        const sp = I.peakSpeed(b.trials, 2000, 1);
        ok(sp > 0, 'EMTeC ET_111.asc: peak speed ' + sp.toFixed(0) + ' per second in the file units');
      }
    } catch (e) { failed++; console.log('FAIL sweep threw: ' + (e.stack || e)); }
  } else console.log('skip whole-file sweep: ' + D + ' not found');

  if (failed) { console.log('\n' + failed + ' check(s) failed'); process.exit(1); }
  console.log('\nall import checks passed');
})().catch(e => { console.error(e); process.exit(1); });
