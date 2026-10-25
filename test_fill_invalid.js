// node test_fill_invalid.js
// Tests the training-copy fill of invalid (NaN / tracker-loss) stretches in
// app.js: a blink labelled 2 on a NaN stretch must keep its label and read
// as a noisy (not flat) velocity block, exactly like the native Android
// trainer's fill_invalid. Without the fill, make_dataset drops the label and
// 3-class training collapses (saccades come back labelled as blinks).
//
// app.js is an IIFE, so the functions are extracted from its source and
// evaluated in isolation (they only use isFinite / Math / typed arrays).
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8');
const androidSrc = fs.readFileSync(path.join(__dirname, 'android/app/src/main/assets/web/app.js'), 'utf8');
if (src !== androidSrc) {
    console.error('FAIL: app.js and android/app/src/main/assets/web/app.js are out of sync');
    process.exit(1);
}

// pull out the helper block between the markers we inserted
const start = src.indexOf('function noiseStdOf(v)');
const end = src.indexOf('function preprocessTrial(');
if (start < 0 || end < 0 || end <= start) { console.error('FAIL: helper block not found in app.js'); process.exit(1); }
const block = src.slice(start, end);
// the Savitzky-Golay helpers of the display filter (networkInputOf smooths the network input with them when the filter is on)
const sgStart = src.indexOf('function makeSavitzkyGolay(w)'), sgEnd = src.indexOf('// Speed limit of the velocities used by the pre-labeler');
if (sgStart < 0 || sgEnd <= sgStart) { console.error('FAIL: Savitzky-Golay helpers not found in app.js'); process.exit(1); }
const sgBlock = src.slice(sgStart, sgEnd);
const make = (state) => new Function('state', sgBlock + block + '\nreturn { noiseStdOf, fillInvalidCopy, trainingCopyOf, networkInputOf, makeSavitzkyGolay, applySavitzkyGolay, postprocessSaccades };')(state);
const helpers = make({ filterSettings: { enabled: false }, rateHz: 1000 });

let failed = 0;
const ok = (c, msg) => { console.log((c ? 'ok   ' : 'FAIL ') + msg); if (!c) failed++; };

// 1. noiseStdOf: MAD of first differences, robust to saccade steps
{
    let s = 12345;
    const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x80000000 - 0.5;
    const v = new Float32Array(2000);
    let p = 0;
    for (let i = 0; i < v.length; i++) { p += rnd() * 0.004; v[i] = p; }
    for (let i = 0; i < 20; i++) v[100 * i] += 0.5;   // 20 huge saccade steps
    const sd = helpers.noiseStdOf(v);
    console.log('     noiseStdOf on fixation noise 0.002 + 20 saccade steps: ' + sd.toFixed(5));
    ok(sd > 0.0005 && sd < 0.01, 'noiseStdOf in the fixation-noise range (MAD ignores the saccade steps)');
}

// 2. fillInvalidCopy: only non-finite samples change, centred on the neighbour
{
    const v = Float32Array.from([1, 2, NaN, NaN, NaN, 3, Infinity, 4]);
    const out = helpers.fillInvalidCopy(v, 0.01, { s: 7 });
    ok(Number.isFinite(out[2]) && Number.isFinite(out[3]) && Number.isFinite(out[4]) && Number.isFinite(out[6]), 'all non-finite samples filled');
    ok(out[0] === 1 && out[1] === 2 && out[5] === 3 && out[7] === 4, 'valid samples untouched');
    ok(Math.abs(out[2] - 2) < 0.2 && Math.abs(out[6] - 3) < 0.2, 'filled samples centred on the neighbour value');
    ok(v[2] !== v[2] && !Number.isFinite(v[6]), 'input array not modified');
}

// 3. trainingCopyOf: the app-like case — fixation noise, saccades, a NaN
//    blink; the copy must keep the blink as a noisy block, the raw trial stays NaN
{
    const N = 6000;
    const x = new Float32Array(N), y = new Float32Array(N);
    let s = 7;
    const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x80000000 - 0.5;
    let cx = 0, cy = 0;
    for (let t = 0; t < N; t++) {
        if (t >= 1000 && t < 1300) { x[t] = NaN; y[t] = NaN; continue; }        // blink: tracker lost
        if ((t >= 500 && t < 550) || (t >= 2500 && t < 2550) || (t >= 4500 && t < 4550)) {
            const ph = (t % 1000 - 500) / 50, env = Math.sin(Math.PI * ph);
            cx += 0.03 * env; cy += 0.012 * env;                               // saccade
        } else { cx += rnd() * 0.003; cy += rnd() * 0.003; }                   // fixation noise
        x[t] = cx; y[t] = cy;
    }
    const tr = { x, y };
    const tc = helpers.trainingCopyOf(tr, 1001);
    let blinkFinite = 0, blinkNonZeroVel = 0, rawStillNaN = 0;
    for (let t = 1000; t < 1300; t++) {
        if (Number.isFinite(tc.x[t]) && Number.isFinite(tc.y[t])) blinkFinite++;
        if (t > 1000 && Number.isFinite(tc.x[t]) && Math.abs(tc.x[t] - tc.x[t - 1]) > 1e-4) blinkNonZeroVel++;
        if (!Number.isFinite(tr.x[t])) rawStillNaN++;
    }
    ok(blinkFinite === 300, `blink stretch fully filled in the copy (${blinkFinite}/300 finite)`);
    ok(blinkNonZeroVel > 250, `the filled blink reads as noise, not a flat block (${blinkNonZeroVel}/299 non-zero velocities)`);
    ok(rawStillNaN === 300, 'the raw trial keeps its NaN stretch (display / prediction untouched)');
    // saccade samples must be untouched by the fill
    let saccUntouched = true;
    for (let t = 500; t < 550; t++) if (tc.x[t] !== tr.x[t] || tc.y[t] !== tr.y[t]) saccUntouched = false;
    ok(saccUntouched, 'saccade samples identical in the copy');
}

// 4. determinism: same seed -> same copy (stable training across rounds)
{
    const v = Float32Array.from([0, NaN, NaN, 1]);
    const a = helpers.fillInvalidCopy(v, 0.01, { s: 42 });
    const b = helpers.fillInvalidCopy(v, 0.01, { s: 42 });
    ok(a[1] === b[1] && a[2] === b[2], 'same seed gives the same fill');
}


// network input with the display filter on: the smoothed trace (identical to the display's Savitzky-Golay), NaN where invalid, then filled
{
    const hf = make({ filterSettings: { enabled: true, windowMs: 11 }, rateHz: 1000 });
    const n = 600, x = new Float32Array(n), y = new Float32Array(n), valid = new Uint8Array(n).fill(1);
    for (let i = 0; i < n; i++) { x[i] = Math.sin(i / 40) + (i > 300 ? 0.5 : 0); y[i] = Math.cos(i / 55); }
    for (let i = 200; i < 230; i++) { x[i] = NaN; y[i] = NaN; valid[i] = 0; }
    const tr = { x, y, valid };
    const inp = hf.networkInputOf(tr);
    const ref = hf.applySavitzkyGolay(x, valid, 1000, hf.makeSavitzkyGolay(11)).smooth;
    let same = true, nanOk = true;
    for (let i = 0; i < n; i++) {
        if (valid[i]) { if (Math.abs(inp.x[i] - ref[i]) > 1e-6) same = false; }
        else if (!isNaN(inp.x[i])) nanOk = false;
    }
    ok(same, 'filter on: the network input is the display\'s Savitzky-Golay trace');
    ok(nanOk, 'filter on: invalid samples stay invalid before the noise fill');
    const tc = hf.trainingCopyOf(tr, 7);
    ok(Array.from(tc.x).every(Number.isFinite), 'filter on: the training copy has no non-finite sample');
    const raw = make({ filterSettings: { enabled: false }, rateHz: 1000 }).networkInputOf(tr);
    ok(raw.x === x, 'filter off: the network input is the raw trace');
}
// post-processing of the saccade class: gaps < 5 ms merged, saccades < 5 ms removed, blinks untouched
{
    const p = new Uint8Array(60);
    for (let i = 5; i < 15; i++) p[i] = 1; for (let i = 18; i < 25; i++) p[i] = 1;     // gap of 3 samples -> merged
    for (let i = 35; i < 38; i++) p[i] = 1;                                             // 3-sample saccade -> removed
    for (let i = 45; i < 48; i++) p[i] = 2;                                             // short blink -> kept
    const q = helpers.postprocessSaccades(p, 1000);
    ok(q.slice(5, 25).every(v => v === 1), 'post-processing merges saccades separated by less than 5 ms');
    ok(q.slice(35, 38).every(v => v === 0), 'post-processing removes saccades shorter than 5 ms');
    ok(q.slice(45, 48).every(v => v === 2), 'post-processing leaves blinks untouched');
}
console.log(failed ? failed + ' check(s) FAILED' : 'all checks passed');
process.exit(failed ? 1 : 0);
