// node test_nearmiss.js
// 1. nearmiss.js (UnEyeProb): U'n'Eye post-processing of a probability trace, runs of pred_labels, near-miss candidates (bounds at half the
//    peak, exclusions, order by decreasing peak), ranges, Uint16 storage of a probability trace.
// 2. formats.js: prob and pred_labels read next to the positions, cut like them (xy + trial_offset in .npz, 2-D trials x time, MAT cells,
//    and .h5 if h5wasm is found: H5WASM=/path/to/h5wasm/dist/node/hdf5_hl.js or h5wasm installed next to this file).
// 3. nearmiss.js is in the Android assets (same content).
const fs = require('fs'), path = require('path');
const P = require('./nearmiss.js'), F = require('./formats.js');
let failed = 0;
const ok = (c, msg) => { console.log((c ? 'ok   ' : 'FAIL ') + msg); if (!c) failed++; };
const J = JSON.stringify;

// ---- 1. pure functions
{
    // threshold 0.5 (strict), gap of 3 samples merged (< 5), event of 3 samples dropped (< 5), NaN = below
    const p = new Float32Array(60).fill(0.05);
    for (let i = 10; i < 20; i++) p[i] = 0.9;         // event 10-19
    for (let i = 23; i < 28; i++) p[i] = 0.8;         // gap 20-22 (3 samples) -> merged: 10-27
    for (let i = 40; i < 43; i++) p[i] = 0.7;         // 3 samples -> dropped
    p[50] = NaN; p[30] = 0.5;                         // exactly the threshold: not above
    const ev = P.eventsFromProb(p, 1000, { threshold: 0.5, mergeMs: 5, minDurMs: 5 });
    ok(J(ev) === J([{ onset: 10, offset: 27 }]), 'eventsFromProb: threshold, merge of a 3-ms gap, removal of a 3-ms event: ' + J(ev));
    const ev500 = P.eventsFromProb(p, 500, { mergeMs: 5, minDurMs: 5 });      // at 500 Hz 3 samples = 6 ms: the gap is not merged, the event stays
    ok(ev500.length === 3 && ev500[1].onset === 23 && ev500[2].onset === 40, 'eventsFromProb at 500 Hz scales the durations: ' + J(ev500));
    const lab = Int8Array.from([0, 1, 1, 0, 2, 2, 2, 0, 1]);
    ok(J(P.eventsFromLabels(lab)) === J([{ onset: 1, offset: 2, cls: 1 }, { onset: 4, offset: 6, cls: 2 }, { onset: 8, offset: 8, cls: 1 }]), 'eventsFromLabels: runs of 1 and 2');
}
{
    // candidates: a bump of peak 0.43 (half 0.215), a bump of peak 0.2, a crossing covered by a proposal, a bump overlapping an expert event,
    // a run broken by NaN, a short crossing that no proposal covers
    const p = new Float32Array(400).fill(0.02);
    const bump = (c, peak, hw) => { for (let i = c - hw; i <= c + hw; i++) p[i] = Math.max(p[i], peak * (1 - Math.abs(i - c) / (hw + 1))); };
    bump(50, 0.43, 9);          // run where p >= 0.1: |i-50| <= 7 (0.43*(1-d/10) >= 0.1 <=> d <= 7.67); half 0.215: d <= 5
    bump(120, 0.2, 9);          // p >= 0.1 for d <= 4; half 0.1: same
    bump(200, 0.9, 9);          // crosses the threshold, covered by a proposal 196-204
    bump(260, 0.3, 9);          // overlaps the expert event 262-270
    bump(330, 0.35, 9); p[328] = NaN;                    // NaN at d = -2 splits the run: left part peak 0.35*(1-3/10)=0.245, right part 0.35
    bump(380, 0.6, 3);          // crossing, no proposal covers it (removed by the post-processing)
    const exclude = [[196, 204], { onset: 262, offset: 270 }];
    const c = P.nearMissCandidates(p, { lower: 0.1, threshold: 0.5, exclude });
    const byAt = Object.fromEntries(c.map(x => [x.at, x]));
    ok(byAt[50] && byAt[50].onset === 45 && byAt[50].offset === 55 && Math.abs(byAt[50].peak - 0.43) < 1e-6, 'candidate at the peak 0.43: bounds at half the peak (45-55): ' + J(byAt[50]));
    ok(byAt[120] && byAt[120].onset === 115 && byAt[120].offset === 125, 'candidate of peak 0.2: bounds = the run (half the peak is below the lower bound): ' + J(byAt[120]));
    ok(!byAt[200], 'a crossing covered by a proposal is not a near miss');
    ok(!c.some(x => x.at >= 250 && x.at <= 272), 'a bump overlapping an expert event is not a near miss');
    ok(byAt[330] && byAt[327] && byAt[327].offset === 327 && byAt[330].onset === 329, 'NaN splits a run in two candidates: ' + J([byAt[327], byAt[330]]));
    ok(!byAt[380] && c.length === 4, 'a crossing that no proposal covers (dropped by the post-processing) is not a near miss by default');
    const wide = P.nearMissCandidates(p, { lower: 0.1, threshold: 0.5, exclude, includeUnproposedCrossings: true });
    ok(wide.some(x => x.at === 380 && x.peak > 0.5) && wide.length === c.length + 1, 'includeUnproposedCrossings: true adds it');
    const c2 = P.nearMissCandidates(p, { lower: 0.25, threshold: 0.5, exclude });
    ok(!c2.some(x => x.at === 120) && c2.some(x => x.at === 50), 'a higher lower bound drops the weak bump');

    // queue over trials, sorted by decreasing peak; exclusions per trial
    const p2 = new Float32Array(100);
    const q = P.buildNearMissQueue([p, null, p2.map((v, i) => (Math.abs(i - 30) <= 5 ? 0.48 * (1 - Math.abs(i - 30) / 6) : 0.01))],
        { lower: 0.1, threshold: 0.5, excludeOf: t => (t === 0 ? exclude : []) });
    const peaks = q.map(x => x.peak);
    ok(q.length === c.length + 1 && peaks.every((v, i) => i === 0 || peaks[i - 1] >= v), 'queue of all trials sorted by decreasing peak: ' + q.map(x => `t${x.trial}@${x.at}:${x.peak.toFixed(2)}`).join(' '));
    ok(q[0].trial === 2 && q[0].at === 30 && q[1].at === 50, 'first candidate is the 0.48 of trial 2, then the 0.43 of trial 0');
}
{
    const r = P.mergeRanges([[10, 20], [21, 30], [50, 40], [45, 60], [100, 200]], 150);
    ok(J(r) === J([[10, 30], [45, 60], [100, 149]]), 'mergeRanges: touching merged, reversed dropped, clipped: ' + J(r));
    const same = P.addRange(r, 12, 25);
    ok(same === r, 'addRange returns the same list when already covered');
    ok(J(P.addRange(r, 25, 50)) === J([[10, 60], [100, 149]]) && P.rangeLength([[0, 9], [20, 20]]) === 11, 'addRange merges, rangeLength');
    const pr = Float32Array.from([0, 0.12345, 0.5, 1, NaN, 0.99999]);
    const back = P.decodeProb(P.encodeProb(pr));
    ok(back.every((v, i) => (isNaN(pr[i]) ? isNaN(v) : Math.abs(v - pr[i]) < 1e-5)) && P.encodeProb(pr).BYTES_PER_ELEMENT === 2, 'Uint16 storage of prob: error < 1e-5, NaN kept');
}

// ---- 2. formats.js
(async () => {
    const lens = [120, 80, 150];
    const offs = [0]; for (const l of lens) offs.push(offs[offs.length - 1] + l);
    const N = offs[3], xy = new Float32Array(2 * N), prob = new Float32Array(N), pred = new Int8Array(N);
    for (let i = 0; i < N; i++) { xy[2 * i] = Math.sin(i / 10); xy[2 * i + 1] = Math.cos(i / 10); prob[i] = (i % 50) / 50; pred[i] = prob[i] > 0.5 ? 1 : 0; }
    prob[5] = NaN;
    const check = (r, name) => {
        const good = r.prob && r.predLabels && r.prob.length === 3 && r.prob.every((p, k) => p instanceof Float32Array && p.length === lens[k]) &&
            r.predLabels.every((p, k) => p instanceof Int8Array && p.length === lens[k]) &&
            Math.abs(r.prob[1][7] - prob[offs[1] + 7]) < 1e-6 && r.predLabels[2][149] === pred[offs[2] + 149] && isNaN(r.prob[0][5]) && r.labels === null;
        ok(good, `${name}: prob (Float32Array) and pred_labels (Int8Array) per trial, cut like the positions, NaN kept, not read as labels`);
    };
    const npz = F.buildNpz([{ name: 'xy', type: 'f4', shape: [N, 2], data: xy }, { name: 'trial_offset', type: 'i4', shape: [4], data: Int32Array.from(offs) },
        { name: 'prob', type: 'f4', shape: [N], data: prob }, { name: 'pred_labels', type: 'i1', shape: [N], data: pred }, { name: 'fs', type: 'f8', shape: [], data: Float64Array.of(1000) }]);
    check(await F.readTrialsFile('a.npz', npz.buffer.slice(npz.byteOffset, npz.byteOffset + npz.byteLength)), 'npz xy + trial_offset');
    const npzNoProb = F.buildNpz([{ name: 'xy', type: 'f4', shape: [N, 2], data: xy }, { name: 'trial_offset', type: 'i4', shape: [4], data: Int32Array.from(offs) }]);
    const r0 = await F.readTrialsFile('b.npz', npzNoProb.buffer);
    ok(r0.prob === null && r0.predLabels === null && r0.trials.length === 3, 'a file without them: prob and predLabels are null');

    // 2-D trials x time (rows padded with NaN)
    const T = 150, X = new Float32Array(3 * T).fill(NaN), Y = new Float32Array(3 * T).fill(NaN), P2 = new Float32Array(3 * T).fill(NaN), L2 = new Int8Array(3 * T);
    for (let k = 0; k < 3; k++) for (let i = 0; i < lens[k]; i++) { X[k * T + i] = xy[2 * (offs[k] + i)]; Y[k * T + i] = xy[2 * (offs[k] + i) + 1]; P2[k * T + i] = prob[offs[k] + i]; L2[k * T + i] = pred[offs[k] + i]; }
    const npz2 = F.buildNpz([{ name: 'X', type: 'f4', shape: [3, T], data: X }, { name: 'Y', type: 'f4', shape: [3, T], data: Y }, { name: 'prob', type: 'f4', shape: [3, T], data: P2 }, { name: 'pred_labels', type: 'i1', shape: [3, T], data: L2 }]);
    check(await F.readTrialsFile('c.npz', npz2.buffer), 'npz 2-D X, Y trials x time');

    // MAT cells (trials, prob, pred_labels)
    const cell = (k, f) => { const d = new Float64Array(lens[k]); for (let i = 0; i < lens[k]; i++) d[i] = f(offs[k] + i); return { dims: [lens[k], 1], data: d }; };
    const mat = F.buildMat5([{ name: 'trials', cell: lens.map((l, k) => { const d = new Float64Array(2 * l); for (let i = 0; i < l; i++) { d[i] = xy[2 * (offs[k] + i)]; d[l + i] = xy[2 * (offs[k] + i) + 1]; } return { dims: [l, 2], data: d }; }) },
        { name: 'prob', cell: lens.map((l, k) => cell(k, i => prob[i])) }, { name: 'pred_labels', cell: lens.map((l, k) => cell(k, i => pred[i])) }]);
    check(await F.readTrialsFile('d.mat', mat.buffer), 'MAT cell arrays');

    // HDF5 (the mocol layout) when h5wasm is available
    let h5mod = null;
    try { const hm = await import(process.env.H5WASM || 'h5wasm/node'); h5mod = hm.default && hm.default.ready ? hm.default : hm; } catch (e) { console.log('skip HDF5 (h5wasm not found; set H5WASM)'); }
    if (h5mod) {
        const bytes = await F.buildH5([{ name: 'xy', type: 'f4', shape: [N, 2], data: xy }, { name: 'trial_offset', type: 'i4', shape: [4], data: Int32Array.from(offs) },
            { name: 'prob', type: 'f4', shape: [N], data: prob }, { name: 'pred_labels', type: 'i1', shape: [N], data: pred }], { fs: 1000, threshold: 0.5, merge_ms: 5, min_dur_ms: 5 }, h5mod);
        const r = await F.readTrialsFile('a.h5', bytes.buffer, h5mod);
        check(r, 'h5 xy + trial_offset');
        ok(r.probParams && r.probParams.threshold === 0.5 && r.probParams.merge_ms === 5 && r.probParams.min_dur_ms === 5 && r.fs === 1000, 'h5 attributes threshold / merge_ms / min_dur_ms: ' + J(r.probParams));
        if (process.env.MOCOL_H5) {
            const b = fs.readFileSync(process.env.MOCOL_H5), t0 = Date.now(), rm = await F.readTrialsFile('m.h5', b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength), h5mod);
            const nS = rm.trials.reduce((a, t) => a + t.x.length, 0), t1 = Date.now();
            const q = P.buildNearMissQueue(rm.prob, { lower: 0.1, threshold: 0.5, excludeOf: t => (rm.predLabels ? P.eventsFromLabels(rm.predLabels[t]) : P.eventsFromProb(rm.prob[t], 1000)) });
            ok(rm.prob && rm.prob.length === rm.trials.length, `${path.basename(process.env.MOCOL_H5)}: ${rm.trials.length} trials, ${nS} samples, prob ${!!rm.prob}, pred_labels ${!!rm.predLabels}, read in ${t1 - t0} ms; ` +
                `${q.length} near misses (peak < 0.5) in ${Date.now() - t1} ms, top peaks ${q.slice(0, 5).map(c => c.peak.toFixed(2)).join(' ')}`);
        }
    }

    // ---- 3. Android copy
    const a = path.join(__dirname, 'android/app/src/main/assets/web/nearmiss.js');
    ok(fs.existsSync(a) && fs.readFileSync(a, 'utf8') === fs.readFileSync(path.join(__dirname, 'nearmiss.js'), 'utf8'), 'nearmiss.js is in the Android assets');

    console.log(failed ? failed + ' check(s) FAILED' : 'all checks passed');
    process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
