// Network probability helpers of the labelers (no DOM, works in the browser as the global UnEyeProb and in Node via require):
//   - eventsFromProb: the U'n'Eye post-processing of a softmax P(saccade) trace (threshold, merge of short gaps, removal of short events),
//     used to derive saccade proposals when a file holds `prob` but no `pred_labels`
//   - eventsFromLabels: runs of a machine label trace (`pred_labels`: 1 saccade, 2 blink, 0 none)
//   - nearMissCandidates / buildNearMissQueue: potential events the network did not detect, for the "Near misses" review queue
//   - range helpers (merged [start, end] lists) and a compact Uint16 encoding of a probability trace for the autosave
// Tests: test_nearmiss.js.
(function (root) {
    'use strict';

    // U'n'Eye's binary_prediction (uneye/functions.py): prob > threshold, interior gaps shorter than mergeMs filled, events shorter than minDurMs removed.
    // prob: Float32Array (NaN = no prediction), fs in Hz. Returns [{onset, offset}] (inclusive sample indices).
    function eventsFromProb(prob, fs, opts) {
        const o = Object.assign({ threshold: 0.5, mergeMs: 5, minDurMs: 5 }, opts || {});
        const n = prob ? prob.length : 0, rate = fs > 0 ? fs : 1000;
        const mergeN = Math.round(o.mergeMs * rate / 1000), minN = Math.round(o.minDurMs * rate / 1000);
        let runs = [];
        for (let i = 0; i < n;) {
            if (!(prob[i] > o.threshold)) { i++; continue; }
            let j = i; while (j + 1 < n && prob[j + 1] > o.threshold) j++;
            runs.push({ onset: i, offset: j }); i = j + 1;
        }
        if (mergeN > 1 && runs.length > 1) {                    // gaps of fewer than mergeN samples between two events are filled
            const m = [runs[0]];
            for (let k = 1; k < runs.length; k++) {
                const prev = m[m.length - 1], gap = runs[k].onset - prev.offset - 1;
                if (gap < mergeN) prev.offset = runs[k].offset; else m.push(runs[k]);
            }
            runs = m;
        }
        if (minN > 1) runs = runs.filter(r => r.offset - r.onset + 1 >= minN);
        return runs;
    }

    // runs of 1 (saccade) and 2 (blink) of a label trace -> [{onset, offset, cls}]
    function eventsFromLabels(lab) {
        const out = [], n = lab ? lab.length : 0;
        for (let i = 0; i < n;) {
            const c = lab[i];
            if (c !== 1 && c !== 2) { i++; continue; }
            let j = i; while (j + 1 < n && lab[j + 1] === c) j++;
            out.push({ onset: i, offset: j, cls: c }); i = j + 1;
        }
        return out;
    }

    const overlaps = (a0, a1, list) => {
        for (const r of list || []) { const s = Array.isArray(r) ? r[0] : r.onset, e = Array.isArray(r) ? r[1] : r.offset; if (!(a1 < s || a0 > e)) return true; }
        return false;
    };

    // Near misses of one trial. A candidate is a contiguous run where prob >= lower (NaN breaks a run) whose peak is below the threshold
    // (with opts.includeUnproposedCrossings, also runs above it that no proposal covers, i.e. crossings removed by the post-processing:
    // off by default, in a real session (mocol 12) they are mostly 1-4 ms crossings at p ~ 1 at the trial starts, edge artefacts that would
    // fill the top of the queue). Runs overlapping an interval of opts.exclude (expert events, saccade proposals, rejected near misses:
    // [[s, e]] or [{onset, offset}]) are skipped. Bounds: where prob falls below half of the peak
    // on each side of the peak, inside the run; at = argmax.
    // Returns [{onset, offset, at, peak}].
    function nearMissCandidates(prob, opts) {
        const o = Object.assign({ lower: 0.1, threshold: 0.5, exclude: [], includeUnproposedCrossings: false }, opts || {});
        const out = [], n = prob ? prob.length : 0;
        for (let i = 0; i < n;) {
            if (!(prob[i] >= o.lower)) { i++; continue; }
            let j = i, at = i;
            while (j + 1 < n && prob[j + 1] >= o.lower) { j++; if (prob[j] > prob[at]) at = j; }
            const peak = prob[at];
            if ((peak < o.threshold || o.includeUnproposedCrossings) && !overlaps(i, j, o.exclude)) {
                const half = peak / 2;
                let on = at, off = at;
                while (on - 1 >= i && prob[on - 1] >= half) on--;
                while (off + 1 <= j && prob[off + 1] >= half) off++;
                out.push({ onset: on, offset: off, at, peak });
            }
            i = j + 1;
        }
        return out;
    }

    // Candidates of every trial, sorted by decreasing peak (ties: trial, then time).
    // probs: [Float32Array|null] per trial; opts as nearMissCandidates plus excludeOf(t) -> exclusion intervals of trial t.
    function buildNearMissQueue(probs, opts) {
        const o = opts || {}, q = [];
        for (let t = 0; t < (probs || []).length; t++) {
            if (!probs[t]) continue;
            const ex = o.excludeOf ? o.excludeOf(t) : [];
            for (const c of nearMissCandidates(probs[t], Object.assign({}, o, { exclude: ex }))) q.push(Object.assign({ trial: t }, c));
        }
        q.sort((a, b) => b.peak - a.peak || a.trial - b.trial || a.at - b.at);
        return q;
    }

    // ---- ranges: sorted, merged, inclusive [start, end] lists
    function mergeRanges(ranges, maxLen) {
        const clean = [];
        for (const r of ranges || []) {
            if (!Array.isArray(r) || r.length < 2) continue;
            let s = Math.round(r[0]), e = Math.round(r[1]);
            if (maxLen !== undefined) { s = Math.max(0, Math.min(maxLen - 1, s)); e = Math.max(0, Math.min(maxLen - 1, e)); }
            if (s <= e) clean.push([s, e]);
        }
        clean.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        const out = [];
        for (const r of clean) { const p = out[out.length - 1]; if (p && r[0] <= p[1] + 1) p[1] = Math.max(p[1], r[1]); else out.push(r); }
        return out;
    }
    // adds [s, e] to a merged list; returns the same list when it was already covered (no change), a new merged list otherwise
    function addRange(ranges, s, e, maxLen) {
        for (const r of ranges || []) if (r[0] <= s && e <= r[1]) return ranges;
        return mergeRanges([...(ranges || []), [s, e]], maxLen);
    }
    function rangeLength(ranges) { let n = 0; for (const r of ranges || []) n += r[1] - r[0] + 1; return n; }

    // ---- compact storage of a probability trace: Uint16 (0..65534 = 0..1, 65535 = NaN): half the size of float32, step 1.5e-5
    function encodeProb(p) {
        const out = new Uint16Array(p.length);
        for (let i = 0; i < p.length; i++) { const v = p[i]; out[i] = v === v ? Math.round(Math.max(0, Math.min(1, v)) * 65534) : 65535; }
        return out;
    }
    function decodeProb(u) {
        const out = new Float32Array(u.length);
        for (let i = 0; i < u.length; i++) out[i] = u[i] === 65535 ? NaN : u[i] / 65534;
        return out;
    }

    const api = { eventsFromProb, eventsFromLabels, nearMissCandidates, buildNearMissQueue, mergeRanges, addRange, rangeLength, encodeProb, decodeProb };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    root.UnEyeProb = api;
})(typeof self !== 'undefined' ? self : this);
