/**
 * U'n'Eye — Offline Saccade Labeler (Web / iPad / Touch)
 * Client-side high-performance canvas engine with touch, pinch-zoom, boundary handles,
 * file IO (CSV & MAT v5), and convergence tracking.
 */

(() => {
    'use strict';

    // --- State Constants ---
    const CLS_BACKGROUND = 0;
    const CLS_SACCADE = 1;
    const CLS_BLINK = 2;

    const PASS_ONE = 1; // Point marking
    const PASS_TWO = 2; // Onset / offset boundary refinement

    // --- Global App State ---
    const state = {
        rateHz: 1000.0,
        degPerUnit: 1.0,
        trials: [],          // Array of { x: Float32Array, y: Float32Array, valid: Uint8Array }
        currentTrialIdx: 0,
        pass: PASS_ONE,
        activeClass: CLS_SACCADE,

        // Window / Viewport (samples in current trial)
        fromSample: 0,
        toSample: 999,
        trialLength: 1000,

        // Annotation Data per Trial:
        // labels[trialIdx] = Uint8Array(trialLength) -> 0 bg, 1 saccade, 2 blink
        // reviewed[trialIdx] = Array of [start, end]
        // pointMarks[trialIdx] = Array of { at: sample, cls: 1|2 }
        // expertEvents[trialIdx] = Array of { onset: int, offset: int, cls: 1|2, fromPrelabel: bool }
        // prelabels[trialIdx] = Array of { onset: int, offset: int, cls: 1|2 }
        labels: [],
        reviewedRanges: [],
        pointMarks: [],
        expertEvents: [],
        prelabels: [],
        undoStack: [],

        // Pass 2 state: One event at a time, centered [-50ms .. +50ms]
        refineQueue: [],
        refineIdx: 0,
        currentMarkAt: -1,
        currentPutativeOnset: -1,
        currentPutativeOffset: -1,
        zoomMultiplier: 1.0,

        // Touch & Gesture state
        pointerDown: false,
        pointers: new Map(),
        initialPinchDist: 0,
        initialPinchFrom: 0,
        initialPinchTo: 0,
        dragStartSample: -1,
        dragCurrentSample: -1,
        dragPointerPos: { x: 0, y: 0 },
        isDraggingSpan: false,
        handleDrag: null,       // {ev, side, onset, offset}: a boundary handle being dragged (ev null = the pass-2 proposal on screen)
        lastTapTime: 0,
        lastTapSample: -1,
        autoSlide: false,           // auto-slide window 50ms before event after manual selection (disabled by default)
        eraser: false,              // eraser tool: drag between two times to delete every event / mark / pre-label in between
        gestureMulti: false,        // a second finger touched during this gesture (pinch zoom)
        gestureIgnored: false,      // the gesture started inside the refractory period after a pinch
        lastMultiTouchEnd: 0,

        // Pass 1 viewing windows: winMs wide, 10 % overlap, one after the other inside each trial
        winMs: 500,
        winIdx: 0,
        // scanWins[trial] = [from, to] windows that were displayed in pass 1 and left with Next (the only data that can ever be trained on)
        scanWins: [],

        // Convergence & Training History
        convergenceHistory: [],
        convergenceState: 'SILENT', // SILENT | COLLECTING | CONVERGED
        convergenceAlertShown: false,
        testTrials: new Set(),
        reviewedAtLastTraining: 0.0,
        isTraining: false,
        trainedModelBin: null,
        trainedModelJson: null,
        previousPrelabels: null,
        wakeLock: null,
        initialWeightsMode: 'pretrained',
        prelabelerMode: 'hmm',      // 'hmm' (label-free HMM, hmm_prelabel.js) or 'none'
        netSuggestions: true,       // once a network has been trained: show its proposals (on) or the HMM's (off); toggle in the menu
        minNewSecondsForAutoTrain: 2.0,   // automatic training after a pass 2 needs at least this much newly reviewed data

        // Signal Filtering & Preprocessing (Savitzky-Golay + Adaptive Engbert/HMM Threshold)
        filterSettings: {
            enabled: true,
            filterType: 'savitzky_golay',
            windowMs: 21,
            polyOrder: 3,
            lambda: 6.0,          // adaptive threshold multiplier (Engbert-Kliegl lambda)
            minPeakVel: 50.0,      // min peak speed (deg/s) to reject noise
            minAmplitude: 0.20,    // min amplitude (deg)
            minDurationMs: 10.0,   // min duration (ms)
            mergeGapMs: 20.0       // merge events closer than this gap
        },
    };

    // --- DOM Elements ---
    const elements = {
        canvas: document.getElementById('traceCanvas'),
        canvasContainer: document.getElementById('canvas-container'),
        passIndicator: document.getElementById('passIndicator'),
        convergenceBadge: document.getElementById('convergenceBadge'),
        saccadeBtn: document.getElementById('saccadeBtn'),
        blinkBtn: document.getElementById('blinkBtn'),
        classBtnGroup: document.querySelector('.class-btn-group'),
        classToggle: document.getElementById('classToggle'),
        undoBtn: document.getElementById('undoBtn'),
        menuBtn: document.getElementById('menuBtn'),
        toolsPanel: document.getElementById('toolsPanel'),
        eraserBtn: document.getElementById('eraserBtn'),
        autoSlideBtn: document.getElementById('autoSlideBtn'),
        autoSlideToggle: document.getElementById('autoSlideToggle'),
        saveLabelsBtn: document.getElementById('saveLabelsBtn'),
        loadLabelsBtn: document.getElementById('loadLabelsBtn'),
        labelsInput: document.getElementById('labelsInput'),
        winMsInput: document.getElementById('winMsInput'),
        modalWinMs: document.getElementById('modalWinMs'),
        prevEpochBtn: document.getElementById('prevEpochBtn'),
        trialNavGroup: document.getElementById('trialNavGroup'),
        startModal: document.getElementById('startModal'),
        startOpenBtn: document.getElementById('startOpenBtn'),
        netSuggestBtn: document.getElementById('netSuggestBtn'),
        openBtn: document.getElementById('openBtn'),
        filterBtn: document.getElementById('filterBtn'),
        metricsBtn: document.getElementById('metricsBtn'),
        trainNowBtn: document.getElementById('trainNowBtn'),
        exportBtn: document.getElementById('exportBtn'),
        prevTrialBtn: document.getElementById('prevTrialBtn'),
        nextTrialBtn: document.getElementById('nextTrialBtn'),
        switchPassBtn: document.getElementById('switchPassBtn'),
        nextEpochBtn: document.getElementById('nextEpochBtn'),
        pass1Controls: document.getElementById('pass1Controls'),
        pass2Controls: document.getElementById('pass2Controls'),
        prevEventBtn: document.getElementById('prevEventBtn'),
        nextEventBtn: document.getElementById('nextEventBtn'),
        eventCounterBadge: document.getElementById('eventCounterBadge'),
        notEventBtn: document.getElementById('notEventBtn'),
        widerBtn: document.getElementById('widerBtn'),
        acceptPreBtn: document.getElementById('acceptPreBtn'),
        trialStatus: document.getElementById('trialStatus'),
        reviewedStatus: document.getElementById('reviewedStatus'),
        eventsStatus: document.getElementById('eventsStatus'),
        trainStatus: document.getElementById('trainStatus'),
        cancelTrainBtn: document.getElementById('cancelTrainBtn'),
        downloadModelBtn: document.getElementById('downloadModelBtn'),
        initialModelSelect: document.getElementById('initialModelSelect'),
        fileInput: document.getElementById('fileInput'),
        exportFormat: document.getElementById('exportFormat'),
        loadModal: document.getElementById('loadModal'),
        modalFileName: document.getElementById('modalFileName'),
        modalChangeTraceBtn: document.getElementById('modalChangeTraceBtn'),
        rateInput: document.getElementById('rateInput'),
        scaleInput: document.getElementById('scaleInput'),
        modalSgWindow: document.getElementById('modalSgWindow'),
        localModelRow: document.getElementById('localModelRow'),
        modalModelFileName: document.getElementById('modalModelFileName'),
        modalPickModelBtn: document.getElementById('modalPickModelBtn'),
        modalModelFile: document.getElementById('modalModelFile'),
        modalLabelsFileName: document.getElementById('modalLabelsFileName'),
        modalPickLabelsBtn: document.getElementById('modalPickLabelsBtn'),
        modalLabelsFile: document.getElementById('modalLabelsFile'),
        prelabelerSelect: document.getElementById('prelabelerSelect'),
        alwaysExplorer: document.getElementById('alwaysExplorer'),
        modalConfirmBtn: document.getElementById('modalConfirmBtn'),
        modalCancelBtn: document.getElementById('modalCancelBtn'),
        filterModal: document.getElementById('filterModal'),
        filterCloseBtn: document.getElementById('filterCloseBtn'),
        filterCancelBtn: document.getElementById('filterCancelBtn'),
        filterApplyBtn: document.getElementById('filterApplyBtn'),
        enableSg: document.getElementById('enableSg'),
        sgWindowMs: document.getElementById('sgWindowMs'),
        metricsModal: document.getElementById('metricsModal'),
        metricsCanvas: document.getElementById('metricsCanvas'),
        metricsCloseBtn: document.getElementById('metricsCloseBtn'),
        alertModal: document.getElementById('alertModal'),
        alertMessage: document.getElementById('alertMessage'),
        alertOkBtn: document.getElementById('alertOkBtn'),
    };

    const ctx = elements.canvas.getContext('2d');
    let pendingFileToLoad = null;

    // --- Synthetic Data Generator (Demo) ---
    function generateDemoDataset() {
        const nTrials = 8;
        const nSamples = 1000;
        const trials = [];

        for (let t = 0; t < nTrials; t++) {
            const x = new Float32Array(nSamples);
            const y = new Float32Array(nSamples);
            const valid = new Uint8Array(nSamples).fill(1);

            let px = 0.0, py = 0.0;
            let i = 0;

            function gaussianNoise() {
                let u = 0, v = 0;
                while (u === 0) u = Math.random();
                while (v === 0) v = Math.random();
                return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
            }

            function addFixation(len) {
                for (let k = 0; k < len && i < nSamples; k++, i++) {
                    x[i] = px + 0.015 * gaussianNoise();
                    y[i] = py + 0.015 * gaussianNoise();
                }
            }

            function addSaccade(amp) {
                const theta = 2 * Math.PI * Math.random();
                const len = Math.max(15, Math.floor(20 + 2.5 * amp));
                const x0 = px, y0 = py;
                for (let k = 0; k < len && i < nSamples; k++, i++) {
                    const u = (k + 1) / len;
                    const s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u); // Smooth quintic polynomial
                    x[i] = x0 + amp * Math.cos(theta) * s + 0.015 * gaussianNoise();
                    y[i] = y0 + amp * Math.sin(theta) * s + 0.015 * gaussianNoise();
                }
                px += amp * Math.cos(theta);
                py += amp * Math.sin(theta);
            }

            // Pattern: Fixation -> Saccade 4 deg -> Fixation -> Saccade 7 deg -> Fixation -> Blink (NaN)
            addFixation(250);
            addSaccade(4.0);
            addFixation(160);
            addSaccade(7.0);
            addFixation(140);
            // Blink: 180ms NaN
            for (let k = 0; k < 180 && i < nSamples; k++, i++) {
                x[i] = NaN;
                y[i] = NaN;
                valid[i] = 0;
            }
            addFixation(nSamples - i);

            trials.push({ x, y, valid });
        }

        loadDataset(trials, 1000.0, 1.0, 'hmm');

        // Populate initial point marks from Pass 1 for demo trials so user can immediately test Pass 1 and Pass 2!
        for (let t = 0; t < nTrials; t++) {
            for (const p of state.prelabels[t]) {
                const mid = Math.round((p.onset + p.offset) / 2);
                state.pointMarks[t].push({ at: mid, cls: p.cls });
            }
            state.pointMarks[t].sort((a, b) => a.at - b.at);
        }

        updateStatus();
        render();
        requestAnimationFrame(() => {
            syncCanvasDimensions();
            render();
        });
    }

    // --- Savitzky-Golay (Cubic, scipy mode="interp") Kinematics Engine ---
    function makeSavitzkyGolay(w) {
        const h = Math.floor((w - 1) / 2);
        const n = 4; // cubic fit: 1, x, x^2, x^3
        const VtV = Array.from({ length: 4 }, () => new Float64Array(4));
        const V = Array.from({ length: w }, () => new Float64Array(4));
        for (let i = 0; i < w; i++) {
            const x = i - h;
            V[i][0] = 1; V[i][1] = x; V[i][2] = x * x; V[i][3] = x * x * x;
        }
        for (let a = 0; a < n; a++) {
            for (let b = 0; b < n; b++) {
                for (let i = 0; i < w; i++) VtV[a][b] += V[i][a] * V[i][b];
            }
        }
        const M = Array.from({ length: 4 }, (v, r) => {
            const row = new Float64Array(8);
            for (let c = 0; c < 4; c++) row[c] = VtV[r][c];
            row[4 + r] = 1.0;
            return row;
        });
        for (let c = 0; c < n; c++) {
            let p = c;
            for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
            const tmp = M[c]; M[c] = M[p]; M[p] = tmp;
            const d = M[c][c];
            for (let k = 0; k < 8; k++) M[c][k] /= d;
            for (let r = 0; r < n; r++) {
                if (r !== c) {
                    const f = M[r][c];
                    for (let k = 0; k < 8; k++) M[r][k] -= f * M[c][k];
                }
            }
        }
        const P = Array.from({ length: 4 }, () => new Float64Array(w));
        for (let a = 0; a < n; a++) {
            for (let i = 0; i < w; i++) {
                let acc = 0;
                for (let b = 0; b < n; b++) acc += M[a][4 + b] * V[i][b];
                P[a][i] = acc;
            }
        }
        return { w, h, P };
    }

    function applySavitzkyGolay(data, valid, fs, sg) {
        const T = data.length;
        const smooth = new Float32Array(T), d1 = new Float32Array(T), d2 = new Float32Array(T);
        const w = sg.w, h = sg.h;
        const y = new Float64Array(T);
        let lastVal = 0.0;
        for (let i = 0; i < T; i++) {
            if (valid[i] && !isNaN(data[i]) && isFinite(data[i])) { lastVal = data[i]; y[i] = data[i]; }
            else y[i] = lastVal;
        }
        for (let i = 0; i < T; i++) {
            const s0 = Math.min(Math.max(i - h, 0), T - w);
            let c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (let j = 0; j < w; j++) {
                const val = y[s0 + j];
                c0 += sg.P[0][j] * val; c1 += sg.P[1][j] * val; c2 += sg.P[2][j] * val; c3 += sg.P[3][j] * val;
            }
            const u = i - (s0 + h);
            smooth[i] = c0 + c1 * u + c2 * u * u + c3 * u * u * u;
            d1[i] = (c1 + 2 * c2 * u + 3 * c3 * u * u) * fs;
            d2[i] = (2 * c2 + 6 * c3 * u) * fs * fs;
        }
        return { smooth, d1, d2 };
    }

    // Speed limit of the velocities used by the pre-labeler, the thresholds and the display (deg/s): the fastest saccades of monkeys
    // reach 1000-1300 deg/s (humans about 700); more is a blink or a tracker loss. Same value as the C++ engine, the Python package and the Android app.
    const MAX_SPEED_DEG_S = 1300.0;

    // Version of the trainer that produced a saved model. Models saved by earlier versions (random 3-class head:
    // saccades proposed as blinks, test F1 0) are dropped when a session is restored, with the proposals they made.
    const MODEL_VERSION = 2;
    const APP_BUILD = '2026-10-25b';

    function computeRobustSigma(arr, valid) {
        const vals = [];
        for (let i = 0; i < arr.length; i++) if (valid[i]) vals.push(Math.abs(arr[i]));
        if (vals.length === 0) return 1.0;
        vals.sort((a, b) => a - b);
        const med = vals[Math.floor(vals.length / 2)];
        return Math.max(med / 0.6745, 1e-4);
    }

    // --- Training-copy fill of invalid (blink / tracker-loss) stretches ---
    // The velocity input maps any non-finite difference to 0, so a blink
    // labelled 2 on a NaN stretch would read as a flat zero-velocity block:
    // indistinguishable from fixation, and the trainer drops the label
    // exactly where the blink is -> 3-class training collapses and saccades
    // come back labelled as blinks (red). A real tracker returns noise during
    // a blink: fill the invalid stretches of the TRAINING COPY with Gaussian
    // noise at 3x the local fixation noise (same rule as the native Android
    // trainer). The raw trace (display, pre-labeler, prediction) is untouched.
    function noiseStdOf(v) {
        const d = [];
        for (let i = 1; i < v.length; i++) if (isFinite(v[i]) && isFinite(v[i - 1])) d.push(Math.abs(v[i] - v[i - 1]));
        if (d.length < 8) return 0.0;
        d.sort((a, b) => a - b);
        return d[Math.floor(d.length / 2)] / 0.6745;
    }

    // Box-Muller, deterministic per call sequence (seeded by the sample index
    // mix, good enough for a fill that only has to look like tracker noise).
    function gauss(rngState) {
        let u = 0, v = 0;
        do { u = (rngState.s = (rngState.s * 1103515245 + 12345) & 0x7fffffff) / 0x80000000; } while (u <= 1e-12);
        v = (rngState.s = (rngState.s * 1103515245 + 12345) & 0x7fffffff) / 0x80000000;
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function fillInvalidCopy(v, std, rngState) {
        const out = v.slice();
        let i = 0;
        while (i < out.length) {
            if (isFinite(out[i])) { i++; continue; }
            let j = i;
            while (j < out.length && !isFinite(out[j])) j++;
            const c = i > 0 ? out[i - 1] : (j < out.length ? out[j] : 0.0);
            for (let k = i; k < j; k++) out[k] = c + std * gauss(rngState);
            i = j;
        }
        return out;
    }

    // Returns a training copy of the trial: NaN stretches filled with noise at
    // 3x the local fixation noise. The trial itself is not modified.
    function trainingCopyOf(tr, seed) {
        const rngState = { s: (seed >>> 0) || 1 };
        const sx = Math.max(3.0 * noiseStdOf(tr.x), 1e-3);
        const sy = Math.max(3.0 * noiseStdOf(tr.y), 1e-3);
        return { x: fillInvalidCopy(tr.x, sx, rngState), y: fillInvalidCopy(tr.y, sy, rngState) };
    }

    function preprocessTrial(trialIdx) {
        const tr = state.trials[trialIdx];
        if (!tr) return;
        const n = tr.x.length;
        const fs = state.rateHz;
        const cfg = state.filterSettings;

        if (!cfg.enabled) {
            tr.smoothX = tr.x;
            tr.smoothY = tr.y;
            tr.speed = new Float32Array(n);
            const dt = 1.0 / fs;
            for (let i = 1; i < n; i++) {
                if (tr.valid[i] && tr.valid[i - 1]) {
                    const dx = (tr.x[i] - tr.x[i - 1]) * state.degPerUnit;
                    const dy = (tr.y[i] - tr.y[i - 1]) * state.degPerUnit;
                    const sp = Math.sqrt(dx * dx + dy * dy) / dt;
                    tr.speed[i] = sp > MAX_SPEED_DEG_S ? MAX_SPEED_DEG_S : sp;   // speed limit
                }
            }
            return;
        }

        let w = Math.round(cfg.windowMs * fs / 1000.0);
        if (w % 2 === 0) w += 1;
        w = Math.max(5, Math.min(w, Math.floor((n / 2) * 2 - 1)));

        const sg = makeSavitzkyGolay(w);
        const rx = applySavitzkyGolay(tr.x, tr.valid, fs, sg);
        const ry = applySavitzkyGolay(tr.y, tr.valid, fs, sg);

        tr.smoothX = rx.smooth;
        tr.smoothY = ry.smooth;
        tr.speed = new Float32Array(n);

        const sc = state.degPerUnit;
        for (let i = 0; i < n; i++) {
            if (tr.valid[i]) {
                const sp = Math.sqrt(rx.d1[i] * rx.d1[i] + ry.d1[i] * ry.d1[i]) * sc;
                tr.speed[i] = sp > MAX_SPEED_DEG_S ? MAX_SPEED_DEG_S : sp;   // speed limit
            }
        }
    }

    // The smoothed traces and the speed are only read for the trial on screen (and by the position export), so they are
    // computed on demand and only a few trials keep them: a long dataset no longer holds three extra arrays per trial.
    const DERIVED_CACHE_TRIALS = 4;
    const derivedOrder = [];
    function dropDerived(tr) { tr.smoothX = tr.smoothY = tr.speed = null; }
    function ensureDerived(trialIdx) {
        const tr = state.trials[trialIdx];
        if (!tr) return;
        if (!tr.speed) preprocessTrial(trialIdx);
        const at = derivedOrder.indexOf(trialIdx);
        if (at >= 0) derivedOrder.splice(at, 1);
        derivedOrder.push(trialIdx);
        while (derivedOrder.length > DERIVED_CACHE_TRIALS) {
            const old = state.trials[derivedOrder.shift()];
            if (old) dropDerived(old);
        }
    }

    // filter settings or dataset changed: forget every derived trace, the next render recomputes the visible trial
    function preprocessAllTrials() {
        derivedOrder.length = 0;
        for (const tr of state.trials) dropDerived(tr);
    }

    // --- Dataset Loader & State Initializer ---
    // trials have their own durations (variable-length files): always use the length of the trial in question
    function trialLenOf(i) { const t = state.trials[i]; return t ? t.x.length : 1000; }
    function curLen() { return trialLenOf(state.currentTrialIdx); }

    const TOUCH_REFRACTORY_MS = 600;   // after a pinch zoom, touches within this time are not marks
    const HANDLE_R_PX = 7;             // boundary handles of an event (dots at mid-height on its onset / offset lines)
    const HANDLE_HIT_X_PX = 22;        // a touch this close to a handle grabs it
    const HANDLE_HIT_Y_PX = 70;
    const WIN_OVERLAP = 0.10;

    function winSamples() { return Math.max(20, Math.round(state.winMs * state.rateHz / 1000)); }
    function winStep() { return Math.max(1, Math.round(winSamples() * (1 - WIN_OVERLAP))); }
    function numWindows(t) { const n = trialLenOf(t), w = winSamples(); return n <= w ? 1 : Math.ceil((n - w) / winStep()) + 1; }
    function gridWindow(t, k) {
        const n = trialLenOf(t), w = winSamples();
        if (n <= w) return [0, n - 1];
        let from = k * winStep();
        if (from + w > n) from = n - w;
        return [from, from + w - 1];
    }
    function applyGridWindow() {
        state.winIdx = Math.max(0, Math.min(state.winIdx, numWindows(state.currentTrialIdx) - 1));
        const [f, e] = gridWindow(state.currentTrialIdx, state.winIdx);
        state.fromSample = f; state.toSample = e;
    }

    function mergeRangeList(ranges, maxLen) {
        const clean = [];
        for (const r of (ranges || [])) {
            if (!Array.isArray(r) || r.length < 2) continue;
            const s = Math.max(0, Math.min(maxLen - 1, Math.round(r[0])));
            const e = Math.max(0, Math.min(maxLen - 1, Math.round(r[1])));
            if (s <= e) clean.push([s, e]);
        }
        clean.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        const merged = [];
        for (const r of clean) {
            const prev = merged[merged.length - 1];
            if (prev && r[0] <= prev[1] + 1) {
                prev[1] = Math.max(prev[1], r[1]);
            } else {
                merged.push([r[0], r[1]]);
            }
        }
        return merged;
    }

    // Merge any two overlapping (or touching) expert events of the same class into one single event
    function mergeOverlappingEvents(trialIdx) {
        const evs = state.expertEvents[trialIdx];
        if (!evs || evs.length === 0) return;
        evs.sort((a, b) => a.onset - b.onset || a.offset - b.offset);
        const merged = [];
        for (const e of evs) {
            let didMerge = false;
            for (let i = merged.length - 1; i >= 0; i--) {
                if (merged[i].cls === e.cls && e.onset <= merged[i].offset) {
                    merged[i].onset = Math.min(merged[i].onset, e.onset);
                    merged[i].offset = Math.max(merged[i].offset, e.offset);
                    merged[i].fromPrelabel = Boolean(merged[i].fromPrelabel && e.fromPrelabel);
                    didMerge = true;
                    break;
                }
            }
            if (!didMerge) {
                merged.push({ onset: e.onset, offset: e.offset, cls: e.cls, fromPrelabel: Boolean(e.fromPrelabel) });
            }
        }
        state.expertEvents[trialIdx] = merged;
        if (state.pointMarks[trialIdx]) {
            state.pointMarks[trialIdx] = state.pointMarks[trialIdx].filter(
                m => !merged.some(e => e.onset <= m.at && m.at <= e.offset)
            );
        }
    }

    // Reviewed = displayed in pass 1 (scanned) AND without a point mark still waiting for its refinement in pass 2,
    // plus any explicitly refined/accepted expert event intervals.
    function recomputeReviewed(t, notify) {
        const marks = state.pointMarks[t] || [];
        const ok = (state.scanWins[t] || []).filter(([f, e]) => !marks.some(m => m.at >= f && m.at <= e)).map(r => r.slice());
        for (const ev of (state.expertEvents[t] || [])) {
            ok.push([ev.onset, ev.offset]);
        }
        state.reviewedRanges[t] = mergeRangeList(ok, trialLenOf(t));
        if (notify) saveToIndexedDB();          // no automatic training here: it starts when a pass 2 is finished (maybeAutoTrain)
    }

    function scanCurrentWindow() {
        const t = state.currentTrialIdx;
        if (state.pass !== PASS_ONE || !state.scanWins[t]) return;
        let [f, e] = gridWindow(t, state.winIdx);
        if (state.fromSample <= f && state.toSample >= e) { f = Math.max(0, state.fromSample); e = Math.min(trialLenOf(t) - 1, state.toSample); }   // zoomed out: all of it was seen
        if (!state.scanWins[t].some(r => r[0] === f && r[1] === e)) state.scanWins[t].push([f, e]);
        recomputeReviewed(t, true);
    }

    function inScanned(t, at) { return (state.scanWins[t] || []).some(([f, e]) => at >= f && at <= e); }

    // labels of the human only: the refined events. Automatic pre-labels never count as ground truth.
    function humanLabels(t) {
        const out = new Uint8Array(trialLenOf(t));
        for (const e of state.expertEvents[t]) for (let i = e.onset; i <= e.offset && i < out.length; i++) out[i] = e.cls;
        return out;
    }

    function loadDataset(trials, rateHz = 1000.0, degPerUnit = 1.0, prelabeler = 'hmm') {
        state.trials = trials;
        state.prelabelerMode = prelabeler === 'none' ? 'none' : 'hmm';
        if (elements.startModal) elements.startModal.classList.add('hidden');
        state.rateHz = rateHz;
        state.degPerUnit = degPerUnit;
        state.currentTrialIdx = 0;
        state.trialLength = trials[0] ? trials[0].x.length : 1000;
        state.winIdx = 0;
        applyGridWindow();

        // Clear and initialize per-trial annotation arrays
        const n = trials.length;
        state.labels = Array.from({ length: n }, (_, t) => new Uint8Array(trials[t].x.length));      // trials may have different durations
        state.trialNames = trials.map(t => t.name || '');
        state.reviewedRanges = Array.from({ length: n }, () => []);
        state.scanWins = Array.from({ length: n }, () => []);
        state.pointMarks = Array.from({ length: n }, () => []);
        state.expertEvents = Array.from({ length: n }, () => []);
        state.prelabels = Array.from({ length: n }, () => []);
        state.undoStack = Array.from({ length: n }, () => []);

        // Pick test set (every 4th trial, min 2)
        state.testTrials = new Set();
        for (let t = 0; t < n; t += 4) {
            state.testTrials.add(t);
        }

        // 1. Preprocess all trials with Savitzky-Golay
        preprocessAllTrials();

        // 2. Proposals: HMM (a network trained in this session replaces them only if the toggle is on and a model is loaded)
        for (let t = 0; t < n; t++) {
            runPrelabeler(t, state.prelabelerMode);
        }
        if (state.netSuggestions && state.trainedModelBin) refreshSuggestions();

        state.convergenceHistory = [];
        state.convergenceState = 'SILENT';
        state.convergenceAlertShown = false;
        state.reviewedAtLastTraining = 0.0;

        setPass(PASS_ONE);
        updateStatus();
        render();
        requestAnimationFrame(() => {
            syncCanvasDimensions();
            render();
        });
    }

    // --- Pre-labeler: label-free universal HMM + acceleration-zero boundaries (hmm_prelabel.js = the C++ detector "hmm_accel"), blinks = NaN runs ---
    // Proposals only: they never count as labels. A trained network replaces them only when the "Network suggestions" toggle is on.
    function runPrelabeler(trialIdx, mode) {
        if (mode === 'none') {
            state.prelabels[trialIdx] = [];
            return;
        }
        const tr = state.trials[trialIdx];
        if (!tr) return;
        const n = tr.x.length;
        const pre = [];

        // 1. Blinks: invalid runs (>= 3 samples)
        let blinkStart = -1;
        for (let i = 0; i <= n; i++) {
            const isInvalid = i < n && (tr.valid[i] === 0 || isNaN(tr.x[i]) || isNaN(tr.y[i]));
            if (isInvalid && blinkStart === -1) {
                blinkStart = i;
            } else if (!isInvalid && blinkStart !== -1) {
                if (i - blinkStart >= 3) pre.push({ onset: blinkStart, offset: i - 1, cls: CLS_BLINK });
                blinkStart = -1;
            }
        }

        // 2. Saccades: HMM on the raw trace (unit-free score; the display filter is not used)
        for (const e of UnEyeHMM.detect(tr.x, tr.y, tr.valid, state.rateHz)) {
            const overlapsBlink = pre.some(b => b.cls === CLS_BLINK && !(e.offset < b.onset || e.onset > b.offset));
            if (!overlapsBlink) pre.push({ onset: e.onset, offset: e.offset, cls: CLS_SACCADE });
        }

        pre.sort((a, b) => a.onset - b.onset);
        state.prelabels[trialIdx] = pre;
    }

    // proposals of every trial from the current source: the trained network if there is one and the toggle is on, otherwise the HMM
    async function refreshSuggestions() {
        if (!state.trials.length) return;
        if (state.netSuggestions && state.trainedModelBin) {
            await applyModelAsPrelabeler();
        } else {
            for (let t = 0; t < state.trials.length; t++) { runPrelabeler(t, state.prelabelerMode); syncRenderedLabels(t); }
        }
        if (state.pass === PASS_TWO) { buildRefineQueue(); state.refineIdx = Math.min(state.refineIdx, Math.max(0, state.refineQueue.length - 1)); focusCurrentMark(); }
        updateStatus();
        render();
    }

    function setNetSuggestions(on) {
        state.netSuggestions = Boolean(on);
        try { localStorage.setItem('uneye_netSuggestions', on ? '1' : '0'); } catch (e) { }
        if (elements.netSuggestBtn) {
            elements.netSuggestBtn.textContent = 'Network suggestions: ' + (on ? 'On' : 'Off');
            elements.netSuggestBtn.classList.toggle('btn-accent', Boolean(on));
            elements.netSuggestBtn.title = on ? 'Proposals come from the network trained on your labels (HMM until a network has been trained). Tap for HMM proposals.'
                : 'Proposals come from the label-free HMM. Tap to show the proposals of the network trained on your labels.';
        }
    }

    // automatic training: only when a pass 2 is finished (the durations are refined then), never during pass 1
    function maybeAutoTrain() {
        if (state.isTraining || !state.trials.length) return;
        if (totalReviewedSeconds() - state.reviewedAtLastTraining >= state.minNewSecondsForAutoTrain) runTrainingRound();
    }

    // --- State Operations: Point Marks & Refinement ---
    function saveUndo(trialIdx) {
        if (!state.undoStack[trialIdx]) state.undoStack[trialIdx] = [];      // e.g. a session restored from the autosave
        state.undoStack[trialIdx].push({
            pointMarks: JSON.parse(JSON.stringify(state.pointMarks[trialIdx])),
            expertEvents: JSON.parse(JSON.stringify(state.expertEvents[trialIdx])),
        });
        if (state.undoStack[trialIdx].length > 30) state.undoStack[trialIdx].shift();
    }

    function undo(trialIdx) {
        const stack = state.undoStack[trialIdx];
        if (!stack || stack.length === 0) return;
        const prev = stack.pop();
        state.pointMarks[trialIdx] = prev.pointMarks;
        state.expertEvents[trialIdx] = prev.expertEvents;
        recomputeReviewed(trialIdx, false);
        syncRenderedLabels(trialIdx);
        if (state.pass === PASS_TWO) {
            buildRefineQueue();
            focusCurrentMark();
        }
        updateStatus();
        render();
        requestAnimationFrame(() => {
            syncCanvasDimensions();
            render();
        });
    }

    function addPointMark(trialIdx, at, cls) {
        saveUndo(trialIdx);
        // Remove existing mark if tapped within 15 samples
        const existingIdx = state.pointMarks[trialIdx].findIndex(m => Math.abs(m.at - at) <= 15);
        if (existingIdx !== -1) {
            state.pointMarks[trialIdx].splice(existingIdx, 1);
            recomputeReviewed(trialIdx, false);
            return;
        }

        // Snap to nearest prelabel within 30 samples
        let snappedAt = at;
        for (const p of state.prelabels[trialIdx]) {
            if (Math.abs(p.onset - at) <= 30) {
                snappedAt = p.onset;
                break;
            }
        }

        state.pointMarks[trialIdx].push({ at: snappedAt, cls });
        state.pointMarks[trialIdx].sort((a, b) => a.at - b.at);
        recomputeReviewed(trialIdx, false);
    }

    function refineMark(trialIdx, at, onset, offset, cls) {
        saveUndo(trialIdx);
        onset = Math.max(0, onset);
        offset = Math.min(trialLenOf(trialIdx) - 1, offset);
        if (onset > offset) return false;

        // Remove any point mark inside [onset, offset] or closest to `at`
        state.pointMarks[trialIdx] = (state.pointMarks[trialIdx] || []).filter(
            m => !(m.at >= onset && m.at <= offset) && Math.abs(m.at - at) > 35
        );

        if (state.pass === PASS_TWO) {
            // In Pass 2, update the event covering `at`, then merge any same-class overlaps
            state.expertEvents[trialIdx] = state.expertEvents[trialIdx].filter(e => !(e.onset <= at && at <= e.offset));
        } else {
            // In Pass 1, only remove overlapping events of a DIFFERENT class covering `at`; same-class overlapping events merge!
            state.expertEvents[trialIdx] = state.expertEvents[trialIdx].filter(e => !(e.cls !== cls && e.onset <= at && at <= e.offset));
        }

        // Add refined event and merge any overlapping events of the same class
        state.expertEvents[trialIdx].push({ onset, offset, cls, fromPrelabel: false });
        mergeOverlappingEvents(trialIdx);

        if (state.pass === PASS_ONE && state.scanWins[trialIdx]) {
            let [f, e] = gridWindow(trialIdx, state.winIdx);
            if (state.fromSample <= f && state.toSample >= e) {
                f = Math.max(0, state.fromSample);
                e = Math.min(trialLenOf(trialIdx) - 1, state.toSample);
            }
            f = Math.min(f, onset);
            e = Math.max(e, offset);
            if (!state.scanWins[trialIdx].some(r => r[0] === f && r[1] === e)) {
                state.scanWins[trialIdx].push([f, e]);
            }
        }

        syncRenderedLabels(trialIdx);
        recomputeReviewed(trialIdx, true);
        return true;
    }

    // Eraser: removes every event (saccade or blink), point mark and automatic pre-label that overlaps [a, b]
    function eraseRange(trialIdx, a, b) {
        saveUndo(trialIdx);
        if (a > b) { const x = a; a = b; b = x; }
        state.pointMarks[trialIdx] = state.pointMarks[trialIdx].filter(m => m.at < a || m.at > b);
        state.expertEvents[trialIdx] = state.expertEvents[trialIdx].filter(e => e.offset < a || e.onset > b);
        state.prelabels[trialIdx] = state.prelabels[trialIdx].filter(p => p.offset < a || p.onset > b);
        syncRenderedLabels(trialIdx);
        recomputeReviewed(trialIdx, true);
        if (state.pass === PASS_TWO) {
            buildRefineQueue();
            state.refineIdx = Math.min(state.refineIdx, state.refineQueue.length);
            focusCurrentMark();
        }
        updateStatus();
        render();
    }

    function deleteMarkAt(trialIdx, at) {
        saveUndo(trialIdx);
        state.pointMarks[trialIdx] = state.pointMarks[trialIdx].filter(m => Math.abs(m.at - at) > 25);
        state.expertEvents[trialIdx] = state.expertEvents[trialIdx].filter(e => !(e.onset <= at && at <= e.offset));
        if (state.pass === PASS_TWO && state.prelabels[trialIdx]) {
            state.prelabels[trialIdx] = state.prelabels[trialIdx].filter(p => !(p.onset <= at && at <= p.offset) && Math.abs(p.onset - at) > 25);
        }
        syncRenderedLabels(trialIdx);
        recomputeReviewed(trialIdx, true);
    }

    function syncRenderedLabels(trialIdx) {
        const lab = state.labels[trialIdx];
        if (!lab) return;
        lab.fill(0);

        // Fill prelabels on unreviewed parts
        for (const p of state.prelabels[trialIdx]) {
            for (let i = p.onset; i <= p.offset && i < lab.length; i++) {
                lab[i] = p.cls;
            }
        }

        // Fill expert events (always takes precedence)
        for (const e of state.expertEvents[trialIdx]) {
            for (let i = e.onset; i <= e.offset && i < lab.length; i++) {
                lab[i] = e.cls;
            }
        }
    }

    function totalReviewedSeconds() {
        let totalSamples = 0;
        for (const ranges of state.reviewedRanges) {
            for (const [s, e] of ranges) {
                totalSamples += (e - s + 1);
            }
        }
        return totalSamples / state.rateHz;
    }

    function totalExpertEventCounts() {
        let saccades = 0, blinks = 0;
        for (const evs of state.expertEvents) {
            for (const e of evs) {
                if (e.cls === CLS_SACCADE) saccades++;
                else if (e.cls === CLS_BLINK) blinks++;
            }
        }
        return { saccades, blinks };
    }

    // A selection being dragged (or a lost pointerup) must never survive a change of window, trial or pass
    function resetGesture() {
        state.pointers.clear();
        state.pointerDown = false;
        state.isDraggingSpan = false;
        state.handleDrag = null;
        state.dragStartSample = -1;
        state.dragCurrentSample = -1;
    }

    // --- Pass Switching & Navigation ---
    function setPass(newPass) {
        resetGesture();
        const leavingPassTwo = state.pass === PASS_TWO && newPass === PASS_ONE;
        if (state.pass === PASS_ONE && newPass === PASS_TWO) scanCurrentWindow();     // the window on screen has been seen
        state.pass = newPass;
        state.zoomMultiplier = 1.0;
        if (newPass === PASS_ONE) {
            state.currentMarkAt = -1;
            state.currentPutativeOnset = -1;
            state.currentPutativeOffset = -1;
            elements.passIndicator.textContent = "Pass 1: Mark";
            elements.passIndicator.className = "badge pass-badge";
            elements.pass1Controls.classList.remove('hidden');
            elements.pass2Controls.classList.add('hidden');
            if (elements.classBtnGroup) elements.classBtnGroup.classList.remove('hidden');
            if (elements.classToggle) elements.classToggle.classList.remove('hidden');
            if (elements.trialNavGroup) elements.trialNavGroup.classList.remove('hidden');

            state.winIdx = Math.floor(Math.max(0, state.fromSample) / winStep());
            applyGridWindow();
        } else {
            elements.passIndicator.textContent = "Pass 2: Refine (Onset / Offset)";
            elements.passIndicator.className = "badge pass-badge pass-two";
            elements.pass1Controls.classList.add('hidden');
            elements.pass2Controls.classList.remove('hidden');
            if (elements.classBtnGroup) elements.classBtnGroup.classList.add('hidden');
            if (elements.classToggle) elements.classToggle.classList.add('hidden');
            if (elements.trialNavGroup) elements.trialNavGroup.classList.add('hidden');

            buildRefineQueue();
            state.refineIdx = 0;
            focusCurrentMark();
        }
        updateStatus();
        render();
        requestAnimationFrame(() => {
            syncCanvasDimensions();
            render();
        });
        if (leavingPassTwo) maybeAutoTrain();
    }

    // Pass 2 shows all events (both pending point marks and refined/accepted expert events) that lie in windows displayed in pass 1
    function trialQueue(t) {
        const items = [];
        const exps = state.expertEvents[t] || [];
        for (let i = 0; i < exps.length; i++) {
            const e = exps[i];
            const mid = Math.round((e.onset + e.offset) / 2);
            if (inScanned(t, mid) || inScanned(t, e.onset) || inScanned(t, e.offset)) {
                items.push({ at: mid, onset: e.onset, offset: e.offset, cls: e.cls, isExpert: true });
            }
        }
        const pms = state.pointMarks[t] || [];
        for (const m of pms) {
            if (!inScanned(t, m.at)) continue;
            if (exps.some(e => e.onset <= m.at && m.at <= e.offset)) continue;
            items.push({ at: m.at, cls: m.cls, isExpert: false });
        }
        items.sort((a, b) => a.at - b.at);
        return items;
    }
    function buildRefineQueue() { state.refineQueue = trialQueue(state.currentTrialIdx); }
    function findTrialWithEvents(from, dir) {
        for (let t = from + dir; t >= 0 && t < state.trials.length; t += dir) if (trialQueue(t).length > 0) return t;
        return -1;
    }
    function enterTrialForRefine(t, atEnd) {
        resetGesture();
        state.currentTrialIdx = t;
        state.zoomMultiplier = 1.0;
        buildRefineQueue();
        state.refineIdx = atEnd ? Math.max(0, state.refineQueue.length - 1) : 0;
    }

    /**
     * Pass 2: Always center the window on one labeled event at a time, zoomed in time.
     * Window: centered on [putativeOnset, putativeOffset] with ±50 ms padding around the event.
     */
    function focusCurrentMark() {
        const trialIdx = state.currentTrialIdx;
        if (state.refineQueue.length === 0) {
            const nt = findTrialWithEvents(trialIdx, 1);
            if (nt >= 0) { enterTrialForRefine(nt, false); focusCurrentMark(); return; }
            const pt = findTrialWithEvents(trialIdx, -1);
            if (pt >= 0) { enterTrialForRefine(pt, true); focusCurrentMark(); return; }
            state.currentMarkAt = -1;
            state.currentPutativeOnset = -1;
            state.currentPutativeOffset = -1;
            if (elements.eventCounterBadge) {
                elements.eventCounterBadge.textContent = "No Events";
            }
            return;
        }

        if (state.refineIdx < 0) state.refineIdx = 0;
        if (state.refineIdx >= state.refineQueue.length) {
            // end of this trial: the next event is the first one of the next trial that has some
            const nt = findTrialWithEvents(trialIdx, 1);
            if (nt >= 0) { enterTrialForRefine(nt, false); focusCurrentMark(); return; }
            state.refineIdx = state.refineQueue.length - 1;
            state.currentMarkAt = -1;
            if (elements.eventCounterBadge) {
                elements.eventCounterBadge.textContent = "All Reviewed ✓";
            }
            updateStatus();
            render();
            return;
        }

        const mark = state.refineQueue[state.refineIdx];
        state.currentMarkAt = mark.at;
        state.activeClass = mark.cls;

        // Find putative onset and offset:
        let putativeOnset = mark.onset !== undefined ? mark.onset : mark.at;
        let putativeOffset = mark.offset !== undefined ? mark.offset : mark.at;

        if (!mark.isExpert) {
            // 1) Check existing expert event
            const exp = state.expertEvents[trialIdx].find(e => 
                (e.onset <= mark.at && mark.at <= e.offset) || 
                Math.abs(e.onset - mark.at) <= 30 || 
                Math.abs(e.offset - mark.at) <= 30
            );

            // 2) Check prelabel covering or closest to mark.at
            const pre = state.prelabels[trialIdx].find(p => 
                (p.onset <= mark.at && mark.at <= p.offset) || 
                Math.abs(p.onset - mark.at) <= 40 || 
                Math.abs(p.offset - mark.at) <= 40
            );

            if (exp) {
                putativeOnset = exp.onset;
                putativeOffset = exp.offset;
                state.activeClass = exp.cls;
            } else if (pre) {
                putativeOnset = pre.onset;
                putativeOffset = pre.offset;      // the class stays the one of the expert's mark
            } else {
                const halfDef = Math.max(5, Math.round(15.0 * (state.rateHz / 1000.0)));
                putativeOnset = Math.max(0, mark.at - halfDef);
                putativeOffset = Math.min(curLen() - 1, mark.at + halfDef);
            }
        }

        state.currentPutativeOnset = putativeOnset;
        state.currentPutativeOffset = putativeOffset;

        // Zoom in time and center on the event: ±50 ms padding around [putativeOnset, putativeOffset]
        const mult = state.zoomMultiplier || 1.0;
        const pad50ms = Math.round(50.0 * (state.rateHz / 1000.0) * mult);
        const evCenter = Math.round((putativeOnset + putativeOffset) / 2);
        const halfSpan = Math.max(pad50ms, Math.round((putativeOffset - putativeOnset) / 2) + pad50ms);

        state.fromSample = Math.max(0, evCenter - halfSpan);
        state.toSample = Math.min(curLen() - 1, evCenter + halfSpan);

        if (elements.eventCounterBadge) {
            const clsName = state.activeClass === CLS_SACCADE ? "Saccade" : "Blink";
            elements.eventCounterBadge.textContent = `Event ${state.refineIdx + 1}/${state.refineQueue.length} (${clsName})`;
        }
    }

    // Pass 1: next window of the same trial (10 % overlap), then the first window of the next trial
    function nextEpoch() {
        resetGesture();
        scanCurrentWindow();
        const t = state.currentTrialIdx;
        const n = trialLenOf(t);
        const w = winSamples();
        const step = winStep();
        if (state.fromSample + w >= n) {
            if (t < state.trials.length - 1) {
                state.currentTrialIdx++;
                state.winIdx = 0;
                applyGridWindow();
            }
        } else {
            const nextFrom = Math.min(state.fromSample + step, Math.max(0, n - w));
            state.fromSample = nextFrom;
            state.toSample = Math.min(n - 1, nextFrom + w - 1);
            state.winIdx = Math.floor(state.fromSample / step);
        }
        state.zoomMultiplier = 1.0;
        updateStatus();
        render();
    }

    function prevEpoch() {
        resetGesture();
        scanCurrentWindow();
        const t = state.currentTrialIdx;
        const n = trialLenOf(t);
        const w = winSamples();
        const step = winStep();
        if (state.fromSample <= 0) {
            if (t > 0) {
                state.currentTrialIdx--;
                state.winIdx = Math.max(0, numWindows(state.currentTrialIdx) - 1);
                applyGridWindow();
            } else {
                return;
            }
        } else {
            const prevFrom = Math.max(0, state.fromSample - step);
            state.fromSample = prevFrom;
            state.toSample = Math.min(n - 1, prevFrom + w - 1);
            state.winIdx = Math.floor(state.fromSample / step);
        }
        updateStatus();
        render();
    }

    function setWinMs(ms) {
        if (!(ms >= 20)) return;
        state.winMs = ms;
        state.winIdx = Math.floor(Math.max(0, state.fromSample) / winStep());
        try { localStorage.setItem('uneye_winMs', String(ms)); } catch (e) { }
        if (elements.winMsInput) elements.winMsInput.value = String(ms);
        if (state.pass === PASS_ONE) { applyGridWindow(); render(); }
        updateStatus();
    }

    // --- High-DPI Canvas Rendering Engine ---
    function syncCanvasDimensions() {
        const dpr = window.devicePixelRatio || 1;
        const rect = elements.canvasContainer.getBoundingClientRect();
        const targetW = Math.max(1, Math.floor(rect.width * dpr));
        const targetH = Math.max(1, Math.floor(rect.height * dpr));

        if (elements.canvas.width !== targetW || elements.canvas.height !== targetH) {
            elements.canvas.width = targetW;
            elements.canvas.height = targetH;
        }
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        return { width: rect.width, height: rect.height };
    }

    function resizeCanvas() {
        syncCanvasDimensions();
        render();
    }

    window.addEventListener('resize', resizeCanvas);
    if (window.ResizeObserver) {
        const ro = new ResizeObserver(() => {
            syncCanvasDimensions();
            render();
        });
        ro.observe(elements.canvasContainer);
    }

    function sampleToPx(s, width) {
        if (state.toSample <= state.fromSample) return 0;
        return ((s - state.fromSample) / (state.toSample - state.fromSample)) * width;
    }

    function pxToSample(px, width) {
        if (width <= 0) return state.fromSample;
        const f = Math.max(0, Math.min(1, px / width));
        return Math.round(state.fromSample + f * (state.toSample - state.fromSample));
    }

    function render() {
        const { width, height } = syncCanvasDimensions();
        if (width <= 0 || height <= 0) return;

        // Clear entire viewport cleanly
        ctx.clearRect(0, 0, width, height);

        const tr = state.trials[state.currentTrialIdx];
        if (!tr) {
            drawEmptyState(width, height);
            return;
        }

        ensureDerived(state.currentTrialIdx);

        // Layout lanes (2 lanes: position X and Y on the same axes, then speed) with top margin in Pass 2 so header never collides with the trace
        const topMargin = state.pass === PASS_TWO ? 26 : 0;
        const availH = height - topMargin;
        const posLane = { top: topMargin, bottom: topMargin + availH * 0.66 };
        const speedLane = { top: posLane.bottom, bottom: height };

        // 1. Draw Grid & Lane Dividers
        ctx.strokeStyle = '#ECEFF1';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, posLane.bottom); ctx.lineTo(width, posLane.bottom);
        ctx.stroke();

        // 2. Draw Label Bands & Reviewed Regions in Background
        drawLabelBands(width, height);

        // 3. Draw Position X & Y Traces (Smooth if filter enabled)
        const traceX = (state.filterSettings.enabled && tr.smoothX) ? tr.smoothX : tr.x;
        const traceY = (state.filterSettings.enabled && tr.smoothY) ? tr.smoothY : tr.y;
        drawPositionLane(traceX, traceY, tr.valid, posLane, width);

        // 4. Draw Velocity Lane
        drawVelocity(tr, speedLane, '#7E57C2', 'Velocity (°/s)', width);

        // 5. Draw Prelabels & Annotations
        drawAnnotations(width, height);

    }

    function drawEmptyState(w, h) {
        ctx.fillStyle = '#78909C';
        ctx.font = '600 18px var(--font-stack)';
        ctx.textAlign = 'center';
        ctx.fillText("No trace loaded — Tap '⚡ Demo' or 'Open…'", w / 2, h / 2);
    }

    function drawLabelBands(w, h) {
        const lab = state.labels[state.currentTrialIdx];
        if (!lab) return;

        const from = state.fromSample;
        const to = state.toSample;

        // Reviewed range indicators (top hairline)
        for (const [rStart, rEnd] of state.reviewedRanges[state.currentTrialIdx]) {
            if (rEnd < from || rStart > to) continue;
            const x1 = sampleToPx(Math.max(from, rStart), w);
            const x2 = sampleToPx(Math.min(to, rEnd), w);
            ctx.fillStyle = 'rgba(46, 125, 50, 0.08)';
            ctx.fillRect(x1, 0, x2 - x1, h);
            ctx.fillStyle = '#2E7D32';
            ctx.fillRect(x1, 0, x2 - x1, 4); // Green bar at top
        }

        // Sample labels highlight
        let spanStart = -1, spanCls = 0;
        for (let s = from; s <= to + 1; s++) {
            const c = s <= to ? lab[s] : 0;
            if (c !== spanCls) {
                if (spanCls !== 0 && spanStart !== -1) {
                    const x1 = sampleToPx(spanStart, w);
                    const x2 = sampleToPx(s - 1, w);
                    ctx.fillStyle = spanCls === CLS_SACCADE ? 'rgba(21, 101, 192, 0.16)' : 'rgba(198, 40, 40, 0.18)';
                    ctx.fillRect(x1, 0, Math.max(2, x2 - x1), h);
                }
                spanStart = s;
                spanCls = c;
            }
        }
    }

    // X (blue) and Y (green) on the same axes, zoomed as much as possible on the extreme values of the visible window.
    // The scale is never tighter than 1.25 degrees so that the 1-degree scale bar always fits.
    function drawPositionLane(arrX, arrY, valid, lane, w) {
        const top = lane.top + 20;
        const bottom = lane.bottom - 12;
        const h = bottom - top;

        const from = state.fromSample;
        const to = state.toSample;

        let minV = Infinity, maxV = -Infinity;
        for (const arr of [arrX, arrY]) {
            for (let i = from; i <= to && i < arr.length; i++) {
                if (valid[i] && !isNaN(arr[i])) {
                    if (arr[i] < minV) minV = arr[i];
                    if (arr[i] > maxV) maxV = arr[i];
                }
            }
        }
        if (minV === Infinity) { minV = -5.0; maxV = 5.0; }
        const oneDeg = 1 / (state.degPerUnit || 1);
        const pad = Math.max(0.02 * oneDeg, (maxV - minV) * 0.04);
        minV -= pad;
        maxV += pad;
        if (maxV - minV < 1.25 * oneDeg) { const c = (minV + maxV) / 2; minV = c - 0.625 * oneDeg; maxV = c + 0.625 * oneDeg; }

        ctx.font = '500 13px var(--font-stack)';
        ctx.textAlign = 'left';
        ctx.fillStyle = '#1E88E5';
        ctx.fillText('X', 16, lane.top + 24);
        ctx.fillStyle = '#43A047';
        ctx.fillText('Y', 32, lane.top + 24);
        ctx.fillStyle = '#78909C';
        ctx.fillText('Position (°)', 52, lane.top + 24);
        ctx.textAlign = 'right';
        ctx.fillText(`${((maxV - minV) * state.degPerUnit).toFixed(1)}°`, w - 16, lane.top + 24);

        // 1-degree scale bar at the right edge: tells a microsaccade from a large saccade whatever the automatic zoom
        const barH = (oneDeg / (maxV - minV)) * h;
        ctx.strokeStyle = '#000000';
        ctx.fillStyle = '#000000';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(w - 10, bottom); ctx.lineTo(w - 10, bottom - barH);
        ctx.stroke();
        ctx.font = '600 12px var(--font-stack)';
        ctx.textAlign = 'right';
        ctx.fillText('1°', w - 17, bottom - barH / 2 + 4);

        // Line traces
        ctx.lineWidth = 2.0;
        for (const [arr, color] of [[arrX, '#1E88E5'], [arrY, '#43A047']]) {
            ctx.strokeStyle = color;
            ctx.beginPath();
            const yOf = (v) => bottom - ((v - minV) / (maxV - minV)) * h;
            if (to - from + 1 > 2 * w) {
                // more samples than pixels: one min-max pair per pixel column (gaps still break the line)
                let started = false, col = NaN, lo = 0, hi = 0;
                const flush = () => {
                    const px = col + 0.5;
                    if (!started) { ctx.moveTo(px, yOf(lo)); started = true; } else ctx.lineTo(px, yOf(lo));
                    if (hi !== lo) ctx.lineTo(px, yOf(hi));
                };
                for (let i = from; i <= to && i < arr.length; i++) {
                    if (!valid[i] || isNaN(arr[i])) {
                        if (col === col) { flush(); col = NaN; }
                        started = false;
                        continue;
                    }
                    const c = Math.floor(sampleToPx(i, w));
                    if (c !== col) {
                        if (col === col) flush();
                        col = c; lo = hi = arr[i];
                    } else {
                        if (arr[i] < lo) lo = arr[i];
                        if (arr[i] > hi) hi = arr[i];
                    }
                }
                if (col === col) flush();
            } else {
                let started = false;
                for (let i = from; i <= to && i < arr.length; i++) {
                    if (!valid[i] || isNaN(arr[i])) { started = false; continue; }
                    const px = sampleToPx(i, w);
                    if (!started) { ctx.moveTo(px, yOf(arr[i])); started = true; } else { ctx.lineTo(px, yOf(arr[i])); }
                }
            }
            ctx.stroke();
        }

        // Invalid / blink gap markers
        ctx.fillStyle = 'rgba(198, 40, 40, 0.25)';
        let invalidStart = -1;
        for (let i = from; i <= to + 1; i++) {
            const isInv = i <= to && (!valid[i] || isNaN(arrX[i]) || isNaN(arrY[i]));
            if (isInv && invalidStart === -1) {
                invalidStart = i;
            } else if (!isInv && invalidStart !== -1) {
                const x1 = sampleToPx(invalidStart, w);
                const x2 = sampleToPx(i - 1, w);
                ctx.fillRect(x1, lane.top, Math.max(2, x2 - x1), lane.bottom - lane.top);
                invalidStart = -1;
            }
        }
    }

    function drawVelocity(tr, lane, color, title, w) {
        const top = lane.top + 20;
        const bottom = lane.bottom - 12;
        const h = bottom - top;
        const from = state.fromSample;
        const to = state.toSample;

        const speedArr = tr.speed;
        const speedAt = (i) => (speedArr && speedArr.length > i) ? speedArr[i] : 0.0;
        const last = Math.min(to, tr.x.length - 1);
        let maxSpeed = 50.0;
        for (let i = from; i <= last; i++) {
            const s = speedAt(i);
            if (s > maxSpeed) maxSpeed = s;
        }
        maxSpeed = Math.max(100.0, maxSpeed * 1.15);

        ctx.fillStyle = '#78909C';
        ctx.font = '500 13px var(--font-stack)';
        ctx.textAlign = 'left';
        ctx.fillText(title, 16, lane.top + 24);
        ctx.textAlign = 'right';
        ctx.fillText(`${Math.round(maxSpeed)} °/s`, w - 16, lane.top + 24);

        // Draw Adaptive Saccade Threshold Line if active
        if (tr.currentThreshold && tr.currentThreshold > 0 && tr.currentThreshold <= maxSpeed) {
            const threshY = bottom - (tr.currentThreshold / maxSpeed) * h;
            ctx.save();
            ctx.strokeStyle = 'rgba(211, 47, 47, 0.65)';
            ctx.lineWidth = 1.2;
            ctx.setLineDash([5, 4]);
            ctx.beginPath();
            ctx.moveTo(16, threshY);
            ctx.lineTo(w - 16, threshY);
            ctx.stroke();
            ctx.fillStyle = '#D32F2F';
            ctx.font = '500 11px var(--font-stack)';
            ctx.textAlign = 'right';
            ctx.fillText(`Threshold: ${tr.currentThreshold.toFixed(1)}°/s (${state.filterSettings.lambda}σ)`, w - 16, threshY - 4);
            ctx.restore();
        }

        ctx.strokeStyle = color;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        const yOf = (s) => bottom - (s / maxSpeed) * h;
        if (last - from + 1 > 2 * w) {
            // more samples than pixels: one min-max pair per pixel column keeps every peak with a fraction of the segments
            let col = NaN, lo = 0, hi = 0, started = false;
            const flush = () => {
                const px = col + 0.5;
                if (!started) { ctx.moveTo(px, yOf(lo)); started = true; } else ctx.lineTo(px, yOf(lo));
                if (hi !== lo) ctx.lineTo(px, yOf(hi));
            };
            for (let i = from; i <= last; i++) {
                const s = speedAt(i);
                const c = Math.floor(sampleToPx(i, w));
                if (c !== col) {
                    if (col === col) flush();
                    col = c; lo = hi = s;
                } else {
                    if (s < lo) lo = s;
                    if (s > hi) hi = s;
                }
            }
            if (col === col) flush();
        } else {
            let started = false;
            for (let i = from; i <= last; i++) {
                const px = sampleToPx(i, w);
                const py = yOf(speedAt(i));
                if (!started) { ctx.moveTo(px, py); started = true; }
                else { ctx.lineTo(px, py); }
            }
        }
        ctx.stroke();
    }

    function drawAnnotations(w, h) {
        const trialIdx = state.currentTrialIdx;

        // 1. Prelabels (Dimmed brackets)
        for (const p of state.prelabels[trialIdx]) {
            if (p.offset < state.fromSample || p.onset > state.toSample) continue;
            const x1 = sampleToPx(p.onset, w);
            const x2 = sampleToPx(p.offset, w);
            ctx.strokeStyle = p.cls === CLS_SACCADE ? 'rgba(21, 101, 192, 0.45)' : 'rgba(198, 40, 40, 0.45)';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(x1, 10, Math.max(2, x2 - x1), h - 20);
            ctx.setLineDash([]);
        }

        // 2. Point marks (Pass 1 - Triangles)
        for (const m of state.pointMarks[trialIdx]) {
            if (m.at < state.fromSample || m.at > state.toSample) continue;
            const px = sampleToPx(m.at, w);
            ctx.fillStyle = m.cls === CLS_SACCADE ? '#1565C0' : '#C62828';
            ctx.beginPath();
            ctx.moveTo(px - 8, 4);
            ctx.lineTo(px + 8, 4);
            ctx.lineTo(px, 18);
            ctx.closePath();
            ctx.fill();
        }

        // 3. Expert events (Pass 2 - Solid brackets with handles)
        for (const e of state.expertEvents[trialIdx]) {
            const hd = state.handleDrag && state.handleDrag.ev === e ? state.handleDrag : e;     // live bounds while a handle is dragged
            if (hd.offset < state.fromSample || hd.onset > state.toSample) continue;
            const x1 = sampleToPx(hd.onset, w);
            const x2 = sampleToPx(hd.offset, w);
            const color = e.cls === CLS_SACCADE ? '#1565C0' : '#C62828';
            ctx.strokeStyle = color;
            ctx.lineWidth = 3.5;
            ctx.strokeRect(x1, 8, Math.max(4, x2 - x1), h - 16);

            // Circular handles on boundaries
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(x1, h / 2, HANDLE_R_PX, 0, 2 * Math.PI); ctx.fill();
            ctx.beginPath(); ctx.arc(x2, h / 2, HANDLE_R_PX, 0, 2 * Math.PI); ctx.fill();
        }

        // 4. Live Drag Preview
        if (state.isDraggingSpan && state.dragStartSample >= 0 && state.dragCurrentSample >= 0) {
            const onset = Math.min(state.dragStartSample, state.dragCurrentSample);
            const offset = Math.max(state.dragStartSample, state.dragCurrentSample);
            const x1 = sampleToPx(onset, w);
            const x2 = sampleToPx(offset, w);
            const color = state.activeClass === CLS_SACCADE ? '#1565C0' : '#C62828';

            ctx.fillStyle = state.eraser ? 'rgba(0, 0, 0, 0.22)' : (state.activeClass === CLS_SACCADE ? 'rgba(21, 101, 192, 0.22)' : 'rgba(198, 40, 40, 0.22)');
            ctx.fillRect(x1, 0, Math.max(2, x2 - x1), h);

            ctx.strokeStyle = state.eraser ? '#000000' : color;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(x1, 6, Math.max(4, x2 - x1), h - 12);
        }

        // 5. In Pass 2: Draw target frame & header banner for the active event
        if (state.pass === PASS_TWO && state.currentMarkAt >= 0) {
            const onset = state.currentPutativeOnset >= 0 ? state.currentPutativeOnset : state.currentMarkAt;
            const offset = state.currentPutativeOffset >= 0 ? state.currentPutativeOffset : state.currentMarkAt;
            const x1 = sampleToPx(onset, w);
            const x2 = sampleToPx(offset, w);
            const color = state.activeClass === CLS_SACCADE ? '#1565C0' : '#C62828';

            // Top banner info bar
            ctx.fillStyle = state.activeClass === CLS_SACCADE ? 'rgba(21, 101, 192, 0.92)' : 'rgba(198, 40, 40, 0.92)';
            ctx.fillRect(0, 0, w, 24);
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '600 12px var(--font-stack)';
            ctx.textAlign = 'center';
            const clsName = state.activeClass === CLS_SACCADE ? 'Saccade' : 'Blink';
            const durMs = Math.round((offset - onset) * (1000.0 / state.rateHz));
            const bannerText = w < 520
                ? `Event ${state.refineIdx + 1}/${state.refineQueue.length} (${clsName}, ${durMs} ms) [${onset}..${offset}]`
                : `Pass 2: Event ${state.refineIdx + 1}/${state.refineQueue.length} (${clsName}) — Putative: [${onset}..${offset}] (${durMs} ms) — Window: [-50ms .. +50ms]`;
            ctx.fillText(bannerText, w / 2, 17);

            // Shaded region between putative onset & offset
            ctx.fillStyle = state.activeClass === CLS_SACCADE ? 'rgba(21, 101, 192, 0.12)' : 'rgba(198, 40, 40, 0.12)';
            ctx.fillRect(x1, 24, Math.max(3, x2 - x1), h - 24);

            // Left/right dashed boundary guides
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(x1, 24); ctx.lineTo(x1, h);
            ctx.moveTo(x2, 24); ctx.lineTo(x2, h);
            ctx.stroke();
            ctx.setLineDash([]);

            // its boundary handles (draggable)
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(x1, h / 2, HANDLE_R_PX, 0, 2 * Math.PI); ctx.fill();
            ctx.beginPath(); ctx.arc(x2, h / 2, HANDLE_R_PX, 0, 2 * Math.PI); ctx.fill();
        }
    }

    // --- Touch & Pointer Event Handlers (iPad & Desktop) ---
    function setupPointerEvents() {
        const canvas = elements.canvas;
        // position in the canvas from clientX/Y (the offset of synthetic events is wrong in the Android WebView)
        // the rect is measured once per gesture (pointerdown), not on every move event
        let canvasRect = null;
        const local = (e) => {
            const r = canvasRect || (canvasRect = canvas.getBoundingClientRect());
            return { x: e.clientX - r.left, y: e.clientY - r.top };
        };
        // pointermove fires faster than the screen refreshes: draw at most once per frame
        let renderQueued = false;
        const requestRender = () => {
            if (renderQueued) return;
            renderQueued = true;
            requestAnimationFrame(() => { renderQueued = false; render(); });
        };

        // boundary handle under the finger: the dots at mid-height on the onset / offset lines of the refined events,
        // and in pass 2 of the event under refinement. Dragging one moves that boundary only.
        const handleAt = (pos) => {
            const w = canvas.clientWidth, h = canvas.clientHeight;
            if (state.eraser || Math.abs(pos.y - h / 2) > HANDLE_HIT_Y_PX) return null;
            let best = null, bestD = HANDLE_HIT_X_PX;
            const consider = (ev, onset, offset, strict) => {
                for (const side of ['on', 'off']) {
                    const d = Math.abs(sampleToPx(side === 'on' ? onset : offset, w) - pos.x);
                    if (strict ? d < bestD : d <= bestD) { bestD = d; best = { ev, side, onset, offset }; }
                }
            };
            const cur = state.pass === PASS_TWO && state.currentMarkAt >= 0 && state.currentPutativeOnset >= 0;
            if (cur) consider(null, state.currentPutativeOnset, state.currentPutativeOffset, true);
            for (const e of state.expertEvents[state.currentTrialIdx] || []) {      // a refined event wins over the proposal drawn on it
                if (e.offset >= state.fromSample && e.onset <= state.toSample) consider(e, e.onset, e.offset, false);
            }
            if (!best) return null;
            best.on0 = best.onset; best.off0 = best.offset;
            best.current = cur && best.onset === state.currentPutativeOnset && best.offset === state.currentPutativeOffset;
            return best;
        };

        canvas.addEventListener('pointerdown', (e) => {
            canvasRect = canvas.getBoundingClientRect();
            try { canvas.setPointerCapture(e.pointerId); } catch (err) { }
            if (e.isPrimary) state.pointers.clear();      // a primary pointer starts a new gesture: drop pointers whose pointerup never arrived
            if (state.pointers.size === 0) {
                state.gestureMulti = false;
                state.gestureIgnored = (performance.now() - state.lastMultiTouchEnd) < TOUCH_REFRACTORY_MS;   // refractory period after a pinch
            }
            state.pointers.set(e.pointerId, local(e));
            if (state.pointers.size >= 2) state.gestureMulti = true;

            if (state.pointers.size === 1) {
                // 1 Finger / Apple Pencil
                state.pointerDown = true;
                const sample = pxToSample(local(e).x, canvas.clientWidth);
                state.dragStartSample = sample;
                state.dragCurrentSample = sample;
                state.dragPointerPos = local(e);
                state.isDraggingSpan = false;
                state.handleDrag = state.gestureIgnored ? null : handleAt(local(e));
            } else if (state.pointers.size === 2) {
                // Start 2-Finger Pinch Zoom
                const pts = Array.from(state.pointers.values());
                state.initialPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
                state.initialPinchFrom = state.fromSample;
                state.initialPinchTo = state.toSample;
                state.isDraggingSpan = false;
                state.handleDrag = null;
            }
            render();
        });

        // move / up / cancel are listened to on the window: they still arrive when the pointer capture is lost
        window.addEventListener('pointermove', (e) => {
            if (!state.pointers.has(e.pointerId)) return;
            state.pointers.set(e.pointerId, local(e));

            if (state.pointers.size === 1 && state.pointerDown && !state.gestureMulti && !state.gestureIgnored) {
                state.dragPointerPos = local(e);
                const sample = pxToSample(local(e).x, canvas.clientWidth);
                state.dragCurrentSample = sample;

                const hd = state.handleDrag;
                if (hd) {
                    if (hd.side === 'on') hd.onset = Math.max(0, Math.min(sample, hd.offset - 1));
                    else hd.offset = Math.min(curLen() - 1, Math.max(sample, hd.onset + 1));
                    if (hd.current) { state.currentPutativeOnset = hd.onset; state.currentPutativeOffset = hd.offset; }
                } else if (!state.isDraggingSpan && Math.abs(sample - state.dragStartSample) > 8) {
                    // Threshold to enter drag mode vs tap
                    state.isDraggingSpan = true;
                }
                requestRender();
            } else if (state.pointers.size === 2) {
                // 2-Finger Pinch Zoom & Pan
                const pts = Array.from(state.pointers.values());
                const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
                if (state.initialPinchDist > 0) {
                    const scale = state.initialPinchDist / Math.max(10, dist);
                    const origSpan = state.initialPinchTo - state.initialPinchFrom;
                    const newSpan = Math.max(50, Math.min(curLen(), Math.round(origSpan * scale)));
                    const center = (state.initialPinchFrom + state.initialPinchTo) / 2;

                    state.fromSample = Math.max(0, Math.round(center - newSpan / 2));
                    state.toSample = Math.min(curLen() - 1, state.fromSample + newSpan);
                    requestRender();
                }
            }
        });

        window.addEventListener('pointerup', (e) => {
            if (!state.pointers.has(e.pointerId)) return;
            try { canvas.releasePointerCapture(e.pointerId); } catch (err) { }
            state.pointers.delete(e.pointerId);
            if (state.gestureMulti) state.lastMultiTouchEnd = performance.now();

            if (state.pointers.size === 0 && state.pointerDown) {
                state.pointerDown = false;
                try {
                    if (state.gestureMulti || state.gestureIgnored) {
                        // end of a pinch zoom (or a touch right after one): never a mark
                        state.isDraggingSpan = false;
                    } else if (state.handleDrag) {
                        commitHandleDrag(state.handleDrag);
                    } else if (state.eraser) {
                        const a = state.isDraggingSpan ? Math.min(state.dragStartSample, state.dragCurrentSample) : state.dragStartSample - 8;
                        const b = state.isDraggingSpan ? Math.max(state.dragStartSample, state.dragCurrentSample) : state.dragStartSample + 8;
                        eraseRange(state.currentTrialIdx, a, b);
                    } else if (!state.isDraggingSpan) {
                        // Tap Event
                        handleTap(state.dragStartSample);
                    } else {
                        // Completed Span Drag
                        const onset = Math.min(state.dragStartSample, state.dragCurrentSample);
                        const offset = Math.max(state.dragStartSample, state.dragCurrentSample);
                        if (offset - onset >= 5) {
                            const markAt = state.currentMarkAt >= 0 ? state.currentMarkAt : state.dragStartSample;
                            const prevLen = state.refineQueue.length;
                            refineMark(state.currentTrialIdx, markAt, onset, offset, state.activeClass);
                            if (state.pass === PASS_TWO) {
                                buildRefineQueue();
                                const newLen = state.refineQueue.length;
                                state.refineIdx = newLen < prevLen ? Math.min(state.refineIdx + 1, newLen) : state.refineIdx + 1;
                                state.zoomMultiplier = 1.0;
                                focusCurrentMark();
                            } else if (state.pass === PASS_ONE) {
                                slideWindowAfterEvent(onset);
                            }
                        }
                    }
                } finally {          // the selection never stays on screen, even if storing the event failed
                    state.isDraggingSpan = false;
                    state.handleDrag = null;
                    state.dragStartSample = -1;
                    state.dragCurrentSample = -1;
                    updateStatus();
                    render();
                }
            }
        });

        window.addEventListener('pointercancel', (e) => {
            if (!state.pointers.has(e.pointerId)) return;
            try { canvas.releasePointerCapture(e.pointerId); } catch (err) { }
            if (state.gestureMulti) state.lastMultiTouchEnd = performance.now();
            const hd = state.handleDrag;
            if (hd && hd.current) { state.currentPutativeOnset = hd.on0; state.currentPutativeOffset = hd.off0; }
            resetGesture();
            render();
        });

        // Mouse Wheel / Trackpad Pinch Zoom
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const span = state.toSample - state.fromSample;
            const factor = e.deltaY > 0 ? 1.15 : 0.85;
            const mouseSample = pxToSample(local(e).x, canvas.clientWidth);
            const frac = (mouseSample - state.fromSample) / Math.max(1, span);

            const newSpan = Math.max(50, Math.min(curLen(), Math.round(span * factor)));
            state.fromSample = Math.max(0, Math.round(mouseSample - frac * newSpan));
            state.toSample = Math.min(curLen() - 1, state.fromSample + newSpan);
            render();
        }, { passive: false });
    }

    // end of a boundary-handle drag: the event keeps its other boundary; in pass 2 the view stays on the same event (no skip, no re-zoom)
    function commitHandleDrag(hd) {
        const t = state.currentTrialIdx;
        if (hd.onset === hd.on0 && hd.offset === hd.off0) return;
        if (hd.ev) {
            saveUndo(t);
            const i = state.expertEvents[t].indexOf(hd.ev);
            if (i >= 0) state.expertEvents[t][i] = { onset: hd.onset, offset: hd.offset, cls: hd.ev.cls, fromPrelabel: false };
            mergeOverlappingEvents(t);
            syncRenderedLabels(t);
            recomputeReviewed(t, true);
        } else {
            refineMark(t, state.currentMarkAt, hd.onset, hd.offset, state.activeClass);      // the proposal under refinement becomes a refined event
        }
        if (state.pass !== PASS_TWO) return;
        const at = hd.current ? Math.round((hd.onset + hd.offset) / 2) : state.currentMarkAt;
        buildRefineQueue();
        const i = state.refineQueue.findIndex(q => q.isExpert ? q.onset <= at && at <= q.offset : q.at === at);
        if (i < 0) return;
        const q = state.refineQueue[i];
        state.refineIdx = i;
        state.currentMarkAt = q.at;
        if (q.isExpert) { state.currentPutativeOnset = q.onset; state.currentPutativeOffset = q.offset; }
        if (elements.eventCounterBadge) {
            elements.eventCounterBadge.textContent = `Event ${i + 1}/${state.refineQueue.length} (${state.activeClass === CLS_SACCADE ? "Saccade" : "Blink"})`;
        }
    }

    function slideWindowAfterEvent(eventOnset) {
        if (!state.autoSlide) return;
        if (state.pass !== PASS_ONE) return;
        const t = state.currentTrialIdx;
        const n = trialLenOf(t);
        const w = winSamples();
        const pad50 = Math.round(50.0 * state.rateHz / 1000);

        // Record current visible window before sliding
        scanCurrentWindow();

        let newFrom = Math.max(0, eventOnset - pad50);
        if (n <= w) {
            newFrom = 0;
        } else if (newFrom + w > n) {
            newFrom = Math.max(0, n - w);
        }
        const newTo = Math.min(n - 1, newFrom + w - 1);

        state.fromSample = newFrom;
        state.toSample = newTo;
        state.winIdx = Math.floor(newFrom / winStep());

        // Record new window as scanned
        scanCurrentWindow();
    }

    function handleTap(sample) {
        const now = Date.now();
        const isDoubleTap = (now - state.lastTapTime < 320) && Math.abs(sample - state.lastTapSample) <= 25;
        state.lastTapTime = now;
        state.lastTapSample = sample;

        if (isDoubleTap) {
            if (state.pass === PASS_TWO && state.currentMarkAt >= 0) {
                const onset = state.currentPutativeOnset >= 0 ? state.currentPutativeOnset : sample - 15;
                const offset = state.currentPutativeOffset >= 0 ? state.currentPutativeOffset : sample + 15;
                const prevLen = state.refineQueue.length;
                refineMark(state.currentTrialIdx, state.currentMarkAt, onset, offset, state.activeClass);
                buildRefineQueue();
                const newLen = state.refineQueue.length;
                state.refineIdx = newLen < prevLen ? Math.min(state.refineIdx + 1, newLen) : state.refineIdx + 1;
                state.zoomMultiplier = 1.0;
                focusCurrentMark();
                updateStatus();
                render();
                return;
            }
        }

        if (state.pass === PASS_ONE) {
            const trialIdx = state.currentTrialIdx;

            // 1. Tap on existing point mark removes it (tolerance up to 15 samples)
            const existingPmIdx = (state.pointMarks[trialIdx] || []).findIndex(m => Math.abs(m.at - sample) <= 15);
            if (existingPmIdx !== -1) {
                state.pointMarks[trialIdx].splice(existingPmIdx, 1);
                recomputeReviewed(trialIdx, false);
                updateStatus();
                render();
                return;
            }

            // 2. Tap on existing expert mark removes it
            const existingExpIdx = (state.expertEvents[trialIdx] || []).findIndex(e => (e.onset <= sample && sample <= e.offset) || Math.abs(e.onset - sample) <= 15 || Math.abs(e.offset - sample) <= 15);
            if (existingExpIdx !== -1) {
                deleteMarkAt(trialIdx, sample);
                updateStatus();
                render();
                return;
            }

            // 3. Tap on proposed event: accept whole proposed duration as ground truth, then slide.
            // The class is the one selected by the expert (Saccade / Blink buttons): a saccade the model proposed
            // as a blink (red) becomes a saccade when tapped with the Saccade tool.
            const pre = (state.prelabels[trialIdx] || []).find(p => (p.onset <= sample && sample <= p.offset) || Math.abs(p.onset - sample) <= 25 || Math.abs(p.offset - sample) <= 25);
            if (pre) {
                refineMark(trialIdx, sample, pre.onset, pre.offset, state.activeClass);
                slideWindowAfterEvent(pre.onset);
                updateStatus();
                render();
                requestAnimationFrame(() => {
                    syncCanvasDimensions();
                    render();
                });
                return;
            }

            // 4. Empty space: add point mark, then slide
            addPointMark(trialIdx, sample, state.activeClass);
            slideWindowAfterEvent(sample);
            updateStatus();
            render();
            requestAnimationFrame(() => {
                syncCanvasDimensions();
                render();
            });
            return;
        } else {
            // In pass 2: tap selects or places
            addPointMark(state.currentTrialIdx, sample, state.activeClass);
            buildRefineQueue();
            focusCurrentMark();
        }
        updateStatus();
        render();
        requestAnimationFrame(() => {
            syncCanvasDimensions();
            render();
        });
    }

    // --- UI Updates & Convergence Logic ---
    function updateStatus() {
        const nTrials = state.trials.length;
        elements.trialStatus.textContent = `Trial ${state.currentTrialIdx + 1}/${Math.max(1, nTrials)}${state.testTrials.has(state.currentTrialIdx) ? ' (Test)' : ''}${(state.trialNames && state.trialNames[state.currentTrialIdx]) ? ' · ' + state.trialNames[state.currentTrialIdx] : ''}${state.pass === PASS_ONE && nTrials ? ' · window ' + (state.winIdx + 1) + '/' + numWindows(state.currentTrialIdx) + (state.currentTrialIdx === nTrials - 1 && state.winIdx >= numWindows(state.currentTrialIdx) - 1 ? ' (end)' : '') : ''}`;
        elements.reviewedStatus.textContent = `Reviewed: ${totalReviewedSeconds().toFixed(1)} s`;

        const counts = totalExpertEventCounts();
        elements.eventsStatus.textContent = `${counts.saccades} saccades, ${counts.blinks} blinks`;

        // Update convergence badge
        updateConvergenceBadge();
    }

    function updateConvergenceBadge() {
        const counts = totalExpertEventCounts();
        const minSaccades = 100;
        const minBlinks = 20;

        if (counts.saccades < minSaccades || counts.blinks < minBlinks) {
            state.convergenceState = 'SILENT';
            elements.convergenceBadge.classList.add('hidden');
        } else {
            if (state.convergenceState === 'CONVERGED') {
                elements.convergenceBadge.className = 'badge conv-badge converged';
                elements.convergenceBadge.textContent = "Converged ✓";
                elements.convergenceBadge.classList.remove('hidden');
            } else {
                state.convergenceState = 'COLLECTING';
                elements.convergenceBadge.className = 'badge conv-badge';
                elements.convergenceBadge.textContent = `Collecting: ${counts.saccades}/${minSaccades} sac, ${counts.blinks}/${minBlinks} blk`;
                elements.convergenceBadge.classList.remove('hidden');
            }
        }
    }

    // --- Web Worker & On-Device Training (WASM SIMD) ---
    let trainWorker = null;
    function getTrainWorker() {
        if (!trainWorker) {
            trainWorker = new Worker('train_worker.js');
        }
        return trainWorker;
    }

    async function runTrainingRound() {
        if (state.isTraining) return;
        const reviewedS = totalReviewedSeconds();
        if (reviewedS < 0.5) {
            alert("Please label at least 0.5 seconds of reviewed data before training.");
            return;
        }

        state.isTraining = true;
        elements.trainStatus.className = 'train-active';
        elements.trainStatus.textContent = "Preparing training windows…";
        if (elements.cancelTrainBtn) elements.cancelTrainBtn.classList.remove('hidden');

        // Keep screen awake during training (iPad Safari & Chrome)
        try {
            if ('wakeLock' in navigator) {
                state.wakeLock = await navigator.wakeLock.request('screen');
            }
        } catch (e) {
            console.log("WakeLock:", e);
        }

        // Training set: reviewed stretches of NON-test trials only (point marks never train)
        const traces = [];
        const winSamples = Math.max(25, Math.round(state.rateHz / 25.0) * 25);
        for (let t = 0; t < state.trials.length; t++) {
            if (state.testTrials.has(t)) continue;
            const tr = state.trials[t];
            const rev = state.reviewedRanges[t];
            if (!rev || rev.length === 0) continue;
            let maxRev = 0;
            for (const [s, e] of rev) {
                if (e > maxRev) maxRev = e;
            }
            const endSample = Math.min(tr.x.length, Math.max(winSamples, maxRev + 1));
            const lab = new Uint8Array(endSample).fill(255);
            const hl = humanLabels(t); // 255 = ignore
            for (const [s, e] of rev) {
                for (let i = s; i <= e && i < endSample; i++) {
                    lab[i] = hl[i];
                }
            }
            // Training copy: invalid (NaN) stretches filled with noise at 3x
            // the local fixation noise, so a blink labelled 2 on a tracker
            // loss keeps its label and stays distinguishable from fixation.
            // The raw trace (display, prediction) is not modified.
            const tc = trainingCopyOf(tr, 1000 + t);
            traces.push({
                x: tc.x.subarray(0, endSample),
                y: tc.y.subarray(0, endSample),
                labels: lab,
                rateHz: state.rateHz
            });
        }

        // Held-out test trials for unbiased F1 & kappa: the same fill, so the
        // metrics measure the class-2 signal the network was trained on.
        const testTrials = [];
        for (const t of state.testTrials) {
            const tr = state.trials[t];
            const rev = state.reviewedRanges[t];
            if (!rev || rev.length === 0) continue;
            const lab = new Uint8Array(tr.x.length).fill(255); const hl = humanLabels(t);
            for (const [s, e] of rev) {
                for (let i = s; i <= e && i < lab.length; i++) {
                    lab[i] = hl[i];
                }
            }
            const tc = trainingCopyOf(tr, 2000 + t);
            testTrials.push({
                x: tc.x,
                y: tc.y,
                labels: lab,
                rateHz: state.rateHz
            });
        }

        if (traces.length === 0) {
            finishTraining("No non-test reviewed trials to train on");
            return;
        }

        // Pretrained weights or scratch
        let initBytes = null;
        if (state.initialWeightsMode === 'pretrained' && !state.trainedModelBin) {
            try {
                const resp = await fetch('models/combined.bin');
                if (resp.ok) {
                    const buf = await resp.arrayBuffer();
                    initBytes = new Uint8Array(buf);
                }
            } catch (e) {
                console.log("Could not load combined.bin:", e);
            }
        } else if (state.trainedModelBin) {
            initBytes = state.trainedModelBin;
        }

        const worker = getTrainWorker();
        const trainId = Date.now();

        let lastEpoch = 0;
        worker.onmessage = async function(evt) {
            const msg = evt.data;
            if (msg.id !== trainId) return;

            if (msg.type === 'PROGRESS') {
                const p = msg.data;
                lastEpoch = p.epoch;
                const etaStr = p.eta_s > 0 ? ` · ETA ${p.eta_s.toFixed(0)}s` : '';
                elements.trainStatus.className = 'train-active';
                elements.trainStatus.textContent = `Epoch ${p.epoch}/60 · Loss: ${p.train_loss.toFixed(3)} · Val F1: ${p.val_event_f1.toFixed(2)}${etaStr}`;
            } else if (msg.type === 'CANCELLED') {
                finishTraining("Training cancelled");
            } else if (msg.type === 'ERROR') {
                finishTraining("Training error: " + msg.message);
                alert("Training error: " + msg.message);
            } else if (msg.type === 'TRAIN_COMPLETE') {
                state.trainedModelBin = msg.binModel;
                state.trainedModelJson = msg.jsonSidecar;

                const counts = totalExpertEventCounts();
                const m = msg.metrics || {};
                const round = {
                    atMs: Date.now(),
                    labeledSeconds: reviewedS,
                    epochs: lastEpoch,
                    valF1: m.testF1Saccade || 0.0,
                    testF1Saccade: m.testF1Saccade || 0.0,
                    testF1Blink: m.testF1Blink || 0.0,
                    testKappaSaccade: m.testKappaSaccade || 0.0,
                    testKappaBlink: m.testKappaBlink || 0.0,
                    nSaccades: counts.saccades,
                    nBlinks: counts.blinks,
                };
                state.convergenceHistory.push(round);
                state.reviewedAtLastTraining = reviewedS;

                evaluateConvergenceFlag();
                finishTraining(`Trained (${round.epochs} ep — Test F1: ${round.testF1Saccade.toFixed(2)})`);

                // proposals from the new network only if the toggle is on (otherwise the HMM's stay)
                if (state.netSuggestions) await refreshSuggestions();
                saveToIndexedDB();
            }
        };

        function finishTraining(statusText) {
            state.isTraining = false;
            elements.trainStatus.className = 'train-done';
            elements.trainStatus.textContent = statusText;
            if (elements.cancelTrainBtn) elements.cancelTrainBtn.classList.add('hidden');
            if (state.wakeLock) {
                state.wakeLock.release().catch(() => {});
                state.wakeLock = null;
            }
            updateStatus();
        }

        worker.postMessage({
            type: 'START_TRAIN',
            id: trainId,
            data: {
                config: {
                    n_classes: 3,
                    max_seconds_of_data: 3600.0,        // all the reviewed labels (the budget is a total over the traces)
                    rotation_aug: true,
                    rotations: 4,
                    val_fraction: 0.2,
                    max_speed_deg_s: 1300.0,
                    max_epochs: 60,
                    patience: 4,
                    learning_rate: 1e-3,
                    seed: 1,
                    init_model_bytes: initBytes
                },
                traces,
                testTrials
            }
        });
    }

    async function applyModelAsPrelabeler() {
        if (!state.trainedModelBin || state.trials.length === 0) return;
        const worker = getTrainWorker();
        state.previousPrelabels = JSON.parse(JSON.stringify(state.prelabels));

        for (let t = 0; t < state.trials.length; t++) {
            // same NaN fill as the training copy, otherwise a blink reads as zero velocity and is never predicted
            const tr = trainingCopyOf(state.trials[t], 3000 + t);
            const predPromise = new Promise((resolve) => {
                const reqId = Date.now() + Math.random();
                const handler = (e) => {
                    if (e.data.id === reqId && e.data.type === 'PREDICT_RESULT') {
                        worker.removeEventListener('message', handler);
                        resolve(e.data.pred);
                    }
                };
                worker.addEventListener('message', handler);
                worker.postMessage({
                    type: 'PREDICT',
                    id: reqId,
                    data: { x: tr.x, y: tr.y, rateHz: state.rateHz, maxSpeedDegS: 1300.0 }
                });
            });

            const pred = await predPromise;
            if (!pred) continue;

            const newPre = [];
            const n = pred.length;
            for (let i = 0; i < n;) {
                const c = pred[i];
                if (c === 1 || c === 2) {
                    let j = i;
                    while (j < n && pred[j] === c) j++;
                    if (j - i >= 3) {
                        newPre.push({ onset: i, offset: j - 1, cls: c });
                    }
                    i = j;
                } else {
                    i++;
                }
            }
            state.prelabels[t] = newPre;
            syncRenderedLabels(t);
        }
        render();
    }

    // --- IndexedDB Persistence & Model Export ---
    const DB_NAME = 'uneye_db';
    const DB_VERSION = 1;
    const STORE_NAME = 'uneye_store';

    function openDB() {
        return new Promise((resolve, reject) => {
            if (!('indexedDB' in window)) return resolve(null);
            const req = indexedDB.open(DB_NAME, DB_VERSION);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    db.createObjectStore(STORE_NAME);
                }
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => resolve(null);
        });
    }

    // Autosave, debounced: one write per burst of edits. The traces go to their own key, written only when a new dataset is loaded,
    // as typed arrays (copying every trial as plain arrays on each label blocked the screen for > 1 s on a phone).
    let saveTimer = null, savedTrials = null;
    function saveToIndexedDB() {
        clearTimeout(saveTimer);
        saveTimer = setTimeout(flushSave, 800);
    }

    async function flushSave() {
        clearTimeout(saveTimer);
        saveTimer = null;
        try {
            const db = await openDB();
            if (!db) return;
            const tx = db.transaction(STORE_NAME, 'readwrite');
            const store = tx.objectStore(STORE_NAME);
            if (savedTrials !== state.trials) {
                store.put(state.trials.map(tr => ({ x: tr.x, y: tr.y, valid: tr.valid, name: tr.name })), 'autosave_trials');
                savedTrials = state.trials;
            }
            const payload = {
                nTrials: state.trials.length,
                labels: state.labels,
                reviewedRanges: state.reviewedRanges,
                scanWins: state.scanWins,
                winMs: state.winMs,
                expertEvents: state.expertEvents,
                pointMarks: state.pointMarks,
                prelabels: state.prelabels,
                prelabelerMode: state.prelabelerMode,
                rateHz: state.rateHz,
                scale: state.scale,
                convergenceHistory: state.convergenceHistory,
                trainedModelBin: state.trainedModelBin,
                trainedModelJson: state.trainedModelJson,
                modelVersion: MODEL_VERSION
            };
            store.put(payload, 'autosave');
        } catch (e) {
            console.log("IndexedDB save:", e);
        }
    }
    // a pending save is written before the page goes away (app switched, tab closed)
    document.addEventListener('visibilitychange', () => { if (document.hidden && saveTimer) flushSave(); });
    window.addEventListener('pagehide', () => { if (saveTimer) flushSave(); });

    async function restoreFromIndexedDB() {
        try {
            const db = await openDB();
            if (!db) return false;
            const tx = db.transaction(STORE_NAME, 'readonly');
            const store = tx.objectStore(STORE_NAME);
            const req = store.get('autosave');
            const reqTrials = store.get('autosave_trials');
            return await new Promise((resolve) => {
            req.onerror = reqTrials.onerror = () => resolve(false);
            reqTrials.onsuccess = () => {        // requests of one transaction complete in order: 'autosave' is ready too
                const data = req.result;
                const savedTr = (data && data.trials) || reqTrials.result;      // sessions saved before the split keep the traces in 'autosave'
                if (!data || !savedTr || savedTr.length === 0 || (data.nTrials && data.nTrials !== savedTr.length)) { resolve(false); return; }
                if (elements.startModal) elements.startModal.classList.add('hidden');
                state.trials = savedTr.map(tr => ({
                    x: new Float32Array(tr.x),
                    y: new Float32Array(tr.y),
                    name: tr.name || '',
                    valid: tr.valid ? new Uint8Array(tr.valid) : new Uint8Array(tr.x.length).fill(1)
                }));
                savedTrials = data.trials ? null : state.trials;     // an old-format session is rewritten in the new layout at the next save
                const n = state.trials.length;
                state.undoStack = Array.from({ length: n }, () => []);
                state.labels = data.labels.map(l => new Uint8Array(l));
                state.reviewedRanges = data.reviewedRanges || Array.from({ length: n }, () => []);
                state.expertEvents = data.expertEvents || Array.from({ length: n }, () => []);
                state.pointMarks = data.pointMarks || Array.from({ length: n }, () => []);
                state.scanWins = data.scanWins || state.reviewedRanges.map(r => r.map(x => x.slice()));
                state.currentTrialIdx = 0; state.winIdx = 0;
                applyGridWindow();
                state.prelabels = data.prelabels || Array.from({ length: n }, () => []);
                state.rateHz = data.rateHz || 1000.0;
                state.scale = data.scale || 1.0;
                state.prelabelerMode = data.prelabelerMode === 'none' ? 'none' : 'hmm';
                state.convergenceHistory = data.convergenceHistory || [];
                state.trainedModelBin = data.trainedModelBin || null;
                state.trainedModelJson = data.trainedModelJson || null;
                const staleModel = state.trainedModelBin && data.modelVersion !== MODEL_VERSION;
                if (staleModel) {          // trained by an earlier, broken version: start again from the pretrained model
                    state.trainedModelBin = null;
                    state.trainedModelJson = null;
                    state.convergenceHistory = [];
                    state.reviewedAtLastTraining = 0.0;
                }

                state.testTrials = new Set();
                for (let t = 0; t < n; t += 4) state.testTrials.add(t);

                for (let t = 0; t < n; t++) recomputeReviewed(t, false);
                preprocessAllTrials();
                if (state.trainedModelBin) {
                    const worker = getTrainWorker();
                    worker.postMessage({ type: 'LOAD_MODEL', data: state.trainedModelBin });
                }
                // proposals are recomputed from the current source (the saved ones may come from an older detector or from the network)
                refreshSuggestions().then(() => { if (staleModel) saveToIndexedDB(); });
                updateStatus();
                render();
                resolve(true);
            };
            });
        } catch (e) {
            console.log("IndexedDB restore:", e);
            return false;
        }
    }

    function downloadModel() {
        if (!state.trainedModelBin) {
            alert("No trained model available yet. Train a model first.");
            return;
        }
        downloadFile('uneye_model.bin', state.trainedModelBin, 'application/octet-stream');
        let sidecar = {};
        try {
            if (state.trainedModelJson) sidecar = JSON.parse(state.trainedModelJson);
        } catch (e) {}
        sidecar.classes = sidecar.classes || 3;
        sidecar.window = sidecar.window || 1000;
        sidecar.ks = 5;
        sidecar.mp = 5;
        sidecar.rate_hz = state.rateHz;
        sidecar.deg_per_unit = state.scale || 1.0;
        sidecar.max_speed_deg_s = 1300.0;
        sidecar.preprocessing = {
            filter: state.filterSettings.enabled ? state.filterSettings.filterType : "none",
            savgol_window_ms: state.filterSettings.enabled ? state.filterSettings.windowMs : 0,
            savgol_poly_order: state.filterSettings.polyOrder || 3,
            max_speed_deg_s: 1300.0
        };
        downloadFile('uneye_model.json', JSON.stringify(sidecar, null, 2), 'application/json');
    }

    function evaluateConvergenceFlag() {
        const counts = totalExpertEventCounts();
        if (counts.saccades < 100 || counts.blinks < 20) {
            state.convergenceState = 'SILENT';
            return;
        }

        const WINDOW = 3;
        const gains = [];
        for (let i = 1; i < state.convergenceHistory.length; i++) {
            gains.push(state.convergenceHistory[i].testF1Saccade - state.convergenceHistory[i - 1].testF1Saccade);
        }

        if (gains.length < WINDOW) {
            state.convergenceState = 'COLLECTING';
            return;
        }

        const windowGains = gains.slice(-WINDOW);
        const meanGain = windowGains.reduce((a, b) => a + b, 0) / WINDOW;

        // Bootstrap 95% CI (2000 resamples)
        const N_BOOT = 2000;
        const bootMeans = new Float32Array(N_BOOT);
        for (let b = 0; b < N_BOOT; b++) {
            let s = 0;
            for (let k = 0; k < WINDOW; k++) {
                s += windowGains[Math.floor(Math.random() * WINDOW)];
            }
            bootMeans[b] = s / WINDOW;
        }
        bootMeans.sort();
        const lo = bootMeans[Math.floor(0.025 * N_BOOT)];
        const hi = bootMeans[Math.floor(0.975 * N_BOOT)];
        const halfWidth = (hi - lo) / 2.0;

        if ((lo <= 0.0 && hi >= 0.0) || Math.abs(meanGain) <= halfWidth || Math.abs(meanGain) < 0.005) {
            state.convergenceState = 'CONVERGED';
            if (!state.convergenceAlertShown) {
                state.convergenceAlertShown = true;
                const last = state.convergenceHistory[state.convergenceHistory.length - 1];
                elements.alertMessage.textContent = `The on-device model has converged to your ground truth: Saccade F1 ${last.testF1Saccade.toFixed(2)}, kappa ${last.testKappaSaccade.toFixed(2)}. You have labeled enough data.`;
                elements.alertModal.classList.remove('hidden');
            }
        } else {
            state.convergenceState = 'COLLECTING';
        }
    }

    // --- Metrics Learning Curve Plot ---
    function showMetricsDialog() {
        elements.metricsModal.classList.remove('hidden');
        renderMetricsCanvas();
    }

    function renderMetricsCanvas() {
        const canvas = elements.metricsCanvas;
        const rect = canvas.parentElement.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(rect.width * dpr);
        canvas.height = Math.floor(rect.height * dpr);

        const mctx = canvas.getContext('2d');
        mctx.scale(dpr, dpr);
        const w = rect.width;
        const h = rect.height;

        mctx.clearRect(0, 0, w, h);

        if (state.convergenceHistory.length === 0) {
            mctx.fillStyle = '#78909C';
            mctx.font = '500 16px var(--font-stack)';
            mctx.textAlign = 'center';
            mctx.fillText("No training rounds recorded yet — Tap 'Train'", w / 2, h / 2);
            return;
        }

        const midX = w / 2;
        const top = 35;
        const bottom = h - 25;

        // Draw left & right axes
        mctx.strokeStyle = '#ECEFF1';
        mctx.strokeRect(0, top, midX - 10, bottom - top);
        mctx.strokeRect(midX + 10, top, w - midX - 20, bottom - top);

        // Titles
        mctx.fillStyle = '#37474F';
        mctx.font = '600 13px var(--font-stack)';
        mctx.textAlign = 'left';
        mctx.fillText("Event F1 Score", 10, 22);
        mctx.fillText("Cohen's Kappa", midX + 20, 22);

        const maxSec = Math.max(10, state.convergenceHistory[state.convergenceHistory.length - 1].labeledSeconds);

        function drawSeries(key, color, xOffset, plotW) {
            mctx.strokeStyle = color;
            mctx.lineWidth = 2.5;
            mctx.beginPath();
            let started = false;
            for (let i = 0; i < state.convergenceHistory.length; i++) {
                const r = state.convergenceHistory[i];
                const px = xOffset + (r.labeledSeconds / maxSec) * plotW;
                const py = bottom - (r[key] * (bottom - top));
                if (!started) { mctx.moveTo(px, py); started = true; }
                else { mctx.lineTo(px, py); }
            }
            mctx.stroke();
        }

        drawSeries('testF1Saccade', '#1565C0', 0, midX - 10);
        drawSeries('testF1Blink', '#C62828', 0, midX - 10);
        drawSeries('testKappaSaccade', '#1565C0', midX + 10, w - midX - 20);
        drawSeries('testKappaBlink', '#C62828', midX + 10, w - midX - 20);
    }

    // --- File Parser (CSV & MAT v5) ---
    function parseCSV(text) {
        const lines = text.trim().split(/\r?\n/);
        const rows = [];
        for (const line of lines) {
            if (!line.trim()) continue;
            const parts = line.split(/[,\t ]+/).map(v => {
                const f = parseFloat(v);
                return isNaN(f) ? NaN : f;
            });
            rows.push(new Float32Array(parts));
        }
        return rows;
    }

    let pendingModelFile = null;
    let pendingLabelsFile = null;

    async function loadCustomModel(modelFile) {
        if (!modelFile) return;
        try {
            const buf = await modelFile.arrayBuffer();
            state.trainedModelBin = new Uint8Array(buf);
            state.initialWeightsMode = 'local';
            const worker = getTrainWorker();
            worker.postMessage({ type: 'LOAD_MODEL', data: state.trainedModelBin });
        } catch (err) {
            console.error('Failed to load model weights:', err);
            alert('Cannot load custom model weights: ' + err.message);
        }
    }

    async function loadPreviousLabels(labelsFile) {
        if (!labelsFile) return;
        const ext = (labelsFile.name.split('.').pop() || '').toLowerCase();
        try {
            if (ext === 'json') {
                const text = await labelsFile.text();
                applyLabelsJson(JSON.parse(text));
            } else if (['mat', 'npz', 'h5', 'hdf5', 'hdf'].includes(ext)) {
                const buf = await labelsFile.arrayBuffer();
                const r = await UnEyeFormats.readTrialsFile(labelsFile.name, buf, loadH5Lib);
                if (r && r.labels) applyLoadedLabels(r.labels);
            } else if (ext === 'csv' || ext === 'txt') {
                const text = await labelsFile.text();
                const rows = parseCSV(text);
                if (rows.length > 0) {
                    const loaded = rows.map(r => {
                        const arr = new Int8Array(r.length);
                        for (let i = 0; i < r.length; i++) {
                            const v = r[i];
                            arr[i] = (isNaN(v) || v < 0) ? -1 : Math.round(v);
                        }
                        return arr;
                    });
                    applyLoadedLabels(loaded);
                }
            }
        } catch (err) {
            console.error('Failed to load previous labels:', err);
            alert('Cannot load previous labels: ' + err.message);
        }
    }

    function handleFile(file) {
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (ext === 'json') { loadLabelsFile(file); return; }
        if (!['csv', 'tsv', 'txt', 'dat', 'arff', 'asc', 'mat', 'npz', 'h5', 'hdf5', 'hdf'].includes(ext)) {
            alert('Unsupported file type ".' + ext + '". Open a .csv, .tsv, .txt, .arff, .asc, .mat, .npz or .h5 file (or a saved labels .json file).');
            return;
        }
        const modalAlreadyOpen = elements.loadModal && !elements.loadModal.classList.contains('hidden');
        pendingFileToLoad = file;
        elements.modalFileName.textContent = file.name;
        if (!modalAlreadyOpen) {
            pendingModelFile = null;
            pendingLabelsFile = null;
            if (elements.modalModelFileName) elements.modalModelFileName.textContent = 'No file chosen';
            if (elements.modalLabelsFileName) elements.modalLabelsFileName.textContent = '(none)';
            if (elements.localModelRow) elements.localModelRow.classList.add('hidden');
            if (elements.initialModelSelect) elements.initialModelSelect.value = 'pretrained';
            elements.loadModal.classList.remove('hidden');
        }
    }

    async function confirmLoadFile() {
        const file = pendingFileToLoad;
        if (!file) return;

        const rate = parseFloat(elements.rateInput.value) || 1000.0;
        const scale = parseFloat(elements.scaleInput.value) || 1.0;
        const prelabeler = elements.prelabelerSelect.value;
        if (elements.modalWinMs) setWinMs(parseFloat(elements.modalWinMs.value) || state.winMs);
        const sgVal = parseInt(elements.modalSgWindow.value, 10);
        state.filterSettings.enabled = sgVal > 0;
        if (sgVal > 0) state.filterSettings.windowMs = sgVal;

        if (elements.initialModelSelect) {
            const mVal = elements.initialModelSelect.value;
            if (mVal === 'local' && pendingModelFile) {
                await loadCustomModel(pendingModelFile);
            } else if (mVal === 'scratch') {
                state.initialWeightsMode = 'scratch';
                state.trainedModelBin = null;
            } else {
                state.initialWeightsMode = 'pretrained';
                state.trainedModelBin = null;      // "Pretrained" starts from combined.bin, not from a model kept from an earlier session
                state.trainedModelJson = null;
            }
        }

        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (['mat', 'npz', 'h5', 'hdf5', 'hdf'].includes(ext)) {      // binary formats: trials of different durations (formats.js)
            const binReader = new FileReader();
            binReader.onload = async (e) => {
                try {
                    elements.modalFileName.textContent = 'Reading ' + file.name + '…';
                    // a layout this page knows (its own exports) loads directly; any other file, or "Show the column explorer", opens the explorer
                    let r = null;
                    if (!(elements.alwaysExplorer && elements.alwaysExplorer.checked)) { try { r = await UnEyeFormats.readTrialsFile(file.name, e.target.result, loadH5Lib); } catch (err) { r = null; } }
                    if (!r) { await loadWithExplorer(file, rate, scale, prelabeler); return; }
                    elements.loadModal.classList.add('hidden');
                    loadDataset(r.trials, r.fs || rate, scale, prelabeler);
                    if (r.labels) {
                        applyLoadedLabels(r.labels);
                    }
                    if (pendingLabelsFile) {
                        await loadPreviousLabels(pendingLabelsFile);
                    }
                    const lens = r.trials.map(t => t.x.length);
                    console.info(`${file.name}: ${r.trials.length} trials, ${Math.min(...lens)}-${Math.max(...lens)} samples${r.fs ? ', ' + r.fs + ' Hz' : ''}${r.labels ? ', with labels' : ''}`);
                } catch (err) {
                    elements.loadModal.classList.add('hidden');
                    alert('Cannot read ' + file.name + ': ' + err.message);
                }
            };
            binReader.readAsArrayBuffer(file);
            return;
        }

        // text files: plain numbers (no header, 1-2 columns, or trials x time) keep the old reader; anything else (a header, many
        // columns, ARFF, EyeLink .asc) goes through the column explorer, which also handles the time stamps
        const forceExplorer = elements.alwaysExplorer && elements.alwaysExplorer.checked;
        if (forceExplorer || ext === 'arff' || ext === 'asc' || !(await isPlainTextFile(file))) {
            await loadWithExplorer(file, rate, scale, prelabeler);
            return;
        }

        const reader = new FileReader();
        reader.onload = async (e) => {
            const text = e.target.result;
            const rows = parseCSV(text);
            const trials = [];

            if (rows.length >= 2 && rows[0].length >= 100) {
                // Multi-trial CSV (UnEye convention: trials x time)
                for (let i = 0; i < rows.length; i += 2) {
                    const x = rows[i];
                    const y = (i + 1 < rows.length) ? rows[i + 1] : new Float32Array(x.length);
                    const valid = new Uint8Array(x.length);
                    for (let k = 0; k < x.length; k++) {
                        valid[k] = (!isNaN(x[k]) && !isNaN(y[k])) ? 1 : 0;
                    }
                    trials.push({ x, y, valid });
                }
            } else {
                // 2 columns or single column
                const n = rows.length;
                const x = new Float32Array(n);
                const y = new Float32Array(n);
                const valid = new Uint8Array(n);
                for (let i = 0; i < n; i++) {
                    x[i] = rows[i][0] || 0;
                    y[i] = rows[i][1] || 0;
                    valid[i] = (!isNaN(x[i]) && !isNaN(y[i])) ? 1 : 0;
                }
                trials.push({ x, y, valid });
            }

            elements.loadModal.classList.add('hidden');
            loadDataset(trials, rate, scale, prelabeler);
            if (pendingLabelsFile) {
                await loadPreviousLabels(pendingLabelsFile);
            }
        };
        reader.readAsText(file);
    }

    async function isPlainTextFile(file) {
        try { return UnEyeImport.sniffHead(await file.slice(0, 65536).text(), file.name).plain; }
        catch (err) { return true; }
    }

    async function loadWithExplorer(file, rate, scale, prelabeler) {
        elements.loadModal.classList.add('hidden');
        const r = await UnEyeImportUI.run(file, { rate, scale, h5: loadH5Lib });
        if (!r) { elements.modalFileName.textContent = file.name; elements.loadModal.classList.remove('hidden'); return; }      // cancelled: back to the load dialog
        loadDataset(r.trials, r.fs, r.degPerUnit, prelabeler);
        if (pendingLabelsFile) await loadPreviousLabels(pendingLabelsFile);
        const lens = r.trials.map(t => t.x.length);
        console.info(`${file.name}: ${r.trials.length} trials, ${Math.min(...lens)}-${Math.max(...lens)} samples at ${r.fs} Hz (${r.source}, ${r.degPerUnit} deg/unit)` + (r.notes.length ? ' · ' + r.notes.join(' ') : ''));
    }

    // --- HDF5 needs h5wasm (5 MB, loaded the first time an .h5 file is read or written; needs a connection once) ---
    async function loadH5Lib() {
        if (!window.h5wasm) {
            await new Promise((resolve, reject) => {
                const sc = document.createElement('script');
                sc.src = 'https://cdn.jsdelivr.net/npm/h5wasm@0.10.3/dist/iife/h5wasm.js';
                sc.onload = resolve;
                sc.onerror = () => reject(new Error('Cannot load the HDF5 library (h5wasm from cdn.jsdelivr.net): connect to the internet once, or use .npz / .mat.'));
                document.head.appendChild(sc);
            });
        }
        return window.h5wasm;
    }

    // labels read from a file (Int8Array per trial: 0 fixation, 1 saccade, 2 blink, -1 not reviewed) -> merge into reviewed ranges + expert events
    function applyLoadedLabels(labels) {
        for (let t = 0; t < state.trials.length; t++) {
            const L = labels[t];
            if (!L) continue;
            const n = state.trials[t].x.length, reviewed = [], events = [];
            for (let i = 0; i < n && i < L.length;) {
                if (!(L[i] >= 0)) { i++; continue; }
                let j = i; while (j + 1 < n && j + 1 < L.length && L[j + 1] >= 0) j++;
                reviewed.push([i, j]); i = j + 1;
            }
            for (let i = 0; i < n && i < L.length;) {
                const c = L[i];
                if (c !== 1 && c !== 2) { i++; continue; }
                let j = i; while (j + 1 < n && j + 1 < L.length && L[j + 1] === c) j++;
                events.push({ onset: i, offset: j, cls: c, fromPrelabel: false }); i = j + 1;
            }
            if (reviewed.length === 0 && events.length === 0) continue;

            // Merge loaded reviewed windows with existing scanWins
            state.scanWins[t] = mergeRangeList([...(state.scanWins[t] || []), ...reviewed, ...events.map(e => [e.onset, e.offset])], n);
            // Append loaded events and merge overlapping events of the same class
            state.expertEvents[t] = [...(state.expertEvents[t] || []), ...events];
            mergeOverlappingEvents(t);
            // Remove any pending point marks inside the loaded reviewed ranges
            if (state.pointMarks[t] && reviewed.length > 0) {
                state.pointMarks[t] = state.pointMarks[t].filter(
                    m => !reviewed.some(([s, e]) => m.at >= s && m.at <= e)
                );
            }
            recomputeReviewed(t, false);
            syncRenderedLabels(t);
        }
        if (state.pass === PASS_TWO) { buildRefineQueue(); state.refineIdx = 0; focusCurrentMark(); }
        updateStatus();
        render();
        saveToIndexedDB();
    }

    // positions (raw, so that the index of a label is the index of the sample in the file) and labels (-1 = not reviewed) in .npz / .mat / .h5
    async function exportBinary(fmt) {
        const labels = state.trials.map((tr, t) => {
            const out = new Int8Array(tr.x.length).fill(-1), lab = humanLabels(t);
            for (const [s, e] of state.reviewedRanges[t]) for (let i = s; i <= e && i < out.length; i++) out[i] = lab[i];
            return out;
        });
        const names = (state.trialNames && state.trialNames.some(n => n)) ? state.trialNames.map((n, i) => n || ('trial_' + i)) : null;
        const trials = state.trials.map(tr => ({ x: tr.x, y: tr.y }));
        try {
            if (fmt === 'npz') downloadFile('unEye_labeled.npz', UnEyeFormats.writeNpzFile(trials, labels, names, state.rateHz), 'application/octet-stream');
            else if (fmt === 'mat') downloadFile('unEye_labeled.mat', UnEyeFormats.writeMatFile(trials, labels, names, state.rateHz), 'application/octet-stream');
            else downloadFile('unEye_labeled.h5', await UnEyeFormats.writeH5File(trials, labels, names, state.rateHz, loadH5Lib), 'application/octet-stream');
        } catch (err) {
            alert('Cannot write the ' + fmt + ' file: ' + err.message);
        }
    }

    // --- Save / load the manual labels (a session file: marks, refined events, windows seen) ---
    function saveLabelsJson() {
        if (state.trials.length === 0) return;
        for (let t = 0; t < state.trials.length; t++) {
            mergeOverlappingEvents(t);
            recomputeReviewed(t, false);
        }
        const payload = {
            format: 'uneye-labels-v1',
            saved: new Date().toISOString(),
            rateHz: state.rateHz,
            winMs: state.winMs,
            trialNames: state.trials.map((tr, i) => (state.trialNames && state.trialNames[i]) || tr.name || ''),
            trialLengths: state.trials.map(tr => tr.x.length),
            observedPeriods: state.reviewedRanges,
            reviewedRanges: state.reviewedRanges,
            trials: state.trials.map((tr, t) => ({
                scanWins: state.scanWins[t],
                observedPeriods: state.reviewedRanges[t],
                reviewedRanges: state.reviewedRanges[t],
                expertEvents: state.expertEvents[t].map(e => ({ onset: e.onset, offset: e.offset, cls: e.cls })),
                pointMarks: state.pointMarks[t],
            })),
        };
        const d = new Date(), p = v => String(v).padStart(2, '0');
        downloadFile(`uneye_labels_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}.json`, JSON.stringify(payload), 'application/json');
    }

    function loadLabelsFile(file) {
        loadPreviousLabels(file);
    }

    // Merge / append loaded session labels with any labels already in the current session
    function applyLabelsJson(obj) {
        if (!obj || obj.format !== 'uneye-labels-v1' || !Array.isArray(obj.trials)) throw new Error('not a U\'n\'Eye labels file');
        if (state.trials.length === 0) throw new Error('open the eye traces first, then load the labels');
        const names = obj.trialNames || [], lens = obj.trialLengths || [];
        const curNames = state.trials.map((tr, i) => (state.trialNames && state.trialNames[i]) || tr.name || '');
        const byName = names.length && names.every(n => n) && curNames.every(n => n) && new Set(curNames).size === curNames.length;
        let applied = 0, skipped = 0;
        const index = new Map(); if (byName) names.forEach((n, i) => index.set(n, i));
        for (let t = 0; t < state.trials.length; t++) {
            let k = byName ? index.get(curNames[t]) : undefined;
            if (k === undefined && t < obj.trials.length) k = t;
            if (k === undefined || (lens[k] !== undefined && lens[k] !== state.trials[t].x.length)) { skipped++; continue; }
            const src = obj.trials[k] || {};
            const n = state.trials[t].x.length;

            const existingRev = (state.reviewedRanges[t] || []).map(r => [r[0], r[1]]);
            const loadedEvents = (src.expertEvents || []).map(e => ({ onset: e.onset, offset: e.offset, cls: e.cls, fromPrelabel: false }));
            const loadedRev = mergeRangeList([
                ...(src.scanWins || []),
                ...(src.observedPeriods || []),
                ...(src.reviewedRanges || []),
                ...((obj.observedPeriods && obj.observedPeriods[k]) || []),
                ...((obj.reviewedRanges && obj.reviewedRanges[k]) || []),
                ...loadedEvents.map(e => [e.onset, e.offset]),
            ], n);

            // 1) Merge scanned / observed windows (existing + loaded)
            state.scanWins[t] = mergeRangeList([...(state.scanWins[t] || []), ...loadedRev], n);

            // 2) Append loaded expertEvents to existing expertEvents and merge overlapping same-class events
            state.expertEvents[t] = [...(state.expertEvents[t] || []), ...loadedEvents];
            mergeOverlappingEvents(t);

            // 3) Merge pointMarks:
            //    - remove any pre-existing pointMarks that fall inside loaded reviewed ranges (e.g. untouched demo marks)
            //    - append loaded pointMarks that do not fall inside already-reviewed ranges or expertEvents
            let keptMarks = (state.pointMarks[t] || []).filter(
                m => !loadedRev.some(([s, e]) => m.at >= s && m.at <= e) &&
                     !state.expertEvents[t].some(ev => ev.onset <= m.at && m.at <= ev.offset)
            );
            for (const m of (src.pointMarks || [])) {
                if (existingRev.some(([s, e]) => m.at >= s && m.at <= e)) continue;
                if (state.expertEvents[t].some(ev => ev.onset <= m.at && m.at <= ev.offset)) continue;
                if (!keptMarks.some(km => Math.abs(km.at - m.at) <= 5 && km.cls === m.cls)) {
                    keptMarks.push({ at: m.at, cls: m.cls });
                }
            }
            keptMarks.sort((a, b) => a.at - b.at);
            state.pointMarks[t] = keptMarks;

            recomputeReviewed(t, false);
            syncRenderedLabels(t);
            applied++;
        }
        if (state.pass === PASS_TWO) { buildRefineQueue(); state.refineIdx = 0; focusCurrentMark(); }
        updateStatus(); render(); saveToIndexedDB();
        alert(`Labels loaded for ${applied} trial(s)${skipped ? `; ${skipped} trial(s) skipped (name or length does not match)` : ''}.`);
    }

    // --- CSV & Preprocessing Config Export (UnEye convention) ---
    function exportDataset() {
        if (state.trials.length === 0) return;
        const nTrials = state.trials.length;
        for (let t = 0; t < nTrials; t++) {
            mergeOverlappingEvents(t);
            recomputeReviewed(t, false);
        }

        const fmt = elements.exportFormat ? elements.exportFormat.value : 'csv';
        if (fmt !== 'csv') {
            exportBinary(fmt);          // one file with all trials (different durations), the labels and the names
        } else {
            // 1. Labels CSV (L.csv) - NaN for unreviewed samples
            let lCsv = "";
            for (let t = 0; t < nTrials; t++) {
                const row = [];
                const lab = humanLabels(t);
                const rev = state.reviewedRanges[t];
                for (let i = 0; i < lab.length; i++) {
                    const isRev = rev.some(([s, e]) => s <= i && i <= e);
                    row.push(isRev ? lab[i] : "nan");
                }
                lCsv += row.join(",") + "\n";
            }
            downloadFile("L.csv", lCsv, "text/csv");

            // 1b. Observed periods mask CSV (observed.csv) - 1 for reviewed/observed samples, 0 for unreviewed
            let obsCsv = "";
            for (let t = 0; t < nTrials; t++) {
                const row = [];
                const rev = state.reviewedRanges[t] || [];
                const len = state.trials[t].x.length;
                for (let i = 0; i < len; i++) {
                    const isRev = rev.some(([s, e]) => s <= i && i <= e);
                    row.push(isRev ? "1" : "0");
                }
                obsCsv += row.join(",") + "\n";
            }
            downloadFile("observed.csv", obsCsv, "text/csv");

            // 2. Positions CSV (X.csv, Y.csv) - smooth positions for reproducibility
            let xCsv = "", yCsv = "";
            for (let t = 0; t < nTrials; t++) {
                ensureDerived(t);
                const tr = state.trials[t];
                const px = (state.filterSettings.enabled && tr.smoothX) ? tr.smoothX : tr.x;
                const py = (state.filterSettings.enabled && tr.smoothY) ? tr.smoothY : tr.y;
                xCsv += Array.from(px).map(v => isNaN(v) ? "nan" : v.toFixed(5)).join(",") + "\n";
                yCsv += Array.from(py).map(v => isNaN(v) ? "nan" : v.toFixed(5)).join(",") + "\n";
            }
            downloadFile("X.csv", xCsv, "text/csv");
            downloadFile("Y.csv", yCsv, "text/csv");
        }

        // 3. Preprocessing Config & Python Code (uneye_config.json)
        const cfg = {
            uneye_version: "1.0",
            export_date: new Date().toISOString(),
            sampling_rate_hz: state.rateHz,
            deg_per_unit: state.degPerUnit,
            preprocessing: {
                filter: state.filterSettings.enabled ? state.filterSettings.filterType : "none",
                savgol_window_ms: state.filterSettings.windowMs,
                savgol_window_samples: Math.round(state.filterSettings.windowMs * state.rateHz / 1000.0) | 1,
                savgol_poly_order: state.filterSettings.polyOrder,
                adaptive_threshold_lambda: state.filterSettings.lambda,
                min_peak_velocity_deg_s: state.filterSettings.minPeakVel,
                min_amplitude_deg: state.filterSettings.minAmplitude,
                max_speed_deg_s: 1300.0
            },
            python_preprocessing_code: `# Preprocessing pipeline matching web labeler:\nimport scipy.signal\nimport numpy as np\n\n# Apply Savitzky-Golay filter to test traces before passing to UnEye:\nwindow_samples = int(round(${state.filterSettings.windowMs} * ${state.rateHz} / 1000.0)) | 1\nX_clean = scipy.signal.savgol_filter(X, window_length=window_samples, polyorder=3, mode="interp", axis=-1)\nY_clean = scipy.signal.savgol_filter(Y, window_length=window_samples, polyorder=3, mode="interp", axis=-1)\n`
        };
        downloadFile("uneye_config.json", JSON.stringify(cfg, null, 2), "application/json");
    }

    function downloadFile(filename, content, type) {
        if (window.AndroidBridge && typeof window.AndroidBridge.saveFile === 'function') {
            try {
                let base64 = '';
                if (typeof content === 'string') {
                    base64 = btoa(unescape(encodeURIComponent(content)));
                } else if (content instanceof ArrayBuffer) {
                    const bytes = new Uint8Array(content);
                    let binary = '';
                    const len = bytes.byteLength;
                    for (let i = 0; i < len; i++) binary += String.fromCharCode(bytes[i]);
                    base64 = btoa(binary);
                } else if (content instanceof Uint8Array) {
                    let binary = '';
                    const len = content.byteLength;
                    for (let i = 0; i < len; i++) binary += String.fromCharCode(content[i]);
                    base64 = btoa(binary);
                }
                if (base64) {
                    window.AndroidBridge.saveFile(filename, base64, type || 'application/octet-stream');
                    return;
                }
            } catch (e) {
                console.warn('AndroidBridge.saveFile error, falling back to browser blob:', e);
            }
        }
        const blob = new Blob([content], { type: `${type};charset=utf-8;` });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", filename);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    // A file opened from another Android app (file manager, "Open with"): same path as the Open… button, every format
    window.loadFromAndroid = function(filename, base64) {
        try {
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            handleFile(new File([bytes], filename));
        } catch (e) {
            alert('Cannot open ' + filename + ': ' + e.message);
        }
    };

    // --- Setup UI Listeners ---
    function setActiveClass(cls) {
        state.activeClass = cls;
        if (elements.saccadeBtn && elements.blinkBtn) {
            if (cls === CLS_SACCADE) {
                elements.saccadeBtn.classList.add('active');
                elements.saccadeBtn.classList.remove('inactive');
                elements.blinkBtn.classList.remove('active');
                elements.blinkBtn.classList.add('inactive');
            } else {
                elements.blinkBtn.classList.add('active');
                elements.blinkBtn.classList.remove('inactive');
                elements.saccadeBtn.classList.remove('active');
                elements.saccadeBtn.classList.add('inactive');
            }
        }
        if (elements.classToggle) {
            elements.classToggle.textContent = state.activeClass === CLS_SACCADE ? "Saccade" : "Blink";
            elements.classToggle.className = `btn btn-class ${state.activeClass === CLS_SACCADE ? 'saccade' : 'blink'}`;
        }
        if (state.eraser) {
            state.eraser = false;
            elements.eraserBtn.classList.remove('active');
            elements.eraserBtn.textContent = 'Eraser';
            render();
        }
    }

    function setAutoSlide(enabled) {
        state.autoSlide = !!enabled;
        if (elements.autoSlideBtn) {
            elements.autoSlideBtn.classList.toggle('active', state.autoSlide);
            elements.autoSlideBtn.textContent = state.autoSlide ? 'Auto-slide: On' : 'Auto-slide: Off';
        }
        if (elements.autoSlideToggle) {
            elements.autoSlideToggle.checked = state.autoSlide;
        }
    }

    function setupListeners() {
        // Class selection (side-by-side buttons: selecting one grises the other and deactivates eraser)
        if (elements.saccadeBtn) elements.saccadeBtn.addEventListener('click', () => setActiveClass(CLS_SACCADE));
        if (elements.blinkBtn) elements.blinkBtn.addEventListener('click', () => setActiveClass(CLS_BLINK));
        elements.classToggle.addEventListener('click', () => {
            setActiveClass(state.activeClass === CLS_SACCADE ? CLS_BLINK : CLS_SACCADE);
        });

        elements.undoBtn.addEventListener('click', () => undo(state.currentTrialIdx));
        elements.menuBtn.addEventListener('click', () => {
            elements.toolsPanel.classList.toggle('hidden');
            elements.menuBtn.classList.toggle('active', !elements.toolsPanel.classList.contains('hidden'));
            requestAnimationFrame(() => { syncCanvasDimensions(); render(); });
        });
        elements.eraserBtn.addEventListener('click', () => {
            state.eraser = !state.eraser;
            elements.eraserBtn.classList.toggle('active', state.eraser);
            elements.eraserBtn.textContent = state.eraser ? 'Eraser ON' : 'Eraser';
        });

        if (elements.autoSlideBtn) {
            elements.autoSlideBtn.addEventListener('click', () => {
                setAutoSlide(!state.autoSlide);
            });
        }
        if (elements.autoSlideToggle) {
            elements.autoSlideToggle.addEventListener('change', (e) => {
                setAutoSlide(e.target.checked);
            });
        }

        elements.saveLabelsBtn.addEventListener('click', saveLabelsJson);
        elements.loadLabelsBtn.addEventListener('click', () => elements.labelsInput.click());
        elements.labelsInput.addEventListener('change', e => { if (e.target.files && e.target.files[0]) loadLabelsFile(e.target.files[0]); e.target.value = ''; });
        elements.winMsInput.addEventListener('change', e => setWinMs(parseFloat(e.target.value)));
        elements.prevEpochBtn.addEventListener('click', prevEpoch);

        if (elements.startOpenBtn) elements.startOpenBtn.addEventListener('click', () => elements.fileInput.click());
        if (elements.netSuggestBtn) elements.netSuggestBtn.addEventListener('click', () => { setNetSuggestions(!state.netSuggestions); refreshSuggestions(); });

        elements.openBtn.addEventListener('click', () => elements.fileInput.click());

        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFile(e.target.files[0]);
            }
        });

        elements.modalConfirmBtn.addEventListener('click', confirmLoadFile);
        elements.modalCancelBtn.addEventListener('click', () => elements.loadModal.classList.add('hidden'));

        elements.metricsBtn.addEventListener('click', showMetricsDialog);
        elements.metricsCloseBtn.addEventListener('click', () => elements.metricsModal.classList.add('hidden'));

        elements.trainNowBtn.addEventListener('click', runTrainingRound);
        if (elements.cancelTrainBtn) {
            elements.cancelTrainBtn.addEventListener('click', () => {
                if (trainWorker) {
                    trainWorker.postMessage({ type: 'CANCEL' });
                    elements.trainStatus.textContent = 'Cancelling…';
                }
            });
        }
        if (elements.downloadModelBtn) {
            elements.downloadModelBtn.addEventListener('click', downloadModel);
        }
        if (elements.initialModelSelect) {
            elements.initialModelSelect.addEventListener('change', (e) => {
                state.initialWeightsMode = e.target.value;
                if (elements.localModelRow) {
                    elements.localModelRow.classList.toggle('hidden', e.target.value !== 'local');
                }
            });
        }
        if (elements.modalPickModelBtn && elements.modalModelFile) {
            elements.modalPickModelBtn.addEventListener('click', () => elements.modalModelFile.click());
            elements.modalModelFile.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) {
                    pendingModelFile = e.target.files[0];
                    if (elements.modalModelFileName) elements.modalModelFileName.textContent = pendingModelFile.name;
                }
            });
        }
        if (elements.modalPickLabelsBtn && elements.modalLabelsFile) {
            elements.modalPickLabelsBtn.addEventListener('click', () => elements.modalLabelsFile.click());
            elements.modalLabelsFile.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) {
                    pendingLabelsFile = e.target.files[0];
                    if (elements.modalLabelsFileName) elements.modalLabelsFileName.textContent = pendingLabelsFile.name;
                }
            });
        }
        if (elements.modalChangeTraceBtn) {
            elements.modalChangeTraceBtn.addEventListener('click', () => elements.fileInput.click());
        }
        elements.exportBtn.addEventListener('click', exportDataset);

        // Filter Settings Dialog
        elements.filterBtn.addEventListener('click', () => {
            elements.enableSg.checked = state.filterSettings.enabled;
            elements.sgWindowMs.value = String(state.filterSettings.windowMs);
            elements.filterModal.classList.remove('hidden');
        });

        elements.filterCloseBtn.addEventListener('click', () => elements.filterModal.classList.add('hidden'));
        elements.filterCancelBtn.addEventListener('click', () => elements.filterModal.classList.add('hidden'));

        elements.filterApplyBtn.addEventListener('click', () => {
            state.filterSettings.enabled = elements.enableSg.checked;
            state.filterSettings.windowMs = parseInt(elements.sgWindowMs.value, 10) || 21;
            // the smoothing is for the display; the proposals (HMM or network) do not depend on it
            preprocessAllTrials();
            elements.filterModal.classList.add('hidden');
            updateStatus();
            render();
        });

        elements.alertOkBtn.addEventListener('click', () => elements.alertModal.classList.add('hidden'));

        elements.switchPassBtn.addEventListener('click', () => {
            setPass(state.pass === PASS_ONE ? PASS_TWO : PASS_ONE);
        });

        elements.nextEpochBtn.addEventListener('click', nextEpoch);

        elements.prevTrialBtn.addEventListener('click', () => {
            if (state.currentTrialIdx > 0) {
                resetGesture();
                if (state.pass === PASS_ONE) scanCurrentWindow();
                state.currentTrialIdx--;
                state.zoomMultiplier = 1.0;
                if (state.pass === PASS_TWO) {
                    buildRefineQueue();
                    state.refineIdx = 0;
                    focusCurrentMark();
                } else {
                    state.winIdx = 0;
                    applyGridWindow();
                }
                updateStatus();
                render();
            }
        });

        elements.nextTrialBtn.addEventListener('click', () => {
            if (state.currentTrialIdx < state.trials.length - 1) {
                resetGesture();
                if (state.pass === PASS_ONE) scanCurrentWindow();
                state.currentTrialIdx++;
                state.zoomMultiplier = 1.0;
                if (state.pass === PASS_TWO) {
                    buildRefineQueue();
                    state.refineIdx = 0;
                    focusCurrentMark();
                } else {
                    state.winIdx = 0;
                    applyGridWindow();
                }
                updateStatus();
                render();
            }
        });

        // Pass 2: Navigation across events
        elements.prevEventBtn.addEventListener('click', () => {
            if (state.refineIdx > 0) {
                state.refineIdx--;
            } else {
                const pt = findTrialWithEvents(state.currentTrialIdx, -1);
                if (pt < 0) return;
                enterTrialForRefine(pt, true);
            }
            state.zoomMultiplier = 1.0;
            focusCurrentMark();
            updateStatus();
            render();
        });

        elements.nextEventBtn.addEventListener('click', () => {
            if (state.refineIdx < state.refineQueue.length - 1) {
                state.refineIdx++;
            } else {
                const nt = findTrialWithEvents(state.currentTrialIdx, 1);
                if (nt < 0) { maybeAutoTrain(); return; }   // last event of the last trial: pass 2 is finished
                enterTrialForRefine(nt, false);
            }
            state.zoomMultiplier = 1.0;
            focusCurrentMark();
            updateStatus();
            render();
        });

        elements.notEventBtn.addEventListener('click', () => {
            if (state.currentMarkAt >= 0) {
                deleteMarkAt(state.currentTrialIdx, state.currentMarkAt);
                buildRefineQueue();
                state.zoomMultiplier = 1.0;
                focusCurrentMark();
                updateStatus();
                render();
            }
        });

        elements.widerBtn.addEventListener('click', () => {
            state.zoomMultiplier = (state.zoomMultiplier || 1.0) === 1.0 ? 2.5 : ((state.zoomMultiplier === 2.5) ? 5.0 : 1.0);
            focusCurrentMark();
            render();
        });

        elements.acceptPreBtn.addEventListener('click', () => {
            if (state.currentMarkAt >= 0) {
                const onset = state.currentPutativeOnset >= 0 ? state.currentPutativeOnset : state.currentMarkAt - 15;
                const offset = state.currentPutativeOffset >= 0 ? state.currentPutativeOffset : state.currentMarkAt + 15;
                const prevLen = state.refineQueue.length;
                refineMark(state.currentTrialIdx, state.currentMarkAt, onset, offset, state.activeClass);
                buildRefineQueue();
                const newLen = state.refineQueue.length;
                state.refineIdx = newLen < prevLen ? Math.min(state.refineIdx + 1, newLen) : state.refineIdx + 1;
                state.zoomMultiplier = 1.0;
                focusCurrentMark();
                updateStatus();
                render();
            }
        });
    }

    // --- Init ---
    function init() {
        try { const w = parseFloat(localStorage.getItem('uneye_winMs')); if (w >= 20) state.winMs = w; } catch (e) { }
        if (elements.winMsInput) elements.winMsInput.value = String(state.winMs);
        if (elements.modalWinMs) elements.modalWinMs.value = String(state.winMs);
        setupListeners();
        setupPointerEvents();
        resizeCanvas();

        try { const v = localStorage.getItem('uneye_netSuggestions'); setNetSuggestions(v === null ? true : v === '1'); } catch (e) { setNetSuggestions(true); }

        // No simulated data: resume the autosaved session, otherwise only a window to choose a file
        restoreFromIndexedDB().then(restored => {
            if (!restored && !state.trials.length && elements.startModal) elements.startModal.classList.remove('hidden');
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    { const b = document.getElementById('appBuild'); if (b) b.textContent = 'Build ' + APP_BUILD; }
    window.state = state;
    window.flushSave = flushSave;
    window.generateDemoDataset = generateDemoDataset;      // simulated traces for the automated tests only (no button, never loaded at start)
    window.refreshSuggestions = refreshSuggestions;
    window.setNetSuggestions = setNetSuggestions;
    window.maybeAutoTrain = maybeAutoTrain;
    window.setPass = setPass;
    window.nextEpoch = nextEpoch;
    window.applyGridWindow = applyGridWindow;
    window.humanLabels = humanLabels;
    window.focusCurrentMark = focusCurrentMark;
    window.render = render;
    window.updateStatus = updateStatus;
    window.syncRenderedLabels = syncRenderedLabels;
    window.slideWindowAfterEvent = slideWindowAfterEvent;
    window.setAutoSlide = setAutoSlide;
    window.handleTap = handleTap;
    window.refineMark = refineMark;
    window.mergeOverlappingEvents = mergeOverlappingEvents;
    window.applyLabelsJson = applyLabelsJson;
    window.applyLoadedLabels = applyLoadedLabels;
    window.saveLabelsJson = saveLabelsJson;
    window.recomputeReviewed = recomputeReviewed;
    window.buildRefineQueue = buildRefineQueue;
})();
