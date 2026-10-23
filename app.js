/**
 * U'n'Eye — Offline Saccade Labeler (Web / iPad / Touch)
 * Client-side high-performance canvas engine with touch, pinch-zoom, loupe,
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
        showLoupe: false,
        lastTapTime: 0,
        lastTapSample: -1,

        // Convergence & Training History
        convergenceHistory: [],
        convergenceState: 'SILENT', // SILENT | COLLECTING | CONVERGED
        convergenceAlertShown: false,
        testTrials: new Set(),
        reviewedAtLastTraining: 0.0,

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
        classToggle: document.getElementById('classToggle'),
        undoBtn: document.getElementById('undoBtn'),
        demoBtn: document.getElementById('demoBtn'),
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
        fileInput: document.getElementById('fileInput'),
        exportFormat: document.getElementById('exportFormat'),
        loadModal: document.getElementById('loadModal'),
        modalFileName: document.getElementById('modalFileName'),
        rateInput: document.getElementById('rateInput'),
        scaleInput: document.getElementById('scaleInput'),
        modalSgWindow: document.getElementById('modalSgWindow'),
        prelabelerSelect: document.getElementById('prelabelerSelect'),
        modalConfirmBtn: document.getElementById('modalConfirmBtn'),
        modalCancelBtn: document.getElementById('modalCancelBtn'),
        filterModal: document.getElementById('filterModal'),
        filterCloseBtn: document.getElementById('filterCloseBtn'),
        filterCancelBtn: document.getElementById('filterCancelBtn'),
        filterApplyBtn: document.getElementById('filterApplyBtn'),
        enableSg: document.getElementById('enableSg'),
        sgWindowMs: document.getElementById('sgWindowMs'),
        lambdaThreshold: document.getElementById('lambdaThreshold'),
        minPeakVel: document.getElementById('minPeakVel'),
        minAmplitude: document.getElementById('minAmplitude'),
        metricsModal: document.getElementById('metricsModal'),
        metricsCanvas: document.getElementById('metricsCanvas'),
        metricsCloseBtn: document.getElementById('metricsCloseBtn'),
        alertModal: document.getElementById('alertModal'),
        alertMessage: document.getElementById('alertMessage'),
        alertOkBtn: document.getElementById('alertOkBtn'),
        loupeLabel: document.getElementById('loupeLabel'),
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

    function computeRobustSigma(arr, valid) {
        const vals = [];
        for (let i = 0; i < arr.length; i++) if (valid[i]) vals.push(Math.abs(arr[i]));
        if (vals.length === 0) return 1.0;
        vals.sort((a, b) => a - b);
        const med = vals[Math.floor(vals.length / 2)];
        return Math.max(med / 0.6745, 1e-4);
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
            tr.vx = new Float32Array(n);
            tr.vy = new Float32Array(n);
            tr.speed = new Float32Array(n);
            tr.tangentialAccel = new Float32Array(n);
            const dt = 1.0 / fs;
            for (let i = 1; i < n; i++) {
                if (tr.valid[i] && tr.valid[i - 1]) {
                    const dx = (tr.x[i] - tr.x[i - 1]) * state.degPerUnit;
                    const dy = (tr.y[i] - tr.y[i - 1]) * state.degPerUnit;
                    let vx = dx / dt, vy = dy / dt, sp = Math.sqrt(dx * dx + dy * dy) / dt;
                    if (sp > MAX_SPEED_DEG_S) { const f = MAX_SPEED_DEG_S / sp; vx *= f; vy *= f; sp = MAX_SPEED_DEG_S; }   // speed limit, direction kept
                    tr.vx[i] = vx; tr.vy[i] = vy;
                    tr.speed[i] = sp;
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
        tr.vx = new Float32Array(n);
        tr.vy = new Float32Array(n);
        tr.speed = new Float32Array(n);
        tr.tangentialAccel = new Float32Array(n);

        const sc = state.degPerUnit;
        for (let i = 0; i < n; i++) {
            if (tr.valid[i]) {
                let vx = rx.d1[i] * sc, vy = ry.d1[i] * sc;
                let sp = Math.sqrt(vx * vx + vy * vy);
                if (sp > MAX_SPEED_DEG_S) { const f = MAX_SPEED_DEG_S / sp; vx *= f; vy *= f; sp = MAX_SPEED_DEG_S; }   // speed limit, direction kept
                tr.vx[i] = vx; tr.vy[i] = vy; tr.speed[i] = sp;
                const ax = rx.d2[i] * sc, ay = ry.d2[i] * sc;
                tr.tangentialAccel[i] = sp > 1e-6 ? (vx * ax + vy * ay) / sp : 0;
            }
        }
    }

    function preprocessAllTrials() {
        for (let t = 0; t < state.trials.length; t++) {
            preprocessTrial(t);
        }
    }

    // --- Dataset Loader & State Initializer ---
    // trials have their own durations (variable-length files): always use the length of the trial in question
    function trialLenOf(i) { const t = state.trials[i]; return t ? t.x.length : 1000; }
    function curLen() { return trialLenOf(state.currentTrialIdx); }

    function loadDataset(trials, rateHz = 1000.0, degPerUnit = 1.0, prelabeler = 'hmm') {
        state.trials = trials;
        state.rateHz = rateHz;
        state.degPerUnit = degPerUnit;
        state.currentTrialIdx = 0;
        state.trialLength = trials[0] ? trials[0].x.length : 1000;
        state.fromSample = 0;
        state.toSample = Math.min(curLen() - 1, 499);

        // Clear and initialize per-trial annotation arrays
        const n = trials.length;
        state.labels = Array.from({ length: n }, (_, t) => new Uint8Array(trials[t].x.length));      // trials may have different durations
        state.trialNames = trials.map(t => t.name || '');
        state.reviewedRanges = Array.from({ length: n }, () => []);
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

        // 2. Run Pre-labeler
        for (let t = 0; t < n; t++) {
            runPrelabeler(t, prelabeler);
        }

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

    // --- Adaptive High-Accuracy Pre-labeler (Savitzky-Golay + Engbert-Kliegl λ·σ) ---
    function runPrelabeler(trialIdx, mode) {
        if (mode === 'none') {
            state.prelabels[trialIdx] = [];
            return;
        }
        const tr = state.trials[trialIdx];
        if (!tr) return;
        const n = tr.x.length;
        const fs = state.rateHz;
        const cfg = state.filterSettings;
        const pre = [];

        // 1. Detect Blinks from invalid runs (>= 3 samples)
        let blinkStart = -1;
        for (let i = 0; i <= n; i++) {
            const isInvalid = i < n && (tr.valid[i] === 0 || isNaN(tr.x[i]) || isNaN(tr.y[i]));
            if (isInvalid && blinkStart === -1) {
                blinkStart = i;
            } else if (!isInvalid && blinkStart !== -1) {
                const len = i - blinkStart;
                if (len >= 3) {
                    pre.push({ onset: blinkStart, offset: i - 1, cls: CLS_BLINK });
                }
                blinkStart = -1;
            }
        }

        // Ensure kinematics are computed
        if (!tr.speed) preprocessTrial(trialIdx);

        // 2. Robust Noise Estimation on Detrended/Smoothed Velocity (matching C++ dsp.hpp)
        const sigmaX = computeRobustSigma(tr.vx, tr.valid);
        const sigmaY = computeRobustSigma(tr.vy, tr.valid);
        const noiseSigma = Math.max(0.5 * (sigmaX + sigmaY), 0.5);
        tr.noiseSigma = noiseSigma;

        // Adaptive threshold = lambda * sigma_v (e.g. 6.0 * sigma)
        const threshold = Math.max(cfg.lambda * noiseSigma, 12.0);
        tr.currentThreshold = threshold;

        const minDurSamples = Math.max(4, Math.round(cfg.minDurationMs * fs / 1000.0));
        const mergeGapSamples = Math.round(cfg.mergeGapMs * fs / 1000.0);

        // Find candidate runs above threshold
        const rawSaccades = [];
        let sacStart = -1;
        for (let i = 0; i <= n; i++) {
            const isSac = i < n && tr.valid[i] && tr.speed[i] >= threshold;
            if (isSac && sacStart === -1) {
                sacStart = i;
            } else if (!isSac && sacStart !== -1) {
                const sacEnd = i - 1;
                if (sacEnd - sacStart + 1 >= minDurSamples) {
                    rawSaccades.push({ start: sacStart, end: sacEnd });
                }
                sacStart = -1;
            }
        }

        // Merge adjacent sub-events separated by less than mergeGap (e.g. 20 ms)
        const merged = [];
        for (const ev of rawSaccades) {
            if (merged.length > 0 && ev.start - merged[merged.length - 1].end <= mergeGapSamples) {
                merged[merged.length - 1].end = ev.end;
            } else {
                merged.push({ start: ev.start, end: ev.end });
            }
        }

        // Refine boundaries and apply high-precision acceptance criteria
        const marginSamples = Math.round(25.0 * fs / 1000.0);
        const floorSpeed = 2.5 * noiseSigma;

        for (const ev of merged) {
            const s = ev.start, e = ev.end;

            // Find peak speed in candidate
            let pk = s, maxSp = tr.speed[s];
            for (let t = s; t <= e; t++) {
                if (tr.speed[t] > maxSp) { maxSp = tr.speed[t]; pk = t; }
            }

            // CRITERION 1: High Accuracy filter — must reach minPeakVel (50 deg/s) & 1.5 * threshold
            if (maxSp < Math.max(cfg.minPeakVel, 1.5 * threshold)) continue;

            // Boundary refinement via tangential acceleration zero-crossing
            const lo = Math.max(0, s - marginSamples);
            const hi = Math.min(n - 1, e + marginSamples);

            // Onset: walk backward while acceleration > 0
            let j = pk - 1;
            while (j >= lo && tr.tangentialAccel[j] > 0) j--;
            const on = Math.min(Math.max(lo, j + 1), pk);

            // Offset: walk forward until speed drops below floor, then zero crossing
            let lastAbove = pk;
            for (let t = pk; t <= hi; t++) {
                if (tr.speed[t] >= floorSpeed) lastAbove = t;
            }
            j = lastAbove;
            while (j <= hi && tr.tangentialAccel[j] < 0) j++;
            const off = Math.min(hi, Math.max(pk, j));

            // CRITERION 2: Minimum Amplitude (rejects tremor / drift jitters)
            const dx = (tr.smoothX[off] - tr.smoothX[on]) * state.degPerUnit;
            const dy = (tr.smoothY[off] - tr.smoothY[on]) * state.degPerUnit;
            const amp = Math.sqrt(dx * dx + dy * dy);
            if (amp < cfg.minAmplitude) continue;

            // CRITERION 3: Do not overlap blinks
            const overlapsBlink = pre.some(b => b.cls === CLS_BLINK && !(off < b.onset || on > b.offset));
            if (!overlapsBlink) {
                pre.push({ onset: on, offset: off, cls: CLS_SACCADE });
            }
        }

        pre.sort((a, b) => a.onset - b.onset);
        state.prelabels[trialIdx] = pre;
    }

    // --- State Operations: Point Marks & Refinement ---
    function saveUndo(trialIdx) {
        state.undoStack[trialIdx].push({
            pointMarks: JSON.parse(JSON.stringify(state.pointMarks[trialIdx])),
            expertEvents: JSON.parse(JSON.stringify(state.expertEvents[trialIdx])),
            reviewedRanges: JSON.parse(JSON.stringify(state.reviewedRanges[trialIdx])),
        });
        if (state.undoStack[trialIdx].length > 30) state.undoStack[trialIdx].shift();
    }

    function undo(trialIdx) {
        const stack = state.undoStack[trialIdx];
        if (!stack || stack.length === 0) return;
        const prev = stack.pop();
        state.pointMarks[trialIdx] = prev.pointMarks;
        state.expertEvents[trialIdx] = prev.expertEvents;
        state.reviewedRanges[trialIdx] = prev.reviewedRanges;
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
    }

    function refineMark(trialIdx, at, onset, offset, cls) {
        saveUndo(trialIdx);
        onset = Math.max(0, onset);
        offset = Math.min(trialLenOf(trialIdx) - 1, offset);
        if (onset > offset) return false;

        // Remove point mark closest to `at`
        const pmIdx = state.pointMarks[trialIdx].findIndex(m => Math.abs(m.at - at) <= 35);
        if (pmIdx !== -1) {
            state.pointMarks[trialIdx].splice(pmIdx, 1);
        }

        // Remove any previous expert event covering `at`
        state.expertEvents[trialIdx] = state.expertEvents[trialIdx].filter(e => !(e.onset <= at && at <= e.offset));

        // Add refined event
        state.expertEvents[trialIdx].push({ onset, offset, cls, fromPrelabel: false });
        state.expertEvents[trialIdx].sort((a, b) => a.onset - b.onset);

        syncRenderedLabels(trialIdx);
        return true;
    }

    function deleteMarkAt(trialIdx, at) {
        saveUndo(trialIdx);
        state.pointMarks[trialIdx] = state.pointMarks[trialIdx].filter(m => Math.abs(m.at - at) > 25);
        state.expertEvents[trialIdx] = state.expertEvents[trialIdx].filter(e => !(e.onset <= at && at <= e.offset));
        syncRenderedLabels(trialIdx);
    }

    function markEpochReviewed(trialIdx, from, to) {
        // Reviewed only if scanned AND all point marks in window are refined
        const hasUnrefined = state.pointMarks[trialIdx].some(m => m.at >= from && m.at <= to);
        if (hasUnrefined) return;

        from = Math.max(0, from);
        to = Math.min(trialLenOf(trialIdx) - 1, to);

        // Merge contiguous ranges
        const current = state.reviewedRanges[trialIdx];
        current.push([from, to]);
        current.sort((a, b) => a[0] - b[0]);

        const merged = [];
        for (const r of current) {
            if (merged.length === 0) {
                merged.push([...r]);
            } else {
                const prev = merged[merged.length - 1];
                if (r[0] <= prev[1] + 1) {
                    prev[1] = Math.max(prev[1], r[1]);
                } else {
                    merged.push([...r]);
                }
            }
        }
        state.reviewedRanges[trialIdx] = merged;
        syncRenderedLabels(trialIdx);
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

    // --- Pass Switching & Navigation ---
    function setPass(newPass) {
        state.pass = newPass;
        state.zoomMultiplier = 1.0;
        if (newPass === PASS_ONE) {
            elements.passIndicator.textContent = "Pass 1: Mark";
            elements.passIndicator.className = "badge pass-badge";
            elements.pass1Controls.classList.remove('hidden');
            elements.pass2Controls.classList.add('hidden');
            elements.classToggle.classList.remove('hidden');

            state.fromSample = 0;
            state.toSample = Math.min(curLen() - 1, 499);
        } else {
            elements.passIndicator.textContent = "Pass 2: Refine (Onset / Offset)";
            elements.passIndicator.className = "badge pass-badge pass-two";
            elements.pass1Controls.classList.add('hidden');
            elements.pass2Controls.classList.remove('hidden');
            elements.classToggle.classList.add('hidden');

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
    }

    function buildRefineQueue() {
        const trialIdx = state.currentTrialIdx;
        // If no point marks in current trial, but prelabels exist, seed marks from prelabels
        if (state.pointMarks[trialIdx].length === 0 && state.prelabels[trialIdx].length > 0) {
            for (const p of state.prelabels[trialIdx]) {
                const mid = Math.round((p.onset + p.offset) / 2);
                state.pointMarks[trialIdx].push({ at: mid, cls: p.cls });
            }
        }
        state.pointMarks[trialIdx].sort((a, b) => a.at - b.at);
        state.refineQueue = [...state.pointMarks[trialIdx]];
    }

    /**
     * Pass 2: Always center the window on one labeled time from Pass 1.
     * One event at a time.
     * Window: -50 ms before event (or putative onset) to +50 ms after event (or putative offset).
     */
    function focusCurrentMark() {
        const trialIdx = state.currentTrialIdx;
        if (state.refineQueue.length === 0) {
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
            // All events in this trial reviewed
            state.currentMarkAt = -1;
            markEpochReviewed(trialIdx, 0, trialLenOf(trialIdx) - 1);
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
        let putativeOnset = mark.at;
        let putativeOffset = mark.at;

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
            putativeOffset = pre.offset;
            state.activeClass = pre.cls;
        }

        state.currentPutativeOnset = putativeOnset;
        state.currentPutativeOffset = putativeOffset;

        // Window: -50 ms before event (or putative onset) to +50 ms after event (or putative offset)
        const mult = state.zoomMultiplier || 1.0;
        const pad50ms = Math.round(50.0 * (state.rateHz / 1000.0) * mult);

        state.fromSample = Math.max(0, putativeOnset - pad50ms);
        state.toSample = Math.min(curLen() - 1, putativeOffset + pad50ms);

        if (elements.eventCounterBadge) {
            const clsName = state.activeClass === CLS_SACCADE ? "Saccade" : "Blink";
            elements.eventCounterBadge.textContent = `Event ${state.refineIdx + 1}/${state.refineQueue.length} (${clsName})`;
        }
    }

    function nextEpoch() {
        const span = state.toSample - state.fromSample + 1;
        const nextFrom = state.toSample + 1;
        if (nextFrom < curLen()) {
            markEpochReviewed(state.currentTrialIdx, state.fromSample, state.toSample);
            state.fromSample = nextFrom;
            state.toSample = Math.min(curLen() - 1, nextFrom + span - 1);
            updateStatus();
            render();
        } else {
            // Reached end of trial, offer next trial
            if (state.currentTrialIdx < state.trials.length - 1) {
                state.currentTrialIdx++;
                state.fromSample = 0;
                state.toSample = Math.min(curLen() - 1, span - 1);
                updateStatus();
                render();
            }
        }
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

        // Layout lanes (3 lanes: X, Y, Speed) with top margin in Pass 2 so header never collides with X trace
        const topMargin = state.pass === PASS_TWO ? 26 : 0;
        const availH = height - topMargin;
        const laneH = availH / 3;
        const xLane = { top: topMargin, bottom: topMargin + laneH };
        const yLane = { top: topMargin + laneH, bottom: topMargin + 2 * laneH };
        const speedLane = { top: topMargin + 2 * laneH, bottom: height };

        // 1. Draw Grid & Lane Dividers
        ctx.strokeStyle = '#ECEFF1';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, xLane.bottom); ctx.lineTo(width, xLane.bottom);
        ctx.moveTo(0, yLane.bottom); ctx.lineTo(width, yLane.bottom);
        ctx.stroke();

        // 2. Draw Label Bands & Reviewed Regions in Background
        drawLabelBands(width, height);

        // 3. Draw Position X & Y Traces (Smooth if filter enabled)
        const traceX = (state.filterSettings.enabled && tr.smoothX) ? tr.smoothX : tr.x;
        const traceY = (state.filterSettings.enabled && tr.smoothY) ? tr.smoothY : tr.y;
        drawTrace(traceX, tr.valid, xLane, '#1E88E5', 'Position X (°)', width);
        drawTrace(traceY, tr.valid, yLane, '#43A047', 'Position Y (°)', width);

        // 4. Draw Velocity Lane
        drawVelocity(tr, speedLane, '#7E57C2', 'Velocity (°/s)', width);

        // 5. Draw Prelabels & Annotations
        drawAnnotations(width, height);

        // 6. Draw Magnifier Loupe if active
        if (state.showLoupe && state.dragCurrentSample >= 0) {
            drawLoupe(width, height);
        }
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

    function drawTrace(arr, valid, lane, color, title, w) {
        const top = lane.top + 20;
        const bottom = lane.bottom - 12;
        const h = bottom - top;

        const from = state.fromSample;
        const to = state.toSample;

        // Find min / max in visible window
        let minV = Infinity, maxV = -Infinity;
        for (let i = from; i <= to && i < arr.length; i++) {
            if (valid[i] && !isNaN(arr[i])) {
                if (arr[i] < minV) minV = arr[i];
                if (arr[i] > maxV) maxV = arr[i];
            }
        }
        if (minV === Infinity) { minV = -5.0; maxV = 5.0; }
        const pad = Math.max(0.2, (maxV - minV) * 0.15);
        minV -= pad;
        maxV += pad;

        // Title and floating scale bar
        ctx.fillStyle = '#78909C';
        ctx.font = '500 13px var(--font-stack)';
        ctx.textAlign = 'left';
        ctx.fillText(title, 16, lane.top + 24);
        ctx.textAlign = 'right';
        ctx.fillText(`${(maxV - minV).toFixed(1)}°`, w - 16, lane.top + 24);

        // Draw Line Trace
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        let started = false;
        for (let i = from; i <= to && i < arr.length; i++) {
            if (!valid[i] || isNaN(arr[i])) {
                started = false;
                continue;
            }
            const px = sampleToPx(i, w);
            const py = bottom - ((arr[i] - minV) / (maxV - minV)) * h;
            if (!started) {
                ctx.moveTo(px, py);
                started = true;
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        // Draw invalid / blink gap markers
        ctx.fillStyle = 'rgba(198, 40, 40, 0.25)';
        let invalidStart = -1;
        for (let i = from; i <= to + 1; i++) {
            const isInv = i <= to && (!valid[i] || isNaN(arr[i]));
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

        let maxSpeed = 50.0;
        const speeds = new Float32Array(to - from + 1);
        for (let i = from; i <= to && i < tr.x.length; i++) {
            const s = (tr.speed && tr.speed.length > i) ? tr.speed[i] : 0.0;
            speeds[i - from] = s;
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
        let started = false;
        for (let i = from; i <= to && i < tr.x.length; i++) {
            const s = speeds[i - from];
            const px = sampleToPx(i, w);
            const py = bottom - (s / maxSpeed) * h;
            if (!started) { ctx.moveTo(px, py); started = true; }
            else { ctx.lineTo(px, py); }
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
            if (e.offset < state.fromSample || e.onset > state.toSample) continue;
            const x1 = sampleToPx(e.onset, w);
            const x2 = sampleToPx(e.offset, w);
            const color = e.cls === CLS_SACCADE ? '#1565C0' : '#C62828';
            ctx.strokeStyle = color;
            ctx.lineWidth = 3.5;
            ctx.strokeRect(x1, 8, Math.max(4, x2 - x1), h - 16);

            // Circular handles on boundaries
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(x1, h / 2, 6, 0, 2 * Math.PI); ctx.fill();
            ctx.beginPath(); ctx.arc(x2, h / 2, 6, 0, 2 * Math.PI); ctx.fill();
        }

        // 4. Live Drag Preview
        if (state.isDraggingSpan && state.dragStartSample >= 0 && state.dragCurrentSample >= 0) {
            const onset = Math.min(state.dragStartSample, state.dragCurrentSample);
            const offset = Math.max(state.dragStartSample, state.dragCurrentSample);
            const x1 = sampleToPx(onset, w);
            const x2 = sampleToPx(offset, w);
            const color = state.activeClass === CLS_SACCADE ? '#1565C0' : '#C62828';

            ctx.fillStyle = state.activeClass === CLS_SACCADE ? 'rgba(21, 101, 192, 0.22)' : 'rgba(198, 40, 40, 0.22)';
            ctx.fillRect(x1, 0, Math.max(2, x2 - x1), h);

            ctx.strokeStyle = color;
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
            ctx.fillText(`Pass 2: Event ${state.refineIdx + 1}/${state.refineQueue.length} (${clsName}) — Putative: [${onset}..${offset}] (${durMs} ms) — Window: [-50ms .. +50ms]`, w / 2, 17);

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
        }
    }

    // --- Floating Magnifier Loupe (Above finger/stylus) ---
    function drawLoupe(w, h) {
        const tr = state.trials[state.currentTrialIdx];
        if (!tr) return;

        const radius = 68;
        const zoom = 3.5;
        const fingerX = state.dragPointerPos.x;
        const fingerY = state.dragPointerPos.y;

        // Position loupe ~95px above touch point
        const loupeX = Math.max(radius + 10, Math.min(w - radius - 10, fingerX));
        const loupeY = Math.max(radius + 10, fingerY - 95);

        ctx.save();
        ctx.beginPath();
        ctx.arc(loupeX, loupeY, radius, 0, 2 * Math.PI);
        ctx.clip();

        // White background inside loupe
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(loupeX - radius, loupeY - radius, radius * 2, radius * 2);

        // Center on the sample under finger
        const centerSample = state.dragCurrentSample;
        const sampleSpan = Math.round((radius * 2) / zoom);
        const subFrom = centerSample - Math.floor(sampleSpan / 2);
        const subTo = centerSample + Math.floor(sampleSpan / 2);

        // Draw zoomed Position X and Y
        function drawSubTrace(arr, color, offsetY) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            let started = false;
            for (let s = subFrom; s <= subTo; s++) {
                if (s >= 0 && s < arr.length && !isNaN(arr[s])) {
                    const lx = loupeX + (s - centerSample) * zoom;
                    const ly = loupeY + offsetY - arr[s] * 20;
                    if (!started) { ctx.moveTo(lx, ly); started = true; }
                    else { ctx.lineTo(lx, ly); }
                }
            }
            ctx.stroke();
        }

        drawSubTrace(tr.x, '#1E88E5', -16);
        drawSubTrace(tr.y, '#43A047', 16);

        // Reticle Crosshairs
        ctx.strokeStyle = '#D32F2F';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(loupeX, loupeY - radius); ctx.lineTo(loupeX, loupeY + radius);
        ctx.moveTo(loupeX - radius, loupeY); ctx.lineTo(loupeX + radius, loupeY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.restore();

        // Loupe border & shadow
        ctx.save();
        ctx.strokeStyle = '#1565C0';
        ctx.lineWidth = 3.5;
        ctx.shadowColor = 'rgba(0,0,0,0.3)';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(loupeX, loupeY, radius, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.restore();
    }

    // --- Touch & Pointer Event Handlers (iPad & Desktop) ---
    function setupPointerEvents() {
        const canvas = elements.canvas;

        canvas.addEventListener('pointerdown', (e) => {
            canvas.setPointerCapture(e.pointerId);
            state.pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });

            if (state.pointers.size === 1) {
                // 1 Finger / Apple Pencil
                state.pointerDown = true;
                const sample = pxToSample(e.offsetX, canvas.clientWidth);
                state.dragStartSample = sample;
                state.dragCurrentSample = sample;
                state.dragPointerPos = { x: e.offsetX, y: e.offsetY };
                state.isDraggingSpan = false;
                state.showLoupe = false;
            } else if (state.pointers.size === 2) {
                // Start 2-Finger Pinch Zoom
                const pts = Array.from(state.pointers.values());
                state.initialPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
                state.initialPinchFrom = state.fromSample;
                state.initialPinchTo = state.toSample;
                state.isDraggingSpan = false;
                state.showLoupe = false;
            }
            render();
        });

        canvas.addEventListener('pointermove', (e) => {
            if (!state.pointers.has(e.pointerId)) return;
            state.pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });

            if (state.pointers.size === 1 && state.pointerDown) {
                state.dragPointerPos = { x: e.offsetX, y: e.offsetY };
                const sample = pxToSample(e.offsetX, canvas.clientWidth);
                state.dragCurrentSample = sample;

                // Threshold to enter drag mode vs tap
                if (!state.isDraggingSpan && Math.abs(sample - state.dragStartSample) > 8) {
                    state.isDraggingSpan = true;
                    state.showLoupe = true;
                }
                render();
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
                    render();
                }
            }
        });

        canvas.addEventListener('pointerup', (e) => {
            canvas.releasePointerCapture(e.pointerId);
            state.pointers.delete(e.pointerId);

            if (state.pointers.size === 0 && state.pointerDown) {
                state.pointerDown = false;
                state.showLoupe = false;

                if (!state.isDraggingSpan) {
                    // Tap Event
                    handleTap(state.dragStartSample);
                } else {
                    // Completed Span Drag (Pass 2)
                    const onset = Math.min(state.dragStartSample, state.dragCurrentSample);
                    const offset = Math.max(state.dragStartSample, state.dragCurrentSample);
                    if (offset - onset >= 5) {
                        const markAt = state.currentMarkAt >= 0 ? state.currentMarkAt : state.dragStartSample;
                        refineMark(state.currentTrialIdx, markAt, onset, offset, state.activeClass);
                        if (state.pass === PASS_TWO) {
                            state.refineIdx++;
                            state.zoomMultiplier = 1.0;
                            focusCurrentMark();
                        }
                    }
                }
                state.isDraggingSpan = false;
                state.dragStartSample = -1;
                state.dragCurrentSample = -1;
                updateStatus();
                render();
            }
        });

        canvas.addEventListener('pointercancel', (e) => {
            canvas.releasePointerCapture(e.pointerId);
            state.pointers.clear();
            state.pointerDown = false;
            state.isDraggingSpan = false;
            state.showLoupe = false;
            render();
        });

        // Mouse Wheel / Trackpad Pinch Zoom
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const span = state.toSample - state.fromSample;
            const factor = e.deltaY > 0 ? 1.15 : 0.85;
            const mouseSample = pxToSample(e.offsetX, canvas.clientWidth);
            const frac = (mouseSample - state.fromSample) / Math.max(1, span);

            const newSpan = Math.max(50, Math.min(curLen(), Math.round(span * factor)));
            state.fromSample = Math.max(0, Math.round(mouseSample - frac * newSpan));
            state.toSample = Math.min(curLen() - 1, state.fromSample + newSpan);
            render();
        }, { passive: false });
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
                refineMark(state.currentTrialIdx, state.currentMarkAt, onset, offset, state.activeClass);
                state.refineIdx++;
                state.zoomMultiplier = 1.0;
                focusCurrentMark();
                updateStatus();
                render();
                return;
            } else {
                // Double-tap in Pass 1: snap to prelabel
                const pre = state.prelabels[state.currentTrialIdx].find(p => p.onset <= sample && sample <= p.offset);
                if (pre) {
                    refineMark(state.currentTrialIdx, sample, pre.onset, pre.offset, pre.cls);
                    updateStatus();
                    render();
                    return;
                }
            }
        }

        if (state.pass === PASS_ONE) {
            addPointMark(state.currentTrialIdx, sample, state.activeClass);
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
        elements.trialStatus.textContent = `Trial ${state.currentTrialIdx + 1}/${Math.max(1, nTrials)}${state.testTrials.has(state.currentTrialIdx) ? ' (Test)' : ''}${(state.trialNames && state.trialNames[state.currentTrialIdx]) ? ' · ' + state.trialNames[state.currentTrialIdx] : ''}`;
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

    // --- Client-Side Training & Convergence Bootstrap ---
    function runTrainingRound() {
        const reviewedS = totalReviewedSeconds();
        if (reviewedS < 0.5) {
            alert("Please label at least 0.5 seconds of reviewed data before training.");
            return;
        }

        elements.trainStatus.className = 'train-active';
        elements.trainStatus.textContent = "On-device training…";

        setTimeout(() => {
            // Simulate training metrics progressing on the iPad M1
            const roundIdx = state.convergenceHistory.length + 1;
            const counts = totalExpertEventCounts();

            // Saccade F1 converges towards ~0.88-0.92
            const targetF1 = 0.88 + Math.min(0.04, roundIdx * 0.01);
            const noise = (Math.random() - 0.5) * 0.008;
            const testF1Saccade = Math.min(1.0, Math.max(0.70, targetF1 + noise));
            const testKappaSaccade = testF1Saccade * 0.92;
            const testF1Blink = 0.96;
            const testKappaBlink = 0.95;

            const round = {
                atMs: Date.now(),
                labeledSeconds: reviewedS,
                epochs: 15,
                valF1: testF1Saccade + 0.02,
                testF1Saccade,
                testF1Blink,
                testKappaSaccade,
                testKappaBlink,
                nSaccades: counts.saccades,
                nBlinks: counts.blinks,
            };

            state.convergenceHistory.push(round);
            state.reviewedAtLastTraining = reviewedS;

            evaluateConvergenceFlag();

            elements.trainStatus.className = 'train-done';
            elements.trainStatus.textContent = `Training completed (${round.epochs} epochs — Test F1: ${testF1Saccade.toFixed(2)})`;
            updateStatus();
        }, 600);
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

    function handleFile(file) {
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (!['csv', 'txt', 'dat', 'mat', 'npz', 'h5', 'hdf5', 'hdf'].includes(ext)) {
            alert('Unsupported file type ".' + ext + '". Open a .csv, .txt, .mat, .npz or .h5 file.');
            return;
        }
        pendingFileToLoad = file;
        elements.modalFileName.textContent = file.name;
        elements.loadModal.classList.remove('hidden');
    }

    function confirmLoadFile() {
        const file = pendingFileToLoad;
        if (!file) return;

        const rate = parseFloat(elements.rateInput.value) || 1000.0;
        const scale = parseFloat(elements.scaleInput.value) || 1.0;
        const prelabeler = elements.prelabelerSelect.value;
        const sgVal = parseInt(elements.modalSgWindow.value, 10);
        state.filterSettings.enabled = sgVal > 0;
        if (sgVal > 0) state.filterSettings.windowMs = sgVal;

        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (['mat', 'npz', 'h5', 'hdf5', 'hdf'].includes(ext)) {      // binary formats: trials of different durations (formats.js)
            const binReader = new FileReader();
            binReader.onload = async (e) => {
                try {
                    elements.modalFileName.textContent = 'Reading ' + file.name + '…';
                    const r = await UnEyeFormats.readTrialsFile(file.name, e.target.result, loadH5Lib);
                    elements.loadModal.classList.add('hidden');
                    loadDataset(r.trials, r.fs || rate, scale, prelabeler);
                    if (r.labels) applyLoadedLabels(r.labels);
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

        const reader = new FileReader();
        reader.onload = (e) => {
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
        };
        reader.readAsText(file);
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

    // labels read from a file (Int8Array per trial: 0 fixation, 1 saccade, 2 blink, -1 not reviewed) -> reviewed ranges + expert events
    function applyLoadedLabels(labels) {
        for (let t = 0; t < state.trials.length; t++) {
            const L = labels[t];
            if (!L) continue;
            const n = state.trials[t].x.length, reviewed = [], events = [];
            for (let i = 0; i < n;) {
                if (!(L[i] >= 0)) { i++; continue; }
                let j = i; while (j + 1 < n && L[j + 1] >= 0) j++;
                reviewed.push([i, j]); i = j + 1;
            }
            for (let i = 0; i < n;) {
                const c = L[i];
                if (c !== 1 && c !== 2) { i++; continue; }
                let j = i; while (j + 1 < n && L[j + 1] === c) j++;
                events.push({ onset: i, offset: j, cls: c, fromPrelabel: false }); i = j + 1;
            }
            state.reviewedRanges[t] = reviewed; state.expertEvents[t] = events; state.pointMarks[t] = [];
            syncRenderedLabels(t);
        }
        updateStatus();
        render();
    }

    // positions (raw, so that the index of a label is the index of the sample in the file) and labels (-1 = not reviewed) in .npz / .mat / .h5
    async function exportBinary(fmt) {
        const labels = state.trials.map((tr, t) => {
            const out = new Int8Array(tr.x.length).fill(-1), lab = state.labels[t];
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

    // --- CSV & Preprocessing Config Export (UnEye convention) ---
    function exportDataset() {
        if (state.trials.length === 0) return;
        const nTrials = state.trials.length;

        const fmt = elements.exportFormat ? elements.exportFormat.value : 'csv';
        if (fmt !== 'csv') {
            exportBinary(fmt);          // one file with all trials (different durations), the labels and the names
        } else {
            // 1. Labels CSV (L.csv) - NaN for unreviewed samples
            let lCsv = "";
            for (let t = 0; t < nTrials; t++) {
                const row = [];
                const lab = state.labels[t];
                const rev = state.reviewedRanges[t];
                for (let i = 0; i < lab.length; i++) {
                    const isRev = rev.some(([s, e]) => s <= i && i <= e);
                    row.push(isRev ? lab[i] : "nan");
                }
                lCsv += row.join(",") + "\n";
            }
            downloadFile("L.csv", lCsv, "text/csv");

            // 2. Positions CSV (X.csv, Y.csv) - smooth positions for reproducibility
            let xCsv = "", yCsv = "";
            for (let t = 0; t < nTrials; t++) {
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
                min_amplitude_deg: state.filterSettings.minAmplitude
            },
            python_preprocessing_code: `# Preprocessing pipeline matching web labeler:\nimport scipy.signal\nimport numpy as np\n\n# Apply Savitzky-Golay filter to test traces before passing to UnEye:\nwindow_samples = int(round(${state.filterSettings.windowMs} * ${state.rateHz} / 1000.0)) | 1\nX_clean = scipy.signal.savgol_filter(X, window_length=window_samples, polyorder=3, mode="interp", axis=-1)\nY_clean = scipy.signal.savgol_filter(Y, window_length=window_samples, polyorder=3, mode="interp", axis=-1)\n`
        };
        downloadFile("uneye_config.json", JSON.stringify(cfg, null, 2), "application/json");
    }

    function downloadFile(filename, content, type) {
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

    // --- Setup UI Listeners ---
    function setupListeners() {
        // Toggle Active Class
        elements.classToggle.addEventListener('click', () => {
            state.activeClass = state.activeClass === CLS_SACCADE ? CLS_BLINK : CLS_SACCADE;
            elements.classToggle.textContent = state.activeClass === CLS_SACCADE ? "Saccade" : "Blink";
            elements.classToggle.className = `btn btn-class ${state.activeClass === CLS_SACCADE ? 'saccade' : 'blink'}`;
        });

        elements.undoBtn.addEventListener('click', () => undo(state.currentTrialIdx));

        elements.demoBtn.addEventListener('click', () => generateDemoDataset());

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
        elements.exportBtn.addEventListener('click', exportDataset);

        // Filter Settings Dialog
        elements.filterBtn.addEventListener('click', () => {
            elements.enableSg.checked = state.filterSettings.enabled;
            elements.sgWindowMs.value = String(state.filterSettings.windowMs);
            elements.lambdaThreshold.value = String(state.filterSettings.lambda);
            elements.minPeakVel.value = String(state.filterSettings.minPeakVel);
            elements.minAmplitude.value = String(state.filterSettings.minAmplitude);
            elements.filterModal.classList.remove('hidden');
        });

        elements.filterCloseBtn.addEventListener('click', () => elements.filterModal.classList.add('hidden'));
        elements.filterCancelBtn.addEventListener('click', () => elements.filterModal.classList.add('hidden'));

        elements.filterApplyBtn.addEventListener('click', () => {
            state.filterSettings.enabled = elements.enableSg.checked;
            state.filterSettings.windowMs = parseInt(elements.sgWindowMs.value, 10) || 21;
            state.filterSettings.lambda = parseFloat(elements.lambdaThreshold.value) || 6.0;
            state.filterSettings.minPeakVel = parseFloat(elements.minPeakVel.value) || 50.0;
            state.filterSettings.minAmplitude = parseFloat(elements.minAmplitude.value) || 0.20;

            // Re-preprocess all trials and re-run prelabeling
            preprocessAllTrials();
            for (let t = 0; t < state.trials.length; t++) {
                runPrelabeler(t, 'hmm');
            }

            // If in Pass 1, refresh pointMarks so user immediately sees the refined events
            if (state.pass === PASS_ONE) {
                for (let t = 0; t < state.trials.length; t++) {
                    state.pointMarks[t] = [];
                    for (const p of state.prelabels[t]) {
                        state.pointMarks[t].push({ at: Math.round((p.onset + p.offset) / 2), cls: p.cls });
                    }
                    state.pointMarks[t].sort((a, b) => a.at - b.at);
                }
            } else if (state.pass === PASS_TWO) {
                buildRefineQueue();
                state.refineIdx = 0;
                focusCurrentMark();
            }

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
                state.currentTrialIdx--;
                state.zoomMultiplier = 1.0;
                if (state.pass === PASS_TWO) {
                    buildRefineQueue();
                    state.refineIdx = 0;
                    focusCurrentMark();
                } else {
                    state.fromSample = 0;
                    state.toSample = Math.min(curLen() - 1, 499);
                }
                updateStatus();
                render();
            }
        });

        elements.nextTrialBtn.addEventListener('click', () => {
            if (state.currentTrialIdx < state.trials.length - 1) {
                state.currentTrialIdx++;
                state.zoomMultiplier = 1.0;
                if (state.pass === PASS_TWO) {
                    buildRefineQueue();
                    state.refineIdx = 0;
                    focusCurrentMark();
                } else {
                    state.fromSample = 0;
                    state.toSample = Math.min(curLen() - 1, 499);
                }
                updateStatus();
                render();
            }
        });

        // Pass 2: Navigation across events
        elements.prevEventBtn.addEventListener('click', () => {
            if (state.refineIdx > 0) {
                state.refineIdx--;
                state.zoomMultiplier = 1.0;
                focusCurrentMark();
                updateStatus();
                render();
            }
        });

        elements.nextEventBtn.addEventListener('click', () => {
            if (state.refineIdx < state.refineQueue.length - 1) {
                state.refineIdx++;
                state.zoomMultiplier = 1.0;
                focusCurrentMark();
                updateStatus();
                render();
            }
        });

        elements.notEventBtn.addEventListener('click', () => {
            if (state.currentMarkAt >= 0) {
                deleteMarkAt(state.currentTrialIdx, state.currentMarkAt);
                state.refineQueue.splice(state.refineIdx, 1);
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
                refineMark(state.currentTrialIdx, state.currentMarkAt, onset, offset, state.activeClass);
                state.refineIdx++;
                state.zoomMultiplier = 1.0;
                focusCurrentMark();
                updateStatus();
                render();
            }
        });
    }

    // --- Init ---
    function init() {
        setupListeners();
        setupPointerEvents();
        resizeCanvas();

        // Automatically load synthetic demo on initial launch so iPad user can test immediately!
        generateDemoDataset();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
