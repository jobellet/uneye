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
        globalUndoStack: [],

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
        sessionTrainedModelBin: null,
        customModelBin: null,
        customModelName: '',
        pretrainedModelBin: null,
        activeModelSource: 'hmm',   // 'hmm' | 'pretrained' | 'session' | 'custom' | 'none'
        activeModelName: '',
        modelPredictedTrials: new Set(),
        predictingTrialInFlight: false,
        previousPrelabels: null,
        wakeLock: null,
        initialWeightsMode: 'pretrained',
        prelabelerMode: 'hmm',      // 'hmm' (label-free HMM, hmm_prelabel.js), 'file' (the machine detection stored in the file: pred_labels, or prob thresholded), 'pretrained', 'custom' or 'none'

        // Machine detection read from the file (never labels, never reviewed):
        // prob[t] = Float32Array P(saccade) of the network (NaN = none) or null; fileEvents[t] = [{onset, offset, cls}] its detected events
        // (pred_labels runs, or prob thresholded with the U'n'Eye post-processing) = the proposals of the 'file' source
        prob: [],
        fileProb: [],
        hasProb: false,
        probThreshold: 0.5,
        fileEvents: [],
        // viewedRanges[t] = merged [start, end] of every window drawn on screen (any pass, near misses)
        viewedRanges: [],
        // rejectedNearMisses[t] = [{onset, offset, at, peak}]: near misses the user judged "Not an Event"
        rejectedNearMisses: [],
        // Near-miss review (inside pass 2): queue of candidates of all trials sorted by decreasing peak probability
        nearMiss: { active: false, queue: [], idx: 0, lower: 0.1 },
        netSuggestions: true,       // once a network has been trained: show its proposals (on) or the HMM's (off); toggle in the menu
        minNewSecondsForAutoTrain: 2.0,   // automatic training after a pass 2 needs at least this much newly reviewed data
        hoverSample: -1,
        hoverPos: null,
        shortcuts: {},
        recordingShortcutAction: null,

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
        loadWeightsBtn: document.getElementById('loadWeightsBtn'),
        downloadModelBtn: document.getElementById('downloadModelBtn'),
        mainModelSelect: document.getElementById('mainModelSelect'),
        mainModelInput: document.getElementById('mainModelInput'),
        initialModelSelect: document.getElementById('initialModelSelect'),
        fileInput: document.getElementById('fileInput'),
        exportFormat: document.getElementById('exportFormat'),
        loadModal: document.getElementById('loadModal'),
        modalFileName: document.getElementById('modalFileName'),
        modalFileSummary: document.getElementById('modalFileSummary'),
        modalChangeTraceBtn: document.getElementById('modalChangeTraceBtn'),
        modalLabelsVarRow: document.getElementById('modalLabelsVarRow'),
        modalLabelsVar: document.getElementById('modalLabelsVar'),
        modalLabelsRole: document.getElementById('modalLabelsRole'),
        modalProbVarRow: document.getElementById('modalProbVarRow'),
        modalProbVar: document.getElementById('modalProbVar'),
        rateInput: document.getElementById('rateInput'),
        scaleInput: document.getElementById('scaleInput'),
        modalSgWindow: document.getElementById('modalSgWindow'),
        localModelRow: document.getElementById('localModelRow'),
        modalModelFileName: document.getElementById('modalModelFileName'),
        modalPickModelBtn: document.getElementById('modalPickModelBtn'),
        modalModelFile: document.getElementById('modalModelFile'),
        prelabelerModelRow: document.getElementById('prelabelerModelRow'),
        prelabelerModelFileName: document.getElementById('prelabelerModelFileName'),
        prelabelerPickModelBtn: document.getElementById('prelabelerPickModelBtn'),
        modalLabelsFileName: document.getElementById('modalLabelsFileName'),
        modalPrevLabelsVar: document.getElementById('modalPrevLabelsVar'),
        modalPickLabelsBtn: document.getElementById('modalPickLabelsBtn'),
        modalLabelsFile: document.getElementById('modalLabelsFile'),
        pickLabelsVarModal: document.getElementById('pickLabelsVarModal'),
        pickLabelsVarSummary: document.getElementById('pickLabelsVarSummary'),
        pickLabelsVarSelect: document.getElementById('pickLabelsVarSelect'),
        pickLabelsVarConfirmBtn: document.getElementById('pickLabelsVarConfirmBtn'),
        pickLabelsVarCancelBtn: document.getElementById('pickLabelsVarCancelBtn'),
        pickLabelsVarCloseBtn: document.getElementById('pickLabelsVarCloseBtn'),
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
        nearMissBtn: document.getElementById('nearMissBtn'),
        nearMissMenuBtn: document.getElementById('nearMissMenuBtn'),
        nearMissMenuGroup: document.getElementById('nearMissMenuGroup'),
        nearMissLowerInput: document.getElementById('nearMissLowerInput'),
        viewedBtn: document.getElementById('viewedBtn'),
        viewedModal: document.getElementById('viewedModal'),
        viewedResetBtn: document.getElementById('viewedResetBtn'),
        viewedCloseBtn: document.getElementById('viewedCloseBtn'),
        viewedScroll: document.getElementById('viewedScroll'),
        viewedCanvas: document.getElementById('viewedCanvas'),
        viewedSizer: document.getElementById('viewedSizer'),
        viewedSummary: document.getElementById('viewedSummary'),
        viewedTip: document.getElementById('viewedTip'),
        shortcutsBtn: document.getElementById('shortcutsBtn'),
        shortcutsTopBtn: document.getElementById('shortcutsTopBtn'),
        shortcutsModal: document.getElementById('shortcutsModal'),
        shortcutsCloseBtn: document.getElementById('shortcutsCloseBtn'),
        shortcutsDoneBtn: document.getElementById('shortcutsDoneBtn'),
        shortcutsResetBtn: document.getElementById('shortcutsResetBtn'),
        shortcutsPresetQwerty: document.getElementById('shortcutsPresetQwerty'),
        shortcutsPresetAzerty: document.getElementById('shortcutsPresetAzerty'),
        shortcutsList: document.getElementById('shortcutsList'),
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
    const APP_BUILD = '2026-10-26a';

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
    // plus any intervals explicitly refined/accepted by the user (stored in scanWins).
    function recomputeReviewed(t, notify) {
        const marks = state.pointMarks[t] || [];
        const ok = (state.scanWins[t] || []).filter(([f, e]) => !marks.some(m => m.at >= f && m.at <= e)).map(r => r.slice());
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

    // extra (optional): {prob: [Float32Array], predLabels: [Int8Array], probParams: {threshold, merge_ms, min_dur_ms}} = the machine detection of the file
    function loadDataset(trials, rateHz = 1000.0, degPerUnit = 1.0, prelabeler = 'hmm', extra = null) {
        state.trials = trials;
        setFileDetection(trials, rateHz, extra);
        const useModelPrelabeler = (prelabeler === 'pretrained' || prelabeler === 'custom') && Boolean(state.trainedModelBin);
        state.prelabelerMode = prelabeler === 'none'
            ? 'none'
            : useModelPrelabeler
            ? prelabeler
            : (state.fileEvents.some(Boolean) ? 'file' : 'hmm');
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
        state.globalUndoStack = [];
        state.viewedRanges = Array.from({ length: n }, () => []);
        state.rejectedNearMisses = Array.from({ length: n }, () => []);
        state.nearMiss.active = false; state.nearMiss.queue = []; state.nearMiss.idx = 0;
        state.modelPredictedTrials = new Set();

        // Pick test set (every 4th trial, min 2)
        state.testTrials = new Set();
        for (let t = 0; t < n; t += 4) {
            state.testTrials.add(t);
        }

        // 1. Preprocess all trials with Savitzky-Golay
        preprocessAllTrials();

        // 2. Proposals & initial file pre-labels (file pre-labels are loaded directly as ground truth so they don't need to be confirmed one by one)
        if (useModelPrelabeler) {
            state.activeModelSource = prelabeler;
            setNetSuggestions(true);
            syncMainModelSelectUI();
            applyModelAsPrelabeler({ asGroundTruthIfEmpty: true, updateProbs: true });
        } else {
            state.activeModelSource = prelabeler === 'none' ? 'none' : 'hmm';
            for (let t = 0; t < n; t++) {
                runPrelabeler(t, state.prelabelerMode);
                if (state.prelabelerMode === 'file' && state.fileEvents[t] && state.fileEvents[t].length > 0) {
                    state.expertEvents[t] = state.fileEvents[t].map(e => ({ onset: e.onset, offset: e.offset, cls: e.cls, fromPrelabel: false }));
                    mergeOverlappingEvents(t);
                    recomputeReviewed(t, false);
                }
                syncRenderedLabels(t);
            }
            if (state.netSuggestions && state.trainedModelBin) refreshSuggestions();
            syncMainModelSelectUI();
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

    // --- Machine detection stored in the file (U'n'Eye output: pred_labels, prob) ---
    // pred_labels -> proposals as they are; prob alone -> thresholded with the U'n'Eye post-processing (merge gaps < 5 ms, drop events < 5 ms)
    function setFileDetection(trials, rateHz, extra) {
        const n = trials.length, ex = extra || {};
        const pp = ex.probParams || {};
        state.probThreshold = pp.threshold > 0 && pp.threshold < 1 ? pp.threshold : 0.5;
        state.prob = Array.from({ length: n }, (_, t) => (ex.prob && ex.prob[t] && ex.prob[t].length === trials[t].x.length) ? ex.prob[t] : null);
        state.fileProb = state.prob.map(p => p ? new Float32Array(p) : null);
        state.hasProb = state.prob.some(Boolean);
        state.fileEvents = Array.from({ length: n }, (_, t) => {
            if (ex.predLabels && ex.predLabels[t]) return UnEyeProb.eventsFromLabels(ex.predLabels[t]);
            if (state.prob[t]) return UnEyeProb.eventsFromProb(state.prob[t], rateHz, { threshold: state.probThreshold, mergeMs: pp.merge_ms !== undefined ? pp.merge_ms : 5, minDurMs: pp.min_dur_ms !== undefined ? pp.min_dur_ms : 5 }).map(e => ({ onset: e.onset, offset: e.offset, cls: CLS_SACCADE }));
            return null;
        });
        updateProbUI();
    }

    // the near-miss buttons exist only when the file gave a probability trace
    function updateProbUI() {
        if (elements.nearMissBtn) elements.nearMissBtn.classList.toggle('hidden', !state.hasProb);
        if (elements.nearMissMenuGroup) elements.nearMissMenuGroup.classList.toggle('hidden', !state.hasProb);
        if (elements.nearMissBtn) elements.nearMissBtn.textContent = state.nearMiss.active ? 'Exit near misses' : 'Near misses';
        if (elements.nearMissMenuBtn) elements.nearMissMenuBtn.textContent = state.nearMiss.active ? 'Exit near misses' : 'Near misses';
        if (elements.nearMissBtn) elements.nearMissBtn.classList.toggle('btn-accent', state.nearMiss.active);
        if (elements.acceptPreBtn) elements.acceptPreBtn.textContent = state.nearMiss.active ? 'Accept' : 'Accept Pre-label';
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

        // 2. Saccades: the file's own detection (pred_labels / prob), or the HMM on the raw trace (unit-free score; the display filter is not used)
        const fileEv = mode === 'file' ? (state.fileEvents[trialIdx] || []) : null;
        if (fileEv) for (const e of fileEv) {
            if (e.cls === CLS_BLINK && !pre.some(b => b.cls === CLS_BLINK && !(e.offset < b.onset || e.onset > b.offset))) pre.push({ onset: e.onset, offset: e.offset, cls: CLS_BLINK });
        }
        for (const e of (fileEv ? fileEv.filter(e => e.cls === CLS_SACCADE) : UnEyeHMM.detect(tr.x, tr.y, tr.valid, state.rateHz))) {
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
        syncMainModelSelectUI();
    }

    function syncMainModelSelectUI() {
        const sel = elements.mainModelSelect;
        if (!sel) return;
        const optHmm = sel.querySelector('option[value="hmm"]');
        const optSession = sel.querySelector('option[value="session"]');
        const optCustom = sel.querySelector('option[value="custom"]');
        if (optHmm) {
            optHmm.textContent = (state.fileEvents && state.fileEvents.some(Boolean)) ? 'Predictions: HMM / File' : 'Predictions: HMM';
        }
        if (optSession) {
            optSession.hidden = !state.sessionTrainedModelBin;
        }
        if (optCustom) {
            optCustom.hidden = !state.customModelBin;
            optCustom.textContent = state.customModelName ? `Predictions: ${state.customModelName}` : 'Predictions: Custom weights (.bin)';
        }
        if (state.netSuggestions && state.trainedModelBin) {
            if (state.activeModelSource === 'custom' && state.customModelBin) sel.value = 'custom';
            else if (state.activeModelSource === 'session' && state.sessionTrainedModelBin) sel.value = 'session';
            else sel.value = 'pretrained';
        } else {
            sel.value = state.prelabelerMode === 'none' ? 'none' : 'hmm';
        }
    }

    // automatic training: only when a pass 2 is finished (the durations are refined then), never during pass 1
    function maybeAutoTrain() {
        if (state.isTraining || !state.trials.length) return;
        if (totalReviewedSeconds() - state.reviewedAtLastTraining >= state.minNewSecondsForAutoTrain) runTrainingRound();
    }

    // --- State Operations: Point Marks & Refinement ---
    function saveUndo(trialIdx, meta = {}) {
        if (!state.undoStack[trialIdx]) state.undoStack[trialIdx] = [];      // e.g. a session restored from the autosave
        if (!state.globalUndoStack) state.globalUndoStack = [];
        const snap = {
            trialIdx,
            pointMarks: JSON.parse(JSON.stringify(state.pointMarks[trialIdx] || [])),
            expertEvents: JSON.parse(JSON.stringify(state.expertEvents[trialIdx] || [])),
            prelabels: JSON.parse(JSON.stringify(state.prelabels[trialIdx] || [])),
            rejectedNearMisses: JSON.parse(JSON.stringify((state.rejectedNearMisses && state.rejectedNearMisses[trialIdx]) || [])),
            scanWins: JSON.parse(JSON.stringify((state.scanWins && state.scanWins[trialIdx]) || [])),
            nearMiss: meta.nearMiss || null,
        };
        state.undoStack[trialIdx].push(snap);
        if (state.undoStack[trialIdx].length > 30) state.undoStack[trialIdx].shift();
        state.globalUndoStack.push(snap);
        if (state.globalUndoStack.length > 50) state.globalUndoStack.shift();
    }

    function undo(trialIdx) {
        let snap = null;
        if (!state.globalUndoStack) state.globalUndoStack = [];
        if (state.nearMiss.active || trialIdx === undefined || (!state.undoStack[trialIdx]?.length && state.globalUndoStack.length)) {
            if (state.globalUndoStack.length === 0) return;
            snap = state.globalUndoStack.pop();
            const t = snap.trialIdx;
            const tStack = state.undoStack[t];
            if (tStack && tStack.length) {
                const idx = tStack.lastIndexOf(snap);
                if (idx >= 0) tStack.splice(idx, 1);
                else tStack.pop();
            }
        } else {
            const stack = state.undoStack[trialIdx];
            if (!stack || stack.length === 0) {
                if (state.globalUndoStack.length === 0) return;
                snap = state.globalUndoStack.pop();
            } else {
                snap = stack.pop();
                const gIdx = state.globalUndoStack.lastIndexOf(snap);
                if (gIdx >= 0) state.globalUndoStack.splice(gIdx, 1);
            }
        }
        if (!snap) return;

        const t = snap.trialIdx;
        state.pointMarks[t] = snap.pointMarks;
        state.expertEvents[t] = snap.expertEvents;
        if (snap.prelabels !== undefined && state.prelabels) {
            state.prelabels[t] = snap.prelabels;
        }
        if (snap.rejectedNearMisses !== undefined && state.rejectedNearMisses) {
            state.rejectedNearMisses[t] = snap.rejectedNearMisses;
        }
        if (snap.scanWins !== undefined && state.scanWins) {
            state.scanWins[t] = snap.scanWins;
        }

        if (snap.nearMiss) {
            const { candidate, idx } = snap.nearMiss;
            const alreadyInQueue = state.nearMiss.queue.some(
                c => c.trial === candidate.trial && c.at === candidate.at
            );
            if (!alreadyInQueue) {
                const insertIdx = Math.min(idx, state.nearMiss.queue.length);
                state.nearMiss.queue.splice(insertIdx, 0, candidate);
            }
            if (state.nearMiss.active) {
                state.currentTrialIdx = candidate.trial;
                state.nearMiss.idx = Math.min(idx, state.nearMiss.queue.length - 1);
                state.zoomMultiplier = 1.0;
                resetGesture();
                nmFocus();
            }
        }

        recomputeReviewed(t, false);
        syncRenderedLabels(t);

        if (state.nearMiss.active) {
            nmFocus();
        } else if (state.pass === PASS_TWO) {
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

    function refineMark(trialIdx, at, onset, offset, cls, skipUndo = false) {
        if (!skipUndo) saveUndo(trialIdx);
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

        if (state.rejectedNearMisses && state.rejectedNearMisses[trialIdx]) {
            state.rejectedNearMisses[trialIdx] = state.rejectedNearMisses[trialIdx].filter(
                r => r.offset < onset || r.onset > offset
            );
        }

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
        } else if (state.scanWins[trialIdx]) {
            if (!state.scanWins[trialIdx].some(r => r[0] <= onset && r[1] >= offset)) {
                state.scanWins[trialIdx].push([onset, offset]);
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
        if (state.rejectedNearMisses && state.rejectedNearMisses[trialIdx]) {
            state.rejectedNearMisses[trialIdx] = state.rejectedNearMisses[trialIdx].filter(r => r.offset < a || r.onset > b);
        }
        syncRenderedLabels(trialIdx);
        recomputeReviewed(trialIdx, true);
        if (state.pass === PASS_TWO && !state.nearMiss.active) {
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
        state.expertEvents[trialIdx] = state.expertEvents[trialIdx].filter(e => !((e.onset <= at && at <= e.offset) || Math.abs(e.onset - at) <= 15 || Math.abs(e.offset - at) <= 15));
        if (state.prelabels[trialIdx]) {
            state.prelabels[trialIdx] = state.prelabels[trialIdx].filter(p => !((p.onset <= at && at <= p.offset) || Math.abs(p.onset - at) <= 25 || Math.abs(p.offset - at) <= 15));
        }
        if (state.rejectedNearMisses && state.rejectedNearMisses[trialIdx]) {
            state.rejectedNearMisses[trialIdx] = state.rejectedNearMisses[trialIdx].filter(r => !(r.onset <= at && at <= r.offset) && Math.abs(r.at - at) > 25);
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
        if (state.nearMiss.active) { state.nearMiss.active = false; updateProbUI(); }
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

    // Pass 2 shows all events (confirmed/pre-labeled expert events and pending point marks in scanned windows)
    function trialQueue(t) {
        const items = [];
        const exps = state.expertEvents[t] || [];
        for (let i = 0; i < exps.length; i++) {
            const e = exps[i];
            const mid = Math.round((e.onset + e.offset) / 2);
            items.push({ at: mid, onset: e.onset, offset: e.offset, cls: e.cls, isExpert: true });
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
        if (state.nearMiss.active) { nmFocus(); return; }
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

    // --- Near misses: potential events the network did not detect (P(saccade) peak below the threshold), all trials, highest peak first ---
    // A sub-mode of pass 2 (same frame, handles, gestures); Accept = an expert saccade with the current bounds, Not an Event = rejected
    // (kept, saved) and the window on screen counts as seen (reviewed as fixation).
    function nearMissExclusions(t) {
        const ex = [];
        for (const e of state.expertEvents[t] || []) ex.push([e.onset, e.offset]);
        for (const p of state.prelabels[t] || []) if (p.cls === CLS_SACCADE) ex.push([p.onset, p.offset]);
        for (const r of state.rejectedNearMisses[t] || []) ex.push([r.onset, r.offset]);
        return ex;
    }
    function buildNearMissQueue() {
        state.nearMiss.queue = UnEyeProb.buildNearMissQueue(state.prob, { lower: state.nearMiss.lower, threshold: state.probThreshold, excludeOf: nearMissExclusions });
        state.nearMiss.idx = 0;
        return state.nearMiss.queue;
    }
    function enterNearMiss() {
        if (!state.hasProb || !state.trials.length) return;
        if (state.pass !== PASS_TWO) setPass(PASS_TWO);
        resetGesture();
        state.nearMiss.active = true;
        state.zoomMultiplier = 1.0;
        buildNearMissQueue();
        elements.passIndicator.textContent = `Near misses (P < ${state.probThreshold})`;
        elements.passIndicator.className = "badge pass-badge pass-two";
        updateProbUI();
        nmFocus();
        updateStatus();
        render();
    }
    function exitNearMiss() {
        if (!state.nearMiss.active) return;
        resetGesture();
        state.nearMiss.active = false;
        updateProbUI();
        elements.passIndicator.textContent = "Pass 2: Refine (Onset / Offset)";
        state.zoomMultiplier = 1.0;
        buildRefineQueue();
        state.refineIdx = 0;
        focusCurrentMark();
        updateStatus();
        render();
    }
    const nmOverlapsExpert = c => (state.expertEvents[c.trial] || []).some(e => !(c.offset < e.onset || c.onset > e.offset));
    function nmFocus() {
        const nm = state.nearMiss, q = nm.queue;
        if (nm.idx >= q.length) nm.idx = q.length - 1;
        if (nm.idx < 0) nm.idx = 0;
        while (q.length && nmOverlapsExpert(q[nm.idx])) {          // labelled in the meantime (pass 1, a gesture): no longer a near miss
            q.splice(nm.idx, 1);
            if (nm.idx >= q.length) nm.idx = q.length - 1;
            if (nm.idx < 0) nm.idx = 0;
        }
        if (!q.length) {
            state.currentMarkAt = -1;
            state.currentPutativeOnset = -1;
            state.currentPutativeOffset = -1;
            if (elements.eventCounterBadge) elements.eventCounterBadge.textContent = 'No near misses left ✓';
            return;
        }
        const c = q[nm.idx];
        if (state.currentTrialIdx !== c.trial) { state.currentTrialIdx = c.trial; state.undoStack[c.trial] = state.undoStack[c.trial] || []; }
        state.activeClass = CLS_SACCADE;
        state.currentMarkAt = c.at;
        state.currentPutativeOnset = c.onset;
        state.currentPutativeOffset = c.offset;
        const mult = state.zoomMultiplier || 1.0;
        const pad50ms = Math.round(50.0 * (state.rateHz / 1000.0) * mult);
        const evCenter = Math.round((c.onset + c.offset) / 2);
        const halfSpan = Math.max(pad50ms, Math.round((c.offset - c.onset) / 2) + pad50ms);
        state.fromSample = Math.max(0, evCenter - halfSpan);
        state.toSample = Math.min(curLen() - 1, evCenter + halfSpan);
        if (elements.eventCounterBadge) elements.eventCounterBadge.textContent = nmLabel();
    }
    function nmLabel() {
        const q = state.nearMiss.queue, c = q[state.nearMiss.idx];
        return c ? `Near miss ${state.nearMiss.idx + 1}/${q.length} (p = ${(Math.floor(c.peak * 100) / 100).toFixed(2)})` : 'No near misses left ✓';      // truncated: 0.499 reads 0.49, not 0.50
    }
    // Accept: an expert saccade with the given bounds (as a pass-2 refinement), then the next candidate
    function nmAccept(onset, offset) {
        const nm = state.nearMiss, c = nm.queue[nm.idx];
        if (!c) return false;
        if (onset === undefined) { onset = state.currentPutativeOnset >= 0 ? state.currentPutativeOnset : c.onset; offset = state.currentPutativeOffset >= 0 ? state.currentPutativeOffset : c.offset; }
        const t = c.trial;
        saveUndo(t, {
            nearMiss: {
                action: 'accept',
                candidate: { onset: c.onset, offset: c.offset, at: c.at, peak: c.peak, trial: c.trial },
                idx: nm.idx,
            }
        });
        refineMark(t, c.at, onset, offset, CLS_SACCADE, true);
        nm.queue.splice(nm.idx, 1);
        state.zoomMultiplier = 1.0;
        nmFocus();
        return true;
    }
    // Not an Event: rejected (kept in the session and the saved labels), the window on screen counts as seen (reviewed as fixation)
    function nmReject() {
        const nm = state.nearMiss, c = nm.queue[nm.idx];
        if (!c) return false;
        const t = c.trial;
        saveUndo(t, {
            nearMiss: {
                action: 'reject',
                candidate: { onset: c.onset, offset: c.offset, at: c.at, peak: c.peak, trial: c.trial },
                idx: nm.idx,
            }
        });
        if (!state.rejectedNearMisses[t]) state.rejectedNearMisses[t] = [];
        state.rejectedNearMisses[t].push({ onset: c.onset, offset: c.offset, at: c.at, peak: Math.round(c.peak * 1e4) / 1e4 });
        if (state.scanWins[t] && t === state.currentTrialIdx) {
            const f = Math.max(0, state.fromSample), e = Math.min(trialLenOf(t) - 1, state.toSample);
            if (!state.scanWins[t].some(r => r[0] === f && r[1] === e)) state.scanWins[t].push([f, e]);
        }
        recomputeReviewed(t, true);
        nm.queue.splice(nm.idx, 1);
        state.zoomMultiplier = 1.0;
        nmFocus();
        return true;
    }
    function nmStep(dir) {
        const nm = state.nearMiss;
        if (!nm.queue.length) return;
        nm.idx = Math.max(0, Math.min(nm.queue.length - 1, nm.idx + dir));
        state.zoomMultiplier = 1.0;
        resetGesture();
        nmFocus();
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

    function render(recordViewed = true) {
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
        if (state.netSuggestions && state.trainedModelBin && !state.modelPredictedTrials.has(state.currentTrialIdx) && !state.isTraining && !state.predictingTrialInFlight) {
            const targetTrial = state.currentTrialIdx;
            state.predictingTrialInFlight = true;
            predictSingleTrialWithModel(targetTrial, { asGroundTruthIfEmpty: false, updateProbs: true }).then(() => {
                state.predictingTrialInFlight = false;
                updateProbUI();
                render(false);
            }).catch(() => {
                state.predictingTrialInFlight = false;
            });
        }
        if (recordViewed !== false && (!elements.viewedModal || elements.viewedModal.classList.contains('hidden'))) {
            markViewed();
        }

        // Layout lanes (2 lanes: position X and Y on the same axes, then speed) with top margin in Pass 2 so header never collides with the trace
        const topMargin = state.pass === PASS_TWO ? 26 : 0;
        const availH = height - topMargin;
        const posLane = { top: topMargin, bottom: topMargin + availH * 0.66 };
        // a file with the network's P(saccade): a thin band at the bottom (0..1), under the speed
        const probArr = state.prob[state.currentTrialIdx] || null;
        const probH = probArr ? Math.max(30, Math.min(64, Math.round(availH * 0.14))) : 0;
        const speedLane = { top: posLane.bottom, bottom: height - probH };

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
        if (probArr) drawProbBand(probArr, { top: height - probH, bottom: height }, width);

        // 5. Draw Prelabels & Annotations
        drawAnnotations(width, height);

    }

    // every window drawn on screen is recorded as viewed (merged ranges per trial, autosaved, saved in the labels file)
    function markViewed() {
        const t = state.currentTrialIdx;
        if (!state.trials[t]) return;
        if (!state.viewedRanges[t]) state.viewedRanges[t] = [];
        const n = trialLenOf(t), s = Math.max(0, Math.min(n - 1, state.fromSample)), e = Math.max(0, Math.min(n - 1, state.toSample));
        if (e < s) return;
        const next = UnEyeProb.addRange(state.viewedRanges[t], s, e, n);
        if (next !== state.viewedRanges[t]) { state.viewedRanges[t] = next; saveToIndexedDB(); }
    }

    // P(saccade) of the network (file variable prob): line 0..1, dashed line at the threshold; in near-miss mode a dotted line at the lower bound
    function drawProbBand(prob, lane, w) {
        const from = state.fromSample, last = Math.min(state.toSample, prob.length - 1);
        const y0 = lane.bottom - 3, y1 = lane.top + 4, yOf = p => y0 - Math.max(0, Math.min(1, p)) * (y0 - y1);
        ctx.save();
        ctx.fillStyle = 'rgba(255, 243, 224, 0.55)';
        ctx.fillRect(0, lane.top, w, lane.bottom - lane.top);
        ctx.strokeStyle = '#ECEFF1'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(0, lane.top + 0.5); ctx.lineTo(w, lane.top + 0.5); ctx.stroke();
        const thr = state.probThreshold;
        ctx.strokeStyle = 'rgba(230, 81, 0, 0.75)'; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(0, yOf(thr)); ctx.lineTo(w, yOf(thr)); ctx.stroke();
        if (state.nearMiss.active) {
            ctx.strokeStyle = 'rgba(230, 81, 0, 0.45)'; ctx.setLineDash([2, 3]);
            ctx.beginPath(); ctx.moveTo(0, yOf(state.nearMiss.lower)); ctx.lineTo(w, yOf(state.nearMiss.lower)); ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.font = '500 11px var(--font-stack)';
        ctx.fillStyle = '#E65100';
        ctx.textAlign = 'left';
        const probTitle = (state.netSuggestions && state.trainedModelBin && state.activeModelName)
            ? `P(saccade) · ${state.activeModelName}`
            : 'P(saccade)';
        ctx.fillText(probTitle, 16, lane.top + 12);
        ctx.textAlign = 'right';
        ctx.fillText(String(thr), w - 16, yOf(thr) - 2);
        ctx.strokeStyle = '#EF6C00';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        if (last - from + 1 > 2 * w) {                  // one min-max pair per pixel column; NaN breaks the line
            let col = NaN, lo = 0, hi = 0;
            const flush = () => { const px = col + 0.5; if (!started) { ctx.moveTo(px, yOf(lo)); started = true; } else ctx.lineTo(px, yOf(lo)); if (hi !== lo) ctx.lineTo(px, yOf(hi)); };
            for (let i = from; i <= last; i++) {
                const v = prob[i];
                if (v !== v) { if (col === col) flush(); col = NaN; started = false; continue; }
                const c = Math.floor(sampleToPx(i, w));
                if (c !== col) { if (col === col) flush(); col = c; lo = hi = v; } else { if (v < lo) lo = v; if (v > hi) hi = v; }
            }
            if (col === col) flush();
        } else {
            for (let i = from; i <= last; i++) {
                const v = prob[i];
                if (v !== v) { started = false; continue; }
                const px = sampleToPx(i, w), py = yOf(v);
                if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
            }
        }
        ctx.stroke();
        ctx.restore();
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

        // 1. Prelabels (Dimmed brackets - when viewing a neural network model's predictions, also draw dashed orange brackets so they can be compared against existing ground-truth expertEvents)
        const showingModelPreds = Boolean(state.netSuggestions && state.trainedModelBin);
        for (const p of state.prelabels[trialIdx]) {
            if (p.offset < state.fromSample || p.onset > state.toSample) continue;
            const anyOverlap = (state.expertEvents[trialIdx] || []).some(e => !(p.offset < e.onset || p.onset > e.offset));
            if (!showingModelPreds && anyOverlap) continue;
            const x1 = sampleToPx(p.onset, w);
            const x2 = sampleToPx(p.offset, w);
            if (showingModelPreds && anyOverlap) {
                ctx.strokeStyle = p.cls === CLS_SACCADE ? 'rgba(230, 81, 0, 0.85)' : 'rgba(198, 40, 40, 0.85)';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 3]);
                ctx.strokeRect(x1, 14, Math.max(2, x2 - x1), h - 28);
                ctx.setLineDash([]);
            } else {
                ctx.strokeStyle = p.cls === CLS_SACCADE ? (showingModelPreds ? 'rgba(230, 81, 0, 0.75)' : 'rgba(21, 101, 192, 0.45)') : 'rgba(198, 40, 40, 0.45)';
                ctx.lineWidth = 2;
                ctx.setLineDash([4, 4]);
                ctx.strokeRect(x1, 10, Math.max(2, x2 - x1), h - 20);
                ctx.setLineDash([]);
            }
        }

        // 1b. Rejected near misses: an orange bar at the top
        for (const r of (state.rejectedNearMisses[trialIdx] || [])) {
            if (r.offset < state.fromSample || r.onset > state.toSample) continue;
            const x1 = sampleToPx(r.onset, w), x2 = sampleToPx(r.offset, w);
            ctx.fillStyle = 'rgba(251, 140, 0, 0.8)';
            ctx.fillRect(x1, 4, Math.max(3, x2 - x1), 5);
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
            const bannerText = state.nearMiss.active
                ? (w < 520 ? `${nmLabel()} [${onset}..${offset}]` : `${nmLabel()}: trial ${state.currentTrialIdx + 1}, [${onset}..${offset}] (${durMs} ms) — Accept, drag the bounds, or Not an Event`)
                : w < 520
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

            // Highlight active grabbed handle & draw shortcut key badges on handles
            if (state.handleDrag && state.handleDrag.current) {
                const gx = state.handleDrag.side === 'on' ? x1 : x2;
                ctx.save();
                ctx.strokeStyle = '#FFA000';
                ctx.fillStyle = 'rgba(255, 193, 7, 0.35)';
                ctx.lineWidth = 2.5;
                ctx.beginPath();
                ctx.arc(gx, h / 2, HANDLE_R_PX + 6, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
                ctx.restore();
            }
            const onKey = state.shortcuts && state.shortcuts.grabOnset ? formatShortcutKey(state.shortcuts.grabOnset) : '';
            const offKey = state.shortcuts && state.shortcuts.grabOffset ? formatShortcutKey(state.shortcuts.grabOffset) : '';
            if (onKey || offKey) {
                ctx.save();
                ctx.fillStyle = '#FFFFFF';
                ctx.font = 'bold 9px var(--font-stack)';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                if (onKey && onKey.length <= 2) ctx.fillText(onKey, x1, h / 2);
                if (offKey && offKey.length <= 2) ctx.fillText(offKey, x2, h / 2);
                ctx.restore();
            }
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

        canvas.addEventListener('pointerleave', () => {
            state.hoverSample = -1;
            state.hoverPos = null;
        });

        canvas.addEventListener('pointerdown', (e) => {
            canvasRect = canvas.getBoundingClientRect();
            if (state.handleDrag && state.handleDrag.viaKey) {
                const sample = pxToSample(local(e).x, canvas.clientWidth);
                const hd = state.handleDrag;
                if (hd.side === 'on') hd.onset = Math.max(0, Math.min(sample, hd.offset - 1));
                else hd.offset = Math.min(curLen() - 1, Math.max(sample, hd.onset + 1));
                if (hd.current) { state.currentPutativeOnset = hd.onset; state.currentPutativeOffset = hd.offset; }
                commitHandleDrag(hd);
                state.handleDrag = null;
                updateStatus();
                render();
                return;
            }
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
            const r = canvas.getBoundingClientRect();
            const lx = e.clientX - r.left, ly = e.clientY - r.top;
            const inside = lx >= 0 && lx <= r.width && ly >= 0 && ly <= r.height;
            if (inside && r.width > 0) {
                state.hoverSample = pxToSample(lx, r.width);
                state.hoverPos = { x: lx, y: ly };
            }
            if (state.handleDrag && state.handleDrag.viaKey && r.width > 0) {
                const sample = pxToSample(lx, r.width);
                const hd = state.handleDrag;
                if (hd.side === 'on') hd.onset = Math.max(0, Math.min(sample, hd.offset - 1));
                else hd.offset = Math.min(curLen() - 1, Math.max(sample, hd.onset + 1));
                if (hd.current) { state.currentPutativeOnset = hd.onset; state.currentPutativeOffset = hd.offset; }
                hd.moved = true;
                requestRender();
                return;
            }
            if (inside && state.pointers.size === 0 && !state.pointerDown) {
                const h = handleAt({ x: lx, y: ly });
                canvas.style.cursor = h ? 'ew-resize' : (state.eraser ? 'cell' : 'crosshair');
            }
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
                        if (offset - onset >= 5 && state.nearMiss.active && state.currentMarkAt >= 0) {
                            nmAccept(onset, offset);          // near miss: the dragged span is the accepted saccade
                        } else if (offset - onset >= 5) {
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
        if (!hd.ev && state.nearMiss.active) {          // near miss: the bounds move, Accept stores them
            const c = state.nearMiss.queue[state.nearMiss.idx];
            if (c) { c.onset = hd.onset; c.offset = hd.offset; }
            return;
        }
        if (hd.ev) {
            saveUndo(t);
            const i = state.expertEvents[t].indexOf(hd.ev);
            if (i >= 0) state.expertEvents[t][i] = { onset: hd.onset, offset: hd.offset, cls: hd.ev.cls, fromPrelabel: false };
            if (state.prelabels[t]) {
                const lo = Math.min(hd.on0, hd.onset), hi = Math.max(hd.off0, hd.offset);
                state.prelabels[t] = state.prelabels[t].filter(p => p.offset < lo || p.onset > hi);
            }
            if (state.scanWins[t] && !state.scanWins[t].some(r => r[0] <= hd.onset && r[1] >= hd.offset)) {
                state.scanWins[t].push([hd.onset, hd.offset]);
            }
            mergeOverlappingEvents(t);
            syncRenderedLabels(t);
            recomputeReviewed(t, true);
        } else {
            refineMark(t, state.currentMarkAt, hd.onset, hd.offset, state.activeClass);      // the proposal under refinement becomes a refined event
        }
        if (state.pass !== PASS_TWO || state.nearMiss.active) return;
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
            if (state.pass === PASS_TWO && state.currentMarkAt >= 0 && state.nearMiss.active) {
                nmAccept();
                updateStatus();
                render();
                return;
            }
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

            // 2b. Tap on existing rejected near miss removes it (cancels the rejection)
            const existingRejIdx = (state.rejectedNearMisses[trialIdx] || []).findIndex(r => (r.onset <= sample && sample <= r.offset) || Math.abs(r.onset - sample) <= 15 || Math.abs(r.offset - sample) <= 15 || Math.abs(r.at - sample) <= 15);
            if (existingRejIdx !== -1) {
                saveUndo(trialIdx);
                state.rejectedNearMisses[trialIdx].splice(existingRejIdx, 1);
                recomputeReviewed(trialIdx, false);
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
        const counts = totalExpertEventCounts();
        const hasGroundTruthEvents = (counts.saccades + counts.blinks) > 0;
        if (reviewedS < 0.5 && !hasGroundTruthEvents) {
            alert("Please label at least 0.5 seconds of reviewed data before training.");
            return;
        }

        state.isTraining = true;
        elements.trainStatus.className = 'train-active';
        elements.trainStatus.textContent = "Preparing training windows…";
        if (elements.cancelTrainBtn) elements.cancelTrainBtn.classList.remove('hidden');
        await new Promise(r => setTimeout(r, 10));

        // Keep screen awake during training (iPad Safari & Chrome)
        try {
            if ('wakeLock' in navigator) {
                state.wakeLock = await navigator.wakeLock.request('screen');
            }
        } catch (e) {
            console.log("WakeLock:", e);
        }

        // Build candidate trial order: current & reviewed/viewed trials first, then trials with ground-truth expertEvents
        const nTrials = state.trials.length;
        const winSamples = Math.max(25, Math.round(state.rateHz / 25.0) * 25);
        const maxTrainSamples = Math.round(50.0 * state.rateHz);
        const maxTestSamples = Math.round(20.0 * state.rateHz);

        const trialPriority = (t) => {
            const revLen = UnEyeProb.rangeLength(state.reviewedRanges[t]);
            const viewLen = UnEyeProb.rangeLength(state.viewedRanges[t]);
            const evCount = (state.expertEvents[t] || []).length;
            const dist = Math.abs(t - state.currentTrialIdx);
            if (revLen >= winSamples) return 0 + dist * 1e-4;
            if (viewLen >= winSamples) return 10 + dist * 1e-4;
            if (revLen > 0 || viewLen > 0) return 20 + dist * 1e-4;
            if (evCount > 0) return 30 + dist * 1e-4;
            return 1000;
        };

        const orderedTrials = Array.from({ length: nTrials }, (_, t) => t)
            .filter(t => trialPriority(t) < 1000)
            .sort((a, b) => trialPriority(a) - trialPriority(b));

        // Helper to build a labeled trace for trial t.
        // If reviewed/viewed ranges cover at least a window (with both fixation and event context), mask to those ranges;
        // otherwise if the trial has ground-truth expertEvents (e.g. pre-labels loaded as ground truth or after resetting viewed periods),
        // use the full trial ground-truth labels (humanLabels(t): 0=fixation, 1=saccade, 2=blink).
        const buildTrialTrainingData = (t, seedBase) => {
            const tr = state.trials[t];
            if (!tr || tr.x.length < winSamples) return null;
            const hl = humanLabels(t);
            const marks = state.pointMarks[t] || [];
            const rev = (state.reviewedRanges[t] || []).filter(([f, e]) => !marks.some(m => m.at >= f && m.at <= e));
            const view = (state.viewedRanges[t] || []).filter(([f, e]) => !marks.some(m => m.at >= f && m.at <= e));
            const combinedRanges = mergeRangeList([...rev, ...view], tr.x.length);
            const covSamples = UnEyeProb.rangeLength(combinedRanges);
            const evSamples = (state.expertEvents[t] || []).reduce((acc, e) => acc + (e.offset - e.onset + 1), 0);

            // Check whether combinedRanges includes fixation context beyond just the event spans themselves
            const hasFixationContext = covSamples >= Math.min(tr.x.length, Math.max(Math.round(0.4 * state.rateHz), evSamples + Math.round(0.2 * state.rateHz)));
            let endSample = tr.x.length;
            const lab = new Uint8Array(endSample).fill(255);
            if (hasFixationContext) {
                let maxEnd = 0;
                for (const [s, e] of combinedRanges) {
                    if (e > maxEnd) maxEnd = e;
                    for (let i = s; i <= e && i < endSample; i++) lab[i] = hl[i];
                }
                for (const ev of (state.expertEvents[t] || [])) {
                    if (ev.offset > maxEnd) maxEnd = ev.offset;
                    for (let i = ev.onset; i <= ev.offset && i < endSample; i++) lab[i] = ev.cls;
                }
                endSample = Math.min(tr.x.length, Math.max(winSamples, maxEnd + 1));
            } else if ((state.expertEvents[t] || []).length > 0 || covSamples > 0) {
                for (let i = 0; i < endSample; i++) lab[i] = hl[i];
            } else {
                return null;
            }
            const tc = trainingCopyOf(tr, seedBase + t);
            return {
                x: tc.x.subarray(0, endSample),
                y: tc.y.subarray(0, endSample),
                labels: lab.subarray(0, endSample),
                rateHz: state.rateHz
            };
        };

        const traces = [];
        let trainSamplesCount = 0;
        for (const t of orderedTrials) {
            if (state.testTrials.has(t)) continue;
            if (trainSamplesCount >= maxTrainSamples && traces.length >= 2) break;
            const item = buildTrialTrainingData(t, 1000);
            if (item) {
                traces.push(item);
                trainSamplesCount += item.x.length;
            }
        }

        const testTrials = [];
        let testSamplesCount = 0;
        for (const t of orderedTrials) {
            if (!state.testTrials.has(t)) continue;
            if (testSamplesCount >= maxTestSamples && testTrials.length >= 2) break;
            const item = buildTrialTrainingData(t, 2000);
            if (item) {
                testTrials.push(item);
                testSamplesCount += item.x.length;
            }
        }

        if (traces.length === 0) {
            finishTraining("No non-test labeled trials to train on");
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
        const maxEpochs = 30;
        const trainSec = trainSamplesCount / state.rateHz;

        let lastEpoch = 0;
        worker.onmessage = async function(evt) {
            const msg = evt.data;
            if (msg.id !== trainId) return;

            if (msg.type === 'STATUS') {
                elements.trainStatus.className = 'train-active';
                elements.trainStatus.textContent = msg.message;
            } else if (msg.type === 'PROGRESS') {
                const p = msg.data;
                lastEpoch = p.epoch;
                const etaStr = p.eta_s > 0 ? ` · ETA ${p.eta_s.toFixed(0)}s` : '';
                elements.trainStatus.className = 'train-active';
                elements.trainStatus.textContent = `Epoch ${p.epoch}/${maxEpochs} · Loss: ${p.train_loss.toFixed(3)} · Val F1: ${p.val_event_f1.toFixed(2)}${etaStr}`;
            } else if (msg.type === 'CANCELLED') {
                finishTraining("Training cancelled");
            } else if (msg.type === 'ERROR') {
                finishTraining("Training error: " + msg.message);
                alert("Training error: " + msg.message);
            } else if (msg.type === 'TRAIN_COMPLETE') {
                state.trainedModelBin = msg.binModel;
                state.trainedModelJson = msg.jsonSidecar;
                state.sessionTrainedModelBin = msg.binModel;
                state.activeModelSource = 'session';
                state.activeModelName = 'session trained';
                state.modelPredictedTrials = new Set();
                syncMainModelSelectUI();

                const curCounts = totalExpertEventCounts();
                const m = msg.metrics || {};
                const round = {
                    atMs: Date.now(),
                    labeledSeconds: Math.max(reviewedS, trainSec),
                    epochs: lastEpoch,
                    valF1: m.testF1Saccade || 0.0,
                    testF1Saccade: m.testF1Saccade || 0.0,
                    testF1Blink: m.testF1Blink || 0.0,
                    testKappaSaccade: m.testKappaSaccade || 0.0,
                    testKappaBlink: m.testKappaBlink || 0.0,
                    nSaccades: curCounts.saccades,
                    nBlinks: curCounts.blinks,
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

        const useRotAug = trainSec < 25.0;
        worker.postMessage({
            type: 'START_TRAIN',
            id: trainId,
            data: {
                config: {
                    n_classes: 3,
                    max_seconds_of_data: 50.0,
                    rotation_aug: useRotAug,
                    rotations: useRotAug ? 2 : 0,
                    val_fraction: 0.2,
                    max_speed_deg_s: 1300.0,
                    max_epochs: maxEpochs,
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

    function loadBytesIntoWorker(bytes) {
        if (!bytes || !bytes.length) return Promise.resolve(false);
        const worker = getTrainWorker();
        return new Promise((resolve) => {
            const reqId = Date.now() + Math.random();
            const timer = setTimeout(() => {
                worker.removeEventListener('message', handler);
                resolve(false);
            }, 15000);
            const handler = (e) => {
                if (e.data && e.data.id === reqId && e.data.type === 'LOAD_MODEL_RESULT') {
                    clearTimeout(timer);
                    worker.removeEventListener('message', handler);
                    resolve(Boolean(e.data.success));
                }
            };
            worker.addEventListener('message', handler);
            worker.postMessage({ type: 'LOAD_MODEL', id: reqId, data: bytes });
        });
    }

    async function loadPretrainedCombinedModel() {
        try {
            if (!state.pretrainedModelBin) {
                const resp = await fetch('models/combined.bin');
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                const buf = await resp.arrayBuffer();
                state.pretrainedModelBin = new Uint8Array(buf);
            }
            const ok = await loadBytesIntoWorker(state.pretrainedModelBin);
            if (!ok) throw new Error('WASM trainer could not load models/combined.bin');
            state.trainedModelBin = state.pretrainedModelBin;
            state.activeModelSource = 'pretrained';
            state.activeModelName = 'combined.bin';
            state.modelPredictedTrials = new Set();
            syncMainModelSelectUI();
            return true;
        } catch (err) {
            console.error('Failed to load pretrained model:', err);
            alert('Cannot load pretrained model (models/combined.bin): ' + err.message);
            return false;
        }
    }

    async function predictSingleTrialWithModel(t, opts = {}) {
        if (!state.trainedModelBin || !state.trials[t] || state.isTraining) return false;
        const worker = getTrainWorker();
        const rawTr = state.trials[t];
        // same NaN fill as the training copy, otherwise a blink reads as zero velocity and is never predicted
        const tr = trainingCopyOf(rawTr, 3000 + t);
        const includeProbs = opts.updateProbs !== false;
        const res = await new Promise((resolve) => {
            const reqId = Date.now() + Math.random();
            const handler = (e) => {
                if (e.data && e.data.id === reqId && e.data.type === 'PREDICT_RESULT') {
                    worker.removeEventListener('message', handler);
                    resolve({ pred: e.data.pred, prob: e.data.prob });
                }
            };
            worker.addEventListener('message', handler);
            worker.postMessage({
                type: 'PREDICT',
                id: reqId,
                data: { x: tr.x, y: tr.y, rateHz: state.rateHz, maxSpeedDegS: 1300.0, includeProbs }
            });
        });

        const pred = res && res.pred;
        if (!pred) return false;

        const newPre = [];
        const n = pred.length;
        // Include NaN/invalid blink stretches (>= 3 samples) so 2-class models (like combined.bin) still mark blinks
        let blinkStart = -1;
        for (let i = 0; i <= n; i++) {
            const isInvalid = i < n && (rawTr.valid[i] === 0 || isNaN(rawTr.x[i]) || isNaN(rawTr.y[i]));
            if (isInvalid && blinkStart === -1) {
                blinkStart = i;
            } else if (!isInvalid && blinkStart !== -1) {
                if (i - blinkStart >= 3) newPre.push({ onset: blinkStart, offset: i - 1, cls: CLS_BLINK });
                blinkStart = -1;
            }
        }

        for (let i = 0; i < n;) {
            const c = pred[i];
            if (c === 1 || c === 2) {
                let j = i;
                while (j < n && pred[j] === c) j++;
                if (j - i >= 3) {
                    const onset = i, offset = j - 1;
                    const overlapsBlink = c === CLS_SACCADE && newPre.some(b => b.cls === CLS_BLINK && !(offset < b.onset || onset > b.offset));
                    if (!overlapsBlink) {
                        newPre.push({ onset, offset, cls: c });
                    }
                }
                i = j;
            } else {
                i++;
            }
        }
        newPre.sort((a, b) => a.onset - b.onset);
        state.prelabels[t] = newPre;

        if (opts.asGroundTruthIfEmpty && (!state.expertEvents[t] || state.expertEvents[t].length === 0) && newPre.length > 0) {
            state.expertEvents[t] = newPre.map(e => ({ onset: e.onset, offset: e.offset, cls: e.cls, fromPrelabel: false }));
            mergeOverlappingEvents(t);
            recomputeReviewed(t, false);
        }

        if (includeProbs && res.prob && res.prob.length === n) {
            const pArr = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                pArr[i] = (!rawTr.valid[i] || isNaN(rawTr.x[i]) || isNaN(rawTr.y[i])) ? NaN : res.prob[i];
            }
            state.prob[t] = pArr;
            state.hasProb = true;
        }

        state.modelPredictedTrials.add(t);
        syncRenderedLabels(t);
        return true;
    }

    async function applyModelAsPrelabeler(opts = {}) {
        if (!state.trainedModelBin || state.trials.length === 0) return;
        state.previousPrelabels = JSON.parse(JSON.stringify(state.prelabels));

        // On large datasets (> 60 trials), predict around the current trial so we don't block the worker for minutes
        const nTotal = state.trials.length;
        const indices = [];
        if (nTotal <= 60) {
            for (let t = 0; t < nTotal; t++) indices.push(t);
        } else {
            const cur = state.currentTrialIdx;
            indices.push(cur);
            for (let d = 1; indices.length < 25 && (cur - d >= 0 || cur + d < nTotal); d++) {
                if (cur + d < nTotal) indices.push(cur + d);
                if (cur - d >= 0 && indices.length < 25) indices.push(cur - d);
            }
        }

        for (const t of indices) {
            if (state.isTraining) break;
            await predictSingleTrialWithModel(t, opts);
        }
        updateProbUI();
        if (state.nearMiss.active) {
            buildNearMissQueue();
            nmFocus();
        } else if (state.pass === PASS_TWO) {
            buildRefineQueue();
            state.refineIdx = Math.min(state.refineIdx, Math.max(0, state.refineQueue.length - 1));
            focusCurrentMark();
        }
        updateStatus();
        render(false);
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
                // the file's probability trace goes with the traces (written once per dataset), as Uint16 (half the size of float32)
                store.put(state.trials.map((tr, t) => ({ x: tr.x, y: tr.y, valid: tr.valid, name: tr.name,
                    probQ: state.prob[t] ? UnEyeProb.encodeProb(state.prob[t]) : null, fileEvents: state.fileEvents[t] || null })), 'autosave_trials');
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
                probThreshold: state.probThreshold,
                viewedRanges: state.viewedRanges,
                rejectedNearMisses: state.rejectedNearMisses,
                rateHz: state.rateHz,
                scale: state.scale,
                convergenceHistory: state.convergenceHistory,
                trainedModelBin: state.trainedModelBin,
                trainedModelJson: state.trainedModelJson,
                sessionTrainedModelBin: state.sessionTrainedModelBin,
                customModelBin: state.customModelBin,
                customModelName: state.customModelName,
                activeModelSource: state.activeModelSource,
                activeModelName: state.activeModelName,
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
                state.prob = savedTr.map(tr => (tr.probQ && tr.probQ.length ? UnEyeProb.decodeProb(tr.probQ) : null));
                state.fileProb = state.prob.map(p => p ? new Float32Array(p) : null);
                state.hasProb = state.prob.some(Boolean);
                state.fileEvents = savedTr.map(tr => tr.fileEvents || null);
                state.probThreshold = data.probThreshold > 0 && data.probThreshold < 1 ? data.probThreshold : 0.5;
                state.viewedRanges = Array.from({ length: n }, (_, t) => (data.viewedRanges && data.viewedRanges[t]) || []);
                state.rejectedNearMisses = Array.from({ length: n }, (_, t) => (data.rejectedNearMisses && data.rejectedNearMisses[t]) || []);
                state.nearMiss.active = false; state.nearMiss.queue = []; state.nearMiss.idx = 0;
                state.modelPredictedTrials = new Set();
                updateProbUI();
                state.undoStack = Array.from({ length: n }, () => []);
                state.globalUndoStack = [];
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
                state.prelabelerMode = data.prelabelerMode === 'none' ? 'none' : ((data.prelabelerMode === 'file' && state.fileEvents.some(Boolean)) ? 'file' : (data.prelabelerMode || 'hmm'));
                state.convergenceHistory = data.convergenceHistory || [];
                state.trainedModelBin = data.trainedModelBin || null;
                state.trainedModelJson = data.trainedModelJson || null;
                state.sessionTrainedModelBin = data.sessionTrainedModelBin || null;
                state.customModelBin = data.customModelBin || null;
                state.customModelName = data.customModelName || '';
                state.activeModelSource = data.activeModelSource || (state.trainedModelBin ? 'session' : 'hmm');
                state.activeModelName = data.activeModelName || '';
                const staleModel = state.trainedModelBin && data.modelVersion !== MODEL_VERSION && !state.customModelBin;
                if (staleModel) {          // trained by an earlier, broken version: start again from the pretrained model
                    state.trainedModelBin = null;
                    state.trainedModelJson = null;
                    state.sessionTrainedModelBin = null;
                    state.convergenceHistory = [];
                    state.reviewedAtLastTraining = 0.0;
                }
                syncMainModelSelectUI();

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
            const parts = line.trim().split(/[,\t; ]+/).map(v => {
                const s = v.replace(/^["']|["']$/g, '').trim();
                const sl = s.toLowerCase();
                if (sl === 'true') return 1;
                if (sl === 'false') return 0;
                const f = parseFloat(s);
                return isNaN(f) ? NaN : f;
            });
            rows.push(new Float32Array(parts));
        }
        return rows;
    }

    let pendingModelFile = null;
    let pendingLabelsFile = null;
    let pendingFileInspection = null;

    function inferRoleForModalVar(varName, info) {
        if (!varName) return 'prelabels';
        const nm = String(varName).toLowerCase();
        if (/pred|pre_?label|propos|auto|detect|hmm|model|unet|uneye/.test(nm)) return 'prelabels';
        const v = info && info.vars ? info.vars.find(it => it.name === varName) : null;
        if (v && v.hasNeg) return 'reviewed';
        if (/^labels?$|human|expert|ground_?truth|gt|manual|annot/.test(nm)) return 'reviewed';
        return 'prelabels';
    }

    function escapeOptionHtml(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function inspectFileForModal(file) {
        if (!elements.modalLabelsVar || !elements.modalProbVar) return;
        elements.modalLabelsVar.innerHTML = '<option value="__auto__">(auto-detecting…)</option><option value="">(none)</option>';
        elements.modalProbVar.innerHTML = '<option value="__auto__">(auto-detecting…)</option><option value="">(none)</option>';
        elements.modalLabelsVar.value = '__auto__';
        elements.modalProbVar.value = '__auto__';
        if (elements.modalLabelsRole) {
            elements.modalLabelsRole.value = 'prelabels';
            elements.modalLabelsRole.disabled = false;
        }
        if (elements.modalFileSummary) {
            elements.modalFileSummary.textContent = 'Inspecting file variables…';
            elements.modalFileSummary.classList.remove('hidden');
        }
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        const promise = (async () => {
            try {
                if (['mat', 'npz', 'h5', 'hdf5', 'hdf'].includes(ext)) {
                    const buf = await file.arrayBuffer();
                    const vars = await UnEyeFormats.readVarsFile(file.name, buf, loadH5Lib);
                    const info = UnEyeFormats.inspectVars ? UnEyeFormats.inspectVars(vars) : null;
                    if (pendingFileToLoad === file && info) {
                        if (info.fs > 0 && elements.rateInput) elements.rateInput.value = String(info.fs);
                        if (info.savgolMs > 0 && elements.modalSgWindow) {
                            const sgStr = String(Math.round(info.savgolMs));
                            if (Array.from(elements.modalSgWindow.options).some(o => o.value === sgStr)) {
                                elements.modalSgWindow.value = sgStr;
                            }
                        }
                        let candidates = info.vars || [];
                        if (!candidates.length && window.UnEyeImport) {
                            const tables = UnEyeImport.tablesFromVars(vars, file.name, ext === 'mat' ? 'mat' : ext === 'npz' ? 'npz' : 'h5');
                            if (tables && tables[0]) {
                                const st = UnEyeImport.describeColumns(tables[0]);
                                const sug = UnEyeImport.suggest(tables[0], st);
                                candidates = tables[0].cols
                                    .map((c, idx) => ({ name: c.name, summary: st[idx].nominal ? 'text' : `${st[idx].min.toFixed(2)}..${st[idx].max.toFixed(2)}`, hasNeg: st[idx].min < 0, nominal: st[idx].nominal }))
                                    .filter(c => !c.nominal);
                                if (!info.defaultLabelsVar && sug.labels >= 0 && tables[0].cols[sug.labels]) {
                                    info.defaultLabelsVar = tables[0].cols[sug.labels].name;
                                    info.defaultLabelsRole = sug.labelsRole || 'prelabels';
                                }
                                if (!info.defaultProbVar && sug.prob >= 0 && tables[0].cols[sug.prob]) {
                                    info.defaultProbVar = tables[0].cols[sug.prob].name;
                                }
                            }
                        }
                        const optsHtml = '<option value="">(none)</option>' + candidates.map(v => `<option value="${escapeOptionHtml(v.name)}">${escapeOptionHtml(v.name)}${v.summary ? ' (' + escapeOptionHtml(v.summary) + ')' : ''}</option>`).join('');
                        elements.modalLabelsVar.innerHTML = optsHtml;
                        elements.modalProbVar.innerHTML = optsHtml;
                        elements.modalLabelsVar.value = info.defaultLabelsVar || '';
                        if (elements.modalLabelsRole) {
                            elements.modalLabelsRole.value = info.defaultLabelsRole || inferRoleForModalVar(info.defaultLabelsVar, info);
                            elements.modalLabelsRole.disabled = !elements.modalLabelsVar.value;
                        }
                        elements.modalProbVar.value = info.defaultProbVar || '';
                        if (elements.modalFileSummary) {
                            const parts = [];
                            if (info.nTrials > 0) parts.push(`${info.nTrials} trial${info.nTrials === 1 ? '' : 's'}`);
                            if (info.nSamples > 0) parts.push(`${info.nSamples.toLocaleString()} samples`);
                            if (info.fs > 0) parts.push(`${info.fs} Hz`);
                            if (candidates.length) parts.push(`variables: ${candidates.map(v => v.name).join(', ')}`);
                            elements.modalFileSummary.textContent = parts.join(' · ');
                            elements.modalFileSummary.classList.toggle('hidden', parts.length === 0);
                        }
                    }
                    return { vars, info };
                } else {
                    const headText = await file.slice(0, 65536).text();
                    const sniff = window.UnEyeImport ? UnEyeImport.sniffHead(headText, file.name) : { plain: true };
                    let colNames = [];
                    if (!sniff.plain && ext !== 'asc' && ext !== 'arff') {
                        const firstLine = (headText.split(/\r?\n/).find(l => l.trim() && !l.trim().startsWith('#')) || '').trim();
                        const delim = firstLine.includes('\t') ? '\t' : firstLine.includes(';') ? ';' : firstLine.includes(',') ? ',' : /\s+/;
                        colNames = firstLine.split(delim).map(s => s.replace(/^["']|["']$/g, '').trim()).filter(Boolean);
                    }
                    if (pendingFileToLoad === file) {
                        const optsHtml = '<option value="">(none)</option>' + colNames.map(nm => `<option value="${escapeOptionHtml(nm)}">${escapeOptionHtml(nm)}</option>`).join('');
                        elements.modalLabelsVar.innerHTML = optsHtml;
                        elements.modalProbVar.innerHTML = optsHtml;
                        const defLbl = colNames.find(nm => /^(pred_?labels?|pre_?labels?|labels?|saccade_?labels?|event_?labels?)$/i.test(nm)) || '';
                        const defProb = colNames.find(nm => /^(prob|probability|probabilities|pred_?prob|p_?saccade|saccade_?prob)$/i.test(nm)) || '';
                        elements.modalLabelsVar.value = defLbl;
                        if (elements.modalLabelsRole) {
                            elements.modalLabelsRole.value = inferRoleForModalVar(defLbl, null);
                            elements.modalLabelsRole.disabled = !defLbl;
                        }
                        elements.modalProbVar.value = defProb;
                        if (elements.modalFileSummary) {
                            elements.modalFileSummary.textContent = colNames.length ? `Columns: ${colNames.join(', ')}` : '';
                            elements.modalFileSummary.classList.toggle('hidden', colNames.length === 0);
                        }
                    }
                    return { sniff, colNames };
                }
            } catch (err) {
                if (pendingFileToLoad === file) {
                    elements.modalLabelsVar.innerHTML = '<option value="">(none)</option>';
                    elements.modalProbVar.innerHTML = '<option value="">(none)</option>';
                    if (elements.modalLabelsRole) elements.modalLabelsRole.disabled = true;
                    if (elements.modalFileSummary) elements.modalFileSummary.classList.add('hidden');
                }
                return null;
            }
        })();
        pendingFileInspection = { file, promise, info: null };
        promise.then(res => { if (pendingFileInspection && pendingFileInspection.file === file && res) pendingFileInspection.info = res.info || null; });
    }

    async function loadCustomModel(modelFile, opts = {}) {
        if (!modelFile) return false;
        try {
            const buf = await modelFile.arrayBuffer();
            const bytes = new Uint8Array(buf);
            if (!bytes.length) throw new Error('Empty weights file');
            const ok = await loadBytesIntoWorker(bytes);
            if (!ok) throw new Error('Invalid or unsupported U\'n\'Eye weights (.bin) file');
            state.customModelBin = bytes;
            state.customModelName = modelFile.name;
            state.trainedModelBin = bytes;
            state.activeModelSource = 'custom';
            state.activeModelName = modelFile.name;
            state.initialWeightsMode = 'local';
            state.modelPredictedTrials = new Set();
            setNetSuggestions(true);
            syncMainModelSelectUI();
            if (opts.runPredictions && state.trials.length > 0) {
                if (elements.trainStatus) {
                    elements.trainStatus.className = 'train-active';
                    elements.trainStatus.textContent = `Computing predictions (${modelFile.name})…`;
                }
                await applyModelAsPrelabeler({
                    asGroundTruthIfEmpty: Boolean(opts.asGroundTruthIfEmpty),
                    updateProbs: opts.updateProbs !== false
                });
                if (elements.trainStatus) {
                    elements.trainStatus.className = 'train-done';
                    elements.trainStatus.textContent = `Model loaded: ${modelFile.name}`;
                }
                saveToIndexedDB();
            }
            return true;
        } catch (err) {
            console.error('Failed to load model weights:', err);
            alert('Cannot load custom model weights: ' + err.message);
            syncMainModelSelectUI();
            return false;
        }
    }

    async function switchActiveModelSource(source) {
        if (source === 'pick_custom') {
            syncMainModelSelectUI();
            if (elements.mainModelInput) elements.mainModelInput.click();
            return;
        }
        if (source === 'pretrained') {
            if (elements.trainStatus && state.trials.length > 0) {
                elements.trainStatus.className = 'train-active';
                elements.trainStatus.textContent = 'Computing predictions (combined.bin)…';
            }
            const ok = await loadPretrainedCombinedModel();
            if (ok) {
                setNetSuggestions(true);
                syncMainModelSelectUI();
                if (state.trials.length > 0) {
                    await applyModelAsPrelabeler({ asGroundTruthIfEmpty: false, updateProbs: true });
                    if (elements.trainStatus) {
                        elements.trainStatus.className = 'train-done';
                        elements.trainStatus.textContent = 'Model: combined.bin';
                    }
                }
            }
            return;
        }
        if (source === 'custom' && state.customModelBin) {
            const ok = await loadBytesIntoWorker(state.customModelBin);
            if (ok) {
                state.trainedModelBin = state.customModelBin;
                state.activeModelSource = 'custom';
                state.activeModelName = state.customModelName || 'custom weights';
                state.modelPredictedTrials = new Set();
                setNetSuggestions(true);
                syncMainModelSelectUI();
                if (state.trials.length > 0) {
                    if (elements.trainStatus) {
                        elements.trainStatus.className = 'train-active';
                        elements.trainStatus.textContent = `Computing predictions (${state.activeModelName})…`;
                    }
                    await applyModelAsPrelabeler({ asGroundTruthIfEmpty: false, updateProbs: true });
                    if (elements.trainStatus) {
                        elements.trainStatus.className = 'train-done';
                        elements.trainStatus.textContent = `Model: ${state.activeModelName}`;
                    }
                }
            }
            return;
        }
        if (source === 'session' && state.sessionTrainedModelBin) {
            const ok = await loadBytesIntoWorker(state.sessionTrainedModelBin);
            if (ok) {
                state.trainedModelBin = state.sessionTrainedModelBin;
                state.activeModelSource = 'session';
                state.activeModelName = 'session trained';
                state.modelPredictedTrials = new Set();
                setNetSuggestions(true);
                syncMainModelSelectUI();
                if (state.trials.length > 0) {
                    await applyModelAsPrelabeler({ asGroundTruthIfEmpty: false, updateProbs: true });
                    if (elements.trainStatus) {
                        elements.trainStatus.className = 'train-done';
                        elements.trainStatus.textContent = 'Model: session trained';
                    }
                }
            }
            return;
        }
        if (source === 'none') {
            state.activeModelSource = 'none';
            state.prelabelerMode = 'none';
            setNetSuggestions(false);
            state.prob = Array.from({ length: state.trials.length }, () => null);
            state.hasProb = false;
            updateProbUI();
            syncMainModelSelectUI();
            await refreshSuggestions();
            return;
        }
        // Default: 'hmm' (or file's own pred_labels / prob when present)
        state.activeModelSource = 'hmm';
        state.prelabelerMode = (state.fileEvents && state.fileEvents.some(Boolean)) ? 'file' : 'hmm';
        setNetSuggestions(false);
        state.prob = Array.from({ length: state.trials.length }, (_, t) => (state.fileProb && state.fileProb[t]) ? new Float32Array(state.fileProb[t]) : null);
        state.hasProb = state.prob.some(Boolean);
        updateProbUI();
        syncMainModelSelectUI();
        await refreshSuggestions();
    }

    function promptPickLabelsVar(fileName, candidates, defaultVar) {
        if (!elements.pickLabelsVarModal || !elements.pickLabelsVarSelect) return Promise.resolve(defaultVar || (candidates[0] && candidates[0].name) || '');
        return new Promise((resolve) => {
            if (elements.pickLabelsFileName) elements.pickLabelsFileName.textContent = fileName;
            elements.pickLabelsVarSelect.innerHTML = candidates.map(c => `<option value="${escapeOptionHtml(c.name)}">${escapeOptionHtml(c.name)}${c.summary ? ' (' + escapeOptionHtml(c.summary) + ')' : ''}</option>`).join('');
            elements.pickLabelsVarSelect.value = defaultVar && candidates.some(c => c.name === defaultVar) ? defaultVar : (candidates[0] && candidates[0].name) || '';
            elements.pickLabelsVarModal.classList.remove('hidden');
            const done = (val) => {
                elements.pickLabelsVarModal.classList.add('hidden');
                if (elements.pickLabelsVarConfirmBtn) elements.pickLabelsVarConfirmBtn.onclick = null;
                if (elements.pickLabelsVarCancelBtn) elements.pickLabelsVarCancelBtn.onclick = null;
                if (elements.pickLabelsVarCloseBtn) elements.pickLabelsVarCloseBtn.onclick = null;
                resolve(val);
            };
            if (elements.pickLabelsVarConfirmBtn) elements.pickLabelsVarConfirmBtn.onclick = () => done(elements.pickLabelsVarSelect.value);
            if (elements.pickLabelsVarCancelBtn) elements.pickLabelsVarCancelBtn.onclick = () => done(null);
            if (elements.pickLabelsVarCloseBtn) elements.pickLabelsVarCloseBtn.onclick = () => done(null);
        });
    }

    async function inspectLabelsFileForModal(file) {
        if (!elements.modalPrevLabelsVar) return;
        elements.modalPrevLabelsVar.innerHTML = '';
        elements.modalPrevLabelsVar.classList.add('hidden');
        if (!file) return;
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        try {
            if (['mat', 'npz', 'h5', 'hdf5', 'hdf'].includes(ext)) {
                const buf = await file.arrayBuffer();
                const vars = await UnEyeFormats.readVarsFile(file.name, buf, loadH5Lib);
                const listed = UnEyeFormats.listLabelVars ? UnEyeFormats.listLabelVars(vars, null) : { vars: [], defaultVar: '' };
                if (pendingLabelsFile === file && listed.vars.length > 0) {
                    elements.modalPrevLabelsVar.innerHTML = listed.vars.map(v => `<option value="${escapeOptionHtml(v.name)}">${escapeOptionHtml(v.name)}${v.summary ? ' (' + escapeOptionHtml(v.summary) + ')' : ''}</option>`).join('');
                    elements.modalPrevLabelsVar.value = listed.defaultVar || listed.vars[0].name;
                    elements.modalPrevLabelsVar.classList.remove('hidden');
                }
            } else if (['csv', 'tsv', 'txt', 'dat', 'arff'].includes(ext)) {
                const headText = await file.slice(0, 65536).text();
                const sniff = window.UnEyeImport ? UnEyeImport.sniffHead(headText, file.name) : { plain: true };
                if (!sniff.plain && window.UnEyeImport) {
                    const parsed = await UnEyeImport.parseFile(file, loadH5Lib);
                    const table = (parsed && parsed.cols) ? parsed : (parsed && parsed.tables && parsed.tables[0]);
                    if (pendingLabelsFile === file && table && table.cols) {
                        const st = UnEyeImport.describeColumns(table);
                        const sug = UnEyeImport.suggest(table, st);
                        const cands = table.cols.map((c, idx) => ({ name: c.name, st: st[idx] }))
                            .filter(x => x.st.isBool || (x.st.isInt && x.st.min >= -1 && x.st.max <= 9) || (x.st.nominal && x.st.distinct <= 2))
                            .map(x => ({ name: x.name, summary: x.st.isBool ? 'bool (0/1)' : x.st.nominal ? 'bool/nominal' : `${x.st.min}..${x.st.max}` }));
                        if (cands.length > 0) {
                            elements.modalPrevLabelsVar.innerHTML = cands.map(v => `<option value="${escapeOptionHtml(v.name)}">${escapeOptionHtml(v.name)} (${escapeOptionHtml(v.summary)})</option>`).join('');
                            const defName = sug.labels >= 0 && table.cols[sug.labels] ? table.cols[sug.labels].name : cands[0].name;
                            elements.modalPrevLabelsVar.value = cands.some(c => c.name === defName) ? defName : cands[0].name;
                            elements.modalPrevLabelsVar.classList.remove('hidden');
                        }
                    }
                }
            }
        } catch (err) {
            console.warn('inspectLabelsFileForModal:', err);
        }
    }

    async function loadPreviousLabels(labelsFile, chosenVarName) {
        if (!labelsFile) return;
        const ext = (labelsFile.name.split('.').pop() || '').toLowerCase();
        try {
            if (ext === 'json') {
                const text = await labelsFile.text();
                applyLabelsJson(JSON.parse(text));
            } else if (['mat', 'npz', 'h5', 'hdf5', 'hdf'].includes(ext)) {
                const buf = await labelsFile.arrayBuffer();
                const vars = await UnEyeFormats.readVarsFile(labelsFile.name, buf, loadH5Lib);
                const listed = UnEyeFormats.listLabelVars ? UnEyeFormats.listLabelVars(vars, state.trials) : { vars: [], defaultVar: '' };
                let varToUse = chosenVarName || '';
                if (!varToUse) {
                    if (listed.vars.length > 1) {
                        varToUse = await promptPickLabelsVar(labelsFile.name, listed.vars, listed.defaultVar);
                        if (!varToUse) return;
                    } else if (listed.vars.length === 1) {
                        varToUse = listed.vars[0].name;
                    }
                }
                let loaded = UnEyeFormats.extractLabelsFromVars ? UnEyeFormats.extractLabelsFromVars(vars, state.trials, varToUse) : null;
                if (!loaded) {
                    const r = UnEyeFormats.trialsFromVars(vars, varToUse ? { labelsVar: varToUse, labelsRole: 'reviewed' } : undefined);
                    if (r) loaded = r.labels || r.predLabels || null;
                }
                if (loaded) applyLoadedLabels(loaded);
                else throw new Error('No compatible label or boolean variable found in ' + labelsFile.name);
            } else if (['csv', 'tsv', 'txt', 'dat', 'arff'].includes(ext)) {
                const text = await labelsFile.text();
                const sniff = window.UnEyeImport ? UnEyeImport.sniffHead(text.slice(0, 65536), labelsFile.name) : { plain: true };
                if ((!sniff.plain || ext === 'arff') && window.UnEyeImport) {
                    const parsed = await UnEyeImport.parseFile(labelsFile, loadH5Lib);
                    const table = (parsed && parsed.cols) ? parsed : (parsed && parsed.tables && parsed.tables[0]);
                    if (table && table.cols && table.cols.length > 0) {
                        const st = UnEyeImport.describeColumns(table);
                        const sug = UnEyeImport.suggest(table, st);
                        const cands = table.cols.map((c, idx) => ({ idx, name: c.name, st: st[idx] }))
                            .filter(x => x.st.isBool || (x.st.isInt && x.st.min >= -1 && x.st.max <= 9) || (x.st.nominal && x.st.distinct <= 2))
                            .map(x => ({ idx: x.idx, name: x.name, summary: x.st.isBool ? 'bool (0/1)' : x.st.nominal ? 'bool/nominal' : `${x.st.min}..${x.st.max}` }));
                        if (cands.length > 0) {
                            let varToUse = chosenVarName || '';
                            if (!varToUse) {
                                const defName = sug.labels >= 0 && table.cols[sug.labels] ? table.cols[sug.labels].name : cands[0].name;
                                if (cands.length > 1) {
                                    varToUse = await promptPickLabelsVar(labelsFile.name, cands, defName);
                                    if (!varToUse) return;
                                } else {
                                    varToUse = cands[0].name;
                                }
                            }
                            const chosen = cands.find(c => c.name === varToUse) || cands[0];
                            const col = table.cols[chosen.idx], colSt = st[chosen.idx];
                            const rawVals = new Int8Array(table.nRows);
                            let targetPos = 1;
                            if (colSt.nominal && col.labels) {
                                const posIdx = col.labels.findIndex(s => /^(true|1|yes|saccade|blink|event)$/i.test(String(s).trim()));
                                const negIdx = col.labels.findIndex(s => /^(false|0|no|fixation|none)$/i.test(String(s).trim()));
                                targetPos = posIdx >= 0 ? posIdx : (negIdx === 0 ? 1 : negIdx === 1 ? 0 : 1);
                            }
                            for (let i = 0; i < table.nRows; i++) {
                                const v = col.data[i];
                                if (colSt.nominal && col.labels) {
                                    rawVals[i] = (isNaN(v) || v < 0) ? -1 : (Math.round(v) === targetPos ? 1 : 0);
                                } else {
                                    rawVals[i] = (isNaN(v) || v < 0) ? -1 : (v >= 2 ? 2 : v >= 1 ? 1 : 0);
                                }
                            }
                            const loaded = [];
                            let off = 0;
                            for (let t = 0; t < state.trials.length; t++) {
                                const len = state.trials[t].x.length;
                                const arr = new Int8Array(len).fill(-1);
                                for (let i = 0; i < len && off + i < rawVals.length; i++) arr[i] = rawVals[off + i];
                                off += len;
                                loaded.push(arr);
                            }
                            applyLoadedLabels(loaded);
                            return;
                        }
                    }
                }
                const rows = parseCSV(text);
                if (rows.length > 0) {
                    let loaded;
                    if (rows.length > 1 && rows[0].length === 1 && state.trials.length >= 1) {
                        // Single column of 0/1 or true/false values across trials
                        let off = 0;
                        loaded = state.trials.map(tr => {
                            const arr = new Int8Array(tr.x.length).fill(-1);
                            for (let i = 0; i < tr.x.length && off < rows.length; i++, off++) {
                                const v = rows[off][0];
                                arr[i] = (isNaN(v) || v < 0) ? -1 : Math.round(v);
                            }
                            return arr;
                        });
                    } else {
                        loaded = rows.map(r => {
                            const arr = new Int8Array(r.length);
                            for (let i = 0; i < r.length; i++) {
                                const v = r[i];
                                arr[i] = (isNaN(v) || v < 0) ? -1 : Math.round(v);
                            }
                            return arr;
                        });
                    }
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
        inspectFileForModal(file);
        if (!modalAlreadyOpen) {
            pendingModelFile = null;
            pendingLabelsFile = null;
            if (elements.modalModelFileName) elements.modalModelFileName.textContent = 'No file chosen';
            if (elements.prelabelerModelFileName) elements.prelabelerModelFileName.textContent = 'No file chosen';
            if (elements.modalLabelsFileName) elements.modalLabelsFileName.textContent = '(none)';
            if (elements.modalPrevLabelsVar) { elements.modalPrevLabelsVar.innerHTML = ''; elements.modalPrevLabelsVar.classList.add('hidden'); }
            if (elements.localModelRow) elements.localModelRow.classList.add('hidden');
            if (elements.prelabelerModelRow) elements.prelabelerModelRow.classList.add('hidden');
            if (elements.initialModelSelect) elements.initialModelSelect.value = 'pretrained';
            if (elements.prelabelerSelect) elements.prelabelerSelect.value = 'hmm';
            elements.loadModal.classList.remove('hidden');
        }
    }

    async function confirmLoadFile() {
        const file = pendingFileToLoad;
        if (!file) return;

        const rate = parseFloat(elements.rateInput.value) || 1000.0;
        const scale = parseFloat(elements.scaleInput.value) || 1.0;
        const prelabeler = elements.prelabelerSelect.value;
        if (prelabeler === 'custom' && !pendingModelFile) {
            alert('Please choose a custom pretrained weights file (.bin) first.');
            if (elements.modalModelFile) elements.modalModelFile.click();
            return;
        }
        if (elements.modalWinMs) setWinMs(parseFloat(elements.modalWinMs.value) || state.winMs);
        const sgVal = parseInt(elements.modalSgWindow.value, 10);
        state.filterSettings.enabled = sgVal > 0;
        if (sgVal > 0) state.filterSettings.windowMs = sgVal;

        const rawLabelsVar = elements.modalLabelsVar ? elements.modalLabelsVar.value : '__auto__';
        const rawProbVar = elements.modalProbVar ? elements.modalProbVar.value : '__auto__';
        const varOpts = {
            labelsVar: rawLabelsVar === '__auto__' ? undefined : rawLabelsVar,
            labelsRole: elements.modalLabelsRole ? elements.modalLabelsRole.value : undefined,
            probVar: rawProbVar === '__auto__' ? undefined : rawProbVar,
        };

        if (prelabeler === 'custom' && pendingModelFile) {
            const ok = await loadCustomModel(pendingModelFile, { runPredictions: false });
            if (!ok) return;
        } else if (prelabeler === 'pretrained') {
            const ok = await loadPretrainedCombinedModel();
            if (!ok) return;
            state.initialWeightsMode = 'pretrained';
        } else if (elements.initialModelSelect) {
            const mVal = elements.initialModelSelect.value;
            if (mVal === 'local' && pendingModelFile) {
                const ok = await loadCustomModel(pendingModelFile, { runPredictions: false });
                if (!ok) return;
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
            try {
                elements.modalFileName.textContent = 'Reading ' + file.name + '…';
                const insp = (pendingFileInspection && pendingFileInspection.file === file) ? await pendingFileInspection.promise : null;
                const vars = (insp && insp.vars) ? insp.vars : await UnEyeFormats.readVarsFile(file.name, await file.arrayBuffer(), loadH5Lib);
                // a layout this page knows (its own exports) loads directly; any other file, or "Show the column explorer", opens the explorer
                let r = null;
                if (!(elements.alwaysExplorer && elements.alwaysExplorer.checked)) {
                    try { r = UnEyeFormats.trialsFromVars(vars, varOpts); } catch (err) { r = null; }
                }
                if (!r) { await loadWithExplorer(file, rate, scale, prelabeler, { ...varOpts, vars }); return; }
                elements.loadModal.classList.add('hidden');
                loadDataset(r.trials, r.fs || rate, scale, prelabeler, { prob: r.prob, predLabels: r.predLabels, probParams: r.probParams });
                if (r.labels) {
                    applyLoadedLabels(r.labels);
                }
                if (pendingLabelsFile) {
                    const chosenPrevVar = (elements.modalPrevLabelsVar && !elements.modalPrevLabelsVar.classList.contains('hidden')) ? elements.modalPrevLabelsVar.value : '';
                    await loadPreviousLabels(pendingLabelsFile, chosenPrevVar);
                }
                const lens = r.trials.map(t => t.x.length);
                console.info(`${file.name}: ${r.trials.length} trials, ${Math.min(...lens)}-${Math.max(...lens)} samples${r.fs ? ', ' + r.fs + ' Hz' : ''}${r.labels ? ', with labels' : ''}${r.predLabels ? ', with pred_labels' : ''}${r.prob ? ', with prob' : ''}`);
            } catch (err) {
                elements.loadModal.classList.add('hidden');
                alert('Cannot read ' + file.name + ': ' + err.message);
            }
            return;
        }

        // text files: plain numbers (no header, 1-2 columns, or trials x time) keep the old reader; anything else (a header, many
        // columns, ARFF, EyeLink .asc) goes through the column explorer, which also handles the time stamps
        const forceExplorer = elements.alwaysExplorer && elements.alwaysExplorer.checked;
        if (forceExplorer || ext === 'arff' || ext === 'asc' || !(await isPlainTextFile(file))) {
            await loadWithExplorer(file, rate, scale, prelabeler, varOpts);
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

    async function loadWithExplorer(file, rate, scale, prelabeler, extraOpts) {
        elements.loadModal.classList.add('hidden');
        const r = await UnEyeImportUI.run(file, { rate, scale, h5: loadH5Lib, ...(extraOpts || {}) });
        if (!r) { elements.modalFileName.textContent = file.name; elements.loadModal.classList.remove('hidden'); return; }      // cancelled: back to the load dialog
        loadDataset(r.trials, r.fs, r.degPerUnit, prelabeler, { prob: r.prob, predLabels: r.predLabels, probParams: r.probParams });
        if (r.labels) applyLoadedLabels(r.labels);
        if (pendingLabelsFile) {
            const chosenPrevVar = (elements.modalPrevLabelsVar && !elements.modalPrevLabelsVar.classList.contains('hidden')) ? elements.modalPrevLabelsVar.value : '';
            await loadPreviousLabels(pendingLabelsFile, chosenPrevVar);
        }
        const lens = r.trials.map(t => t.x.length);
        console.info(`${file.name}: ${r.trials.length} trials, ${Math.min(...lens)}-${Math.max(...lens)} samples at ${r.fs} Hz (${r.source}, ${r.degPerUnit} deg/unit)` + (r.notes.length ? ' · ' + r.notes.join(' ') : ''));
    }

    // --- HDF5 needs h5wasm (5.7 MB, web/h5wasm.js, shipped with the app; loaded the first time an .h5 / v7.3 .mat file is read or written) ---
    async function loadH5Lib() {
        if (!window.h5wasm) {
            await new Promise((resolve, reject) => {
                const sc = document.createElement('script');
                sc.src = 'web/h5wasm.js?v=10';
                sc.onload = resolve;
                sc.onerror = () => reject(new Error('Cannot load the HDF5 library (web/h5wasm.js): reload the page, or use .npz / .mat (v5).'));
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
            for (const ev of (state.expertEvents[t] || [])) for (let i = ev.onset; i <= ev.offset && i < out.length; i++) out[i] = ev.cls;
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
                viewedRanges: state.viewedRanges[t] || [],
                rejectedNearMisses: state.rejectedNearMisses[t] || [],
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

            // 4) windows displayed and rejected near misses (files saved before these fields existed have none)
            if (Array.isArray(src.viewedRanges)) state.viewedRanges[t] = UnEyeProb.mergeRanges([...(state.viewedRanges[t] || []), ...src.viewedRanges], n);
            if (Array.isArray(src.rejectedNearMisses)) {
                const rej = state.rejectedNearMisses[t] || (state.rejectedNearMisses[t] = []);
                for (const r of src.rejectedNearMisses) {
                    if (!(r && r.onset >= 0 && r.offset >= r.onset && r.offset < n)) continue;
                    if (!rej.some(q => q.onset === r.onset && q.offset === r.offset)) rej.push({ onset: r.onset, offset: r.offset, at: r.at !== undefined ? r.at : r.onset, peak: r.peak });
                }
            }

            recomputeReviewed(t, false);
            syncRenderedLabels(t);
            applied++;
        }
        if (state.nearMiss.active) buildNearMissQueue();
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
                    row.push((isRev || lab[i] > 0) ? lab[i] : "nan");
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

    // lower bound of P(saccade) of a near miss (menu field, remembered)
    function setNearMissLower(v) {
        if (!(v > 0 && v < state.probThreshold)) { if (elements.nearMissLowerInput) elements.nearMissLowerInput.value = String(state.nearMiss.lower); return; }
        state.nearMiss.lower = v;
        try { localStorage.setItem('uneye_nearMissLower', String(v)); } catch (e) { }
        if (elements.nearMissLowerInput) elements.nearMissLowerInput.value = String(v);
        if (state.nearMiss.active) { buildNearMissQueue(); nmFocus(); updateStatus(); render(); }
    }

    // --- Viewed periods: one row per trial, time on x (samples from the trial start, all trials on the scale of the longest) ---
    // The rows are drawn once per opening into an offscreen image of one pixel row per trial (one column per screen pixel, the highest
    // category of the samples of each column), then scaled into the visible part of the scroll area: fast for thousands of trials.
    const VIEW_RGB = [[255, 255, 255], [207, 216, 220], [144, 202, 249], [102, 187, 106], [251, 140, 0], [13, 71, 161], [198, 40, 40]];   // beyond the trial, not viewed, viewed, reviewed, rejected near miss, saccade, blink
    const VIEW_LEFT = 46, VIEW_TOP = 18;
    const viewed = { img: null, W: 0, maxLen: 1, rowH: 4 };

    function viewedTotals() {
        let total = 0, seen = 0, rev = 0, full = 0, rej = 0, nEv = 0;
        for (let t = 0; t < state.trials.length; t++) {
            const n = trialLenOf(t), v = UnEyeProb.rangeLength(state.viewedRanges[t]);
            total += n; seen += v; rev += UnEyeProb.rangeLength(state.reviewedRanges[t]);
            if (v >= n) full++;
            rej += (state.rejectedNearMisses[t] || []).length;
            nEv += (state.expertEvents[t] || []).length;
        }
        return { total, seen, rev, full, rej, nEv, n: state.trials.length };
    }

    function buildViewedImage(W) {
        const n = state.trials.length;
        let maxLen = 1; for (let t = 0; t < n; t++) maxLen = Math.max(maxLen, trialLenOf(t));
        const off = document.createElement('canvas'); off.width = W; off.height = Math.max(1, n);
        const octx = off.getContext('2d'), img = octx.createImageData(W, Math.max(1, n)), px = img.data;
        const lvl = new Uint8Array(W), col = s => Math.max(0, Math.min(W - 1, Math.floor(s * W / maxLen)));
        const paint = (s, e, level) => { for (let c = col(s), c1 = col(e); c <= c1; c++) if (lvl[c] < level) lvl[c] = level; };
        for (let t = 0; t < n; t++) {
            lvl.fill(0);
            const L = trialLenOf(t);
            lvl.fill(1, 0, Math.max(1, Math.ceil(L * W / maxLen)));
            for (const r of state.viewedRanges[t] || []) paint(r[0], r[1], 2);
            for (const r of state.reviewedRanges[t] || []) paint(r[0], r[1], 3);
            for (const r of state.rejectedNearMisses[t] || []) paint(r.onset, r.offset, 4);
            for (const e of state.expertEvents[t] || []) paint(e.onset, e.offset, e.cls === CLS_BLINK ? 6 : 5);
            for (let c = 0, o = t * W * 4; c < W; c++, o += 4) { const rgb = VIEW_RGB[lvl[c]]; px[o] = rgb[0]; px[o + 1] = rgb[1]; px[o + 2] = rgb[2]; px[o + 3] = 255; }
        }
        octx.putImageData(img, 0, 0);
        viewed.img = off; viewed.W = W; viewed.maxLen = maxLen;
    }

    function drawViewed() {
        const sc = elements.viewedScroll, cv = elements.viewedCanvas;
        if (!sc || !cv || elements.viewedModal.classList.contains('hidden')) return;
        const dpr = window.devicePixelRatio || 1, cw = sc.clientWidth, ch = sc.clientHeight, n = state.trials.length;
        if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr); }
        cv.style.height = ch + 'px';
        if (elements.viewedSizer) elements.viewedSizer.style.height = Math.max(0, VIEW_TOP + n * viewed.rowH - ch) + 'px';
        const c = cv.getContext('2d');
        c.setTransform(dpr, 0, 0, dpr, 0, 0);
        c.clearRect(0, 0, cw, ch);
        const plotW = Math.max(10, cw - VIEW_LEFT - 8), rowH = viewed.rowH, top = sc.scrollTop;
        if (!viewed.img || viewed.W !== Math.min(4000, Math.round(plotW * dpr))) buildViewedImage(Math.min(4000, Math.round(plotW * dpr)));
        const first = Math.max(0, Math.floor(top / rowH)), last = Math.min(n - 1, Math.ceil((top + ch - VIEW_TOP) / rowH));
        c.imageSmoothingEnabled = false;
        if (last >= first) c.drawImage(viewed.img, 0, first, viewed.W, last - first + 1, VIEW_LEFT, VIEW_TOP + first * rowH - top, plotW, (last - first + 1) * rowH);
        // current trial
        const cy = VIEW_TOP + state.currentTrialIdx * rowH - top;
        if (cy + rowH >= VIEW_TOP && cy <= ch) { c.strokeStyle = '#000'; c.lineWidth = 1; c.strokeRect(VIEW_LEFT - 0.5, cy - 0.5, plotW + 1, rowH + 1); }
        // trial numbers on the left
        c.clearRect(0, 0, VIEW_LEFT - 1, ch);
        c.fillStyle = '#546E7A'; c.font = '500 10px sans-serif'; c.textAlign = 'right'; c.textBaseline = 'middle';
        let step = 1; for (const k of [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]) { step = k; if (k * rowH >= 14) break; }
        for (let t = Math.ceil(first / step) * step; t <= last; t += step) { const y = VIEW_TOP + t * rowH - top + rowH / 2; if (y > VIEW_TOP) c.fillText(String(t + 1), VIEW_LEFT - 5, y); }
        // time axis (top strip)
        c.clearRect(0, 0, cw, VIEW_TOP);
        c.textBaseline = 'alphabetic'; c.textAlign = 'center';
        const totalMs = viewed.maxLen * 1000 / state.rateHz;
        let tick = 100; for (const k of [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000]) { tick = k; if (plotW * k / totalMs >= 60) break; }
        for (let ms = 0; ms <= totalMs; ms += tick) {
            const x = VIEW_LEFT + plotW * ms / totalMs;
            c.textAlign = ms === 0 ? 'left' : (x > VIEW_LEFT + plotW - 20 ? 'right' : 'center');
            c.fillText(ms >= 1000 ? (ms / 1000) + ' s' : ms + ' ms', x, 12);
        }
        c.textAlign = 'right'; c.fillText('trial', VIEW_LEFT - 5, 12);
    }

    function viewedHit(e) {
        const r = elements.viewedCanvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
        const plotW = Math.max(10, elements.viewedScroll.clientWidth - VIEW_LEFT - 8);
        if (x < VIEW_LEFT || y < VIEW_TOP) return null;
        const t = Math.floor((y - VIEW_TOP + elements.viewedScroll.scrollTop) / viewed.rowH);
        if (t < 0 || t >= state.trials.length) return null;
        const sample = Math.max(0, Math.min(trialLenOf(t) - 1, Math.round((x - VIEW_LEFT) / plotW * viewed.maxLen)));
        return { t, sample, beyond: (x - VIEW_LEFT) / plotW * viewed.maxLen >= trialLenOf(t) };
    }

    // a click on a row: that trial, pass 1, the window that contains the time clicked
    function jumpToTrialTime(t, sample) {
        if (!state.trials[t]) return;
        if (elements.viewedModal) elements.viewedModal.classList.add('hidden');
        if (state.pass !== PASS_ONE) setPass(PASS_ONE); else scanCurrentWindow();
        resetGesture();
        state.currentTrialIdx = t;
        state.zoomMultiplier = 1.0;
        state.winIdx = Math.max(0, Math.min(numWindows(t) - 1, Math.floor(sample / winStep())));
        applyGridWindow();
        updateStatus();
        render();
    }

    function updateViewedSummary() {
        const tot = viewedTotals(), pct = (a, b) => (b ? (100 * a / b).toFixed(1) : '0.0') + ' %';
        if (elements.viewedSummary) elements.viewedSummary.textContent = `Viewed ${pct(tot.seen, tot.total)} of the samples (${(tot.seen / state.rateHz).toFixed(1)} of ${(tot.total / state.rateHz).toFixed(1)} s), ` +
            `${tot.full}/${tot.n} trials fully viewed · reviewed ${pct(tot.rev, tot.total)} · ${tot.nEv} expert events · ${tot.rej} rejected near misses`;
        if (elements.viewedResetBtn) {
            elements.viewedResetBtn.disabled = (tot.seen === 0 && tot.rev === 0);
        }
    }

    // Reset all viewed and reviewed periods to zero (mark all as unseen) while keeping ground-truth labels (state.expertEvents) as they are
    function resetViewedPeriods() {
        const n = state.trials.length;
        if (!n) return;
        for (let t = 0; t < n; t++) {
            state.viewedRanges[t] = [];
            state.scanWins[t] = [];
            state.reviewedRanges[t] = [];
            syncRenderedLabels(t);
        }
        viewed.img = null;
        updateViewedSummary();
        drawViewed();
        updateStatus();
        render(false);
        saveToIndexedDB();
    }

    function openViewedPeriods() {
        if (!state.trials.length || !elements.viewedModal) return;
        const n = state.trials.length;
        viewed.rowH = n <= 40 ? 14 : n <= 150 ? 8 : n <= 600 ? 5 : 4;
        viewed.img = null;
        updateViewedSummary();
        elements.viewedModal.classList.remove('hidden');
        const sc = elements.viewedScroll;
        requestAnimationFrame(() => {
            sc.scrollTop = Math.max(0, state.currentTrialIdx * viewed.rowH - sc.clientHeight / 2);
            drawViewed();
        });
    }

    function setupViewedPeriods() {
        const sc = elements.viewedScroll, cv = elements.viewedCanvas;
        if (!sc || !cv) return;
        let queued = false;
        sc.addEventListener('scroll', () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; drawViewed(); }); });
        window.addEventListener('resize', () => { viewed.img = null; drawViewed(); });
        cv.addEventListener('mousemove', (e) => {
            const h = viewedHit(e);
            if (!h) { cv.title = ''; return; }
            const t = h.t, n = trialLenOf(t), name = (state.trialNames && state.trialNames[t]) || state.trials[t].name || '';
            const v = UnEyeProb.rangeLength(state.viewedRanges[t]), r = UnEyeProb.rangeLength(state.reviewedRanges[t]);
            const txt = `Trial ${t + 1}/${state.trials.length}${name ? ' · ' + name : ''} · ${h.beyond ? 'after the end' : Math.round(h.sample * 1000 / state.rateHz) + ' ms'} (of ${Math.round(n * 1000 / state.rateHz)} ms) · viewed ${(100 * v / n).toFixed(0)} % · reviewed ${(100 * r / n).toFixed(0)} %`;
            cv.title = txt;
            if (elements.viewedTip) elements.viewedTip.textContent = txt;
        });
        cv.addEventListener('click', (e) => { const h = viewedHit(e); if (h) jumpToTrialTime(h.t, h.sample); });
    }

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
                e.target.value = '';
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
        if (elements.loadWeightsBtn && elements.mainModelInput) {
            elements.loadWeightsBtn.addEventListener('click', () => elements.mainModelInput.click());
        }
        if (elements.mainModelSelect) {
            elements.mainModelSelect.addEventListener('change', (e) => {
                switchActiveModelSource(e.target.value);
            });
        }
        if (elements.mainModelInput) {
            elements.mainModelInput.addEventListener('change', async (e) => {
                if (e.target.files && e.target.files[0]) {
                    const f = e.target.files[0];
                    e.target.value = '';
                    await loadCustomModel(f, { runPredictions: true, asGroundTruthIfEmpty: false, updateProbs: true });
                } else {
                    syncMainModelSelectUI();
                }
            });
        }
        if (elements.initialModelSelect) {
            elements.initialModelSelect.addEventListener('change', (e) => {
                state.initialWeightsMode = e.target.value;
                if (elements.localModelRow) {
                    elements.localModelRow.classList.toggle('hidden', e.target.value !== 'local');
                }
            });
        }
        if (elements.prelabelerSelect) {
            elements.prelabelerSelect.addEventListener('change', (e) => {
                const isCustom = e.target.value === 'custom';
                if (elements.prelabelerModelRow) {
                    elements.prelabelerModelRow.classList.toggle('hidden', !isCustom);
                }
                if (isCustom) {
                    if (elements.initialModelSelect) {
                        elements.initialModelSelect.value = 'local';
                        state.initialWeightsMode = 'local';
                    }
                    if (elements.localModelRow) elements.localModelRow.classList.remove('hidden');
                    if (!pendingModelFile && elements.modalModelFile) {
                        elements.modalModelFile.click();
                    }
                }
            });
        }
        if (elements.prelabelerPickModelBtn && elements.modalModelFile) {
            elements.prelabelerPickModelBtn.addEventListener('click', () => elements.modalModelFile.click());
        }
        if (elements.modalPickModelBtn && elements.modalModelFile) {
            elements.modalPickModelBtn.addEventListener('click', () => elements.modalModelFile.click());
            elements.modalModelFile.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) {
                    pendingModelFile = e.target.files[0];
                    if (elements.modalModelFileName) elements.modalModelFileName.textContent = pendingModelFile.name;
                    if (elements.prelabelerModelFileName) elements.prelabelerModelFileName.textContent = pendingModelFile.name;
                }
            });
        }
        if (elements.modalPickLabelsBtn && elements.modalLabelsFile) {
            elements.modalPickLabelsBtn.addEventListener('click', () => elements.modalLabelsFile.click());
            elements.modalLabelsFile.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) {
                    pendingLabelsFile = e.target.files[0];
                    if (elements.modalLabelsFileName) elements.modalLabelsFileName.textContent = pendingLabelsFile.name;
                    inspectLabelsFileForModal(pendingLabelsFile);
                    e.target.value = '';
                }
            });
        }
        if (elements.modalChangeTraceBtn) {
            elements.modalChangeTraceBtn.addEventListener('click', () => elements.fileInput.click());
        }
        if (elements.modalLabelsVar) {
            elements.modalLabelsVar.addEventListener('change', () => {
                const val = elements.modalLabelsVar.value;
                if (elements.modalLabelsRole) {
                    const hasVal = Boolean(val && val !== '__auto__');
                    elements.modalLabelsRole.disabled = !hasVal;
                    if (hasVal) {
                        elements.modalLabelsRole.value = inferRoleForModalVar(val, pendingFileInspection && pendingFileInspection.info);
                    }
                }
            });
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
            if (state.nearMiss.active) { nmStep(-1); updateStatus(); render(); return; }
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
            if (state.nearMiss.active) { nmStep(1); updateStatus(); render(); return; }
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
            if (state.nearMiss.active) { nmReject(); updateStatus(); render(); return; }
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

        // Near misses (only with a probability trace): pass-2 button and menu button toggle the mode; the lower bound is remembered
        const toggleNearMiss = () => { if (state.nearMiss.active) exitNearMiss(); else enterNearMiss(); };
        if (elements.nearMissBtn) elements.nearMissBtn.addEventListener('click', toggleNearMiss);
        if (elements.nearMissMenuBtn) elements.nearMissMenuBtn.addEventListener('click', toggleNearMiss);
        if (elements.nearMissLowerInput) elements.nearMissLowerInput.addEventListener('change', e => setNearMissLower(parseFloat(e.target.value)));
        if (elements.viewedBtn) elements.viewedBtn.addEventListener('click', openViewedPeriods);
        if (elements.viewedResetBtn) elements.viewedResetBtn.addEventListener('click', resetViewedPeriods);
        if (elements.viewedCloseBtn) elements.viewedCloseBtn.addEventListener('click', () => elements.viewedModal.classList.add('hidden'));
        setupViewedPeriods();

        elements.acceptPreBtn.addEventListener('click', () => {
            if (state.handleDrag && state.handleDrag.viaKey) {
                commitHandleDrag(state.handleDrag);
                state.handleDrag = null;
            }
            if (state.nearMiss.active) { nmAccept(); updateStatus(); render(); return; }
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

    // --- Keyboard Shortcuts & Desktop Boundary Grabbing ---
    const SHORTCUT_ACTIONS = [
        // Navigation
        { id: 'prevWindowOrEvent', group: 'Navigation', label: 'Previous window / event', sub: 'Pass 1: Prev window · Pass 2: Prev event', btnIds: ['prevEpochBtn', 'prevEventBtn'], defaultKey: 'ArrowLeft', azertyKey: 'ArrowLeft' },
        { id: 'nextWindowOrEvent', group: 'Navigation', label: 'Next window / event', sub: 'Pass 1: Next window · Pass 2: Next event', btnIds: ['nextEpochBtn', 'nextEventBtn'], defaultKey: 'ArrowRight', azertyKey: 'ArrowRight' },
        { id: 'prevTrial', group: 'Navigation', label: 'Previous trial', sub: 'Jump to previous trial', btnIds: ['prevTrialBtn'], defaultKey: 'ArrowUp', azertyKey: 'ArrowUp' },
        { id: 'nextTrial', group: 'Navigation', label: 'Next trial', sub: 'Jump to next trial', btnIds: ['nextTrialBtn'], defaultKey: 'ArrowDown', azertyKey: 'ArrowDown' },
        { id: 'switchPass', group: 'Navigation', label: 'Switch Pass (Pass 1 ⇄ Pass 2)', sub: 'Toggle between marking and refinement', btnIds: ['switchPassBtn'], defaultKey: 'p', azertyKey: 'p' },

        // Pass 2 & Near Misses Refinement
        { id: 'grabOnset', group: 'Pass 2 & Boundary Refinement', label: 'Grab start (onset) boundary', sub: 'Hold/tap while hovering trace, or hold + ←/→', btnIds: [], defaultKey: 'a', azertyKey: 'q' },
        { id: 'grabOffset', group: 'Pass 2 & Boundary Refinement', label: 'Grab end (offset) boundary', sub: 'Hold/tap while hovering trace, or hold + ←/→', btnIds: [], defaultKey: 's', azertyKey: 's' },
        { id: 'acceptEvent', group: 'Pass 2 & Boundary Refinement', label: 'Accept Pre-label / candidate', sub: 'Confirm current event or near miss (Enter also works)', btnIds: ['acceptPreBtn'], defaultKey: 'Space', azertyKey: 'Space' },
        { id: 'notAnEvent', group: 'Pass 2 & Boundary Refinement', label: 'Not an Event (reject / delete)', sub: 'Reject candidate or delete mark (Delete/Backspace also work)', btnIds: ['notEventBtn'], defaultKey: 'd', azertyKey: 'd' },
        { id: 'wider', group: 'Pass 2 & Boundary Refinement', label: 'Wider zoom', sub: 'Cycle zoom width (±50ms → 2.5× → 5×)', btnIds: ['widerBtn'], defaultKey: 'w', azertyKey: 'z' },
        { id: 'nearMisses', group: 'Pass 2 & Boundary Refinement', label: 'Toggle Near misses mode', sub: 'Review sub-threshold P(saccade) peaks', btnIds: ['nearMissBtn', 'nearMissMenuBtn'], defaultKey: 'n', azertyKey: 'n' },

        // Annotation Tools & Classes
        { id: 'saccadeTool', group: 'Annotation Tools & Classes', label: 'Select Saccade tool', sub: 'Class 1 (blue)', btnIds: ['saccadeBtn'], defaultKey: '1', azertyKey: '1' },
        { id: 'blinkTool', group: 'Annotation Tools & Classes', label: 'Select Blink tool', sub: 'Class 2 (red)', btnIds: ['blinkBtn'], defaultKey: '2', azertyKey: '2' },
        { id: 'toggleClass', group: 'Annotation Tools & Classes', label: 'Toggle Saccade / Blink class', sub: 'Switch active event class', btnIds: ['classToggle'], defaultKey: 'c', azertyKey: 'c' },
        { id: 'eraserTool', group: 'Annotation Tools & Classes', label: 'Toggle Eraser tool', sub: 'Erase marks/events in dragged range', btnIds: ['eraserBtn'], defaultKey: 'e', azertyKey: 'e' },
        { id: 'undo', group: 'Annotation Tools & Classes', label: 'Undo last action', sub: 'Ctrl+Z / Cmd+Z also works', btnIds: ['undoBtn'], defaultKey: 'z', azertyKey: 'w' },
        { id: 'autoSlide', group: 'Annotation Tools & Classes', label: 'Toggle Auto-slide', sub: 'Slide window after marking in Pass 1', btnIds: ['autoSlideBtn'], defaultKey: 'l', azertyKey: 'l' },

        // Files, Model & Views
        { id: 'viewedPeriods', group: 'Files, Model & Views', label: 'Viewed periods overview', sub: 'Open trial overview map', btnIds: ['viewedBtn'], defaultKey: 'v', azertyKey: 'v' },
        { id: 'netSuggestions', group: 'Files, Model & Views', label: 'Toggle Network suggestions', sub: 'Switch between trained network and HMM proposals', btnIds: ['netSuggestBtn'], defaultKey: 'g', azertyKey: 'g' },
        { id: 'openFile', group: 'Files, Model & Views', label: 'Open eye trace file…', sub: 'Load a new dataset', btnIds: ['openBtn'], defaultKey: 'o', azertyKey: 'o' },
        { id: 'saveLabels', group: 'Files, Model & Views', label: 'Save labels (.json)', sub: 'Ctrl+S / Cmd+S also works', btnIds: ['saveLabelsBtn'], defaultKey: 'k', azertyKey: 'k' },
        { id: 'loadLabels', group: 'Files, Model & Views', label: 'Load saved labels (.json)', sub: 'Import previously saved labels', btnIds: ['loadLabelsBtn'], defaultKey: 'j', azertyKey: 'j' },
        { id: 'filterSettings', group: 'Files, Model & Views', label: 'Filter settings', sub: 'Savitzky-Golay display smoothing', btnIds: ['filterBtn'], defaultKey: 'f', azertyKey: 'f' },
        { id: 'metricsModal', group: 'Files, Model & Views', label: 'Learning curves (Metrics)', sub: 'Show test-set F1 & kappa curves', btnIds: ['metricsBtn'], defaultKey: 'm', azertyKey: 'm' },
        { id: 'trainModel', group: 'Files, Model & Views', label: 'Train network now', sub: 'Start an on-device training round', btnIds: ['trainNowBtn'], defaultKey: 't', azertyKey: 't' },
        { id: 'downloadModel', group: 'Files, Model & Views', label: 'Download trained model', sub: 'Save .bin and .json weights', btnIds: ['downloadModelBtn'], defaultKey: 'b', azertyKey: 'b' },
        { id: 'exportData', group: 'Files, Model & Views', label: 'Export dataset', sub: 'Export labels & traces in selected format', btnIds: ['exportBtn'], defaultKey: 'x', azertyKey: 'x' },
        { id: 'toggleMenu', group: 'Files, Model & Views', label: 'Show / hide settings bar (☰)', sub: 'Toggle tools panel', btnIds: ['menuBtn'], defaultKey: 'u', azertyKey: 'u' },
        { id: 'openShortcuts', group: 'Files, Model & Views', label: 'Keyboard shortcuts list', sub: 'Open this shortcuts window', btnIds: ['shortcutsBtn', 'shortcutsTopBtn'], defaultKey: '?', azertyKey: '?' },
    ];

    function normalizeShortcutKey(rawKey) {
        if (rawKey === undefined || rawKey === null) return '';
        const k = typeof rawKey === 'string' ? rawKey : rawKey.key;
        if (!k || k === 'Shift' || k === 'Control' || k === 'Alt' || k === 'Meta' || k === 'CapsLock') return '';
        if (k === ' ' || k === 'Spacebar') return 'Space';
        if (k.length === 1) return k.toLowerCase();
        return k;
    }

    function formatShortcutKey(k) {
        if (!k) return '';
        const map = {
            ArrowLeft: '←',
            ArrowRight: '→',
            ArrowUp: '↑',
            ArrowDown: '↓',
            Space: 'Space',
            Enter: 'Enter',
            Backspace: '⌫',
            Delete: 'Del',
            Escape: 'Esc',
            Tab: 'Tab',
            PageUp: 'PgUp',
            PageDown: 'PgDn',
        };
        if (map[k]) return map[k];
        return k.length === 1 ? k.toUpperCase() : k;
    }

    function defaultShortcutsForPreset(preset) {
        const out = {};
        for (const a of SHORTCUT_ACTIONS) {
            out[a.id] = preset === 'azerty' ? a.azertyKey : a.defaultKey;
        }
        return out;
    }

    function loadShortcuts() {
        const base = defaultShortcutsForPreset('qwerty');
        try {
            const raw = localStorage.getItem('uneye_shortcuts');
            if (raw) {
                const parsed = JSON.parse(raw);
                if (parsed && typeof parsed === 'object') {
                    for (const a of SHORTCUT_ACTIONS) {
                        if (Object.prototype.hasOwnProperty.call(parsed, a.id)) {
                            base[a.id] = parsed[a.id] ? normalizeShortcutKey(parsed[a.id]) : '';
                        }
                    }
                }
            }
        } catch (e) { }
        state.shortcuts = base;
        syncShortcutBadgesOnButtons();
    }

    function saveShortcuts() {
        try {
            localStorage.setItem('uneye_shortcuts', JSON.stringify(state.shortcuts));
        } catch (e) { }
        syncShortcutBadgesOnButtons();
        render();
    }

    function syncShortcutBadgesOnButtons() {
        for (const a of SHORTCUT_ACTIONS) {
            const key = state.shortcuts[a.id] || '';
            const disp = formatShortcutKey(key);
            for (const bid of a.btnIds) {
                const el = document.getElementById(bid);
                if (!el) continue;
                el.dataset.shortcut = disp;
                if (!el.dataset.baseTitle) {
                    el.dataset.baseTitle = el.getAttribute('title') || a.label;
                }
                el.title = disp ? `${el.dataset.baseTitle} [${disp}]` : el.dataset.baseTitle;
            }
        }
    }

    function setShortcut(actionId, newKey) {
        const act = SHORTCUT_ACTIONS.find(a => a.id === actionId);
        if (!act) return false;
        const norm = newKey ? normalizeShortcutKey(newKey) : '';
        if (norm) {
            for (const a of SHORTCUT_ACTIONS) {
                if (a.id !== actionId && state.shortcuts[a.id] === norm) {
                    state.shortcuts[a.id] = '';
                }
            }
        }
        state.shortcuts[actionId] = norm;
        saveShortcuts();
        if (elements.shortcutsModal && !elements.shortcutsModal.classList.contains('hidden')) {
            renderShortcutsList();
        }
        return true;
    }

    function applyShortcutPreset(preset) {
        state.recordingShortcutAction = null;
        state.shortcuts = defaultShortcutsForPreset(preset === 'azerty' ? 'azerty' : 'qwerty');
        saveShortcuts();
        if (elements.shortcutsModal && !elements.shortcutsModal.classList.contains('hidden')) {
            renderShortcutsList();
        }
    }

    function renderShortcutsList() {
        const container = elements.shortcutsList;
        if (!container) return;
        container.innerHTML = '';
        let currentGroup = '';
        for (const a of SHORTCUT_ACTIONS) {
            if (a.group !== currentGroup) {
                currentGroup = a.group;
                const h = document.createElement('div');
                h.className = 'shortcut-group-title';
                h.textContent = currentGroup;
                container.appendChild(h);
            }
            const row = document.createElement('div');
            row.className = 'shortcut-row';

            const lblWrap = document.createElement('div');
            lblWrap.className = 'shortcut-label';
            lblWrap.textContent = a.label;
            if (a.sub) {
                const sub = document.createElement('span');
                sub.className = 'shortcut-sub';
                sub.textContent = `(${a.sub})`;
                lblWrap.appendChild(sub);
            }

            const ctrls = document.createElement('div');
            ctrls.className = 'shortcut-controls';

            const keyBtn = document.createElement('button');
            keyBtn.type = 'button';
            keyBtn.dataset.action = a.id;
            const isRec = state.recordingShortcutAction === a.id;
            const curKey = state.shortcuts[a.id] || '';
            keyBtn.className = 'shortcut-key-btn' + (isRec ? ' recording' : (!curKey ? ' unbound' : ''));
            keyBtn.textContent = isRec ? 'Press a key…' : (curKey ? formatShortcutKey(curKey) : 'Unbound');
            keyBtn.title = 'Click and press a key to rebind';
            keyBtn.addEventListener('click', () => {
                state.recordingShortcutAction = (state.recordingShortcutAction === a.id) ? null : a.id;
                renderShortcutsList();
            });

            const clearBtn = document.createElement('button');
            clearBtn.type = 'button';
            clearBtn.className = 'shortcut-clear-btn';
            clearBtn.dataset.clear = a.id;
            clearBtn.textContent = '×';
            clearBtn.title = 'Unbind this shortcut';
            clearBtn.addEventListener('click', () => {
                state.recordingShortcutAction = null;
                setShortcut(a.id, '');
            });

            ctrls.appendChild(keyBtn);
            ctrls.appendChild(clearBtn);
            row.appendChild(lblWrap);
            row.appendChild(ctrls);
            container.appendChild(row);
        }
    }

    function openShortcutsModal() {
        if (!elements.shortcutsModal) return;
        state.recordingShortcutAction = null;
        renderShortcutsList();
        elements.shortcutsModal.classList.remove('hidden');
    }

    function closeShortcutsModal() {
        if (!elements.shortcutsModal) return;
        state.recordingShortcutAction = null;
        elements.shortcutsModal.classList.add('hidden');
    }

    function startKeyGrab(side, normKey) {
        if (!state.trials.length) return false;
        // If already in a sticky key-grab for the same side, pressing the key again commits & ends it
        if (state.handleDrag && state.handleDrag.viaKey && state.handleDrag.sticky && state.handleDrag.side === side) {
            commitHandleDrag(state.handleDrag);
            state.handleDrag = null;
            updateStatus();
            render();
            return true;
        }
        // If already in a key-grab for the other side, commit it first
        if (state.handleDrag && state.handleDrag.viaKey) {
            commitHandleDrag(state.handleDrag);
            state.handleDrag = null;
        }
        const t = state.currentTrialIdx;
        let ev = null, on0 = -1, off0 = -1, isCurrent = false;
        if (state.pass === PASS_TWO && state.currentMarkAt >= 0 && state.currentPutativeOnset >= 0) {
            on0 = state.currentPutativeOnset;
            off0 = state.currentPutativeOffset;
            isCurrent = true;
            if (!state.nearMiss.active) {
                ev = (state.expertEvents[t] || []).find(e => e.onset === on0 && e.offset === off0) ||
                     (state.expertEvents[t] || []).find(e => e.onset <= state.currentMarkAt && state.currentMarkAt <= e.offset) ||
                     null;
            }
        } else if (state.pass === PASS_ONE) {
            const vis = (state.expertEvents[t] || []).filter(e => e.offset >= state.fromSample && e.onset <= state.toSample);
            if (state.hoverSample >= 0) {
                ev = vis.find(e => e.onset - 25 <= state.hoverSample && state.hoverSample <= e.offset + 25) || null;
            }
            if (!ev && vis.length === 1) ev = vis[0];
            if (ev) {
                on0 = ev.onset;
                off0 = ev.offset;
            }
        }
        if (on0 < 0 || off0 < 0) return false;

        const hd = {
            ev,
            side,
            onset: on0,
            offset: off0,
            on0,
            off0,
            current: isCurrent,
            viaKey: true,
            key: normKey,
            moved: false,
            sticky: false,
        };
        if (state.hoverSample >= 0) {
            if (side === 'on') hd.onset = Math.max(0, Math.min(state.hoverSample, hd.offset - 1));
            else hd.offset = Math.min(curLen() - 1, Math.max(state.hoverSample, hd.onset + 1));
            if (hd.current) {
                state.currentPutativeOnset = hd.onset;
                state.currentPutativeOffset = hd.offset;
            }
            hd.moved = (hd.onset !== hd.on0 || hd.offset !== hd.off0);
        }
        state.handleDrag = hd;
        render();
        return true;
    }

    function finishKeyGrab(normKey) {
        const hd = state.handleDrag;
        if (!hd || !hd.viaKey || hd.key !== normKey) return;
        if (hd.sticky) return;
        if (hd.moved || state.hoverSample >= 0) {
            commitHandleDrag(hd);
            state.handleDrag = null;
            updateStatus();
            render();
        } else {
            hd.sticky = true;
            render();
        }
    }

    function anyModalOpen() {
        const modals = [
            elements.startModal,
            elements.loadModal,
            document.getElementById('importModal'),
            elements.pickLabelsVarModal,
            elements.filterModal,
            elements.metricsModal,
            elements.viewedModal,
            elements.alertModal,
            elements.shortcutsModal,
        ];
        return modals.some(m => m && !m.classList.contains('hidden'));
    }

    function executeShortcutAction(actionId, normKey) {
        switch (actionId) {
            case 'prevWindowOrEvent':
                if (state.pass === PASS_ONE) elements.prevEpochBtn.click();
                else elements.prevEventBtn.click();
                return true;
            case 'nextWindowOrEvent':
                if (state.pass === PASS_ONE) elements.nextEpochBtn.click();
                else elements.nextEventBtn.click();
                return true;
            case 'prevTrial':
                elements.prevTrialBtn.click();
                return true;
            case 'nextTrial':
                elements.nextTrialBtn.click();
                return true;
            case 'switchPass':
                elements.switchPassBtn.click();
                return true;
            case 'grabOnset':
                return startKeyGrab('on', normKey);
            case 'grabOffset':
                return startKeyGrab('off', normKey);
            case 'acceptEvent':
                if (state.handleDrag && state.handleDrag.viaKey) {
                    commitHandleDrag(state.handleDrag);
                    state.handleDrag = null;
                }
                if (state.pass === PASS_TWO) {
                    elements.acceptPreBtn.click();
                    return true;
                }
                if (state.pass === PASS_ONE && state.hoverSample >= 0) {
                    handleTap(state.hoverSample);
                    return true;
                }
                return false;
            case 'notAnEvent':
                if (state.handleDrag && state.handleDrag.viaKey) {
                    state.handleDrag = null;
                }
                if (state.pass === PASS_TWO) {
                    elements.notEventBtn.click();
                    return true;
                }
                if (state.pass === PASS_ONE && state.hoverSample >= 0) {
                    deleteMarkAt(state.currentTrialIdx, state.hoverSample);
                    updateStatus();
                    render();
                    return true;
                }
                return false;
            case 'wider':
                if (state.pass === PASS_TWO) elements.widerBtn.click();
                return true;
            case 'nearMisses':
                if (state.hasProb && elements.nearMissBtn) elements.nearMissBtn.click();
                return true;
            case 'saccadeTool':
                if (elements.saccadeBtn) elements.saccadeBtn.click();
                return true;
            case 'blinkTool':
                if (elements.blinkBtn) elements.blinkBtn.click();
                return true;
            case 'toggleClass':
                if (elements.classToggle) elements.classToggle.click();
                return true;
            case 'eraserTool':
                if (elements.eraserBtn) elements.eraserBtn.click();
                return true;
            case 'undo':
                if (elements.undoBtn) elements.undoBtn.click();
                return true;
            case 'autoSlide':
                if (elements.autoSlideBtn) elements.autoSlideBtn.click();
                return true;
            case 'viewedPeriods':
                if (elements.viewedBtn) elements.viewedBtn.click();
                return true;
            case 'netSuggestions':
                if (elements.netSuggestBtn) elements.netSuggestBtn.click();
                return true;
            case 'openFile':
                if (elements.openBtn) elements.openBtn.click();
                return true;
            case 'saveLabels':
                if (elements.saveLabelsBtn) elements.saveLabelsBtn.click();
                return true;
            case 'loadLabels':
                if (elements.loadLabelsBtn) elements.loadLabelsBtn.click();
                return true;
            case 'filterSettings':
                if (elements.filterBtn) elements.filterBtn.click();
                return true;
            case 'metricsModal':
                if (elements.metricsBtn) elements.metricsBtn.click();
                return true;
            case 'trainModel':
                if (elements.trainNowBtn) elements.trainNowBtn.click();
                return true;
            case 'downloadModel':
                if (elements.downloadModelBtn) elements.downloadModelBtn.click();
                return true;
            case 'exportData':
                if (elements.exportBtn) elements.exportBtn.click();
                return true;
            case 'toggleMenu':
                if (elements.menuBtn) elements.menuBtn.click();
                return true;
            case 'openShortcuts':
                openShortcutsModal();
                return true;
            default:
                return false;
        }
    }

    function setupKeyboardShortcuts() {
        loadShortcuts();
        if (elements.shortcutsBtn) elements.shortcutsBtn.addEventListener('click', openShortcutsModal);
        if (elements.shortcutsTopBtn) elements.shortcutsTopBtn.addEventListener('click', openShortcutsModal);
        if (elements.shortcutsCloseBtn) elements.shortcutsCloseBtn.addEventListener('click', closeShortcutsModal);
        if (elements.shortcutsDoneBtn) elements.shortcutsDoneBtn.addEventListener('click', closeShortcutsModal);
        if (elements.shortcutsResetBtn) elements.shortcutsResetBtn.addEventListener('click', () => applyShortcutPreset('qwerty'));
        if (elements.shortcutsPresetQwerty) elements.shortcutsPresetQwerty.addEventListener('click', () => applyShortcutPreset('qwerty'));
        if (elements.shortcutsPresetAzerty) elements.shortcutsPresetAzerty.addEventListener('click', () => applyShortcutPreset('azerty'));

        window.addEventListener('keydown', (e) => {
            // 1. If recording a custom shortcut in the Shortcuts Modal
            if (state.recordingShortcutAction) {
                const norm = normalizeShortcutKey(e);
                if (!norm) return;
                e.preventDefault();
                e.stopPropagation();
                const actId = state.recordingShortcutAction;
                state.recordingShortcutAction = null;
                if (norm === 'Escape') {
                    renderShortcutsList();
                    return;
                }
                if (norm === 'Backspace' || norm === 'Delete') {
                    setShortcut(actId, '');
                    return;
                }
                setShortcut(actId, norm);
                return;
            }

            // 2. Ignore when typing in input, select, textarea, or contentEditable
            const tag = e.target && e.target.tagName ? e.target.tagName.toUpperCase() : '';
            if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable)) {
                return;
            }

            // 3. Standard Ctrl / Cmd combinations (Undo, Save)
            if (e.ctrlKey || e.metaKey) {
                const k = e.key ? e.key.toLowerCase() : '';
                if (k === 'z') {
                    e.preventDefault();
                    undo(state.currentTrialIdx);
                } else if (k === 's') {
                    e.preventDefault();
                    saveLabelsJson();
                }
                return;
            }
            if (e.altKey) return;

            const norm = normalizeShortcutKey(e);
            if (!norm) return;

            // 4. Escape cancels active handle grab or closes open modals
            if (norm === 'Escape') {
                if (state.handleDrag && state.handleDrag.viaKey) {
                    e.preventDefault();
                    const hd = state.handleDrag;
                    if (hd.current) {
                        state.currentPutativeOnset = hd.on0;
                        state.currentPutativeOffset = hd.off0;
                    }
                    state.handleDrag = null;
                    render();
                    return;
                }
                if (elements.shortcutsModal && !elements.shortcutsModal.classList.contains('hidden')) { closeShortcutsModal(); return; }
                if (elements.viewedModal && !elements.viewedModal.classList.contains('hidden')) { elements.viewedModal.classList.add('hidden'); return; }
                if (elements.metricsModal && !elements.metricsModal.classList.contains('hidden')) { elements.metricsModal.classList.add('hidden'); return; }
                if (elements.filterModal && !elements.filterModal.classList.contains('hidden')) { elements.filterModal.classList.add('hidden'); return; }
                return;
            }

            // 5. If any modal is open, only allow closing shortcutsModal with its shortcut key
            if (anyModalOpen()) {
                if (elements.shortcutsModal && !elements.shortcutsModal.classList.contains('hidden') && state.shortcuts.openShortcuts === norm) {
                    e.preventDefault();
                    closeShortcutsModal();
                }
                return;
            }

            // 6. While grabbing a boundary handle via keyboard, ArrowLeft/ArrowRight nudge the boundary, and Enter commits
            if (state.handleDrag && state.handleDrag.viaKey) {
                const hd = state.handleDrag;
                if (norm === 'ArrowLeft' || norm === 'ArrowRight' || norm === state.shortcuts.prevWindowOrEvent || norm === state.shortcuts.nextWindowOrEvent) {
                    const isLeft = (norm === 'ArrowLeft' || norm === state.shortcuts.prevWindowOrEvent);
                    const step = (e.shiftKey ? 5 : 1) * (isLeft ? -1 : 1);
                    e.preventDefault();
                    if (hd.side === 'on') hd.onset = Math.max(0, Math.min(hd.onset + step, hd.offset - 1));
                    else hd.offset = Math.min(curLen() - 1, Math.max(hd.offset + step, hd.onset + 1));
                    if (hd.current) {
                        state.currentPutativeOnset = hd.onset;
                        state.currentPutativeOffset = hd.offset;
                    }
                    hd.moved = true;
                    render();
                    return;
                }
                if (norm === 'Enter') {
                    e.preventDefault();
                    commitHandleDrag(hd);
                    state.handleDrag = null;
                    updateStatus();
                    render();
                    return;
                }
            }

            // 7. Match configured shortcut action
            const matched = SHORTCUT_ACTIONS.find(a => state.shortcuts[a.id] === norm);
            if (matched) {
                if ((matched.id === 'grabOnset' || matched.id === 'grabOffset') && e.repeat) {
                    e.preventDefault();
                    return;
                }
                e.preventDefault();
                executeShortcutAction(matched.id, norm);
                return;
            }

            // 8. Built-in fallbacks in Pass 2 if not bound elsewhere (Enter = Accept, Delete/Backspace = Not an Event)
            if (state.pass === PASS_TWO) {
                if (norm === 'Enter') {
                    e.preventDefault();
                    executeShortcutAction('acceptEvent', norm);
                } else if (norm === 'Delete' || norm === 'Backspace') {
                    e.preventDefault();
                    executeShortcutAction('notAnEvent', norm);
                }
            }
        });

        window.addEventListener('keyup', (e) => {
            const norm = normalizeShortcutKey(e);
            if (!norm) return;
            finishKeyGrab(norm);
        });
    }

    // --- Init ---
    function init() {
        try { const w = parseFloat(localStorage.getItem('uneye_winMs')); if (w >= 20) state.winMs = w; } catch (e) { }
        if (elements.winMsInput) elements.winMsInput.value = String(state.winMs);
        if (elements.modalWinMs) elements.modalWinMs.value = String(state.winMs);
        try { const v = parseFloat(localStorage.getItem('uneye_nearMissLower')); if (v > 0 && v < 0.5) state.nearMiss.lower = v; } catch (e) { }
        if (elements.nearMissLowerInput) elements.nearMissLowerInput.value = String(state.nearMiss.lower);
        setupListeners();
        setupPointerEvents();
        setupKeyboardShortcuts();
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
    window.loadPreviousLabels = loadPreviousLabels;
    window.loadCustomModel = loadCustomModel;
    window.switchActiveModelSource = switchActiveModelSource;
    window.saveLabelsJson = saveLabelsJson;
    window.recomputeReviewed = recomputeReviewed;
    window.buildRefineQueue = buildRefineQueue;
    window.loadDataset = loadDataset;
    window.enterNearMiss = enterNearMiss;
    window.exitNearMiss = exitNearMiss;
    window.nearMissAccept = nmAccept;
    window.nearMissReject = nmReject;
    window.setNearMissLower = setNearMissLower;
    window.openViewedPeriods = openViewedPeriods;
    window.resetViewedPeriods = resetViewedPeriods;
    window.jumpToTrialTime = jumpToTrialTime;
    window.viewedTotals = viewedTotals;
    window.SHORTCUT_ACTIONS = SHORTCUT_ACTIONS;
    window.setShortcut = setShortcut;
    window.applyShortcutPreset = applyShortcutPreset;
    window.openShortcutsModal = openShortcutsModal;
    window.closeShortcutsModal = closeShortcutsModal;
})();
