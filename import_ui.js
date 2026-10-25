// import_ui.js - the column explorer: opens a text export (CSV, ARFF, EyeLink .asc ...), proposes X, Y and time (import.js),
// shows the choice on a live preview and returns the trials. UnEyeImportUI.run(file, {rate, scale}) -> Promise<result | null>.
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const COLORS = { x: '#1E88E5', y: '#43A047', speed: '#7E57C2', bad: 'rgba(198, 40, 40, 0.28)', axis: '#90A4AE' };
  const OPS = ['=', '!=', '>', '>=', '<', '<='];

  let S = null;          // the session being edited

  function el() {
    return {
      modal: $('importModal'), summary: $('impSummary'), warn: $('impWarn'), body: $('impBody'), list: $('impColList'),
      canvas: $('impPreview'), info: $('impPreviewInfo'), prev: $('impPrevBtn'), next: $('impNextBtn'),
      x: $('impX'), y: $('impY'), t: $('impT'), unit: $('impUnit'), rate: $('impRate'), deg: $('impDeg'), hint: $('impTimeHint'),
      resRow: $('impResampleRow'), resample: $('impResample'), vcol: $('impVCol'), vop: $('impVOp'), vval: $('impVVal'),
      split: $('impSplit'), splitCol: $('impSplitCol'), splitLbl: $('impSplitColLabel'), gap: $('impGapSec'), chunk: $('impChunk'),
      result: $('impResult'), load: $('impLoadBtn'), cancel: $('impCancelBtn'), close: $('impCloseBtn'),
      scrPx: $('impScrPx'), scrCm: $('impScrCm'), scrDist: $('impScrDist'), scrApply: $('impScrApply'),
    };
  }

  const fmt = (v) => {
    if (!(Math.abs(v) < Infinity)) return '–';
    const a = Math.abs(v);
    return a === 0 ? '0' : a >= 1e6 || a < 1e-3 ? v.toExponential(1) : a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(1) : v.toFixed(3);
  };
  const optionsFor = (table, stats, pred, none) => {
    let h = none ? '<option value="-1">' + none + '</option>' : '';
    table.cols.forEach((c, i) => { if (pred(c, stats[i], i)) h += '<option value="' + i + '">' + c.name.replace(/&/g, '&amp;').replace(/</g, '&lt;') + '</option>'; });
    return h;
  };

  // ------------------------------------------------------------ run
  function run(file, opts) {
    opts = opts || {};
    const E = el();
    return new Promise((resolve) => {
      S = { E, done: resolve, file, opts, table: null, busy: true };
      E.modal.classList.remove('hidden');
      setBusy('Reading ' + file.name + '…');
      const finish = (val) => { E.modal.classList.add('hidden'); unbind(); S = null; resolve(val); };
      S.finish = finish;
      bind();
      window.UnEyeImport.parseFile(file).then((table) => {
        if (!S) return;
        if (table.nRows < 2 || table.cols.length < 2) throw new Error('The file has fewer than 2 columns or 2 rows of numbers.');
        S.table = table; S.stats = window.UnEyeImport.describeColumns(table);
        init();
      }).catch((err) => { alert('Cannot read ' + file.name + ': ' + err.message); finish(null); });
    });
  }

  function setBusy(msg) {
    const E = S.E;
    S.busy = true;
    E.summary.textContent = msg;
    E.body.classList.add('hidden'); E.warn.classList.add('hidden'); E.result.textContent = '';
    E.load.disabled = true;
  }

  // ------------------------------------------------------------ session setup
  function init() {
    const I = window.UnEyeImport, E = S.E, T = S.table, st = S.stats;
    const sug = I.suggest(T, st);
    const rate0 = sug.time && sug.time.rate ? sug.time.rate : (S.opts.rate || 1000);
    S.m = {
      x: sug.x, y: sug.y, t: sug.t, unit: sug.time ? sug.time.unit : 'samples', fs: rate0,
      valid: sug.valid ? { col: sug.valid.col, op: sug.valid.op, val: sug.valid.val } : null,
      split: sug.trial >= 0 ? { mode: 'column', col: sug.trial, gapSec: 1 } : { mode: 'none', col: -1, gapSec: 1 },
      chunkSec: 0, resample: sug.time ? !sug.time.regular : false, gapFactor: 3,
    };
    S.timeInfo = sug.time;
    S.deg = S.opts.scale || 1;
    S.prevStart = 0;
    S.notes = sug.notes.slice();

    const numeric = (c, s) => !s.nominal && s.finite >= 0.05 * s.n;
    E.x.innerHTML = optionsFor(T, st, numeric, '(choose)');
    E.y.innerHTML = optionsFor(T, st, numeric, '(choose)');
    E.t.innerHTML = optionsFor(T, st, numeric, '(none: use the sampling rate)');
    E.vcol.innerHTML = optionsFor(T, st, numeric, '(all samples)');
    E.splitCol.innerHTML = optionsFor(T, st, (c, s) => s.nominal || s.isInt, '');
    if (sug.trial < 0 && E.splitCol.options.length) S.m.split.col = parseInt(E.splitCol.options[0].value, 10);

    const info = T.kind === 'asc' ? 'EyeLink ASCII, ' + T.info.blocks + ' recording block' + (T.info.blocks === 1 ? '' : 's') + (T.info.rate ? ', ' + T.info.rate + ' Hz' : '')
      : T.kind === 'arff' ? 'ARFF' + (T.info.relation ? ' "' + T.info.relation + '"' : '')
      : (T.info.delimiter === '\t' ? 'tab' : T.info.delimiter === 'ws' ? 'space' : T.info.delimiter === null ? 'single column' : '"' + T.info.delimiter + '"') + ' separated' + (T.info.header ? ', with a header' : ', no header');
    E.summary.textContent = S.file.name + ' · ' + info + ' · ' + T.nRows.toLocaleString() + ' rows × ' + T.cols.length + ' variables';
    E.body.classList.remove('hidden');
    S.busy = false;

    // a very long single recording is hard to label: propose parts
    if (S.m.split.mode === 'none' && S.timeInfo && S.timeInfo.rate && T.nRows / S.timeInfo.rate > 300) S.m.chunkSec = 60;
    buildList();
    syncForm();
    refresh();
  }

  function buildList() {
    const E = S.E, T = S.table, st = S.stats;
    E.list.innerHTML = '';
    T.cols.forEach((c, i) => {
      const s = st[i];
      const row = document.createElement('div');
      row.className = 'import-col';
      row.innerHTML = '<span class="nm"></span><div class="roles"></div><span class="st"></span><canvas></canvas>';
      row.querySelector('.nm').textContent = c.name;
      row.querySelector('.st').textContent = s.nominal ? 'text · ' + s.distinct + ' values' : 'range ' + fmt(s.min) + ' … ' + fmt(s.max) + (s.nanFrac > 0.005 ? ' · ' + Math.round(s.nanFrac * 100) + '% empty' : '');
      const roles = row.querySelector('.roles');
      if (!s.nominal) {
        for (const [key, lbl] of [['x', 'X'], ['y', 'Y'], ['t', 'T']]) {
          const b = document.createElement('button');
          b.type = 'button'; b.className = 'import-role'; b.textContent = lbl; b.dataset.role = key; b.dataset.i = i;
          b.addEventListener('click', () => { S.m[key] = (S.m[key] === i && key === 't') ? -1 : i; if (key === 't') onTimeChosen(); syncForm(); refresh(); });
          roles.appendChild(b);
        }
        spark(row.querySelector('canvas'), c.data, T.nRows);
      } else row.querySelector('canvas').remove();
      E.list.appendChild(row);
    });
  }

  function spark(cv, data, n) {
    const w = 160, h = 22, dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const g = cv.getContext('2d'); g.scale(dpr, dpr);
    const k = Math.min(w, n), step = n / k;
    let lo = Infinity, hi = -Infinity; const mm = [];
    for (let j = 0; j < k; j++) {
      let a = Infinity, b = -Infinity;
      const s0 = Math.floor(j * step), s1 = Math.max(s0 + 1, Math.floor((j + 1) * step));
      for (let i = s0; i < s1 && i < n; i += Math.max(1, Math.floor((s1 - s0) / 8))) { const v = data[i]; if (v === v && Math.abs(v) !== Infinity) { if (v < a) a = v; if (v > b) b = v; } }
      mm.push([a, b]); if (a < lo) lo = a; if (b > hi) hi = b;
    }
    if (!(hi > lo)) { hi = lo + 1; }
    g.strokeStyle = '#78909C'; g.lineWidth = 1; g.beginPath();
    mm.forEach(([a, b], j) => { if (a === Infinity) return; const x = (j + 0.5) * w / k; g.moveTo(x, h - 2 - (a - lo) / (hi - lo) * (h - 4)); g.lineTo(x, h - 2 - (b - lo) / (hi - lo) * (h - 4)); });
    g.stroke();
  }

  // ------------------------------------------------------------ form <-> mapping
  function syncForm() {
    const E = S.E, m = S.m, I = window.UnEyeImport;
    E.x.value = String(m.x); E.y.value = String(m.y); E.t.value = String(m.t);
    E.unit.value = m.unit;
    E.unit.disabled = m.t < 0;
    E.rate.value = m.fs ? String(m.fs) : '';
    E.deg.value = String(S.deg);
    E.vcol.value = String(m.valid ? m.valid.col : -1);
    E.vop.disabled = E.vval.disabled = !m.valid;
    if (m.valid) { E.vop.value = m.valid.op; E.vval.value = String(m.valid.val); }
    E.split.value = m.split.mode;
    E.splitCol.classList.toggle('hidden', m.split.mode === 'gap'); E.splitCol.value = String(m.split.col);
    E.gap.classList.toggle('hidden', m.split.mode !== 'gap'); E.gap.value = String(m.split.gapSec);
    E.splitLbl.textContent = m.split.mode === 'gap' ? 'Gap longer than (s)' : 'Variable';
    E.splitLbl.parentNode.classList.toggle('hidden', m.split.mode === 'none');
    E.chunk.value = String(m.chunkSec || 0);
    const ti = S.timeInfo;
    E.resRow.classList.toggle('hidden', !(m.t >= 0 && m.unit !== 'samples'));
    E.resample.checked = !!m.resample;
    let hint = '';
    if (m.t >= 0 && ti && ti.ok && m.unit !== 'samples') {
      hint = 'Measured ' + (I.rateFor(ti.meanStep, m.unit) || '?') + ' Hz from "' + S.table.cols[m.t].name + '" (' + { s: 'seconds', ms: 'milliseconds', us: 'microseconds' }[m.unit] + ')'
        + (ti.regular ? ', regular.' : ', not regular: steps vary' + (ti.gaps ? ' and there are ' + ti.gaps + ' gap' + (ti.gaps === 1 ? '' : 's') : '') + ', so resampling is recommended.');
    } else if (m.t >= 0 && m.unit === 'samples') hint = 'The time column is a sample index: set the sampling rate by hand.';
    else if (m.t < 0) hint = 'No time column: samples are taken to be evenly spaced at the rate above.';
    E.hint.textContent = hint;
    document.querySelectorAll('#impColList .import-role').forEach((b) => {
      const key = b.dataset.role, on = m[key] === parseInt(b.dataset.i, 10);
      b.className = 'import-role' + (on ? ' on-' + key : '');
    });
  }

  function onTimeChosen() {
    const I = window.UnEyeImport, m = S.m;
    S.timeInfo = null;
    if (m.t < 0) { m.unit = 'samples'; m.resample = false; return; }
    const ti = I.analyzeTimeColumn(S.table, m.t, S.stats);
    S.timeInfo = ti;
    if (ti.ok) { m.unit = ti.unit; if (ti.rate) m.fs = ti.rate; m.resample = ti.unit !== 'samples' && !ti.regular; }
    else { m.unit = 'ms'; m.resample = false; S.notes = ['The steps of "' + S.table.cols[m.t].name + '" do not look like time: pick the unit and the rate by hand.']; }
  }

  function readForm() {
    const E = S.E, m = S.m, I = window.UnEyeImport;
    const prevT = m.t, prevUnit = m.unit;
    m.x = parseInt(E.x.value, 10); m.y = parseInt(E.y.value, 10); m.t = parseInt(E.t.value, 10);
    if (m.t !== prevT) onTimeChosen();
    else if (m.t >= 0) {
      m.unit = E.unit.value;
      if (m.unit !== prevUnit && S.timeInfo && S.timeInfo.meanStep) { const r = m.unit === 'samples' ? null : I.rateFor(S.timeInfo.meanStep, m.unit); if (r) m.fs = r; }
      m.resample = E.resample.checked;
    }
    const fs = parseFloat(E.rate.value); if (fs > 0) m.fs = fs;
    const dg = parseFloat(E.deg.value); if (dg > 0) S.deg = dg;
    const vc = parseInt(E.vcol.value, 10);
    m.valid = vc >= 0 ? { col: vc, op: E.vop.value, val: parseFloat(E.vval.value) || 0 } : null;
    m.split.mode = E.split.value;
    m.split.col = parseInt(E.splitCol.value, 10); if (isNaN(m.split.col)) m.split.col = -1;
    m.split.gapSec = parseFloat(E.gap.value) > 0 ? parseFloat(E.gap.value) : 1;
    m.chunkSec = parseFloat(E.chunk.value) > 0 ? parseFloat(E.chunk.value) : 0;
    if (m.split.mode === 'gap' && !(m.t >= 0 && m.unit !== 'samples')) { m.split.mode = 'none'; S.notes = ['Splitting at gaps needs a time column.']; }
    if (m.split.mode === 'column' && m.split.col < 0) m.split.mode = 'none';
  }

  // ------------------------------------------------------------ preview and checks
  let raf = 0;
  function refresh() {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => { raf = 0; if (S && !S.busy) { draw(); check(); } });
  }

  function windowRows() {
    const m = S.m, n = S.table.nRows, span = Math.max(50, Math.round((m.fs || 1000) * 10));
    const r0 = Math.max(0, Math.min(S.prevStart, Math.max(0, n - 2)));
    return { r0, len: Math.min(span, n - r0), span };
  }

  function draw() {
    const E = S.E, m = S.m, T = S.table, I = window.UnEyeImport;
    const cv = E.canvas, dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth || 600, H = cv.clientHeight || 220;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const { r0, len } = windowRows();
    E.info.textContent = m.x >= 0 && m.y >= 0 ? 'rows ' + (r0 + 1).toLocaleString() + ' – ' + (r0 + len).toLocaleString() + ' of ' + T.nRows.toLocaleString() : '';
    if (m.x < 0 || m.y < 0) { g.fillStyle = '#78909C'; g.font = '14px sans-serif'; g.fillText('Choose X and Y to see the signal.', 14, 28); return; }
    const sub = { nRows: len, cols: T.cols.map((c) => ({ data: c.data.subarray(r0, r0 + len) })) };
    const ok = I.validArray(sub, { x: m.x, y: m.y, valid: m.valid });
    const X = sub.cols[m.x].data, Y = sub.cols[m.y].data;
    const laneH = Math.floor(H * 0.62), pad = 16, left = 4, right = 4, w = W - left - right;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < len; i++) if (ok[i]) { for (const v of [X[i], Y[i]]) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    if (!(hi > lo)) { lo = (isFinite(lo) ? lo : 0) - 1; hi = lo + 2; }
    const yPos = (v) => pad + (1 - (v - lo) / (hi - lo)) * (laneH - 2 * pad);
    const px = (i) => left + (len > 1 ? i / (len - 1) : 0) * w;
    const trace = (arr, color, f) => {            // min / max per pixel column when there are more samples than pixels
      g.strokeStyle = color; g.lineWidth = 1.4; g.beginPath();
      let started = false;
      if (len > 2 * w) {
        let col = -1, a = 0, b = 0;
        const flush = () => { const x = left + col + 0.5; if (!started) { g.moveTo(x, f(a)); started = true; } else g.lineTo(x, f(a)); if (b !== a) g.lineTo(x, f(b)); };
        for (let i = 0; i < len; i++) {
          if (!ok[i]) { if (col >= 0) { flush(); col = -1; } started = false; continue; }
          const c = Math.floor((px(i) - left)), v = arr[i];
          if (c !== col) { if (col >= 0) flush(); col = c; a = b = v; } else { if (v < a) a = v; if (v > b) b = v; }
        }
        if (col >= 0) flush();
      } else {
        for (let i = 0; i < len; i++) { if (!ok[i]) { started = false; continue; } if (!started) { g.moveTo(px(i), f(arr[i])); started = true; } else g.lineTo(px(i), f(arr[i])); }
      }
      g.stroke();
    };
    // invalid stretches
    g.fillStyle = COLORS.bad;
    for (let i = 0, s = -1; i <= len; i++) {
      const bad = i < len && !ok[i];
      if (bad && s < 0) s = i; else if (!bad && s >= 0) { g.fillRect(px(s), 0, Math.max(2, px(i - 1) - px(s)), laneH); s = -1; }
    }
    trace(X, COLORS.x, yPos); trace(Y, COLORS.y, yPos);
    g.fillStyle = COLORS.axis; g.font = '11px sans-serif';
    g.fillText(fmt(hi), left + 2, 12); g.fillText(fmt(lo), left + 2, laneH - 4);
    // speed lane
    const fs = m.fs || 1000, sp = new Float32Array(len);
    let big = [];
    for (let i = 1; i < len; i++) if (ok[i] && ok[i - 1]) { sp[i] = Math.hypot(X[i] - X[i - 1], Y[i] - Y[i - 1]) * S.deg * fs; big.push(sp[i]); }
    big.sort((a, b) => a - b);
    const top = Math.max(50, (big.length ? big[Math.floor(big.length * 0.995)] : 50) * 1.2);
    const y0 = laneH + 4, y1 = H - 6;
    g.strokeStyle = '#ECEFF1'; g.beginPath(); g.moveTo(0, laneH); g.lineTo(W, laneH); g.stroke();
    const ys = (v) => y1 - Math.min(v, top) / top * (y1 - y0);
    g.strokeStyle = COLORS.speed; g.lineWidth = 1.2; g.beginPath();
    if (len > 2 * w) {
      let col = -1, b = 0, started = false;
      const flush = () => { const x = left + col + 0.5; if (!started) { g.moveTo(x, ys(0)); started = true; } g.lineTo(x, ys(b)); };
      for (let i = 1; i < len; i++) { const c = Math.floor(px(i) - left); if (c !== col) { if (col >= 0) flush(); col = c; b = sp[i]; } else if (sp[i] > b) b = sp[i]; }
      if (col >= 0) flush();
    } else for (let i = 1; i < len; i++) { if (i === 1) g.moveTo(px(i), ys(sp[i])); else g.lineTo(px(i), ys(sp[i])); }
    g.stroke();
    g.fillStyle = COLORS.axis; g.fillText(Math.round(top) + ' °/s', left + 2, y0 + 10);
  }

  function peakOfStart() {
    const m = S.m, T = S.table, n = Math.min(T.nRows, 200000);
    const I = window.UnEyeImport;
    const sub = { nRows: n, cols: T.cols.map((c) => ({ data: c.data.subarray(0, n) })) };
    const ok = I.validArray(sub, { x: m.x, y: m.y, valid: m.valid }), X = sub.cols[m.x].data, Y = sub.cols[m.y].data;
    const v = [];
    for (let i = 1; i < n; i++) if (ok[i] && ok[i - 1]) v.push(Math.hypot(X[i] - X[i - 1], Y[i] - Y[i - 1]) * S.deg * (m.fs || 1000));
    if (v.length < 10) return NaN;
    v.sort((a, b) => a - b);
    return v[Math.min(v.length - 1, Math.floor(v.length * 0.999))];
  }

  function check() {
    const E = S.E, m = S.m, I = window.UnEyeImport;
    const msgs = S.notes.slice();
    let good = true;
    if (m.x < 0 || m.y < 0) { good = false; E.result.className = 'import-result bad'; E.result.textContent = 'Choose the X and Y variables.'; }
    else if (m.x === m.y) { good = false; E.result.className = 'import-result bad'; E.result.textContent = 'X and Y must be two different variables.'; }
    else if (!(m.fs > 0)) { good = false; E.result.className = 'import-result bad'; E.result.textContent = 'Set the sampling rate.'; }
    else {
      const c = I.countTrials(S.table, m);
      const peak = peakOfStart();
      let verdict = '', cls = 'good';
      if (isNaN(peak)) { verdict = 'not enough valid samples to check the units'; cls = 'bad'; }
      else if (peak > 1500) { verdict = 'peak speed ' + Math.round(peak).toLocaleString() + ' °/s is too high for eye movements: check "Degrees per unit" and the rate'; cls = 'bad'; }
      else if (peak < 30) { verdict = 'peak speed only ' + Math.round(peak) + ' °/s: check "Degrees per unit" and the rate'; cls = 'bad'; }
      else verdict = 'peak speed ≈ ' + Math.round(peak) + ' °/s, plausible';
      E.result.className = 'import-result ' + cls;
      E.result.textContent = '→ ' + c.trials + ' trial' + (c.trials === 1 ? '' : 's') + ', ' + (c.samples / m.fs).toFixed(0) + ' s at ' + m.fs + ' Hz · ' + verdict + '.';
      if (c.trials > 5000) msgs.push('More than 5000 trials: is the trial variable the right one?');
    }
    E.warn.textContent = msgs.join(' ');
    E.warn.classList.toggle('hidden', !msgs.length);
    E.load.disabled = !good;
  }

  // ------------------------------------------------------------ events
  let handlers = [];
  function on(target, type, fn) { target.addEventListener(type, fn); handlers.push([target, type, fn]); }
  function bind() {
    const E = S.E;
    const changed = () => { if (!S || S.busy) return; S.notes = []; readForm(); syncForm(); refresh(); };
    for (const id of ['x', 'y', 't', 'unit', 'vcol', 'vop', 'split', 'splitCol', 'resample']) on(E[id], 'change', changed);
    for (const id of ['rate', 'deg', 'vval', 'gap', 'chunk']) on(E[id], 'input', () => { if (!S || S.busy) return; S.notes = []; readForm(); refresh(); });
    on(E.prev, 'click', () => { S.prevStart = Math.max(0, S.prevStart - windowRows().span); refresh(); });
    on(E.next, 'click', () => { const w = windowRows(); if (S.prevStart + w.span < S.table.nRows - 2) S.prevStart += w.span; refresh(); });
    on(E.scrApply, 'click', () => {
      const d = window.UnEyeImport.degPerPixel(parseFloat(E.scrPx.value), parseFloat(E.scrCm.value), parseFloat(E.scrDist.value));
      if (d > 0) { S.deg = Math.round(d * 1e6) / 1e6; E.deg.value = String(S.deg); refresh(); }
    });
    on(E.cancel, 'click', () => S.finish(null));
    on(E.close, 'click', () => S.finish(null));
    on(E.load, 'click', load);
    on(window, 'resize', () => { if (S && !S.busy) refresh(); });
  }
  function unbind() { for (const [t, ty, fn] of handlers) t.removeEventListener(ty, fn); handlers = []; }

  function load() {
    const E = S.E, m = S.m;
    readForm();
    const sess = S;
    setBusy('Building trials…');
    setTimeout(() => {
      try {
        const r = window.UnEyeImport.buildTrials(sess.table, m);
        if (!r.trials.length) {
          sess.busy = false; sess.E.body.classList.remove('hidden'); E.load.disabled = false;
          E.summary.textContent = sess.file.name;
          S.notes = ['No trial has enough valid samples with this choice (check X, Y and the validity rule).'];
          check();
          return;
        }
        sess.finish({ trials: r.trials, fs: r.fs, degPerUnit: S.deg, notes: r.notes, mapping: JSON.parse(JSON.stringify(m)), source: sess.table.kind });
      } catch (err) {
        sess.busy = false; sess.E.body.classList.remove('hidden');
        alert('Cannot build the trials: ' + err.message);
        E.load.disabled = false;
      }
    }, 30);
  }

  window.UnEyeImportUI = { run };
}());
