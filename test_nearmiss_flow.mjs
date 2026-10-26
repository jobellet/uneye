// Headless-Chrome test of a file with the network's output (prob, pred_labels): proposals from the file, probability band, near-miss queue
// (sorted by decreasing peak, accept with a dragged bound, reject), viewed periods window, save / reload / autosave of the new fields.
// Usage: node test_nearmiss_flow.mjs [url]   (puppeteer-core; CHROME=/path/to/chrome; serve the repo with `python3 -m http.server 8765`)
// MOCOL_H5=/path/to/session.h5 also loads a real session (xy, trial_offset, names, fs, pred_labels, prob) and times the near-miss queue
// and the viewed-periods window. Screenshots: SHOTS=/tmp/uneye (default) -> _nearmiss.png, _viewed.png (and _session_*.png).
import puppeteer from 'puppeteer-core';
const URL = process.argv[2] || 'http://127.0.0.1:8765/index.html';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const SHOTS = process.env.SHOTS || '/tmp/uneye';
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('dialog', async d => { await d.accept().catch(() => {}); });
await page.setViewport({ width: 1100, height: 680 });
await page.evaluateOnNewDocument(() => {
  Element.prototype.setPointerCapture = () => {}; Element.prototype.releasePointerCapture = () => {};
  const blobs = new Map(), orig = URL.createObjectURL.bind(URL);
  URL.createObjectURL = b => { const u = orig(b); blobs.set(u, b); return u; };
  window.__dl = [];
  HTMLAnchorElement.prototype.click = function () { if (!this.download) return; const name = this.download, b = blobs.get(this.href), fr = new FileReader(); fr.onload = () => window.__dl.push({ name, text: String(fr.result) }); fr.readAsText(b); };
});
await page.goto(URL, { waitUntil: 'networkidle0', timeout: 120000 });
await page.evaluate(() => new Promise(r => { const q = indexedDB.deleteDatabase('uneye_db'); q.onsuccess = q.onerror = q.onblocked = () => r(); }));
await page.reload({ waitUntil: 'networkidle0' });
let ok = true; const check = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) ok = false; };
const ev = (f, ...a) => page.evaluate(f, ...a);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const waitFor = async (f, ms = 60000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await ev(f)) return true; await sleep(100); } return false; };

// --- a file of 20 trials x 2000 samples made in the page (h5wasm of the app): one saccade per trial (prob 0.97, pred_labels 1), a bump of
// P(saccade) below the threshold at sample 700 (peaks 0.05 .. 0.43, different per trial), a bump of 0.3 at 1600 on even trials, NaN in trial 3
const nT = 20, T = 2000;
await ev(async (nT, T) => {
  const h5 = await (async () => { const sc = document.createElement('script'); sc.src = 'web/h5wasm.js'; document.head.appendChild(sc); await new Promise(r => sc.onload = r); return window.h5wasm; })();
  let s = 7; const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x80000000;
  const N = nT * T, xy = new Float32Array(2 * N), prob = new Float32Array(N), pred = new Int8Array(N), offs = [];
  for (let k = 0; k < nT; k++) {
    offs.push(k * T);
    const sac = 300 + 40 * k, nmPeak = 0.05 + 0.38 * ((k * 7) % 20) / 19;
    let px = 0;
    for (let i = 0; i < T; i++) {
      const g = k * T + i;
      if (i >= sac && i < sac + 20) px += 0.05;
      xy[2 * g] = px + 0.01 * (rnd() - 0.5); xy[2 * g + 1] = 0.01 * (rnd() - 0.5);
      let p = 0.02;
      if (i >= sac - 2 && i < sac + 22) p = 0.97;
      p = Math.max(p, nmPeak * Math.max(0, 1 - Math.abs(i - 700) / 12));
      if (k % 2 === 0) p = Math.max(p, 0.3 * Math.max(0, 1 - Math.abs(i - 1600) / 10));
      if (k === 3 && i >= 1800 && i < 1850) { xy[2 * g] = NaN; xy[2 * g + 1] = NaN; p = NaN; }
      prob[g] = p; pred[g] = p > 0.5 ? 1 : 0;
    }
  }
  offs.push(N);
  const names = Array.from({ length: nT }, (_, k) => 'sess_t' + String(k).padStart(3, '0'));
  const bytes = await UnEyeFormats.buildH5([{ name: 'xy', type: 'f4', shape: [N, 2], data: xy }, { name: 'trial_offset', type: 'i4', shape: [nT + 1], data: Int32Array.from(offs) },
    { name: 'pred_labels', type: 'i1', shape: [N], data: pred }, { name: 'prob', type: 'f4', shape: [N], data: prob }, { name: 'names', strings: names }], { fs: 1000, threshold: 0.5 }, h5);
  const dt = new DataTransfer(); dt.items.add(new File([bytes], 'nearmiss_test.h5'));
  const inp = document.getElementById('fileInput'); inp.files = dt.files; inp.dispatchEvent(new Event('change'));
}, nT, T);
await waitFor(() => !document.getElementById('loadModal').classList.contains('hidden'));
await page.click('#modalConfirmBtn');
check(await waitFor(() => state.trials.length === 20), 'the .h5 file with prob and pred_labels loads (20 trials)');
await sleep(300);

// --- pre-labels loaded at start = immediately treated as ground truth (expertEvents) without needing to confirm one by one
const st = await ev(() => ({ mode: state.prelabelerMode, hasProb: state.hasProb, pre0: state.prelabels[0].map(p => [p.onset, p.offset, p.cls]), exp0: state.expertEvents[0].map(e => [e.onset, e.offset, e.cls]), pre3: state.prelabels[3].map(p => [p.onset, p.offset, p.cls]),
  ev: state.expertEvents.reduce((a, e) => a + e.length, 0), rev: state.reviewedRanges.reduce((a, r) => a + r.length, 0), nmBtn: !document.getElementById('nearMissBtn').classList.contains('hidden') }));
check(st.mode === 'file' && st.hasProb && st.nmBtn, `proposal source 'file', near-miss button shown: ${JSON.stringify([st.mode, st.hasProb, st.nmBtn])}`);
check(JSON.stringify(st.pre0) === JSON.stringify([[298, 321, 1]]), 'the proposals are the runs of pred_labels: ' + JSON.stringify(st.pre0));
check(st.pre3.some(p => p[2] === 2 && p[0] === 1800 && p[1] === 1849), 'blinks still come from the invalid runs: ' + JSON.stringify(st.pre3));
check(st.ev === 20 && JSON.stringify(st.exp0) === JSON.stringify([[298, 321, 1]]), `pre-labels loaded at start are immediately ground-truth expertEvents (ev=${st.ev}, exp0=${JSON.stringify(st.exp0)})`);

// clicking a ground-truth pre-label in Pass 1 deletes it without leaving a dashed prelabel behind, and Undo restores it
await ev(() => handleTap(310));
const afterDel = await ev(() => ({ exp0: state.expertEvents[0].length, pre0: state.prelabels[0].length }));
await page.click('#undoBtn');
const afterUndoDel = await ev(() => ({ exp0: state.expertEvents[0].length, pre0: state.prelabels[0].length }));
check(afterDel.exp0 === 0 && afterDel.pre0 === 0 && afterUndoDel.exp0 === 1 && afterUndoDel.pre0 === 1,
  `clicking a pre-label in Pass 1 deletes it completely (${JSON.stringify(afterDel)}) and Undo restores it (${JSON.stringify(afterUndoDel)})`);

// --- probability band: orange line in the bottom band of the trace
const band = await ev(() => {
  const c = document.getElementById('traceCanvas'), x = c.getContext('2d'), h = c.height, w = c.width;
  const d = x.getImageData(0, Math.round(h * 0.82), w, h - Math.round(h * 0.82)).data; let n = 0;
  for (let i = 0; i < d.length; i += 4) if (d[i] > 200 && d[i + 1] > 80 && d[i + 1] < 140 && d[i + 2] < 60) n++;
  return n;
});
check(band > 100, `the probability band is drawn (${band} orange pixels at the bottom)`);

// --- near misses: from the menu, all trials, decreasing peak
await page.click('#menuBtn');
await page.click('#nearMissMenuBtn');
const nm = await ev(() => ({ active: state.nearMiss.active, pass: state.pass, n: state.nearMiss.queue.length, peaks: state.nearMiss.queue.map(c => c.peak), trials: new Set(state.nearMiss.queue.map(c => c.trial)).size,
  badge: document.getElementById('eventCounterBadge').textContent, cur: [state.currentTrialIdx, state.fromSample, state.toSample, state.currentPutativeOnset, state.currentPutativeOffset], first: state.nearMiss.queue[0] }));
check(nm.active && nm.pass === 2 && nm.n > 10 && nm.trials > 10, `near-miss mode in pass 2, ${nm.n} candidates over ${nm.trials} trials`);
check(nm.peaks.every((p, i) => i === 0 || nm.peaks[i - 1] >= p) && nm.peaks[0] < 0.5 && nm.peaks[nm.peaks.length - 1] >= 0.1, 'sorted by decreasing peak, all in [0.1, 0.5): ' + nm.peaks.map(p => p.toFixed(2)).join(' '));
check(/^Near miss 1\/\d+ \(p = 0\.4\d\)$/.test(nm.badge), 'counter: ' + nm.badge);
check(nm.cur[0] === nm.first.trial && nm.cur[3] === nm.first.onset && nm.cur[4] === nm.first.offset && nm.cur[1] <= nm.first.onset - 45 && nm.cur[2] >= nm.first.offset + 45 && nm.cur[2] - nm.cur[1] < 140,
  'the view is the candidate +-50 ms: ' + JSON.stringify(nm.cur));
await page.click('#menuBtn');
await sleep(200);
await page.screenshot({ path: SHOTS + '_nearmiss.png' });

// Wider, Next, Prev
await page.click('#widerBtn');
const wide = await ev(() => state.toSample - state.fromSample);
await page.click('#widerBtn'); await page.click('#widerBtn');
await page.click('#nextEventBtn');
const nx = await ev(() => [state.nearMiss.idx, document.getElementById('eventCounterBadge').textContent]);
await page.click('#prevEventBtn');
check(wide > 200 && nx[0] === 1 && /^Near miss 2\//.test(nx[1]) && (await ev(() => state.nearMiss.idx)) === 0, `Wider (${wide} samples), Next (${nx[1]}), Prev`);

// accept with the onset handle dragged 6 samples earlier
const drag = await ev(() => {
  const c = document.getElementById('traceCanvas'), r = c.getBoundingClientRect(), w = c.clientWidth, h = c.clientHeight;
  const px = s => r.left + (s - state.fromSample) / (state.toSample - state.fromSample) * w;
  const on = state.currentPutativeOnset, target = on - 6, y = r.top + h / 2;
  const fire = (type, x) => c.dispatchEvent(new PointerEvent(type, { pointerId: 9, clientX: x, clientY: y, bubbles: true, pointerType: 'touch', isPrimary: true }));
  fire('pointerdown', px(on)); fire('pointermove', px(target)); fire('pointerup', px(target));
  return { target, on, after: state.currentPutativeOnset, ev: state.expertEvents[state.currentTrialIdx].length, c: state.nearMiss.queue[0] };
});
check(Math.abs(drag.after - drag.target) <= 1 && drag.ev === 1, `dragging the onset handle moves the bound only (still 1 pre-labeled event): ${drag.on} -> ${drag.after}`);
const n0 = await ev(() => state.nearMiss.queue.length);
await page.click('#acceptPreBtn');
const acc = await ev((t) => ({ evs: state.expertEvents[t].map(e => [e.onset, e.offset, e.cls]), n: state.nearMiss.queue.length, rev: state.reviewedRanges[t] }), drag.c.trial);
check(acc.n === n0 - 1 && acc.evs.length === 2 && acc.evs.some(e => e[0] === drag.after && e[1] === drag.c.offset && e[2] === 1),
  `Accept: an expert saccade with the dragged bounds ${JSON.stringify(acc.evs)}, removed from the queue (${n0} -> ${acc.n})`);
// reject the next one
const rj = await ev(() => ({ c: state.nearMiss.queue[state.nearMiss.idx], win: [state.fromSample, state.toSample] }));
await page.click('#notEventBtn');
const rej = await ev((t) => ({ rej: state.rejectedNearMisses[t], n: state.nearMiss.queue.length, rev: state.reviewedRanges[t], scan: state.scanWins[t] }), rj.c.trial);
check(rej.n === n0 - 2 && rej.rej.length === 1 && rej.rej[0].at === rj.c.at && rej.rev.some(r => r[0] <= rj.win[0] && r[1] >= rj.win[1]),
  `Not an Event: kept as rejected ${JSON.stringify(rej.rej)}, the window ${JSON.stringify(rj.win)} is reviewed ${JSON.stringify(rej.rev)}`);

// undo cancels the rejection: brings back the candidate to the queue, removes it from rejectedNearMisses, re-focuses it
await page.click('#undoBtn');
const unrej = await ev((t) => ({ rej: state.rejectedNearMisses[t], n: state.nearMiss.queue.length, idx: state.nearMiss.idx, badge: document.getElementById('eventCounterBadge').textContent, c: state.nearMiss.queue[state.nearMiss.idx] }), rj.c.trial);
check(unrej.n === n0 - 1 && unrej.rej.length === 0 && unrej.c.at === rj.c.at,
  `Undo cancels the rejection: queue restored (${unrej.n}), candidate removed from rejectedNearMisses, focused (${unrej.badge})`);
// reject it again to continue the rest of the flow
await page.click('#notEventBtn');
// double tap = accept with the proposed bounds
const dt = await ev(() => {
  const c = document.getElementById('traceCanvas'), r = c.getBoundingClientRect(), w = c.clientWidth, h = c.clientHeight, q = state.nearMiss.queue[state.nearMiss.idx];
  const x = r.left + (q.at - state.fromSample) / (state.toSample - state.fromSample) * w, y = r.top + h * 0.3;
  for (let k = 0; k < 2; k++) { c.dispatchEvent(new PointerEvent('pointerdown', { pointerId: 20 + k, clientX: x, clientY: y, bubbles: true, pointerType: 'touch', isPrimary: true })); window.dispatchEvent(new PointerEvent('pointerup', { pointerId: 20 + k, clientX: x, clientY: y, bubbles: true, pointerType: 'touch', isPrimary: true })); }
  return { q, evs: state.expertEvents[q.trial].map(e => [e.onset, e.offset]), n: state.nearMiss.queue.length, marks: state.pointMarks[q.trial].length };
});
check(dt.evs.some(e => e[0] === dt.q.onset && e[1] === dt.q.offset) && dt.n === n0 - 3 && dt.marks === 0, 'a double tap accepts the near miss with its bounds (no stray mark)');
// a rejected near miss is not offered again
await page.click('#nearMissBtn');
check((await ev(() => [state.nearMiss.active, document.getElementById('passIndicator').textContent, state.pass])).join() === 'false,Pass 2: Refine (Onset / Offset),2', 'leaving near-miss mode returns to pass 2');
await page.click('#nearMissBtn');
const again = await ev((at, t) => ({ n: state.nearMiss.queue.length, has: state.nearMiss.queue.some(c => c.trial === t && c.at === at) }), rj.c.at, rj.c.trial);
check(again.n === n0 - 3 && !again.has, `re-opened: ${again.n} candidates, neither the accepted nor the rejected ones`);
// lower bound
await ev(() => setNearMissLower(0.3));
const hi = await ev(() => state.nearMiss.queue.every(c => c.peak >= 0.3) && state.nearMiss.queue.length);
await ev(() => setNearMissLower(0.1));
check(hi > 0 && hi < again.n, `lower bound 0.3: ${hi} candidates`);
await page.click('#switchPassBtn');
check((await ev(() => [state.pass, state.nearMiss.active])).join() === '1,false', 'Switch Pass leaves near-miss mode (pass 1)');

// --- viewed periods
const vr = await ev(() => ({ n: state.viewedRanges.filter(r => r.length).length, t: viewedTotals() }));
check(vr.n >= 4 && vr.t.seen > 0 && vr.t.seen < vr.t.total, `windows drawn on screen are recorded as viewed (${vr.n} trials, ${vr.t.seen} of ${vr.t.total} samples)`);
await page.click('#menuBtn');
await page.click('#viewedBtn');
await sleep(400);
const vw = await ev(() => {
  const c = document.getElementById('viewedCanvas'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; const col = {};
  for (let i = 0; i < d.length; i += 4) { const k = d[i] + ',' + d[i + 1] + ',' + d[i + 2]; col[k] = (col[k] || 0) + 1; }
  return { open: !document.getElementById('viewedModal').classList.contains('hidden'), summary: document.getElementById('viewedSummary').textContent,
    viewed: col['144,202,249'] || 0, reviewed: col['102,187,106'] || 0, sacc: col['13,71,161'] || 0, rej: col['251,140,0'] || 0, none: col['207,216,220'] || 0 };
});
check(vw.open && /Viewed \d+\.\d %/.test(vw.summary) && /trials fully viewed/.test(vw.summary), 'Viewed periods window: ' + vw.summary);
check(vw.viewed > 0 && vw.reviewed > 0 && vw.sacc > 0 && vw.rej > 0 && vw.none > 0, `rows show not viewed / viewed / reviewed / saccades / rejected near misses: ${JSON.stringify(vw)}`);
const box = await (await page.$('#viewedCanvas')).boundingBox();
const rowH = await ev(() => 14);          // 20 trials -> 14 px rows
await page.mouse.move(box.x + 46 + (box.width - 54) * 0.5, box.y + 18 + 7 * rowH + 3);
await sleep(100);
const tip = await ev(() => document.getElementById('viewedTip').textContent);
check(/^Trial 8\/20 · sess_t007 · \d+ ms/.test(tip), 'hover shows the trial: ' + tip);
await page.screenshot({ path: SHOTS + '_viewed.png' });
await page.mouse.click(box.x + 46 + (box.width - 54) * 0.5, box.y + 18 + 7 * rowH + 3);
const jump = await ev(() => [document.getElementById('viewedModal').classList.contains('hidden'), state.pass, state.currentTrialIdx, state.fromSample, state.toSample]);
check(jump[0] && jump[1] === 1 && jump[2] === 7 && jump[3] <= 1000 && jump[4] >= 1000, 'a click goes to that trial and time (pass 1 window holding it): ' + JSON.stringify(jump));

// --- reset viewed/reviewed periods to zero while keeping ground-truth labels (expertEvents) intact
const beforeReset = await ev(() => ({
  viewed: JSON.parse(JSON.stringify(state.viewedRanges)),
  scan: JSON.parse(JSON.stringify(state.scanWins)),
  rev: JSON.parse(JSON.stringify(state.reviewedRanges)),
  nEv: state.expertEvents.reduce((a, e) => a + e.length, 0),
}));
await page.click('#viewedBtn');
await sleep(200);
await page.click('#viewedResetBtn');
await sleep(100);
const afterReset = await ev(() => {
  const c = document.getElementById('viewedCanvas'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; const col = {};
  for (let i = 0; i < d.length; i += 4) { const k = d[i] + ',' + d[i + 1] + ',' + d[i + 2]; col[k] = (col[k] || 0) + 1; }
  const tot = viewedTotals();
  return {
    tot,
    summary: document.getElementById('viewedSummary').textContent,
    resetDisabled: document.getElementById('viewedResetBtn').disabled,
    viewedPx: col['144,202,249'] || 0,
    reviewedPx: col['102,187,106'] || 0,
    saccPx: col['13,71,161'] || 0,
  };
});
check(afterReset.tot.seen === 0 && afterReset.tot.rev === 0 && afterReset.tot.nEv === beforeReset.nEv &&
  afterReset.viewedPx === 0 && afterReset.reviewedPx === 0 && afterReset.saccPx > 0 && afterReset.resetDisabled,
  `Mark all as unseen resets viewed & reviewed to 0 while keeping all ${afterReset.tot.nEv} ground-truth events: ${afterReset.summary}`);
await page.click('#viewedCloseBtn');
// Restore viewed/scan/rev ranges for the subsequent save/reload checks
await ev((b) => { state.viewedRanges = b.viewed; state.scanWins = b.scan; state.reviewedRanges = b.rev; updateStatus(); }, beforeReset);

// --- save labels: viewedRanges and rejectedNearMisses; an old file without them still loads
await ev(() => { window.__dl = []; saveLabelsJson(); });
await waitFor(() => window.__dl.length > 0, 5000);
const saved = JSON.parse(await ev(() => window.__dl[0].text));
check(saved.trials.every(t => Array.isArray(t.viewedRanges)) && saved.trials[rj.c.trial].rejectedNearMisses.length === 1, 'saved labels hold viewedRanges and rejectedNearMisses per trial');
const old = JSON.parse(JSON.stringify(saved)); old.trials.forEach(t => { delete t.viewedRanges; delete t.rejectedNearMisses; });
const ld = await ev((o) => { try { applyLabelsJson(o); return 'ok'; } catch (e) { return e.message; } }, old);
check(ld === 'ok', 'a labels file without viewedRanges / rejectedNearMisses still loads');

// --- autosave: prob, proposals source, viewed and rejected come back after a reload
await ev(() => flushSave());
await sleep(500);
const before = await ev(() => ({ v: JSON.stringify(state.viewedRanges), p: Array.from(state.prob[5].slice(690, 710)) }));
await page.reload({ waitUntil: 'networkidle0' });
await waitFor(() => state.trials.length === 20, 20000);
await sleep(500);
const after = await ev(() => ({ v: state.viewedRanges, p: Array.from(state.prob[5].slice(690, 710)), mode: state.prelabelerMode, hasProb: state.hasProb, rej: state.rejectedNearMisses.reduce((a, r) => a + r.length, 0),
  pre0: state.prelabels[0].map(p => [p.onset, p.offset, p.cls]), btn: !document.getElementById('nearMissBtn').classList.contains('hidden') }));
const vSame = JSON.parse(before.v).every((r, t) => r.every(([s, e]) => after.v[t].some(q => q[0] <= s && q[1] >= e)));
check(after.hasProb && after.btn && after.mode === 'file' && JSON.stringify(after.pre0) === JSON.stringify([[298, 321, 1]]) && after.rej === 1 && vSame &&
  after.p.every((v, i) => Math.abs(v - before.p[i]) < 2e-5), 'after a reload: prob (Uint16 copy, error < 2e-5), file proposals, rejected near misses and viewed ranges are back');

// --- optional: a real session
if (process.env.MOCOL_H5) {
  await ev(() => new Promise(r => { const q = indexedDB.deleteDatabase('uneye_db'); q.onsuccess = q.onerror = q.onblocked = () => r(); }));
  const inp = await page.$('#fileInput');
  const t0 = Date.now();
  await inp.uploadFile(process.env.MOCOL_H5);
  await waitFor(() => !document.getElementById('loadModal').classList.contains('hidden'));
  // wait for file inspection to populate #modalLabelsVar and #modalProbVar
  const inspected = await waitFor(() => document.getElementById('modalLabelsVar').value === 'pred_labels' && document.getElementById('modalProbVar').value === 'prob', 30000);
  const modalVars = await ev(() => ({
    summary: document.getElementById('modalFileSummary').textContent,
    labelsVar: document.getElementById('modalLabelsVar').value,
    labelsRole: document.getElementById('modalLabelsRole').value,
    probVar: document.getElementById('modalProbVar').value,
    labelOpts: Array.from(document.getElementById('modalLabelsVar').options).map(o => o.value),
    probOpts: Array.from(document.getElementById('modalProbVar').options).map(o => o.value),
  }));
  check(inspected && modalVars.labelsRole === 'prelabels' && modalVars.labelOpts.includes('pred_labels') && modalVars.probOpts.includes('prob'),
    `loadModal variable selectors populated from ${process.env.MOCOL_H5.split('/').pop()}: labels='${modalVars.labelsVar}' (${modalVars.labelsRole}), prob='${modalVars.probVar}', summary='${modalVars.summary}'`);
  await page.click('#modalConfirmBtn');
  const loaded = await waitFor(() => state.trials.length > 1000 && state.hasProb, 180000);
  const tl = Date.now() - t0;
  const r = await ev(() => { const t0 = performance.now(); enterNearMiss(); const t1 = performance.now(); return { n: state.nearMiss.queue.length, ms: Math.round(t1 - t0), top: state.nearMiss.queue.slice(0, 5).map(c => c.peak.toFixed(3)), badge: document.getElementById('eventCounterBadge').textContent, mode: state.prelabelerMode, trials: state.trials.length }; });
  check(loaded && r.n > 0, `session ${process.env.MOCOL_H5.split('/').pop()}: ${r.trials} trials loaded in ${tl} ms, proposals '${r.mode}', near-miss queue ${r.n} in ${r.ms} ms, top ${r.top.join(' ')}, ${r.badge}`);
  await page.screenshot({ path: SHOTS + '_session_nearmiss.png' });
  const tv = await ev(() => { const t0 = performance.now(); openViewedPeriods(); return new Promise(res => requestAnimationFrame(() => requestAnimationFrame(() => res(Math.round(performance.now() - t0))))); });
  await sleep(300);
  check(tv < 3000, `viewed periods of ${r.trials} trials drawn in ${tv} ms`);
  await ev(() => { document.getElementById('viewedScroll').scrollTop = 0; });
  await sleep(200);
  await page.screenshot({ path: SHOTS + '_session_viewed.png' });
  await page.click('#viewedCloseBtn');

  // Also test loading with pred_labels chosen as 'reviewed' ground-truth labels and prob disabled ('(none)')
  await inp.uploadFile(process.env.MOCOL_H5);
  await waitFor(() => !document.getElementById('loadModal').classList.contains('hidden') && document.getElementById('modalLabelsVar').value === 'pred_labels', 30000);
  await ev(() => {
    document.getElementById('modalLabelsVar').value = 'pred_labels';
    document.getElementById('modalLabelsRole').value = 'reviewed';
    document.getElementById('modalProbVar').value = '';
  });
  await page.click('#modalConfirmBtn');
  const loadedCustom = await waitFor(() => state.trials.length > 1000 && !state.hasProb && state.expertEvents[0].length > 0, 60000);
  const customSt = await ev(() => ({ hasProb: state.hasProb, ev0: state.expertEvents[0].length, rev0: state.reviewedRanges[0].length }));
  check(loadedCustom && !customSt.hasProb && customSt.ev0 > 0 && customSt.rev0 > 0,
    `custom variable selection (labels='pred_labels' as reviewed, prob='(none)'): hasProb=${customSt.hasProb}, trial 0 expertEvents=${customSt.ev0}, reviewedRanges=${customSt.rev0}`);
}

// --- test loading boolean variables as labels (L) via Load labels (both HDF5/NPZ boolean variables with variable picker and CSV true/false)
const boolRes = await ev(async () => {
  const nSamples = state.trials[0].x.length;
  const totalLen = state.trials.reduce((a, tr) => a + tr.x.length, 0);
  // Create an H5 file with two boolean-like variables (bool_sacc_a and bool_sacc_b) and no xy
  const h5 = window.h5wasm || await (async () => { const sc = document.createElement('script'); sc.src = 'web/h5wasm.js'; document.head.appendChild(sc); await new Promise(r => sc.onload = r); return window.h5wasm; })();
  const a = new Uint8Array(totalLen), b = new Uint8Array(totalLen);
  for (let i = 100; i <= 130; i++) a[i] = 1;
  for (let i = 500; i <= 540; i++) b[i] = 1;
  const bytes = await UnEyeFormats.buildH5([
    { name: 'bool_sacc_a', type: 'u1', shape: [totalLen], data: a },
    { name: 'bool_sacc_b', type: 'u1', shape: [totalLen], data: b },
  ], {}, h5);
  state.expertEvents[0] = [];
  state.reviewedRanges[0] = [];
  state.scanWins[0] = [];
  const p = window.loadPreviousLabels(new File([bytes], 'bool_vars.h5'));
  await new Promise(r => setTimeout(r, 200));
  const modalOpen = !document.getElementById('pickLabelsVarModal').classList.contains('hidden');
  const opts = Array.from(document.getElementById('pickLabelsVarSelect').options).map(o => o.value);
  document.getElementById('pickLabelsVarSelect').value = 'bool_sacc_b';
  document.getElementById('pickLabelsVarConfirmBtn').click();
  await p;
  const evAfterH5 = state.expertEvents[0].map(e => [e.onset, e.offset, e.cls]);

  // Also test CSV with true/false column via Load labels
  const csvLines = ['time_ms,is_saccade'];
  for (let i = 0; i < nSamples; i++) {
    csvLines.push(`${i},${(i >= 800 && i <= 835) ? 'true' : 'false'}`);
  }
  state.expertEvents[0] = [];
  await window.loadPreviousLabels(new File([csvLines.join('\n')], 'bool_labels.csv'), 'is_saccade');
  const evAfterCsv = state.expertEvents[0].map(e => [e.onset, e.offset, e.cls]);
  return { modalOpen, opts, evAfterH5, evAfterCsv };
});
check(boolRes.modalOpen && boolRes.opts.includes('bool_sacc_a') && boolRes.opts.includes('bool_sacc_b') &&
  JSON.stringify(boolRes.evAfterH5) === JSON.stringify([[500, 540, 1]]) &&
  JSON.stringify(boolRes.evAfterCsv) === JSON.stringify([[800, 835, 1]]),
  `boolean variables selectable for Load labels (L): picker opts=${JSON.stringify(boolRes.opts)}, H5=${JSON.stringify(boolRes.evAfterH5)}, CSV(true/false)=${JSON.stringify(boolRes.evAfterCsv)}`);

check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors.join(' | ') : ''));
await browser.close();
console.log(ok ? 'all near-miss flow checks passed' : 'SOME CHECKS FAILED');
process.exit(ok ? 0 : 1);
