// Headless-Chrome test of the labeling flow: pinch refractory period, pass-1 windows (10 % overlap), reviewed = windows seen,
// pass 2 event navigation across trials, save / reload of the labels.
// Usage: node test_labeler_flow.mjs [url]   (needs puppeteer-core and CHROME=/path/to/chrome; serve the repo with `python3 -m http.server 8765`)
import puppeteer from 'puppeteer-core';
import fs from 'fs';
const URL = process.argv[2] || 'http://127.0.0.1:8765/index.html';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('dialog', async d => { await d.accept(); });
await page.setViewport({ width: 1180, height: 820 });
await page.evaluateOnNewDocument(() => {
  Element.prototype.setPointerCapture = () => {}; Element.prototype.releasePointerCapture = () => {};
  const blobs = new Map(), orig = URL.createObjectURL.bind(URL);
  URL.createObjectURL = b => { const u = orig(b); blobs.set(u, b); return u; };
  window.__dl = [];
  HTMLAnchorElement.prototype.click = function () { if (!this.download) return; const name = this.download, b = blobs.get(this.href), fr = new FileReader(); fr.onload = () => window.__dl.push({ name, text: String(fr.result) }); fr.readAsText(b); };
});
await page.goto(URL, { waitUntil: 'networkidle0' });
let ok = true; const check = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) ok = false; };
const ev = f => page.evaluate(f);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const ptr = (type, id, x, y) => page.evaluate((type, id, x, y) => {
  const c = document.getElementById('traceCanvas'), r = c.getBoundingClientRect();
  c.dispatchEvent(new PointerEvent(type, { pointerId: id, clientX: r.left + x, clientY: r.top + y, bubbles: true, pointerType: 'touch' }));
}, type, id, x, y);
const nMarks = () => ev(() => state.pointMarks[0].length);

// --- windows
const g = await ev(() => [state.trials[0].x.length, state.fromSample, state.toSample]);
check(g[1] === 0 && g[2] === 499, `first window is 500 samples: ${g[1]}..${g[2]} (trial of ${g[0]} samples)`);
await page.click('#nextEpochBtn');
const w2 = await ev(() => [state.fromSample, state.toSample]);
check(w2[0] === 450 && w2[1] === 949, `Next window starts 10 % before the end of the previous one: ${w2[0]}..${w2[1]}`);
check((await ev(() => state.scanWins[0].length)) === 1, 'the window left with Next is recorded as seen');

// --- pinch refractory period
await page.click('#prevEpochBtn');
const base = await nMarks();
await ptr('pointerdown', 1, 300, 100); await ptr('pointerdown', 2, 600, 100);
await ptr('pointermove', 2, 700, 100); await ptr('pointerup', 2, 700, 100); await ptr('pointerup', 1, 300, 100);
check((await nMarks()) === base, 'releasing the fingers of a pinch makes no mark');
await sleep(100); await ptr('pointerdown', 3, 400, 100); await ptr('pointerup', 3, 400, 100);
check((await nMarks()) === base, 'a touch 100 ms after a pinch is ignored (refractory period)');
await sleep(800); await ptr('pointerdown', 4, 400, 100); await ptr('pointerup', 4, 400, 100);
check((await nMarks()) === base + 1, 'a normal tap after the refractory period makes a mark');

// --- reviewed = seen windows without a pending mark
await ev(() => { state.currentTrialIdx = 0; state.winIdx = 0; applyGridWindow(); state.scanWins[0] = []; render(); });
await page.click('#nextEpochBtn');                                   // window 0 (holds the mark near sample 400) is now seen
let rev = await ev(() => JSON.stringify(state.reviewedRanges[0]));
check(rev === '[]', `a seen window that still has a pending mark is not reviewed (${rev})`);
await ev(() => { state.pointMarks[0] = []; });
await page.click('#nextEpochBtn');                                   // window 1 seen, no mark
rev = await ev(() => JSON.stringify(state.reviewedRanges[0]));
check(rev === '[[0,949]]', `once the mark is gone, the two seen windows (10 % overlap) merge into one reviewed range (${rev})`);
const tl = await ev(() => { const h = humanLabels(0); return h.reduce((a, b) => a + b, 0); });
check(tl === 0, 'training labels hold only the refined events (pre-labels do not count)');

// --- pass 2: events across trials
await ev(() => {
  for (let t = 0; t < 3; t++) { state.expertEvents[t] = []; state.scanWins[t] = [[0, 999]]; state.pointMarks[t] = []; }
  state.pointMarks[0] = [{ at: 200, cls: 1 }, { at: 600, cls: 1 }]; state.pointMarks[2] = [{ at: 300, cls: 1 }];
  state.currentTrialIdx = 0; state.winIdx = 0; applyGridWindow();
});
await page.click('#switchPassBtn');
let s = await ev(() => [state.currentTrialIdx, state.refineIdx, state.refineQueue.length]);
check(s[0] === 0 && s[1] === 0 && s[2] === 2, `pass 2 starts at the first event: ${s}`);
check(await ev(() => document.getElementById('trialNavGroup').classList.contains('hidden')), 'no Next Trial button in pass 2');
await page.click('#nextEventBtn'); await page.click('#nextEventBtn');
s = await ev(() => [state.currentTrialIdx, state.refineIdx]);
check(s[0] === 2 && s[1] === 0, `Next event after the last event of a trial goes to the next trial with events (trial 3, skipping trial 2): ${s}`);
await page.click('#prevEventBtn');
s = await ev(() => [state.currentTrialIdx, state.refineIdx]);
check(s[0] === 0 && s[1] === 1, `Prev event goes back to the last event of the previous trial: ${s}`);
await ev(() => { state.scanWins[2] = []; });
await page.click('#nextEventBtn');
check((await ev(() => state.currentTrialIdx)) === 0, 'a mark in a window that was never displayed is not offered in pass 2');

// refining the last event of a trial moves on to the next trial by itself
await ev(() => { state.scanWins[2] = [[0, 999]]; state.refineIdx = 1; focusCurrentMark && 0; });
await page.click('#acceptPreBtn');
s = await ev(() => [state.currentTrialIdx, state.refineIdx]);
check(s[0] === 2, `after the last event of a trial the next one is in the next trial: ${s}`);

// --- eraser: one drag deletes every event, mark and pre-label between two times
await ev(() => { state.pass === 2 && document.getElementById('switchPassBtn').click(); state.currentTrialIdx = 0; state.winIdx = 0; applyGridWindow();
  state.expertEvents[0] = [{ onset: 100, offset: 120, cls: 1 }, { onset: 300, offset: 330, cls: 2 }, { onset: 800, offset: 820, cls: 1 }];
  state.pointMarks[0] = [{ at: 200, cls: 1 }]; state.prelabels[0] = [{ onset: 250, offset: 262, cls: 1 }, { onset: 900, offset: 910, cls: 1 }]; render(); });
await page.click('#eraserBtn');
const px = await ev(() => { const w = document.getElementById('traceCanvas').clientWidth; return [w * 90 / 500, w * 400 / 500]; });   // samples 90..400 of the 0..499 window
await ptr('pointerdown', 9, px[0], 100); await ptr('pointermove', 9, px[1], 100); await ptr('pointerup', 9, px[1], 100);
const er = await ev(() => [state.expertEvents[0].map(e => e.onset).join(), state.pointMarks[0].length, state.prelabels[0].map(p => p.onset).join()]);
check(er[0] === '800' && er[1] === 0 && er[2] === '900', `eraser removed saccade, blink, mark and pre-label between the two times, kept the rest: ${er}`);
await page.click('#eraserBtn');

// --- save / reload labels
await ev(() => { state.expertEvents[0] = [{ onset: 190, offset: 215, cls: 1, fromPrelabel: false }]; });
await page.click('#saveLabelsBtn'); await sleep(300);
const saved = JSON.parse(await ev(() => (window.__dl.find(d => /uneye_labels_.*\.json$/.test(d.name)) || { text: 'null' }).text));
check(saved && saved.format === 'uneye-labels-v1' && saved.trials[0].expertEvents.length >= 1, 'Save labels downloads a json file with the refined events and the windows seen');
await ev(() => { state.expertEvents[0] = []; state.scanWins[0] = []; });
fs.writeFileSync('/tmp/uneye_labels_test.json', JSON.stringify(saved));
await (await page.$('#labelsInput')).uploadFile('/tmp/uneye_labels_test.json'); await sleep(500);
const back = await ev(() => [state.expertEvents[0].length, state.scanWins[0].length]);
check(back[0] >= 1 && back[1] >= 1, `Load labels restores events and windows seen: ${back}`);

check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors.join(' | ') : ''));
await page.screenshot({ path: process.env.SHOT || '/tmp/labeler.png' });
await browser.close();
console.log(ok ? 'all labeler flow checks passed' : 'SOME CHECKS FAILED');
process.exit(ok ? 0 : 1);
