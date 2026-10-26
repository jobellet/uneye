// Headless-Chrome test of the labeling flow: pinch refractory period, pass-1 windows (10 % overlap), reviewed = windows seen,
// pass 2 event navigation across trials, save / reload of the labels.
// Usage: node test_labeler_flow.mjs [url]   (needs puppeteer-core and CHROME=/path/to/chrome; serve the repo with `python3 -m http.server 8765`)
import puppeteer from 'puppeteer-core';
import fs from 'fs';
const URL = process.argv[2] || 'http://127.0.0.1:8765/index.html';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
// ANDROID_WS=ws://localhost:9333/devtools/browser runs the test inside the Android app's WebView
// (debug build; adb forward tcp:9333 localabstract:webview_devtools_remote_<pid>)
const ANDROID_WS = process.env.ANDROID_WS;
const browser = ANDROID_WS ? await puppeteer.connect({ browserWSEndpoint: ANDROID_WS, defaultViewport: null })
  : await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
const page = ANDROID_WS ? (await browser.pages()).find(p => p.url().includes('index.html')) : await browser.newPage();
const errors = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('dialog', async d => { await d.accept().catch(() => {}); });
if (!ANDROID_WS) await page.setViewport({ width: 1024, height: 620 });
await page.evaluateOnNewDocument(() => {
  Element.prototype.setPointerCapture = () => {}; Element.prototype.releasePointerCapture = () => {};
  const blobs = new Map(), orig = URL.createObjectURL.bind(URL);
  URL.createObjectURL = b => { const u = orig(b); blobs.set(u, b); return u; };
  window.__dl = [];
  HTMLAnchorElement.prototype.click = function () { if (!this.download) return; const name = this.download, b = blobs.get(this.href), fr = new FileReader(); fr.onload = () => window.__dl.push({ name, text: String(fr.result) }); fr.readAsText(b); };
});
await page.goto(ANDROID_WS ? 'https://appassets.androidplatform.net/assets/web/index.html' : URL, { waitUntil: ANDROID_WS ? 'load' : 'networkidle0', timeout: 120000 });
if (ANDROID_WS) {   // the app keeps its session in IndexedDB: start from an empty one, as the headless browser does
  await page.evaluate(() => new Promise(r => { try { localStorage.clear(); } catch (e) { } const q = indexedDB.deleteDatabase('uneye_db'); q.onsuccess = q.onerror = q.onblocked = () => r(); }));
  await page.reload({ waitUntil: 'load', timeout: 120000 });
  await new Promise(r => setTimeout(r, 4000));
  // native dialogs would block the page until tapped; files go through AndroidBridge.saveFile (Downloads): record them too
  await page.evaluate(() => {
    window.alert = m => console.log('alert: ' + m); window.confirm = () => true;
    const bridge = window.AndroidBridge;
    window.AndroidBridge = { saveFile: (name, b64, type) => { const bin = atob(b64), u8 = Uint8Array.from(bin, c => c.charCodeAt(0)); window.__dl.push({ name, text: new TextDecoder().decode(u8) }); bridge.saveFile(name, b64, type); }, showToast: m => bridge.showToast(m) };
  });
}
let ok = true; const check = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) ok = false; };
const ev = f => page.evaluate(f);
const sleep = ms => new Promise(r => setTimeout(r, ms));

// --- start: no simulated data, only a window to choose a file; the simulated traces of the tests are loaded explicitly
await sleep(500);
const start = await ev(() => ({ n: state.trials.length, modal: !document.getElementById('startModal').classList.contains('hidden'), demoBtn: !!document.getElementById('demoBtn') }));
check(start.n === 0 && start.modal && !start.demoBtn, `nothing is loaded at start, only the file window: ${JSON.stringify(start)}`);
await ev(() => generateDemoDataset());
// --- proposals come from the label-free HMM (hmm_prelabel.js)
const hmm = await ev(() => { const tr = state.trials[0]; const ref = UnEyeHMM.detect(tr.x, tr.y, tr.valid, state.rateHz).map(e => e.onset + '-' + e.offset).join();
  const got = state.prelabels[0].filter(p => p.cls === 1).map(p => p.onset + '-' + p.offset).join(); return [got === ref, state.prelabels[0].length, document.getElementById('startModal').classList.contains('hidden')]; });
check(hmm[0] && hmm[1] > 0 && hmm[2], `saccade proposals are the HMM's (${hmm[1]} proposals) and the start window is closed`);
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
const K = (await ev(() => document.getElementById('traceCanvas').clientWidth)) / 1024;   // touch positions below are for a 1024 px wide canvas
const tp = (type, id, x, y) => ptr(type, id, x * K, y);
await tp('pointerdown', 1, 300, 100); await tp('pointerdown', 2, 600, 100);
await tp('pointermove', 2, 700, 100); await tp('pointerup', 2, 700, 100); await tp('pointerup', 1, 300, 100);
check((await nMarks()) === base, 'releasing the fingers of a pinch makes no mark');
await sleep(100); await tp('pointerdown', 3, 400, 100); await tp('pointerup', 3, 400, 100);
check((await nMarks()) === base, 'a touch 100 ms after a pinch is ignored (refractory period)');
await sleep(800); await tp('pointerdown', 4, 400, 100); await tp('pointerup', 4, 400, 100);
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
// Switching class while eraser is active must automatically turn eraser off
check(await ev(() => state.eraser), 'eraser is on');
await page.click('#blinkBtn');
check(!(await ev(() => state.eraser)), 'clicking blinkBtn automatically deactivates eraser');
check(await ev(() => state.activeClass === 2), 'class switched to blink via blinkBtn');
check(await ev(() => document.getElementById('blinkBtn').classList.contains('active')), 'blinkBtn is active');
check(await ev(() => document.getElementById('saccadeBtn').classList.contains('inactive')), 'saccadeBtn is greyed out/inactive');

await page.click('#saccadeBtn');
check(await ev(() => state.activeClass === 1), 'class switched to saccade via saccadeBtn');
check(await ev(() => document.getElementById('saccadeBtn').classList.contains('active')), 'saccadeBtn is active');
check(await ev(() => document.getElementById('blinkBtn').classList.contains('inactive')), 'blinkBtn is greyed out/inactive');

await page.click('#classToggle');
check(await ev(() => state.activeClass === 2), 'class switched to blink via classToggle');
await page.click('#classToggle');
check(await ev(() => state.activeClass === 1), 'class switched back to saccade via classToggle');

// --- save / reload labels
await ev(() => { state.expertEvents[0] = [{ onset: 190, offset: 215, cls: 1, fromPrelabel: false }]; });
check(await ev(() => document.getElementById('toolsPanel').classList.contains('hidden')), 'the settings panel is hidden by default');
await page.click('#menuBtn');
check(!(await ev(() => document.getElementById('toolsPanel').classList.contains('hidden'))), 'the menu button shows the settings panel');
await page.click('#saveLabelsBtn'); await sleep(300);
const saved = JSON.parse(await ev(() => (window.__dl.find(d => /uneye_labels_.*\.json$/.test(d.name)) || { text: 'null' }).text));
check(saved && saved.format === 'uneye-labels-v1' && saved.trials[0].expertEvents.length >= 1, 'Save labels downloads a json file with the refined events and the windows seen');
check(saved && Array.isArray(saved.observedPeriods), 'saved labels JSON includes observedPeriods variable');

await page.click('#exportBtn'); await sleep(300);
const obsDl = await ev(() => window.__dl.find(d => d.name === 'observed.csv'));
check(obsDl && obsDl.text && obsDl.text.length > 0, 'exportDataset exports observed.csv alongside labels');
await page.click('#menuBtn');
await ev(() => { state.expertEvents[0] = []; state.scanWins[0] = []; });
fs.writeFileSync('/tmp/uneye_labels_test.json', JSON.stringify(saved));
if (ANDROID_WS) await page.evaluate(j => applyLabelsJson(JSON.parse(j)), JSON.stringify(saved));   // the Android file picker cannot be driven from here
else await (await page.$('#labelsInput')).uploadFile('/tmp/uneye_labels_test.json');
await sleep(500);
const back = await ev(() => [state.expertEvents[0].length, state.scanWins[0].length]);
check(back[0] >= 1 && back[1] >= 1, `Load labels restores events and windows seen: ${back}`);

// --- auto-scroll window sliding & prelabel ground truth acceptance on tap
await ev(() => {
  state.currentTrialIdx = 0;
  state.pass = 1;
  state.winMs = 500;
  state.fromSample = 0;
  state.toSample = 499;
  state.scanWins[0] = [];
  state.pointMarks[0] = [];
  state.expertEvents[0] = [];
  state.prelabels[0] = [{ onset: 200, offset: 235, cls: 1 }, { onset: 700, offset: 740, cls: 2 }];
  render();
});

// Auto-slide toggle check: must be disabled by default
const defaultAutoSlide = await ev(() => state.autoSlide);
check(defaultAutoSlide === false, 'auto-slide is disabled by default');

// Tap on proposed prelabel at sample 210 with autoSlide disabled:
// 1) Must accept full [200, 235] as ground truth (expertEvents), without adding point mark
// 2) Must NOT slide window because autoSlide is disabled by default (window stays 0..499)
await ev(() => handleTap(210));
const tapPreResult = await ev(() => ({
  expCount: state.expertEvents[0].length,
  expEvent: state.expertEvents[0][0],
  pmCount: state.pointMarks[0].length,
  from: state.fromSample,
  to: state.toSample,
}));
check(tapPreResult.expCount === 1 && tapPreResult.expEvent.onset === 200 && tapPreResult.expEvent.offset === 235 && tapPreResult.pmCount === 0,
  `tapping proposed event accepts full duration [200..235] as ground truth without adding point mark: ${JSON.stringify(tapPreResult.expEvent)}`);
check(tapPreResult.from === 0 && tapPreResult.to === 499,
  `window does not slide when autoSlide is disabled (remains 0..499): from=${tapPreResult.from}, to=${tapPreResult.to}`);

// Now open menu and toggle autoSlide ON via autoSlideBtn
await page.click('#menuBtn');
await page.click('#autoSlideBtn');
const enabledAutoSlide = await ev(() => state.autoSlide);
check(enabledAutoSlide === true, 'clicking autoSlideBtn enables autoSlide');
await page.click('#menuBtn');

// Tap on empty space at sample 450 with autoSlide enabled:
// 1) Adds a point mark at 450
// 2) Glides window so it starts 50 ms before sample (450 - 50 = 400) and ends at 400 + 500 - 1 = 899
await ev(() => handleTap(450));
const tapEmptyResult = await ev(() => ({
  pmCount: state.pointMarks[0].length,
  pmAt: state.pointMarks[0][0].at,
  from: state.fromSample,
  to: state.toSample,
}));
check(tapEmptyResult.pmCount === 1 && tapEmptyResult.pmAt === 450,
  `tapping empty space places point mark at 450: ${JSON.stringify(tapEmptyResult)}`);
check(tapEmptyResult.from === 400 && tapEmptyResult.to === 899,
  `with autoSlide enabled, window glides to start 50 ms before new mark (400..899): from=${tapEmptyResult.from}, to=${tapEmptyResult.to}`);

// Toggle autoSlide back OFF
await page.click('#menuBtn');
await page.click('#autoSlideBtn');
const disabledAgain = await ev(() => state.autoSlide);
check(disabledAgain === false, 'clicking autoSlideBtn again disables autoSlide');
await page.click('#menuBtn');

// --- Mobile phone viewport (360x740): hamburger menu and all controls must fit within screen width
if (!ANDROID_WS) await page.setViewport({ width: 360, height: 740 });   // on Android: the real phone screen
await sleep(100);
const mobileCheck = await ev(() => {
  const menuRect = document.getElementById('menuBtn').getBoundingClientRect();
  const topRect = document.getElementById('top-bar').getBoundingClientRect();
  const botRect = document.getElementById('bottom-bar').getBoundingClientRect();
  return {
    menuVisible: menuRect.width > 0 && menuRect.left >= 0 && menuRect.right <= innerWidth + 1,
    topFits: topRect.right <= innerWidth + 1 && topRect.height <= 50,
    botFits: botRect.right <= innerWidth + 1,
  };
});
check(mobileCheck.menuVisible && mobileCheck.topFits && mobileCheck.botFits,
  `on 360x740 phone screen, hamburger menuBtn and all bars fit inside screen dimensions: ${JSON.stringify(mobileCheck)}`);
if (!ANDROID_WS) await page.setViewport({ width: 1024, height: 620 });

// --- Overlapping events of the same class merge into one
const mergeCheck = await ev(() => {
  state.pass = 1;
  state.currentTrialIdx = 0;
  state.expertEvents[0] = [];
  state.pointMarks[0] = [];
  refineMark(0, 110, 100, 150, 1);
  refineMark(0, 140, 130, 180, 1); // overlaps [100, 150] with same class 1 -> must merge into [100, 180]
  refineMark(0, 260, 250, 290, 2); // blink class 2 -> stays separate
  refineMark(0, 280, 275, 320, 2); // overlaps [250, 290] with same class 2 -> must merge into [250, 320]
  return state.expertEvents[0];
});
check(
  mergeCheck.length === 2 &&
  mergeCheck[0].onset === 100 && mergeCheck[0].offset === 180 && mergeCheck[0].cls === 1 &&
  mergeCheck[1].onset === 250 && mergeCheck[1].offset === 320 && mergeCheck[1].cls === 2,
  `overlapping events of the same class merge into one: ${JSON.stringify(mergeCheck)}`
);

// --- Loading labels from a previous session and saving new labels appends/merges both
const appendSessionCheck = await ev(() => {
  // Simulate a clean dataset
  for (let t = 0; t < state.trials.length; t++) {
    state.scanWins[t] = [];
    state.reviewedRanges[t] = [];
    state.expertEvents[t] = [];
    state.pointMarks[t] = [];
  }
  // Load previous session labels (Trial 0 window 0..499 has saccade [100..130], Trial 1 window 0..499 has blink [200..260])
  const prevSession = {
    format: 'uneye-labels-v1',
    rateHz: 1000,
    winMs: 500,
    trialLengths: state.trials.map(tr => tr.x.length),
    trials: state.trials.map((_, t) => {
      if (t === 0) return { scanWins: [[0, 499]], expertEvents: [{ onset: 100, offset: 130, cls: 1 }], pointMarks: [] };
      if (t === 1) return { scanWins: [[0, 499]], expertEvents: [{ onset: 200, offset: 260, cls: 2 }], pointMarks: [] };
      return { scanWins: [], expertEvents: [], pointMarks: [] };
    })
  };
  applyLabelsJson(prevSession);

  // Now add new labels in the current session:
  // 1) In Trial 0 window 450..949, add saccade [600..640]
  state.currentTrialIdx = 0;
  state.pass = 1;
  state.winIdx = 1;
  applyGridWindow();
  refineMark(0, 610, 600, 640, 1);
  // 2) In Trial 2 window 0..499, add saccade [150..185]
  state.currentTrialIdx = 2;
  state.winIdx = 0;
  applyGridWindow();
  refineMark(2, 160, 150, 185, 1);

  window.__dl = [];
  saveLabelsJson();
  return {
    t0Events: state.expertEvents[0],
    t0Rev: state.reviewedRanges[0],
    t1Events: state.expertEvents[1],
    t1Rev: state.reviewedRanges[1],
    t2Events: state.expertEvents[2],
    t2Rev: state.reviewedRanges[2],
  };
});
await sleep(200);
const mergedSaved = JSON.parse(await ev(() => (window.__dl.find(d => /uneye_labels_.*\.json$/.test(d.name)) || { text: 'null' }).text));
check(
  mergedSaved &&
  mergedSaved.trials[0].expertEvents.length === 2 &&
  mergedSaved.trials[0].expertEvents[0].onset === 100 &&
  mergedSaved.trials[0].expertEvents[1].onset === 600 &&
  mergedSaved.trials[1].expertEvents.length === 1 &&
  mergedSaved.trials[1].expertEvents[0].onset === 200 &&
  mergedSaved.trials[2].expertEvents.length === 1 &&
  mergedSaved.trials[2].expertEvents[0].onset === 150,
  `saving after loading previous labels preserves both loaded and newly added labels across trials: T0=${JSON.stringify(mergedSaved?.trials[0]?.expertEvents)}, T1=${JSON.stringify(mergedSaved?.trials[1]?.expertEvents)}, T2=${JSON.stringify(mergedSaved?.trials[2]?.expertEvents)}`
);

// --- Pass 2 navigation across windows & trials (including expertEvents) and centered time zoom
await ev(() => {
  state.pass = 1;
  state.currentTrialIdx = 0;
  state.winIdx = 0;
  applyGridWindow();
});
await page.click('#switchPassBtn'); // enters Pass 2 on Trial 0 (which has events at [100..130] in win 0 and [600..640] in win 1, plus Trial 1 [200..260] and Trial 2 [150..185])
const p2Ev0 = await ev(() => ({
  trial: state.currentTrialIdx,
  idx: state.refineIdx,
  from: state.fromSample,
  to: state.toSample,
  on: state.currentPutativeOnset,
  off: state.currentPutativeOffset,
}));
check(
  p2Ev0.trial === 0 && p2Ev0.idx === 0 && p2Ev0.on === 100 && p2Ev0.off === 130 &&
  p2Ev0.from === 50 && p2Ev0.to === 180 &&
  Math.abs((p2Ev0.from + p2Ev0.to) / 2 - (p2Ev0.on + p2Ev0.off) / 2) <= 1,
  `Pass 2 Event 1 (Trial 0, Win 0) is zoomed in time [50..180] and centered on [100..130]: ${JSON.stringify(p2Ev0)}`
);

await page.click('#nextEventBtn'); // advances to second event of Trial 0 (in Window 1: [600..640])
const p2Ev1 = await ev(() => ({
  trial: state.currentTrialIdx,
  idx: state.refineIdx,
  from: state.fromSample,
  to: state.toSample,
  on: state.currentPutativeOnset,
  off: state.currentPutativeOffset,
}));
check(
  p2Ev1.trial === 0 && p2Ev1.idx === 1 && p2Ev1.on === 600 && p2Ev1.off === 640 &&
  p2Ev1.from === 550 && p2Ev1.to === 690 &&
  Math.abs((p2Ev1.from + p2Ev1.to) / 2 - (p2Ev1.on + p2Ev1.off) / 2) <= 1,
  `Pass 2 Next Event moves to another window of Trial 0 [600..640], zoomed [550..690] & centered: ${JSON.stringify(p2Ev1)}`
);

await page.click('#nextEventBtn'); // advances across trial boundary to Trial 1 [200..260]
const p2Ev2 = await ev(() => ({
  trial: state.currentTrialIdx,
  idx: state.refineIdx,
  from: state.fromSample,
  to: state.toSample,
  on: state.currentPutativeOnset,
  off: state.currentPutativeOffset,
}));
check(
  p2Ev2.trial === 1 && p2Ev2.idx === 0 && p2Ev2.on === 200 && p2Ev2.off === 260 &&
  p2Ev2.from === 150 && p2Ev2.to === 310 &&
  Math.abs((p2Ev2.from + p2Ev2.to) / 2 - (p2Ev2.on + p2Ev2.off) / 2) <= 1,
  `Pass 2 Next Event moves to Trial 1 [200..260], zoomed [150..310] & centered: ${JSON.stringify(p2Ev2)}`
);

await page.click('#prevEventBtn'); // goes back across trial boundary to Trial 0 Event 2 [600..640]
const p2Back = await ev(() => ({
  trial: state.currentTrialIdx,
  idx: state.refineIdx,
  from: state.fromSample,
  to: state.toSample,
  on: state.currentPutativeOnset,
  off: state.currentPutativeOffset,
}));
check(
  p2Back.trial === 0 && p2Back.idx === 1 && p2Back.on === 600 && p2Back.off === 640,
  `Pass 2 Prev Event returns to Trial 0 Event 2 [600..640]: ${JSON.stringify(p2Back)}`
);

// --- a pointerup whose releasePointerCapture throws (iPad Safari / Android WebView) must not leave the selection on screen
await ev(() => { if (state.pass !== 1) document.getElementById('switchPassBtn').click(); state.currentTrialIdx = 1; state.winIdx = 0; applyGridWindow(); state.eraser = false; render(); });
await ev(() => { window.__rel = Element.prototype.releasePointerCapture; Element.prototype.releasePointerCapture = () => { throw new DOMException('No active pointer', 'NotFoundError'); }; });
const evBefore = await ev(() => state.expertEvents[1].length);
await ptr('pointerdown', 41, 300, 150); await ptr('pointermove', 41, 360, 150); await ptr('pointermove', 41, 420, 150); await ptr('pointerup', 41, 420, 150);
const afterUp = await ev(() => ({ drag: state.isDraggingSpan, pointers: state.pointers.size, ev: state.expertEvents[1].length }));
check(!afterUp.drag && afterUp.pointers === 0 && afterUp.ev === evBefore + 1, `a drag whose releasePointerCapture throws still ends (event added, no selection left): ${JSON.stringify(afterUp)}`);
await ev(() => { Element.prototype.releasePointerCapture = window.__rel; });
// a pointerup that never arrives: the next window / trial drops the half-finished selection
await ptr('pointerdown', 42, 300, 150); await ptr('pointermove', 42, 400, 150);
await page.click('#nextTrialBtn');
const lost = await ev(() => ({ t: state.currentTrialIdx, drag: state.isDraggingSpan, pointers: state.pointers.size }));
check(lost.t === 2 && !lost.drag && lost.pointers === 0, `Next Trial drops a selection whose pointerup was lost: ${JSON.stringify(lost)}`);
await ptr('pointerdown', 43, 500, 150); await ptr('pointerup', 43, 500, 150);   // a new primary gesture works normally afterwards
// --- tapping a proposed event with the Saccade tool accepts it as a saccade, even when the model proposed a blink (red)
await sleep(700);
const tapCls = await ev(() => {
  state.currentTrialIdx = 3; state.winIdx = 0; applyGridWindow(); state.autoSlide = false;
  state.expertEvents[3] = []; state.pointMarks[3] = []; state.prelabels[3] = [{ onset: 300, offset: 330, cls: 2 }];
  document.getElementById('saccadeBtn').click();
  handleTap(315);
  return state.expertEvents[3].map(e => [e.onset, e.offset, e.cls]);
});
check(JSON.stringify(tapCls) === '[[300,330,1]]', `a red (blink) proposal tapped with the Saccade tool becomes a saccade: ${JSON.stringify(tapCls)}`);
// --- leaving pass 2 forgets the event that was being refined
const p1back = await ev(() => { document.getElementById('switchPassBtn').click(); const a = state.currentMarkAt; document.getElementById('switchPassBtn').click(); return [a >= 0, state.pass, state.currentMarkAt, state.currentPutativeOnset]; });
check(p1back[0] && p1back[1] === 1 && p1back[2] === -1 && p1back[3] === -1, `back in pass 1 there is no event under refinement: ${JSON.stringify(p1back)}`);

// --- boundary handles: the dot at mid-height on an onset / offset line moves that boundary only; in pass 2 the view stays on the event
const hx = (smp) => page.evaluate(smp => { const c = document.getElementById('traceCanvas'); return [c.clientWidth * (smp - state.fromSample) / (state.toSample - state.fromSample), c.clientHeight / 2]; }, smp);
await sleep(700);
await ev(() => { state.currentTrialIdx = 3; state.winIdx = 0; applyGridWindow(); state.eraser = false; state.scanWins[3] = [[0, 499]];
  state.expertEvents[3] = [{ onset: 300, offset: 330, cls: 1 }, { onset: 400, offset: 430, cls: 1 }]; state.pointMarks[3] = []; render(); });
let [x0, y0] = await hx(300), [x1] = await hx(290);
await ptr('pointerdown', 51, x0 + 3, y0 + 20); await ptr('pointermove', 51, (x0 + x1) / 2, y0); await ptr('pointermove', 51, x1, y0); await ptr('pointerup', 51, x1, y0);
const h1 = await ev(() => ({ evs: state.expertEvents[3].map(e => [e.onset, e.offset]), win: [state.fromSample, state.toSample], drag: state.isDraggingSpan, hd: state.handleDrag }));
check(JSON.stringify(h1.evs) === '[[290,330],[400,430]]' && h1.win.join() === '0,499' && !h1.drag && h1.hd === null,
  `pass 1: dragging the onset handle moves the onset only, no new event, no slide: ${JSON.stringify(h1)}`);
await ev(() => setPass(2));
const p2a = await ev(() => [state.currentTrialIdx, state.refineIdx, state.currentPutativeOnset, state.currentPutativeOffset, state.fromSample, state.toSample]);
[x0, y0] = await hx(330); [x1] = await hx(345);
await ptr('pointerdown', 52, x0, y0); await ptr('pointermove', 52, x1, y0); await ptr('pointerup', 52, x1, y0);
const p2b = await ev(() => [state.currentTrialIdx, state.refineIdx, state.currentPutativeOnset, state.currentPutativeOffset, state.fromSample, state.toSample]);
const p2e = await ev(() => state.expertEvents[3].map(e => [e.onset, e.offset]));
check(p2a[0] === 3 && p2a[2] === 290 && JSON.stringify(p2e) === '[[290,345],[400,430]]' && p2b[0] === 3 && p2b[1] === p2a[1] && p2b[2] === 290 && p2b[3] === 345 && p2b[4] === p2a[4] && p2b[5] === p2a[5],
  `pass 2: dragging the offset handle adjusts the offset and stays on the same event, same view: ${JSON.stringify({ p2a, p2b, p2e })}`);
await page.click('#undoBtn');
const p2u = await ev(() => state.expertEvents[3].map(e => [e.onset, e.offset]));
check(JSON.stringify(p2u) === '[[290,330],[400,430]]', `Undo restores the boundary moved with a handle: ${JSON.stringify(p2u)}`);

// --- Desktop keyboard shortcuts: ArrowLeft/ArrowRight navigation, "a"/"s" in Pass 2 to grab start/end, and customizable shortcuts modal
const kbNav1 = await ev(() => {
  setPass(1);
  state.currentTrialIdx = 3;
  state.winIdx = 0;
  applyGridWindow();
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
  const afterRight = [state.fromSample, state.toSample];
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
  const afterLeft = [state.fromSample, state.toSample];
  return { afterRight, afterLeft };
});
check(kbNav1.afterRight[0] === 450 && kbNav1.afterLeft[0] === 0,
  `ArrowRight / ArrowLeft navigate to next / previous window in Pass 1: ${JSON.stringify(kbNav1)}`);

const kbPass2Grab = await ev(() => {
  setPass(2);
  // Event 0 on trial 3 is [290, 330]; hover at sample 282 and press 'a' (grabOnset), then hover at 342 and press 's' (grabOffset)
  state.hoverSample = 282;
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', bubbles: true }));
  window.dispatchEvent(new KeyboardEvent('keyup', { key: 'a', bubbles: true }));
  const afterA = state.expertEvents[3][0] ? [state.expertEvents[3][0].onset, state.expertEvents[3][0].offset] : null;

  state.hoverSample = 342;
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true }));
  window.dispatchEvent(new KeyboardEvent('keyup', { key: 's', bubbles: true }));
  const afterS = state.expertEvents[3][0] ? [state.expertEvents[3][0].onset, state.expertEvents[3][0].offset] : null;
  state.hoverSample = -1;
  return { afterA, afterS };
});
check(
  JSON.stringify(kbPass2Grab.afterA) === '[282,330]' && JSON.stringify(kbPass2Grab.afterS) === '[282,342]',
  `Pass 2 "a" and "s" shortcuts grab start (onset) and end (offset) boundaries: ${JSON.stringify(kbPass2Grab)}`
);

// Test customizing shortcuts via the Shortcuts Modal (e.g. rebinding grabOnset to "q" for AZERTY)
await page.click('#shortcutsTopBtn');
const modalInfo = await ev(() => {
  const open = !document.getElementById('shortcutsModal').classList.contains('hidden');
  const rows = document.querySelectorAll('#shortcutsList .shortcut-row').length;
  // Click the grabOnset key button to enter recording mode, then press 'q'
  const btn = document.querySelector('#shortcutsList .shortcut-key-btn[data-action="grabOnset"]');
  if (btn) btn.click();
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'q', bubbles: true }));
  const newBinding = state.shortcuts.grabOnset;
  const saved = JSON.parse(localStorage.getItem('uneye_shortcuts') || '{}').grabOnset;
  closeShortcutsModal();
  // Verify 'q' now grabs the onset boundary in Pass 2
  state.hoverSample = 290;
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'q', bubbles: true }));
  window.dispatchEvent(new KeyboardEvent('keyup', { key: 'q', bubbles: true }));
  state.hoverSample = -1;
  const afterCustomQ = state.expertEvents[3][0] ? [state.expertEvents[3][0].onset, state.expertEvents[3][0].offset] : null;
  // Restore offset to 330 and reset shortcuts to default
  state.hoverSample = 330;
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true }));
  window.dispatchEvent(new KeyboardEvent('keyup', { key: 's', bubbles: true }));
  state.hoverSample = -1;
  applyShortcutPreset('qwerty');
  return { open, rows, newBinding, saved, afterCustomQ, resetBinding: state.shortcuts.grabOnset };
});
check(
  modalInfo.open && modalInfo.rows >= 20 && modalInfo.newBinding === 'q' && modalInfo.saved === 'q' &&
  JSON.stringify(modalInfo.afterCustomQ) === '[290,342]' && modalInfo.resetBinding === 'a',
  `Shortcuts modal lists ${modalInfo.rows} shortcuts and supports custom key rebinding ('a' -> 'q'): ${JSON.stringify(modalInfo)}`
);

await ev(() => setPass(1));
// --- no automatic training during pass 1, whatever the amount reviewed; it starts when a pass 2 is left
// (the pass-2 navigation above reached the last event, which ends a pass 2 and may have started a training: stop it first)
await ev(() => { const b = document.getElementById('cancelTrainBtn'); if (state.isTraining && b) b.click(); });
for (let k = 0; k < 40 && await ev(() => state.isTraining); k++) await sleep(250);
const auto1 = await ev(() => { setPass(1); state.currentTrialIdx = 0; state.winIdx = 0; applyGridWindow(); state.reviewedAtLastTraining = -30;   // > 20 s 'new' data: the former rule trained during pass 1 here
  for (let k = 0; k < 400; k++) nextEpoch(); return [state.isTraining, totalReviewed()]; function totalReviewed() { return state.reviewedRanges.flat().reduce((a, r) => a + r[1] - r[0] + 1, 0) / state.rateHz; } });
check(auto1[0] === false && auto1[1] > 2, `no training during pass 1 (${auto1[1].toFixed(1)} s reviewed, 30 s more than at the last training)`);
const auto2 = await ev(() => { setPass(2); setPass(1); return [state.isTraining, document.getElementById('trainStatus').textContent]; });
check(auto2[0] === true, `training starts when pass 2 is left: ${JSON.stringify(auto2)}`);
await ev(() => { const b = document.getElementById('cancelTrainBtn'); if (b) b.click(); });
await sleep(1500);
// --- network-suggestion toggle off: the HMM proposes even when a network exists
const tog = await ev(async () => { setNetSuggestions(false); await refreshSuggestions(); const tr = state.trials[1];
  return [document.getElementById('netSuggestBtn').textContent, state.prelabels[1].filter(p => p.cls === 1).map(p => p.onset).join() === UnEyeHMM.detect(tr.x, tr.y, tr.valid, state.rateHz).map(e => e.onset).join()]; });
check(tog[0].endsWith('Off') && tog[1], `toggle off -> HMM proposals: ${JSON.stringify(tog)}`);
await ev(() => setNetSuggestions(true));

// --- the session is saved (debounced) and restored on reload; the traces sit in their own key, the labels are saved without them
await ev(() => { setPass(1); state.expertEvents[3] = [{ onset: 290, offset: 345, cls: 1 }]; recomputeReviewed(3, true); });
await sleep(1500);
const idb = await page.evaluate(() => new Promise(res => { const q = indexedDB.open('uneye_db', 1); q.onsuccess = () => { const st = q.result.transaction('uneye_store').objectStore('uneye_store');
  const a = st.get('autosave'), b = st.get('autosave_trials'); b.onsuccess = () => res({ trialsInAutosave: 'trials' in a.result, n: b.result.length, typed: b.result[0].x instanceof Float32Array }); }; }));
check(!idb.trialsInAutosave && idb.n === (await ev(() => state.trials.length)) && idb.typed, `autosave: labels without the traces, traces as typed arrays in their own key: ${JSON.stringify(idb)}`);
await page.reload({ waitUntil: ANDROID_WS ? 'load' : 'networkidle0' }); await sleep(ANDROID_WS ? 5000 : 1500);
const rest = await ev(() => ({ n: state.trials.length, ev3: state.expertEvents[3].map(e => [e.onset, e.offset]) }));
check(rest.n > 0 && JSON.stringify(rest.ev3) === '[[290,345]]', `the session comes back after a reload: ${JSON.stringify(rest)}`);
// labeling in a restored session: the release used to throw (no undo history) and leave the selection on screen without the event
await ev(() => { state.currentTrialIdx = 6; state.winIdx = 0; applyGridWindow(); state.expertEvents[6] = []; state.prelabels[6] = []; state.pointMarks[6] = []; render(); });
await ptr('pointerdown', 61, 100, 150); await ptr('pointermove', 61, 160, 150); await ptr('pointermove', 61, 220, 150); await ptr('pointerup', 61, 220, 150);
const rl = await ev(() => ({ n: state.expertEvents[6].length, drag: state.isDraggingSpan }));
check(rl.n === 1 && !rl.drag, `a drag in a session restored after a reload stores the event and ends the selection: ${JSON.stringify(rl)}`);

// --- a session saved by an earlier version (model without modelVersion, every proposal a blink) comes back without that model
await page.evaluate(() => new Promise(res => {
  const q = indexedDB.open('uneye_db', 1);
  q.onsuccess = () => {
    const db = q.result, tx = db.transaction('uneye_store', 'readwrite');
    tx.objectStore('uneye_store').put({
      trials: state.trials.map(tr => ({ x: Array.from(tr.x), y: Array.from(tr.y), name: tr.name })), labels: state.labels.map(l => Array.from(l)),
      reviewedRanges: state.reviewedRanges, scanWins: state.scanWins, expertEvents: state.expertEvents, pointMarks: state.pointMarks,
      prelabels: state.prelabels.map(P => P.map(p => ({ onset: p.onset, offset: p.offset, cls: 2 }))), rateHz: state.rateHz, convergenceHistory: [{ testF1Saccade: 0 }],
      trainedModelBin: new Uint8Array([85, 78, 69, 89, 1, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0]), trainedModelJson: '{}'
    }, 'autosave');
    tx.oncomplete = () => res();
  };
}));
await page.reload({ waitUntil: ANDROID_WS ? 'load' : 'networkidle0' }); await sleep(ANDROID_WS ? 5000 : 1500);
const stale = await ev(() => ({ model: state.trainedModelBin, hist: state.convergenceHistory.length, cls: [...new Set(state.prelabels.flat().map(p => p.cls))], n: state.prelabels.flat().length, build: document.getElementById('appBuild').textContent }));
check(stale.model === null && stale.hist === 0 && stale.cls.includes(1),   // the rule-based detector proposes saccades again (and blinks on NaN stretches)
  `a model saved by an earlier version is dropped on restore, with its (red) proposals: ${JSON.stringify(stale)}`);

check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors.join(' | ') : ''));
await page.screenshot({ path: process.env.SHOT || '/tmp/labeler.png' });
if (ANDROID_WS) await browser.disconnect(); else await browser.close();
console.log(ok ? 'all labeler flow checks passed' : 'SOME CHECKS FAILED');
process.exit(ok ? 0 : 1);
