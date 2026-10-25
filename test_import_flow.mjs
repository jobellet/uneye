// Headless-Chrome test of the column explorer (import.js + import_ui.js) on trimmed real files from public datasets (test_data/import):
// plain files keep the old reader; files with a header / ARFF / EyeLink .asc open the explorer with X, Y and time proposed; the choice
// can be changed; loading gives the trials at the measured rate.
// Usage: node test_import_flow.mjs [url]   (needs puppeteer-core and CHROME=/path/to/chrome; serve the repo with `python3 -m http.server 8765`)
import puppeteer from 'puppeteer-core';
import path from 'path';
import { fileURLToPath } from 'url';
const URL = process.argv[2] || 'http://127.0.0.1:8765/index.html';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const here = path.dirname(fileURLToPath(import.meta.url));
const fx = (f) => path.join(here, 'test_data', 'import', f);
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
const errors = [], alerts = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('dialog', async d => { alerts.push(d.message()); await d.accept().catch(() => {}); });
await page.setViewport({ width: 1100, height: 760 });
await page.goto(URL, { waitUntil: 'networkidle0', timeout: 120000 });
let ok = true; const check = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) ok = false; };
const ev = f => page.evaluate(f);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const hidden = (id) => page.evaluate((id) => document.getElementById(id).classList.contains('hidden'), id);
const val = (id) => page.evaluate((id) => document.getElementById(id).value, id);
const optText = (id) => page.evaluate((id) => { const s = document.getElementById(id); return s.options[s.selectedIndex].textContent; }, id);

async function open(file, { always } = {}) {
  await page.evaluate(() => { document.getElementById('loadModal').classList.add('hidden'); });
  const input = await page.$('#fileInput');
  await input.uploadFile(fx(file));
  await page.waitForFunction(() => !document.getElementById('loadModal').classList.contains('hidden'), { timeout: 10000 });
  if (always) await page.evaluate(() => { document.getElementById('alwaysExplorer').checked = true; });
  await page.click('#modalConfirmBtn');
}
const waitExplorer = () => page.waitForFunction(() => { const m = document.getElementById('importModal'); return !m.classList.contains('hidden') && !document.getElementById('impBody').classList.contains('hidden'); }, { timeout: 20000 });

// --- GazeBaseVR: header, 3-D vectors for both eyes; "n" is a time in ms at 250 Hz
await open('gazebasevr.csv');
await waitExplorer();
check(await optText('impX') === 'x' && await optText('impY') === 'y' && await optText('impT') === 'n', `GazeBaseVR: proposed X=${await optText('impX')}, Y=${await optText('impY')}, time=${await optText('impT')}`);
check(await val('impUnit') === 'ms' && await val('impRate') === '250', `GazeBaseVR: unit ${await val('impUnit')}, rate ${await val('impRate')} Hz`);
await sleep(300);
const nCols = await ev(() => document.querySelectorAll('#impColList .import-col').length);
check(nCols >= 18, `the variable list shows all ${nCols} variables (with the mean-of-eyes columns)`);
const painted = await ev(() => { const c = document.getElementById('impPreview'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n; });
check(painted > 500, `the preview is drawn (${painted} painted pixels)`);
check((await ev(() => document.getElementById('impResult').textContent)).startsWith('→ 1 trial'), 'summary: ' + await ev(() => document.getElementById('impResult').textContent));
// the units check reacts: 1 unit = 100 degrees is far too fast
await page.$eval('#impDeg', el => { el.value = '100'; el.dispatchEvent(new Event('input', { bubbles: true })); });
await sleep(300);
check(await ev(() => document.getElementById('impResult').classList.contains('bad')), 'a wrong "degrees per unit" is flagged: ' + await ev(() => document.getElementById('impResult').textContent));
await page.$eval('#impDeg', el => { el.value = '1'; el.dispatchEvent(new Event('input', { bubbles: true })); });
// choose the left eye instead
await page.select('#impX', await ev(() => String([...document.getElementById('impX').options].find(o => o.textContent === 'lx').value)));
await page.select('#impY', await ev(() => String([...document.getElementById('impY').options].find(o => o.textContent === 'ly').value)));
await page.click('#impLoadBtn');
await page.waitForFunction(() => document.getElementById('importModal').classList.contains('hidden'), { timeout: 20000 });
await sleep(300);
let st = await ev(() => ({ n: state.trials.length, len: state.trials[0].x.length, rate: state.rateHz, x0: state.trials[0].x[0] }));
check(st.n === 1 && st.len === 2999 && st.rate === 250, `loaded: ${JSON.stringify(st)}`);
check(Math.abs(st.x0 - 0.3396) < 1e-3, 'the left-eye column (lx = 0.3396) was loaded, not the cyclopean x');

// --- EyeLink asc: 3 blocks = 3 trials at 2000 Hz, blinks invalid
await open('emtec.asc');
await waitExplorer();
check(await optText('impSplit') === 'New trial when a variable changes' && await optText('impSplitCol') === 'block', 'asc: trials split by recording block');
check((await ev(() => document.getElementById('impSummary').textContent)).includes('EyeLink') && await val('impRate') === '2000', 'asc: ' + await ev(() => document.getElementById('impSummary').textContent));
await page.click('#impLoadBtn');
await page.waitForFunction(() => document.getElementById('importModal').classList.contains('hidden'), { timeout: 20000 });
await sleep(300);
st = await ev(() => ({ n: state.trials.length, lens: state.trials.map(t => t.x.length), rate: state.rateHz, inval: state.trials[0].valid.reduce((a, v) => a + (v ? 0 : 1), 0), names: state.trialNames }));
check(st.n === 3 && st.lens.every(l => l === 1500) && st.rate === 2000 && st.inval > 0, `asc loaded: ${JSON.stringify(st)}`);

// --- ARFF with irregular time stamps: resampled onto the 250 Hz grid
await open('gazecom.arff');
await waitExplorer();
check(await ev(() => document.getElementById('impResample').checked) && await val('impUnit') === 'us' && await val('impRate') === '250', 'ARFF: microseconds, 250 Hz, irregular -> resample ticked');
check(await optText('impVCol') === 'confidence', 'ARFF: confidence proposed as the validity variable');
await page.click('#impLoadBtn');
await page.waitForFunction(() => document.getElementById('importModal').classList.contains('hidden'), { timeout: 20000 });
await sleep(300);
st = await ev(() => ({ n: state.trials.length, len: state.trials[0].x.length, rate: state.rateHz }));
check(st.n === 1 && st.len === 1203 && st.rate === 250, `ARFF loaded: ${JSON.stringify(st)}`);

// --- GazeBase: the rate is not in the file (time is a sample index): the dialog's rate is kept
await page.evaluate(() => { document.getElementById('rateInput').value = '1000'; });
await open('gazebase_hss.csv');
await waitExplorer();
check(await val('impUnit') === 'samples' && await val('impRate') === '1000' && await optText('impVCol') === 'val', 'GazeBase: sample index -> rate from the load dialog (1000 Hz), validity = val');
// cancel returns to the load dialog
await page.click('#impCancelBtn');
await sleep(200);
check(await hidden('importModal') && !(await hidden('loadModal')), 'Cancel in the explorer goes back to the load dialog');
await page.click('#modalCancelBtn');

// --- plain two-column file: old reader, no explorer
await open('../sample_two_column_data.txt');
await sleep(800);
st = await ev(() => ({ n: state.trials.length, explorerHidden: document.getElementById('importModal').classList.contains('hidden') }));
check(st.explorerHidden && st.n === 1, `a plain two-column file does not open the explorer: ${JSON.stringify(st)}`);

check(errors.length === 0, 'no page error' + (errors.length ? ': ' + errors.join(' | ') : ''));
check(alerts.length === 0, 'no alert' + (alerts.length ? ': ' + alerts.join(' | ') : ''));
await browser.close();
console.log(ok ? 'all import flow checks passed' : 'SOME CHECKS FAILED');
process.exit(ok ? 0 : 1);
