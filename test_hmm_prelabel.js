// Parity of the labelers' HMM proposals (hmm_prelabel.js) with the C++ batch detector "hmm_accel" (cpp/app/include/eyelive/dsp.hpp, the default options of
// cpp/app/detect_hmm/src/hmm_detector.cpp::detect_all): same events on simulated traces (fixation noise, drift, micro- and larger saccades, NaN gaps) at
// 1000 and 500 Hz. Needs a C++17 compiler (CXX, default c++). Usage: node test_hmm_prelabel.js
const fs = require('fs'), os = require('os'), path = require('path'), { execFileSync } = require('child_process');
const HMM = require('./hmm_prelabel.js');

let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 4294967296; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

function simulate(T, fs, withNaN) {
    const x = new Float64Array(T), y = new Float64Array(T); let px = 0, py = 0, t = 0;
    while (t < T) {
        const fix = Math.round((0.15 + 0.6 * rnd()) * fs);
        for (let k = 0; k < fix && t < T; k++, t++) { px += 0.002 * gauss(); py += 0.002 * gauss(); x[t] = px + 0.01 * gauss(); y[t] = py + 0.01 * gauss(); }
        const amp = rnd() < 0.7 ? 0.1 + 0.8 * rnd() : 2 + 8 * rnd(), ang = 2 * Math.PI * rnd(), dur = Math.max(3, Math.round((0.012 + 0.0022 * amp) * fs));
        const x0 = px, y0 = py;
        for (let k = 0; k < dur && t < T; k++, t++) { const s = 0.5 - 0.5 * Math.cos(Math.PI * (k + 1) / dur); px = x0 + amp * s * Math.cos(ang); py = y0 + amp * s * Math.sin(ang); x[t] = px + 0.01 * gauss(); y[t] = py + 0.01 * gauss(); }
    }
    if (withNaN) { const a = Math.round(T * 0.4), b = a + Math.round(0.08 * fs); for (let i = a; i < b && i < T; i++) { x[i] = NaN; y[i] = NaN; } }
    return { x, y };
}

const cpp = `#include "eyelive/dsp.hpp"
#include <cstdio>
#include <cmath>
using namespace eyelive;
int main(int argc, char** argv) {
  double fs = atof(argv[2]); FILE* f = fopen(argv[1], "rb"); long T; fread(&T, 8, 1, f); std::vector<double> x(T), y(T); fread(x.data(), 8, T, f); fread(y.data(), 8, T, f); fclose(f);
  dsp::Mask valid(T, 1); for (long i = 0; i < T; ++i) valid[i] = std::isfinite(x[i]) && std::isfinite(y[i]);
  dsp::HmmParams h; if (std::fabs(fs - 1000.0) < 50.0) h = {fs, 6, 0.416573763, 1.32039225, 0.201105669, 0.766239583, 0.00554218027, 0.949469511};
  else h = {fs, 3, 0.406975567, 1.3010565, 0.20460619, 0.573560297, 0.00969478964, 0.928820044};
  dsp::Kin k; dsp::score_trace(x, y, valid, fs, k); dsp::Events ev = dsp::runs_of(dsp::hmm_detect(k, h));
  dsp::Refine r; dsp::Events fin = dsp::runs_of(dsp::refine_events(ev, k.fx, k.fy, valid, fs, r));
  for (auto& e : fin) printf("%d %d\\n", e.first, e.second); return 0; }
`;
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hmmpar-'));
fs.writeFileSync(path.join(dir, 'p.cpp'), cpp);
const exe = path.join(dir, 'p');
execFileSync(process.env.CXX || 'c++', ['-O2', '-std=c++17', '-I', path.join(__dirname, 'cpp/app/include'), path.join(dir, 'p.cpp'), '-o', exe]);

let ok = true, nEv = 0;
for (const [fsHz, T, withNaN] of [[1000, 6000, false], [1000, 6000, true], [1000, 2500, false], [500, 4000, true]]) {
    for (let rep = 0; rep < 3; rep++) {
        const { x, y } = simulate(T, fsHz, withNaN);
        const buf = Buffer.alloc(8 + 16 * T); buf.writeBigInt64LE(BigInt(T), 0);
        for (let i = 0; i < T; i++) { buf.writeDoubleLE(x[i], 8 + 8 * i); buf.writeDoubleLE(y[i], 8 + 8 * (T + i)); }
        const bin = path.join(dir, 't.bin'); fs.writeFileSync(bin, buf);
        const ref = execFileSync(exe, [bin, String(fsHz)]).toString().trim().split('\n').filter(Boolean).map(l => l.split(' ').map(Number));
        const js = HMM.detect(x, y, null, fsHz).map(e => [e.onset, e.offset]);
        const same = JSON.stringify(ref) === JSON.stringify(js); nEv += ref.length;
        console.log((same ? 'ok   ' : 'FAIL ') + `${fsHz} Hz, ${T} samples${withNaN ? ', NaN gap' : ''}, rep ${rep}: C++ ${ref.length} events, JS ${js.length} events`);
        if (!same) { ok = false; console.log('  C++', JSON.stringify(ref.slice(0, 8)), '\n  JS ', JSON.stringify(js.slice(0, 8))); }
    }
}
fs.rmSync(dir, { recursive: true, force: true });
console.log(ok ? `HMM proposals identical to the C++ detector (${nEv} events)` : 'SOME CHECKS FAILED');
process.exit(ok ? 0 : 1);
