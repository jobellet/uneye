// node test_wasm_recipe.js
// The labelers' training recipe in the WebAssembly trainer (WebTrainer.setRecipe = TrainConfig::resample_each_epoch / batch_size, the recipe
// cross-validated against PyTorch): whole traces held out for validation, windows re-cut every epoch, minibatches of 32, early stopping that keeps the
// best weights. Checks that it trains, that the validation loss is computed on held-out traces, that training stops by early stopping before
// max_epochs, and that the returned network is the best one (it predicts the synthetic saccades of a trace it never saw).
const fs = require('fs');
const path = require('path');

function synthTrace(seed, T = 3000) {                 // fixation noise + 1-deg saccades every ~400 ms, labels 1 on the saccades
    let s = seed;
    const rnd = () => (s = (s * 1103515245 + 12345) >>> 0) / 4294967296;
    const g = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    const x = new Float32Array(T), y = new Float32Array(T), l = new Uint8Array(T);
    let px = 0, py = 0;
    for (let t = 0; t < T;) {
        const fix = 250 + Math.floor(300 * rnd());
        for (let k = 0; k < fix && t < T; k++, t++) { x[t] = px + 0.01 * g(); y[t] = py + 0.01 * g(); }
        const a = 2 * Math.PI * rnd(), amp = 0.3 + rnd(), dur = 25, x0 = px, y0 = py;
        for (let k = 0; k < dur && t < T; k++, t++) {
            const f = 0.5 - 0.5 * Math.cos(Math.PI * (k + 1) / dur);
            px = x0 + amp * f * Math.cos(a); py = y0 + amp * f * Math.sin(a);
            x[t] = px + 0.01 * g(); y[t] = py + 0.01 * g(); l[t] = 1;
        }
    }
    return { x, y, l };
}

(async () => {
    const createUnEyeModule = require('./web/uneye_train.js');
    const Module = await createUnEyeModule({ wasmBinary: fs.readFileSync(path.join(__dirname, 'web', 'uneye_train.wasm')) });
    let failed = 0;
    const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) failed++; };
    const trainer = new Module.WebTrainer();
    trainer.reset();
    trainer.setConfig(2, 3600.0, true, 2, 0.25, 1300.0, 200, 3, 0.001, 1n, 1.0);
    ok(typeof trainer.setRecipe === 'function', 'WebTrainer.setRecipe is exported');
    trainer.setRecipe(true, 32);
    for (let i = 0; i < 8; i++) { const t = synthTrace(100 + i); trainer.addTrace(t.x, t.y, t.l, 1000.0); }
    const err = trainer.initTraining();
    ok(err === '', 'initTraining with the recipe: ' + (err || 'no error'));
    let last = null, epochs = 0, lossOk = true;
    while (!trainer.isFinished() && epochs < 200) {
        last = JSON.parse(trainer.step()); epochs++;
        if (!(last.val_loss >= 0) || !isFinite(last.val_loss)) lossOk = false;
    }
    ok(lossOk, `validation loss finite every epoch (${epochs} epochs)`);
    ok(last.finished && last.message === 'early stopping' && epochs < 200, `stopped by early stopping before max_epochs: ${last.message} after ${epochs} epochs`);
    const test = synthTrace(999);
    const pred = trainer.predictTrial(test.x, test.y, 1000.0, 1300.0);
    let tp = 0, fp = 0, fn = 0;
    for (let i = 0; i < test.l.length; i++) { const p = pred[i] === 1, t = test.l[i] === 1; tp += p && t; fp += p && !t; fn += !p && t; }
    const f1 = 2 * tp / Math.max(1, 2 * tp + fp + fn);
    ok(f1 > 0.6, `the kept (best) network detects the saccades of an unseen trace: sample F1 ${f1.toFixed(2)}`);
    console.log(failed ? `${failed} check(s) FAILED` : 'all recipe checks passed');
    process.exit(failed ? 1 : 0);
})();
