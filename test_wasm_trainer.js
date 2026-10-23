const fs = require('fs');
const path = require('path');

async function test() {
    console.log("Loading WebAssembly module...");
    const createUnEyeModule = require('./web/uneye_train.js');
    const wasmBinary = fs.readFileSync(path.join(__dirname, 'web', 'uneye_train.wasm'));
    
    const Module = await createUnEyeModule({
        wasmBinary: wasmBinary
    });

    console.log("Creating WebTrainer instance...");
    const trainer = new Module.WebTrainer();
    trainer.reset();

    // Configure: 2 classes, 50s, rot_aug=true, 4 rots, val=0.2, max_speed=1300, max_epochs=5, patience=4, lr=1e-3, seed=1, deg_per_unit=1.0
    trainer.setConfig(2, 50.0, true, 4, 0.2, 1300.0, 5, 4, 0.001, 1n, 1.0);

    // Create a 50s synthetic trace (50,000 samples @ 1000 Hz)
    console.log("Generating synthetic trace (50s, 1000 Hz)...");
    const N = 50000;
    const x = new Float32Array(N);
    const y = new Float32Array(N);
    const labels = new Uint8Array(N); // 0=fix, 1=sac

    let px = 0, py = 0;
    for (let i = 0; i < N; i++) {
        if (i % 200 >= 170 && i % 200 < 190) {
            // Saccade
            labels[i] = 1;
            px += 0.5;
            py += 0.2;
        } else {
            // Fixation
            labels[i] = 0;
            px += (Math.random() - 0.5) * 0.02;
            py += (Math.random() - 0.5) * 0.02;
        }
        x[i] = px;
        y[i] = py;
    }

    console.log("Adding trace to trainer...");
    trainer.addTrace(x, y, labels, 1000.0);

    console.log("Initializing training...");
    const initErr = trainer.initTraining();
    if (initErr) {
        throw new Error("initTraining failed: " + initErr);
    }

    console.log("Running training steps (epochs)...");
    const t0 = Date.now();
    for (let ep = 1; ep <= 3; ep++) {
        const stepT0 = Date.now();
        const jsonStr = trainer.step();
        const stepTime = (Date.now() - stepT0) / 1000;
        const p = JSON.parse(jsonStr);
        console.log(`Epoch ${p.epoch}: train_loss=${p.train_loss.toFixed(4)}, val_loss=${p.val_loss.toFixed(4)}, val_F1=${p.val_event_f1.toFixed(3)}, val_kappa=${p.val_kappa.toFixed(3)} (${stepTime.toFixed(2)}s)`);
    }
    const totalTime = (Date.now() - t0) / 1000;
    console.log(`Finished 3 epochs in ${totalTime.toFixed(2)}s (${(totalTime / 3).toFixed(2)}s/epoch)`);

    console.log("Testing exportBin...");
    const binBytes = trainer.exportBin();
    console.log(`Exported bin model: ${binBytes.length} bytes`);
    if (binBytes.length !== 75708) { // 2 classes = 75708 bytes
        throw new Error(`Unexpected bin size: ${binBytes.length}`);
    }

    console.log("Testing predictTrial...");
    const pred = trainer.predictTrial(x.subarray(0, 1000), y.subarray(0, 1000), 1000.0, 1300.0);
    console.log(`Predicted 1000 samples, saccade count: ${pred.filter(v => v === 1).length}`);

    console.log("Testing evaluateScoresJson...");
    const scoresJson = Module.WebTrainer.evaluateScoresJson(pred, labels.subarray(0, 1000), 2);
    console.log("Scores:", scoresJson);

    trainer.delete();
    console.log("\nALL WASM TRAINER TESTS PASSED!");
}

test().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});
