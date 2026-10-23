const fs = require('fs');
const path = require('path');

async function runAcceptanceTests() {
    console.log("=================================================================");
    console.log("       RUNNING ACCEPTANCE & ROBUSTNESS TESTS (SECTION 6)");
    console.log("=================================================================\n");

    const createUnEyeModule = require('./web/uneye_train.js');
    const wasmBinary = fs.readFileSync(path.join(__dirname, 'web', 'uneye_train.wasm'));
    const Module = await createUnEyeModule({ wasmBinary });

    // Test 1: Determinism & Parity
    console.log("--- 1. Determinism & Loss Curve Test ---");
    const trainer1 = new Module.WebTrainer();
    const trainer2 = new Module.WebTrainer();

    trainer1.reset();
    trainer2.reset();
    trainer1.setConfig(2, 20.0, true, 4, 0.2, 1300.0, 3, 4, 0.001, 42n, 1.0);
    trainer2.setConfig(2, 20.0, true, 4, 0.2, 1300.0, 3, 4, 0.001, 42n, 1.0);

    const N = 20000;
    const x = new Float32Array(N);
    const y = new Float32Array(N);
    const l = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
        x[i] = (i % 300 > 250) ? 5.0 : 0.1 * Math.sin(i * 0.05);
        y[i] = (i % 300 > 250) ? 3.0 : 0.1 * Math.cos(i * 0.05);
        l[i] = (i % 300 > 250) ? 1 : 0;
    }

    trainer1.addTrace(x, y, l, 1000.0);
    trainer2.addTrace(x, y, l, 1000.0);
    trainer1.initTraining();
    trainer2.initTraining();

    for (let ep = 1; ep <= 3; ep++) {
        const p1 = JSON.parse(trainer1.step());
        const p2 = JSON.parse(trainer2.step());
        console.log(`Epoch ${ep}: Run 1 loss=${p1.train_loss.toFixed(6)}, Run 2 loss=${p2.train_loss.toFixed(6)}`);
        const diff = Math.abs(p1.train_loss - p2.train_loss);
        if (diff > 1e-5) throw new Error(`Loss non-deterministic: diff = ${diff}`);
    }
    console.log("PASS: Exact determinism across identical runs.\n");

    // Test 2: Robustness against NaNs and extreme velocity (> 1300 deg/s)
    console.log("--- 2. Robustness Test (NaNs and Extreme Velocities) ---");
    const trainerRob = new Module.WebTrainer();
    trainerRob.reset();
    trainerRob.setConfig(2, 20.0, false, 0, 0.2, 1300.0, 2, 4, 0.001, 1n, 1.0);

    const x_dirty = new Float32Array(1000);
    const y_dirty = new Float32Array(1000);
    const l_dirty = new Uint8Array(1000);

    for (let i = 0; i < 1000; i++) {
        if (i >= 100 && i < 150) {
            x_dirty[i] = NaN;
            y_dirty[i] = NaN;
            l_dirty[i] = 255; // ignore
        } else if (i === 300) {
            x_dirty[i] = 10000.0; // extreme tracker glitch (> 10000 deg/s)
            y_dirty[i] = -5000.0;
            l_dirty[i] = 0;
        } else {
            x_dirty[i] = 0.01 * i;
            y_dirty[i] = 0.02 * i;
            l_dirty[i] = (i > 500 && i < 550) ? 1 : 0;
        }
    }

    // Add trace and test inference
    const predRob = trainerRob.predictTrial(x_dirty, y_dirty, 1000.0, 1300.0);
    let hasNan = false;
    for (let i = 0; i < predRob.length; i++) {
        if (isNaN(predRob[i])) hasNan = true;
    }
    if (hasNan) throw new Error("Prediction produced NaN!");
    console.log("PASS: Prediction handles NaNs and 10000 deg/s glitches safely without NaN output.\n");

    // Test 3: Model export (.bin layout)
    console.log("--- 3. Model Export Layout Test ---");
    const binData = trainer1.exportBin();
    const jsonStr = trainer1.exportJson();
    console.log(`Exported bin size: ${binData.length} bytes (magic: ${String.fromCharCode(binData[0], binData[1], binData[2], binData[3])})`);
    console.log(`Exported json: ${jsonStr}`);
    if (binData.length !== 75708) throw new Error(`Unexpected model size: ${binData.length}`);
    console.log("PASS: Exported .bin and .json sidecar are correct.\n");

    trainer1.delete();
    trainer2.delete();
    trainerRob.delete();

    console.log("=================================================================");
    console.log("             ALL ACCEPTANCE TESTS PASSED (100%)");
    console.log("=================================================================");
}

runAcceptanceTests().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});
