// U'n'Eye Web Worker: On-device background neural network training and inference.
// Communicates with app.js without blocking the main UI thread (maintaining 60 fps).

let Module = null;
let trainer = null;
let isCancelled = false;
let isTraining = false;

async function ensureModule() {
    if (!Module) {
        if (typeof importScripts === 'function') {
            importScripts('web/uneye_train.js');
        }
        Module = await createUnEyeModule({
            locateFile: (path) => 'web/' + path
        });
    }
    if (!trainer) {
        trainer = new Module.WebTrainer();
    }
}

self.onmessage = async function(e) {
    const { type, data, id } = e.data;

    try {
        await ensureModule();

        if (type === 'CANCEL') {
            isCancelled = true;
            if (trainer) trainer.cancel();
            self.postMessage({ type: 'CANCELLED', id });
            return;
        }

        if (type === 'PREDICT') {
            // data: { x, y, rateHz, maxSpeedDegS }
            const pred = trainer.predictTrial(data.x, data.y, data.rateHz || 1000.0, data.maxSpeedDegS || 1300.0);
            self.postMessage({ type: 'PREDICT_RESULT', id, pred });
            return;
        }

        if (type === 'LOAD_MODEL') {
            // data: Uint8Array
            const ok = trainer.loadModelFromBytes(data);
            self.postMessage({ type: 'LOAD_MODEL_RESULT', id, success: ok });
            return;
        }

        if (type === 'EXPORT_MODEL') {
            const bin = trainer.exportBin();
            const json = trainer.exportJson();
            self.postMessage({ type: 'EXPORT_MODEL_RESULT', id, bin, json });
            return;
        }

        if (type === 'START_TRAIN') {
            const { config, traces, testTrials } = data;
            isCancelled = false;
            isTraining = true;

            trainer.reset();
            trainer.setConfig(
                config.n_classes || 3,
                config.max_seconds_of_data || 50.0,
                config.rotation_aug !== false,
                config.rotations || 4,
                config.val_fraction || 0.2,
                config.max_speed_deg_s || 1300.0,
                config.max_epochs || 60,
                config.patience || 4,
                config.learning_rate || 1e-3,
                BigInt(config.seed || 1),
                config.deg_per_unit || 1.0
            );

            if (trainer.setRecipe) {                       // whole traces held out for validation, windows re-cut every epoch, minibatches of batch_size
                trainer.setRecipe(config.resample_each_epoch === true, config.batch_size || 0);
            }

            if (config.init_model_bytes) {
                trainer.setInitModelBytes(config.init_model_bytes);
            }

            for (const tr of traces) {
                trainer.addTrace(tr.x, tr.y, tr.labels, tr.rateHz || 1000.0);
            }

            const initErr = trainer.initTraining();
            if (initErr) {
                isTraining = false;
                self.postMessage({ type: 'ERROR', id, message: initErr });
                return;
            }

            // Train epoch by epoch cooperatively so cancel can interrupt
            while (!trainer.isFinished() && !isCancelled) {
                const progJson = trainer.step();
                const prog = JSON.parse(progJson);
                self.postMessage({ type: 'PROGRESS', id, data: prog });

                if (prog.finished || prog.cancelled) {
                    break;
                }

                // Small yield to process incoming worker messages (such as CANCEL)
                await new Promise(resolve => setTimeout(resolve, 0));
            }

            isTraining = false;

            if (isCancelled || trainer.isCancelled()) {
                self.postMessage({ type: 'CANCELLED', id });
                return;
            }

            // Training finished! Evaluate on held-out test trials
            let testMetrics = {
                testF1Saccade: 0.0,
                testKappaSaccade: 0.0,
                testF1Blink: 0.0,
                testKappaBlink: 0.0
            };

            if (testTrials && testTrials.length > 0) {
                let allPred = [];
                let allTruth = [];
                for (const tt of testTrials) {
                    const raw = trainer.predictTrial(tt.x, tt.y, tt.rateHz || 1000.0, config.max_speed_deg_s || 1300.0);
                    const p = config.postprocess === false ? raw : postprocessSaccades(raw, tt.rateHz || 1000.0);
                    for (let i = 0; i < tt.labels.length; i++) {
                        if (tt.labels[i] < 250) {
                            allPred.push(p[i]);
                            allTruth.push(tt.labels[i]);
                        }
                    }
                }
                if (allTruth.length > 0) {
                    const predArr = new Uint8Array(allPred);
                    const truthArr = new Uint8Array(allTruth);
                    const scoreJson = Module.WebTrainer.evaluateScoresJson(predArr, truthArr, config.n_classes || 3);
                    testMetrics = JSON.parse(scoreJson);
                }
            }

            const binModel = trainer.exportBin();
            const jsonSidecar = trainer.exportJson();

            self.postMessage({
                type: 'TRAIN_COMPLETE',
                id,
                metrics: testMetrics,
                binModel,
                jsonSidecar
            });
        }

    } catch (err) {
        isTraining = false;
        self.postMessage({ type: 'ERROR', id, message: err.message || String(err) });
    }
};

// U'n'Eye post-processing of the saccade class (as app.js::postprocessSaccades and uneye/functions.py::binary_prediction): saccades separated by less
// than mergeMs are merged, saccades shorter than minMs are removed; the test-set scores are computed on the post-processed predictions
function postprocessSaccades(pred, fs, mergeMs = 5, minMs = 5) {
    const n = pred.length, merge = Math.round(mergeMs * fs / 1000), minLen = Math.round(minMs * fs / 1000);
    const p = Uint8Array.from(pred);
    let lastEnd = -1;
    for (let i = 0; i < n; i++) {
        if (p[i] !== 1) continue;
        const gap = i - lastEnd - 1;
        if (lastEnd >= 0 && gap > 0 && gap < merge) {
            let fixation = true;
            for (let k = lastEnd + 1; k < i; k++) if (p[k] !== 0) { fixation = false; break; }
            if (fixation) for (let k = lastEnd + 1; k < i; k++) p[k] = 1;
        }
        let j = i; while (j < n && p[j] === 1) j++;
        lastEnd = j - 1; i = j - 1;
    }
    for (let i = 0; i < n;) {
        if (p[i] !== 1) { i++; continue; }
        let j = i; while (j < n && p[j] === 1) j++;
        if (j - i < minLen) for (let k = i; k < j; k++) p[k] = 0;
        i = j;
    }
    return p;
}
