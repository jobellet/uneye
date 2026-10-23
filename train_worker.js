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
                    const p = trainer.predictTrial(tt.x, tt.y, tt.rateHz || 1000.0, config.max_speed_deg_s || 1300.0);
                    allPred.push(...p);
                    allTruth.push(...tt.labels);
                }
                const predArr = new Uint8Array(allPred);
                const truthArr = new Uint8Array(allTruth);
                const scoreJson = Module.WebTrainer.evaluateScoresJson(predArr, truthArr, config.n_classes || 3);
                testMetrics = JSON.parse(scoreJson);
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
