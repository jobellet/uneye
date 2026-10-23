package com.uneye.offline

import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.coroutines.resume

/**
 * Background training worker (phase 3). Runs the C++ training cycle through
 * the OfflineSession facade on the app's low-priority worker, with a
 * foreground-service notification so Android does not kill it mid-epoch.
 *
 * Triggered by [TrainingScheduler] after every N newly labeled seconds
 * (and manually from the UI). Each run starts from the last trained model
 * (init_model), then saves the new model and swaps the pre-labeler to it.
 *
 * Constraints (set by the scheduler): requires charging + device idle for
 * large data sets; the small default run only requires the device to be
 * idle-ish (no constraint beyond battery-not-low is enforced here — the
 * scheduler decides).
 */
class TrainingWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val app = (applicationContext as? UneyeApp)
            ?: (applicationContext.applicationContext as? UneyeApp)
            ?: return@withContext Result.failure()
        val session = app.session
        if (!session.loaded.value) return@withContext Result.failure()

        setForegroundSafely()

        val modelPath = File(app.modelsDir(), "model_round.bin").absolutePath
        val labelsPath = app.labelsPathFor(session, 0)
        val result = suspendCancellableCoroutine<Pair<Boolean, Session.TrainUi>> { cont ->
            var lastUi = Session.TrainUi()
            var started = false
            started = session.startTraining(
                maxSeconds = 50.0, nThreads = 2, maxEpochs = 500, patience = 4,
                rotationAug = true, initModel = "", useLastModelAsInit = true,
            ) { ui ->
                lastUi = ui
                app.trainingState.value = ui   // for the UI (phase 4 metrics)
                if (ui.finished || ui.cancelled || ui.failed) {
                    if (cont.isActive) cont.resume(Pair(true, lastUi))
                }
            }
            if (!started) {
                if (cont.isActive) cont.resume(Pair(false, lastUi))
            }
            cont.invokeOnCancellation { session.cancelTraining() }
        }
        val (started, lastUi) = result
        if (isStopped || lastUi.cancelled) return@withContext Result.failure()
        if (!started || lastUi.failed) return@withContext Result.failure()

        // save the model, swap the pre-labeler, re-prelabel the unreviewed parts
        if (!session.saveTrainedModel(modelPath)) return@withContext Result.failure()
        session.setPrelabeler(Native.PRELABELER_LAST_TRAINED, modelPath)
        session.rePrelabelAll()
        session.saveLabels(labelsPath)

        // record the round (phase 4 convergence history)
        val counts = session.scoreTestSetEventCounts(modelPath)
        val scores = try { session.scoreTestSet(modelPath) } catch (e: Exception) { null }
        app.convergenceStore.appendRound(
            ConvergenceStore.Round(
                atMs = System.currentTimeMillis(),
                labeledSeconds = session.reviewedSeconds.value,
                epochs = lastUi.epoch,
                valF1 = lastUi.valF1,          // biased (used for early stop): recorded, never used for the flag
                valKappa = lastUi.kappa,
                testF1Saccade = scores?.eventF1?.getOrElse(1) { 0.0 } ?: 0.0,
                testF1Blink = scores?.eventF1?.getOrElse(2) { 0.0 } ?: 0.0,
                testKappaSaccade = scores?.kappa?.getOrElse(1) { 0.0 } ?: 0.0,
                testKappaBlink = scores?.kappa?.getOrElse(2) { 0.0 } ?: 0.0,
                nSaccades = counts.getOrElse(1) { 0 },
                nBlinks = counts.getOrElse(2) { 0 },
                modelPath = modelPath,
            )
        )
        app.trainingState.value = lastUi.copy(running = false, finished = true)
        session.markTrainingBaseline()
        Result.success()
    }

    private suspend fun setForegroundSafely() {
        try {
            setForeground(makeForegroundInfo())
        } catch (_: Exception) {
            // foreground service type restrictions on newer APIs: training still runs
        }
    }

    private fun makeForegroundInfo(): ForegroundInfo {
        val channelId = "training"
        val nm = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as android.app.NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(android.app.NotificationChannel(channelId, "Training", android.app.NotificationManager.IMPORTANCE_LOW))
        }
        val notification = androidx.core.app.NotificationCompat.Builder(applicationContext, channelId)
            .setForegroundServiceBehavior(androidx.core.app.NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setContentTitle(applicationContext.getString(R.string.app_name))
            .setContentText(applicationContext.getString(R.string.training_running))
            .setSmallIcon(android.R.drawable.ic_menu_rotate)
            .setOngoing(true)
            .build()
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ForegroundInfo(1001, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            ForegroundInfo(1001, notification)
        }
    }
}
