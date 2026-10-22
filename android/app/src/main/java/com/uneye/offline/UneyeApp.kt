package com.uneye.offline

import android.app.Application
import java.io.File

/**
 * Application object: owns the [Session] (the single native OfflineSession
 * handle) and the app's file layout (labels + trained models under filesDir).
 */
class UneyeApp : Application() {

    val session: Session by lazy { Session(filesDir) }

    /** Live training progress (phase 3), posted by the TrainingWorker. */
    val trainingState = kotlinx.coroutines.flow.MutableStateFlow(Session.TrainUi())

    /** The convergence history (phase 4): one JSON file in filesDir. */
    val convergenceStore: ConvergenceStore by lazy {
        ConvergenceStore(File(filesDir, "convergence.json"))
    }

    /** Directory for the trained models of this app. */
    fun modelsDir(): File = session.appDir("models")

    /** Labels path for the current recording (per trial: path + ".trialK"). */
    fun labelsPathFor(session: Session, trial: Long): String =
        File(session.appDir("labels"), "labels.bin").absolutePath

    /**
     * The shipped pre-trained model, copied from the assets on first use
     * (cpp/models/combined.bin: U-Net 2 classes, window 200, 1 kHz).
     */
    fun shippedModelPath(): String {
        val dir = modelsDir()
        val bin = File(dir, "combined.bin")
        val json = File(dir, "combined.json")
        if (!bin.exists()) {
            assets.open("models/combined.bin").use { input -> bin.outputStream().use { input.copyTo(it) } }
            assets.open("models/combined.json").use { input -> json.outputStream().use { input.copyTo(it) } }
        }
        return bin.absolutePath
    }

    override fun onTerminate() {
        session.destroy()
        super.onTerminate()
    }
}
