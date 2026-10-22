package com.uneye.offline

/**
 * The single native entry point of the app: the OfflineSession facade
 * (android/offline_session/). Every function maps 1:1 to a JNI function in
 * offline_session_jni.cpp. All functions are thread-safe; the training
 * progress callback arrives on an arbitrary thread and must be posted to the
 * UI thread by the caller.
 *
 * Sample indices are in samples of the loaded trace (1 kHz after load()).
 */
object Native {

    init {
        System.loadLibrary("offline_session")
    }

    /** Pre-labelers for [load] / [setPrelabeler]. */
    const val PRELABELER_NONE = 0
    const val PRELABELER_HMM = 1
    const val PRELABELER_SHIPPED_MODEL = 2
    const val PRELABELER_LAST_TRAINED = 3

    /** Class codes: 0 fixation (background), 1 saccade, 2 blink. */
    const val CLS_FIXATION = 0
    const val CLS_SACCADE = 1
    const val CLS_BLINK = 2

    data class LoadResult(
        val ok: Boolean,
        val error: String,
        val nSamples: Long,
        val rateHz: Double,
        val degPerUnit: Double,
        val scaleEstimated: Boolean,
        val hmmRateWarning: Boolean,
        val prelabelerError: String,
        val prelabelOnset: IntArray,
        val prelabelOffset: IntArray,
        val prelabelCls: IntArray,
        val prelabelFromPrelabel: IntArray,
    )

    /** Training progress, delivered from a worker thread. */
    interface ProgressListener {
        fun onProgress(
            epoch: Int, trainLoss: Double, valLoss: Double, valF1: Double, kappa: Double,
            elapsedS: Double, finished: Boolean, cancelled: Boolean, failed: Boolean,
            message: String,
        )
    }

    data class TestScores(
        val ok: Boolean,
        val nSamples: Long,
        val eventF1: DoubleArray,   // per class, size 5
        val kappa: DoubleArray,     // per class, size 5
    )

    private var handle: Long = 0

    @Synchronized
    fun create() {
        if (handle == 0L) handle = nativeCreate()
    }

    @Synchronized
    fun destroy() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0
        }
    }

    private fun h(): Long {
        if (handle == 0L) throw IllegalStateException("Native.create() not called")
        return handle
    }

    fun load(
        xRows: Array<DoubleArray>,
        yRows: Array<DoubleArray>,
        rateHz: Double,
        degPerUnit: Double,
        estimateScale: Boolean,
        prelabeler: Int,
        modelBin: String,
    ): LoadResult {
        val r = nativeLoad(h(), xRows, yRows, rateHz, degPerUnit, estimateScale, prelabeler, modelBin)
        return LoadResult(
            ok = r[0] as Boolean,
            error = r[1] as String,
            nSamples = r[2] as Long,
            rateHz = r[3] as Double,
            degPerUnit = r[4] as Double,
            scaleEstimated = r[5] as Boolean,
            hmmRateWarning = r[6] as Boolean,
            prelabelerError = r[7] as String,
            prelabelOnset = r[8] as IntArray,
            prelabelOffset = r[9] as IntArray,
            prelabelCls = r[10] as IntArray,
            prelabelFromPrelabel = r[11] as IntArray,
        )
    }

    fun isLoaded(): Boolean = nativeIsLoaded(h())
    fun nSamples(): Long = nativeNSamples(h())
    fun nTrials(): Long = nativeNTrials(h())
    fun degPerUnit(): Double = nativeDegPerUnit(h())

    /** Positions in degrees; returns Pair(x, y) of equal length. */
    fun trace(trial: Long, from: Long, to: Long): Pair<FloatArray, FloatArray> {
        val r = nativeTrace(h(), trial, from, to)
        return Pair(r[0] as FloatArray, r[1] as FloatArray)
    }

    fun labels(trial: Long, from: Long, to: Long, onlyReviewed: Boolean): ByteArray =
        nativeLabels(h(), trial, from, to, onlyReviewed)

    // --- pass 1: point marks -------------------------------------------------

    fun addPointMark(trial: Long, at: Long, cls: Int) = nativeAddPointMark(h(), trial, at, cls)
    fun removePointMark(trial: Long, at: Long): Boolean = nativeRemovePointMark(h(), trial, at)
    fun clearPointMarks(trial: Long) = nativeClearPointMarks(h(), trial)

    /** Returns Pair(at, cls) arrays of the point marks of one trial. */
    fun pointMarks(trial: Long): Pair<IntArray, IntArray> {
        val r = nativePointMarks(h(), trial)
        return Pair(r[0] as IntArray, r[1] as IntArray)
    }

    /** Nearest pre-label onset within [tolerance] samples, or -1. */
    fun snapToPrelabel(trial: Long, at: Long, tolerance: Long): Long =
        nativeSnapToPrelabel(h(), trial, at, tolerance)

    // --- pass 2: refinement ----------------------------------------------------

    fun refineMark(trial: Long, at: Long, onset: Long, offset: Long, cls: Int): Boolean =
        nativeRefineMark(h(), trial, at, onset, offset, cls)

    fun deleteMark(trial: Long, at: Long): Boolean = nativeDeleteMark(h(), trial, at)
    fun markEpochReviewed(trial: Long, from: Long, to: Long) =
        nativeMarkEpochReviewed(h(), trial, from, to)
    fun undo(trial: Long): Boolean = nativeUndo(h(), trial)
    fun redo(trial: Long): Boolean = nativeRedo(h(), trial)

    /** Returns { onset, offset, cls, fromPrelabel } arrays of the expert events. */
    fun events(trial: Long): Array<IntArray> = nativeEvents(h(), trial)

    /** Returns { onset, offset, cls, fromPrelabel } arrays of the pre-labels. */
    fun prelabels(trial: Long): Array<IntArray> = nativePrelabels(h(), trial)

    fun reviewedSeconds(): Double = nativeReviewedSeconds(h())

    // --- persistence ------------------------------------------------------------

    fun saveLabels(path: String): Boolean = nativeSaveLabels(h(), path)
    fun loadLabels(path: String): Boolean = nativeLoadLabels(h(), path)

    // --- training (phase 3) -------------------------------------------------------

    fun startTraining(
        maxSeconds: Double, nThreads: Int, maxEpochs: Int, patience: Int,
        rotationAug: Boolean, initModel: String, useLastModelAsInit: Boolean,
        listener: ProgressListener,
    ): Boolean = nativeStartTraining(
        h(), maxSeconds, nThreads, maxEpochs, patience, rotationAug,
        initModel, useLastModelAsInit, listener,
    )

    fun cancelTraining() = nativeCancelTraining(h())
    fun trainingRunning(): Boolean = nativeTrainingRunning(h())
    fun waitTrainingDone() = nativeWaitTrainingDone(h())
    fun saveTrainedModel(path: String): Boolean = nativeSaveTrainedModel(h(), path)
    fun lastTrainedModel(): String = nativeLastTrainedModel(h())

    // --- convergence test set (phase 4) ---------------------------------------------

    fun chooseTestSet(everyNth: Long, minTrials: Long) = nativeChooseTestSet(h(), everyNth, minTrials)
    fun testTrials(): LongArray = nativeTestTrials(h())

    /** Returns { ok(0/1), nSamples, f1[5], kappa[5] } as a DoubleArray of 11. */
    fun scoreTestSet(modelBin: String): TestScores {
        val r = nativeScoreTestSet(h(), modelBin)
        return TestScores(
            ok = r[0] > 0.5,
            nSamples = r[1].toLong(),
            eventF1 = DoubleArray(5) { i -> r[2 + i] },
            kappa = DoubleArray(5) { i -> r[7 + i] },
        )
    }

    /** Expert event counts per class on the test set (size 5). */
    fun scoreTestSetEventCounts(modelBin: String): IntArray =
        nativeScoreTestSetEventCounts(h(), modelBin)

    // --- pre-labeling -----------------------------------------------------------------

    fun prelabelTrial(trial: Long): Boolean = nativePrelabelTrial(h(), trial)

    /**
     * Re-runs the current pre-labeler on every trial (phase 3 swap). Reviewed
     * ranges and expert events are never touched (apply_auto_labels); only the
     * automatic labels of the unreviewed parts are replaced.
     */
    fun rePrelabelAll(): Boolean = nativeRePrelabelAll(h())
    fun setPrelabeler(prelabeler: Int, modelBin: String) = nativeSetPrelabeler(h(), prelabeler, modelBin)
    fun prelabeler(): Int = nativePrelabeler(h())

    // --- JNI -----------------------------------------------------------------------

    private external fun nativeCreate(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeLoad(
        handle: Long, xRows: Array<DoubleArray>, yRows: Array<DoubleArray>,
        rateHz: Double, degPerUnit: Double, estimateScale: Boolean,
        prelabeler: Int, modelBin: String,
    ): Array<Any>
    private external fun nativeIsLoaded(handle: Long): Boolean
    private external fun nativeNSamples(handle: Long): Long
    private external fun nativeNTrials(handle: Long): Long
    private external fun nativeDegPerUnit(handle: Long): Double
    private external fun nativeTrace(handle: Long, trial: Long, from: Long, to: Long): Array<FloatArray>
    private external fun nativeLabels(handle: Long, trial: Long, from: Long, to: Long, onlyReviewed: Boolean): ByteArray
    private external fun nativeAddPointMark(handle: Long, trial: Long, at: Long, cls: Int)
    private external fun nativeRemovePointMark(handle: Long, trial: Long, at: Long): Boolean
    private external fun nativeClearPointMarks(handle: Long, trial: Long)
    private external fun nativePointMarks(handle: Long, trial: Long): Array<IntArray>
    private external fun nativeSnapToPrelabel(handle: Long, trial: Long, at: Long, tolerance: Long): Long
    private external fun nativeRefineMark(handle: Long, trial: Long, at: Long, onset: Long, offset: Long, cls: Int): Boolean
    private external fun nativeDeleteMark(handle: Long, trial: Long, at: Long): Boolean
    private external fun nativeMarkEpochReviewed(handle: Long, trial: Long, from: Long, to: Long)
    private external fun nativeUndo(handle: Long, trial: Long): Boolean
    private external fun nativeRedo(handle: Long, trial: Long): Boolean
    private external fun nativeEvents(handle: Long, trial: Long): Array<IntArray>
    private external fun nativePrelabels(handle: Long, trial: Long): Array<IntArray>
    private external fun nativeReviewedSeconds(handle: Long): Double
    private external fun nativeSaveLabels(handle: Long, path: String): Boolean
    private external fun nativeLoadLabels(handle: Long, path: String): Boolean
    private external fun nativeStartTraining(
        handle: Long, maxSeconds: Double, nThreads: Int, maxEpochs: Int, patience: Int,
        rotationAug: Boolean, initModel: String, useLastModelAsInit: Boolean,
        listener: ProgressListener,
    ): Boolean
    private external fun nativeCancelTraining(handle: Long)
    private external fun nativeTrainingRunning(handle: Long): Boolean
    private external fun nativeWaitTrainingDone(handle: Long)
    private external fun nativeSaveTrainedModel(handle: Long, path: String): Boolean
    private external fun nativeLastTrainedModel(handle: Long): String
    private external fun nativeChooseTestSet(handle: Long, everyNth: Long, minTrials: Long)
    private external fun nativeTestTrials(handle: Long): LongArray
    private external fun nativeScoreTestSet(handle: Long, modelBin: String): DoubleArray
    private external fun nativeScoreTestSetEventCounts(handle: Long, modelBin: String): IntArray
    private external fun nativePrelabelTrial(handle: Long, trial: Long): Boolean
    private external fun nativeRePrelabelAll(handle: Long): Boolean
    private external fun nativeSetPrelabeler(handle: Long, prelabeler: Int, modelBin: String)
    private external fun nativePrelabeler(handle: Long): Int
}
