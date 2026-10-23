package com.uneye.offline

import android.os.Handler
import android.os.Looper
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

/**
 * The labeling session state machine (phase 2). Kotlin owns only the UI state
 * (which trial, which pass, the cursor, the zoom); every label / edit decision
 * lives in the C++ OfflineSession through [Native] — Kotlin draws and handles
 * gestures, nothing else.
 *
 * Pass 1: one-tap point marks with auto-scroll (an epoch is scanned when the
 * view passes its end; the epoch becomes reviewed only when scanned AND all
 * its marks are refined — enforced by the facade).
 * Pass 2: press-drag-lift sets onset/offset per mark, in order.
 */
class Session(private val filesDir: File) {

    enum class Pass { ONE, TWO }

    data class MarkUi(
        val onset: Long, val offset: Long, val cls: Int, val fromPrelabel: Boolean,
    )

    data class TrialUi(
        val index: Long,
        val nSamples: Long,
        val marks: List<MarkUi>,          // expert events (pass 2)
        val prelabels: List<MarkUi>,      // dimmed pre-labels
        val pointMarks: List<Pair<Long, Int>>,   // pass-1 marks (at, cls)
        val isTest: Boolean,
    )

    data class TrainUi(
        val running: Boolean = false,
        val epoch: Int = 0,
        val trainLoss: Double = 0.0,
        val valLoss: Double = 0.0,
        val valF1: Double = 0.0,
        val kappa: Double = 0.0,
        val elapsedS: Double = 0.0,
        val finished: Boolean = false,
        val cancelled: Boolean = false,
        val failed: Boolean = false,
        val message: String = "",
    )

    private val native = Native
    private val uiHandler = Handler(Looper.getMainLooper())

    private val _loaded = MutableStateFlow(false)
    val loaded: StateFlow<Boolean> = _loaded.asStateFlow()

    private val _nTrials = MutableStateFlow(0L)
    val nTrials: StateFlow<Long> = _nTrials.asStateFlow()

    private val _reviewedSeconds = MutableStateFlow(0.0)
    val reviewedSeconds: StateFlow<Double> = _reviewedSeconds.asStateFlow()

    /** Seconds reviewed at the last training round (phase 3 scheduler baseline). */
    var reviewedAtLastTraining: Double = 0.0
        private set

    /** Marks the current reviewed-seconds value as the baseline for the next training trigger. */
    fun markTrainingBaseline() { reviewedAtLastTraining = _reviewedSeconds.value }

    private val _train = MutableStateFlow(TrainUi())
    val train: StateFlow<TrainUi> = _train.asStateFlow()

    private val _pass = MutableStateFlow(Pass.ONE)
    val pass: StateFlow<Pass> = _pass.asStateFlow()

    private val _trial = MutableStateFlow(0L)
    val trial: StateFlow<Long> = _trial.asStateFlow()

    var trialSamples: Long = 1000   // samples per trial at 1 kHz (1 s)
        private set

    var degPerUnit: Double = 1.0
        private set

    /** Display name of the loaded recording (for export file names). */
    var recordingName: String = "recording"
        private set

    private var testTrials: Set<Long> = emptySet()
    private var lastPrelabels: List<MarkUi> = emptyList()

    val prelabels: List<MarkUi> get() = lastPrelabels

    init {
        native.create()
    }

    fun destroy() {
        native.destroy()
    }

    // --- loading ---------------------------------------------------------------

    /**
     * Loads a recording: [trace] (X/Y rows), declared [rateHz], [degPerUnit],
     * [estimateScale] (select_scale against the HMM reference), [prelabeler]
     * and the optional shipped model path. Runs on a worker thread; the
     * result is posted to [onDone] on the main thread.
     */
    fun load(
        trace: TraceFiles.Trace, rateHz: Double, degPerUnit: Double, estimateScale: Boolean,
        prelabeler: Int, modelBin: String, recordingName: String = "recording",
        onDone: (ok: Boolean, error: String) -> Unit,
    ) {
        Thread {
            val res = native.load(
                trace.x.toTypedArray(), trace.y.toTypedArray(),
                rateHz, degPerUnit, estimateScale, prelabeler, modelBin,
            )
            val marks = res.prelabelOnset.indices.map { i ->
                MarkUi(res.prelabelOnset[i].toLong(), res.prelabelOffset[i].toLong(),
                       res.prelabelCls[i], res.prelabelFromPrelabel[i] == 1)
            }
            // the facade pre-labels every trial; split the flat list per trial
            val perTrial = mutableListOf<MutableList<MarkUi>>()
            val nT = native.nTrials()
            val per = if (nT > 0) res.nSamples / nT else 0
            var idx = 0
            for (k in 0 until nT) {
                val list = mutableListOf<MarkUi>()
                // pre-labels are sorted by trial then onset; count how many fall in [k*per, (k+1)*per)
                while (idx < marks.size && marks[idx].onset < (k + 1) * per) { list.add(marks[idx]); ++idx }
                perTrial.add(list)
            }
            lastPrelabels = marks
            uiHandler.post {
                _nTrials.value = nT
                trialSamples = if (nT > 0) per else 1000
                this.degPerUnit = res.degPerUnit
                this.recordingName = recordingName
                _loaded.value = res.ok
                _reviewedSeconds.value = native.reviewedSeconds()
                onDone(res.ok, if (res.ok) "" else res.error.ifEmpty { res.prelabelerError })
            }
        }.start()
    }

    fun chooseTestSet(everyNth: Long, minTrials: Long) {
        native.chooseTestSet(everyNth, minTrials)
        testTrials = native.testTrials().toSet()
    }

    fun isTestTrial(k: Long): Boolean = k in testTrials

    // --- trace / labels for drawing -------------------------------------------------

    /** Positions in degrees for [trial], samples [from]..[to] inclusive. Off the UI thread. */
    fun trace(trial: Long, from: Long, to: Long): Pair<FloatArray, FloatArray> =
        native.trace(trial, from, to)

    /** Rendered per-sample labels (0 background, 1 saccade, 2 blink, -1 ignored). */
    fun labels(trial: Long, from: Long, to: Long, onlyReviewed: Boolean): ByteArray =
        native.labels(trial, from, to, onlyReviewed)

    fun trialUi(k: Long): TrialUi {
        val marks = native.events(k).let { e ->
            e[0].indices.map { i -> MarkUi(e[0][i].toLong(), e[1][i].toLong(), e[2][i], e[3][i] == 1) }
        }
        val pre = native.prelabels(k).let { e ->
            e[0].indices.map { i -> MarkUi(e[0][i].toLong(), e[1][i].toLong(), e[2][i], e[3][i] == 1) }
        }
        val pm = native.pointMarks(k)
        val points = pm.first.indices.map { i -> Pair(pm.first[i].toLong(), pm.second[i]) }
        return TrialUi(k, trialSamples, marks, pre, points, k in testTrials)
    }

    // --- pass 1 ----------------------------------------------------------------------

    fun addPointMark(trial: Long, at: Long, cls: Int) = native.addPointMark(trial, at, cls)
    fun removePointMark(trial: Long, at: Long): Boolean = native.removePointMark(trial, at)
    fun clearPointMarks(trial: Long) = native.clearPointMarks(trial)
    fun snapToPrelabel(trial: Long, at: Long, tolerance: Long): Long =
        native.snapToPrelabel(trial, at, tolerance)

    // --- pass 2 ----------------------------------------------------------------------

    fun refineMark(trial: Long, at: Long, onset: Long, offset: Long, cls: Int): Boolean =
        native.refineMark(trial, at, onset, offset, cls)

    fun deleteMark(trial: Long, at: Long): Boolean = native.deleteMark(trial, at)

    /** The epoch [from]..[to] was scanned and all its marks refined (facade checks). */
    fun markEpochReviewed(trial: Long, from: Long, to: Long) {
        native.markEpochReviewed(trial, from, to)
        _reviewedSeconds.value = native.reviewedSeconds()
    }

    fun undo(trial: Long): Boolean = native.undo(trial)
    fun redo(trial: Long): Boolean = native.redo(trial)

    fun setPass(p: Pass) { _pass.value = p }
    fun setTrial(k: Long) { _trial.value = k }

    // --- persistence ------------------------------------------------------------------

    fun saveLabels(path: String): Boolean = native.saveLabels(path)
    fun loadLabels(path: String): Boolean = native.loadLabels(path)

    // --- training (phase 3) --------------------------------------------------------------

    /**
     * Starts the background trainer. Progress arrives on a worker thread and
     * is posted to the main thread through [onProgress].
     */
    fun startTraining(
        maxSeconds: Double, nThreads: Int, maxEpochs: Int, patience: Int,
        rotationAug: Boolean, initModel: String, useLastModelAsInit: Boolean,
        onProgress: (TrainUi) -> Unit,
    ): Boolean {
        val listener = object : Native.ProgressListener {
            override fun onProgress(
                epoch: Int, trainLoss: Double, valLoss: Double, valF1: Double, kappa: Double,
                elapsedS: Double, finished: Boolean, cancelled: Boolean, failed: Boolean,
                message: String,
            ) {
                val ui = TrainUi(!finished && !cancelled && !failed, epoch, trainLoss, valLoss,
                                 valF1, kappa, elapsedS, finished, cancelled, failed, message)
                uiHandler.post {
                    _train.value = ui
                    onProgress(ui)
                }
            }
        }
        val ok = native.startTraining(maxSeconds, nThreads, maxEpochs, patience, rotationAug,
                                      initModel, useLastModelAsInit, listener)
        if (ok) _train.value = TrainUi(running = true)
        return ok
    }

    fun cancelTraining() = native.cancelTraining()
    fun trainingRunning(): Boolean = native.trainingRunning()
    fun saveTrainedModel(path: String): Boolean = native.saveTrainedModel(path)
    fun lastTrainedModel(): String = native.lastTrainedModel()

    // --- convergence (phase 4) --------------------------------------------------------------

    fun scoreTestSet(modelBin: String): Native.TestScores = native.scoreTestSet(modelBin)
    fun scoreTestSetEventCounts(modelBin: String): IntArray = native.scoreTestSetEventCounts(modelBin)

    // --- pre-labeler ---------------------------------------------------------------------------

    fun setPrelabeler(prelabeler: Int, modelBin: String) = native.setPrelabeler(prelabeler, modelBin)
    fun prelabelTrial(trial: Long): Boolean = native.prelabelTrial(trial)
    fun rePrelabelAll(): Boolean = native.rePrelabelAll()
    fun prelabeler(): Int = native.prelabeler()

    /** Directory for the app's own files (labels, trained models). */
    fun appDir(sub: String): File = File(filesDir, sub).apply { mkdirs() }
}
