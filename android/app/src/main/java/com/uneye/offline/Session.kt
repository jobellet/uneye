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

    var rateHz: Double = 1000.0
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
        _loaded.value = false
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
                this.rateHz = rateHz
                this.degPerUnit = res.degPerUnit
                this.recordingName = recordingName
                scans.clear(); erased.clear()
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

    // --- windows seen in pass 1 -----------------------------------------------------
    // Only a stretch that was on screen in pass 1 can become reviewed (= can be trained on). The facade reviews it
    // only when no point mark is still waiting for its refinement, so the scans are re-applied after every edit.
    private val scans = HashMap<Long, MutableList<LongArray>>()
    // ranges wiped with the eraser: their automatic pre-labels are no longer drawn nor snapped to
    private val erased = HashMap<Long, MutableList<LongArray>>()

    fun addScan(trial: Long, from: Long, to: Long) {
        if (to < from) return
        val l = scans.getOrPut(trial) { mutableListOf() }
        if (l.none { it[0] == from && it[1] == to }) l.add(longArrayOf(from, to))
        applyScans(trial)
    }

    fun inScanned(trial: Long, at: Long): Boolean = scans[trial]?.any { at >= it[0] && at <= it[1] } == true

    fun applyScans(trial: Long) {
        scans[trial]?.forEach { native.markEpochReviewed(trial, it[0], it[1]) }
        _reviewedSeconds.value = native.reviewedSeconds()
    }

    /** Eraser: removes every event (saccade or blink), point mark and pre-label overlapping [a]..[b]. */
    fun eraseRange(trial: Long, a: Long, b: Long) {
        val lo = minOf(a, b); val hi = maxOf(a, b)
        val ui = trialUi(trial)
        for (m in ui.marks) if (m.offset >= lo && m.onset <= hi) native.deleteMark(trial, m.onset)
        for (p in ui.pointMarks) if (p.first in lo..hi) native.removePointMark(trial, p.first)
        erased.getOrPut(trial) { mutableListOf() }.add(longArrayOf(lo, hi))
        applyScans(trial)
    }

    private fun isErased(trial: Long, onset: Long, offset: Long): Boolean =
        erased[trial]?.any { offset >= it[0] && onset <= it[1] } == true

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
        }.filter { !isErased(k, it.onset, it.offset) }
        val pm = native.pointMarks(k)
        val points = pm.first.indices.map { i -> Pair(pm.first[i].toLong(), pm.second[i]) }
        return TrialUi(k, trialSamples, marks, pre, points, k in testTrials)
    }

    // --- pass 1 ----------------------------------------------------------------------

    fun addPointMark(trial: Long, at: Long, cls: Int) = native.addPointMark(trial, at, cls)
    fun removePointMark(trial: Long, at: Long): Boolean {
        val ok = native.removePointMark(trial, at)
        if (ok) applyScans(trial)
        return ok
    }
    fun clearPointMarks(trial: Long) = native.clearPointMarks(trial)
    fun snapToPrelabel(trial: Long, at: Long, tolerance: Long): Long {
        val s = native.snapToPrelabel(trial, at, tolerance)
        return if (s >= 0 && erased[trial]?.any { s >= it[0] && s <= it[1] } == true) -1L else s
    }

    // --- pass 2 ----------------------------------------------------------------------

    fun refineMark(trial: Long, at: Long, onset: Long, offset: Long, cls: Int): Boolean {
        val ok = native.refineMark(trial, at, onset, offset, cls)
        if (ok) applyScans(trial)          // the stretch may have no pending mark left
        return ok
    }

    fun deleteMark(trial: Long, at: Long): Boolean {
        val ok = native.deleteMark(trial, at)
        applyScans(trial)
        return ok
    }

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

    fun saveLabels(path: String): Boolean {
        val ok = native.saveLabels(path)
        try {
            val lines = scans.flatMap { (t, l) -> l.map { "$t ${it[0]} ${it[1]}" } }
            File(File(path).parentFile, "scans.txt").writeText(lines.joinToString("\n"))
        } catch (_: Exception) { }
        return ok
    }

    fun loadLabels(path: String): Boolean {
        val ok = native.loadLabels(path)
        try {
            scans.clear()
            File(File(path).parentFile, "scans.txt").takeIf { it.exists() }?.readLines()?.forEach { line ->
                val p = line.trim().split(" ")
                if (p.size == 3) scans.getOrPut(p[0].toLong()) { mutableListOf() }.add(longArrayOf(p[1].toLong(), p[2].toLong()))
            }
        } catch (_: Exception) { }
        return ok
    }

    /** Imports labels from a JSON string (uneye-labels-v1 session file format). */
    fun importLabelsJson(jsonStr: String): Boolean {
        return try {
            val root = org.json.JSONObject(jsonStr)
            val trials = root.optJSONArray("trials") ?: return false
            val nT = _nTrials.value
            for (t in 0 until minOf(nT.toInt(), trials.length())) {
                val obj = trials.getJSONObject(t)
                val scanWins = obj.optJSONArray("scanWins")
                if (scanWins != null) {
                    for (i in 0 until scanWins.length()) {
                        val w = scanWins.getJSONArray(i)
                        addScan(t.toLong(), w.getLong(0), w.getLong(1))
                    }
                }
                val exp = obj.optJSONArray("expertEvents")
                if (exp != null) {
                    for (i in 0 until exp.length()) {
                        val e = exp.getJSONObject(i)
                        val on = e.getLong("onset")
                        val off = e.getLong("offset")
                        val c = e.getInt("cls")
                        refineMark(t.toLong(), on, on, off, c)
                    }
                }
                val pm = obj.optJSONArray("pointMarks")
                if (pm != null) {
                    for (i in 0 until pm.length()) {
                        val p = pm.getJSONObject(i)
                        addPointMark(t.toLong(), p.getLong("at"), p.getInt("cls"))
                    }
                }
                applyScans(t.toLong())
            }
            true
        } catch (_: Exception) {
            false
        }
    }

    /** Imports labels from a 2D matrix (trials x samples: 0 fixation, 1 saccade, 2 blink, NaN / -1 unreviewed). */
    fun importLabelsMatrix(labels: Array<DoubleArray>): Boolean {
        return try {
            val nT = _nTrials.value
            for (t in 0 until minOf(nT.toInt(), labels.size)) {
                val row = labels[t]
                val n = row.size
                var i = 0
                while (i < n) {
                    if (row[i].isNaN() || row[i] < 0) { i++; continue }
                    var j = i
                    while (j + 1 < n && !row[j + 1].isNaN() && row[j + 1] >= 0) j++
                    addScan(t.toLong(), i.toLong(), j.toLong())
                    i = j + 1
                }
                i = 0
                while (i < n) {
                    val c = if (row[i].isNaN()) -1 else row[i].toInt()
                    if (c != 1 && c != 2) { i++; continue }
                    var j = i
                    while (j + 1 < n && !row[j + 1].isNaN() && row[j + 1].toInt() == c) j++
                    refineMark(t.toLong(), i.toLong(), i.toLong(), j.toLong(), c)
                    i = j + 1
                }
                applyScans(t.toLong())
            }
            true
        } catch (_: Exception) {
            false
        }
    }

    /** Imports labels from a CSV string (rows = trials, values = 0/1/2/nan). */
    fun importLabelsCsv(csvStr: String): Boolean {
        return try {
            val lines = csvStr.lines().filter { it.isNotBlank() }
            val rows = lines.map { line ->
                val tokens = line.split(',').map { it.trim() }
                DoubleArray(tokens.size) { idx ->
                    val tok = tokens[idx]
                    if (tok.equals("nan", ignoreCase = true) || tok.isEmpty()) Double.NaN else (tok.toDoubleOrNull() ?: Double.NaN)
                }
            }.toTypedArray()
            importLabelsMatrix(rows)
        } catch (_: Exception) {
            false
        }
    }

    /** Imports labels from text (auto-detecting JSON or CSV). */
    fun importLabelsText(text: String): Boolean {
        val trimmed = text.trim()
        return if (trimmed.startsWith("{")) importLabelsJson(trimmed) else importLabelsCsv(trimmed)
    }

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
