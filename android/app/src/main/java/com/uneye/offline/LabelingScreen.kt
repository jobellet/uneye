package com.uneye.offline

import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Touch-optimized two-pass saccade and blink labeling screen (aligned with web app).
 *
 * Pass 1 (one-tap mark with auto-scroll):
 *  - Tools: Saccade (blue), Blink (red), Eraser (clear).
 *  - Tap places a point mark (or removes it with eraser), snapping to putative pre-labels.
 *  - Auto-scrolls after each marked event.
 *  - Epoch and Trial navigation buttons (10% epoch overlap).
 *
 * Pass 2 (event-by-event focused refinement):
 *  - Always centers strictly on one labeled event from Pass 1:
 *    Window framed [-50 ms .. +50 ms] around event boundaries ([onset - 50ms, offset + 50ms]).
 *  - Event counter badge: "Event X/Y (Saccade, 120 ms)".
 *  - Single-event navigation: "◀ Prev", "Next ▶".
 *  - Action buttons: "Accept Pre-label", "Not an Event", "Wider (±50ms)".
 *  - Press-drag-lift on canvas sets fine-grained onset/offset.
 *
 * Filter & Preprocessing Settings:
 *  - Accessible via "⚙ Filter" button on top bar.
 *  - Configurable Savitzky-Golay cubic filter window (default 21 ms) and adaptive threshold (λ·σ).
 *  - Saved in uneye_config.json alongside exports for exact Google Colab reproduction.
 */
class LabelingScreen : AppCompatActivity() {

    private lateinit var session: Session
    private lateinit var traceView: TraceView

    // Top Bar Views
    private lateinit var passIndicator: TextView
    private lateinit var toolGroup: View
    private lateinit var saccadeToolBtn: Button
    private lateinit var blinkToolBtn: Button
    private lateinit var eraserToolBtn: Button
    private lateinit var eventCounterBadge: TextView
    private lateinit var filterBtn: Button
    private lateinit var undoBtn: Button

    // Status & Feedback
    private lateinit var statusText: TextView
    private lateinit var trainStatusText: TextView
    private lateinit var convergenceBadge: TextView

    // Contextual Controls
    private lateinit var pass1Controls: View
    private lateinit var prevTrialBtn: Button
    private lateinit var nextTrialBtn: Button
    private lateinit var prevEpochBtn: Button
    private lateinit var nextEpochBtn: Button
    private lateinit var winMsEdit: EditText
    private lateinit var menuBtn: Button
    private lateinit var settingsPanel: View

    private lateinit var pass2Controls: View
    private lateinit var prevEventBtn: Button
    private lateinit var nextEventBtn: Button
    private lateinit var acceptPreBtn: Button
    private lateinit var notEventBtn: Button
    private lateinit var widerBtn: Button

    // Global Actions
    private lateinit var switchPassBtn: Button
    private lateinit var saveBtn: Button
    private lateinit var trainNowBtn: Button
    private lateinit var metricsBtn: Button
    private lateinit var exportBtn: Button

    // Settings & State
    private var filterSettings = FilterSettings.DEFAULT
    private var epochMs = 500L            // epoch length: 500 ms default
    private var activeTool = TraceView.TOOL_SACCADE
    private var cls = Native.CLS_SACCADE  // active class for marking
    private var zoomMultiplier = 1.0

    private val trial: Long get() = session.trial.value
    private val pass: Session.Pass get() = session.pass.value

    // Pass 1 state
    private var lastMarkSample = -1L

    // Pass 2 state: the queue of marks to refine
    private var refineQueue: MutableList<Pair<Long, Int>> = mutableListOf()   // (at, cls)
    private var refineIdx = 0
    private var currentMarkAt = -1L
    private var currentPutativeOnset = -1L
    private var currentPutativeOffset = -1L
    private var currentMarkCls = Native.CLS_SACCADE

    private val ui = Handler(Looper.getMainLooper())
    private lateinit var exportHelper: ExportHelper
    private var convergenceAlertShown = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_labeling)

        session = (application as UneyeApp).session
        filterSettings = FilterSettings.load(this)

        traceView = findViewById(R.id.traceView)
        passIndicator = findViewById(R.id.passIndicator)
        toolGroup = findViewById(R.id.toolGroup)
        saccadeToolBtn = findViewById(R.id.saccadeToolBtn)
        blinkToolBtn = findViewById(R.id.blinkToolBtn)
        eraserToolBtn = findViewById(R.id.eraserToolBtn)
        eventCounterBadge = findViewById(R.id.eventCounterBadge)
        filterBtn = findViewById(R.id.filterBtn)
        undoBtn = findViewById(R.id.undoBtn)

        statusText = findViewById(R.id.statusText)
        trainStatusText = findViewById(R.id.trainStatusText)
        convergenceBadge = findViewById(R.id.convergenceBadge)

        pass1Controls = findViewById(R.id.pass1Controls)
        prevTrialBtn = findViewById(R.id.prevTrialBtn)
        nextTrialBtn = findViewById(R.id.nextTrialBtn)
        prevEpochBtn = findViewById(R.id.prevEpochBtn)
        nextEpochBtn = findViewById(R.id.nextEpochBtn)
        winMsEdit = findViewById(R.id.winMsEdit)
        menuBtn = findViewById(R.id.menuBtn)
        settingsPanel = findViewById(R.id.settingsPanel)
        epochMs = getSharedPreferences("uneye_prefs", MODE_PRIVATE).getLong("winMs", 500L).coerceIn(20L, 60_000L)
        winMsEdit.setText(epochMs.toString())

        pass2Controls = findViewById(R.id.pass2Controls)
        prevEventBtn = findViewById(R.id.prevEventBtn)
        nextEventBtn = findViewById(R.id.nextEventBtn)
        acceptPreBtn = findViewById(R.id.acceptPreBtn)
        notEventBtn = findViewById(R.id.notEventBtn)
        widerBtn = findViewById(R.id.widerBtn)

        switchPassBtn = findViewById(R.id.switchPassBtn)
        saveBtn = findViewById(R.id.saveBtn)
        trainNowBtn = findViewById(R.id.trainNowBtn)
        metricsBtn = findViewById(R.id.metricsBtn)
        exportBtn = findViewById(R.id.exportBtn)

        exportHelper = ExportHelper(this)

        setupToolSelection()
        setupListeners()
        setupObservers()

        updatePassUi()
        refresh()
        refreshConvergence()
    }

    private fun setupToolSelection() {
        saccadeToolBtn.setOnClickListener { selectTool(TraceView.TOOL_SACCADE, Native.CLS_SACCADE) }
        blinkToolBtn.setOnClickListener { selectTool(TraceView.TOOL_BLINK, Native.CLS_BLINK) }
        eraserToolBtn.setOnClickListener {
            if (activeTool == TraceView.TOOL_ERASER) {
                // Toggling eraser off restores current class tool (saccade or blink)
                val backTool = if (cls == Native.CLS_BLINK) TraceView.TOOL_BLINK else TraceView.TOOL_SACCADE
                selectTool(backTool, cls)
            } else {
                selectTool(TraceView.TOOL_ERASER, 0)
            }
        }
        selectTool(TraceView.TOOL_SACCADE, Native.CLS_SACCADE)
    }

    private fun selectTool(tool: Int, classId: Int) {
        activeTool = tool
        if (classId != 0) cls = classId
        traceView.activeTool = tool

        // Update button visual styles
        saccadeToolBtn.setBackgroundColor(if (tool == TraceView.TOOL_SACCADE) 0xFF1565C0.toInt() else 0xFFE0E0E0.toInt())
        saccadeToolBtn.setTextColor(if (tool == TraceView.TOOL_SACCADE) 0xFFFFFFFF.toInt() else 0xFF37474F.toInt())

        blinkToolBtn.setBackgroundColor(if (tool == TraceView.TOOL_BLINK) 0xFFC62828.toInt() else 0xFFE0E0E0.toInt())
        blinkToolBtn.setTextColor(if (tool == TraceView.TOOL_BLINK) 0xFFFFFFFF.toInt() else 0xFF37474F.toInt())

        eraserToolBtn.setBackgroundColor(if (tool == TraceView.TOOL_ERASER) 0xFF455A64.toInt() else 0xFFE0E0E0.toInt())
        eraserToolBtn.setTextColor(if (tool == TraceView.TOOL_ERASER) 0xFFFFFFFF.toInt() else 0xFF37474F.toInt())

        traceView.invalidate()
    }

    private fun applyWinMs() {
        val v = winMsEdit.text.toString().toLongOrNull() ?: return
        val ms = v.coerceIn(20L, 60_000L)
        epochMs = ms
        getSharedPreferences("uneye_prefs", MODE_PRIVATE).edit().putLong("winMs", ms).apply()
        if (pass == Session.Pass.ONE) { traceView.centerOn(traceView.fromSample + ms / 2, ms); refresh() }
    }

    private fun setupListeners() {
        menuBtn.setOnClickListener {
            settingsPanel.visibility = if (settingsPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        winMsEdit.setOnEditorActionListener { _, _, _ -> applyWinMs(); true }
        winMsEdit.setOnFocusChangeListener { _, hasFocus -> if (!hasFocus) applyWinMs() }
        undoBtn.setOnClickListener { session.undo(trial); refresh() }

        // Pass 1 Navigation
        prevEpochBtn.setOnClickListener { moveEpoch(-1) }
        nextEpochBtn.setOnClickListener { moveEpoch(+1) }
        prevTrialBtn.setOnClickListener {
            if (trial > 0) {
                session.setTrial(trial - 1)
                traceView.centerOn(epochMs / 2, epochMs)
                refresh()
            }
        }
        nextTrialBtn.setOnClickListener {
            if (trial < session.nTrials.value - 1) {
                session.setTrial(trial + 1)
                traceView.centerOn(epochMs / 2, epochMs)
                refresh()
            }
        }

        // Pass 2 Navigation & Actions
        prevEventBtn.setOnClickListener {
            if (refineIdx > 0) {
                refineIdx--
            } else {
                val pt = findTrialWithEvents(trial, -1)
                if (pt < 0) return@setOnClickListener
                enterTrialForRefine(pt, atEnd = true)
            }
            zoomMultiplier = 1.0
            focusCurrentMark()
        }
        nextEventBtn.setOnClickListener {
            if (refineIdx < refineQueue.size - 1) {
                refineIdx++
            } else {
                val nt = findTrialWithEvents(trial, +1)       // after the last event of a trial: first event of the next trial that has some
                if (nt < 0) return@setOnClickListener
                enterTrialForRefine(nt, atEnd = false)
            }
            zoomMultiplier = 1.0
            focusCurrentMark()
        }
        acceptPreBtn.setOnClickListener {
            if (currentMarkAt >= 0) {
                val onset = if (currentPutativeOnset >= 0) currentPutativeOnset else max(0, currentMarkAt - 15)
                val offset = if (currentPutativeOffset >= 0) currentPutativeOffset else min(session.trialSamples - 1, currentMarkAt + 15)
                session.refineMark(trial, currentMarkAt, onset, offset, currentMarkCls)
                refineIdx++
                zoomMultiplier = 1.0
                focusCurrentMark()
                refresh()
            }
        }
        notEventBtn.setOnClickListener {
            if (currentMarkAt >= 0) {
                session.deleteMark(trial, currentMarkAt)
                if (refineIdx < refineQueue.size) {
                    refineQueue.removeAt(refineIdx)
                }
                zoomMultiplier = 1.0
                focusCurrentMark()
                refresh()
            }
        }
        widerBtn.setOnClickListener {
            zoomMultiplier = when (zoomMultiplier) {
                1.0 -> 2.5
                2.5 -> 5.0
                else -> 1.0
            }
            focusCurrentMark()
            traceView.invalidate()
        }

        // Canvas gestures
        traceView.onTapSample = { s -> onTap(s) }
        traceView.onDoubleTapSample = { s -> onDoubleTap(s) }
        traceView.onDragSpan = { onset, offset -> onDragSpan(onset, offset) }
        traceView.onScrolled = { _, _ -> refreshWindow() }

        // Top & Bottom Action Buttons
        filterBtn.setOnClickListener { showFilterDialog() }
        switchPassBtn.setOnClickListener { switchPass() }
        saveBtn.setOnClickListener { saveLabels() }
        trainNowBtn.setOnClickListener {
            TrainingScheduler.scheduleNow(this)
            Toast.makeText(this, R.string.training_scheduled, Toast.LENGTH_SHORT).show()
        }
        metricsBtn.setOnClickListener { showMetricsDialog() }
        exportBtn.setOnClickListener { showExportDialog() }
    }

    private fun setupObservers() {
        session.trial.observeState(this) { refresh() }
        session.pass.observeState(this) { updatePassUi() }
        session.reviewedSeconds.observeState(this) { seconds ->
            val app = application as UneyeApp
            val newSeconds = seconds - session.reviewedAtLastTraining
            TrainingScheduler.maybeSchedule(this, seconds, newSeconds)
        }
        (application as UneyeApp).trainingState.observeState(this) { ui ->
            trainStatusText.text = when {
                ui.running -> getString(R.string.training_progress, ui.epoch, ui.trainLoss, ui.valLoss)
                ui.finished -> getString(R.string.training_done, ui.epoch)
                ui.failed -> getString(R.string.training_failed, ui.message)
                else -> getString(R.string.training_idle)
            }
            if (ui.finished || ui.failed) refreshConvergence()
        }
    }

    // --- Pass Switching & UI Setup ---------------------------------------------------

    private fun switchPass() {
        if (pass == Session.Pass.ONE) {
            // Enter Pass 2: the window on screen has been seen; the queue holds the marks of windows that were seen
            session.addScan(trial, traceView.fromSample, min(traceView.toSample, session.trialSamples - 1))
            refineQueue = queueFor(trial).toMutableList()
            refineIdx = 0
            zoomMultiplier = 1.0
            session.setPass(Session.Pass.TWO)
        } else {
            session.setPass(Session.Pass.ONE)
        }
        updatePassUi()
        refresh()
    }

    private fun updatePassUi() {
        val isTwo = pass == Session.Pass.TWO
        traceView.isPassTwo = isTwo

        if (isTwo) {
            passIndicator.text = getString(R.string.pass2_title)
            toolGroup.visibility = View.GONE
            eventCounterBadge.visibility = View.VISIBLE
            pass1Controls.visibility = View.GONE
            pass2Controls.visibility = View.VISIBLE

            if (refineQueue.isEmpty()) {
                refineQueue = queueFor(trial).toMutableList()
            }
            refineIdx = 0
            focusCurrentMark()
        } else {
            passIndicator.text = getString(R.string.pass1_title)
            toolGroup.visibility = View.VISIBLE
            eventCounterBadge.visibility = View.GONE
            pass1Controls.visibility = View.VISIBLE
            pass2Controls.visibility = View.GONE
            traceView.activePutativeRange = null
        }
    }

    // --- Pass 1 Gestures & Logic -----------------------------------------------------

    private fun onTap(s: Long) {
        if (pass != Session.Pass.ONE) return

        if (activeTool == TraceView.TOOL_ERASER) {
            // Eraser tool, tap: everything within 8 samples of the tap (a drag erases a whole interval)
            session.eraseRange(trial, s - 8, s + 8)
            refresh()
            return
        }

        // Tap on existing mark removes it
        if (session.removePointMark(trial, s)) {
            refresh()
            return
        }

        // Tap near a pre-label snaps to it
        val snapped = session.snapToPrelabel(trial, s, 30)
        val targetSample = if (snapped >= 0) snapped else s
        session.addPointMark(trial, targetSample, cls)
        lastMarkSample = targetSample

        // Auto-scroll: next epoch starts just after marked event (+30 ms saccade, +300 ms blink)
        val margin = if (cls == Native.CLS_SACCADE) 30L else 300L
        val winStep = (epochMs * 0.9).roundToInt()
        val nextFrom = ((lastMarkSample + margin) / winStep + 1) * winStep
        if (nextFrom > traceView.toSample) {
            session.addScan(trial, traceView.fromSample, min(traceView.toSample, session.trialSamples - 1))
            traceView.centerOn(nextFrom + epochMs / 2, epochMs)
        }
        refresh()
    }

    /**
     * Next / previous window of the same trial (10 % overlap); after the last window of a trial, the first window of the next one.
     * The window left with Next counts as seen: the visible range, or the whole epoch width when the view is at least that wide.
     */
    private fun moveEpoch(dir: Int) {
        val n = session.trialSamples
        val step = max(1L, (epochMs * 0.9).roundToInt().toLong())
        val cur = traceView.fromSample
        if (dir > 0) {
            val vis = traceView.toSample - cur + 1
            val seenTo = if (vis >= (epochMs * 0.9)) max(traceView.toSample, cur + epochMs - 1) else traceView.toSample
            session.addScan(trial, cur, min(n - 1, seenTo))
            if (cur + epochMs >= n) {                               // last window of the trial
                if (trial < session.nTrials.value - 1) {
                    session.setTrial(trial + 1)
                    traceView.centerOn(epochMs / 2, epochMs)
                }
            } else {
                val next = min(cur + step, max(0L, n - epochMs))    // the last window is clipped to the end of the trial
                traceView.centerOn(next + epochMs / 2, epochMs)
            }
        } else {
            if (cur <= 0) {
                if (trial > 0) {
                    session.setTrial(trial - 1)
                    val last = max(0L, n - epochMs)
                    traceView.centerOn(last + epochMs / 2, epochMs)
                }
            } else {
                val prev = max(0L, cur - step)
                traceView.centerOn(prev + epochMs / 2, epochMs)
            }
        }
        refresh()
    }

    // Pass 2 only offers the marks that lie in windows displayed in pass 1
    private fun queueFor(t: Long): List<Pair<Long, Int>> =
        session.trialUi(t).pointMarks.filter { session.inScanned(t, it.first) }.sortedBy { it.first }

    private fun findTrialWithEvents(from: Long, dir: Int): Long {
        var t = from + dir
        while (t >= 0 && t < session.nTrials.value) {
            if (queueFor(t).isNotEmpty()) return t
            t += dir
        }
        return -1L
    }

    private fun enterTrialForRefine(t: Long, atEnd: Boolean) {
        session.setTrial(t)
        refineQueue = queueFor(t).toMutableList()
        refineIdx = if (atEnd) max(0, refineQueue.size - 1) else 0
        zoomMultiplier = 1.0
    }

    // --- Pass 2: Focused Event-by-Event Centering -----------------------------------

    /**
     * Pass 2: Always center the window on one labeled event from Pass 1.
     * One event at a time.
     * Window: -50 ms before event (or putative onset) to +50 ms after event (or putative offset).
     */
    private fun focusCurrentMark() {
        if (refineQueue.isEmpty()) {
            val nt = findTrialWithEvents(trial, +1)
            if (nt >= 0) { enterTrialForRefine(nt, atEnd = false); focusCurrentMark(); return }
            currentMarkAt = -1L
            currentPutativeOnset = -1L
            currentPutativeOffset = -1L
            traceView.activePutativeRange = null
            eventCounterBadge.text = getString(R.string.no_events)
            traceView.invalidate()
            return
        }

        if (refineIdx < 0) refineIdx = 0
        if (refineIdx >= refineQueue.size) {
            // end of this trial: the next event is the first one of the next trial that has events
            val nt = findTrialWithEvents(trial, +1)
            if (nt >= 0) { enterTrialForRefine(nt, atEnd = false); focusCurrentMark(); return }
            currentMarkAt = -1L
            traceView.activePutativeRange = null
            eventCounterBadge.text = getString(R.string.all_reviewed)
            refresh()
            return
        }

        val (at, markCls) = refineQueue[refineIdx]
        currentMarkAt = at
        currentMarkCls = markCls
        traceView.activeClass = markCls

        val trialUi = session.trialUi(trial)

        // Find putative onset & offset from existing expert marks or prelabels
        val exp = trialUi.marks.find { (it.onset <= at && at <= it.offset) || abs(it.onset - at) <= 30 || abs(it.offset - at) <= 30 }
        val pre = trialUi.prelabels.find { (it.onset <= at && at <= it.offset) || abs(it.onset - at) <= 40 || abs(it.offset - at) <= 40 }

        var on = at
        var off = at
        if (exp != null) {
            on = exp.onset
            off = exp.offset
            currentMarkCls = exp.cls
        } else if (pre != null) {
            on = pre.onset
            off = pre.offset
            currentMarkCls = pre.cls
        }

        currentPutativeOnset = on
        currentPutativeOffset = off
        traceView.activePutativeRange = on to off

        // Strict window framing: -50 ms before event to +50 ms after event
        val pad50ms = (50.0 * zoomMultiplier).roundToInt().toLong()
        val winFrom = max(0, on - pad50ms)
        val winTo = min(session.trialSamples - 1, off + pad50ms)
        val center = (winFrom + winTo) / 2
        val span = max(100L, winTo - winFrom + 1)

        traceView.centerOn(center, span)

        val clsName = if (currentMarkCls == Native.CLS_SACCADE) getString(R.string.tool_saccade) else getString(R.string.tool_blink)
        val durMs = max(1L, off - on + 1)
        eventCounterBadge.text = getString(R.string.event_counter, refineIdx + 1, refineQueue.size, clsName, durMs)
    }

    private fun onDoubleTap(s: Long) {
        if (pass != Session.Pass.TWO || currentMarkAt < 0) return
        acceptPreBtn.performClick()
    }

    private fun onDragSpan(onset: Long, offset: Long) {
        if (activeTool == TraceView.TOOL_ERASER) {
            // eraser: everything between the two times goes (saccades, blinks, marks, pre-labels)
            session.eraseRange(trial, onset, offset)
            if (pass == Session.Pass.TWO) {
                refineQueue = queueFor(trial).toMutableList()
                refineIdx = min(refineIdx, refineQueue.size)
                focusCurrentMark()
            }
            refresh()
            return
        }
        if (pass != Session.Pass.TWO || currentMarkAt < 0) return
        if (session.refineMark(trial, currentMarkAt, onset, offset, currentMarkCls)) {
            refineIdx++
            zoomMultiplier = 1.0
            focusCurrentMark()
            refresh()
        } else {
            Toast.makeText(this, R.string.invalid_span, Toast.LENGTH_SHORT).show()
        }
    }

    // --- Filter Settings Dialog ------------------------------------------------------

    private fun showFilterDialog() {
        val view = layoutInflater.inflate(R.layout.dialog_filter, null)
        val sgWindowEdit = view.findViewById<EditText>(R.id.sgWindowEdit)
        val lambdaEdit = view.findViewById<EditText>(R.id.lambdaEdit)
        val minVelEdit = view.findViewById<EditText>(R.id.minVelEdit)
        val minAmpEdit = view.findViewById<EditText>(R.id.minAmpEdit)
        val resetBtn = view.findViewById<Button>(R.id.resetFilterBtn)

        sgWindowEdit.setText(filterSettings.windowMs.toString())
        lambdaEdit.setText(String.format(java.util.Locale.US, "%.1f", filterSettings.lambda))
        minVelEdit.setText(String.format(java.util.Locale.US, "%.1f", filterSettings.minPeakVel))
        minAmpEdit.setText(String.format(java.util.Locale.US, "%.2f", filterSettings.minAmplitude))

        resetBtn.setOnClickListener {
            val def = FilterSettings.DEFAULT
            sgWindowEdit.setText(def.windowMs.toString())
            lambdaEdit.setText(String.format(java.util.Locale.US, "%.1f", def.lambda))
            minVelEdit.setText(String.format(java.util.Locale.US, "%.1f", def.minPeakVel))
            minAmpEdit.setText(String.format(java.util.Locale.US, "%.2f", def.minAmplitude))
        }

        AlertDialog.Builder(this)
            .setTitle(R.string.filter_dialog_title)
            .setView(view)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.filter_apply) { _, _ ->
                var w = sgWindowEdit.text.toString().toIntOrNull() ?: FilterSettings.DEFAULT.windowMs
                if (w % 2 == 0) w += 1
                w = max(5, w)

                val lam = lambdaEdit.text.toString().toDoubleOrNull() ?: FilterSettings.DEFAULT.lambda
                val mVel = minVelEdit.text.toString().toDoubleOrNull() ?: FilterSettings.DEFAULT.minPeakVel
                val mAmp = minAmpEdit.text.toString().toDoubleOrNull() ?: FilterSettings.DEFAULT.minAmplitude

                filterSettings = FilterSettings(
                    enabled = true,
                    windowMs = w,
                    polyOrder = 3,
                    lambda = max(1.0, lam),
                    minPeakVel = max(10.0, mVel),
                    minAmplitude = max(0.01, mAmp)
                )
                FilterSettings.save(this, filterSettings)
                refresh()
                Toast.makeText(this, "Filter applied (SG window: ${w}ms, λ=${lam})", Toast.LENGTH_SHORT).show()
            }
            .show()
    }

    // --- Data Rendering & Kinematics Engine -------------------------------------------

    private fun refresh() {
        val k = trial
        if (!session.loaded.value) return

        Thread {
            val uiState = session.trialUi(k)
            val from = traceView.fromSample
            val to = min(traceView.toSample, session.trialSamples - 1)
            val (rawX, rawY) = session.trace(k, from, to)
            val n = rawX.size

            val invalid = BooleanArray(n) { !rawX[it].isFinite() || !rawY[it].isFinite() }

            // Compute Savitzky-Golay cubic kinematics & adaptive threshold
            val kin = Kinematics.computeKinematics(
                rawX, rawY, invalid, 1000.0, session.degPerUnit, filterSettings
            )

            val lab = session.labels(k, from, to, true)
            val reviewed = BooleanArray(n) { lab[it].toInt() and 0xFF != 255 }

            val nMarks = uiState.pointMarks.size
            val nExpert = uiState.marks.count { !it.fromPrelabel }

            ui.post {
                traceView.degPerUnit = session.degPerUnit
                traceView.currentThreshold = kin.threshold
                traceView.thresholdLambda = filterSettings.lambda

                traceView.setWindow(
                    from, to,
                    TraceView.Lane(kin.smoothX, invalid),
                    TraceView.Lane(kin.smoothY, invalid),
                    TraceView.Lane(kin.speed)
                )
                traceView.labelBand = lab
                traceView.reviewedBand = reviewed
                traceView.prelabels = uiState.prelabels.map { it.onset to it.offset }
                traceView.prelabelCls = uiState.prelabels.map { it.cls }
                traceView.pointMarks = uiState.pointMarks
                traceView.expertMarks = uiState.marks.map { Triple(it.onset, it.offset, it.cls) }
                traceView.invalidate()

                statusText.text = getString(
                    R.string.status_marks,
                    k + 1, session.nTrials.value,
                    nExpert, nMarks, session.reviewedSeconds.value
                )
            }
        }.start()
    }

    private fun refreshWindow() {
        refresh()
    }

    private fun saveLabels() {
        val path = (application as UneyeApp).labelsPathFor(session, trial)
        val ok = session.saveLabels(path)
        Toast.makeText(this, if (ok) R.string.labels_saved else R.string.labels_save_failed, Toast.LENGTH_SHORT).show()
    }

    // --- Metrics & Convergence -------------------------------------------------------

    private fun refreshConvergence() {
        val app = application as UneyeApp
        val rounds = app.convergenceStore.rounds()
        val flag = ConvergenceFlag.evaluate(rounds)
        when (flag.state) {
            ConvergenceFlag.State.SILENT -> {
                convergenceBadge.visibility = View.GONE
                convergenceBadge.text = ""
            }
            ConvergenceFlag.State.COLLECTING -> {
                convergenceBadge.visibility = View.VISIBLE
                convergenceBadge.setBackgroundColor(0xFFFFB300.toInt())
                convergenceBadge.text = getString(
                    R.string.convergence_collecting,
                    flag.nSaccades, ConvergenceFlag.MIN_SACCADES, flag.nBlinks, ConvergenceFlag.MIN_BLINKS
                )
            }
            ConvergenceFlag.State.CONVERGED -> {
                convergenceBadge.visibility = View.VISIBLE
                convergenceBadge.setBackgroundColor(0xFF2E7D32.toInt())
                convergenceBadge.text = getString(
                    R.string.convergence_converged,
                    flag.f1Saccade, flag.kappaSaccade, flag.f1Blink, flag.kappaBlink
                )
                if (!convergenceAlertShown) {
                    convergenceAlertShown = true
                    AlertDialog.Builder(this)
                        .setTitle(R.string.convergence_alert_title)
                        .setMessage(getString(R.string.convergence_alert_message, flag.f1Saccade, flag.kappaSaccade))
                        .setPositiveButton(android.R.string.ok, null)
                        .show()
                }
            }
        }
    }

    private fun showMetricsDialog() {
        val app = application as UneyeApp
        val rounds = app.convergenceStore.rounds()
        val flag = ConvergenceFlag.evaluate(rounds)
        val view = MetricsView(this)
        view.setSeries(rounds, flag.state == ConvergenceFlag.State.CONVERGED)
        val heightPx = (360 * resources.displayMetrics.density).toInt()
        AlertDialog.Builder(this)
            .setTitle(R.string.metrics_title)
            .setView(view)
            .setPositiveButton(android.R.string.ok, null)
            .show()
        view.post {
            view.layoutParams = android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT, heightPx
            )
            view.requestLayout()
        }
    }

    // --- Export Dialog ---------------------------------------------------------------

    private fun showExportDialog() {
        if (!session.loaded.value) return
        val baseName = session.recordingName.ifBlank { "recording" }
        val hasModel = session.lastTrainedModel().let { it.isNotEmpty() && File(it).exists() }
        val dialog = layoutInflater.inflate(R.layout.dialog_export, null)
        val matBox = dialog.findViewById<CheckBox>(R.id.matBox)
        val modelBox = dialog.findViewById<CheckBox>(R.id.modelBox)
        val shareBox = dialog.findViewById<CheckBox>(R.id.shareBox)
        modelBox.isEnabled = hasModel
        if (!hasModel) modelBox.isChecked = false

        AlertDialog.Builder(this)
            .setTitle(R.string.export_dialog_title)
            .setView(dialog)
            .setNegativeButton(android.R.string.cancel, null)
            .setNeutralButton(R.string.export_pick_destination) { _, _ ->
                if (matBox.isChecked) {
                    exportHelper.exportTo("application/octet-stream", "$baseName.mat",
                        write = { out -> Exporter.writeMat(out, session) },
                        onDone = { ok, uri -> toastExportResult(ok, uri) })
                } else {
                    exportHelper.exportTo("text/csv", "${baseName}_L.csv",
                        write = { out -> Exporter.writeLabelsCsv(out, Exporter.labelsMatrix(session)) },
                        onDone = { ok, uri -> toastExportResult(ok, uri) })
                }
            }
            .setPositiveButton(R.string.export) { _, _ ->
                val includeMat = matBox.isChecked
                val includeModel = modelBox.isChecked && hasModel
                val share = shareBox.isChecked
                Toast.makeText(this, R.string.export_running, Toast.LENGTH_SHORT).show()
                lifecycleScope.launch {
                    val root = getExternalFilesDir("exports") ?: filesDir
                    val dir = File(root, baseName).apply { mkdirs() }
                    val files = exportHelper.exportAll(session, dir, baseName, includeMat, includeModel)
                    if (files.isEmpty()) {
                        Toast.makeText(this@LabelingScreen, R.string.export_failed, Toast.LENGTH_LONG).show()
                        return@launch
                    }
                    if (share) {
                        exportHelper.shareFiles(files, listOf("text/csv", "application/octet-stream", "application/json"))
                    }
                    Toast.makeText(this@LabelingScreen, getString(R.string.export_done, dir.absolutePath), Toast.LENGTH_LONG).show()
                }
            }
            .show()
    }

    private fun toastExportResult(ok: Boolean, uri: Uri?) {
        val msg = when {
            uri == null -> getString(R.string.export_cancelled)
            ok -> getString(R.string.export_done, uri.toString())
            else -> getString(R.string.export_failed)
        }
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        session.saveLabels((application as UneyeApp).labelsPathFor(session, trial))
    }
}

/** Observes a StateFlow on the main thread for the activity's lifetime. */
private fun <T> kotlinx.coroutines.flow.StateFlow<T>.observeState(
    activity: AppCompatActivity, block: (T) -> Unit,
) {
    activity.lifecycleScope.launch {
        collect { block(it) }
    }
}
