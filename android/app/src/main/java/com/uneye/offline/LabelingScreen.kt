package com.uneye.offline

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

/**
 * The two-pass labeling screen (phase 2).
 *
 * Pass 1 (tap + auto-scroll): a class toggle (saccade / blink), one tap places
 * a point mark (a tap on an existing mark removes it), pre-labels are dimmed
 * and a tap near one snaps to it. After each mark the view auto-scrolls so
 * the next epoch starts just after the last marked event (+30 ms for a
 * saccade, +300 ms for a blink). A large "Next" button (or a swipe) moves to
 * the next epoch when it has no event. When the epoch is scanned and all its
 * marks are refined (pass 2), the facade marks it reviewed.
 *
 * Pass 2 (press-drag-lift): for each mark, in order, the view centers a
 * window around it (-50..+50 ms for saccades, -300..+300 ms for blinks,
 * widened when the pre-label is longer). Press = onset, drag, lift = offset;
 * the app then moves to the next mark. Buttons: "not an event" (deletes the
 * mark), "back", "wider" (zoom out). A double tap accepts the pre-label
 * boundaries as they are.
 *
 * The screen loads the visible window from the facade on a worker thread and
 * keeps the TraceView fed; all edits go through the facade.
 */
class LabelingScreen : AppCompatActivity() {

    private lateinit var session: Session
    private lateinit var traceView: TraceView
    private lateinit var classToggle: Button
    private lateinit var undoBtn: Button
    private lateinit var nextBtn: Button
    private lateinit var passIndicator: TextView
    private lateinit var statusText: TextView
    private lateinit var refineButtons: View
    private lateinit var notEventBtn: Button
    private lateinit var backBtn: Button
    private lateinit var widerBtn: Button

    private var epochMs = 500L            // epoch length (setting): 200 or 500 ms
    private var cls = Native.CLS_SACCADE  // the class chosen for pass-1 taps

    private val trial: Long get() = session.trial.value
    private val pass: Session.Pass get() = session.pass.value

    // pass-1 state
    private var lastMarkSample = -1L
    private var epochScanned = false

    // pass-2 state: the queue of marks to refine
    private var refineQueue: List<Pair<Long, Int>> = emptyList()   // (at, cls)
    private var refineIdx = 0
    private var currentMarkAt = -1L
    private var currentMarkCls = 1
    private val ui = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_labeling)
        session = (application as UneyeApp).session
        traceView = findViewById(R.id.traceView)
        classToggle = findViewById(R.id.classToggle)
        undoBtn = findViewById(R.id.undoBtn)
        nextBtn = findViewById(R.id.nextBtn)
        passIndicator = findViewById(R.id.passIndicator)
        statusText = findViewById(R.id.statusText)
        refineButtons = findViewById(R.id.refineButtons)
        notEventBtn = findViewById(R.id.notEventBtn)
        backBtn = findViewById(R.id.backBtn)
        widerBtn = findViewById(R.id.widerBtn)

        classToggle.setOnClickListener {
            cls = if (cls == Native.CLS_SACCADE) Native.CLS_BLINK else Native.CLS_SACCADE
            classToggle.text = if (cls == Native.CLS_SACCADE) "Saccade" else "Clignement"
            classToggle.setBackgroundColor(
                if (cls == Native.CLS_SACCADE) 0xFF1565C0.toInt() else 0xFFC62828.toInt())
        }
        classToggle.text = "Saccade"
        classToggle.setBackgroundColor(0xFF1565C0.toInt())

        undoBtn.setOnClickListener { session.undo(trial); refresh() }

        nextBtn.setOnClickListener { moveEpoch(+1) }

        notEventBtn.setOnClickListener {
            if (currentMarkAt >= 0) {
                session.deleteMark(trial, currentMarkAt)
                advanceRefineQueue()
            }
        }
        backBtn.setOnClickListener {
            if (refineIdx > 0) {
                --refineIdx
                showCurrentMark()
            }
        }
        widerBtn.setOnClickListener {
            val span = traceView.toSample - traceView.fromSample + 1
            traceView.centerOn(currentMarkAt, span * 2)
            refreshWindow()
        }

        traceView.onTapSample = { s -> onTap(s) }
        traceView.onDoubleTapSample = { s -> onDoubleTap(s) }
        traceView.onDragSpan = { onset, offset -> onDragSpan(onset, offset) }
        traceView.onScrolled = { _, _ -> refreshWindow() }

        findViewById<Button>(R.id.switchPassBtn).setOnClickListener { switchPass() }
        findViewById<Button>(R.id.saveBtn).setOnClickListener { saveLabels() }
        findViewById<Button>(R.id.trainNowBtn).setOnClickListener {
            TrainingScheduler.scheduleNow(this)
            Toast.makeText(this, R.string.training_scheduled, Toast.LENGTH_SHORT).show()
        }

        session.trial.observeState(this) { refresh() }
        session.pass.observeState(this) { updatePassUi() }
        session.reviewedSeconds.observeState(this) { seconds ->
            statusText.text = getString(R.string.reviewed_seconds, seconds)
            // phase 3: schedule a background training round after every N newly
            // labeled seconds
            val app = application as UneyeApp
            val newSeconds = seconds - session.reviewedAtLastTraining
            TrainingScheduler.maybeSchedule(this, seconds, newSeconds)
        }
        (application as UneyeApp).trainingState.observeState(this) { ui ->
            val trainStatus = findViewById<TextView>(R.id.trainStatusText)
            trainStatus.text = if (ui.running) {
                getString(R.string.training_progress, ui.epoch, ui.trainLoss, ui.valLoss)
            } else if (ui.finished) {
                getString(R.string.training_done, ui.epoch)
            } else if (ui.failed) {
                getString(R.string.training_failed, ui.message)
            } else {
                getString(R.string.training_idle)
            }
            // phase 4: a finished round appends to the convergence history ->
            // re-evaluate the flag and refresh the badge / metrics
            if (ui.finished || ui.failed) refreshConvergence()
        }

        findViewById<Button>(R.id.metricsBtn).setOnClickListener { showMetricsDialog() }

        updatePassUi()
        refresh()
        refreshConvergence()
    }

    // --- phase 4: convergence flag + metrics -------------------------------------------

    private var convergenceAlertShown = false

    private fun refreshConvergence() {
        val app = application as UneyeApp
        val rounds = app.convergenceStore.rounds()
        val flag = ConvergenceFlag.evaluate(rounds)
        val badge = findViewById<TextView>(R.id.convergenceBadge)
        when (flag.state) {
            ConvergenceFlag.State.SILENT -> {
                badge.visibility = View.GONE
                badge.text = ""
            }
            ConvergenceFlag.State.COLLECTING -> {
                badge.visibility = View.VISIBLE
                badge.setBackgroundColor(0xFFFFB300.toInt())
                badge.text = getString(R.string.convergence_collecting,
                    flag.nSaccades, ConvergenceFlag.MIN_SACCADES, flag.nBlinks, ConvergenceFlag.MIN_BLINKS)
            }
            ConvergenceFlag.State.CONVERGED -> {
                badge.visibility = View.VISIBLE
                badge.setBackgroundColor(0xFF2E7D32.toInt())
                badge.text = getString(R.string.convergence_converged,
                    flag.f1Saccade, flag.kappaSaccade, flag.f1Blink, flag.kappaBlink)
                if (!convergenceAlertShown) {
                    convergenceAlertShown = true
                    AlertDialog.Builder(this)
                        .setTitle(R.string.convergence_alert_title)
                        .setMessage(getString(R.string.convergence_alert_message,
                            flag.f1Saccade, flag.kappaSaccade))
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
            .setView(view, 0, 0, 0, 0)
            .setPositiveButton(android.R.string.ok, null)
            .show()
        // the dialog measures the view after show(); force a fixed height so
        // onDraw always has a canvas
        view.post {
            view.layoutParams = android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.MATCH_PARENT, heightPx)
            view.requestLayout()
        }
    }

    // --- pass switching ---------------------------------------------------------------

    private fun switchPass() {
        if (pass == Session.Pass.ONE) {
            // entering pass 2: build the refinement queue from the point marks
            val pm = session.trialUi(trial).pointMarks
            refineQueue = pm.sortedBy { it.first }
            refineIdx = 0
            session.setPass(Session.Pass.TWO)
        } else {
            session.setPass(Session.Pass.ONE)
        }
        updatePassUi()
        refresh()
    }

    private fun updatePassUi() {
        val two = pass == Session.Pass.TWO
        passIndicator.text = if (two) "Pass 2 : onset / offset" else "Pass 1 : marquer"
        classToggle.visibility = if (two) View.GONE else View.VISIBLE
        nextBtn.visibility = if (two) View.GONE else View.VISIBLE
        refineButtons.visibility = if (two) View.VISIBLE else View.GONE
        if (two && refineQueue.isEmpty()) {
            // no marks to refine on this trial: pull them from the facade
            refineQueue = session.trialUi(trial).pointMarks.sortedBy { it.first }
        }
        if (two) showCurrentMark() else refresh()
    }

    // --- pass 1 -------------------------------------------------------------------------

    private fun onTap(s: Long) {
        if (pass != Session.Pass.ONE) return
        // a tap on an existing point mark removes it
        if (session.removePointMark(trial, s)) { refresh(); return }
        // a tap near a pre-label snaps to it
        val snapped = session.snapToPrelabel(trial, s, 30)
        session.addPointMark(trial, if (snapped >= 0) snapped else s, cls)
        lastMarkSample = if (snapped >= 0) snapped else s
        // auto-scroll: the next epoch starts just after the last marked event
        val margin = if (cls == Native.CLS_SACCADE) 30L else 300L
        val epochSamples = epochMs   // 1 sample = 1 ms at 1 kHz
        val nextFrom = ((lastMarkSample + margin) / epochSamples + 1) * epochSamples
        if (nextFrom > traceView.toSample) {
            traceView.centerOn(nextFrom + epochSamples / 2, epochSamples)
            epochScanned = false
        }
        refresh()
    }

    private fun moveEpoch(dir: Int) {
        val epochSamples = epochMs
        val cur = traceView.fromSample
        val next = (cur + dir * epochSamples).coerceIn(0, maxOf(0, session.trialSamples - 1))
        // the epoch we leave is scanned
        val leaveTo = minOf(cur + epochSamples - 1, session.trialSamples - 1)
        session.markEpochReviewed(trial, cur, leaveTo)
        epochScanned = true
        traceView.centerOn(next + epochSamples / 2, epochSamples)
        refresh()
    }

    // --- pass 2 ---------------------------------------------------------------------------

    private fun onDoubleTap(s: Long) {
        if (pass != Session.Pass.TWO || currentMarkAt < 0) return
        // accept the pre-label boundaries as they are
        val snapped = session.snapToPrelabel(trial, currentMarkAt, 60)
        if (snapped >= 0) {
            val pre = session.trialUi(trial).prelabels.firstOrNull { it.onset == snapped }
            if (pre != null && session.refineMark(trial, currentMarkAt, pre.onset, pre.offset, currentMarkCls)) {
                advanceRefineQueue()
            }
        }
    }

    private fun onDragSpan(onset: Long, offset: Long) {
        if (pass != Session.Pass.TWO || currentMarkAt < 0) return
        if (session.refineMark(trial, currentMarkAt, onset, offset, currentMarkCls)) {
            advanceRefineQueue()
        } else {
            Toast.makeText(this, R.string.invalid_span, Toast.LENGTH_SHORT).show()
        }
    }

    private fun showCurrentMark() {
        if (refineIdx >= refineQueue.size) {
            // all marks refined: the epoch can be reviewed
            val from = traceView.fromSample
            val to = minOf(from + epochMs - 1, session.trialSamples - 1)
            session.markEpochReviewed(trial, from, to)
            Toast.makeText(this, R.string.all_marks_refined, Toast.LENGTH_SHORT).show()
            refresh()
            return
        }
        val (at, markCls) = refineQueue[refineIdx]
        currentMarkAt = at
        currentMarkCls = markCls
        // window: -50..+50 ms for saccades, -300..+300 ms for blinks, widened
        // when the pre-label is longer
        val half = if (markCls == Native.CLS_BLINK) 300L else 50L
        val pre = session.trialUi(trial).prelabels.firstOrNull { it.onset <= at && at <= it.offset }
        val span = if (pre != null) maxOf(2 * half + 1, pre.offset - pre.onset + 2 * half + 1) else 2 * half + 1
        traceView.centerOn(at, span)
        refresh()
    }

    private fun advanceRefineQueue() {
        ++refineIdx
        showCurrentMark()
    }

    // --- data -----------------------------------------------------------------------------

    private fun refresh() {
        val k = trial
        if (!session.loaded.value) return
        Thread {
            val ui = session.trialUi(k)
            val from = traceView.fromSample
            val to = minOf(traceView.toSample, session.trialSamples - 1)
            val (x, y) = session.trace(k, from, to)
            val n = x.size
            val invalid = BooleanArray(n) { !x[it].isFinite() || !y[it].isFinite() }
            val speed = FloatArray(n) { i ->
                if (i == 0 || invalid[i] || invalid[i - 1]) 0f
                else kotlin.math.hypot((x[i] - x[i - 1]).toDouble(), (y[i] - y[i - 1]).toDouble()).toFloat()
            }
            val lab = session.labels(k, from, to, true)
            val reviewed = BooleanArray(n) { lab[it].toInt() and 0xFF != 255 }
            uiHandlerPost {
                traceView.setWindow(from, to,
                    TraceView.Lane(x, invalid), TraceView.Lane(y, invalid), TraceView.Lane(speed))
                traceView.labelBand = lab
                traceView.reviewedBand = reviewed
                traceView.prelabels = ui.prelabels.map { it.onset to it.offset }
                traceView.prelabelCls = ui.prelabels.map { it.cls }
                traceView.pointMarks = ui.pointMarks
                traceView.expertMarks = ui.marks.map { Triple(it.onset, it.offset, it.cls) }
                traceView.invalidate()
                val nMarks = ui.pointMarks.size
                val nExpert = ui.marks.count { !it.fromPrelabel }
                statusText.text = getString(R.string.status_marks, k + 1, session.nTrials.value,
                    nExpert, nMarks, session.reviewedSeconds.value)
            }
        }.start()
    }

    private fun refreshWindow() {
        // called after a pan/zoom: only the window-dependent data changes
        refresh()
    }

    private inline fun uiHandlerPost(crossinline block: () -> Unit) {
        ui.post { block() }
    }

    private fun saveLabels() {
        val path = (application as UneyeApp).labelsPathFor(session, trial)
        val ok = session.saveLabels(path)
        Toast.makeText(this, if (ok) R.string.labels_saved else R.string.labels_save_failed,
            Toast.LENGTH_SHORT).show()
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
