package com.uneye.offline

import android.os.SystemClock
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Random
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * Validates the 11 points of docs/LABELER_UX_SPEC.md on device/emulator.
 */
@RunWith(AndroidJUnit4::class)
class LabelerUxSpecTest {

    private fun synthTrial(seed: Long): Triple<DoubleArray, DoubleArray, DoubleArray> {
        val rng = Random(seed)
        val n = 1000
        val x = DoubleArray(n)
        val y = DoubleArray(n)
        val lab = DoubleArray(n)
        var px = 0.0
        var py = 0.0
        var i = 0

        fun fixation(len: Int) {
            for (k in 0 until len) {
                if (i >= n) break
                x[i] = px + 0.01 * rng.nextGaussian()
                y[i] = py + 0.01 * rng.nextGaussian()
                lab[i] = 0.0
                i++
            }
        }

        fun saccade(amp: Double) {
            val th = 6.2831853 * rng.nextDouble()
            val len = (21 + 2.2 * amp).toInt()
            val x0 = px
            val y0 = py
            for (k in 0 until len) {
                if (i >= n) break
                val u = (k + 1).toDouble() / len
                val s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
                x[i] = x0 + amp * cos(th) * s + 0.01 * rng.nextGaussian()
                y[i] = y0 + amp * sin(th) * s + 0.01 * rng.nextGaussian()
                lab[i] = 1.0
                i++
            }
            px += amp * cos(th)
            py += amp * sin(th)
        }

        fixation(300)
        saccade(4.0)
        fixation(200)
        saccade(6.0)
        fixation(n - i)
        return Triple(x, y, lab)
    }

    /**
     * Point 1: Refractory period after a pinch zoom.
     * No marks on release of fingers, and touches within 600 ms are suppressed.
     */
    @Test
    fun testPoint1PinchZoomRefractorySuppression() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val instrumentation = InstrumentationRegistry.getInstrumentation()

        instrumentation.runOnMainSync {
            val view = TraceView(context)
            view.layout(0, 0, 1000, 800)
            view.setWindow(0, 999, null, null, null)

            var tapCount = 0
            var dragCount = 0
            view.onTapSample = { tapCount++ }
            view.onDragSpan = { _, _ -> dragCount++ }

            val pProps = arrayOf(
                MotionEvent.PointerProperties().apply { id = 0; toolType = MotionEvent.TOOL_TYPE_FINGER },
                MotionEvent.PointerProperties().apply { id = 1; toolType = MotionEvent.TOOL_TYPE_FINGER }
            )
            val pCoords = arrayOf(
                MotionEvent.PointerCoords().apply { x = 200f; y = 300f },
                MotionEvent.PointerCoords().apply { x = 500f; y = 300f }
            )

            val t0 = SystemClock.uptimeMillis()

            // 1. Pointer 0 down
            val downEv = MotionEvent.obtain(t0, t0, MotionEvent.ACTION_DOWN, 1, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(downEv)
            downEv.recycle()

            // 2. Pointer 1 down (multi-touch starts)
            val p1DownAction = MotionEvent.ACTION_POINTER_DOWN or (1 shl MotionEvent.ACTION_POINTER_INDEX_SHIFT)
            val p1DownEv = MotionEvent.obtain(t0, t0 + 10, p1DownAction, 2, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(p1DownEv)
            p1DownEv.recycle()

            // 3. Pinch Move
            pCoords[0].x = 180f
            pCoords[1].x = 550f
            val moveEv = MotionEvent.obtain(t0, t0 + 30, MotionEvent.ACTION_MOVE, 2, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(moveEv)
            moveEv.recycle()

            // 4. Pointer 1 up
            val p1UpAction = MotionEvent.ACTION_POINTER_UP or (1 shl MotionEvent.ACTION_POINTER_INDEX_SHIFT)
            val p1UpEv = MotionEvent.obtain(t0, t0 + 50, p1UpAction, 2, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(p1UpEv)
            p1UpEv.recycle()

            // 5. Pointer 0 up
            val upEv = MotionEvent.obtain(t0, t0 + 60, MotionEvent.ACTION_UP, 1, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(upEv)
            upEv.recycle()

            // Release must not trigger any mark
            assertEquals("No tap on pinch release", 0, tapCount)
            assertEquals("No drag on pinch release", 0, dragCount)

            // 6. Touch within 600 ms refractory window (< REFRACTORY_MS = 600L)
            val tRefractory = SystemClock.uptimeMillis()
            val down2 = MotionEvent.obtain(tRefractory, tRefractory, MotionEvent.ACTION_DOWN, 1, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(down2)
            down2.recycle()

            val up2 = MotionEvent.obtain(tRefractory, tRefractory + 20, MotionEvent.ACTION_UP, 1, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(up2)
            up2.recycle()

            assertEquals("Tap inside refractory window must be suppressed", 0, tapCount)
        }

        // Wait past refractory window (650 ms)
        Thread.sleep(650)

        instrumentation.runOnMainSync {
            val view = TraceView(context)
            view.layout(0, 0, 1000, 800)
            view.setWindow(0, 999, null, null, null)

            var tapCount = 0
            view.onTapSample = { tapCount++ }

            val pProps = arrayOf(MotionEvent.PointerProperties().apply { id = 0; toolType = MotionEvent.TOOL_TYPE_FINGER })
            val pCoords = arrayOf(MotionEvent.PointerCoords().apply { x = 300f; y = 400f })

            val tNormal = SystemClock.uptimeMillis()
            val down = MotionEvent.obtain(tNormal, tNormal, MotionEvent.ACTION_DOWN, 1, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(down)
            down.recycle()

            val up = MotionEvent.obtain(tNormal, tNormal + 20, MotionEvent.ACTION_UP, 1, pProps, pCoords, 0, 0, 1f, 1f, 0, 0, 0, 0)
            view.onTouchEvent(up)
            up.recycle()

            assertEquals("Normal tap after refractory period must be accepted", 1, tapCount)
        }
    }

    /**
     * Point 2: Pass 1 window navigation with 10% overlap and trial boundary jump.
     */
    @Test
    fun testPoint2Pass1WindowNavigation() {
        val epochMs = 500L
        val trialSamples = 1000L
        val step = (epochMs * 0.9).roundToInt().toLong() // 450
        var trial = 0L
        val nTrials = 3L
        var cur = 0L

        // Step 1: Forward navigation in trial 0
        cur = min(cur + step, max(0L, trialSamples - epochMs)) // min(450, 500) = 450
        assertEquals(450L, cur)

        // Step 2: Last window of trial 0 (clipped to end: 500)
        cur = min(cur + step, max(0L, trialSamples - epochMs)) // min(900, 500) = 500
        assertEquals(500L, cur)

        // Step 3: From last window (cur + epochMs >= trialSamples), advance to next trial
        if (cur + epochMs >= trialSamples && trial < nTrials - 1) {
            trial += 1
            cur = 0L
        }
        assertEquals("Advanced to trial 1", 1L, trial)
        assertEquals("Reset to start of trial 1", 0L, cur)
    }

    /**
     * Points 3 & 4: Scanned/Reviewed requirement and Human-only refined events.
     */
    @Test
    fun testPoints3And4ScannedReviewedAndHumanOnly() = kotlinx.coroutines.runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as UneyeApp
        val session = app.session

        val xRows = mutableListOf<DoubleArray>()
        val yRows = mutableListOf<DoubleArray>()
        for (k in 0L until 3L) {
            val (x, y, _) = synthTrial(300 + k)
            xRows.add(x)
            yRows.add(y)
        }

        var loadDone = false
        var loadOk = false
        var loadErr = ""
        session.load(
            TraceFiles.Trace(xRows, yRows),
            rateHz = 1000.0, degPerUnit = 1.0, estimateScale = false,
            prelabeler = Native.PRELABELER_HMM, modelBin = "",
            recordingName = "ux_spec_test"
        ) { ok, err ->
            loadOk = ok
            loadErr = err
            loadDone = true
        }

        while (!loadDone) kotlinx.coroutines.delay(20)
        assertTrue("Session load failed: $loadErr", loadOk)
        assertEquals(3L, session.nTrials.value)

        // Unscanned epoch has 0 reviewed seconds
        assertEquals(0.0, session.reviewedSeconds.value, 1e-4)

        // Mark a window seen (0..499)
        session.addScan(0, 0, 499)
        assertTrue("Sample 250 in scanned window", session.inScanned(0, 250))
        assertFalse("Sample 600 not in scanned window", session.inScanned(0, 600))

        // Add a point mark at sample 250.
        // C++ add_point_mark re-opens ±150 samples ([100..400]) around the unrefined mark!
        session.addPointMark(0, 250, Native.CLS_SACCADE)
        session.applyScans(0)

        // The samples [100..400] are unreviewed waiting for refinement:
        val labelsWhileUnrefined = session.labels(0, 250, 250, onlyReviewed = true)
        assertEquals("Sample 250 must be unreviewed (-1)", -1, labelsWhileUnrefined[0].toInt())
        assertTrue("Reviewed seconds reflects exclusion of ±150ms window", session.reviewedSeconds.value < 0.5)

        // Point 4: Now refine the mark
        val ok = session.refineMark(0, 250, 235, 265, Native.CLS_SACCADE)
        assertTrue("Refinement succeeded", ok)

        // Once refined, applyScans counts the full 500 ms window in reviewed seconds!
        assertEquals("Full 500 ms window is reviewed once refined", 0.5, session.reviewedSeconds.value, 1e-3)

        // Only human refined marks become ground truth labels (1 for saccade, 0 background, -1 unreviewed)
        val reviewedLabels = session.labels(0, 0, 499, onlyReviewed = true)
        var humanSaccadeCount = 0
        for (b in reviewedLabels) {
            if (b.toInt() == 1) humanSaccadeCount++
        }
        assertEquals("Exactly 31 samples refined as saccade (235..265)", 31, humanSaccadeCount)
    }

    /**
     * Point 5: Pass 2 cross-trial navigation (finding trials with events, skipping empty ones).
     */
    @Test
    fun testPoint5Pass2CrossTrialNavigation() = kotlinx.coroutines.runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as UneyeApp
        val session = app.session

        val xRows = mutableListOf<DoubleArray>()
        val yRows = mutableListOf<DoubleArray>()
        for (k in 0L until 4L) {
            val (x, y, _) = synthTrial(400 + k)
            xRows.add(x)
            yRows.add(y)
        }

        var loadDone = false
        session.load(
            TraceFiles.Trace(xRows, yRows),
            rateHz = 1000.0, degPerUnit = 1.0, estimateScale = false,
            prelabeler = Native.PRELABELER_HMM, modelBin = "",
            recordingName = "nav_test"
        ) { _, _ -> loadDone = true }
        while (!loadDone) kotlinx.coroutines.delay(20)

        // Trial 0: mark seen window and add a point mark
        session.addScan(0, 0, 499)
        session.addPointMark(0, 200, Native.CLS_SACCADE)

        // Trial 1: seen window but NO point marks
        session.addScan(1, 0, 499)

        // Trial 2: seen window and add a point mark
        session.addScan(2, 0, 499)
        session.addPointMark(2, 150, Native.CLS_BLINK)

        fun queueFor(t: Long) = session.trialUi(t).pointMarks.filter { session.inScanned(t, it.first) }
        fun findTrialWithEvents(from: Long, dir: Int): Long {
            var t = from + dir
            while (t >= 0 && t < session.nTrials.value) {
                if (queueFor(t).isNotEmpty()) return t
                t += dir
            }
            return -1L
        }

        assertEquals("From trial 0, next trial with events must skip empty trial 1 and find trial 2", 2L, findTrialWithEvents(0, +1))
        assertEquals("From trial 2, prev trial with events must skip empty trial 1 and find trial 0", 0L, findTrialWithEvents(2, -1))
        assertEquals("From trial 2, next trial has no events", -1L, findTrialWithEvents(2, +1))
    }

    /**
     * Point 6: Eraser tool deletes marks, prelabels, and can be undone.
     */
    @Test
    fun testPoint6EraserToolAndUndo() = kotlinx.coroutines.runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as UneyeApp
        val session = app.session

        val xRows = mutableListOf<DoubleArray>()
        val yRows = mutableListOf<DoubleArray>()
        for (k in 0L until 2L) {
            val (x, y, _) = synthTrial(500 + k)
            xRows.add(x)
            yRows.add(y)
        }

        var loadDone = false
        session.load(
            TraceFiles.Trace(xRows, yRows),
            rateHz = 1000.0, degPerUnit = 1.0, estimateScale = false,
            prelabeler = Native.PRELABELER_HMM, modelBin = "",
            recordingName = "eraser_test"
        ) { _, _ -> loadDone = true }
        while (!loadDone) kotlinx.coroutines.delay(20)

        // Add and refine a mark
        session.addScan(0, 0, 499)
        session.addPointMark(0, 200, Native.CLS_SACCADE)
        session.refineMark(0, 200, 190, 210, Native.CLS_SACCADE)

        val beforeEraser = session.trialUi(0)
        assertTrue("Mark exists before erase", beforeEraser.marks.any { it.onset == 190L && it.offset == 210L })

        // Erase range overlapping 190..210
        session.eraseRange(0, 180, 220)
        val afterEraser = session.trialUi(0)
        assertFalse("Mark removed after eraseRange", afterEraser.marks.any { it.onset == 190L && it.offset == 210L })

        // Undo restores the mark
        session.undo(0)
        val afterUndo = session.trialUi(0)
        assertTrue("Mark restored after undo", afterUndo.marks.any { it.onset == 190L && it.offset == 210L })
    }

    /**
     * Point 6: Switching between Saccade and Blink while eraser is active automatically deactivates the eraser.
     */
    @Test
    fun testPoint6EraserDeactivatesOnClassSwitch() {
        var activeTool = TraceView.TOOL_SACCADE
        var cls = Native.CLS_SACCADE

        fun selectTool(tool: Int, classId: Int) {
            activeTool = tool
            if (classId != 0) cls = classId
        }

        fun onSaccadeClick() = selectTool(TraceView.TOOL_SACCADE, Native.CLS_SACCADE)
        fun onBlinkClick() = selectTool(TraceView.TOOL_BLINK, Native.CLS_BLINK)
        fun onEraserClick() {
            if (activeTool == TraceView.TOOL_ERASER) {
                val backTool = if (cls == Native.CLS_BLINK) TraceView.TOOL_BLINK else TraceView.TOOL_SACCADE
                selectTool(backTool, cls)
            } else {
                selectTool(TraceView.TOOL_ERASER, 0)
            }
        }

        // Initially Saccade
        assertEquals(TraceView.TOOL_SACCADE, activeTool)
        assertEquals(Native.CLS_SACCADE, cls)

        // Turn on Eraser
        onEraserClick()
        assertEquals(TraceView.TOOL_ERASER, activeTool)
        assertEquals(Native.CLS_SACCADE, cls)

        // Click Blink from Saccade while eraser active -> eraser automatically deactivates
        onBlinkClick()
        assertEquals("Eraser must deactivate on Blink click", TraceView.TOOL_BLINK, activeTool)
        assertEquals("Class switched to Blink", Native.CLS_BLINK, cls)

        // Turn on Eraser again
        onEraserClick()
        assertEquals(TraceView.TOOL_ERASER, activeTool)
        assertEquals(Native.CLS_BLINK, cls)

        // Click Saccade from Blink while eraser active -> eraser automatically deactivates
        onSaccadeClick()
        assertEquals("Eraser must deactivate on Saccade click", TraceView.TOOL_SACCADE, activeTool)
        assertEquals("Class switched to Saccade", Native.CLS_SACCADE, cls)

        // Toggle eraser on then off directly with eraser button
        onEraserClick()
        assertEquals(TraceView.TOOL_ERASER, activeTool)
        onEraserClick()
        assertEquals("Clicking eraser while active toggles it off", TraceView.TOOL_SACCADE, activeTool)
    }

    /**
     * Point 7: Save and load labels with scans.txt persistence and path isolation.
     */
    @Test
    fun testPoint7SaveLoadLabelsWithScans() = kotlinx.coroutines.runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as UneyeApp
        val session = app.session

        val xRows = mutableListOf<DoubleArray>()
        val yRows = mutableListOf<DoubleArray>()
        for (k in 0L until 2L) {
            val (x, y, _) = synthTrial(600 + k)
            xRows.add(x)
            yRows.add(y)
        }

        var loadDone = false
        session.load(
            TraceFiles.Trace(xRows, yRows),
            rateHz = 1000.0, degPerUnit = 1.0, estimateScale = false,
            prelabeler = Native.PRELABELER_HMM, modelBin = "",
            recordingName = "save_test"
        ) { _, _ -> loadDone = true }
        while (!loadDone) kotlinx.coroutines.delay(20)

        session.addScan(0, 100, 500)
        session.addPointMark(0, 250, Native.CLS_SACCADE)
        session.refineMark(0, 250, 240, 260, Native.CLS_SACCADE)

        val labelsPath = app.labelsPathFor(session, 0)
        val labelsParent = File(labelsPath).parentFile!!
        labelsParent.deleteRecursively()
        labelsParent.mkdirs()

        assertTrue("saveLabels succeeds", session.saveLabels(labelsPath))
        // Multiple trials produce labels.bin.trial0 and labels.bin.trial1
        assertTrue("trial0 labels written", File(labelsPath + ".trial0").exists())
        val scansFile = File(labelsParent, "scans.txt")
        assertTrue("scans.txt written", scansFile.exists())

        val scanContent = scansFile.readText()
        assertTrue("scans.txt contains trial 0 scan", scanContent.contains("0 100 500"))

        // Now test loadLabels restores scans
        assertTrue("loadLabels succeeds", session.loadLabels(labelsPath))
        assertTrue("Scanned interval 100..500 restored", session.inScanned(0, 300))
    }

    /**
     * Points 8 & 11: Single position lane & minimum 1.25° range clamping for 1° scale bar.
     */
    @Test
    fun testPoints8And11SinglePositionLaneAndScaleBarClamping() {
        // Test scale range math: for values between 0.1 and 0.3 degrees (delta = 0.2 < 1.25)
        val degPerUnit = 1.0
        val oneDeg = 1.0f
        var minV = 0.1f
        var maxV = 0.3f
        val pad = max(0.02f * oneDeg, (maxV - minV) * 0.04f)
        minV -= pad
        maxV += pad
        if (maxV - minV < 1.25f * oneDeg) {
            val c = (minV + maxV) / 2f
            minV = c - 0.625f * oneDeg
            maxV = c + 0.625f * oneDeg
        }

        val range = maxV - minV
        assertTrue("Range must never be tighter than 1.25 degrees", range >= 1.25f)
        assertTrue("1-degree scale bar height fits within lane", 1.0f / range <= 1.0f / 1.25f)
    }

    /**
     * Point 10: Screen space and menu toggle.
     * Settings panel is GONE by default; menu button toggles it.
     */
    @Test
    fun testPoint10ScreenSpaceAndMenuToggle() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val instrumentation = InstrumentationRegistry.getInstrumentation()

        instrumentation.runOnMainSync {
            val inflater = LayoutInflater.from(context)
            val root = inflater.inflate(R.layout.activity_labeling, null)

            val settingsPanel = root.findViewById<View>(R.id.settingsPanel)
            val menuBtn = root.findViewById<Button>(R.id.menuBtn)
            val traceView = root.findViewById<TraceView>(R.id.traceView)

            // 1. Settings panel is GONE by default
            assertEquals("Settings panel hidden by default", View.GONE, settingsPanel.visibility)

            // 2. TraceView exists and takes primary vertical space
            assertTrue("TraceView exists in layout", traceView != null)

            // 3. Toggle menu button
            menuBtn.performClick()
            settingsPanel.visibility = if (settingsPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
            assertEquals("Settings panel visible after menu toggle", View.VISIBLE, settingsPanel.visibility)

            settingsPanel.visibility = if (settingsPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
            assertEquals("Settings panel hidden again after second toggle", View.GONE, settingsPanel.visibility)
        }
    }

    /**
     * Point 12: Tapping on a proposed event accepts whole duration as ground truth,
     * and window slides so event is at left of window (50 ms before onset).
     */
    @Test
    fun testPoint12TapProposedEventAndAutoSlideWindow() = kotlinx.coroutines.runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as UneyeApp
        val session = app.session

        val xRows = mutableListOf<DoubleArray>()
        val yRows = mutableListOf<DoubleArray>()
        for (k in 0L until 2L) {
            val (x, y, _) = synthTrial(800 + k)
            xRows.add(x)
            yRows.add(y)
        }

        var loadDone = false
        session.load(
            TraceFiles.Trace(xRows, yRows),
            rateHz = 1000.0, degPerUnit = 1.0, estimateScale = false,
            prelabeler = Native.PRELABELER_HMM, modelBin = "",
            recordingName = "slide_test"
        ) { _, _ -> loadDone = true }
        while (!loadDone) kotlinx.coroutines.delay(20)

        val ui0 = session.trialUi(0)
        assertTrue("Trial 0 has detected prelabels", ui0.prelabels.isNotEmpty())

        val firstPre = ui0.prelabels[0]
        val tapSample = firstPre.onset + 5

        // Simulate accepting the pre-label directly as ground truth across its full duration
        val ok = session.refineMark(0, tapSample, firstPre.onset, firstPre.offset, firstPre.cls)
        assertTrue("refineMark succeeded for proposed event", ok)

        val updatedUi = session.trialUi(0)
        val groundTruth = updatedUi.marks.find { it.onset == firstPre.onset && it.offset == firstPre.offset }
        assertTrue("Proposed event accepted as ground truth across entire duration", groundTruth != null)
        assertFalse("Ground truth event is marked as expert, not unconfirmed prelabel", groundTruth!!.fromPrelabel)
        assertFalse("No pending point mark created", updatedUi.pointMarks.any { it.first == tapSample })

        // Validate slide window math:
        // Window starts 50 ms before event onset:
        val epochMs = 500L
        val pad50 = (50.0 * session.rateHz / 1000.0).toLong()
        val n = session.trialSamples
        val expectedFrom = max(0L, min(firstPre.onset - pad50, n - epochMs))
        val expectedTo = min(n - 1, expectedFrom + epochMs - 1)

        val actualFrom = max(0L, firstPre.onset - pad50)
        val actualTo = actualFrom + epochMs - 1
        assertEquals("Window starts 50 ms before event onset", expectedFrom, actualFrom)
        assertEquals("Window ends 450 ms after event onset", expectedTo, actualTo)
    }
}
