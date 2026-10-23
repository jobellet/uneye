package com.uneye.offline

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.ListenableWorker
import androidx.work.testing.TestListenableWorkerBuilder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Random
import kotlin.math.cos
import kotlin.math.sin

@RunWith(AndroidJUnit4::class)
class Phase3WorkerTest {

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
        fixation(150)
        saccade(7.0)
        fixation(150)
        // blink: 200 ms NaN
        for (k in 0 until 200) {
            if (i >= n) break
            x[i] = Double.NaN
            y[i] = Double.NaN
            lab[i] = 2.0
            i++
        }
        fixation(n - i)
        return Triple(x, y, lab)
    }

    @Test
    fun testPhase3FullTrainingWorkerCycle() = kotlinx.coroutines.runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as UneyeApp
        val session = app.session

        // 8 trials synthetic trace
        val xRows = mutableListOf<DoubleArray>()
        val yRows = mutableListOf<DoubleArray>()
        for (k in 0L until 8L) {
            val (x, y, _) = synthTrial(100 + k)
            xRows.add(x)
            yRows.add(y)
        }

        var loadOk = false
        var loadErr = ""
        var loadDone = false
        session.load(
            TraceFiles.Trace(xRows, yRows),
            rateHz = 1000.0, degPerUnit = 1.0, estimateScale = false,
            prelabeler = Native.PRELABELER_HMM, modelBin = "",
        ) { ok, err ->
            loadOk = ok
            loadErr = err
            loadDone = true
        }

        while (!loadDone) {
            kotlinx.coroutines.delay(20)
        }
        assertTrue("Session load failed: $loadErr", loadOk)
        assertEquals(8L, session.nTrials.value)

        // Mark trials reviewed
        session.chooseTestSet(4, 2)
        for (k in 0L until 8L) {
            session.clearPointMarks(k)
            val ui = session.trialUi(k)
            for (m in ui.prelabels) {
                session.refineMark(k, m.onset, m.onset, m.offset, m.cls)
            }
            session.markEpochReviewed(k, 0, 999)
        }
        assertTrue("Reviewed seconds: ${session.reviewedSeconds.value}", session.reviewedSeconds.value > 5.0)

        // Clean any existing models/history for a clean test run
        val modelFile = File(app.modelsDir(), "model_round.bin")
        if (modelFile.exists()) modelFile.delete()
        val convFile = File(context.filesDir, "convergence.json")
        if (convFile.exists()) convFile.delete()

        // Execute TrainingWorker directly via TestListenableWorkerBuilder
        val worker = TestListenableWorkerBuilder<TrainingWorker>(context.applicationContext).build()
        val result = worker.doWork()
        assertEquals("Worker result should be success", ListenableWorker.Result.success(), result)

        // Verify:
        // 1. model_round.bin saved
        assertTrue("model_round.bin should exist", modelFile.exists())
        assertTrue("model_round.bin should be non-empty", modelFile.length() > 1000)

        // 2. Prelabeler swapped to LAST_TRAINED
        assertEquals(Native.PRELABELER_LAST_TRAINED, session.prelabeler())

        // 3. ConvergenceStore recorded round
        assertTrue("convergence.json should exist", convFile.exists())
        val rounds = app.convergenceStore.rounds()
        assertEquals(1, rounds.size)
        val round = rounds[0]
        assertTrue("epochs > 0", round.epochs > 0)
        assertTrue("labeledSeconds > 0", round.labeledSeconds > 5.0)
        assertEquals(modelFile.absolutePath, round.modelPath)

        // 4. Session baseline updated
        assertTrue("reviewedAtLastTraining updated", session.reviewedAtLastTraining > 5.0)
    }
}
