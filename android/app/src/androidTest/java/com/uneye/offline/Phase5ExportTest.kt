package com.uneye.offline

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Random
import kotlin.math.cos
import kotlin.math.sin

/**
 * Phase 5 instrumented test: end-to-end export on the device.
 *
 * Loads a synthetic 4-trial trace, refines the pre-labels, marks the trials
 * reviewed, then exports the session into the app's external files dir and
 * checks:
 *  - the 3 CSV files exist and the labels CSV is in the UnEye convention
 *    (0/1/2, NaN = not reviewed),
 *  - the .mat file exists and reads back through [MatReader] with the same
 *    X / Y / Sacc values,
 *  - the model .bin + .json are copied when a trained model exists.
 */
@RunWith(AndroidJUnit4::class)
class Phase5ExportTest {

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
    fun testExportAllWritesReadableFiles() = kotlinx.coroutines.runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val app = context.applicationContext as UneyeApp
        val session = app.session

        // 4 trials synthetic trace
        val xRows = mutableListOf<DoubleArray>()
        val yRows = mutableListOf<DoubleArray>()
        for (k in 0L until 4L) {
            val (x, y, _) = synthTrial(200 + k)
            xRows.add(x)
            yRows.add(y)
        }

        var loadOk = false
        var loadErr = ""
        session.load(
            TraceFiles.Trace(xRows, yRows),
            rateHz = 1000.0, degPerUnit = 1.0, estimateScale = false,
            prelabeler = Native.PRELABELER_HMM, modelBin = "",
            recordingName = "export_test",
        ) { ok, err ->
            loadOk = ok
            loadErr = err
        }
        while (!session.loaded.value && loadErr.isEmpty()) {
            kotlinx.coroutines.delay(20)
        }
        assertTrue("Session load failed: $loadErr", loadOk)
        assertEquals(4L, session.nTrials.value)

        // Review every trial: refine each pre-label, then mark the epoch
        for (k in 0L until 4L) {
            session.clearPointMarks(k)
            val ui = session.trialUi(k)
            for (m in ui.prelabels) {
                session.refineMark(k, m.onset, m.onset, m.offset, m.cls)
            }
            session.markEpochReviewed(k, 0, 999)
        }
        assertTrue("Reviewed seconds: ${session.reviewedSeconds.value}",
            session.reviewedSeconds.value >= 4.0)

        // Fake a trained model so the model export path is exercised
        val modelFile = File(app.modelsDir(), "model_round.bin")
        modelFile.parentFile?.mkdirs()
        modelFile.writeBytes(ByteArray(2048) { (it % 251).toByte() })
        File(app.modelsDir(), "model_round.json").writeText("{}")

        // Export everything into the app's external files dir
        val dir = File(context.getExternalFilesDir("exports"), "export_test")
        if (dir.exists()) dir.deleteRecursively()
        dir.mkdirs()
        val written = mutableListOf<File>()
        withContext(Dispatchers.IO) {
            val labels = Exporter.labelsMatrix(session)
            val labelsFile = File(dir, "export_test_L.csv")
            labelsFile.outputStream().use { Exporter.writeLabelsCsv(it, labels) }
            written.add(labelsFile)
            val (xs, ys) = Exporter.positionsMatrix(session)
            val xFile = File(dir, "export_test_X.csv")
            xFile.outputStream().use { Exporter.writePositionsCsv(it, xs) }
            written.add(xFile)
            val yFile = File(dir, "export_test_Y.csv")
            yFile.outputStream().use { Exporter.writePositionsCsv(it, ys) }
            written.add(yFile)
            val matFile = File(dir, "export_test.mat")
            matFile.outputStream().use { Exporter.writeMat(it, session) }
            written.add(matFile)
            val bin = Exporter.exportModel(modelFile.absolutePath, dir, "export_test_model")
            written.add(File(bin))
            written.add(File(dir, "export_test_model.json"))
        }
        assertEquals(6, written.size)
        for (f in written) {
            assertTrue("missing export: ${f.name}", f.exists())
            assertTrue("empty export: ${f.name}", f.length() > 0)
        }

        // --- labels CSV: UnEye convention -------------------------------------
        val labelsText = File(dir, "export_test_L.csv").readText()
        val rows = labelsText.trim().split("\n")
        assertEquals(4, rows.size)                       // one row per trial
        for (row in rows) {
            val vals = row.split(",")
            assertEquals(1000, vals.size)                // 1 s per trial at 1 kHz
            for (v in vals) {
                val d = v.toDouble()
                assertTrue("label out of convention: $v",
                    d.isNaN() || d == 0.0 || d == 1.0 || d == 2.0)
            }
        }
        // the reviewed trials carry at least one saccade and one blink label
        val allLabels = rows.flatMap { it.split(",") }
        assertTrue("no saccade label exported", allLabels.contains("1"))
        assertTrue("no blink label exported", allLabels.contains("2"))

        // --- X CSV: degrees, round-trip against the session trace --------------
        val xText = File(dir, "export_test_X.csv").readText().trim().split("\n")
        assertEquals(4, xText.size)
        val xRow0 = xText[0].split(",").map { it.toDouble() }
        assertEquals(1000, xRow0.size)
        val (sx, _) = session.trace(0, 0, 999)
        for (i in 0 until 1000) {
            assertEquals(sx[i].toDouble(), xRow0[i], 1e-4)
        }

        // --- .mat: reads back through MatReader with the same values -----------
        val matBytes = File(dir, "export_test.mat").readBytes()
        val vars = MatReader.readAll(matBytes)
        assertTrue(vars.containsKey("X"))
        assertTrue(vars.containsKey("Y"))
        assertTrue(vars.containsKey("Sacc"))
        val sacc = vars["Sacc"]!!
        assertEquals(4, sacc.size)
        assertEquals(1000, sacc[0].size)
        // labels match the CSV (NaN stays NaN)
        var nSacc = 0
        for (r in 0 until 4) for (c in 0 until 1000) {
            val csv = rows[r].split(",")[c].toDouble()
            val m = sacc[r][c]
            if (csv.isNaN()) assertTrue("mat should be NaN at $r,$c", m.isNaN())
            else assertEquals(csv, m, 0.0)
            if (m == 1.0) nSacc++
        }
        assertTrue("no saccade in .mat", nSacc > 0)

        // --- model: .bin + .json copied ----------------------------------------
        val binBytes = File(dir, "export_test_model.bin").readBytes()
        assertEquals(modelFile.readBytes().toList(), binBytes.toList())
        assertEquals("{}", File(dir, "export_test_model.json").readText())

        // cleanup
        dir.deleteRecursively()
        modelFile.delete()
        File(app.modelsDir(), "model_round.json").delete()
    }
}
