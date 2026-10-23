package com.uneye.offline

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * Phase 5 unit tests: the export writers must produce files the training
 * pipeline reads back unchanged — labels CSV in the UnEye convention
 * (0/1/2, NaN = not reviewed), X/Y CSV as degrees, and a MATLAB v5 file
 * readable by the app's own [MatReader] (and by scipy.io.loadmat).
 */
class ExporterTest {

    // --- labels CSV ---------------------------------------------------------------

    @Test
    fun labelsCsvWritesNanForUnreviewed() {
        val labels = arrayOf(
            doubleArrayOf(0.0, 1.0, 1.0, Double.NaN, Double.NaN, 2.0),
        )
        val out = ByteArrayOutputStream()
        Exporter.writeLabelsCsv(out, labels)
        val text = out.toString(Charsets.UTF_8)
        assertEquals("0,1,1,nan,nan,2\n", text)
    }

    @Test
    fun labelsCsvColumnWritesOneValuePerLine() {
        val labels = arrayOf(doubleArrayOf(1.0, 0.0, 2.0))
        val out = ByteArrayOutputStream()
        Exporter.writeLabelsCsvColumn(out, labels)
        assertEquals("1\n0\n2\n", out.toString(Charsets.UTF_8))
    }

    @Test
    fun labelsCsvMultipleTrialsOneRowEach() {
        val labels = arrayOf(
            doubleArrayOf(1.0, 1.0),
            doubleArrayOf(0.0, 2.0),
        )
        val out = ByteArrayOutputStream()
        Exporter.writeLabelsCsv(out, labels)
        assertEquals("1,1\n0,2\n", out.toString(Charsets.UTF_8))
    }

    // --- positions CSV ------------------------------------------------------------

    @Test
    fun positionsCsvRoundTripsDoubles() {
        val x = arrayOf(doubleArrayOf(1.5, -0.25, 1e-9))
        val out = ByteArrayOutputStream()
        Exporter.writePositionsCsv(out, x)
        val line = out.toString(Charsets.UTF_8).trim()
        // %.17g round-trips a double exactly
        val parsed = line.split(",").map { it.toDouble() }
        assertArrayEquals(doubleArrayOf(1.5, -0.25, 1e-9), parsed.toDoubleArray(), 0.0)
    }

    @Test
    fun positionsCsvWritesNanAndInf() {
        val x = arrayOf(doubleArrayOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY))
        val out = ByteArrayOutputStream()
        Exporter.writePositionsCsv(out, x)
        assertEquals("nan,inf,-inf\n", out.toString(Charsets.UTF_8))
    }

    // --- MATLAB v5 -----------------------------------------------------------------

    @Test
    fun matWriterRoundTripThroughMatReader() {
        val vars = mapOf(
            "X" to arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)),
            "Y" to arrayOf(doubleArrayOf(0.5, Double.NaN), doubleArrayOf(-1.25, 7.0)),
            "Sacc" to arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(2.0, Double.NaN)),
        )
        val out = ByteArrayOutputStream()
        MatWriter.writeTo(out, vars)
        val back = MatReader.readAll(out.toByteArray())
        assertEquals(3, back.size)
        for ((name, rows) in vars) {
            val got = back[name]!!
            assertEquals("rows of $name", rows.size, got.size)
            for (r in rows.indices) {
                assertArrayEquals("row $r of $name", rows[r], got[r], 0.0)
            }
        }
    }

    @Test
    fun matWriterHeaderIsMatlabV5LittleEndian() {
        val out = ByteArrayOutputStream()
        MatWriter.writeTo(out, mapOf("X" to arrayOf(doubleArrayOf(1.0))))
        val bytes = out.toByteArray()
        assertTrue(bytes.size > 128)
        // version 0x0100 + "IM" endian indicator at offsets 124..127
        assertEquals(0x00, bytes[124].toInt() and 0xFF)
        assertEquals(0x01, bytes[125].toInt() and 0xFF)
        assertEquals('I'.code.toByte(), bytes[126])
        assertEquals('M'.code.toByte(), bytes[127])
        assertTrue(String(bytes, 0, 6, Charsets.US_ASCII).startsWith("MATLAB"))
    }

    @Test
    fun matWriterVariableNamesLongerThan4Chars() {
        val out = ByteArrayOutputStream()
        MatWriter.writeTo(out, mapOf("Saccades" to arrayOf(doubleArrayOf(1.0, 2.0))))
        val back = MatReader.readAll(out.toByteArray())
        assertArrayEquals(doubleArrayOf(1.0, 2.0), back["Saccades"]!![0], 0.0)
    }

    @Test
    fun matReaderReadsWhatWriterWrote_singleRow() {
        val out = ByteArrayOutputStream()
        MatWriter.writeTo(out, mapOf("X" to arrayOf(doubleArrayOf(9.0, 8.0, 7.0, 6.0))))
        // read() needs X and Y; readAll covers a single-variable file
        val vars = MatReader.readAll(out.toByteArray())
        assertArrayEquals(doubleArrayOf(9.0, 8.0, 7.0, 6.0), vars["X"]!![0], 0.0)
    }

    // --- model export ----------------------------------------------------------------

    @Test
    fun exportModelCopiesBinAndJson() {
        val dir = createTempDir(prefix = "model_src")
        val srcBin = File(dir, "model_round.bin")
        val srcJson = File(dir, "model_round.json")
        srcBin.writeBytes(byteArrayOf(1, 2, 3))
        srcJson.writeText("{}")
        val outDir = createTempDir(prefix = "model_dst")
        val binPath = Exporter.exportModel(srcBin.absolutePath, outDir, "rec_model")
        assertTrue(File(binPath).exists())
        assertArrayEquals(byteArrayOf(1, 2, 3), File(binPath).readBytes())
        assertEquals("{}", File(outDir, "rec_model.json").readText())
        dir.deleteRecursively()
        outDir.deleteRecursively()
    }

    @Test
    fun exportModelWithoutJsonSidecarStillCopiesBin() {
        val dir = createTempDir(prefix = "model_src")
        val srcBin = File(dir, "m.bin")
        srcBin.writeBytes(byteArrayOf(9))
        val outDir = createTempDir(prefix = "model_dst")
        val binPath = Exporter.exportModel(srcBin.absolutePath, outDir, "m")
        assertTrue(File(binPath).exists())
        assertTrue(!File(outDir, "m.json").exists())
        dir.deleteRecursively()
        outDir.deleteRecursively()
    }
}
