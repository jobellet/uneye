package com.uneye.offline

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Readers for the eye-trace file formats of the plan (phase 2):
 *
 *  - CSV (UnEye.py convention, `np.loadtxt(delimiter=",")`): a trials x time
 *    matrix or a single column. Rows are trials, columns are time samples.
 *    Non-finite values (nan, inf, empty fields) mark blink / lost samples.
 *  - MATLAB v5 `.mat` (what `io.loadmat` reads): a single file with the
 *    variables X and Y (2-D double arrays, column-major). Only miDOUBLE
 *    (miUINT8/miINT8/miINT16/miINT32/miSINGLE are converted too) and
 *    uncompressed elements are supported — scipy's default save format.
 *
 * Both readers return X / Y as rows = trials, columns = time, ready for
 * [Native.load].
 */
object TraceFiles {

    data class Trace(val x: List<DoubleArray>, val y: List<DoubleArray>)

    /**
     * Reads one `.mat` file (X and Y inside). For CSV the app loads the X file
     * and the Y file separately (UnEye convention): [readCsvPair].
     */
    fun read(input: InputStream, fileName: String): Trace {
        val name = fileName.lowercase()
        return if (name.endsWith(".mat")) {
            val bytes = input.use { it.readBytes() }
            MatReader.read(bytes)
        } else {
            throw IllegalArgumentException("CSV needs an X and a Y file: use readCsvPair(xText, yText)")
        }
    }

    fun read(file: File): Trace = file.inputStream().use { read(it, file.name) }

    /** Reads every variable of a `.mat` file (name -> rows x cols, row-major). */
    fun readAll(bytes: ByteArray): Map<String, Array<DoubleArray>> =
        MatReader.readAll(bytes)

    /** Two CSV files (X axis, Y axis), each trials x time or a single column. */
    fun readCsvPair(xText: String, yText: String): Trace = CsvParser.parsePair(xText, yText)
}

/** CSV reader: comma or whitespace separated, one trial per row (or one column). */
object CsvParser {

    /**
     * One file per axis (the UnEye convention: `-x fileX.csv -y fileY.csv`),
     * each a trials x time matrix or a single column (one long trial, cut into
     * 1 s trials by the facade).
     */
    fun parsePair(xText: String, yText: String): TraceFiles.Trace {
        val xr = parseRows(xText)
        val yr = parseRows(yText)
        if (xr.size != yr.size) throw IllegalArgumentException("X/Y row counts differ (${xr.size} vs ${yr.size})")
        for (i in xr.indices) if (xr[i].size != yr[i].size)
            throw IllegalArgumentException("X/Y row $i lengths differ (${xr[i].size} vs ${yr[i].size})")
        return TraceFiles.Trace(xr, yr)
    }

    private fun parseRows(text: String): List<DoubleArray> {
        val rows = ArrayList<DoubleArray>()
        for (line in text.lineSequence()) {
            val t = line.trim()
            if (t.isEmpty() || t.startsWith("#")) continue
            val fields = if (t.contains(',')) t.split(',') else t.split(Regex("\\s+"))
            rows.add(DoubleArray(fields.size) { i -> parseDouble(fields[i]) })
        }
        if (rows.isEmpty()) throw IllegalArgumentException("empty CSV")
        return rows
    }

    private fun parseDouble(s: String): Double {
        val t = s.trim()
        if (t.isEmpty()) return Double.NaN
        return t.toDoubleOrNull() ?: Double.NaN   // nan/inf parse; garbage -> NaN (lost sample)
    }
}

/**
 * Minimal MATLAB v5 reader: uncompressed miMATRIX elements, numeric real
 * arrays, converted to DoubleArray (column-major flattened). Variables X and
 * Y are returned as rows = trials (a 1-D variable becomes one row).
 */
object MatReader {

    fun read(bytes: ByteArray): TraceFiles.Trace {
        val vars = readAll(bytes)
        val x = vars["X"] ?: throw IllegalArgumentException("MAT file has no variable 'X'")
        val y = vars["Y"] ?: throw IllegalArgumentException("MAT file has no variable 'Y'")
        return TraceFiles.Trace(x.toList(), y.toList())
    }

    /** Reads every numeric real variable of a MAT v5 file (name -> rows x cols). */
    fun readAll(bytes: ByteArray): Map<String, Array<DoubleArray>> {
        if (bytes.size < 128) throw IllegalArgumentException("not a MAT v5 file (too short)")
        val header = bytes.copyOfRange(0, 128)
        val endian = String(header, 126, 2, Charsets.US_ASCII)
        val little = when (endian) {
            "IM" -> true
            "MI" -> false
            else -> throw IllegalArgumentException("not a MAT v5 file (bad endian indicator '$endian')")
        }
        val buf = ByteBuffer.wrap(bytes).order(if (little) ByteOrder.LITTLE_ENDIAN else ByteOrder.BIG_ENDIAN)
        val vars = HashMap<String, Array<DoubleArray>>()
        var pos = 128
        while (pos + 8 <= bytes.size) {
            val type = buf.getInt(pos).toLong() and 0xFFFFFFFFL
            val size = buf.getInt(pos + 4).toLong() and 0xFFFFFFFFL
            if (type == 0L || size <= 0 || pos + 8 + size > bytes.size) break
            if (type == MI_MATRIX) {
                val v = parseMatrix(bytes, pos + 8, size.toInt(), little)
                if (v != null) {
                    // column-major storage -> row-major rows
                    val rows = Array(v.rows) { r -> DoubleArray(v.cols) { c -> v.data[c * v.rows + r] } }
                    vars[v.name] = rows
                }
            }
            pos += (8 + size + ((8 - size % 8) % 8)).toInt()
        }
        return vars
    }

    private const val MI_MATRIX = 14L
    private const val MI_COMPRESSED = 15L
    private const val MI_DOUBLE = 9L
    private const val MI_SINGLE = 7L
    private const val MI_INT32 = 5L
    private const val MI_INT16 = 3L
    private const val MI_UINT8 = 2L
    private const val MI_INT8 = 1L
    private const val MX_DOUBLE = 6

    private class MatVar(val name: String, val rows: Int, val cols: Int, val data: DoubleArray)

    /** Parses one miMATRIX element body; returns null for unsupported variables. */
    private fun parseMatrix(bytes: ByteArray, off: Int, len: Int, little: Boolean): MatVar? {
        val buf = ByteBuffer.wrap(bytes).order(if (little) ByteOrder.LITTLE_ENDIAN else ByteOrder.BIG_ENDIAN)
        var p = off
        val end = off + len

        var classId = -1
        var rows = 0
        var cols = 0
        var name = ""
        var data: DoubleArray? = null

        // Sub-elements: standard MATLAB 5 order is flags -> dims -> name -> data
        while (p + 4 <= end) {
            val (t, s, adv) = readTag(buf, p)
            if (s <= 0 && adv != 4) break
            when (t) {
                6L -> { // Array Flags (miUINT32)
                    if (s >= 8 && p + 8 + 4 <= end) {
                        classId = buf.getInt(p + 8) and 0xFF
                    }
                    p += (adv + s + ((8 - s % 8) % 8)).toInt()
                }
                5L -> { // Dimensions (miINT32)
                    val nDims = (s / 4).toInt()
                    if (nDims >= 1 && p + 8 + nDims * 4 <= end) {
                        val dims = IntArray(nDims) { buf.getInt(p + 8 + it * 4) }
                        rows = dims[0]
                        cols = if (nDims > 1) dims[1] else 1
                    }
                    p += (adv + s + ((8 - s % 8) % 8)).toInt()
                }
                1L, 2L -> { // Array Name (miINT8 or miUINT8)
                    if (adv == 4) {
                        name = String(bytes, p + 4, s.toInt(), Charsets.US_ASCII)
                        p += 8
                    } else {
                        name = String(bytes, p + 8, s.toInt(), Charsets.US_ASCII)
                        p += (8 + s + ((8 - s % 8) % 8)).toInt()
                    }
                }
                MI_DOUBLE, MI_SINGLE, MI_INT32, MI_INT16 -> { // Real data
                    val n = rows.toLong() * cols
                    if (n > 0 && n <= Int.MAX_VALUE / 2) {
                        val dp = p + adv
                        data = when (t) {
                            MI_DOUBLE -> DoubleArray(n.toInt()) { buf.getDouble(dp + it * 8) }
                            MI_SINGLE -> DoubleArray(n.toInt()) { buf.getFloat(dp + it * 4).toDouble() }
                            MI_INT32 -> DoubleArray(n.toInt()) { buf.getInt(dp + it * 4).toDouble() }
                            MI_INT16 -> DoubleArray(n.toInt()) { buf.getShort(dp + it * 2).toDouble() }
                            else -> null
                        }
                    }
                    p += (adv + s + ((8 - s % 8) % 8)).toInt()
                }
                else -> {
                    if (adv == 4) p += 8
                    else p += (8 + s + ((8 - s % 8) % 8)).toInt()
                }
            }
            if (data != null) break
        }

        if (classId != MX_DOUBLE || data == null || rows <= 0 || cols <= 0) return null
        return MatVar(name, rows, cols, data)
    }

    /** Reads a data tag; returns (type, size, bytes consumed by the tag itself). */
    private fun readTag(buf: ByteBuffer, p: Int): Triple<Long, Long, Int> {
        val w = buf.getInt(p).toLong() and 0xFFFFFFFFL
        val sizeInHigh = w ushr 16          // small element format: size in the high 16 bits of the first word
        return if (sizeInHigh in 1..4) {
            // small element format: type = low 16 bits, size = high 16 bits, data follows in the next 4 bytes
            Triple(w and 0xFFFFL, sizeInHigh, 4)
        } else {
            Triple(w, buf.getInt(p + 4).toLong() and 0xFFFFFFFFL, 8)
        }
    }
}
