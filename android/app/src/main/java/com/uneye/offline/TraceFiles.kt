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
        if (bytes.size < 128) throw IllegalArgumentException("not a MAT v5 file (too short)")
        val header = bytes.copyOfRange(0, 128)
        val endian = String(header, 126, 2, Charsets.US_ASCII)
        val little = when (endian) {
            "IM" -> true
            "MI" -> false
            else -> throw IllegalArgumentException("not a MAT v5 file (bad endian indicator '$endian')")
        }
        val buf = ByteBuffer.wrap(bytes).order(if (little) ByteOrder.LITTLE_ENDIAN else ByteOrder.BIG_ENDIAN)
        val vars = HashMap<String, MatVar>()
        var pos = 128
        while (pos + 8 <= bytes.size) {
            val type = buf.getInt(pos).toLong() and 0xFFFFFFFFL
            val size = buf.getInt(pos + 4).toLong() and 0xFFFFFFFFL
            if (type == 0L || size <= 0 || pos + 8 + size > bytes.size) break
            if (type == MI_MATRIX) {
                val v = parseMatrix(bytes, pos + 8, size.toInt(), little)
                if (v != null) vars[v.name] = v
            }
            pos += (8 + size + ((8 - size % 8) % 8)).toInt()
        }
        val x = vars["X"] ?: throw IllegalArgumentException("MAT file has no variable 'X'")
        val y = vars["Y"] ?: throw IllegalArgumentException("MAT file has no variable 'Y'")
        return TraceFiles.Trace(toRows(x), toRows(y))
    }

    private const val MI_MATRIX = 14L
    private const val MI_COMPRESSED = 15L
    private const val MI_DOUBLE = 9L
    private const val MI_SINGLE = 7L
    private const val MI_INT32 = 5L
    private const val MI_INT16 = 3L
    private const val MI_INT8 = 2L
    private const val MI_UINT8 = 1L
    private const val MX_DOUBLE = 6

    private class MatVar(val name: String, val rows: Int, val cols: Int, val data: DoubleArray)

    private fun toRows(v: MatVar): List<DoubleArray> {
        // column-major storage -> row-major rows
        val rows = ArrayList<DoubleArray>(v.rows)
        for (r in 0 until v.rows) {
            val row = DoubleArray(v.cols)
            for (c in 0 until v.cols) row[c] = v.data[c * v.rows + r]
            rows.add(row)
        }
        return rows
    }

    /** Parses one miMATRIX element body; returns null for unsupported variables. */
    private fun parseMatrix(bytes: ByteArray, off: Int, len: Int, little: Boolean): MatVar? {
        val buf = ByteBuffer.wrap(bytes).order(if (little) ByteOrder.LITTLE_ENDIAN else ByteOrder.BIG_ENDIAN)
        var p = off
        val end = off + len
        // array name (miINT8, small or regular format). Small elements pack the
        // tag in 4 bytes and pad the data to 4 (not 8); regular elements pad to 8.
        var name = ""
        run {
            val (t, s, adv) = readTag(buf, p)
            if (adv == 4) {
                name = String(bytes, p + adv, s.toInt(), Charsets.US_ASCII)
                p += 8                      // 4-byte tag + 4 bytes of padded data
            } else {
                name = String(bytes, p + 8, s.toInt(), Charsets.US_ASCII)
                p += (8 + s + ((8 - s % 8) % 8)).toInt()
            }
        }
        // array flags (miUINT32, 8 bytes): class in the low byte of the first word
        val flagsTag = buf.getInt(p); val flagsSize = buf.getInt(p + 4)
        if (flagsTag != 6 || flagsSize != 8) return null
        val classId = buf.getInt(p + 8) and 0xFF
        p += 16
        // dimensions (miINT32)
        val dimTag = buf.getInt(p); val dimSize = buf.getInt(p + 4)
        if (dimTag != 5) return null
        val nDims = dimSize / 4
        val dims = IntArray(nDims) { buf.getInt(p + 8 + it * 4) }
        p += 8 + dimSize + ((8 - dimSize % 8) % 8)
        if (classId != MX_DOUBLE) return null          // only numeric real double arrays
        if (nDims < 1) return null
        val rows = dims[0]
        val cols = if (nDims > 1) dims[1] else 1
        val n = rows.toLong() * cols
        if (n > Int.MAX_VALUE / 2 || p + 8 + n * 8 > end + 8) return null
        // real part (miDOUBLE or a smaller numeric type, converted)
        val (rt, rs, radv) = readTag(buf, p)
        if (rt == MI_COMPRESSED) return null
        val data = when (rt) {
            MI_DOUBLE -> {
                p += radv
                DoubleArray(n.toInt()) { buf.getDouble(p + it * 8) }
            }
            MI_SINGLE -> {
                p += radv
                DoubleArray(n.toInt()) { buf.getFloat(p + it * 4).toDouble() }
            }
            MI_INT32 -> {
                p += radv
                DoubleArray(n.toInt()) { buf.getInt(p + it * 4).toDouble() }
            }
            MI_INT16 -> {
                p += radv
                DoubleArray(n.toInt()) { buf.getShort(p + it * 2).toDouble() }
            }
            MI_INT8, MI_UINT8 -> {
                p += radv
                DoubleArray(n.toInt()) { (if (rt == MI_INT8) buf.get(p + it) else buf.get(p + it).toInt() and 0xFF).toDouble() }
            }
            else -> return null
        }
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
