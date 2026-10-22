package com.uneye.offline

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal MATLAB v5 writer (uncompressed, little-endian, miDOUBLE matrices).
 * Writes exactly what `io.loadmat` reads back: a 128-byte header plus one
 * miMATRIX element per variable, column-major. Round-trip verified against
 * the reader in [MatReader].
 *
 * Variables: X, Y (positions in degrees, rows = trials) and Sacc (labels
 * 0/1/2 as doubles, the UnEye `.mat` convention — `io.loadmat(L)['Sacc']`).
 */
object MatWriter {

    private const val MI_INT8 = 2
    private const val MI_UINT32 = 6
    private const val MI_INT32 = 5
    private const val MI_DOUBLE = 9
    private const val MI_MATRIX = 14
    private const val MX_DOUBLE_CLASS = 6

    /**
     * @param vars variables as name -> rows x cols matrix (row-major input,
     *             written column-major as MATLAB stores them)
     */
    fun write(file: File, vars: Map<String, Array<DoubleArray>>) {
        file.parentFile?.mkdirs()
        FileOutputStream(file).use { writeTo(it, vars) }
    }

    /** Same as [write] but to any output stream (SAF CREATE_DOCUMENT). */
    fun writeTo(stream: OutputStream, vars: Map<String, Array<DoubleArray>>) {
        val out = ByteArrayOutputStream()
        // header: 116 bytes text + 8 bytes subsystem offset + 2 version + 2 endian
        val text = "MATLAB 5.0 MAT-file, Platform: Android, Created by: uneye-offline-labeler"
            .toByteArray(Charsets.US_ASCII)
        out.write(text)
        out.write(ByteArray(116 - text.size))
        out.write(byteArrayOf(0, 0, 0, 0, 0, 0, 0, 0))   // subsystem data offset (unused)
        out.write(byteArrayOf(0x00, 0x01))               // version 0x0100, little-endian
        out.write(byteArrayOf('I'.code.toByte(), 'M'.code.toByte()))
        for ((name, rows) in vars) out.write(matrixElement(name, rows))
        stream.use { it.write(out.toByteArray()) }
    }

    /** One miMATRIX element: name + flags (mxDOUBLE_CLASS) + dims + column-major doubles. */
    private fun matrixElement(name: String, rows: Array<DoubleArray>): ByteArray {
        val nRows = rows.size
        val nCols = if (nRows > 0) rows[0].size else 0
        val body = ByteArrayOutputStream()

        // array name: small element format when <= 4 chars (miINT8), else regular
        if (name.length <= 4) {
            body.write(le16(MI_INT8)); body.write(le16(name.length))
            body.write(name.toByteArray(Charsets.US_ASCII))
            body.write(ByteArray(4 - name.length))
        } else {
            body.write(le32(MI_INT8)); body.write(le32(name.length))
            body.write(name.toByteArray(Charsets.US_ASCII))
            body.write(ByteArray((8 - name.length % 8) % 8))
        }
        // array flags: miUINT32, 8 bytes: [class | flags, 0]
        body.write(le32(MI_UINT32)); body.write(le32(8))
        body.write(le32(MX_DOUBLE_CLASS)); body.write(le32(0))
        // dimensions: miINT32, 2 dims
        body.write(le32(MI_INT32)); body.write(le32(8))
        body.write(le32(nRows)); body.write(le32(nCols))
        // real part: miDOUBLE, column-major
        val n = nRows.toLong() * nCols
        body.write(le32(MI_DOUBLE)); body.write(le32((n * 8).toInt()))
        val bb = ByteBuffer.allocate((n * 8).toInt()).order(ByteOrder.LITTLE_ENDIAN)
        for (c in 0 until nCols) for (r in 0 until nRows) bb.putDouble(rows[r][c])
        body.write(bb.array())
        body.write(ByteArray((8 - (n * 8) % 8) % 8))

        val bodyBytes = body.toByteArray()
        val el = ByteArrayOutputStream()
        el.write(le32(MI_MATRIX)); el.write(le32(bodyBytes.size))
        el.write(bodyBytes)
        el.write(ByteArray((8 - bodyBytes.size % 8) % 8))
        return el.toByteArray()
    }

    private fun le16(v: Int) = byteArrayOf((v and 0xFF).toByte(), ((v ushr 8) and 0xFF).toByte())
    private fun le32(v: Int) = byteArrayOf(
        (v and 0xFF).toByte(), ((v ushr 8) and 0xFF).toByte(),
        ((v ushr 16) and 0xFF).toByte(), ((v ushr 24) and 0xFF).toByte())
}
