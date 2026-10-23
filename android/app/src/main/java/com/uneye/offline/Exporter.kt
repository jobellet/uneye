package com.uneye.offline

import java.io.File
import java.io.OutputStream

/**
 * Export (phase 5): writes the session's data in the formats the training
 * pipeline consumes, so the user can upload eye traces + ground truth to
 * Google Colab / a desktop and train a uneye network there.
 *
 * Formats:
 *  - Labels CSV (UnEye convention): one file, a trials x time matrix or a
 *    single column, values 0 fixation / 1 saccade / 2 blink. Only the
 *    REVIEWED ranges carry the expert labels; unreviewed samples are NaN
 *    (the UnEye convention for missing labels). This matches
 *    `np.loadtxt(L, delimiter=",")`.
 *  - X / Y CSV: the positions in degrees (trials x time or single column),
 *    matching `np.loadtxt(X)` / `np.loadtxt(Y)`.
 *  - `.mat`: one file with variables X, Y and Sacc (the UnEye `.mat`
 *    convention: `io.loadmat(X)['X']`, `io.loadmat(L)['Sacc']`).
 *  - Model: the trained `.bin` + its `.json` sidecar (architecture +
 *    metadata: window, classes, deg_per_unit, rate_hz, epochs, val scores),
 *    readable by `uneye` / the native engine / this app.
 *
 * All writers take an [OutputStream] so they work with SAF
 * (ACTION_CREATE_DOCUMENT) as well as plain files.
 */
object Exporter {

    /**
     * Builds the labels matrix (trials x samples) from the session's rendered
     * reviewed labels: 0 fixation, 1 saccade, 2 blink. Samples outside the
     * reviewed ranges are exported as NaN (the UnEye convention for missing
     * labels: `np.isnan(Labels)` selects the labeled samples), NOT as 0 —
     * writing 0 would tell a Colab training run that the user reviewed those
     * samples as fixations. Runs off the UI thread.
     */
    fun labelsMatrix(session: Session): Array<DoubleArray> {
        val nT = session.nTrials.value
        val per = session.trialSamples
        val out = Array(nT.toInt()) { DoubleArray(per.toInt()) { Double.NaN } }
        for (k in 0 until nT) {
            val lab = session.labels(k, 0, per - 1, true)
            for (i in lab.indices) {
                val v = lab[i].toInt() and 0xFF
                // 255 (ignored) -> NaN: the UnEye/Python convention for a
                // missing label. Writing 255 would count as a 4th class in
                // `np.unique(Labels)`; NaN is excluded (`np.isnan(Labels)`).
                out[k.toInt()][i] = if (v == 255) Double.NaN else v.toDouble()
            }
        }
        return out
    }

    /** Builds the X / Y position matrices (trials x samples, degrees). */
    fun positionsMatrix(session: Session): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        val nT = session.nTrials.value
        val per = session.trialSamples
        val xs = Array(nT.toInt()) { DoubleArray(per.toInt()) }
        val ys = Array(nT.toInt()) { DoubleArray(per.toInt()) }
        for (k in 0 until nT) {
            val (x, y) = session.trace(k, 0, per - 1)
            for (i in x.indices) { xs[k.toInt()][i] = x[i].toDouble(); ys[k.toInt()][i] = y[i].toDouble() }
        }
        return xs to ys
    }

    // --- CSV (UnEye / np.loadtxt convention) -------------------------------------

    /** Labels CSV: trials x time matrix (one row per trial); NaN = not reviewed. */
    fun writeLabelsCsv(stream: OutputStream, labels: Array<DoubleArray>) {
        stream.bufferedWriter(Charsets.UTF_8).use { w ->
            for (row in labels) {
                w.write(row.joinToString(",") { formatLabel(it) })
                w.newLine()
            }
        }
    }

    /** Labels CSV as a single column (one value per line) — when there is one trial. */
    fun writeLabelsCsvColumn(stream: OutputStream, labels: Array<DoubleArray>) {
        stream.bufferedWriter(Charsets.UTF_8).use { w ->
            for (row in labels) for (v in row) { w.write(formatLabel(v)); w.newLine() }
        }
    }

    /** Observed mask CSV: 1 for reviewed/observed samples, 0 for unreviewed. */
    fun writeObservedCsv(stream: OutputStream, labels: Array<DoubleArray>) {
        stream.bufferedWriter(Charsets.UTF_8).use { w ->
            for (row in labels) {
                w.write(row.joinToString(",") { if (it.isNaN()) "0" else "1" })
                w.newLine()
            }
        }
    }

    /** X or Y CSV: trials x time matrix. */
    fun writePositionsCsv(stream: OutputStream, positions: Array<DoubleArray>) {
        stream.bufferedWriter(Charsets.UTF_8).use { w ->
            for (row in positions) {
                w.write(row.joinToString(",") { formatDouble(it) })
                w.newLine()
            }
        }
    }

    // --- MATLAB v5 ---------------------------------------------------------------

    /** One .mat file with X, Y (positions), Sacc (labels; NaN = not reviewed), and observed/reviewed masks. */
    fun writeMat(stream: OutputStream, session: Session) {
        val labels = labelsMatrix(session)
        val (xs, ys) = positionsMatrix(session)
        val observed = Array(labels.size) { k ->
            DoubleArray(labels[k].size) { i -> if (labels[k][i].isNaN()) 0.0 else 1.0 }
        }
        MatWriter.writeTo(stream, mapOf(
            "X" to xs,
            "Y" to ys,
            "Sacc" to labels,
            "labels" to labels,
            "observed" to observed,
            "reviewed" to observed,
        ))
    }

    // --- configuration sidecar -----------------------------------------------------

    /**
     * Writes configuration sidecar JSON with preprocessing filter settings and
     * Python Colab reproduction snippet.
     */
    fun writeConfigJson(stream: OutputStream, session: Session, settings: FilterSettings = FilterSettings.DEFAULT) {
        writeConfigJson(stream, session.degPerUnit, 1000.0, settings)
    }

    /**
     * Writes configuration sidecar JSON with preprocessing filter settings and
     * Python Colab reproduction snippet.
     */
    fun writeConfigJson(stream: OutputStream, degPerUnit: Double, rateHz: Double = 1000.0, settings: FilterSettings = FilterSettings.DEFAULT) {
        val windowMs = settings.windowMs
        val pySnippet = "# Preprocessing pipeline matching UnEye labeler:\n" +
                "import numpy as np\n" +
                "import scipy.signal\n\n" +
                "# Apply Savitzky-Golay filter to test traces before passing to UnEye:\n" +
                "window_samples = int(round($windowMs * $rateHz / 1000.0)) | 1\n" +
                "X_clean = scipy.signal.savgol_filter(X, window_length=window_samples, polyorder=3, mode=\"interp\", axis=-1)\n" +
                "Y_clean = scipy.signal.savgol_filter(Y, window_length=window_samples, polyorder=3, mode=\"interp\", axis=-1)\n"

        val json = """
{
  "uneye_version": "2026.1",
  "rate_hz": ${formatDouble(rateHz)},
  "deg_per_unit": ${formatDouble(degPerUnit)},
  "filter_type": "savitzky_golay",
  "window_ms": $windowMs,
  "poly_order": ${settings.polyOrder},
  "adaptive_threshold_lambda": ${settings.lambda},
  "min_peak_vel_deg_s": ${settings.minPeakVel},
  "min_amplitude_deg": ${settings.minAmplitude},
  "min_duration_ms": ${settings.minDurationMs},
  "merge_gap_ms": ${settings.mergeGapMs},
  "python_preprocessing_code": ${escapeJson(pySnippet)}
}
        """.trimIndent()
        stream.bufferedWriter(Charsets.UTF_8).use { w ->
            w.write(json)
            w.newLine()
        }
    }

    private fun escapeJson(s: String): String {
        val sb = StringBuilder("\"")
        for (c in s) {
            when (c) {
                '\\' -> sb.append("\\\\")
                '"' -> sb.append("\\\"")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                else -> sb.append(c)
            }
        }
        sb.append("\"")
        return sb.toString()
    }

    // --- model ---------------------------------------------------------------------

    /** Copies the trained model .bin + .json sidecar to [dir]/[name]. Returns the .bin path. */
    fun exportModel(modelBin: String, dir: File, name: String): String {
        val srcBin = File(modelBin)
        val srcJson = File(modelBin.replaceAfterLast('.', "json", missingDelimiterValue = modelBin + ".json"))
        dir.mkdirs()
        val dstBin = File(dir, "$name.bin")
        val dstJson = File(dir, "$name.json")
        srcBin.copyTo(dstBin, overwrite = true)
        if (srcJson.exists()) srcJson.copyTo(dstJson, overwrite = true)
        return dstBin.absolutePath
    }

    private fun formatLabel(v: Double): String {
        if (v.isNaN()) return "nan"
        return v.toLong().toString()
    }

    private fun formatDouble(v: Double): String {
        if (v.isNaN()) return "nan"
        if (v.isInfinite()) return if (v > 0) "inf" else "-inf"
        // %.17g round-trips a double exactly; trim for readability
        return String.format(java.util.Locale.US, "%.17g", v)
    }
}
