package com.uneye.offline

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * The convergence history (phase 4): one JSON file with one entry per
 * training round. The flag rule (phase 4) reads this history: silent until
 * the test set has >= 100 saccades and >= 20 blinks, "converged" when the
 * gain over the last N rounds is inside the bootstrap CI. The trainer's
 * val_event_f1 is biased (it chose the model) and is recorded for reference
 * only — never used for the flag.
 */
class ConvergenceStore(private val file: File) {

    data class Round(
        val atMs: Long,
        val labeledSeconds: Double,
        val epochs: Int,
        val valF1: Double,            // biased: early-stopping score, reference only
        val valKappa: Double,
        val testF1Saccade: Double,
        val testF1Blink: Double,
        val testKappaSaccade: Double,
        val testKappaBlink: Double,
        val nSaccades: Int,           // expert events in the fixed test set
        val nBlinks: Int,
        val modelPath: String,
    )

    @Synchronized
    fun appendRound(r: Round) {
        val arr = readArray()
        arr.put(roundToJson(r))
        file.parentFile?.mkdirs()
        file.writeText(arr.toString(2))
    }

    @Synchronized
    fun rounds(): List<Round> {
        val arr = readArray()
        return (0 until arr.length()).map { jsonToRound(arr.getJSONObject(it)) }
    }

    @Synchronized
    fun last(): Round? = rounds().lastOrNull()

    private fun readArray(): JSONArray {
        if (!file.exists()) return JSONArray()
        return try { JSONArray(file.readText()) } catch (_: Exception) { JSONArray() }
    }

    private fun roundToJson(r: Round) = JSONObject().apply {
        put("at_ms", r.atMs)
        put("labeled_seconds", r.labeledSeconds)
        put("epochs", r.epochs)
        put("val_f1", r.valF1)
        put("val_kappa", r.valKappa)
        put("test_f1_saccade", r.testF1Saccade)
        put("test_f1_blink", r.testF1Blink)
        put("test_kappa_saccade", r.testKappaSaccade)
        put("test_kappa_blink", r.testKappaBlink)
        put("n_saccades", r.nSaccades)
        put("n_blinks", r.nBlinks)
        put("model_path", r.modelPath)
    }

    private fun jsonToRound(j: JSONObject) = Round(
        atMs = j.optLong("at_ms"),
        labeledSeconds = j.optDouble("labeled_seconds"),
        epochs = j.optInt("epochs"),
        valF1 = j.optDouble("val_f1"),
        valKappa = j.optDouble("val_kappa"),
        testF1Saccade = j.optDouble("test_f1_saccade"),
        testF1Blink = j.optDouble("test_f1_blink"),
        testKappaSaccade = j.optDouble("test_kappa_saccade"),
        testKappaBlink = j.optDouble("test_kappa_blink"),
        nSaccades = j.optInt("n_saccades"),
        nBlinks = j.optInt("n_blinks"),
        modelPath = j.optString("model_path"),
    )
}
