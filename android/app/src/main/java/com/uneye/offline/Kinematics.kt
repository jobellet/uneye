package com.uneye.offline

import android.content.Context
import android.content.SharedPreferences
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Filter and detection settings (Savitzky-Golay cubic filter + adaptive threshold).
 * Matches the web labeler and Python scipy.signal.savgol_filter conventions.
 */
data class FilterSettings(
    val enabled: Boolean = true,
    val windowMs: Int = 21,           // Savitzky-Golay window in ms (odd integer >= 5)
    val polyOrder: Int = 3,           // Cubic polynomial
    val lambda: Double = 6.0,          // Adaptive threshold multiplier (Engbert-Kliegl lambda)
    val minPeakVel: Double = 50.0,     // Min peak speed (deg/s) to reject noise
    val minAmplitude: Double = 0.20,   // Min amplitude (deg)
    val minDurationMs: Double = 10.0,  // Min duration (ms)
    val mergeGapMs: Double = 20.0      // Merge events closer than this gap (ms)
) {
    companion object {
        private const val PREFS_NAME = "uneye_filter_prefs"
        private const val KEY_ENABLED = "filter_enabled"
        private const val KEY_WINDOW_MS = "window_ms"
        private const val KEY_LAMBDA = "lambda"
        private const val KEY_MIN_PEAK_VEL = "min_peak_vel"
        private const val KEY_MIN_AMPLITUDE = "min_amplitude"

        val DEFAULT = FilterSettings()

        fun load(context: Context): FilterSettings {
            val p = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            return FilterSettings(
                enabled = p.getBoolean(KEY_ENABLED, DEFAULT.enabled),
                windowMs = p.getInt(KEY_WINDOW_MS, DEFAULT.windowMs).coerceAtLeast(5).let { if (it % 2 == 0) it + 1 else it },
                polyOrder = 3,
                lambda = p.getFloat(KEY_LAMBDA, DEFAULT.lambda.toFloat()).toDouble(),
                minPeakVel = p.getFloat(KEY_MIN_PEAK_VEL, DEFAULT.minPeakVel.toFloat()).toDouble(),
                minAmplitude = p.getFloat(KEY_MIN_AMPLITUDE, DEFAULT.minAmplitude.toFloat()).toDouble(),
                minDurationMs = DEFAULT.minDurationMs,
                mergeGapMs = DEFAULT.mergeGapMs
            )
        }

        fun save(context: Context, s: FilterSettings) {
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
                .putBoolean(KEY_ENABLED, s.enabled)
                .putInt(KEY_WINDOW_MS, s.windowMs)
                .putFloat(KEY_LAMBDA, s.lambda.toFloat())
                .putFloat(KEY_MIN_PEAK_VEL, s.minPeakVel.toFloat())
                .putFloat(KEY_MIN_AMPLITUDE, s.minAmplitude.toFloat())
                .apply()
        }
    }
}

/**
 * Filter coefficients for Savitzky-Golay cubic fit.
 */
data class FilterCoeffs(val w: Int, val h: Int, val P: Array<DoubleArray>)

/**
 * Result of Savitzky-Golay filter applied to an eye trace.
 */
data class SavGolResult(val smooth: FloatArray, val d1: FloatArray, val d2: FloatArray)

/**
 * Full kinematics and noise characteristics for a trial.
 */
data class KinematicsResult(
    val smoothX: FloatArray,
    val smoothY: FloatArray,
    val vx: FloatArray,
    val vy: FloatArray,
    val speed: FloatArray,
    val tangentialAccel: FloatArray,
    val noiseSigma: Double,
    val threshold: Double
)

/**
 * Putative event detected by the adaptive Savitzky-Golay prelabeler.
 */
data class PutativeEvent(
    val onset: Long,
    val offset: Long,
    val cls: Int,
    val peakSpeed: Float = 0f,
    val amplitude: Float = 0f
)

/**
 * Savitzky-Golay cubic filter, derivative engine, and adaptive noise-scaled
 * velocity threshold detector for eye movements.
 *
 * Implements:
 * 1. Savitzky-Golay (cubic, scipy mode="interp") smoothing and derivatives.
 * 2. Robust median noise estimation: sigma_v = median(|v|) / 0.6745.
 * 3. Adaptive threshold: lambda * sigma_v (default lambda = 6.0).
 * 4. High-precision candidate detection: min peak velocity (50 deg/s),
 *    min amplitude (0.20 deg), sub-peak merging (< 20 ms), and
 *    tangential acceleration zero-crossing boundary refinement.
 */
object Kinematics {

    const val MAX_SPEED_DEG_S = 1300.0

    /**
     * Builds Savitzky-Golay cubic filter projection matrix (P = (V^T V)^-1 V^T).
     * [w] must be an odd integer >= 5.
     */
    fun makeSavitzkyGolay(w: Int): FilterCoeffs {
        require(w >= 5 && w % 2 == 1) { "Window w must be an odd integer >= 5" }
        val h = (w - 1) / 2
        val n = 4 // cubic: 1, x, x^2, x^3
        val vtV = Array(n) { DoubleArray(n) }
        val v = Array(w) { DoubleArray(n) }

        for (i in 0 until w) {
            val x = (i - h).toDouble()
            v[i][0] = 1.0
            v[i][1] = x
            v[i][2] = x * x
            v[i][3] = x * x * x
        }

        for (a in 0 until n) {
            for (b in 0 until n) {
                var sum = 0.0
                for (i in 0 until w) {
                    sum += v[i][a] * v[i][b]
                }
                vtV[a][b] = sum
            }
        }

        // Augmented matrix [VtV | I]
        val m = Array(n) { r ->
            val row = DoubleArray(8)
            for (c in 0 until n) row[c] = vtV[r][c]
            row[n + r] = 1.0
            row
        }

        // Gauss-Jordan elimination with partial pivoting
        for (c in 0 until n) {
            var p = c
            for (r in c + 1 until n) {
                if (abs(m[r][c]) > abs(m[p][c])) p = r
            }
            val tmp = m[c]
            m[c] = m[p]
            m[p] = tmp
            val d = m[c][c]
            for (k in 0 until 8) m[c][k] /= d
            for (r in 0 until n) {
                if (r != c) {
                    val f = m[r][c]
                    for (k in 0 until 8) m[r][k] -= f * m[c][k]
                }
            }
        }

        // P = (VtV)^-1 * V^T, size 4 x w
        val p = Array(n) { DoubleArray(w) }
        for (a in 0 until n) {
            for (i in 0 until w) {
                var acc = 0.0
                for (b in 0 until n) {
                    acc += m[a][n + b] * v[i][b]
                }
                p[a][i] = acc
            }
        }

        return FilterCoeffs(w, h, p)
    }

    /**
     * Applies Savitzky-Golay cubic filter (scipy mode="interp") to [data].
     */
    fun applySavitzkyGolay(data: FloatArray, valid: BooleanArray, fs: Double, sg: FilterCoeffs): SavGolResult {
        val t = data.size
        val smooth = FloatArray(t)
        val d1 = FloatArray(t)
        val d2 = FloatArray(t)
        val w = sg.w
        val h = sg.h
        val y = DoubleArray(t)
        var lastVal = 0.0

        for (i in 0 until t) {
            if (valid[i] && data[i].isFinite()) {
                lastVal = data[i].toDouble()
                y[i] = data[i].toDouble()
            } else {
                y[i] = lastVal
            }
        }

        for (i in 0 until t) {
            val s0 = (i - h).coerceIn(0, max(0, t - w))
            var c0 = 0.0; var c1 = 0.0; var c2 = 0.0; var c3 = 0.0
            for (j in 0 until w) {
                val v = y[s0 + j]
                c0 += sg.P[0][j] * v
                c1 += sg.P[1][j] * v
                c2 += sg.P[2][j] * v
                c3 += sg.P[3][j] * v
            }
            val u = (i - (s0 + h)).toDouble()
            smooth[i] = (c0 + c1 * u + c2 * u * u + c3 * u * u * u).toFloat()
            d1[i] = ((c1 + 2.0 * c2 * u + 3.0 * c3 * u * u) * fs).toFloat()
            d2[i] = ((2.0 * c2 + 6.0 * c3 * u) * fs * fs).toFloat()
        }

        return SavGolResult(smooth, d1, d2)
    }

    /**
     * Robust noise estimation on detrended velocity: median(|v|) / 0.6745.
     */
    fun computeRobustSigma(arr: FloatArray, valid: BooleanArray): Double {
        val vals = ArrayList<Double>(arr.size)
        for (i in arr.indices) {
            if (valid[i] && arr[i].isFinite()) {
                vals.add(abs(arr[i].toDouble()))
            }
        }
        if (vals.isEmpty()) return 1.0
        vals.sort()
        val med = vals[vals.size / 2]
        return max(med / 0.6745, 1e-4)
    }

    /**
     * Computes smoothed traces, velocities, tangential acceleration, and adaptive threshold.
     */
    fun computeKinematics(
        x: FloatArray, y: FloatArray, valid: BooleanArray,
        fs: Double, degPerUnit: Double, cfg: FilterSettings
    ): KinematicsResult {
        val n = x.size
        if (n == 0) {
            return KinematicsResult(
                FloatArray(0), FloatArray(0), FloatArray(0), FloatArray(0),
                FloatArray(0), FloatArray(0), 1.0, 12.0
            )
        }

        if (!cfg.enabled) {
            val vx = FloatArray(n)
            val vy = FloatArray(n)
            val speed = FloatArray(n)
            val dt = 1.0 / fs
            for (i in 1 until n) {
                if (valid[i] && valid[i - 1]) {
                    var dx = (x[i] - x[i - 1]) * degPerUnit.toFloat()
                    var dy = (y[i] - y[i - 1]) * degPerUnit.toFloat()
                    var sp = hypot(dx.toDouble(), dy.toDouble()).toFloat() / dt.toFloat()
                    if (sp > MAX_SPEED_DEG_S) {
                        val f = (MAX_SPEED_DEG_S / sp).toFloat()
                        dx *= f; dy *= f; sp = MAX_SPEED_DEG_S.toFloat()
                    }
                    vx[i] = dx / dt.toFloat()
                    vy[i] = dy / dt.toFloat()
                    speed[i] = sp
                }
            }
            val sX = computeRobustSigma(vx, valid)
            val sY = computeRobustSigma(vy, valid)
            val ns = max(0.5 * (sX + sY), 0.5)
            val th = max(cfg.lambda * ns, 12.0)
            return KinematicsResult(x, y, vx, vy, speed, FloatArray(n), ns, th)
        }

        var w = (cfg.windowMs * fs / 1000.0).roundToInt()
        if (w % 2 == 0) w += 1
        w = max(5, min(w, max(5, ((n / 2) * 2 - 1))))
        val sg = makeSavitzkyGolay(w)

        val rx = applySavitzkyGolay(x, valid, fs, sg)
        val ry = applySavitzkyGolay(y, valid, fs, sg)

        val vx = FloatArray(n)
        val vy = FloatArray(n)
        val speed = FloatArray(n)
        val tangentialAccel = FloatArray(n)
        val sc = degPerUnit.toFloat()

        for (i in 0 until n) {
            if (valid[i]) {
                var vxi = rx.d1[i] * sc
                var vyi = ry.d1[i] * sc
                var sp = hypot(vxi.toDouble(), vyi.toDouble()).toFloat()
                if (sp > MAX_SPEED_DEG_S) {
                    val f = (MAX_SPEED_DEG_S / sp).toFloat()
                    vxi *= f; vyi *= f; sp = MAX_SPEED_DEG_S.toFloat()
                }
                vx[i] = vxi
                vy[i] = vyi
                speed[i] = sp

                val ax = rx.d2[i] * sc
                val ay = ry.d2[i] * sc
                tangentialAccel[i] = if (sp > 1e-6f) (vxi * ax + vyi * ay) / sp else 0f
            }
        }

        val sigmaX = computeRobustSigma(vx, valid)
        val sigmaY = computeRobustSigma(vy, valid)
        val noiseSigma = max(0.5 * (sigmaX + sigmaY), 0.5)
        val threshold = max(cfg.lambda * noiseSigma, 12.0)

        return KinematicsResult(rx.smooth, ry.smooth, vx, vy, speed, tangentialAccel, noiseSigma, threshold)
    }

    /**
     * Detects putative blinks and saccades with high-precision criteria.
     */
    fun detectPutativeEvents(
        x: FloatArray, y: FloatArray, valid: BooleanArray,
        fs: Double, degPerUnit: Double, cfg: FilterSettings
    ): List<PutativeEvent> {
        val n = x.size
        if (n == 0) return emptyList()

        val pre = ArrayList<PutativeEvent>()

        // 1. Detect blinks from invalid runs (>= 3 samples)
        var blinkStart = -1
        for (i in 0..n) {
            val isInvalid = i < n && (!valid[i] || !x[i].isFinite() || !y[i].isFinite())
            if (isInvalid && blinkStart == -1) {
                blinkStart = i
            } else if (!isInvalid && blinkStart != -1) {
                val len = i - blinkStart
                if (len >= 3) {
                    pre.add(PutativeEvent(blinkStart.toLong(), (i - 1).toLong(), Native.CLS_BLINK))
                }
                blinkStart = -1
            }
        }

        // Kinematics & adaptive threshold
        val kin = computeKinematics(x, y, valid, fs, degPerUnit, cfg)
        val threshold = kin.threshold
        val minDurSamples = max(4, (cfg.minDurationMs * fs / 1000.0).roundToInt())
        val mergeGapSamples = (cfg.mergeGapMs * fs / 1000.0).roundToInt()

        // Candidate runs above threshold
        data class Run(val start: Int, var end: Int)
        val rawSaccades = ArrayList<Run>()
        var sacStart = -1
        for (i in 0..n) {
            val isSac = i < n && valid[i] && kin.speed[i] >= threshold
            if (isSac && sacStart == -1) {
                sacStart = i
            } else if (!isSac && sacStart != -1) {
                val sacEnd = i - 1
                if (sacEnd - sacStart + 1 >= minDurSamples) {
                    rawSaccades.add(Run(sacStart, sacEnd))
                }
                sacStart = -1
            }
        }

        // Merge adjacent sub-events separated by less than mergeGap
        val merged = ArrayList<Run>()
        for (ev in rawSaccades) {
            if (merged.isNotEmpty() && ev.start - merged.last().end <= mergeGapSamples) {
                merged.last().end = ev.end
            } else {
                merged.add(Run(ev.start, ev.end))
            }
        }

        // Refine boundaries and apply acceptance criteria
        val marginSamples = (25.0 * fs / 1000.0).roundToInt()
        val floorSpeed = (2.5 * kin.noiseSigma).toFloat()

        for (ev in merged) {
            val s = ev.start
            val e = ev.end

            // Peak speed in candidate
            var pk = s
            var maxSp = kin.speed[s]
            for (t in s..e) {
                if (kin.speed[t] > maxSp) {
                    maxSp = kin.speed[t]
                    pk = t
                }
            }

            // CRITERION 1: High Accuracy filter — must reach minPeakVel (50 deg/s) & 1.5 * threshold
            if (maxSp < max(cfg.minPeakVel, 1.5 * threshold).toFloat()) continue

            // Boundary refinement via tangential acceleration zero-crossing
            val lo = max(0, s - marginSamples)
            val hi = min(n - 1, e + marginSamples)

            // Onset: walk backward while acceleration > 0
            var j = pk - 1
            while (j >= lo && kin.tangentialAccel[j] > 0) j--
            val on = min(max(lo, j + 1), pk)

            // Offset: walk forward until speed drops below floor, then zero crossing
            var lastAbove = pk
            for (t in pk..hi) {
                if (kin.speed[t] >= floorSpeed) lastAbove = t
            }
            j = lastAbove
            while (j <= hi && kin.tangentialAccel[j] < 0) j++
            val off = min(hi, max(pk, j))

            // CRITERION 2: Minimum Amplitude
            val dx = (kin.smoothX[off] - kin.smoothX[on]) * degPerUnit.toFloat()
            val dy = (kin.smoothY[off] - kin.smoothY[on]) * degPerUnit.toFloat()
            val amp = hypot(dx.toDouble(), dy.toDouble()).toFloat()
            if (amp < cfg.minAmplitude.toFloat()) continue

            // CRITERION 3: Do not overlap blinks
            val overlapsBlink = pre.any { b -> b.cls == Native.CLS_BLINK && !(off < b.onset || on > b.offset) }
            if (!overlapsBlink) {
                pre.add(PutativeEvent(on.toLong(), off.toLong(), Native.CLS_SACCADE, maxSp, amp))
            }
        }

        return pre.sortedBy { it.onset }
    }
}
