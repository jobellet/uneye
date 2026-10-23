package com.uneye.offline

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class KinematicsTest {

    @Test
    fun testMakeSavitzkyGolayCoefficients() {
        val w = 21
        val sg = Kinematics.makeSavitzkyGolay(w)
        assertEquals(21, sg.w)
        assertEquals(10, sg.h)
        assertEquals(4, sg.P.size)
        assertEquals(21, sg.P[0].size)

        // Smoothing coefficients (row 0) must sum to 1.0 (constant conservation)
        var sum0 = 0.0
        for (v in sg.P[0]) sum0 += v
        assertEquals(1.0, sum0, 1e-6)

        // Derivative coefficients (row 1) must sum to 0.0 (constant has 0 derivative)
        var sum1 = 0.0
        for (v in sg.P[1]) sum1 += v
        assertEquals(0.0, sum1, 1e-6)

        // Smoothing coefficients must be symmetric around center h = 10
        for (i in 0 until 10) {
            assertEquals(sg.P[0][i], sg.P[0][w - 1 - i], 1e-6)
        }
    }

    @Test
    fun testApplySavitzkyGolayOnFlatSignal() {
        val w = 21
        val sg = Kinematics.makeSavitzkyGolay(w)
        val n = 100
        val x = FloatArray(n) { 5.0f }
        val valid = BooleanArray(n) { true }
        val res = Kinematics.applySavitzkyGolay(x, valid, 1000.0, sg)

        // Values should remain 5.0
        for (i in 0 until n) {
            assertEquals(5.0f, res.smooth[i], 1e-4f)
            assertEquals(0.0f, res.d1[i], 1e-4f)
        }
    }

    @Test
    fun testApplySavitzkyGolayOnLinearRamp() {
        val w = 21
        val sg = Kinematics.makeSavitzkyGolay(w)
        val n = 100
        val fs = 1000.0
        // Ramp: x(t) = 10 * (t / fs) -> velocity = 10 units/s
        val x = FloatArray(n) { (10.0 * it / fs).toFloat() }
        val valid = BooleanArray(n) { true }
        val res = Kinematics.applySavitzkyGolay(x, valid, fs, sg)

        // Away from edges, derivative should be ~10 units/s
        for (i in 15 until 85) {
            assertEquals(10.0f, res.d1[i], 0.05f)
        }
    }

    @Test
    fun testComputeRobustSigma() {
        // Uniform noise between -1 and +1 -> median of absolute values is 0.5 -> sigma ~ 0.5 / 0.6745 ~ 0.741
        val n = 1000
        val v = FloatArray(n) { ((it.toDouble() / n) * 2.0 - 1.0).toFloat() }
        val valid = BooleanArray(n) { true }
        val sigma = Kinematics.computeRobustSigma(v, valid)
        assertEquals(0.5 / 0.6745, sigma, 0.05)
    }

    @Test
    fun testDetectPutativeEventsOnSyntheticTrace() {
        val n = 1000
        val fs = 1000.0
        val x = FloatArray(n)
        val y = FloatArray(n)
        val valid = BooleanArray(n) { true }

        // Low baseline fixation noise
        for (i in 0 until n) {
            x[i] = (kotlin.math.sin(i * 0.05) * 0.02).toFloat()
            y[i] = (kotlin.math.cos(i * 0.05) * 0.02).toFloat()
        }

        // Add one real saccade at t = 300..340 ms (40 ms duration, amplitude 2.0 deg, peak speed > 70 deg/s)
        val sacStart = 300
        val sacEnd = 340
        for (i in sacStart..sacEnd) {
            val progress = (i - sacStart).toDouble() / (sacEnd - sacStart)
            // Smooth half-cosine rise: d/dt peaks at ~78 deg/s
            val sStep = 0.5 * (1.0 - kotlin.math.cos(progress * Math.PI))
            x[i] += (2.0 * sStep).toFloat()
        }
        for (i in sacEnd + 1 until n) {
            x[i] += 2.0f
        }

        val cfg = FilterSettings.DEFAULT
        val events = Kinematics.detectPutativeEvents(x, y, valid, fs, 1.0, cfg)

        // Must detect the saccade
        val saccades = events.filter { it.cls == Native.CLS_SACCADE }
        assertEquals(1, saccades.size)
        val s = saccades[0]
        assertTrue("Onset should be near 300 ms, was ${s.onset}", abs(s.onset - sacStart) <= 10)
        assertTrue("Offset should be near 340 ms, was ${s.offset}", abs(s.offset - sacEnd) <= 10)
        assertTrue("Peak speed should be > 50 deg/s, was ${s.peakSpeed}", s.peakSpeed >= 50f)
        assertTrue("Amplitude should be >= 0.20 deg, was ${s.amplitude}", s.amplitude >= 0.20f)
    }

    @Test
    fun testRejectsLowVelocityNoiseJitter() {
        val n = 1000
        val fs = 1000.0
        val x = FloatArray(n)
        val y = FloatArray(n)
        val valid = BooleanArray(n) { true }

        // Small tremor jitter at 500 ms (amplitude 0.05 deg, low speed)
        for (i in 500..520) {
            x[i] = 0.05f
        }

        val cfg = FilterSettings.DEFAULT
        val events = Kinematics.detectPutativeEvents(x, y, valid, fs, 1.0, cfg)

        // Saccade should NOT be detected because amplitude < 0.20 deg and peak speed < 50 deg/s
        val saccades = events.filter { it.cls == Native.CLS_SACCADE }
        assertEquals(0, saccades.size)
    }
}
