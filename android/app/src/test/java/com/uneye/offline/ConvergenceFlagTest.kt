package com.uneye.offline

import org.junit.Assert.assertEquals
import org.junit.Test

class ConvergenceFlagTest {

    private fun makeRound(
        saccades: Int, blinks: Int, testF1Saccade: Double, roundIdx: Int = 1,
    ) = ConvergenceStore.Round(
        atMs = 1000L * roundIdx,
        labeledSeconds = 10.0 * roundIdx,
        epochs = 10,
        valF1 = 0.9,
        valKappa = 0.85,
        testF1Saccade = testF1Saccade,
        testF1Blink = 0.95,
        testKappaSaccade = 0.80,
        testKappaBlink = 0.90,
        nSaccades = saccades,
        nBlinks = blinks,
        modelPath = "dummy.bin",
    )

    @Test
    fun testEmptyRoundsSilent() {
        val res = ConvergenceFlag.evaluate(emptyList())
        assertEquals(ConvergenceFlag.State.SILENT, res.state)
    }

    @Test
    fun testTooFewSaccadesSilent() {
        val rounds = listOf(
            makeRound(saccades = 50, blinks = 30, testF1Saccade = 0.80, 1),
            makeRound(saccades = 50, blinks = 30, testF1Saccade = 0.81, 2),
            makeRound(saccades = 50, blinks = 30, testF1Saccade = 0.81, 3),
            makeRound(saccades = 50, blinks = 30, testF1Saccade = 0.81, 4),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.SILENT, res.state)
    }

    @Test
    fun testTooFewBlinksSilent() {
        val rounds = listOf(
            makeRound(saccades = 120, blinks = 15, testF1Saccade = 0.80, 1),
            makeRound(saccades = 120, blinks = 15, testF1Saccade = 0.81, 2),
            makeRound(saccades = 120, blinks = 15, testF1Saccade = 0.81, 3),
            makeRound(saccades = 120, blinks = 15, testF1Saccade = 0.81, 4),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.SILENT, res.state)
    }

    @Test
    fun testFewRoundsCollecting() {
        // Thresholds met, but only 2 rounds (1 gain, window requires 3 gains / 4 rounds)
        val rounds = listOf(
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.80, 1),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.81, 2),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.COLLECTING, res.state)
    }

    @Test
    fun testSharpImprovementCollecting() {
        // Strong improvement (+0.12 per round)
        val rounds = listOf(
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.50, 1),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.62, 2),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.74, 3),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.86, 4),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.COLLECTING, res.state)
    }

    @Test
    fun testPlateauConverged() {
        // Stable F1 (~0.81)
        val rounds = listOf(
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.78, 1),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.810, 2),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.811, 3),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.810, 4),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.CONVERGED, res.state)
    }

    @Test
    fun testNoisyPlateauConverged() {
        // Fluctuations around +/- 0.002
        val rounds = listOf(
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.810, 1),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.812, 2),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.809, 3),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.811, 4),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.CONVERGED, res.state)
    }

    @Test
    fun testSlightDeclineConverged() {
        // -0.001 slight decline
        val rounds = listOf(
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.810, 1),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.809, 2),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.810, 3),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.808, 4),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.CONVERGED, res.state)
    }

    @Test
    fun testBigDropCollecting() {
        // Major drop (-0.05) is not converged
        val rounds = listOf(
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.810, 1),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.760, 2),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.710, 3),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.660, 4),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.COLLECTING, res.state)
    }

    @Test
    fun testConstantF1Converged() {
        // Exactly identical F1 across rounds
        val rounds = listOf(
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.850, 1),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.850, 2),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.850, 3),
            makeRound(saccades = 120, blinks = 25, testF1Saccade = 0.850, 4),
        )
        val res = ConvergenceFlag.evaluate(rounds)
        assertEquals(ConvergenceFlag.State.CONVERGED, res.state)
    }
}
