package com.uneye.offline

import kotlin.math.abs
import kotlin.random.Random

/**
 * The convergence flag (phase 4). Reads the round history of
 * [ConvergenceStore] and answers one question: has the model stopped
 * improving against the expert's ground truth on the fixed test set?
 *
 * Rule (from the plan):
 *  - SILENT until the test set has a minimum size: >= [minSaccades] saccades
 *    and >= [minBlinks] blinks (expert events in the fixed test set).
 *  - Otherwise, "converged" when the gain over the last [window] rounds is
 *    inside the bootstrap confidence interval of that gain.
 *
 * The gain is computed on the saccade test F1 (class 1, the class the app is
 * built for); the blink F1 is reported alongside. The trainer's val F1 is
 * biased (it was used to choose the model) and is NEVER used here.
 *
 * Bootstrap: resample the per-round gains with replacement [nBoot] times,
 * take the [ci] central interval of the resampled means. The gain over the
 * last `window` rounds is the mean of the last `window` per-round gains
 * (round i gain = F1[i] - F1[i-1]); "inside the CI" means |gain| <= the CI
 * half-width, i.e. the interval of the mean gain contains 0.
 */
object ConvergenceFlag {

    const val MIN_SACCADES = 100   // min expert saccades in the fixed test set
    const val MIN_BLINKS = 20
    const val WINDOW = 3          // gain over the last N rounds
    const val N_BOOT = 2000
    const val CI = 0.95

    enum class State { SILENT, COLLECTING, CONVERGED }

    data class Result(
        val state: State,
        val nSaccades: Int,
        val nBlinks: Int,
        val gain: Double,             // mean gain of the last WINDOW rounds (saccade F1)
        val ciHalfWidth: Double,      // bootstrap CI half-width of the mean gain
        val f1Saccade: Double,        // latest saccade test F1
        val kappaSaccade: Double,
        val f1Blink: Double,
        val kappaBlink: Double,
        val rounds: Int,
    )

    /** Evaluates the flag on the round history. Pure function of the history. */
    fun evaluate(rounds: List<ConvergenceStore.Round>, seed: Long = 42): Result {
        if (rounds.isEmpty()) return Result(State.SILENT, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        val last = rounds.last()
        // 1. minimum test-set size: silent until then
        if (last.nSaccades < MIN_SACCADES || last.nBlinks < MIN_BLINKS) {
            return Result(State.SILENT, last.nSaccades, last.nBlinks, 0.0, 0.0,
                last.testF1Saccade, last.testKappaSaccade, last.testF1Blink, last.testKappaBlink, rounds.size)
        }
        // 2. per-round gains of the saccade test F1
        val gains = rounds.zipWithNext { a, b -> b.testF1Saccade - a.testF1Saccade }
        if (gains.size < WINDOW) {
            return Result(State.COLLECTING, last.nSaccades, last.nBlinks, 0.0, 0.0,
                last.testF1Saccade, last.testKappaSaccade, last.testF1Blink, last.testKappaBlink, rounds.size)
        }
        val windowGains = gains.takeLast(WINDOW)
        val gain = windowGains.average()
        // 3. bootstrap CI of the mean gain over the window
        val ciHalf = bootstrapCiHalfWidth(windowGains, N_BOOT, CI, seed)
        // 4. converged when the gain is inside the interval (the interval of the
        //    mean gain contains 0 <=> |gain| <= ciHalf)
        val state = if (abs(gain) <= ciHalf) State.CONVERGED else State.COLLECTING
        return Result(state, last.nSaccades, last.nBlinks, gain, ciHalf,
            last.testF1Saccade, last.testKappaSaccade, last.testF1Blink, last.testKappaBlink, rounds.size)
    }

    /**
     * Bootstrap CI half-width of the mean of [gains]: resample with
     * replacement [nBoot] times, the CI is the central [ci] interval of the
     * resampled means, the half-width is (upper - lower) / 2.
     */
    internal fun bootstrapCiHalfWidth(gains: List<Double>, nBoot: Int, ci: Double, seed: Long): Double {
        if (gains.isEmpty()) return 0.0
        val rng = Random(seed)
        val n = gains.size
        val means = DoubleArray(nBoot)
        repeat(nBoot) { b ->
            var s = 0.0
            repeat(n) { s += gains[rng.nextInt(n)] }
            means[b] = s / n
        }
        means.sort()
        val alpha = (1.0 - ci) / 2.0
        val lo = means[(alpha * nBoot).toInt().coerceIn(0, nBoot - 1)]
        val hi = means[((1.0 - alpha) * nBoot).toInt().coerceIn(0, nBoot - 1)]
        return (hi - lo) / 2.0
    }
}
