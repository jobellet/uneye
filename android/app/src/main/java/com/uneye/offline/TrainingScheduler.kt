package com.uneye.offline

import android.content.Context
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import java.util.concurrent.TimeUnit

/**
 * Schedules the background training (phase 3). A training run is enqueued
 * after every [newlyLabeledSeconds] newly reviewed seconds, coalesced through
 * a unique work name so overlapping requests collapse into one run.
 *
 * Constraints: the default run asks for the device to be idle and charging
 * only when the data set is large (> [largeDataSeconds]); small runs go with
 * battery-not-low only. The WorkManager network constraint is unused (all
 * offline) but kept off explicitly.
 */
object TrainingScheduler {

    private const val UNIQUE_NAME = "uneye_training"
    private const val LARGE_DATA_SECONDS = 120.0

    /** Call from the labeling screen whenever reviewed seconds grow. */
    fun maybeSchedule(context: Context, reviewedSeconds: Double, newlyLabeledSinceLastRun: Double, minNewSeconds: Double = 20.0) {
        if (newlyLabeledSinceLastRun < minNewSeconds) return
        val large = reviewedSeconds >= LARGE_DATA_SECONDS
        val constraints = Constraints.Builder()
            .setRequiresBatteryNotLow(true)
            .setRequiresDeviceIdle(true)
            .apply { if (large) setRequiresCharging(true) }
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .build()
        val request = OneTimeWorkRequestBuilder<TrainingWorker>()
            .setConstraints(constraints)
            .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 1, TimeUnit.MINUTES)
            .addTag("training")
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            UNIQUE_NAME, ExistingWorkPolicy.KEEP, request,
        )
    }

    /** Manual trigger from the UI ("Train now"). Runs without the idle/charging constraints. */
    fun scheduleNow(context: Context) {
        val constraints = Constraints.Builder()
            .setRequiresBatteryNotLow(true)
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .build()
        val request = OneTimeWorkRequestBuilder<TrainingWorker>()
            .setConstraints(constraints)
            .addTag("training")
            .build()
        WorkManager.getInstance(context).enqueueUniqueWork(
            UNIQUE_NAME, ExistingWorkPolicy.REPLACE, request,
        )
    }

    fun cancel(context: Context) {
        WorkManager.getInstance(context).cancelUniqueWork(UNIQUE_NAME)
    }
}
