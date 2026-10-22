package com.uneye.offline

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

/**
 * Learning-curve view (phase 4): plots the test-set event F1 and kappa per
 * class against the reviewed (labeled) seconds, from the round history of
 * [ConvergenceStore]. Read-only; the data is pushed by the screen.
 *
 * Left half = F1 (class 1 saccade blue, class 2 blink red), right half =
 * kappa (same colors). The x axis is the reviewed seconds at each round.
 */
class MetricsView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
    View(context, attrs) {

    data class Series(val seconds: DoubleArray, val values: DoubleArray)

    var f1Saccade: Series? = null
    var f1Blink: Series? = null
    var kappaSaccade: Series? = null
    var kappaBlink: Series? = null
    var convergedAtSeconds: Double = -1.0     // vertical marker when the flag fired

    private val paintSaccade = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 4f; style = Paint.Style.STROKE; color = 0xFF1565C0.toInt() }
    private val paintBlink = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 4f; style = Paint.Style.STROKE; color = 0xFFC62828.toInt() }
    private val paintGrid = Paint().apply { strokeWidth = 1f; color = 0x33000000 }
    private val paintText = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 26f; color = Color.DKGRAY }
    private val paintMarker = Paint().apply { strokeWidth = 3f; color = 0xFF2E7D32.toInt(); pathEffect = android.graphics.DashPathEffect(floatArrayOf(10f, 8f), 0f) }

    fun setSeries(rounds: List<ConvergenceStore.Round>, converged: Boolean) {
        if (rounds.isEmpty()) {
            f1Saccade = null; f1Blink = null; kappaSaccade = null; kappaBlink = null
        } else {
            val s = rounds.map { it.labeledSeconds }.toDoubleArray()
            f1Saccade = Series(s, rounds.map { it.testF1Saccade }.toDoubleArray())
            f1Blink = Series(s, rounds.map { it.testF1Blink }.toDoubleArray())
            kappaSaccade = Series(s, rounds.map { it.testKappaSaccade }.toDoubleArray())
            kappaBlink = Series(s, rounds.map { it.testKappaBlink }.toDoubleArray())
            convergedAtSeconds = if (converged) rounds.last().labeledSeconds else -1.0
        }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return
        val midX = width / 2f
        drawGrid(canvas, 0f, midX)
        drawGrid(canvas, midX, width.toFloat())
        canvas.drawText(context.getString(R.string.metrics_f1), 16f, 34f, paintText)
        canvas.drawText(context.getString(R.string.metrics_kappa), midX + 16f, 34f, paintText)
        canvas.drawText(context.getString(R.string.metrics_saccade), 16f, height - 12f, paintText.apply { color = 0xFF1565C0.toInt() })
        canvas.drawText(context.getString(R.string.metrics_blink), midX + 16f, height - 12f, paintText.apply { color = 0xFFC62828.toInt() })
        drawSeries(canvas, f1Saccade, 0f, midX, paintSaccade)
        drawSeries(canvas, f1Blink, 0f, midX, paintBlink)
        drawSeries(canvas, kappaSaccade, midX, width.toFloat(), paintSaccade)
        drawSeries(canvas, kappaBlink, midX, width.toFloat(), paintBlink)
        if (convergedAtSeconds >= 0) {
            val all = listOfNotNull(f1Saccade, f1Blink, kappaSaccade, kappaBlink)
            val sMax = all.flatMap { it.seconds.toList() }.maxOrNull() ?: 1.0
            val px = (convergedAtSeconds / sMax.coerceAtLeast(1e-9)).toFloat() * width
            canvas.drawLine(px, 40f, px, height - 40f, paintMarker)
        }
    }

    private fun drawGrid(canvas: Canvas, left: Float, right: Float) {
        val top = 48f; val bottom = height - 44f
        canvas.drawRect(left, top, right, bottom, paintGrid)
        for (f in 0..4) {
            val y = top + (bottom - top) * f / 4f
            canvas.drawLine(left, y, right, y, paintGrid)
            canvas.drawText(String.format("%.1f", 1.0 - 0.25 * f), left + 4, y - 4, paintText.apply { textSize = 20f })
        }
        canvas.drawLine((left + right) / 2, top, (left + right) / 2, bottom, paintGrid)
    }

    private fun drawSeries(canvas: Canvas, series: Series?, left: Float, right: Float, paint: Paint) {
        val s = series ?: return
        if (s.seconds.size < 2) return
        val sMax = s.seconds.maxOrNull() ?: 1.0
        val top = 48f; val bottom = height - 44f
        val path = Path()
        for (i in s.seconds.indices) {
            val px = left + (right - left) * (s.seconds[i] / sMax.coerceAtLeast(1e-9)).toFloat()
            val v = s.values[i].coerceIn(0.0, 1.0)
            val py = bottom - (bottom - top) * v.toFloat()
            if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        canvas.drawPath(path, paint)
    }
}
