package com.uneye.offline

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Touch-optimized trace view (phase 2). Draws, for one epoch of one trial:
 *  - the X (top lane) and Y (bottom lane) position traces, plus the speed lane,
 *  - the rendered labels as a background band (reviewed only),
 *  - the invalid (blink / lost) samples as grey gaps,
 *  - the pre-labels as dimmed vertical marks, the point marks (pass 1) as
 *    triangles, the expert events (pass 2) as brackets,
 *  - a magnifier loupe above the finger while dragging (pass 2).
 *
 * Gestures:
 *  - single tap: [onTap(sample)] — pass 1 places / removes a point mark,
 *    pass 2 double-tap accepts the pre-label boundaries.
 *  - long press + drag: pass 2 onset/offset refinement (press = onset,
 *    lift = offset), reported through [onDragSpan].
 *  - two-finger pinch / vertical drag: zoom (time axis) and pan.
 *
 * All label decisions are delegated to the callbacks; the view keeps no label
 * state (the C++ ExpertLabels owns it).
 */
class TraceView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
    View(context, attrs) {

    // --- data (set by the screen before invalidate) --------------------------------

    data class Lane(val values: FloatArray, val invalid: BooleanArray? = null)

    var fromSample: Long = 0
        private set
    var toSample: Long = 999
        private set
    var xLane: Lane? = null
        private set
    var yLane: Lane? = null
        private set
    var speedLane: Lane? = null
        private set
    var labelBand: ByteArray? = null          // per sample, aligned with [fromSample]
    var reviewedBand: BooleanArray? = null     // true where the sample is reviewed
    var prelabels: List<Pair<Long, Long>> = emptyList()     // (onset, offset) dimmed
    var prelabelCls: List<Int> = emptyList()
    var pointMarks: List<Pair<Long, Int>> = emptyList()     // (at, cls)
    var expertMarks: List<Triple<Long, Long, Int>> = emptyList()  // (onset, offset, cls)
    var cursorSample: Long = -1               // drawn as a vertical line (pass 2 finger)
    var showLoupe: Boolean = false

    // --- callbacks -------------------------------------------------------------------

    var onTapSample: ((Long) -> Unit)? = null
    var onDoubleTapSample: ((Long) -> Unit)? = null
    var onLongPressSample: ((Long) -> Unit)? = null
    var onDragSpan: ((onset: Long, offset: Long) -> Unit)? = null
    var onScrolled: ((from: Long, to: Long) -> Unit)? = null   // pan / zoom changed

    // --- colors ------------------------------------------------------------------------

    private val paintTrace = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 3f; style = Paint.Style.STROKE }
    private val paintSpeed = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 2f; style = Paint.Style.STROKE; color = 0xFF7E57C2.toInt() }
    private val paintLabelBand = Paint()
    private val paintInvalid = Paint().apply { color = 0x33000000 }
    private val paintPre = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 4f; style = Paint.Style.STROKE; color = 0x66FFB300.toInt() }
    private val paintPoint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val paintExpert = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 5f; style = Paint.Style.STROKE }
    private val paintCursor = Paint().apply { strokeWidth = 4f; color = 0xFFD32F2F.toInt() }
    private val paintGrid = Paint().apply { strokeWidth = 1f; color = 0x22000000 }
    private val paintLoupeBg = Paint().apply { color = 0xCCFFFFFF.toInt() }
    private val paintLoupeStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 3f; color = 0xFF1565C0.toInt(); style = Paint.Style.STROKE }
    private val paintText = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 30f; color = Color.DKGRAY }

    private var dragStartSample = -1L
    private var dragCurSample = -1L
    private var dragging = false

    private val gestureDetector = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(e: MotionEvent): Boolean = true
        override fun onSingleTapUp(e: MotionEvent): Boolean {
            onTapSample?.invoke(sampleAt(e.x))
            return true
        }
        override fun onDoubleTap(e: MotionEvent): Boolean {
            onDoubleTapSample?.invoke(sampleAt(e.x))
            return true
        }
        override fun onLongPress(e: MotionEvent) {
            onLongPressSample?.invoke(sampleAt(e.x))
        }
    })

    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            zoomBy(d.scaleFactor, d.focusX)
            return true
        }
    })

    // --- data injection ------------------------------------------------------------------

    /** Sets the visible window and the lanes; call before invalidate(). */
    fun setWindow(from: Long, to: Long, x: Lane?, y: Lane?, speed: Lane?) {
        fromSample = from; toSample = to
        xLane = x; yLane = y; speedLane = speed
        invalidate()
    }

    fun samplesPerPixel(): Float = (toSample - fromSample + 1).toFloat() / max(1, width)

    private fun sampleAt(px: Float): Long {
        if (width <= 0) return fromSample
        val f = (px / width.toFloat()).coerceIn(0f, 1f)
        return fromSample + (f * (toSample - fromSample)).toLong()
    }

    private fun pxOf(sample: Long): Float {
        if (toSample <= fromSample) return 0f
        return (sample - fromSample).toFloat() / (toSample - fromSample) * width
    }

    // --- touch ----------------------------------------------------------------------------

    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                dragStartSample = sampleAt(event.x)
                dragCurSample = dragStartSample
                dragging = false
            }
            MotionEvent.ACTION_MOVE -> {
                val s = sampleAt(event.x)
                if (!dragging && abs(s - dragStartSample) > 12) {
                    dragging = true       // a real drag, not a tap: pass 2 refinement
                    showLoupe = true
                }
                if (dragging) {
                    dragCurSample = s
                    cursorSample = s
                    invalidate()
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (dragging) {
                    val onset = min(dragStartSample, dragCurSample)
                    val offset = max(dragStartSample, dragCurSample)
                    if (offset > onset) onDragSpan?.invoke(onset, offset)
                    dragging = false
                    showLoupe = false
                    cursorSample = -1
                    invalidate()
                }
                dragStartSample = -1
            }
        }
        return true
    }

    private fun zoomBy(factor: Float, focusPx: Float) {
        val span = (toSample - fromSample + 1).toFloat()
        val newSpan = (span / factor).coerceIn(20f, 60_000f)
        val focusSample = sampleAt(focusPx)
        val ratio = (focusSample - fromSample).toFloat() / span.coerceAtLeast(1f)
        val newFrom = (focusSample - newSpan * ratio).toLong().coerceIn(0, 1_000_000)
        val newTo = newFrom + newSpan.toLong() - 1
        fromSample = newFrom; toSample = newTo
        onScrolled?.invoke(newFrom, newTo)
        invalidate()
    }

    /** Pans so that [center] is in the middle of the view. */
    fun centerOn(center: Long, span: Long) {
        val from = (center - span / 2).coerceAtLeast(0)
        fromSample = from
        toSample = from + span - 1
        onScrolled?.invoke(from, toSample)
        invalidate()
    }

    // --- drawing ----------------------------------------------------------------------------

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return
        val laneH = height / 3f
        drawLaneBand(canvas, 0f, laneH)          // X
        drawLaneBand(canvas, laneH, 2 * laneH)   // Y
        drawLaneBand(canvas, 2 * laneH, height.toFloat())  // speed
        val xl = xLane; val yl = yLane; val sp = speedLane
        if (xl != null) drawTrace(canvas, xl, 0f, laneH, paintTrace, 0xFF1565C0.toInt())
        if (yl != null) drawTrace(canvas, yl, laneH, 2 * laneH, paintTrace, 0xFF2E7D32.toInt())
        if (sp != null) drawTrace(canvas, sp, 2 * laneH, height.toFloat(), paintSpeed, 0xFF7E57C2.toInt())
        drawLabels(canvas)
        drawInvalidGaps(canvas, xl, 0f, laneH)
        drawInvalidGaps(canvas, yl, laneH, 2 * laneH)
        drawPrelabels(canvas)
        drawPointMarks(canvas)
        drawExpertMarks(canvas)
        if (cursorSample >= 0) {
            val px = pxOf(cursorSample)
            canvas.drawLine(px, 0f, px, height.toFloat(), paintCursor)
        }
        if (dragging) drawLoupe(canvas)
        drawTimeRuler(canvas)
    }

    private fun drawLaneBand(canvas: Canvas, top: Float, bottom: Float) {
        canvas.drawRect(0f, top, width.toFloat(), bottom, paintGrid)
        canvas.drawLine(0f, bottom, width.toFloat(), bottom, paintGrid)
    }

    private fun drawTrace(canvas: Canvas, lane: Lane, top: Float, bottom: Float, paint: Paint, color: Int) {
        paint.color = color
        val mid = (top + bottom) / 2f
        val amp = (bottom - top) / 2f * 0.9f
        var lo = Float.POSITIVE_INFINITY
        var hi = Float.NEGATIVE_INFINITY
        for (v in lane.values) if (v.isFinite()) { if (v < lo) lo = v; if (v > hi) hi = v }
        if (!lo.isFinite()) return
        val range = (hi - lo).coerceAtLeast(1e-6f)
        val path = Path()
        var started = false
        val n = lane.values.size
        for (i in 0 until n) {
            val v = lane.values[i]
            val px = pxOf(fromSample + i)
            if (!v.isFinite()) { started = false; continue }
            val py = mid - (v - (lo + hi) / 2f) / range * amp
            if (!started) { path.moveTo(px, py); started = true } else path.lineTo(px, py)
        }
        canvas.drawPath(path, paint)
    }

    private fun drawLabels(canvas: Canvas) {
        val band = labelBand ?: return
        val top = 0f; val bottom = height.toFloat()
        for (i in band.indices) {
            val l = band[i].toInt() and 0xFF
            if (l == 0 || l == 255) continue
            paintLabelBand.color = when (l) {
                1 -> 0x331565C0
                2 -> 0x33C62828
                else -> 0x22000000
            }
            val x0 = pxOf(fromSample + i)
            val x1 = pxOf(fromSample + i + 1)
            canvas.drawRect(x0, top, x1.coerceAtLeast(x0 + 1f), bottom, paintLabelBand)
        }
    }

    private fun drawInvalidGaps(canvas: Canvas, lane: Lane?, top: Float, bottom: Float) {
        val inv = lane?.invalid ?: return
        for (i in inv.indices) {
            if (!inv[i]) continue
            val x0 = pxOf(fromSample + i)
            val x1 = pxOf(fromSample + i + 1)
            canvas.drawRect(x0, top, x1.coerceAtLeast(x0 + 1f), bottom, paintInvalid)
        }
    }

    private fun drawPrelabels(canvas: Canvas) {
        val mid = height / 2f
        for (i in prelabels.indices) {
            val (on, off) = prelabels[i]
            val cls = prelabelCls.getOrElse(i) { 1 }
            paintPre.color = if (cls == 2) 0x66C62828.toInt() else 0x66FFB300.toInt()
            val x0 = pxOf(on); val x1 = pxOf(off + 1)
            canvas.drawLine(x0, mid - 40, x0, mid + 40, paintPre)
            canvas.drawLine(x1, mid - 40, x1, mid + 40, paintPre)
            canvas.drawLine(x0, mid, x1, mid, paintPre)
        }
    }

    private fun drawPointMarks(canvas: Canvas) {
        val mid = height / 2f
        for ((at, cls) in pointMarks) {
            paintPoint.color = if (cls == 2) 0xFFC62828.toInt() else 0xFF1565C0.toInt()
            paintPoint.style = Paint.Style.FILL
            val px = pxOf(at)
            val path = Path()
            path.moveTo(px, mid - 50); path.lineTo(px - 25, mid - 10); path.lineTo(px + 25, mid - 10)
            path.close()
            canvas.drawPath(path, paintPoint)
        }
    }

    private fun drawExpertMarks(canvas: Canvas) {
        val mid = height / 2f
        for ((on, off, cls) in expertMarks) {
            paintExpert.color = if (cls == 2) 0xFFC62828.toInt() else 0xFF1565C0.toInt()
            val x0 = pxOf(on); val x1 = pxOf(off + 1)
            canvas.drawLine(x0, mid - 60, x0, mid - 20, paintExpert)
            canvas.drawLine(x0, mid - 20, x1, mid - 20, paintExpert)
            canvas.drawLine(x1, mid - 20, x1, mid - 60, paintExpert)
            canvas.drawLine(x0, mid + 20, x0, mid + 60, paintExpert)
            canvas.drawLine(x0, mid + 60, x1, mid + 60, paintExpert)
            canvas.drawLine(x1, mid + 60, x1, mid + 20, paintExpert)
        }
    }

    /** Magnifier above the finger: a zoomed copy of the trace around the cursor. */
    private fun drawLoupe(canvas: Canvas) {
        val r = 140f
        val cx = width / 2f
        val cy = r + 30f
        val rect = RectF(cx - r, cy - r, cx + r, cy + r)
        canvas.drawRoundRect(rect, 24f, 24f, paintLoupeBg)
        canvas.drawRoundRect(rect, 24f, 24f, paintLoupeStroke)
        canvas.save()
        canvas.clipRect(rect)
        val span = 200L    // loupe shows +/- 100 samples around the cursor
        val lo = (cursorSample - span / 2).coerceAtLeast(fromSample)
        val hi = (cursorSample + span / 2).coerceAtMost(toSample)
        val xl = xLane
        if (xl != null) {
            val path = Path()
            var started = false
            var lo2 = Float.POSITIVE_INFINITY; var hi2 = Float.NEGATIVE_INFINITY
            for (i in (lo - fromSample).toInt()..(hi - fromSample).toInt()) {
                if (i < 0 || i >= xl.values.size) continue
                val v = xl.values[i]
                if (v.isFinite()) { if (v < lo2) lo2 = v; if (v > hi2) hi2 = v }
            }
            if (lo2.isFinite()) {
                val range = (hi2 - lo2).coerceAtLeast(1e-6f)
                val mid = cy
                val amp = r * 0.8f
                for (i in (lo - fromSample).toInt()..(hi - fromSample).toInt()) {
                    if (i < 0 || i >= xl.values.size) continue
                    val v = xl.values[i]
                    if (!v.isFinite()) { started = false; continue }
                    val px = rect.left + (i - (lo - fromSample)).toFloat() / (hi - lo + 1).coerceAtLeast(1) * rect.width()
                    val py = mid - (v - (lo2 + hi2) / 2f) / range * amp
                    if (!started) { path.moveTo(px, py); started = true } else path.lineTo(px, py)
                }
                canvas.drawPath(path, paintTrace.apply { color = 0xFF1565C0.toInt() })
            }
        }
        canvas.restore()
        // cursor line inside the loupe
        val cpx = cx
        canvas.drawLine(cpx, rect.top, cpx, rect.bottom, paintCursor)
        canvas.drawText("${cursorSample}", rect.left + 10, rect.bottom - 10, paintText)
    }

    private fun drawTimeRuler(canvas: Canvas) {
        val y = height - 8f
        val spanMs = (toSample - fromSample + 1)   // 1 sample = 1 ms at 1 kHz
        val step = when {
            spanMs <= 500 -> 50L
            spanMs <= 2000 -> 200L
            else -> 500L
        }
        var s = (fromSample / step) * step
        while (s <= toSample) {
            val px = pxOf(s)
            canvas.drawLine(px, y - 14, px, y, paintGrid)
            canvas.drawText("${s}ms", px + 4, y - 18, paintText.apply { textSize = 24f })
            s += step
        }
    }
}
