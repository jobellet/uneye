package com.uneye.offline

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.os.SystemClock
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Touch-optimized trace view with high data-ink ratio (Edward Tufte principles).
 *
 * Displays:
 *  - Lane 0: Position X trace (degrees) with 1° floating scale bar and range.
 *  - Lane 1: Position Y trace (degrees) with 1° floating scale bar and range.
 *  - Lane 2: Velocity trace (°/s) with dashed adaptive threshold (λ·σ) line.
 *  - Reviewed bands, ground truth labels, and pre-label brackets.
 *  - Pass 2: focused event highlight window with boundary drag handles.
 *  - Magnifier loupe during drag refinement.
 */
class TraceView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) :
    View(context, attrs) {

    companion object {
        const val TOOL_SACCADE = 1
        const val TOOL_BLINK = 2
        const val TOOL_ERASER = 3
        const val REFRACTORY_MS = 600L     // no mark for this long after the end of a pinch zoom
    }

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
    var prelabels: List<Pair<Long, Long>> = emptyList()     // (onset, offset)
    var prelabelCls: List<Int> = emptyList()
    var pointMarks: List<Pair<Long, Int>> = emptyList()     // (at, cls)
    var expertMarks: List<Triple<Long, Long, Int>> = emptyList()  // (onset, offset, cls)

    var currentThreshold: Double = 0.0
    var thresholdLambda: Double = 6.0
    var degPerUnit: Double = 1.0

    // Pass 2 state
    var isPassTwo: Boolean = false
    var activePutativeRange: Pair<Long, Long>? = null
    var activeClass: Int = Native.CLS_SACCADE
    var activeTool: Int = TOOL_SACCADE

    var cursorSample: Long = -1               // drawn as a vertical line (pass 2 finger)
    var showLoupe: Boolean = false

    // --- callbacks -------------------------------------------------------------------

    var onTapSample: ((Long) -> Unit)? = null
    var onDoubleTapSample: ((Long) -> Unit)? = null
    var onLongPressSample: ((Long) -> Unit)? = null
    var onDragSpan: ((onset: Long, offset: Long) -> Unit)? = null
    var onScrolled: ((from: Long, to: Long) -> Unit)? = null   // pan / zoom changed

    // --- colors & paints --------------------------------------------------------------

    private val typefaceNormal = Typeface.create(Typeface.DEFAULT, Typeface.NORMAL)

    private val paintBg = Paint().apply { color = 0xFFFAFAFA.toInt() }
    private val paintDivider = Paint().apply { color = 0xFFE0E0E0.toInt(); strokeWidth = 1.5f }

    private val paintTraceX = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 3f; style = Paint.Style.STROKE; color = 0xFF1E88E5.toInt()
    }
    private val paintTraceY = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 3f; style = Paint.Style.STROKE; color = 0xFF43A047.toInt()
    }
    private val paintSpeed = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 2.5f; style = Paint.Style.STROKE; color = 0xFF7E57C2.toInt()
    }

    private val paintThreshold = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 1.5f; style = Paint.Style.STROKE; color = 0xFFD32F2F.toInt()
        pathEffect = DashPathEffect(floatArrayOf(12f, 12f), 0f)
    }

    private val paintLabelBand = Paint()
    private val paintInvalid = Paint().apply { color = 0x22C62828 }
    private val paintPre = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 3f; style = Paint.Style.STROKE; color = 0x88FFB300.toInt()
    }
    private val paintPoint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val paintExpert = Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeWidth = 5f; style = Paint.Style.STROKE }
    private val paintCursor = Paint().apply { strokeWidth = 3f; color = 0xFFD32F2F.toInt() }
    private val paintGrid = Paint().apply { strokeWidth = 1f; color = 0x1A000000 }
    private val paintScaleBar = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 3.5f; style = Paint.Style.STROKE; color = 0xFF37474F.toInt()
    }

    private val paintLoupeBg = Paint().apply { color = 0xEEFFFFFF.toInt() }
    private val paintLoupeStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 2.5f; color = 0xFF1565C0.toInt(); style = Paint.Style.STROKE
    }

    private val paintText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 28f; color = 0xFF546E7A.toInt(); typeface = typefaceNormal
        isFakeBoldText = false
    }
    private val paintThresholdText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = 26f; color = 0xFFD32F2F.toInt(); typeface = typefaceNormal
        isFakeBoldText = false
    }

    private val paintPass2Highlight = Paint().apply { color = 0x261E88E5.toInt() }
    private val paintPass2Bound = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        strokeWidth = 2f; style = Paint.Style.STROKE; color = 0xFF1565C0.toInt()
        pathEffect = DashPathEffect(floatArrayOf(8f, 8f), 0f)
    }

    private var dragStartSample = -1L
    private var dragCurSample = -1L
    private var dragging = false

    // Refractory period after a pinch zoom: the last finger that lifts must not mark an event, and neither may
    // a new touch within REFRACTORY_MS of the end of a multi-touch gesture.
    private var multiTouch = false
    private var ignoreGesture = false
    private var lastMultiEnd = 0L
    private fun suppressed() = multiTouch || ignoreGesture

    private val gestureDetector = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(e: MotionEvent): Boolean = true
        override fun onSingleTapUp(e: MotionEvent): Boolean {
            if (!suppressed()) onTapSample?.invoke(sampleAt(e.x))
            return true
        }
        override fun onDoubleTap(e: MotionEvent): Boolean {
            if (!suppressed()) onDoubleTapSample?.invoke(sampleAt(e.x))
            return true
        }
        override fun onLongPress(e: MotionEvent) {
            if (!suppressed()) onLongPressSample?.invoke(sampleAt(e.x))
        }
    })

    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            zoomBy(d.scaleFactor, d.focusX)
            return true
        }
    })

    // --- data injection ------------------------------------------------------------------

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

    // --- touch gestures -------------------------------------------------------------------

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val now = SystemClock.uptimeMillis()
        // gesture state first: the detectors below call back while they process this very event
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                multiTouch = false
                ignoreGesture = now - lastMultiEnd < REFRACTORY_MS
            }
            MotionEvent.ACTION_POINTER_DOWN -> multiTouch = true
        }
        scaleDetector.onTouchEvent(event)
        gestureDetector.onTouchEvent(event)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                dragStartSample = sampleAt(event.x)
                dragCurSample = dragStartSample
                dragging = false
            }
            MotionEvent.ACTION_POINTER_DOWN -> {
                dragging = false
                showLoupe = false
            }
            MotionEvent.ACTION_POINTER_UP -> lastMultiEnd = now
            MotionEvent.ACTION_MOVE -> {
                val s = sampleAt(event.x)
                if (!suppressed() && !dragging && abs(s - dragStartSample) > 8) {
                    dragging = true       // a real drag: pass 2 refinement
                    showLoupe = true
                }
                if (dragging) {
                    dragCurSample = s
                    cursorSample = s
                    invalidate()
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (multiTouch) lastMultiEnd = now
                if (dragging && suppressed()) { dragging = false; showLoupe = false; cursorSample = -1; invalidate() }
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

        // 1. Solid clear to prevent ghosting between passes or frames
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), paintBg)

        // two lanes: position (X and Y on the same axes) above, speed below
        val posLaneTop = 0f
        val spLaneTop = 0.66f * height

        // Lane Divider
        canvas.drawLine(0f, spLaneTop, width.toFloat(), spLaneTop, paintDivider)

        // 2. Background label bands & reviewed regions
        drawReviewedBands(canvas)
        drawLabelBands(canvas)

        // 3. In Pass 2: highlight the active putative event
        if (isPassTwo) {
            drawPass2Highlight(canvas)
        }

        // 4. Invalid / blink gaps
        val xl = xLane; val yl = yLane; val sp = speedLane
        drawInvalidGaps(canvas, xl, posLaneTop, spLaneTop)
        drawInvalidGaps(canvas, yl, posLaneTop, spLaneTop)

        // 5. Traces
        drawPositionLane(canvas, xl, yl, posLaneTop, spLaneTop)
        if (sp != null) drawVelocityTrace(canvas, sp, spLaneTop, height.toFloat(), "Velocity (°/s)")

        // 6. Pre-labels, point marks, and expert marks
        drawPrelabels(canvas)
        drawPointMarks(canvas)
        drawExpertMarks(canvas)

        // 7. Cursor / drag indicator
        if (cursorSample >= 0) {
            val px = pxOf(cursorSample)
            canvas.drawLine(px, 0f, px, height.toFloat(), paintCursor)
        }

        // 8. Loupe magnifier during drag
        if (dragging && showLoupe) {
            drawLoupe(canvas)
        }

        // 9. Time scale ruler at bottom
        drawTimeRuler(canvas)
    }

    private fun drawReviewedBands(canvas: Canvas) {
        val rev = reviewedBand ?: return
        var start = -1
        val top = 0f; val h = height.toFloat()
        paintLabelBand.color = 0x142E7D32 // faint green tint

        for (i in 0 until rev.size) {
            if (rev[i] && start == -1) {
                start = i
            } else if (!rev[i] && start != -1) {
                val x0 = pxOf(fromSample + start)
                val x1 = pxOf(fromSample + i)
                canvas.drawRect(x0, top, x1, h, paintLabelBand)
                // Top green hairline
                canvas.drawRect(x0, top, x1, top + 6f, paintExpert.apply { color = 0xFF2E7D32.toInt() })
                start = -1
            }
        }
        if (start != -1) {
            val x0 = pxOf(fromSample + start)
            val x1 = pxOf(fromSample + rev.size)
            canvas.drawRect(x0, top, x1, h, paintLabelBand)
            canvas.drawRect(x0, top, x1, top + 6f, paintExpert.apply { color = 0xFF2E7D32.toInt() })
        }
    }

    private fun drawLabelBands(canvas: Canvas) {
        val band = labelBand ?: return
        val top = 0f; val bottom = height.toFloat()
        var start = -1
        var curCls = 0

        for (i in band.indices) {
            val l = band[i].toInt() and 0xFF
            if (l != curCls) {
                if (curCls == 1 || curCls == 2) {
                    val x0 = pxOf(fromSample + start)
                    val x1 = pxOf(fromSample + i)
                    paintLabelBand.color = if (curCls == 1) 0x291E88E5.toInt() else 0x2EC62828.toInt()
                    canvas.drawRect(x0, top, max(x1, x0 + 2f), bottom, paintLabelBand)
                }
                start = i
                curCls = l
            }
        }
        if (curCls == 1 || curCls == 2) {
            val x0 = pxOf(fromSample + start)
            val x1 = pxOf(fromSample + band.size)
            paintLabelBand.color = if (curCls == 1) 0x291E88E5.toInt() else 0x2EC62828.toInt()
            canvas.drawRect(x0, top, max(x1, x0 + 2f), bottom, paintLabelBand)
        }
    }

    private fun drawPass2Highlight(canvas: Canvas) {
        val r = activePutativeRange ?: return
        val (on, off) = r
        if (off < fromSample || on > toSample) return

        val x0 = pxOf(on)
        val x1 = pxOf(off + 1)
        paintPass2Highlight.color = if (activeClass == Native.CLS_BLINK) 0x22C62828.toInt() else 0x221E88E5.toInt()
        canvas.drawRect(x0, 0f, max(x1, x0 + 2f), height.toFloat(), paintPass2Highlight)

        // Dashed boundaries
        paintPass2Bound.color = if (activeClass == Native.CLS_BLINK) 0xFFC62828.toInt() else 0xFF1565C0.toInt()
        canvas.drawLine(x0, 0f, x0, height.toFloat(), paintPass2Bound)
        canvas.drawLine(x1, 0f, x1, height.toFloat(), paintPass2Bound)
    }

    /**
     * X (blue) and Y (green) on the same axes, zoomed as far as possible on the extreme values of the visible window.
     * The scale is never tighter than 1.25 degrees so that the 1-degree scale bar always fits.
     */
    private fun drawPositionLane(canvas: Canvas, xl: Lane?, yl: Lane?, top: Float, bottom: Float) {
        val h = bottom - top
        var lo = Float.POSITIVE_INFINITY
        var hi = Float.NEGATIVE_INFINITY
        for (lane in listOf(xl, yl)) {
            if (lane == null) continue
            for (v in lane.values) {
                if (v.isFinite()) {
                    if (v < lo) lo = v
                    if (v > hi) hi = v
                }
            }
        }
        if (!lo.isFinite()) return

        val oneDeg = (1.0 / max(1e-4, degPerUnit)).toFloat()
        var minV = lo
        var maxV = hi
        val pad = max(0.02f * oneDeg, (maxV - minV) * 0.04f)
        minV -= pad
        maxV += pad
        if (maxV - minV < 1.25f * oneDeg) {
            val c = (minV + maxV) / 2f
            minV = c - 0.625f * oneDeg
            maxV = c + 0.625f * oneDeg
        }

        canvas.drawText("X / Y  Position (°)", 20f, top + 36f, paintText)
        val rangeDeg = (maxV - minV) * degPerUnit.toFloat()
        canvas.drawText(String.format(java.util.Locale.US, "%.1f°", rangeDeg), width - 110f, top + 36f, paintText)

        // 1° scale bar at the right edge
        val barH = (oneDeg / (maxV - minV)) * (h - 32f)
        val barX = width - 24f
        val barY1 = bottom - 16f
        val barY0 = barY1 - barH
        canvas.drawLine(barX, barY1, barX, barY0, paintScaleBar)
        canvas.drawLine(barX - 6f, barY0, barX + 6f, barY0, paintScaleBar)
        canvas.drawLine(barX - 6f, barY1, barX + 6f, barY1, paintScaleBar)

        for ((lane, paint) in listOf(xl to paintTraceX, yl to paintTraceY)) {
            if (lane == null) continue
            val path = Path()
            var started = false
            for (i in lane.values.indices) {
                val v = lane.values[i]
                val px = pxOf(fromSample + i)
                if (!v.isFinite()) { started = false; continue }
                val py = bottom - 16f - ((v - minV) / (maxV - minV)) * (h - 32f)
                if (!started) { path.moveTo(px, py); started = true } else path.lineTo(px, py)
            }
            canvas.drawPath(path, paint)
        }
    }

    private fun drawVelocityTrace(
        canvas: Canvas, lane: Lane, top: Float, bottom: Float, title: String
    ) {
        val h = bottom - top
        var maxSp = 50f
        for (v in lane.values) {
            if (v.isFinite() && v > maxSp) maxSp = v
        }
        maxSp = max(100f, maxSp * 1.15f)

        // Title and max speed
        canvas.drawText(title, 20f, top + 36f, paintText)
        val maxStr = String.format(java.util.Locale.US, "%d °/s", maxSp.roundToInt())
        canvas.drawText(maxStr, width - 110f, top + 36f, paintText)

        // Dashed adaptive threshold line
        if (currentThreshold > 0 && currentThreshold <= maxSp) {
            val thY = bottom - 16f - (currentThreshold.toFloat() / maxSp) * (h - 32f)
            canvas.drawLine(0f, thY, width.toFloat(), thY, paintThreshold)
            val thStr = String.format(java.util.Locale.US, "Threshold: %.1f °/s (%.1fσ)", currentThreshold, thresholdLambda)
            canvas.drawText(thStr, width - 360f, thY - 8f, paintThresholdText)
        }

        val path = Path()
        var started = false
        val n = lane.values.size
        for (i in 0 until n) {
            val v = lane.values[i]
            val px = pxOf(fromSample + i)
            if (!v.isFinite()) {
                started = false
                continue
            }
            val py = bottom - 16f - (v / maxSp) * (h - 32f)
            if (!started) {
                path.moveTo(px, py)
                started = true
            } else {
                path.lineTo(px, py)
            }
        }
        canvas.drawPath(path, paintSpeed)
    }

    private fun drawInvalidGaps(canvas: Canvas, lane: Lane?, top: Float, bottom: Float) {
        val inv = lane?.invalid ?: return
        for (i in inv.indices) {
            if (!inv[i]) continue
            val x0 = pxOf(fromSample + i)
            val x1 = pxOf(fromSample + i + 1)
            canvas.drawRect(x0, top, max(x1, x0 + 1f), bottom, paintInvalid)
        }
    }

    private fun drawPrelabels(canvas: Canvas) {
        val mid = height / 2f
        for (i in prelabels.indices) {
            val (on, off) = prelabels[i]
            val cls = prelabelCls.getOrElse(i) { 1 }
            paintPre.color = if (cls == 2) 0x66C62828.toInt() else 0x66FFB300.toInt()
            val x0 = pxOf(on); val x1 = pxOf(off + 1)
            canvas.drawLine(x0, mid - 35, x0, mid + 35, paintPre)
            canvas.drawLine(x1, mid - 35, x1, mid + 35, paintPre)
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
            path.moveTo(px, mid - 45); path.lineTo(px - 20, mid - 10); path.lineTo(px + 20, mid - 10)
            path.close()
            canvas.drawPath(path, paintPoint)
        }
    }

    private fun drawExpertMarks(canvas: Canvas) {
        val mid = height / 2f
        for ((on, off, cls) in expertMarks) {
            paintExpert.color = if (cls == 2) 0xFFC62828.toInt() else 0xFF1565C0.toInt()
            val x0 = pxOf(on); val x1 = pxOf(off + 1)
            canvas.drawLine(x0, mid - 50, x0, mid - 15, paintExpert)
            canvas.drawLine(x0, mid - 15, x1, mid - 15, paintExpert)
            canvas.drawLine(x1, mid - 15, x1, mid - 50, paintExpert)
            canvas.drawLine(x0, mid + 15, x0, mid + 50, paintExpert)
            canvas.drawLine(x0, mid + 50, x1, mid + 50, paintExpert)
            canvas.drawLine(x1, mid + 50, x1, mid + 15, paintExpert)
        }
    }

    private fun drawLoupe(canvas: Canvas) {
        val r = 130f
        val cx = width / 2f
        val cy = r + 24f
        val rect = RectF(cx - r, cy - r, cx + r, cy + r)
        canvas.drawRoundRect(rect, 20f, 20f, paintLoupeBg)
        canvas.drawRoundRect(rect, 20f, 20f, paintLoupeStroke)
        canvas.save()
        canvas.clipRect(rect)
        val span = 160L
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
                val amp = r * 0.75f
                for (i in (lo - fromSample).toInt()..(hi - fromSample).toInt()) {
                    if (i < 0 || i >= xl.values.size) continue
                    val v = xl.values[i]
                    if (!v.isFinite()) { started = false; continue }
                    val px = rect.left + (i - (lo - fromSample)).toFloat() / (hi - lo + 1).coerceAtLeast(1) * rect.width()
                    val py = mid - (v - (lo2 + hi2) / 2f) / range * amp
                    if (!started) { path.moveTo(px, py) } else path.lineTo(px, py)
                }
                canvas.drawPath(path, paintTraceX)
            }
        }
        canvas.restore()
        val cpx = cx
        canvas.drawLine(cpx, rect.top, cpx, rect.bottom, paintCursor)
        canvas.drawText("${cursorSample} ms", rect.left + 14, rect.bottom - 12, paintText)
    }

    private fun drawTimeRuler(canvas: Canvas) {
        val y = height - 10f
        val spanMs = (toSample - fromSample + 1)
        val step = when {
            spanMs <= 300 -> 50L
            spanMs <= 1000 -> 100L
            spanMs <= 2500 -> 250L
            else -> 500L
        }
        var s = (fromSample / step) * step
        while (s <= toSample) {
            val px = pxOf(s)
            canvas.drawLine(px, y - 12, px, y, paintGrid)
            canvas.drawText("${s} ms", px + 4, y - 14, paintText.apply { textSize = 22f })
            s += step
        }
    }
}
