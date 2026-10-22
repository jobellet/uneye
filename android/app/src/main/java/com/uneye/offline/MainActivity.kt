package com.uneye.offline

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Entry screen (phase 2): the file list (Room) + the load dialog (sampling
 * rate, units/scale, pre-labeler). Opening the app on a CSV / MAT file (from
 * a file manager) lands here with the file pre-filled.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var app: UneyeApp
    private lateinit var fileListText: TextView
    private lateinit var openBtn: Button
    private lateinit var openXBtn: Button
    private lateinit var openYBtn: Button

    private var pendingX: Uri? = null
    private var pendingY: Uri? = null
    private var pendingMat: Uri? = null

    private val pickFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@registerForActivityResult
        val name = displayName(uri).lowercase()
        when {
            name.endsWith(".mat") -> pendingMat = uri
            name.contains("_x") || name.endsWith("x.csv") -> pendingX = uri
            name.contains("_y") || name.endsWith("y.csv") -> pendingY = uri
            else -> pendingX = uri   // first of a pair; the user picks the second
        }
        updatePicked()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        app = application as UneyeApp
        setContentView(R.layout.activity_main)
        fileListText = findViewById(R.id.fileListText)
        openBtn = findViewById(R.id.openBtn)
        openXBtn = findViewById(R.id.openXBtn)
        openYBtn = findViewById(R.id.openYBtn)

        openBtn.setOnClickListener { if (pendingMat != null) showLoadDialog(pendingMat!!, null) else pickFile.launch(arrayOf("text/*", "application/octet-stream")) }
        openXBtn.setOnClickListener { pickFile.launch(arrayOf("text/*", "application/octet-stream")) }
        openYBtn.setOnClickListener { pickFile.launch(arrayOf("text/*", "application/octet-stream")) }
        findViewById<Button>(R.id.startLabelingBtn).setOnClickListener {
            startActivity(Intent(this, LabelingScreen::class.java))
        }

        handleIncomingIntent(intent)
        refreshFileList()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIncomingIntent(intent)
    }

    private fun handleIncomingIntent(intent: Intent?) {
        val uri = intent?.data ?: return
        val name = displayName(uri).lowercase()
        if (name.endsWith(".mat")) pendingMat = uri
        else pendingX = uri
        updatePicked()
    }

    private fun updatePicked() {
        val parts = mutableListOf<String>()
        pendingMat?.let { parts.add("MAT: ${displayName(it)}") }
        pendingX?.let { parts.add("X: ${displayName(it)}") }
        pendingY?.let { parts.add("Y: ${displayName(it)}") }
        fileListText.text = if (parts.isEmpty()) getString(R.string.no_file_picked)
            else parts.joinToString("\n")
    }

    private fun displayName(uri: Uri): String = contentResolver.query(uri, null, null, null, null)?.use { c ->
        val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (i >= 0 && c.moveToFirst()) c.getString(i) else uri.lastPathSegment ?: uri.toString()
    } ?: (uri.lastPathSegment ?: uri.toString())

    // --- load dialog ---------------------------------------------------------------

    private fun showLoadDialog(mat: Uri, y: Uri?) {
        val dialog = layoutInflater.inflate(R.layout.dialog_load, null)
        val rateEdit = dialog.findViewById<EditText>(R.id.rateEdit)
        val unitEdit = dialog.findViewById<EditText>(R.id.unitEdit)
        val preSpinner = dialog.findViewById<Spinner>(R.id.preSpinner)
        val modelEdit = dialog.findViewById<EditText>(R.id.modelEdit)
        val estimateBox = dialog.findViewById<android.widget.CheckBox>(R.id.estimateBox)
        preSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item,
            listOf("HMM", "Modèle embarqué (combined.bin)", "Dernier modèle entraîné", "Aucun"))
        AlertDialog.Builder(this)
            .setTitle(R.string.load_dialog_title)
            .setView(dialog)
            .setPositiveButton(R.string.load) { _, _ ->
                val rate = rateEdit.text.toString().toDoubleOrNull() ?: 1000.0
                val unit = unitEdit.text.toString().toDoubleOrNull() ?: 1.0
                val prelabeler = preSpinner.selectedItemPosition
                val modelBin = modelEdit.text.toString().trim()
                loadFile(mat, y, rate, unit, estimateBox.isChecked, prelabeler, modelBin)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun loadFile(mat: Uri, y: Uri?, rateHz: Double, degPerUnit: Double, estimateScale: Boolean,
                         prelabeler: Int, modelBin: String) {
        val session = app.session
        Toast.makeText(this, R.string.loading, Toast.LENGTH_SHORT).show()
        lifecycleScope.launch {
            val trace = withContext(Dispatchers.IO) {
                try {
                    if (y == null) {
                        contentResolver.openInputStream(mat)?.use { TraceFiles.read(it, displayName(mat)) }
                    } else {
                        val xText = contentResolver.openInputStream(pendingX!!)?.use { it.readBytes().toString(Charsets.UTF_8) }
                        val yText = contentResolver.openInputStream(y)?.use { it.readBytes().toString(Charsets.UTF_8) }
                        if (xText != null && yText != null) TraceFiles.readCsvPair(xText, yText) else null
                    }
                } catch (e: Exception) {
                    null
                }
            }
            if (trace == null) {
                Toast.makeText(this@MainActivity, R.string.load_failed, Toast.LENGTH_LONG).show()
                return@launch
            }
            val modelPath = if (prelabeler == Native.PRELABELER_SHIPPED_MODEL && modelBin.isEmpty())
                app.shippedModelPath() else modelBin
            var result: Pair<Boolean, String>? = null
            session.load(trace, rateHz, degPerUnit, estimateScale, prelabeler, modelPath) { ok, err ->
                result = ok to err
            }
            // session.load posts onDone on the main thread; wait for it
            while (result == null) kotlinx.coroutines.delay(20)
            val (ok, err) = result!!
            if (ok) {
                // the convergence test set is chosen per trial before pass 1
                session.chooseTestSet(5, 2)
                val dao = FileListDb.get(this@MainActivity).traceFiles()
                dao.insert(TraceFileEntity(
                    uri = (pendingMat ?: pendingX!!).toString(),
                    yUri = y?.toString() ?: "",
                    displayName = displayName(pendingMat ?: pendingX!!),
                    rateHz = rateHz,
                    degPerUnit = session.degPerUnit,
                    labelsPath = app.labelsPathFor(session, 0),
                    modelPath = "",
                ))
                refreshFileList()
                startActivity(Intent(this@MainActivity, LabelingScreen::class.java))
            } else {
                Toast.makeText(this@MainActivity, getString(R.string.load_failed_arg, err), Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun refreshFileList() {
        lifecycleScope.launch {
            val files = withContext(Dispatchers.IO) { FileListDb.get(this@MainActivity).traceFiles().all() }
            val list = findViewById<TextView>(R.id.knownFilesText)
            list.text = if (files.isEmpty()) getString(R.string.no_known_files)
            else files.joinToString("\n") { "${it.displayName}  (${it.rateHz.toInt()} Hz, labels: ${it.labelsPath})" }
        }
    }
}
