package com.uneye.offline

import android.content.Intent
import android.net.Uri
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import androidx.fragment.app.FragmentActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Export plumbing (phase 5): Storage Access Framework launchers for
 * ACTION_CREATE_DOCUMENT (write labels / X / Y / .mat / model to any provider:
 * local storage, Google Drive, SD card) and an ACTION_SEND_MULTIPLE share
 * intent (e-mail, Drive upload, Colab-adjacent apps) for the files already
 * exported to the app's external files dir.
 *
 * The actual writing is done by [Exporter]; this class only owns the SAF
 * contracts and the share intent.
 *
 * The launchers must be created during activity/fragment initialization
 * (registerForActivityResult throws once the activity is STARTED), so they
 * are created eagerly in the constructor and reused.
 */
class ExportHelper(private val activity: FragmentActivity) {

    private var pendingCallback: ((Uri?) -> Unit)? = null

    private val createDocument: ActivityResultLauncher<Array<String>> =
        activity.registerForActivityResult(ActivityResultContracts.CreateDocument()) { uri ->
            pendingCallback?.invoke(uri)
        }

    /**
     * Launches CREATE_DOCUMENT (the user picks the destination and the file
     * name) and runs [write] with the returned stream on a background thread.
     * [write] must close the stream itself (use `use {}`). [onDone] receives
     * whether the write succeeded and the Uri the user picked (null when the
     * user cancelled).
     */
    fun exportTo(mimeType: String, fileName: String,
                 write: suspend (java.io.OutputStream) -> Unit,
                 onDone: (ok: Boolean, uri: Uri?) -> Unit) {
        pendingCallback = { uri ->
            if (uri == null) { onDone(false, null); return@registerForActivityResult }
            activity.lifecycleScope.launchWhenResumed {
                val ok = try {
                    withContext(Dispatchers.IO) {
                        activity.contentResolver.openOutputStream(uri)?.use { write(it) } != null
                    }
                } catch (_: Exception) { false }
                onDone(ok, uri)
            }
        }
        createDocument.launch(arrayOf(mimeType, fileName))
    }

    /**
     * Shares files already on disk (app external files dir) via the system
     * share sheet: e-mail, Google Drive, etc. Requires the FileProvider
     * declared in the manifest (res/xml/file_paths.xml).
     */
    fun shareFiles(files: List<File>, mimeTypes: List<String>) {
        val uris = files.filter { it.exists() }.map { f ->
            FileProvider.getUriForFile(activity, "${activity.packageName}.fileprovider", f)
        }
        if (uris.isEmpty()) return
        val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = if (mimeTypes.size == 1) mimeTypes[0] else "*/*"
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        activity.startActivity(Intent.createChooser(intent, activity.getString(R.string.export_share_title)))
    }

    /**
     * Exports the whole session (labels CSV + X CSV + Y CSV + optionally the
     * .mat and the trained model .bin/.json) into [dir] under [baseName].
     * Returns the files written, in a stable order (labels, X, Y, [mat],
     * [model .bin, model .json]). Off the UI thread.
     */
    suspend fun exportAll(session: Session, dir: File, baseName: String,
                          includeMat: Boolean, includeModel: Boolean): List<File> {
        val out = ArrayList<File>()
        withContext(Dispatchers.IO) {
            // labels CSV (UnEye: trials x time, 0/1/2, NaN = not reviewed)
            val labels = Exporter.labelsMatrix(session)
            val labelsFile = File(dir, "${baseName}_L.csv")
            labelsFile.outputStream().use { Exporter.writeLabelsCsv(it, labels) }
            out.add(labelsFile)
            // X / Y CSV (degrees)
            val (xs, ys) = Exporter.positionsMatrix(session)
            val xFile = File(dir, "${baseName}_X.csv")
            xFile.outputStream().use { Exporter.writePositionsCsv(it, xs) }
            out.add(xFile)
            val yFile = File(dir, "${baseName}_Y.csv")
            yFile.outputStream().use { Exporter.writePositionsCsv(it, ys) }
            out.add(yFile)
            // .mat (X, Y, Sacc)
            if (includeMat) {
                val matFile = File(dir, "$baseName.mat")
                matFile.outputStream().use { Exporter.writeMat(it, session) }
                out.add(matFile)
            }
            // model .bin + .json
            val model = session.lastTrainedModel()
            if (includeModel && model.isNotEmpty() && File(model).exists()) {
                val bin = Exporter.exportModel(model, dir, baseName + "_model")
                out.add(File(bin))
                val json = File(bin.replaceAfterLast('.', "json", missingDelimiterValue = bin + ".json"))
                if (json.exists()) out.add(json)
            }
        }
        return out
    }
}
