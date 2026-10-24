package com.uneye.offline

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Base64
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.webkit.WebViewAssetLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Main Activity hosting the unified, touch-optimized U'n'Eye UI.
 *
 * UI frontend assets (HTML, CSS, JS, WASM, Web Worker) are shared directly
 * with the web app and packaged into Android assets by Gradle `syncWebAssets`.
 * This guarantees 100% visual and functional parity: any UI change in web
 * is automatically reflected in Android upon build.
 *
 * Native bridges:
 *  - Native document file picker via WebChromeClient.onShowFileChooser
 *  - Direct file saving to Android Downloads via AndroidBridge.saveFile
 *  - External file intent opening (MAT / CSV / JSON) via handleIncomingIntent
 *  - Full offline capability with WebViewAssetLoader and COOP/COEP headers
 */
class MainActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    // a file opened from another app before the page has loaded waits here (window.loadFromAndroid does not exist yet)
    private var pageLoaded = false
    private var pendingJs: String? = null
    private var fileChooserCallback: ValueCallback<Array<Uri>>? = null

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val uris: Array<Uri>? = if (result.resultCode == RESULT_OK) {
            val data = result.data
            if (data?.clipData != null) {
                val count = data.clipData!!.itemCount
                Array(count) { i -> data.clipData!!.getItemAt(i).uri }
            } else if (data?.data != null) {
                arrayOf(data.data!!)
            } else {
                null
            }
        } else {
            null
        }
        fileChooserCallback?.onReceiveValue(uris)
        fileChooserCallback = null
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        webView = findViewById(R.id.webView)
        // debug builds: the page can be inspected / driven from chrome://inspect (the labeler tests run through it)
        if (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE != 0) WebView.setWebContentsDebuggingEnabled(true)

        val assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            allowContentAccess = true
            useWideViewPort = true
            loadWithOverviewMode = true
            cacheMode = WebSettings.LOAD_DEFAULT
            mediaPlaybackRequiresUserGesture = false
        }

        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView, url: String) {
                pageLoaded = true
                pendingJs?.let { view.evaluateJavascript(it, null) }
                pendingJs = null
            }

            override fun shouldInterceptRequest(
                view: WebView,
                request: WebResourceRequest
            ): WebResourceResponse? {
                return try {
                    val url = request.url ?: return null
                    val response = assetLoader.shouldInterceptRequest(url) ?: return null
                    if (response.data == null) return response
                    val path = url.path?.lowercase() ?: ""
                    val mime = when {
                        path.endsWith(".html") || path.endsWith("/") -> "text/html"
                        path.endsWith(".css") -> "text/css"
                        path.endsWith(".js") || path.endsWith(".mjs") -> "application/javascript"
                        path.endsWith(".wasm") -> "application/wasm"
                        path.endsWith(".json") -> "application/json"
                        path.endsWith(".png") -> "image/png"
                        path.endsWith(".jpg") || path.endsWith(".jpeg") -> "image/jpeg"
                        path.endsWith(".svg") -> "image/svg+xml"
                        path.endsWith(".bin") -> "application/octet-stream"
                        else -> response.mimeType ?: "application/octet-stream"
                    }
                    val headers = HashMap<String, String>()
                    response.responseHeaders?.let { headers.putAll(it) }
                    headers["Cross-Origin-Opener-Policy"] = "same-origin"
                    headers["Cross-Origin-Embedder-Policy"] = "require-corp"
                    headers["Access-Control-Allow-Origin"] = "*"
                    response.mimeType = mime
                    response.encoding = "UTF-8"
                    response.setStatusCodeAndReasonPhrase(200, "OK")
                    response.responseHeaders = headers
                    response
                } catch (e: Throwable) {
                    android.util.Log.e("MainActivity", "shouldInterceptRequest error", e)
                    null
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onJsAlert(
                view: WebView?,
                url: String?,
                message: String?,
                result: android.webkit.JsResult?
            ): Boolean {
                Toast.makeText(this@MainActivity, message ?: "", Toast.LENGTH_SHORT).show()
                result?.confirm()
                return true
            }

            override fun onJsConfirm(
                view: WebView?,
                url: String?,
                message: String?,
                result: android.webkit.JsResult?
            ): Boolean {
                result?.confirm()
                return true
            }

            override fun onShowFileChooser(
                view: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                fileChooserCallback?.onReceiveValue(null)
                fileChooserCallback = filePathCallback
                val intent = fileChooserParams?.createIntent() ?: Intent(Intent.ACTION_GET_CONTENT).apply {
                    type = "*/*"
                    addCategory(Intent.CATEGORY_OPENABLE)
                }
                try {
                    filePickerLauncher.launch(intent)
                } catch (e: Exception) {
                    fileChooserCallback?.onReceiveValue(null)
                    fileChooserCallback = null
                    return false
                }
                return true
            }
        }

        webView.addJavascriptInterface(AndroidBridge(this), "AndroidBridge")

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        webView.loadUrl("https://appassets.androidplatform.net/assets/web/index.html")

        handleIncomingIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIncomingIntent(intent)
    }

    private fun handleIncomingIntent(intent: Intent?) {
        val uri = intent?.data ?: return
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val name = displayName(uri)
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return@launch
                val base64 = Base64.encodeToString(bytes, Base64.NO_WRAP)
                withContext(Dispatchers.Main) {
                    val js = "if (window.loadFromAndroid) window.loadFromAndroid('${name.replace("'", "\\'")}', '$base64');"
                    if (pageLoaded) webView.evaluateJavascript(js, null) else pendingJs = js
                }
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "Failed to load incoming file: $uri", e)
            }
        }
    }

    private fun displayName(uri: Uri): String = contentResolver.query(uri, null, null, null, null)?.use { c ->
        val i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (i >= 0 && c.moveToFirst()) c.getString(i) else uri.lastPathSegment ?: uri.toString()
    } ?: (uri.lastPathSegment ?: uri.toString())

    fun saveFileToDownloads(filename: String, base64Content: String, mimeType: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val bytes = Base64.decode(base64Content, Base64.DEFAULT)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    val values = ContentValues().apply {
                        put(MediaStore.MediaColumns.DISPLAY_NAME, filename)
                        put(MediaStore.MediaColumns.MIME_TYPE, mimeType)
                        put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                    }
                    val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    uri?.let { contentResolver.openOutputStream(it)?.use { stream -> stream.write(bytes) } }
                } else {
                    val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                    val file = File(dir, filename)
                    file.writeBytes(bytes)
                }
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Saved $filename to Downloads", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Save error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    class AndroidBridge(private val activity: MainActivity) {
        @JavascriptInterface
        fun saveFile(filename: String, base64Content: String, mimeType: String) {
            activity.saveFileToDownloads(filename, base64Content, mimeType)
        }

        @JavascriptInterface
        fun showToast(message: String) {
            activity.runOnUiThread {
                Toast.makeText(activity, message, Toast.LENGTH_SHORT).show()
            }
        }
    }
}
