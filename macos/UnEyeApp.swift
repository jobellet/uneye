import Cocoa
import WebKit
import Darwin

// MARK: - Local Web Asset Manager & Auto-Updater
final class WebAssetManager {
    static let shared = WebAssetManager()

    let remoteBaseURL = URL(string: "https://jobellet.github.io/uneye/")!
    let trackedAssets: [String] = [
        "index.html",
        "manifest.json",
        "style.css",
        "app.js",
        "formats.js",
        "import.js",
        "import_ui.js",
        "hmm_prelabel.js",
        "nearmiss.js",
        "train_worker.js",
        "sw.js",
        "logo.jpeg",
        "logo_circle.png",
        "web/uneye_train.js",
        "web/uneye_train.wasm",
        "web/h5wasm.js",
        "models/combined.bin",
        "models/combined.json"
    ]

    let appSupportDir: URL
    let webDir: URL
    let etagsFile: URL
    let bundleWebDir: URL?
    let localRepoDir: URL?

    private var etags: [String: String] = [:]
    private let syncQueue = DispatchQueue(label: "com.uneye.labeler.sync", qos: .utility)
    private var isSyncing = false
    private var lastRemoteSyncTime: Date = .distantPast

    private init() {
        let fm = FileManager.default
        let baseSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        appSupportDir = baseSupport.appendingPathComponent("UnEye", isDirectory: true)
        webDir = appSupportDir.appendingPathComponent("web", isDirectory: true)
        etagsFile = appSupportDir.appendingPathComponent("etags.json")
        bundleWebDir = Bundle.main.resourceURL?.appendingPathComponent("web", isDirectory: true)

        var repoPath = ProcessInfo.processInfo.environment["UNEYE_LOCAL_REPO"]
        if repoPath == nil {
            repoPath = Bundle.main.object(forInfoDictionaryKey: "UnEyeLocalRepoPath") as? String
        }
        if let rp = repoPath, !rp.isEmpty, fm.fileExists(atPath: (rp as NSString).appendingPathComponent("index.html")) {
            localRepoDir = URL(fileURLWithPath: rp, isDirectory: true)
        } else {
            localRepoDir = nil
        }

        try? fm.createDirectory(at: webDir, withIntermediateDirectories: true)
        if let data = try? Data(contentsOf: etagsFile),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            etags = dict
        }

        bootstrapLocalAssets()
    }

    /// Ensures `~/Library/Application Support/UnEye/web/` has the latest files from the app bundle and (if present) the local git repository.
    func bootstrapLocalAssets() {
        let fm = FileManager.default
        for rel in trackedAssets {
            let dest = webDir.appendingPathComponent(rel)
            try? fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)

            if let bDir = bundleWebDir {
                let src = bDir.appendingPathComponent(rel)
                copyIfNewer(from: src, to: dest)
            }
            if let rDir = localRepoDir {
                let src = rDir.appendingPathComponent(rel)
                copyIfNewer(from: src, to: dest)
            }
        }
    }

    private func copyIfNewer(from src: URL, to dest: URL) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: src.path) else { return }
        if !fm.fileExists(atPath: dest.path) {
            try? fm.copyItem(at: src, to: dest)
            return
        }
        let srcDate = (try? fm.attributesOfItem(atPath: src.path)[.modificationDate] as? Date) ?? .distantPast
        let dstDate = (try? fm.attributesOfItem(atPath: dest.path)[.modificationDate] as? Date) ?? .distantPast
        if srcDate > dstDate {
            try? fm.removeItem(at: dest)
            try? fm.copyItem(at: src, to: dest)
        }
    }

    private func saveETags() {
        if let data = try? JSONSerialization.data(withJSONObject: etags, options: [.prettyPrinted]) {
            try? data.write(to: etagsFile, options: .atomic)
        }
    }

    /// Checks `https://jobellet.github.io/uneye/` (and local repo if present) for updated web files.
    func syncWithWeb(force: Bool = false, completion: ((Int, Bool) -> Void)? = nil) {
        bootstrapLocalAssets()
        let now = Date()
        if !force && now.timeIntervalSince(lastRemoteSyncTime) < 45 {
            completion?(0, false)
            return
        }

        syncQueue.async {
            if self.isSyncing {
                DispatchQueue.main.async { completion?(0, false) }
                return
            }
            self.isSyncing = true
            self.lastRemoteSyncTime = Date()

            // Check core code/UI files on every sync; large static binaries only if missing locally or forced
            let coreAssets = [
                "index.html",
                "manifest.json",
                "style.css",
                "app.js",
                "formats.js",
                "import.js",
                "import_ui.js",
                "hmm_prelabel.js",
                "nearmiss.js",
                "train_worker.js",
                "sw.js",
                "web/uneye_train.js",
                "models/combined.json"
            ]
            let heavyAssets = [
                "web/uneye_train.wasm",
                "web/h5wasm.js",
                "models/combined.bin",
                "logo.jpeg",
                "logo_circle.png"
            ]

            let fm = FileManager.default
            var assetsToCheck = coreAssets
            for h in heavyAssets {
                let dest = self.webDir.appendingPathComponent(h)
                if force || !fm.fileExists(atPath: dest.path) {
                    assetsToCheck.append(h)
                }
            }

            let config = URLSessionConfiguration.ephemeral
            config.timeoutIntervalForRequest = 10
            config.timeoutIntervalForResource = 30
            config.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
            let session = URLSession(configuration: config)

            let group = DispatchGroup()
            let lock = NSLock()
            var updatedCount = 0
            var hadNetworkResponse = false

            for rel in assetsToCheck {
                guard let url = URL(string: rel, relativeTo: self.remoteBaseURL) else { continue }
                group.enter()
                var req = URLRequest(url: url)
                req.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
                lock.lock()
                let prevTag = self.etags[rel]
                lock.unlock()
                if let tag = prevTag, !force {
                    req.setValue(tag, forHTTPHeaderField: "If-None-Match")
                }

                session.dataTask(with: req) { data, resp, err in
                    defer { group.leave() }
                    guard err == nil, let http = resp as? HTTPURLResponse else { return }
                    lock.lock()
                    hadNetworkResponse = true
                    lock.unlock()

                    if http.statusCode == 304 {
                        return
                    }
                    guard http.statusCode == 200, let data = data, !data.isEmpty else { return }

                    let dest = self.webDir.appendingPathComponent(rel)
                    // If a local repo file exists, only let remote overwrite if remote has a parsed Last-Modified strictly newer than local
                    if let rDir = self.localRepoDir {
                        let localFile = rDir.appendingPathComponent(rel)
                        if fm.fileExists(atPath: localFile.path) {
                            if let localData = try? Data(contentsOf: localFile), localData == data {
                                if let newTag = http.value(forHTTPHeaderField: "ETag") {
                                    lock.lock()
                                    self.etags[rel] = newTag
                                    lock.unlock()
                                }
                                return
                            }
                            let localDate = (try? fm.attributesOfItem(atPath: localFile.path)[.modificationDate] as? Date) ?? Date()
                            let remoteDate = http.value(forHTTPHeaderField: "Last-Modified").flatMap { WebAssetManager.parseHTTPDate($0) }
                            if remoteDate == nil || remoteDate! <= localDate.addingTimeInterval(30) {
                                return
                            }
                        }
                    }

                    let existingData = try? Data(contentsOf: dest)
                    if existingData != data {
                        try? fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
                        try? data.write(to: dest, options: .atomic)
                        lock.lock()
                        updatedCount += 1
                        lock.unlock()
                    }
                    if let newTag = http.value(forHTTPHeaderField: "ETag") {
                        lock.lock()
                        self.etags[rel] = newTag
                        lock.unlock()
                    }
                }.resume()
            }

            group.wait()
            self.saveETags()
            self.isSyncing = false

            DispatchQueue.main.async {
                completion?(updatedCount, hadNetworkResponse)
            }
        }
    }

    private static func parseHTTPDate(_ str: String) -> Date? {
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.timeZone = TimeZone(secondsFromGMT: 0)
        for pattern in ["EEE, dd MMM yyyy HH:mm:ss 'GMT'", "EEE, dd MMM yyyy HH:mm:ss z", "EEE, dd MMM yyyy HH:mm:ss zzz"] {
            fmt.dateFormat = pattern
            if let d = fmt.date(from: str) {
                return d
            }
        }
        return nil
    }

    func resolveFile(relativePath: String) -> URL? {
        let clean = relativePath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let rel = clean.isEmpty ? "index.html" : clean
        if rel.contains("..") { return nil }

        // Always check if localRepoDir has a fresher version right now
        let target = webDir.appendingPathComponent(rel)
        if let rDir = localRepoDir {
            let repoCandidate = rDir.appendingPathComponent(rel)
            copyIfNewer(from: repoCandidate, to: target)
        }
        if FileManager.default.fileExists(atPath: target.path) {
            return target
        }
        if let bDir = bundleWebDir {
            let bundleCandidate = bDir.appendingPathComponent(rel)
            if FileManager.default.fileExists(atPath: bundleCandidate.path) {
                return bundleCandidate
            }
        }
        return nil
    }
}

// MARK: - Embedded Loopback HTTP Server (127.0.0.1)
final class LocalWebServer {
    private(set) var port: UInt16 = 18765
    private var serverFD: Int32 = -1
    private var isRunning = false
    private let acceptQueue = DispatchQueue(label: "com.uneye.labeler.http.accept", qos: .userInteractive)
    private let clientQueue = DispatchQueue(label: "com.uneye.labeler.http.client", qos: .userInteractive, attributes: .concurrent)

    private var sharedFilesLock = NSLock()
    private var sharedFiles: [String: URL] = [:]

    func registerSharedFile(_ url: URL) -> (token: String, name: String) {
        let token = UUID().uuidString
        sharedFilesLock.lock()
        sharedFiles[token] = url
        sharedFilesLock.unlock()
        return (token, url.lastPathComponent)
    }

    func start() -> UInt16 {
        if isRunning { return port }

        let candidatePorts: [UInt16] = [18765, 18766, 18767, 0]
        for cand in candidatePorts {
            let fd = socket(AF_INET, SOCK_STREAM, 0)
            guard fd >= 0 else { continue }

            var yes: Int32 = 1
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))

            var addr = sockaddr_in()
            addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = cand.bigEndian
            addr.sin_addr.s_addr = inet_addr("127.0.0.1")

            let bindRes = withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            if bindRes == 0 && Darwin.listen(fd, 128) == 0 {
                var actualAddr = sockaddr_in()
                var len = socklen_t(MemoryLayout<sockaddr_in>.size)
                _ = withUnsafeMutablePointer(to: &actualAddr) {
                    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                        getsockname(fd, $0, &len)
                    }
                }
                self.port = UInt16(bigEndian: actualAddr.sin_port)
                self.serverFD = fd
                self.isRunning = true
                break
            }
            Darwin.close(fd)
        }

        guard isRunning else { return port }

        acceptQueue.async { [weak self] in
            while let self = self, self.isRunning && self.serverFD >= 0 {
                var clientAddr = sockaddr_in()
                var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)
                let clientFD = withUnsafeMutablePointer(to: &clientAddr) {
                    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                        Darwin.accept(self.serverFD, $0, &addrLen)
                    }
                }
                if clientFD < 0 { continue }
                var one: Int32 = 1
                setsockopt(clientFD, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
                self.clientQueue.async {
                    self.handleClient(clientFD)
                }
            }
        }
        return port
    }

    private func handleClient(_ fd: Int32) {
        defer { Darwin.close(fd) }

        var buf = [UInt8](repeating: 0, count: 8192)
        let n = Darwin.read(fd, &buf, buf.count - 1)
        guard n > 0 else { return }
        let reqStr = String(decoding: buf[0..<n], as: UTF8.self)
        guard let firstLine = reqStr.split(separator: "\r\n").first ?? reqStr.split(separator: "\n").first else { return }
        let parts = firstLine.split(separator: " ")
        guard parts.count >= 2 else { return }
        let method = String(parts[0])
        let rawTarget = String(parts[1])

        guard method == "GET" || method == "HEAD" else {
            sendResponse(fd: fd, status: "405 Method Not Allowed", mime: "text/plain", body: Data(), headOnly: true)
            return
        }

        let pathOnly = rawTarget.split(separator: "?", maxSplits: 1).first.map(String.init) ?? "/"
        let decodedPath = pathOnly.removingPercentEncoding ?? pathOnly

        if decodedPath == "/__mac_open_file__" {
            if let qIdx = rawTarget.firstIndex(of: "?") {
                let query = String(rawTarget[rawTarget.index(after: qIdx)...])
                var token: String?
                for pair in query.split(separator: "&") {
                    let kv = pair.split(separator: "=", maxSplits: 1)
                    if kv.count == 2 && kv[0] == "id" {
                        token = String(kv[1])
                    }
                }
                if let t = token {
                    sharedFilesLock.lock()
                    let fileURL = sharedFiles[t]
                    sharedFilesLock.unlock()
                    if let fURL = fileURL, let data = try? Data(contentsOf: fURL, options: .mappedIfSafe) {
                        sendResponse(fd: fd, status: "200 OK", mime: "application/octet-stream", body: data, headOnly: method == "HEAD")
                        return
                    }
                }
            }
            sendResponse(fd: fd, status: "404 Not Found", mime: "text/plain", body: Data("Not Found".utf8), headOnly: method == "HEAD")
            return
        }

        guard let fileURL = WebAssetManager.shared.resolveFile(relativePath: decodedPath),
              let data = try? Data(contentsOf: fileURL, options: .mappedIfSafe) else {
            sendResponse(fd: fd, status: "404 Not Found", mime: "text/plain", body: Data("Not Found".utf8), headOnly: method == "HEAD")
            return
        }

        let mime = Self.mimeType(for: fileURL.pathExtension.lowercased())
        sendResponse(fd: fd, status: "200 OK", mime: mime, body: data, headOnly: method == "HEAD")
    }

    private func sendResponse(fd: Int32, status: String, mime: String, body: Data, headOnly: Bool) {
        let headers = [
            "HTTP/1.1 \(status)",
            "Content-Type: \(mime)",
            "Content-Length: \(body.count)",
            "Cache-Control: no-store, no-cache, must-revalidate",
            "Cross-Origin-Opener-Policy: same-origin",
            "Cross-Origin-Embedder-Policy: require-corp",
            "Cross-Origin-Resource-Policy: same-origin",
            "Connection: close",
            "",
            ""
        ].joined(separator: "\r\n")

        if let hData = headers.data(using: .utf8) {
            writeAll(fd: fd, data: hData)
        }
        if !headOnly && !body.isEmpty {
            writeAll(fd: fd, data: body)
        }
    }

    private func writeAll(fd: Int32, data: Data) {
        data.withUnsafeBytes { rawBuf in
            guard let base = rawBuf.baseAddress else { return }
            var sent = 0
            while sent < data.count {
                let w = Darwin.write(fd, base.advanced(by: sent), data.count - sent)
                if w <= 0 { break }
                sent += w
            }
        }
    }

    private static func mimeType(for ext: String) -> String {
        switch ext {
        case "html", "htm": return "text/html; charset=utf-8"
        case "css": return "text/css; charset=utf-8"
        case "js", "mjs": return "application/javascript; charset=utf-8"
        case "json", "webmanifest": return "application/json; charset=utf-8"
        case "wasm": return "application/wasm"
        case "png": return "image/png"
        case "jpg", "jpeg": return "image/jpeg"
        case "svg": return "image/svg+xml"
        case "ico": return "image/x-icon"
        default: return "application/octet-stream"
        }
    }
}

// MARK: - Main Application Delegate & Window Controller
final class AppDelegate: NSObject, NSApplicationDelegate, WKUIDelegate, WKNavigationDelegate, WKScriptMessageHandler {
    var window: NSWindow!
    var webView: WKWebView!
    let server = LocalWebServer()
    var pageLoaded = false
    var pendingOpenFiles: [URL] = []
    var lastSaveDirectory: URL?

    private struct SaveItem {
        let filename: String
        let data: Data
    }
    private var saveQueue: [SaveItem] = []
    private var isShowingSavePanel = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMainMenu()

        let port = server.start()
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()
        config.preferences.setValue(true, forKey: "developerExtrasEnabled")

        let bridgeJS = """
        window.UnEyeMacApp = true;
        window.AndroidBridge = {
            saveFile: function(filename, base64, mimeType) {
                window.webkit.messageHandlers.macBridge.postMessage({
                    action: 'saveFile',
                    filename: String(filename || 'download.bin'),
                    base64: String(base64 || ''),
                    mimeType: String(mimeType || 'application/octet-stream')
                });
            }
        };
        window.__macOpenSharedFile = async function(token, filename) {
            try {
                const resp = await fetch('/__mac_open_file__?id=' + encodeURIComponent(token));
                if (!resp.ok) throw new Error('HTTP ' + resp.status);
                const buf = await resp.arrayBuffer();
                const file = new File([buf], filename);
                const dt = new DataTransfer();
                dt.items.add(file);
                const inp = document.getElementById('fileInput');
                if (inp) {
                    inp.files = dt.files;
                    inp.dispatchEvent(new Event('change'));
                }
            } catch (e) {
                console.error('Failed to open file from macOS:', e);
            }
        };
        """
        let userScript = WKUserScript(source: bridgeJS, injectionTime: .atDocumentStart, forMainFrameOnly: true)
        config.userContentController.addUserScript(userScript)
        config.userContentController.add(self, name: "macBridge")

        let screenRect = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1280, height: 800)
        let winW = min(1440, max(1060, screenRect.width * 0.86))
        let winH = min(920, max(700, screenRect.height * 0.86))
        let winRect = NSRect(
            x: screenRect.origin.x + (screenRect.width - winW) / 2,
            y: screenRect.origin.y + (screenRect.height - winH) / 2,
            width: winW,
            height: winH
        )

        window = NSWindow(
            contentRect: winRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "U'n'Eye — Offline Saccade Labeler"
        window.minSize = NSSize(width: 820, height: 540)
        window.collectionBehavior = [.fullScreenPrimary]
        window.setFrameAutosaveName("UnEyeMainWindow")

        webView = WKWebView(frame: window.contentView!.bounds, configuration: config)
        webView.autoresizingMask = [.width, .height]
        webView.uiDelegate = self
        webView.navigationDelegate = self
        window.contentView?.addSubview(webView)

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        let startURL = URL(string: "http://127.0.0.1:\(port)/index.html")!
        webView.load(URLRequest(url: startURL, cachePolicy: .reloadIgnoringLocalCacheData))

        // Automatically sync with https://jobellet.github.io/uneye/ on startup
        checkForWebUpdates(force: false, silent: true)
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        // Check for web updates when switching back to the app (throttled to every 45s)
        checkForWebUpdates(force: false, silent: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    // Support Finder "Open With -> U'n'Eye" and dragging files onto the Dock icon
    func application(_ sender: NSApplication, openFiles filenames: [String]) {
        for path in filenames {
            let url = URL(fileURLWithPath: path)
            openDocumentURL(url)
        }
        sender.reply(toOpenOrPrint: .success)
    }

    func openDocumentURL(_ url: URL) {
        if !pageLoaded {
            pendingOpenFiles.append(url)
            return
        }
        let info = server.registerSharedFile(url)
        let escapedToken = info.token.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "'", with: "\\'")
        let escapedName = info.name.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "'", with: "\\'")
        webView.evaluateJavaScript("window.__macOpenSharedFile && window.__macOpenSharedFile('\(escapedToken)', '\(escapedName)');", completionHandler: nil)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        pageLoaded = true
        if !pendingOpenFiles.isEmpty {
            let files = pendingOpenFiles
            pendingOpenFiles.removeAll()
            for f in files {
                openDocumentURL(f)
            }
        }
    }

    // MARK: - Web Auto-Update Actions
    @objc func checkForWebUpdatesMenuAction(_ sender: Any?) {
        window.title = "U'n'Eye — Checking for web updates…"
        checkForWebUpdates(force: true, silent: false)
    }

    @objc func reloadWebApp(_ sender: Any?) {
        WebAssetManager.shared.syncWithWeb(force: true) { [weak self] _, _ in
            guard let self = self else { return }
            self.webView.evaluateJavaScript("window.flushSave && window.flushSave();") { _, _ in
                self.webView.reloadFromOrigin()
            }
        }
    }

    @objc func openFileMenuAction(_ sender: Any?) {
        webView.evaluateJavaScript("document.getElementById('openBtn') && document.getElementById('openBtn').click();", completionHandler: nil)
    }

    @objc func saveLabelsMenuAction(_ sender: Any?) {
        webView.evaluateJavaScript("window.saveLabelsJson && window.saveLabelsJson();", completionHandler: nil)
    }

    @objc func showShortcutsMenuAction(_ sender: Any?) {
        webView.evaluateJavaScript("window.openShortcutsModal && window.openShortcutsModal();", completionHandler: nil)
    }

    func checkForWebUpdates(force: Bool, silent: Bool) {
        WebAssetManager.shared.syncWithWeb(force: force) { [weak self] updatedCount, hadNetwork in
            guard let self = self else { return }
            if updatedCount > 0 {
                self.window.title = "U'n'Eye — Offline Saccade Labeler (Updated)"
                self.webView.evaluateJavaScript("window.flushSave && window.flushSave();") { _, _ in
                    self.webView.reloadFromOrigin()
                }
                if !silent {
                    let alert = NSAlert()
                    alert.messageText = "U'n'Eye Updated"
                    alert.informativeText = "\(updatedCount) file(s) were updated from the latest web version and reloaded."
                    alert.alertStyle = .informational
                    alert.runModal()
                }
            } else {
                self.window.title = "U'n'Eye — Offline Saccade Labeler"
                if !silent {
                    let alert = NSAlert()
                    alert.messageText = hadNetwork ? "U'n'Eye is Up to Date" : "Offline Mode"
                    alert.informativeText = hadNetwork
                        ? "You are running the latest version synchronized with https://jobellet.github.io/uneye/."
                        : "Could not reach https://jobellet.github.io/uneye/. Running from the local offline cache."
                    alert.alertStyle = .informational
                    alert.runModal()
                }
            }
        }
    }

    // MARK: - WKScriptMessageHandler (File Saving)
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "macBridge",
              let body = message.body as? [String: Any],
              let action = body["action"] as? String else { return }

        if action == "saveFile" {
            let filename = (body["filename"] as? String) ?? "uneye_export.bin"
            let base64 = (body["base64"] as? String) ?? ""
            guard let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters) else { return }
            saveQueue.append(SaveItem(filename: filename, data: data))
            processNextSaveItem()
        }
    }

    private func processNextSaveItem() {
        guard !isShowingSavePanel, !saveQueue.isEmpty else { return }
        isShowingSavePanel = true
        let item = saveQueue.removeFirst()

        let savePanel = NSSavePanel()
        savePanel.title = "Save \(item.filename)"
        savePanel.nameFieldStringValue = item.filename
        savePanel.canCreateDirectories = true
        if let dir = lastSaveDirectory {
            savePanel.directoryURL = dir
        }

        savePanel.beginSheetModal(for: window) { [weak self] response in
            guard let self = self else { return }
            if response == .OK, let url = savePanel.url {
                self.lastSaveDirectory = url.deletingLastPathComponent()
                try? item.data.write(to: url, options: .atomic)
            }
            self.isShowingSavePanel = false
            self.processNextSaveItem()
        }
    }

    // MARK: - WKUIDelegate (Native File Picker & Alerts)
    func webView(_ webView: WKWebView,
                 runOpenPanelWith parameters: WKOpenPanelParameters,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = parameters.allowsDirectories
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.resolvesAliases = true
        panel.beginSheetModal(for: window) { response in
            if response == .OK {
                completionHandler(panel.urls)
            } else {
                completionHandler(nil)
            }
        }
    }

    func webView(_ webView: WKWebView,
                 runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = "U'n'Eye"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.beginSheetModal(for: window) { _ in
            completionHandler()
        }
    }

    func webView(_ webView: WKWebView,
                 runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = "U'n'Eye"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: window) { resp in
            completionHandler(resp == .alertFirstButtonReturn)
        }
    }

    // MARK: - macOS Menu Bar
    private func setupMainMenu() {
        let mainMenu = NSMenu()
        NSApp.mainMenu = mainMenu

        // 1. App Menu
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenuItem.submenu = appMenu
        appMenu.addItem(withTitle: "About U'n'Eye", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(withTitle: "Check for Web Updates…", action: #selector(checkForWebUpdatesMenuAction(_:)), keyEquivalent: "u")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "Keyboard Shortcuts…", action: #selector(showShortcutsMenuAction(_:)), keyEquivalent: ",")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "Hide U'n'Eye", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = appMenu.addItem(withTitle: "Hide Others", action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        appMenu.addItem(withTitle: "Show All", action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "Quit U'n'Eye", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        // 2. File Menu
        let fileMenuItem = NSMenuItem()
        mainMenu.addItem(fileMenuItem)
        let fileMenu = NSMenu(title: "File")
        fileMenuItem.submenu = fileMenu
        fileMenu.addItem(withTitle: "Open Eye Trace…", action: #selector(openFileMenuAction(_:)), keyEquivalent: "o")
        fileMenu.addItem(withTitle: "Save Labels (.json)…", action: #selector(saveLabelsMenuAction(_:)), keyEquivalent: "s")
        fileMenu.addItem(NSMenuItem.separator())
        fileMenu.addItem(withTitle: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")

        // 3. Edit Menu
        let editMenuItem = NSMenuItem()
        mainMenu.addItem(editMenuItem)
        let editMenu = NSMenu(title: "Edit")
        editMenuItem.submenu = editMenu
        editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

        // 4. View Menu
        let viewMenuItem = NSMenuItem()
        mainMenu.addItem(viewMenuItem)
        let viewMenu = NSMenu(title: "View")
        viewMenuItem.submenu = viewMenu
        viewMenu.addItem(withTitle: "Sync & Reload from Web", action: #selector(reloadWebApp(_:)), keyEquivalent: "r")
        viewMenu.addItem(NSMenuItem.separator())
        let fullScreen = viewMenu.addItem(withTitle: "Toggle Full Screen", action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
        fullScreen.keyEquivalentModifierMask = [.command, .control]

        // 5. Window Menu
        let winMenuItem = NSMenuItem()
        mainMenu.addItem(winMenuItem)
        let winMenu = NSMenu(title: "Window")
        winMenuItem.submenu = winMenu
        winMenu.addItem(withTitle: "Minimize", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        winMenu.addItem(withTitle: "Zoom", action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        NSApp.windowsMenu = winMenu
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
