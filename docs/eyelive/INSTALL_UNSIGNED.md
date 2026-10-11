# Installing EyeLive without a signature (no administrator rights needed)

EyeLive is distributed as an **unsigned** program: operating systems warn about every unsigned download, antivirus programs are more suspicious of programs they do not know. The program does nothing hidden: it opens a UDP port only when you start a session, writes the session file in the folder you choose, and has no installer, no service, no auto-start. The source is public and the release contains `SHA256SUMS.txt`.

## 1. Check the download (all systems)
Compare the SHA-256 of the file with the one in `SHA256SUMS.txt` of the release.
* Windows (PowerShell): `Get-FileHash .\eyelive-windows.zip -Algorithm SHA256`
* macOS / Linux: `shasum -a 256 eyelive-macos.zip`

## 2. Windows
1. Before extracting, right-click the downloaded `.zip` -> **Properties** -> tick **Unblock** -> OK (or in PowerShell: `Unblock-File .\eyelive-windows.zip`). This removes the "downloaded from the internet" mark that triggers most warnings.
2. Extract it into a folder you own (for example `C:\Users\<you>\EyeLive`). No installation, no administrator rights.
3. First start: if **"Windows protected your PC"** (SmartScreen) appears, click **More info** -> **Run anyway**.
4. If Microsoft Defender removes or blocks the file: **Windows Security -> Virus & threat protection -> Protection history**, open the entry and choose **Allow on device** (or **Restore**). To avoid it again, add the EyeLive folder as an exclusion: **Virus & threat protection -> Manage settings -> Exclusions -> Add or remove exclusions -> Add an exclusion -> Folder**.
5. On a computer managed by your institution these settings may be locked by policy: ask the IT department to allow the EyeLive folder (give them `SHA256SUMS.txt` and this page).
6. The first time EyeLive listens for the tracker, Windows Firewall may ask whether to allow it: choose **Private networks**. If the tracker computer is the same machine (UDP to `127.0.0.1`), no firewall rule is needed.

## 3. macOS
1. Unzip. The first time, **Control-click (right-click) the app -> Open -> Open**. On recent macOS versions (Sequoia and later) this may not offer the choice: open **System Settings -> Privacy & Security**, scroll down to the message that EyeLive was blocked and click **Open Anyway**.
2. Alternative from Terminal (removes the quarantine mark, then it opens normally): `xattr -dr com.apple.quarantine /path/to/EyeLive.app`
3. Apple Silicon requires at least an ad-hoc signature; the release is ad-hoc signed (`codesign --force --sign -`), which is not a developer signature and does not remove step 1.

## 4. Linux
`chmod +x eyelive` and run it. If the graphical window does not open, check that OpenGL 3.2 is available (`glxinfo | grep "OpenGL version"`).

## 5. If an antivirus still complains
* A false alarm on a new unsigned program is normal. Keep the SHA-256 check as the proof that the file is the released one.
* Report it as a false positive to the antivirus vendor (Microsoft has a "submit a file for analysis" form for Defender); this usually clears it within days.
* Never disable the antivirus as a whole: add the EyeLive folder as an exclusion instead.
