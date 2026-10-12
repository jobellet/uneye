# Signing (optional)

The owner decided that signed distribution is **not required**: releases are unsigned zip files with SHA-256 files and `INSTALL.txt` (how to unblock them on Windows SmartScreen and macOS Gatekeeper). This page says where a certificate would plug in later; none of it is active.

## Windows (OV or EV code-signing certificate)
Add the repository secrets `WINDOWS_CERT_PFX_BASE64` and `WINDOWS_CERT_PASSWORD`. In `.github/workflows/eyelive-release.yml` the step "Sign (Windows, skipped without secret)" then runs `signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /f cert.pfx /p ... eyelive.exe eyelive_cli.exe`. An EV certificate removes the SmartScreen warning at once; an OV certificate removes it after the file has built reputation. Sign BEFORE the archive and the SHA-256 files are made.

## macOS (Developer ID + notarization)
Secrets: `MACOS_CERT_P12_BASE64`, `MACOS_CERT_PASSWORD`, `APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_APP_PASSWORD`. The step "Sign and notarize (macOS, skipped without secrets)" runs `codesign --force --options runtime --timestamp --sign "Developer ID Application: ..."` on both executables, zips them and runs `xcrun notarytool submit --wait`, then `xcrun stapler staple` (a bare executable cannot be stapled: ship it inside a signed .dmg or .pkg in that case).

Without these secrets the steps are skipped and the build stays green.
