#!/usr/bin/env python3
"""Headless browser test for the U'n'Eye web labeler.

Tests end-to-end functionality in headless Chromium:
- UI load and Demo dataset generation
- Pass 1 & Pass 2 interactions
- On-device Web Worker WASM training
- Live progress updates and convergence flag
- Cancellation
- IndexedDB persistence across reload
"""
import functools
import http.server
import os
import threading
import time
from playwright.sync_api import sync_playwright

PORT = 8899

def run_server(repo_dir):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=repo_dir)
    httpd = http.server.HTTPServer(("127.0.0.1", PORT), handler)
    httpd.serve_forever()

def main():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    print(f"Starting local HTTP server for {repo_dir} on port {PORT}...")
    server_thread = threading.Thread(target=run_server, args=(repo_dir,), daemon=True)
    server_thread.start()
    time.sleep(1)

    print("Launching headless Chromium via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Listen to console logs
        page.on("console", lambda msg: print(f"[Browser Console {msg.type}]: {msg.text}"))

        print(f"Navigating to http://127.0.0.1:{PORT}/index.html ...")
        page.goto(f"http://127.0.0.1:{PORT}/index.html")
        page.wait_for_load_state("networkidle")

        title = page.title()
        print(f"Page title: {title}")
        assert "U'n'Eye" in title

        # Click Demo button
        print("Clicking Demo button...")
        page.click("#demoBtn")
        time.sleep(1)

        trial_status = page.text_content("#trialStatus")
        print(f"Trial status: {trial_status}")
        assert "Trial 1/" in trial_status

        # Review a portion of trial 1 and trial 2 to have reviewed training data
        print("Reviewing demo trials (setting reviewed ranges)...")
        page.evaluate("""() => {
            // Mark trial 1 (trial index 1, non-test) as reviewed
            window.state.reviewedRanges[1] = [[0, 999]];
            // Add a saccade event in trial 1
            window.state.expertEvents[1] = [{ onset: 200, offset: 250, cls: 1, fromPrelabel: false }];
            // Also mark trial 2 as reviewed
            window.state.reviewedRanges[2] = [[0, 999]];
            window.state.expertEvents[2] = [{ onset: 300, offset: 340, cls: 1, fromPrelabel: false }];
            window.updateStatus();
            window.render();
        }""")

        reviewed_text = page.text_content("#reviewedStatus")
        print(f"Reviewed status: {reviewed_text}")

        # Start Training
        print("Triggering On-Device Training (clicking Train button)...")
        page.click("#trainNowBtn")

        # Wait for status to show active training
        page.wait_for_function("""() => {
            const el = document.getElementById('trainStatus');
            return el && (el.textContent.includes('Epoch') || el.textContent.includes('Training') || el.textContent.includes('Preparing'));
        }""", timeout=15000)

        train_status = page.text_content("#trainStatus")
        print(f"Training active status: {train_status}")

        # Test Cancel button
        print("Testing Cancel button...")
        page.click("#cancelTrainBtn")
        time.sleep(2)
        train_status_cancelled = page.text_content("#trainStatus")
        print(f"Status after cancel: {train_status_cancelled}")
        assert "cancelled" in train_status_cancelled.lower() or "idle" in train_status_cancelled.lower()
        print("PASS: Cooperative cancellation verified!")

        # Now test full training
        print("\nStarting full training round...")
        page.click("#trainNowBtn")

        # Wait for training completion (max 60 seconds)
        print("Waiting for training to complete in background worker...")
        page.wait_for_function("""() => {
            const el = document.getElementById('trainStatus');
            return el && (el.textContent.includes('Trained') || el.textContent.includes('completed'));
        }""", timeout=60000)

        final_train_status = page.text_content("#trainStatus")
        print(f"Final training status: {final_train_status}")
        assert "Trained" in final_train_status or "completed" in final_train_status

        # Check convergence history
        conv_hist_len = page.evaluate("() => window.state.convergenceHistory.length")
        print(f"Convergence history records: {conv_hist_len}")
        assert conv_hist_len > 0

        # Check trained model binary presence
        has_model = page.evaluate("() => window.state.trainedModelBin !== null && window.state.trainedModelBin.length > 0")
        print(f"Trained model in memory: {has_model}")
        assert has_model

        # Test page reload and IndexedDB restore
        print("Testing page reload and IndexedDB restore...")
        page.reload()
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        restored_trials = page.evaluate("() => window.state.trials.length")
        print(f"Restored trials count after reload: {restored_trials}")
        assert restored_trials == 8

        restored_model = page.evaluate("() => window.state.trainedModelBin !== null")
        print(f"Restored model after reload: {restored_model}")
        assert restored_model

        print("\n=================================================================")
        print("   ALL HEADLESS BROWSER PLAYWRIGHT TESTS PASSED SUCCESSFULLY!    ")
        print("=================================================================")

        browser.close()

if __name__ == "__main__":
    main()
