"""Open a test or bench page in headless Google Chrome (the installed one: real GPU adapter) and print
window.__result as JSON.

    python tests/run_page.py "tests/parity.html?precision=f32" [--timeout 900] [--browser chrome|webkit]
"""
import argparse
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--browser", default="chrome")
    a = p.parse_args()
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(a.root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with sync_playwright() as pw:
        if a.browser == "chrome":
            browser = pw.chromium.launch(channel="chrome", headless=True,
                                         args=["--headless=new", "--enable-unsafe-webgpu", "--enable-features=Vulkan"])
        else:
            browser = pw.webkit.launch(headless=True)
        page = browser.new_page()
        page.on("console", lambda m: print("console:", m.text, file=sys.stderr))
        page.on("pageerror", lambda e: print("pageerror:", e, file=sys.stderr))
        page.goto(f"http://127.0.0.1:{server.server_port}/{a.path}")
        page.wait_for_function("window.__result !== undefined", timeout=a.timeout * 1000)
        print(json.dumps(page.evaluate("window.__result")))
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main()
