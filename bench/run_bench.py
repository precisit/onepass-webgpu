"""Run the SPEED-PROTOCOL bench: every backend three times in a fresh headless browser, then the
median of the three medians. Writes one JSON record to bench/results/.

    python bench/run_bench.py --data work/c4-v2 [--browser chrome] [--runs 3] [--only onepass-f32]

--data must hold reference/boards.bin and reference/reference.json (tests/make_reference.py), plan.json
(compiler/compile_onepass.py) and the model files onepass-c4-v2.onnx / onepass-c4-v2-int8.onnx.
"""
import argparse
import datetime
import functools
import gzip
import hashlib
import http.server
import json
import os
import platform
import statistics
import subprocess
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {
    "onepass-f32": "backend=onepass&precision=f32",
    "onepass-f16": "backend=onepass&precision=f16",
    "onepass-int8": "backend=onepass&precision=f32&weights=int8&plan=../work/c4-v2-int8/plan.json",
    "ort-wasm-int8": "backend=ort&model=int8",
    "ort-wasm-fp32": "backend=ort&model=fp32",
}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def machine():
    return {"model": sh("sysctl -n hw.model"), "chip": sh("sysctl -n machdep.cpu.brand_string"),
            "gpu_cores": sh("system_profiler SPDisplaysDataType | grep -i 'cores' | head -1").split(":")[-1].strip(),
            "os": platform.platform(), "power": sh("pmset -g batt | head -1").replace("Now drawing from ", "")}


def serve(root: Path):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--browser", default="chrome", choices=["chrome"])
    p.add_argument("--chrome", help="path to a Chrome executable (e.g. Chrome for Testing) instead of the installed Chrome")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--only", nargs="*", default=list(BACKENDS))
    p.add_argument("--out", type=Path, default=ROOT / "bench" / "results")
    a = p.parse_args()
    data = a.data.resolve()
    rel = os.path.relpath(data, ROOT)
    server = serve(ROOT)
    bundle = (ROOT / "dist" / "onepass-webgpu.js").read_bytes()
    record = {
        "commit": (sh(f"git -C {ROOT} rev-parse --short HEAD") + ("-dirty" if sh(f"git -C {ROOT} status --porcelain -- runtime bench") else ""))
                  or ((ROOT / "COMMIT").read_text().strip() if (ROOT / "COMMIT").exists() else ""),
        "protocol_sha256": hashlib.sha256((ROOT / "SPEED-PROTOCOL.md").read_bytes()).hexdigest(),
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "machine": machine(),
        "runtime_bundle": {"bytes": len(bundle), "gzip_bytes": len(gzip.compress(bundle, 9)),
                           "sha256": hashlib.sha256(bundle).hexdigest()},
        "models": {f: {"bytes": (data / f).stat().st_size, "sha256": hashlib.sha256((data / f).read_bytes()).hexdigest()}
                   for f in ("onepass-c4-v2.onnx", "onepass-c4-v2-int8.onnx")},
        "backends": {},
    }
    with sync_playwright() as pw:
        for name in a.only:
            runs = []
            for i in range(a.runs):
                if a.chrome:
                    browser = pw.chromium.launch(executable_path=a.chrome, headless=True, args=["--headless=new"])
                    record["browser"] = f"Chrome for Testing {browser.version} (headless=new)"
                else:
                    browser = pw.chromium.launch(channel="chrome", headless=True, args=["--headless=new"])
                    record["browser"] = f"Chrome {browser.version} (headless=new)"
                page = browser.new_page()
                load_before = os.getloadavg()
                page.goto(f"http://127.0.0.1:{server.server_port}/bench/bench.html?{BACKENDS[name]}&data=../{rel}/")
                page.wait_for_function("window.__result !== undefined", timeout=900_000)
                r = page.evaluate("window.__result")
                r["loadavg_before"] = load_before
                r["loadavg_after"] = os.getloadavg()
                browser.close()
                if "error" in r:
                    raise SystemExit(f"{name}: {r['error']}")
                software = "swiftshader" in json.dumps(r.get("adapter", {})).lower()
                if software:
                    raise SystemExit(f"{name}: software adapter, result invalid")
                runs.append(r)
                warm = r.get("warm", {})
                print(f"{name} run {i + 1}: check {r['check']['agree']}/{r['check']['positions']}, "
                      f"median {warm.get('median')} ms, p95 {warm.get('p95')} ms, load {load_before[0]:.1f}", flush=True)
            summary = {"runs": runs}
            if all("warm" in r for r in runs):
                summary["median_of_medians_ms"] = statistics.median(r["warm"]["median"] for r in runs)
                summary["median_of_p95_ms"] = statistics.median(r["warm"]["p95"] for r in runs)
                summary["median_of_means_ms"] = statistics.median(r["warm"]["mean"] for r in runs)
                summary["cold_total_ms_median"] = statistics.median(r["cold"]["total_ms"] for r in runs)
            record["backends"][name] = summary
    server.shutdown()
    a.out.mkdir(parents=True, exist_ok=True)
    chip = record["machine"]["chip"].replace("Apple ", "").replace(" ", "")
    path = a.out / f"{record['time_utc'][:10]}-{chip}-chrome.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {path}")
    for name, s in record["backends"].items():
        print(f"{name:14s} median {s.get('median_of_medians_ms')} ms  p95 {s.get('median_of_p95_ms')} ms  cold {s.get('cold_total_ms_median')} ms")


if __name__ == "__main__":
    main()
