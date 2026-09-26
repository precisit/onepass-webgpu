# SPEED-PROTOCOL: how onepass-webgpu is timed (frozen before the first measurement)

This file is hashed (`SPEED-PROTOCOL.sha256`) before the first timing run that is reported.
Amendments are dated and additive, appended at the end. A target that is missed is reported as missed.

## What is compared

The same model, v2 of the Connect Four one-pass scorer (7.38 M parameters), on:

| backend | model file |
| --- | --- |
| onnxruntime-web 1.30.0, wasm (the reference in-browser runtime) | `onepass-c4-v2-int8.onnx` (7.8 MB) and `onepass-c4-v2.onnx` (fp32, 29.7 MB) |
| onepass-webgpu, f32 | compiled plan + fp32 weights |
| onepass-webgpu, f16 (when `shader-f16` is available) | compiled plan + fp16 weights |

Later variants (for example ternary weights) are added as new rows under the same rules.

## Where it runs

- **Browsers:** Chrome (stable) and Safari 26. Other browsers are out of scope for now.
- **Machines:** Apple M1 Max first; M4 and M5 Pro when available. Every result records the machine.
- **The adapter must be a hardware GPU.** Every result records `GPUAdapter.info` (vendor, architecture).
  A result from a software adapter (for example `swiftshader`) is invalid and is not reported. Headless
  Chrome is run with the installed Google Chrome (`--headless=new`), which exposes the Metal adapter.
- **Conditions:** power adapter connected, no other heavy jobs started by us, other tabs closed. The load
  average and the time are recorded with every run; we do not control other processes on shared machines,
  so every run is repeated three times and the median of the three medians is reported.

## The workload

- **Positions:** the frozen eval set of the Connect Four project (17 325 positions). For timing, a fixed,
  seeded selection of 520 positions: 20 for warm-up, then 500 timed.
- **One decision** = score all legal columns of one position: write the inputs, run the model, and have
  the seven logits back in JavaScript.

## What is measured

1. **Warm latency per decision (the headline).** `performance.now()` from just before the inputs are
   written to just after the logits are available in JavaScript (for WebGPU: after the resolved
   `mapAsync` and the copy out of the mapped range). Report the median, p95 and mean over the 500 timed
   decisions, one decision at a time (no overlap).
2. **Cold start.** From the start of loading to the first finished decision, split into: model/plan
   fetch (from a local server, so this is not a network measurement), runtime set-up (pipeline and
   buffer creation, or ORT session creation), and the first decision.
3. **GPU time** (WebGPU only, when `timestamp-query` is available): the timestamps around the compute
   pass, reported separately and never mixed into the headline.
4. **Batched throughput:** 8 and 64 positions per submit, reported as time per batch and per position.
5. **Sizes:** the runtime bundle (minified JavaScript including all shader code; also gzip size) and the
   model download. For onnxruntime-web: the JavaScript and wasm files the page actually loads.

## Correctness before speed

A backend is only timed after it passes a correctness check on the same page: its choices on 50 fixed
eval positions must match the reference (ORT CPU, fp32) exactly for f32, and in at least 48 of 50 for
reduced-precision variants (near-ties can flip). The full 17 325-position parity run is a separate test
and is not part of this protocol.

## Targets for track A1 (agreed before measuring)

- warm latency < 10 ms per decision (median) on an M1 Max in Chrome, onepass-webgpu f32 or f16;
- runtime bundle < 1 MB (minified);
- (argmax parity is gated separately: all 17 325 positions, on the page's real code path).

## Reporting

Each result is a JSON record: backend, precision, model file and its sha256, browser and version,
adapter info, machine, load average, time, the three runs' statistics and their median. Results live in
`bench/results/` and are summarised in the README with the command or page that produced them.
