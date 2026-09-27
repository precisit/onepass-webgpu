# onepass-webgpu

A small WebGPU-only runtime for **one-pass option scorers**: models that read a context and a list of
options and return one score per option in a single forward pass (for example the
[Connect Four model](https://huggingface.co/precisit/onepass-c4) behind the
[onepass-web demos](https://github.com/precisit/onepass-web)).

Research-grade. TypeScript + WGSL, no WebAssembly, no dependencies at runtime. Chrome and Safari only for now.

## Status

| Milestone | State |
| --- | --- |
| A1: v2 Connect Four model on WebGPU, offline-compiled plan | done: argmax parity on all 17 325 eval positions (f32), 4.1 ms per move, see below |
| Speed protocol (`SPEED-PROTOCOL.md`) | frozen before the first measurement |
| A2: load the ONNX graph in the browser (no offline plan) | done for the one-pass scorer family: `Engine.fromOnnx(bytes)` recognises the layers of the unchanged v1, v2 and v2-int8 files; parity below |
| int8 weights (load the published int8 file directly) | done: weights stay 8-bit on the GPU, unpacked inside the matmul; parity with its reference on all 17 325 positions, see below |
| Plugin API for custom weight formats | first version: `WeightFormat` (pack at load, a WGSL `w4(k, n4)` decode inside the matmul) |

## Results (A1, v2 Connect Four model)

Correctness ([record](tests/results/2026-09-27-parity-c4-v2.json)): on all 17 325 positions of the Connect Four eval set, run through the demo page's own code path in
headless Chrome, the f32 runtime chooses the same column as ONNX Runtime (CPU, fp32) every time; the largest
score difference is 6.6e-5. With f16 weights, 17 322 of 17 325 match (the other three are near ties).

Speed, under [SPEED-PROTOCOL.md](SPEED-PROTOCOL.md) (500 eval positions after 20 warm-up moves, one decision at a
time, three runs, median of the three medians). Apple M1 Max, Chrome 153 (`--headless=new`, Metal adapter), on AC
power; the machine was busy with other jobs (load average 25 to 75 during the runs), so treat the absolute
numbers as provisional; a run on a quiet machine will be added. Both runtimes were measured under the same
conditions. Record:
[`bench/results/2026-09-26-M1Max-chrome.json`](bench/results/2026-09-26-M1Max-chrome.json), runtime commit `e1823f3`.

| backend | model file | median | p95 | set-up + first move | runtime code (gzip) |
| --- | --- | ---: | ---: | ---: | ---: |
| onepass-webgpu, f32 | fp32 ONNX, 29.7 MB | **4.1 ms** | 5.1 ms | 144 ms | **22 KB (7 KB)** |
| onepass-webgpu, f16 weights | the same file | 3.9 ms | 4.4 ms | 127 ms | 22 KB (7 KB) |
| onnxruntime-web 1.30, wasm | int8 ONNX, 7.8 MB | 20.4 ms | 21.4 ms | 398 ms | 14.3 MB (3.7 MB) |
| onnxruntime-web 1.30, wasm | fp32 ONNX, 29.7 MB | 19.5 ms | 21.2 ms | 385 ms | 14.3 MB (3.7 MB) |

- GPU time per move (timestamp queries around the compute pass): 2.8 ms. The rest of the 4.1 ms is submit and read-back.
- Batched (onepass-webgpu only; the ONNX file has a fixed batch of 1): 64 positions per call take 34 ms,
  0.53 ms per position (f32), or 0.36 ms per position with f16 weights.
- onnxruntime-web ran single-threaded, as on a page without cross-origin isolation (for example GitHub Pages).
- The fp32 download (29.7 MB) is the price of A1; the int8 file (7.8 MB) now loads directly, see below.

### int8 file, weights only

`compile_onepass.py` also reads files quantized by ONNX Runtime's dynamic quantization (`MatMulInteger`,
per-tensor int8 weights). The runtime keeps those weights packed, four per 32-bit word, in GPU memory (7.7 MB
instead of 29.5 MB), and each matmul thread unpacks its word to floats in registers; the math stays f32. That is
weight-only int8: onnxruntime-web also rounds the activations to 8 bits before each matmul, so its scores differ
slightly by design.

- Parity ([record](tests/results/2026-09-27-parity-c4-v2-int8.json)): on all 17 325 eval positions, through the demo
  page, the WebGPU int8 path chooses the same column as its reference (the plan's numpy model with dequantized
  weights), largest score difference 1.7e-5.
- Against ONNX Runtime fp32: same column on 17 128 of 17 325 positions. onnxruntime-web's own int8 on the same file:
  17 004. So the weight-only path stays closer to the fp32 model.
- Speed: at batch 1 about the same as f32 (a move is bound by dispatches, not weight bandwidth), a little faster at
  batch 64. The protocol record follows.

### A2: unchanged ONNX files, no offline step

`Engine.fromOnnx(bytes)` reads the graph in the browser (nodes, attributes, initializers), recognises the same
layer patterns as `compiler/compile_onepass.py` and builds the plan itself. For the three published Connect Four
files the in-browser plan is identical to the Python compiler's. Parity through `Engine.fromOnnx`
([record](tests/results/2026-09-27-parity-fromonnx.json)):

| file | positions | same choice | largest score difference |
| --- | ---: | ---: | ---: |
| v1, `onepass-c4-8x24.onnx` (2 x 128, 224-byte context), vs ONNX Runtime | 2 000 recorded random inputs | 2 000 | 1.9e-5 |
| v2 fp32 vs ONNX Runtime fp32 | 17 325 eval positions | 17 325 | 6.6e-5 |
| v2 int8 vs its weight-only reference | 17 325 eval positions | 17 325 | 1.7e-5 |

Scope: this recognises one-pass scorers exported by the toolkit (float or ONNX Runtime dynamic int8), not general
ONNX graphs; anything else is refused with a message. Sequences longer than 64 tokens (v1's 224-byte context) use a
chunked attention kernel with an online softmax.

## Use

```js
import { Engine } from "./onepass-webgpu.js";

const plan = await (await fetch("onepass-c4-v2.plan.json")).json();
const onnx = await (await fetch("onepass-c4-v2.onnx")).arrayBuffer();   // the unchanged fp32 ONNX file
const engine = await Engine.load(plan, onnx, { precision: "f32" });     // or "f16" (needs shader-f16)
const logits = await engine.score(contextIds, optionIds, optionMask);   // Int32Arrays in, Float32Array(7) out
```

The plan is small (a few KB of JSON). It names the ONNX initializer each weight comes from, so the runtime
reads its weights straight out of the same `.onnx` file that onnxruntime-web would load. `score()` also takes
a batch size for several positions per call (`Engine.load(..., { maxBatch: 64 })`).

**Warm up before an interactive decision.** After one to two idle seconds, the GPU and the browser's GPU
process drop into low-power states. The first decision then pays 10 to 100 ms to wake them, even though the
compute itself still takes 4 ms (measured on an M1 Max, Chrome 153, visible window; `tests/cooldown.html`).
A trivial submit does not wake them; real work does. `engine.wake()` re-runs the model on the inputs already
in its buffers. Call it a few hundred ms before a decision you want fast, for example when the user clicks.
The demo does this while the user's disc drops, and then gets about 5 ms per move after any pause.

`Engine.load` refuses a software WebGPU adapter (such as SwiftShader) unless `allowSoftware` is set: numbers
from a software adapter say nothing about a real GPU.

## How it works

- `compiler/compile_onepass.py` recognises the layers of a one-pass scorer export (byte embedding + positions,
  pre-norm encoder layers, a one-layer option encoder with mean pooling, an attention scoring head), maps each
  weight to its initializer, runs a numpy reference of the plan and refuses to write it unless the result matches
  ONNX Runtime on the same file. Anything it does not recognise is refused.
- `runtime/src/onnx.ts` reads the initializers of an ONNX file (a minimal protobuf reader, about 120 lines).
- `runtime/src/kernels.ts` has six WGSL kernels: embedding + layer norm, split-K matmul (vec4 weights), residual
  add + layer norm, masked multi-head attention, mean pooling + layer norm, and the scoring head.
- A decision is one command encoder, one compute pass with 70 dispatches for the v2 model, one submit, and a
  read-back of the seven logits.
- At batch 1 the matmuls are too small to keep a GPU busy, so each is split along K into partial products;
  the next kernel sums the partials while loading its input. Activations are f32; weights are f32 or f16.

## Reproduce

```sh
npm install && npm run build
python compiler/compile_onepass.py onepass-c4-v2.onnx --out work/c4-v2
python tests/make_reference.py onepass-c4-v2.onnx board.npy --out work/c4-v2/reference   # ORT CPU fp32
python tests/run_page.py "tests/parity.html?precision=f32"                                  # all eval positions
python bench/run_bench.py --data work/c4-v2                                                 # SPEED-PROTOCOL.md
```

`board.npy` is the Connect Four eval set built by the recipe in
[one-pass-specialists/examples/c4](https://github.com/precisit/one-pass-specialists/tree/main/examples/c4).
The Python scripts need `onnx`, `onnxruntime`, `numpy` and `playwright`; the pages run in the installed
Google Chrome (`--headless=new`), which uses the real GPU.

## License

MIT
