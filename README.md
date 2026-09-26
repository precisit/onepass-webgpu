# onepass-webgpu

A small WebGPU-only runtime for **one-pass option scorers**: models that read a context and a list of
options and return one score per option in a single forward pass (for example the
[Connect Four model](https://huggingface.co/precisit/onepass-c4) behind the
[onepass-web demos](https://github.com/precisit/onepass-web)).

Research-grade. TypeScript + WGSL, no WebAssembly, no dependencies at runtime. Chrome and Safari only for now.

## Status

| Milestone | State |
| --- | --- |
| A1: v2 Connect Four model on WebGPU, offline-compiled plan | done: argmax parity on all 17 325 eval positions (f32), see below |
| Speed protocol (`SPEED-PROTOCOL.md`) | frozen before the first measurement |
| A2: load the ONNX graph in the browser (no offline plan) | next |
| int8 weights (load the published int8 file directly) | planned |
| Plugin API for custom weight formats | planned |

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
