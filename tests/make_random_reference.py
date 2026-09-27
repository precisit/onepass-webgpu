"""Reference logits from ONNX Runtime (CPU) on recorded random inputs, for models whose input encoding the tests
do not rebuild (e.g. the v1 Connect Four file). Contexts are printable bytes of random length followed by
padding; a random number of options, each of random length, followed by padding.

    python tests/make_random_reference.py model.onnx --out <dir> [--n 2000]

Writes inputs.bin (int32: per position the context ids, option ids and option mask), logits.bin and
reference.json (with "inputs": "inputs.bin" and the argmax).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("onnx", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    dims = {i.name: i.shape for i in sess.get_inputs()}
    lc = dims["context_ids"][1]
    slots, lo = dims["option_ids"][1], dims["option_ids"][2]
    rng = np.random.default_rng(a.seed)
    rows, logits = [], []
    for _ in range(a.n):
        ctx = np.zeros((1, lc), np.int32)
        n = int(rng.integers(max(1, lc // 3), lc + 1))
        ctx[0, :n] = rng.integers(33, 127, n) + 1
        opt = np.zeros((1, slots, lo), np.int32)
        mask = np.zeros((1, slots), np.int32)
        for s in range(int(rng.integers(1, slots + 1))):
            m = int(rng.integers(1, lo + 1))
            opt[0, s, :m] = rng.integers(33, 127, m) + 1
            mask[0, s] = 1
        out = sess.run(None, {"context_ids": ctx, "option_ids": opt, "option_mask": mask})[0][0]
        rows.append(np.concatenate([ctx.ravel(), opt.ravel(), mask.ravel()]))
        logits.append(out.astype(np.float32))
    inputs = np.stack(rows).astype(np.int32)
    logits = np.stack(logits)
    masks = inputs[:, lc + slots * lo:]
    arg = np.where(masks != 0, logits, -np.inf).argmax(1)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "inputs.bin").write_bytes(inputs.tobytes())
    (a.out / "logits.bin").write_bytes(logits.tobytes())
    meta = {"positions": a.n, "inputs": "inputs.bin", "context_len": lc, "option_slots": slots, "option_len": lo,
            "onnx_sha256": hashlib.sha256(a.onnx.read_bytes()).hexdigest(), "argmax": arg.tolist()}
    (a.out / "reference.json").write_text(json.dumps(meta))
    print({k: v for k, v in meta.items() if k != "argmax"})


if __name__ == "__main__":
    main()
