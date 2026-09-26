"""Reference logits for the runtime parity test: ONNX Runtime CPU, fp32, on the Connect Four eval set.

    python tests/make_reference.py model.onnx board.npy --out work/c4-v2/reference

Writes boards.bin (uint8 [N, 42]: bottom row first, 0 empty, 1 = to move, 2 = opponent),
logits.bin (float32 [N, 7], slot order = legal columns left to right), and reference.json.
The encoding is the demo's: context "<side>:<42 cells from .mt>", options "column k" for the legal
columns, packed into the first slots.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

CELLS = np.frombuffer(b".mt", dtype=np.uint8)


def encode(boards):
    n = len(boards)
    first = (boards > 0).sum(1) % 2 == 0
    ctx = np.empty((n, 44), dtype=np.int32)
    ctx[:, 0] = np.where(first, ord("1"), ord("2")) + 1
    ctx[:, 1] = ord(":") + 1
    ctx[:, 2:] = CELLS[boards].astype(np.int32) + 1
    opts = np.zeros((n, 7, 8), dtype=np.int32)
    mask = np.zeros((n, 7), dtype=np.int32)
    for i, b in enumerate(boards):
        for slot, col in enumerate(np.flatnonzero(b.reshape(6, 7)[5] == 0)):
            opts[i, slot] = np.frombuffer(f"column {col + 1}".encode(), dtype=np.uint8) + 1
            mask[i, slot] = 1
    return ctx, opts, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("onnx", type=Path)
    p.add_argument("boards", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    boards = np.load(a.boards).astype(np.uint8)
    ctx, opts, mask = encode(boards)
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    logits = np.concatenate([sess.run(None, {"context_ids": ctx[i:i + 1], "option_ids": opts[i:i + 1],
                                             "option_mask": mask[i:i + 1]})[0] for i in range(len(boards))])
    logits = logits.astype(np.float32)
    legal = mask != 0
    masked = np.where(legal, logits, -np.inf)
    order = np.sort(masked, 1)
    gap = order[:, -1] - order[:, -2]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "boards.bin").write_bytes(boards.tobytes())
    (a.out / "logits.bin").write_bytes(logits.tobytes())
    meta = {"positions": int(len(boards)), "onnx_sha256": hashlib.sha256(a.onnx.read_bytes()).hexdigest(),
            "argmax": masked.argmax(1).tolist(), "min_top2_gap": float(np.nanmin(np.where(np.isfinite(gap), gap, np.inf))),
            "gaps_below_1e-4": int((gap < 1e-4).sum()), "gaps_below_1e-3": int((gap < 1e-3).sum())}
    (a.out / "reference.json").write_text(json.dumps(meta))
    print({k: v for k, v in meta.items() if k != "argmax"})


if __name__ == "__main__":
    main()
