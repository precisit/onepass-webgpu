"""Reference logits from the compiler's numpy model of a plan (the runtime's exact semantics, float64-free
f32 numpy), for plans whose ONNX Runtime result is not the target, e.g. weight-only int8.

    python tests/make_plan_reference.py model-int8.onnx board.npy --out work/c4-v2-int8/reference \
        [--fp32-reference work/c4-v2/reference]

Writes boards.bin, logits.bin and reference.json like make_reference.py, plus the argmax of ONNX Runtime
on the same file (for information) and agreement with an fp32 reference when given.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compiler"))
from compile_onepass import extract, materialise, reference  # noqa: E402
from make_reference import encode  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("onnx", type=Path)
    p.add_argument("boards", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fp32-reference", type=Path)
    p.add_argument("--probes", type=Path, help="selftest JSON (rows with 'board'): also write expected logits "
                                               "for those positions to <out>/probes.json, for a page's runtime check")
    a = p.parse_args()
    model = onnx.load(str(a.onnx))
    cfg, refs = extract(model)
    tensors = materialise(model, refs)
    boards = np.load(a.boards).astype(np.uint8)
    ctx, opts, mask = encode(boards)
    logits = np.concatenate([reference(cfg, tensors, ctx[i:i + 512], opts[i:i + 512], mask[i:i + 512])
                             for i in range(0, len(boards), 512)]).astype(np.float32)
    legal = mask != 0
    arg = np.where(legal, logits, -np.inf).argmax(1)
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    ort_logits = np.concatenate([sess.run(None, {"context_ids": ctx[i:i + 1], "option_ids": opts[i:i + 1],
                                                 "option_mask": mask[i:i + 1]})[0] for i in range(len(boards))])
    ort_arg = np.where(legal, ort_logits, -np.inf).argmax(1)
    order = np.sort(np.where(legal, logits, -np.inf), 1)
    gap = order[:, -1] - order[:, -2]
    meta = {"positions": int(len(boards)), "onnx_sha256": hashlib.sha256(a.onnx.read_bytes()).hexdigest(),
            "semantics": "numpy model of the plan: weights dequantized, f32 math (weight-only)",
            "argmax": arg.tolist(), "gaps_below_1e-3": int((np.where(np.isfinite(gap), gap, np.inf) < 1e-3).sum()),
            "vs_ort_same_file": {"same_column": int((arg == ort_arg).sum()),
                                 "max_abs_logit_diff": float(np.abs(np.where(legal, logits - ort_logits, 0)).max())}}
    if a.fp32_reference:
        fp = json.loads((a.fp32_reference / "reference.json").read_text())["argmax"]
        fp = np.array(fp)
        meta["vs_fp32"] = {"plan_same_column": int((arg == fp).sum()), "ort_same_file_same_column": int((ort_arg == fp).sum())}
    a.out.mkdir(parents=True, exist_ok=True)
    if a.probes:
        rows = json.loads(a.probes.read_text())
        pb = np.array([r["board"] for r in rows], dtype=np.uint8)
        pc, po, pm = encode(pb)
        pl = reference(cfg, tensors, pc, po, pm)
        probes = {"onnx_sha256": meta["onnx_sha256"], "semantics": meta["semantics"],
                  "rows": [{"name": r.get("name"), "board": r["board"],
                            "logits": [round(float(x), 6) for x in pl[i][pm[i] != 0]]} for i, r in enumerate(rows)]}
        (a.out / "probes.json").write_text(json.dumps(probes) + "\n")
    (a.out / "boards.bin").write_bytes(boards.tobytes())
    (a.out / "logits.bin").write_bytes(logits.tobytes())
    (a.out / "reference.json").write_text(json.dumps(meta))
    print({k: v for k, v in meta.items() if k != "argmax"})


if __name__ == "__main__":
    main()
