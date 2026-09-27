"""Compile a one-pass scorer ONNX file into an onepass-webgpu plan (plan.json).

The one-pass scorer (context bytes + option strings in, one score per option out) is a small
transformer: byte embedding + positions, N pre-norm encoder layers over the context, a one-layer
option encoder with mean pooling, and an attention scoring head. PyTorch's ONNX export spreads that
over ~1 000 nodes, most of them shape plumbing. This compiler recognises the layers, pulls out their
weights, checks the result against ONNX Runtime, and writes a small plan the WebGPU runtime executes
with a handful of kernels. The plan names the ONNX initializers; the weights stay in the ONNX file.

    python compile_onepass.py model.onnx --out work/c4-v2 [--check 200]

Refuses (raises) on any graph whose structure it does not recognise. The check runs a numpy
reference of the plan on real inputs and compares its logits with ONNX Runtime on the same file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

class Unrecognised(ValueError):
    pass


def attrs(node) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def extract(model: onnx.ModelProto) -> tuple[dict, dict]:
    g = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    producer = {out: n for n in g.node for out in n.output}
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in g.input}
    try:
        ctx_len = shapes["context_ids"][1]
        _, opt_slots, opt_len = shapes["option_ids"]
    except KeyError as exc:
        raise Unrecognised(f"expected inputs context_ids / option_ids / option_mask: {exc}")

    # every weight matmul, in execution order, as (effective [in, out] matrix, bias or None, node)
    linears = []
    for node in g.node:
        if node.op_type == "MatMul" and node.input[1] in inits:
            linears.append(((node.input[1], False), None, node))
        elif node.op_type == "Gemm" and node.input[1] in inits:
            a = attrs(node)
            if a.get("transA", 0) or a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0:
                raise Unrecognised(f"unsupported Gemm attributes {a}")
            bias = node.input[2] if len(node.input) > 2 and node.input[2] in inits else None
            linears.append(((node.input[1], bool(a.get("transB", 0))), bias, node))
        elif node.op_type == "MatMulInteger" and node.input[1] in inits:
            # ONNX Runtime dynamic quantization: int8 weights [K, N] with a per-tensor scale; the
            # activations are quantized on the fly, which this runtime does not copy (weights only)
            quant_info(inits, node.input[1])
            linears.append(((node.input[1], False), None, node))
    emb_name = next((n for n in ("model.embedding.weight", "model.embedding.weight_quantized") if n in inits), None)
    if emb_name is None:
        raise Unrecognised("no model.embedding.weight")
    vocab, width = inits[emb_name].shape

    def layer_prefixes(root: str) -> list[str]:
        found = sorted({int(n.split(".")[3]) for n in inits if n.startswith(f"model.{root}.layers.")})
        return [f"model.{root}.layers.{i}" for i in found]

    ctx_layers, opt_layers = layer_prefixes("encoder"), layer_prefixes("option_encoder")
    if len(opt_layers) != 1:
        raise Unrecognised(f"expected one option-encoder layer, found {len(opt_layers)}")
    per_layer = 4  # qkv, out, ff1, ff2
    expected = per_layer * (len(ctx_layers) + 1) + 3
    if len(linears) != expected:
        raise Unrecognised(f"expected {expected} weight matmuls, found {len(linears)}")

    # every tensor is a reference to an ONNX initializer: (name, transpose to [in, out])
    tensors: dict[str, tuple[str, bool]] = {"embedding": (emb_name, False)}

    def shape(ref):
        a = inits[ref[0]]
        return a.T.shape if ref[1] else a.shape
    # positional rows: the exporter constant-folds position[:L] into Add initializers of shape [L, width]
    pos = {a.shape[0]: n for n, a in inits.items() if a.ndim == 2 and a.shape[1] == width and n.startswith("onnx::Add")}
    if ctx_len not in pos or opt_len not in pos:
        raise Unrecognised(f"positional rows for lengths {ctx_len}/{opt_len} not found: {sorted(pos)}")
    tensors["pos_context"] = (pos[ctx_len], False)
    tensors["pos_option"] = (pos[opt_len], False)

    # assign matmuls to layers by the module path in the node names ("/model/encoder/layers.0/..."), since a
    # quantized export may interleave layers; fall back to execution order for unnamed nodes
    named = all(l[2].name.startswith("/model/") for l in linears)
    used: set[int] = set()

    def layer_linears(prefix: str, index: int) -> list:
        if not named:
            return linears[index * per_layer:(index + 1) * per_layer]
        root, sub, layers, i = prefix.split(".")
        mine = [l for l in linears if l[2].name.startswith(f"/{root}/{sub}/{layers}.{i}/")]
        if len(mine) != per_layer:
            raise Unrecognised(f"{prefix}: expected {per_layer} weight matmuls, found {len(mine)}")
        used.update(id(l) for l in mine)
        return mine

    ff = None
    for index, prefix in enumerate(ctx_layers + opt_layers):
        name = f"layer{index}" if index < len(ctx_layers) else "option_layer"
        qkv, out, ff1, ff2 = layer_linears(prefix, index)
        checks = [(shape(qkv[0]), (width, 3 * width)), (shape(out[0]), (width, width)),
                  (shape(ff1[0])[0], width), (shape(ff2[0])[1], width)]
        for got, want in checks:
            if got != want:
                raise Unrecognised(f"{prefix}: unexpected weight shape {got}, wanted {want}")
        ff = shape(ff1[0])[1]
        tensors[f"{name}.qkv.w"] = qkv[0]
        tensors[f"{name}.qkv.b"] = (f"{prefix}.self_attn.in_proj_bias", False)
        tensors[f"{name}.out.w"] = out[0]
        tensors[f"{name}.out.b"] = (out[1] or f"{prefix}.self_attn.out_proj.bias", False)
        tensors[f"{name}.ff1.w"] = ff1[0]
        tensors[f"{name}.ff1.b"] = (f"{prefix}.linear1.bias", False)
        tensors[f"{name}.ff2.w"] = ff2[0]
        tensors[f"{name}.ff2.b"] = (f"{prefix}.linear2.bias", False)
        for norm in ("norm1", "norm2"):
            tensors[f"{name}.{norm}.w"] = (f"{prefix}.{norm}.weight", False)
            tensors[f"{name}.{norm}.b"] = (f"{prefix}.{norm}.bias", False)

    # scoring head: tell the query (fed by option_norm) from key/value (fed by context_norm)
    def fed_by(node) -> str:
        src = producer.get(node.input[0])
        while src is not None and src.op_type != "LayerNormalization":
            src = producer.get(src.input[0])
        if src is None:
            raise Unrecognised("head matmul not fed by a LayerNormalization")
        return src.input[1]

    head = [l for l in linears if id(l) not in used] if named else linears[-3:]
    if len(head) != 3:
        raise Unrecognised(f"expected three head matmuls, found {len(head)}")
    roles = [fed_by(n) for _, _, n in head]
    query = [h for h, r in zip(head, roles) if r == "model.head.option_norm.weight"]
    keyval = [h for h, r in zip(head, roles) if r == "model.head.context_norm.weight"]
    if len(query) != 1 or len(keyval) != 2:
        raise Unrecognised(f"head roles not recognised: {roles}")
    tensors["head.q.w"], tensors["head.k.w"], tensors["head.v.w"] = query[0][0], keyval[0][0], keyval[1][0]
    for norm in ("context_norm", "option_norm"):
        tensors[f"head.{norm}.w"] = (f"model.head.{norm}.weight", False)
        tensors[f"head.{norm}.b"] = (f"model.head.{norm}.bias", False)
    rank = shape(tensors["head.q.w"])[1]

    heads = None
    for node in g.node:  # number of attention heads from the reshape that splits q/k/v: [L, heads, head_dim]
        if node.op_type != "Reshape":
            continue
        src = producer.get(node.input[1])
        if node.input[1] in inits:
            s = inits[node.input[1]]
        elif src is not None and src.op_type == "Constant" and src.attribute and src.attribute[0].name == "value":
            s = numpy_helper.to_array(src.attribute[0].t)
        else:
            continue
        if s.ndim == 1 and len(s) == 3 and s[1] > 0 and s[2] > 0 and s[1] * s[2] == width and s[2] < width:
            heads = int(s[1])
            break
    # heads = None: the export computes its reshape shapes at run time; main() tries the candidates and keeps
    # the one ONNX Runtime confirms
    config = {"vocab": int(vocab), "width": int(width), "heads": int(heads) if heads else None, "layers": len(ctx_layers),
              "ff": int(ff), "rank": int(rank), "context_len": int(ctx_len), "option_slots": int(opt_slots),
              "option_len": int(opt_len), "eps": 1e-5, "activation": "relu", "norm_first": True}
    for name, ref in tensors.items():
        if inits[ref[0]].dtype != np.float32 and quant_info(inits, ref[0]) is None:
            raise Unrecognised(f"{name}: initializer {ref[0]} is {inits[ref[0]].dtype}, only float32 is supported")
    return config, tensors


def quant_info(inits: dict, name: str) -> dict | None:
    """Per-tensor quantization of `<base>_quantized` (ONNX Runtime's naming): w = (q - zero_point) * scale."""
    if not name.endswith("_quantized"):
        return None
    base = name[: -len("_quantized")]
    scale, zp = inits.get(base + "_scale"), inits.get(base + "_zero_point")
    if scale is None or zp is None or scale.size != 1 or zp.size != 1:
        raise Unrecognised(f"{name}: only per-tensor scale and zero point are supported")
    if inits[name].dtype not in (np.int8, np.uint8):
        raise Unrecognised(f"{name}: {inits[name].dtype} weights are not supported")
    return {"dtype": inits[name].dtype.name, "scale": float(scale), "zero_point": int(zp)}


def materialise(model: onnx.ModelProto, refs: dict) -> dict[str, np.ndarray]:
    """Float values of every plan tensor (quantized ones dequantized: the runtime's semantics)."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    out = {}
    for k, (n, t) in refs.items():
        a, q = inits[n], quant_info(inits, n)
        if q:
            a = (a.astype(np.float32) - np.float32(q["zero_point"])) * np.float32(q["scale"])
        out[k] = np.ascontiguousarray(a.T if t else a, dtype=np.float32)
    return out


# ---------------------------------------------------------------- numpy reference of the plan

def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def encoder_layer(x, key_mask, t, name, cfg):
    # x [S, L, W]; key_mask [S, L] True = attend
    s, l, w = x.shape
    h, d = cfg["heads"], w // cfg["heads"]
    y = layer_norm(x, t[f"{name}.norm1.w"], t[f"{name}.norm1.b"], cfg["eps"])
    qkv = y @ t[f"{name}.qkv.w"] + t[f"{name}.qkv.b"]
    q, k, v = np.split(qkv, 3, axis=-1)
    q = q.reshape(s, l, h, d).transpose(0, 2, 1, 3)
    k = k.reshape(s, l, h, d).transpose(0, 2, 1, 3)
    v = v.reshape(s, l, h, d).transpose(0, 2, 1, 3)
    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d)
    att = np.where(key_mask[:, None, None, :], att, -np.inf)
    att = np.exp(att - att.max(-1, keepdims=True))
    att = att / att.sum(-1, keepdims=True)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(s, l, w)
    x = x + o @ t[f"{name}.out.w"] + t[f"{name}.out.b"]
    y = layer_norm(x, t[f"{name}.norm2.w"], t[f"{name}.norm2.b"], cfg["eps"])
    y = np.maximum(y @ t[f"{name}.ff1.w"] + t[f"{name}.ff1.b"], 0)
    return x + y @ t[f"{name}.ff2.w"] + t[f"{name}.ff2.b"]


def reference(cfg, t, context_ids, option_ids, option_mask):
    """context_ids [B, Lc], option_ids [B, O, Lo], option_mask [B, O] (int) -> logits [B, O]."""
    b = context_ids.shape[0]
    ctx_mask = context_ids != 0
    safe = ctx_mask.copy()
    safe[:, 0] = True  # the toolkit never masks position 0 (avoids an all-masked row)
    x = t["embedding"][context_ids] + t["pos_context"][None]
    for i in range(cfg["layers"]):
        x = encoder_layer(x, safe, t, f"layer{i}", cfg)
    o = option_ids.reshape(b * cfg["option_slots"], cfg["option_len"])
    tok = o != 0
    safe_o = tok.copy()
    safe_o[:, 0] = True
    y = t["embedding"][o] + t["pos_option"][None]
    y = encoder_layer(y, safe_o, t, "option_layer", cfg)
    wts = tok[..., None].astype(np.float32)
    pooled = (y * wts).sum(1) / np.maximum(wts.sum(1), 1)
    pooled = pooled.reshape(b, cfg["option_slots"], cfg["width"])
    c = layer_norm(x, t["head.context_norm.w"], t["head.context_norm.b"], cfg["eps"])
    p = layer_norm(pooled, t["head.option_norm.w"], t["head.option_norm.b"], cfg["eps"])
    q, k, v = p @ t["head.q.w"], c @ t["head.k.w"], c @ t["head.v.w"]
    r = cfg["rank"]
    scores = np.einsum("bnr,blr->bnl", q, k) / np.sqrt(r)
    scores = np.where(ctx_mask[:, None, :], scores, -np.inf)
    scores = np.exp(scores - scores.max(-1, keepdims=True))
    scores = scores / scores.sum(-1, keepdims=True)
    attended = np.einsum("bnl,blr->bnr", scores, v)
    logits = (q * attended).sum(-1) / np.sqrt(r)
    return np.where(option_mask != 0, logits, np.finfo(np.float32).min)


# ---------------------------------------------------------------- output

def write_plan(out: Path, cfg: dict, refs: dict, arrays: dict, source: Path, check: dict) -> dict:
    """plan.json: the architecture config plus, for every tensor, the ONNX initializer it comes from.
    The runtime reads the weights straight out of the unchanged ONNX file."""
    out.mkdir(parents=True, exist_ok=True)
    inits = {t.name: numpy_helper.to_array(t) for t in onnx.load(str(source)).graph.initializer}
    plan = {"format": "onepass-plan/1", "architecture": "onepass-scorer", "config": cfg,
            "model": {"file": source.name, "bytes": source.stat().st_size,
                      "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            "inputs": {"context_ids": [cfg["context_len"]], "option_ids": [cfg["option_slots"], cfg["option_len"]],
                       "option_mask": [cfg["option_slots"]]},
            "tensors": {k: {"initializer": n, "transpose": t, "shape": list(arrays[k].shape),
                            **({"quant": q} if (q := quant_info(inits, n)) else {})} for k, (n, t) in refs.items()},
            "compile_check": check}
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=int, default=200, help="random inputs to compare with ONNX Runtime")
    parser.add_argument("--inputs", type=Path, help="optional .npz with context_ids/option_ids/option_mask")
    args = parser.parse_args()

    model = onnx.load(str(args.onnx))
    cfg, refs = extract(model)
    tensors = materialise(model, refs)
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    if args.inputs:
        data = np.load(args.inputs)
        ctx, opt, mask = data["context_ids"][: args.check], data["option_ids"][: args.check], data["option_mask"][: args.check]
    else:  # printable-byte noise with a random number of legal options
        rng = np.random.default_rng(0)
        ctx = rng.integers(33, 127, size=(args.check, cfg["context_len"]), dtype=np.int32)
        opt = rng.integers(33, 127, size=(args.check, cfg["option_slots"], cfg["option_len"]), dtype=np.int32)
        mask = (np.arange(cfg["option_slots"])[None] < rng.integers(2, cfg["option_slots"] + 1, size=(args.check, 1))).astype(np.int32)
        opt = opt * mask[..., None]
    if cfg["heads"] is None:
        def trial(h):
            c = {**cfg, "heads": h}
            err = 0.0
            for i in range(min(10, len(ctx))):
                want = session.run(None, {"context_ids": ctx[i:i + 1], "option_ids": opt[i:i + 1], "option_mask": mask[i:i + 1]})[0][0]
                got = reference(c, tensors, ctx[i:i + 1], opt[i:i + 1], mask[i:i + 1])[0]
                err = max(err, float(np.abs(want - got)[mask[i] != 0].max()))
            return err
        candidates = [h for h in (1, 2, 4, 8, 16, 32) if cfg["width"] % h == 0 and cfg["width"] // h >= 8]
        errors = {h: trial(h) for h in candidates}
        cfg["heads"] = min(errors, key=errors.get)
        print(json.dumps({"heads_tried": errors, "heads": cfg["heads"]}))
    worst, agree = 0.0, 0
    for i in range(len(ctx)):
        want = session.run(None, {"context_ids": ctx[i:i + 1], "option_ids": opt[i:i + 1], "option_mask": mask[i:i + 1]})[0][0]
        got = reference(cfg, tensors, ctx[i:i + 1], opt[i:i + 1], mask[i:i + 1])[0]
        legal = mask[i] != 0
        worst = max(worst, float(np.abs(want[legal] - got[legal]).max()))
        agree += int(np.argmax(np.where(legal, want, -np.inf)) == np.argmax(np.where(legal, got, -np.inf)))
    quantized = any(n.endswith("_quantized") for n, _ in refs.values())
    report = {"config": cfg, "checked": int(len(ctx)), "max_abs_logit_diff": worst, "argmax_agree": agree,
              "weights": "quantized, dequantized (weight-only); ORT also quantizes activations" if quantized else "float"}
    print(json.dumps(report))
    # a quantized file cannot match exactly (ORT quantizes the activations too); a wrong mapping shows up
    # as chance-level agreement, so that is what the looser check catches
    if (worst > 1e-3 or agree != len(ctx)) if not quantized else (worst > 1.0 or agree < 0.9 * len(ctx)):
        raise SystemExit("the compiled plan does not reproduce ONNX Runtime; refusing to write it")
    write_plan(args.out, cfg, refs, tensors, args.onnx, {k: v for k, v in report.items() if k != "config"})
    print(f"wrote {args.out / 'plan.json'}")


if __name__ == "__main__":
    main()
