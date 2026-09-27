// In-browser plan extraction: the same pattern matching as compiler/compile_onepass.py, on the ONNX graph
// itself, so a page can load an unchanged one-pass scorer .onnx file without an offline compile step.
// Anything unrecognised is refused with a message.

import type { Plan, PlanConfig, PlanTensor } from "./index";
import { type Initializer, readInitializers } from "./onnx";

interface Attr {
  name: string;
  i?: number;
  f?: number;
  ints?: number[];
  t?: Initializer;
}

interface Node {
  opType: string;
  name: string;
  inputs: string[];
  outputs: string[];
  attrs: Map<string, Attr>;
}

interface Model {
  nodes: Node[];
  inputs: Map<string, number[]>;
  inits: Map<string, Initializer>;
}

// ---------------------------------------------------------------- protobuf (nodes, attributes, inputs)

class R {
  pos: number;
  constructor(readonly b: Uint8Array, start: number, readonly end: number) {
    this.pos = start;
  }
  /** varint as a signed 64-bit value (exact for |x| < 2^53) */
  varint(): number {
    let lo = 0;
    let shift = 0;
    let b: number;
    // fast path: up to 7 bytes fit a double exactly
    do {
      b = this.b[this.pos++];
      if (shift < 49) lo += (b & 0x7f) * 2 ** shift;
      else return this.bigTail(lo, shift, b);
      shift += 7;
    } while (b & 0x80);
    return lo;
  }
  private bigTail(lo: number, shift: number, first: number): number {
    let v = BigInt(lo) + (BigInt(first & 0x7f) << BigInt(shift));
    let b = first;
    while (b & 0x80) {
      shift += 7;
      b = this.b[this.pos++];
      v += BigInt(b & 0x7f) << BigInt(shift);
    }
    return Number(BigInt.asIntN(64, v));
  }
  str(len: number): string {
    const s = new TextDecoder().decode(this.b.subarray(this.pos, this.pos + len));
    this.pos += len;
    return s;
  }
  skip(wire: number): void {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) {
      const len = this.varint();
      this.pos += len;
    } else if (wire === 5) this.pos += 4;
    else throw new Error(`onnx: unsupported wire type ${wire}`);
  }
}

function each(b: Uint8Array, start: number, end: number, visit: (field: number, wire: number, r: R) => void): void {
  const r = new R(b, start, end);
  while (r.pos < end) {
    const key = r.varint();
    const field = Math.floor(key / 8);
    const wire = key & 7;
    const before = r.pos;
    visit(field, wire, r);
    if (r.pos === before) r.skip(wire);
  }
}

function sub(r: R, visit: (field: number, wire: number, r: R) => void): void {
  const len = r.varint();
  const start = r.pos;
  each(r.b, start, start + len, visit);
  r.pos = start + len;
}

function tensorOf(b: Uint8Array, start: number, end: number): Initializer {
  // reuse the initializer reader on a one-tensor graph view: parse the TensorProto fields directly
  const out: Initializer = { name: "", dims: [], dataType: 0, bytes: new Uint8Array(0) };
  each(b, start, end, (f, w, r) => {
    if (f === 1 && w === 0) out.dims.push(r.varint());
    else if (f === 1 && w === 2) {
      const len = r.varint();
      const stop = r.pos + len;
      while (r.pos < stop) out.dims.push(r.varint());
    } else if (f === 2) out.dataType = r.varint();
    else if (f === 8) out.name = r.str(r.varint());
    else if (f === 9) {
      const len = r.varint();
      out.bytes = b.subarray(r.pos, r.pos + len);
      r.pos += len;
    } else if (f === 7 && w === 2) {           // int64_data, packed
      const len = r.varint();
      const stop = r.pos + len;
      const vals: number[] = [];
      while (r.pos < stop) vals.push(r.varint());
      const arr = new BigInt64Array(vals.map((v) => BigInt(v)));
      out.bytes = new Uint8Array(arr.buffer);
    }
  });
  return out;
}

function readModel(file: Uint8Array): Model {
  const nodes: Node[] = [];
  const inputs = new Map<string, number[]>();
  each(file, 0, file.length, (field, wire, r) => {
    if (field !== 7 || wire !== 2) return;
    sub(r, (gf, gw, g) => {
      if (gf === 1 && gw === 2) {
        const node: Node = { opType: "", name: "", inputs: [], outputs: [], attrs: new Map() };
        sub(g, (nf, _nw, n) => {
          if (nf === 1) node.inputs.push(n.str(n.varint()));
          else if (nf === 2) node.outputs.push(n.str(n.varint()));
          else if (nf === 3) node.name = n.str(n.varint());
          else if (nf === 4) node.opType = n.str(n.varint());
          else if (nf === 5) {
            const attr: Attr = { name: "" };
            sub(n, (af, aw, a) => {
              if (af === 1) attr.name = a.str(a.varint());
              else if (af === 2 && aw === 5) {
                attr.f = new DataView(a.b.buffer, a.b.byteOffset + a.pos, 4).getFloat32(0, true);
                a.pos += 4;
              } else if (af === 3) attr.i = a.varint();
              else if (af === 5 && aw === 2) {
                const len = a.varint();
                attr.t = tensorOf(a.b, a.pos, a.pos + len);
                a.pos += len;
              } else if (af === 8) {
                attr.ints ??= [];
                if (aw === 0) attr.ints.push(a.varint());
                else {
                  const len = a.varint();
                  const stop = a.pos + len;
                  while (a.pos < stop) attr.ints.push(a.varint());
                }
              }
            });
            node.attrs.set(attr.name, attr);
          }
        });
        nodes.push(node);
      } else if (gf === 11 && gw === 2) {
        let name = "";
        const dims: number[] = [];
        sub(g, (vf, _vw, v) => {
          if (vf === 1) name = v.str(v.varint());
          else if (vf === 2) {
            sub(v, (tf, _tw, t) => {            // TypeProto.tensor_type
              if (tf !== 1) return;
              sub(t, (sf, _sw, s) => {          // Tensor.shape
                if (sf !== 2) return;
                sub(s, (df, _dw, d) => {        // TensorShapeProto.dim
                  if (df !== 1) return;
                  let value = -1;
                  sub(d, (xf, _xw, x) => { if (xf === 1) value = x.varint(); });
                  dims.push(value);
                });
              });
            });
          }
        });
        inputs.set(name, dims);
      }
    });
  });
  return { nodes, inputs, inits: readInitializers(file) };
}

// ---------------------------------------------------------------- pattern extraction (mirrors the Python compiler)

class Unrecognised extends Error {}

function scalar(init: Initializer): number {
  const v = new DataView(init.bytes.buffer, init.bytes.byteOffset, init.bytes.byteLength);
  if (init.dataType === 1) return v.getFloat32(0, true);
  if (init.dataType === 3) return v.getInt8(0);
  if (init.dataType === 2) return v.getUint8(0);
  throw new Unrecognised(`${init.name}: unexpected scalar type ${init.dataType}`);
}

function int64s(init: Initializer): number[] {
  if (init.dataType !== 7) return [];
  const a = new BigInt64Array(init.bytes.slice().buffer);
  return Array.from(a, Number);
}

/** Recognise a one-pass scorer export (float, or ONNX Runtime dynamic int8) and build its plan. */
export function planFromOnnx(file: ArrayBuffer | Uint8Array, fileName = "model.onnx"): Plan {
  const bytes = file instanceof Uint8Array ? file : new Uint8Array(file);
  const { nodes, inputs, inits } = readModel(bytes);
  const producer = new Map<string, Node>();
  for (const n of nodes) for (const o of n.outputs) producer.set(o, n);
  const shapeOf = (name: string) => inits.get(name)?.dims ?? [];
  const ctx = inputs.get("context_ids");
  const opt = inputs.get("option_ids");
  if (!ctx || !opt || !inputs.has("option_mask")) throw new Unrecognised("expected inputs context_ids / option_ids / option_mask");
  const [ctxLen, slots, optLen] = [ctx[1], opt[1], opt[2]];

  const quantOf = (name: string): PlanTensor["quant"] | undefined => {
    if (!name.endsWith("_quantized")) return undefined;
    const base = name.slice(0, -"_quantized".length);
    const s = inits.get(`${base}_scale`);
    const z = inits.get(`${base}_zero_point`);
    const q = inits.get(name)!;
    if (!s || !z || s.bytes.byteLength > 4 || z.bytes.byteLength > 1) throw new Unrecognised(`${name}: only per-tensor quantization`);
    if (q.dataType !== 2 && q.dataType !== 3) throw new Unrecognised(`${name}: unsupported quantized type`);
    return { dtype: q.dataType === 3 ? "int8" : "uint8", scale: scalar(s), zero_point: scalar(z) };
  };

  type Linear = { ref: [string, boolean]; bias: string | null; node: Node };
  const linears: Linear[] = [];
  for (const n of nodes) {
    if (!inits.has(n.inputs[1] ?? "")) continue;
    if (n.opType === "MatMul" || n.opType === "MatMulInteger") {
      if (n.opType === "MatMulInteger") quantOf(n.inputs[1]);
      linears.push({ ref: [n.inputs[1], false], bias: null, node: n });
    } else if (n.opType === "Gemm") {
      const a = (k: string, d: number) => n.attrs.get(k)?.i ?? n.attrs.get(k)?.f ?? d;
      if (a("transA", 0) || a("alpha", 1) !== 1 || a("beta", 1) !== 1) throw new Unrecognised("unsupported Gemm attributes");
      linears.push({ ref: [n.inputs[1], !!a("transB", 0)], bias: inits.has(n.inputs[2] ?? "") ? n.inputs[2] : null, node: n });
    }
  }
  const embName = ["model.embedding.weight", "model.embedding.weight_quantized"].find((n) => inits.has(n));
  if (!embName) throw new Unrecognised("no model.embedding.weight");
  const [vocab, width] = shapeOf(embName);

  const layerPrefixes = (root: string) => {
    const found = new Set<number>();
    for (const n of inits.keys()) {
      const m = n.match(new RegExp(`^model\\.${root}\\.layers\\.(\\d+)\\.`));
      if (m) found.add(Number(m[1]));
    }
    return [...found].sort((x, y) => x - y).map((i) => `model.${root}.layers.${i}`);
  };
  const ctxLayers = layerPrefixes("encoder");
  const optLayers = layerPrefixes("option_encoder");
  if (optLayers.length !== 1) throw new Unrecognised(`expected one option-encoder layer, found ${optLayers.length}`);
  const perLayer = 4;
  if (linears.length !== perLayer * (ctxLayers.length + 1) + 3) {
    throw new Unrecognised(`expected ${perLayer * (ctxLayers.length + 1) + 3} weight matmuls, found ${linears.length}`);
  }

  const tensors: Record<string, PlanTensor> = {};
  const shape = ([name, t]: [string, boolean]) => (t ? [...shapeOf(name)].reverse() : shapeOf(name));
  const put = (key: string, ref: [string, boolean]) => {
    const t: PlanTensor = { initializer: ref[0], transpose: ref[1], shape: shape(ref) };
    const q = quantOf(ref[0]);
    if (q) t.quant = q;
    tensors[key] = t;
  };
  put("embedding", [embName, false]);
  const pos = new Map<number, string>();
  for (const [n, t] of inits) if (n.startsWith("onnx::Add") && t.dims.length === 2 && t.dims[1] === width) pos.set(t.dims[0], n);
  if (!pos.has(ctxLen) || !pos.has(optLen)) throw new Unrecognised(`positional rows for lengths ${ctxLen}/${optLen} not found`);
  put("pos_context", [pos.get(ctxLen)!, false]);
  put("pos_option", [pos.get(optLen)!, false]);

  const named = linears.every((l) => l.node.name.startsWith("/model/"));
  const used = new Set<Linear>();
  const layerLinears = (prefix: string, index: number): Linear[] => {
    if (!named) return linears.slice(index * perLayer, (index + 1) * perLayer);
    const [root, sub, layers, i] = prefix.split(".");
    const mine = linears.filter((l) => l.node.name.startsWith(`/${root}/${sub}/${layers}.${i}/`));
    if (mine.length !== perLayer) throw new Unrecognised(`${prefix}: expected ${perLayer} weight matmuls, found ${mine.length}`);
    mine.forEach((l) => used.add(l));
    return mine;
  };
  let ff = 0;
  [...ctxLayers, ...optLayers].forEach((prefix, index) => {
    const name = index < ctxLayers.length ? `layer${index}` : "option_layer";
    const [qkv, out, ff1, ff2] = layerLinears(prefix, index);
    const checks: [number[], number[]][] = [[shape(qkv.ref), [width, 3 * width]], [shape(out.ref), [width, width]]];
    for (const [got, want] of checks) if (got.join() !== want.join()) throw new Unrecognised(`${prefix}: weight shape ${got}, wanted ${want}`);
    if (shape(ff1.ref)[0] !== width || shape(ff2.ref)[1] !== width) throw new Unrecognised(`${prefix}: MLP shapes`);
    ff = shape(ff1.ref)[1];
    put(`${name}.qkv.w`, qkv.ref);
    put(`${name}.qkv.b`, [`${prefix}.self_attn.in_proj_bias`, false]);
    put(`${name}.out.w`, out.ref);
    put(`${name}.out.b`, [out.bias ?? `${prefix}.self_attn.out_proj.bias`, false]);
    put(`${name}.ff1.w`, ff1.ref);
    put(`${name}.ff1.b`, [`${prefix}.linear1.bias`, false]);
    put(`${name}.ff2.w`, ff2.ref);
    put(`${name}.ff2.b`, [`${prefix}.linear2.bias`, false]);
    for (const norm of ["norm1", "norm2"]) {
      put(`${name}.${norm}.w`, [`${prefix}.${norm}.weight`, false]);
      put(`${name}.${norm}.b`, [`${prefix}.${norm}.bias`, false]);
    }
  });

  const fedBy = (n: Node): string => {
    let src = producer.get(n.inputs[0]);
    while (src && src.opType !== "LayerNormalization") src = producer.get(src.inputs[0]);
    if (!src) throw new Unrecognised("head matmul not fed by a LayerNormalization");
    return src.inputs[1];
  };
  const head = named ? linears.filter((l) => !used.has(l)) : linears.slice(-3);
  if (head.length !== 3) throw new Unrecognised(`expected three head matmuls, found ${head.length}`);
  const roles = head.map((l) => fedBy(l.node));
  const query = head.filter((_, i) => roles[i] === "model.head.option_norm.weight");
  const keyval = head.filter((_, i) => roles[i] === "model.head.context_norm.weight");
  if (query.length !== 1 || keyval.length !== 2) throw new Unrecognised(`head roles not recognised: ${roles}`);
  put("head.q.w", query[0].ref);
  put("head.k.w", keyval[0].ref);
  put("head.v.w", keyval[1].ref);
  for (const norm of ["context_norm", "option_norm"]) {
    put(`head.${norm}.w`, [`model.head.${norm}.weight`, false]);
    put(`head.${norm}.b`, [`model.head.${norm}.bias`, false]);
  }
  const rank = tensors["head.q.w"].shape[1];

  let heads = 0;
  for (const n of nodes) {
    if (n.opType !== "Reshape") continue;
    const init = inits.get(n.inputs[1]);
    const src = producer.get(n.inputs[1]);
    const values = init ? int64s(init) : src?.opType === "Constant" && src.attrs.get("value")?.t ? int64s(src.attrs.get("value")!.t!) : [];
    if (values.length === 3 && values[1] > 0 && values[2] > 0 && values[1] * values[2] === width && values[2] < width) {
      heads = values[1];
      break;
    }
  }
  if (!heads) throw new Unrecognised("could not find the number of attention heads");

  for (const [name, t] of Object.entries(tensors)) {
    const init = inits.get(t.initializer);
    if (!init) throw new Unrecognised(`${name}: missing initializer ${t.initializer}`);
    if (init.dataType !== 1 && !t.quant) throw new Unrecognised(`${name}: ${t.initializer} is not float32`);
  }
  const config: PlanConfig = { vocab, width, heads, layers: ctxLayers.length, ff, rank, context_len: ctxLen,
    option_slots: slots, option_len: optLen, eps: 1e-5 };
  return { format: "onepass-plan/1", architecture: "onepass-scorer", config,
    model: { file: fileName, bytes: bytes.byteLength, sha256: "" }, tensors };
}
