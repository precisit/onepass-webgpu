// onepass-webgpu: a small WebGPU runtime for one-pass option scorers.
//
//   const engine = await Engine.load(plan, onnxBytes, { precision: "f16" });
//   const logits = await engine.score(contextIds, optionIds, optionMask);   // Float32Array(slots)
//
// The plan (from compiler/compile_onepass.py) names the ONNX initializers each tensor comes from;
// the weights are read straight out of the unchanged .onnx file.

import * as K from "./kernels";
import { dequantTensor, floatTensor, type Initializer, readInitializers } from "./onnx";

export { floatTensor, halfToFloat, readInitializers, type Initializer } from "./onnx";
export { planFromOnnx } from "./graph";
import { planFromOnnx } from "./graph";

export interface PlanConfig {
  vocab: number;
  width: number;
  heads: number;
  layers: number;
  ff: number;
  rank: number;
  context_len: number;
  option_slots: number;
  option_len: number;
  eps: number;
}

export interface Plan {
  format: string;
  architecture: string;
  config: PlanConfig;
  model: { file: string; bytes: number; sha256: string };
  tensors: Record<string, PlanTensor>;
}

export interface PlanTensor {
  initializer: string;
  transpose: boolean;
  shape: number[];
  /** Per-tensor quantization of an 8-bit initializer: value = (q - zero_point) * scale. */
  quant?: { dtype: "int8" | "uint8"; scale: number; zero_point: number };
  /** Use only rows [start, end) of a 2-D initializer (e.g. the first positions of a position table). */
  rows?: [number, number];
  /** A weight format provided by a plugin (see WeightFormat); `initializer` names its main packed tensor. */
  format?: { kind: string; [key: string]: unknown };
}

/**
 * A plugin weight format for matmul weights ([K, N], x @ W). At load time `pack` turns the tensor into bytes
 * (uploaded as-is into the packed-weight buffer) plus optional float side data (e.g. group scales) and two u32
 * parameters. In the matmul kernel, the format's WGSL must define
 *
 *     fn w4(k: u32, n4: u32) -> vec4<f32>     // W[k, 4 n4 .. 4 n4 + 3]
 *
 * and may use qbyte(i) (byte i of this tensor's packed bytes), fval(i) (float i of its side data) and the
 * uniforms p.K, p.N, p.x0, p.x1 (the two parameters from pack). A format that sets `inner` also defines
 *
 *     fn inner(k0: u32, n4: u32, acc: ptr<function, array<vec4<f32>, RM>>)
 *
 * which runs the whole K-split loop itself: for kk < KS, acc[r] += at[r * KS + kk] * W[k0 + kk, columns], where
 * `at` is the kernel's workgroup tile of activations, RM rows by KS. This lets a format decode each byte or record
 * once instead of once per weight.
 */
export interface WeightFormat {
  kind: string;
  wgsl: string;
  inner?: boolean;
  pack(t: PlanTensor, inits: Map<string, Initializer>): { bytes: Uint8Array; floats?: Float32Array; params?: [number, number] };
}

interface Packed8 {
  offsetWords: number;
  kind: "int8" | "uint8";
  scale: number;
  zeroPoint: number;
  /** plugin formats: the format, byte offset, side-data offset (elements of the float buffer), parameters */
  format?: WeightFormat;
  offsetBytes?: number;
  floatOffset?: number;
  params?: [number, number];
}

export type Precision = "f32" | "f16";

export interface LoadOptions {
  precision?: Precision;
  /** Largest batch (positions per call) the buffers are sized for. Default 1. */
  maxBatch?: number;
  /** Measure GPU time of each call with timestamp queries when the adapter supports it. */
  gpuTiming?: boolean;
  /** Allow a software adapter (for example SwiftShader). Off by default: such results are not real. */
  allowSoftware?: boolean;
  /** Kernel tuning knobs (defaults measured on an M1 Max). */
  tuning?: Partial<Tuning>;
  /** Plugin weight formats, matched by PlanTensor.format.kind. */
  formats?: WeightFormat[];
}

export interface Tuning {
  /** Split matmuls along K until about this many threads run (more threads, more partial sums). */
  splitTarget: number;
  /** The same for the fused q/k/v projection, whose partials attention reads once per block of queries. */
  qkvSplitTarget: number;
}

const DEFAULT_TUNING: Tuning = { splitTarget: 4096, qkvSplitTarget: 4096 };

interface Dispatch {
  label: string;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  groups: [number, number, number];
}

interface Program {
  batch: number;
  dispatches: Dispatch[];
  params: GPUBuffer;
}

const PARAM_STRIDE = 256;

/** How a matmul is split: RM rows per workgroup, S splits along K (enough threads at small M). */
function splits(M: number, K: number, N: number, target: number): { S: number; RM: number } {
  const RM = 4;
  const threads = Math.ceil(N / 256) * 64 * Math.ceil(M / RM);
  let S = 1;
  while (S * 2 <= K / 32 && threads * S < target) S *= 2;
  return { S, RM };
}

/** Floats needed by the two partial buffers (p1: qkv, ff1, head k; p2: out, ff2, head v) and pq. */
function partialSizes(c: PlanConfig, batch: number, t: Tuning): { p1: number; p2: number; pq: number } {
  const size = (M: number, K: number, N: number, target = t.splitTarget) => splits(M, K, N, target).S * M * N;
  const out = { p1: 0, p2: 0, pq: size(batch * c.option_slots, c.width, c.rank) };
  for (const M of [batch * c.context_len, batch * c.option_slots * c.option_len]) {
    out.p1 = Math.max(out.p1, size(M, c.width, 3 * c.width, t.qkvSplitTarget), size(M, c.width, c.ff));
    out.p2 = Math.max(out.p2, size(M, c.width, c.width), size(M, c.ff, c.width));
  }
  const ctx = batch * c.context_len;
  out.p1 = Math.max(out.p1, size(ctx, c.width, c.rank));
  out.p2 = Math.max(out.p2, size(ctx, c.width, c.rank));
  return out;
}

function toHalf(src: Float32Array): Uint16Array {
  const F16 = (globalThis as { Float16Array?: new (a: Float32Array) => ArrayBufferView }).Float16Array;
  if (F16) {
    const h = new F16(src);
    return new Uint16Array(h.buffer, h.byteOffset, src.length);
  }
  // round to nearest even, with subnormals; the weights never overflow f16
  const out = new Uint16Array(src.length);
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  for (let i = 0; i < src.length; i += 1) {
    f[0] = src[i];
    const x = u[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 0xff) { out[i] = sign | 0x7c00 | (mant ? 0x200 : 0); continue; }
    let e = exp - 127 + 15;
    if (e >= 0x1f) { out[i] = sign | 0x7c00; continue; }
    if (e <= 0) {
      if (e < -10) { out[i] = sign; continue; }
      mant |= 0x800000;
      const shift = 14 - e;
      let half = mant >>> shift;
      const rem = mant & ((1 << shift) - 1);
      const mid = 1 << (shift - 1);
      if (rem > mid || (rem === mid && (half & 1))) half += 1;
      out[i] = sign | half;
      continue;
    }
    let half = (e << 10) | (mant >>> 13);
    const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half += 1;
    out[i] = sign | half;
  }
  return out;
}

export class Engine {
  readonly config: PlanConfig;
  /** "f32"/"f16" for float weights, "int8" when the matmul weights are packed 8-bit. */
  get weightFormat(): string {
    if (!this.packed.size) return this.precision;
    const kinds = new Set([...this.packed.values()].map((q) => q.format?.kind ?? q.kind));
    return `${[...kinds].join(" + ")} (${this.precision} for the rest)`;
  }
  /** Bytes of weights held on the GPU. */
  get weightBytes(): number {
    return this.buffers.weights.size + (this.packed.size ? this.buffers.weights8.size : 0);
  }
  lastGpuMs: number | null = null;
  private programs = new Map<number, Program>();
  private pipelines = new Map<string, GPUComputePipeline>();
  private offsets = new Map<string, number>();
  private buffers: Record<string, GPUBuffer> = {};
  private staging: GPUBuffer;
  private querySet: GPUQuerySet | null = null;
  private queryBuffer: GPUBuffer | null = null;
  private busy: Promise<unknown> = Promise.resolve();

  private constructor(
    readonly device: GPUDevice,
    readonly adapterInfo: GPUAdapterInfo,
    readonly precision: Precision,
    readonly maxBatch: number,
    plan: Plan,
    weights: GPUBuffer,
    offsets: Map<string, number>,
    timing: boolean,
    readonly tuning: Tuning,
    weights8: GPUBuffer | null = null,
    private packed = new Map<string, Packed8>(),
  ) {
    this.config = plan.config;
    this.offsets = offsets;
    const c = plan.config;
    const B = maxBatch;
    const ctxRows = B * c.context_len;
    const optRows = B * c.option_slots * c.option_len;
    const rows = Math.max(ctxRows, optRows);
    const f32 = (n: number, usage = 0) =>
      device.createBuffer({ size: Math.max(16, n * 4), usage: GPUBufferUsage.STORAGE | usage });
    const parts = { p1: 0, p2: 0, pq: 0 };
    for (let batch = 1; batch <= B; batch += 1) {
      const need = partialSizes(c, batch, tuning);
      parts.p1 = Math.max(parts.p1, need.p1);
      parts.p2 = Math.max(parts.p2, need.p2);
      parts.pq = Math.max(parts.pq, need.pq);
    }
    this.buffers = {
      weights,
      weights8: weights8 ?? f32(4),
      ids: f32(B * (c.context_len + c.option_slots * c.option_len + c.option_slots), GPUBufferUsage.COPY_DST),
      xc: f32(ctxRows * c.width),
      yc: f32(ctxRows * c.width),
      xo: f32(optRows * c.width),
      yo: f32(optRows * c.width),
      att: f32(rows * c.width),
      p1: f32(parts.p1),
      p2: f32(parts.p2),
      pq: f32(parts.pq),
      pooled: f32(B * c.option_slots * c.width),
      logits: f32(B * c.option_slots, GPUBufferUsage.COPY_SRC),
    };
    this.staging = device.createBuffer({ size: 256 + 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    if (B * c.option_slots * 4 > 256) {
      this.staging.destroy();
      this.staging = device.createBuffer({
        size: Math.ceil((B * c.option_slots * 4) / 256) * 256 + 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
    if (timing) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: 2 });
      this.queryBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
  }

  /** Load an unchanged one-pass scorer ONNX file: recognise its layers in the browser, then load. */
  static async fromOnnx(onnx: ArrayBuffer | Uint8Array, options: LoadOptions = {}): Promise<Engine> {
    return Engine.load(planFromOnnx(onnx), onnx, options);
  }

  /** Load a plan and its ONNX file (bytes) onto the GPU. */
  static async load(plan: Plan, onnx: ArrayBuffer | Uint8Array, options: LoadOptions = {}): Promise<Engine> {
    if (plan.format !== "onepass-plan/1" || plan.architecture !== "onepass-scorer") {
      throw new Error(`unsupported plan ${plan.format} / ${plan.architecture}`);
    }
    if (!navigator.gpu) throw new Error("WebGPU is not available in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const info = adapter.info;
    const software = /swiftshader|llvmpipe|software|basic render/i.test(`${info.vendor} ${info.architecture} ${info.description}`)
      || (adapter as { isFallbackAdapter?: boolean }).isFallbackAdapter === true;
    if (software && !options.allowSoftware) throw new Error(`software WebGPU adapter (${info.vendor} ${info.architecture})`);
    const precision = options.precision ?? "f32";
    if (precision === "f16" && !adapter.features.has("shader-f16")) throw new Error("this GPU has no shader-f16");
    const timing = !!options.gpuTiming && adapter.features.has("timestamp-query");
    const features: GPUFeatureName[] = [];
    if (precision === "f16") features.push("shader-f16");
    if (timing) features.push("timestamp-query");
    const device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });

    // Float tensors go into one weight buffer (offsets in elements). 8-bit matmul weights stay packed in a
    // second buffer, exactly as stored in the file (offsets in u32 words); the embedding, which is only
    // gathered, is dequantized here.
    const inits = readInitializers(onnx);
    const parts: { name: string; data: Float32Array }[] = [];
    const offsets = new Map<string, number>();
    const packed = new Map<string, Packed8>();
    const bytes8: Uint8Array[] = [];
    let total = 0;
    let total8 = 0;
    for (const [name, t] of Object.entries(plan.tensors)) {
      const init = inits.get(t.initializer);
      if (!init) throw new Error(`the ONNX file has no initializer ${t.initializer} (for ${name})`);
      const expected = t.shape.reduce((a, b) => a * b, 1);
      if (t.format) {
        const fmt = (options.formats ?? []).find((f) => f.kind === t.format!.kind);
        if (!fmt) throw new Error(`${name}: no plugin for weight format ${t.format.kind}`);
        const { bytes, floats, params } = fmt.pack(t, inits);
        const entry: Packed8 = { offsetWords: total8 / 4, kind: "int8", scale: 1, zeroPoint: 0, format: fmt,
          offsetBytes: total8, floatOffset: 0, params: params ?? [0, 0] };
        packed.set(name, entry);
        bytes8.push(bytes);
        total8 += Math.ceil(bytes.byteLength / 256) * 256;
        if (floats) {
          entry.floatOffset = total;
          parts.push({ name: `${name}#side`, data: floats });
          offsets.set(`${name}#side`, total);
          total += Math.ceil(floats.length / 64) * 64;
        }
        continue;
      }
      if (t.quant && name !== "embedding") {
        if (t.shape.length !== 2 || t.shape[1] % 4 !== 0) throw new Error(`${name}: unsupported 8-bit layout`);
        if (init.bytes.byteLength !== expected) throw new Error(`${name}: ${init.bytes.byteLength} bytes, plan says ${expected}`);
        let bytes = init.bytes;
        if (t.transpose) {
          const [rows, cols] = init.dims;
          bytes = new Uint8Array(bytes.length);
          for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) bytes[c * rows + r] = init.bytes[r * cols + c];
        }
        packed.set(name, { offsetWords: total8 / 4, kind: t.quant.dtype, scale: t.quant.scale, zeroPoint: t.quant.zero_point });
        bytes8.push(bytes);
        total8 += Math.ceil(init.bytes.byteLength / 256) * 256;
        continue;
      }
      let data = t.quant ? dequantTensor(init, t.quant.scale, t.quant.zero_point) : floatTensor(init, t.transpose);
      if (t.rows) {
        const cols = data.length / (t.transpose ? init.dims[1] : init.dims[0]);
        data = data.slice(t.rows[0] * cols, t.rows[1] * cols);
      }
      if (data.length !== expected) throw new Error(`${name}: ${data.length} values, plan says ${expected}`);
      offsets.set(name, total);
      parts.push({ name, data });
      total += Math.ceil(data.length / 64) * 64;
    }
    const all = new Float32Array(total);
    for (const { name, data } of parts) all.set(data, offsets.get(name)!);
    const upload: ArrayBufferView = precision === "f16" ? toHalf(all) : all;
    const weights = device.createBuffer({ size: upload.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(weights, 0, upload.buffer, upload.byteOffset, upload.byteLength);
    let weights8: GPUBuffer | null = null;
    if (total8 > 0) {
      weights8 = device.createBuffer({ size: total8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      let at = 0;
      for (const b of bytes8) {
        const copy = new Uint8Array(Math.ceil(b.byteLength / 4) * 4);
        copy.set(b);
        device.queue.writeBuffer(weights8, at, copy);
        at += Math.ceil(b.byteLength / 256) * 256;
      }
    }

    const tuning = { ...DEFAULT_TUNING, ...options.tuning };
    const engine = new Engine(device, info, precision, options.maxBatch ?? 1, plan, weights, offsets, timing, tuning,
      weights8, packed);
    engine.program(1);
    await device.queue.onSubmittedWorkDone();
    return engine;
  }

  private pipeline(key: string, code: () => string): GPUComputePipeline {
    let p = this.pipelines.get(key);
    if (!p) {
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code: code() }), entryPoint: "main" },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  private off(name: string): number {
    const o = this.offsets.get(name);
    if (o === undefined) throw new Error(`plan has no tensor ${name}`);
    return o;
  }

  /** The dispatch list for a batch size, built once and cached. */
  private program(batch: number): Program {
    const cached = this.programs.get(batch);
    if (cached) return cached;
    if (batch < 1 || batch > this.maxBatch) throw new Error(`batch ${batch} outside 1..${this.maxBatch}`);
    const c = this.config;
    const b = this.buffers;
    const f16 = this.precision === "f16";
    const words: ArrayBuffer[] = [];
    const specs: { label: string; pipeline: GPUComputePipeline; buffers: GPUBuffer[]; groups: [number, number, number] }[] = [];
    let label = "";
    const add = (pipeline: GPUComputePipeline, params: number[], floats: Record<number, number>, buffers: GPUBuffer[],
      groups: [number, number, number]) => {
      const block = new ArrayBuffer(32);
      const u = new Uint32Array(block);
      const f = new Float32Array(block);
      params.forEach((v, i) => { u[i] = v; });
      for (const [i, v] of Object.entries(floats)) f[Number(i)] = v;
      words.push(block);
      specs.push({ label, pipeline, buffers, groups });
    };

    const ctxIds = 0;
    const optIds = batch * c.context_len;
    const maskIds = optIds + batch * c.option_slots * c.option_len;
    const ctxRows = batch * c.context_len;
    const optSeqs = batch * c.option_slots;
    const optRows = optSeqs * c.option_len;

    const embed = this.pipeline(`embed${f16}`, () => K.embed(f16));
    label = "embed";
    add(embed, [c.width, c.context_len, ctxIds, this.off("embedding"), this.off("pos_context"), this.off("layer0.norm1.w"),
      this.off("layer0.norm1.b")], { 7: c.eps }, [b.weights, b.ids, b.xc, b.yc], [ctxRows, 1, 1]);
    add(embed, [c.width, c.option_len, optIds, this.off("embedding"), this.off("pos_option"), this.off("option_layer.norm1.w"),
      this.off("option_layer.norm1.b")], { 7: c.eps }, [b.weights, b.ids, b.xo, b.yo], [optRows, 1, 1]);

    // A @ W into S partials; returns S. `aSplits` > 0 reads A as act(sum of partials + bias).
    const matmul = (M: number, Kin: number, N: number, a: GPUBuffer, out: GPUBuffer, w: string,
      aSplits = 0, aBias: string | null = null, aRelu = false, target = this.tuning.splitTarget) => {
      const { S, RM } = splits(M, Kin, N, target);
      const q = this.packed.get(w);
      if (q?.format) {
        const v: K.MatmulVariant = { RM, KS: Kin / S, aSplits, aRelu, plugin: q.format.kind, pluginInner: !!q.format.inner };
        const pipe = this.pipeline(`mm${f16}${JSON.stringify(v)}`, () => K.matmul(f16, v, q.format!.wgsl));
        add(pipe, [M, N, Kin, q.offsetBytes!, aBias ? this.off(aBias) : 0, q.floatOffset!, ...q.params!], {},
          [b.weights8, a, out, b.weights], [Math.ceil(N / 256), Math.ceil(M / RM), S]);
        return S;
      }
      const v: K.MatmulVariant = { RM, KS: Kin / S, aSplits, aRelu, ...(q ? { w8: q.kind } : {}) };
      const pipe = this.pipeline(`mm${f16}${JSON.stringify(v)}`, () => K.matmul(f16, v));
      if (q) {
        add(pipe, [M, N, Kin, q.offsetWords, aBias ? this.off(aBias) : 0], { 5: q.zeroPoint, 6: q.scale },
          [b.weights8, a, out, ...(aSplits ? [b.weights] : [])], [Math.ceil(N / 256), Math.ceil(M / RM), S]);
      } else {
        add(pipe, [M, N, Kin, this.off(w), aBias ? this.off(aBias) : 0], {}, [b.weights, a, out],
          [Math.ceil(N / 256), Math.ceil(M / RM), S]);
      }
      return S;
    };
    const residualNorm = (M: number, part: GPUBuffer, S: number, bias: string, x: GPUBuffer, y: GPUBuffer, norm: string) => {
      const pipe = this.pipeline(`rn${f16}${S}`, () => K.residualNorm(f16, S));
      add(pipe, [M, c.width, this.off(bias), this.off(`${norm}.w`), this.off(`${norm}.b`)], { 5: c.eps },
        [b.weights, part, x, y], [M, 1, 1]);
    };
    const D = c.width / c.heads;
    // sequences that fit the short kernel's shared memory use it; longer ones use the chunked kernel
    const fitsShort = (L: number) => L <= 64 && Math.ceil(L / 8) * 8 * D * 2 * 4 <= 13000;
    const shortL = [c.context_len, c.option_len].filter(fitsShort);
    const maxL = shortL.length ? Math.ceil(Math.max(...shortL) / 8) * 8 : 8;
    if (D > 128) throw new Error("head dimension above 128 is not supported");
    const layer = (name: string, next: string, x: GPUBuffer, y: GPUBuffer, seqs: number, L: number, idsOff: number) => {
      const M = seqs * L;
      label = `${name}.qkv`;
      const sQkv = matmul(M, c.width, 3 * c.width, y, b.p1, `${name}.qkv.w`, 0, null, false, this.tuning.qkvSplitTarget);
      label = `${name}.attention`;
      const att = fitsShort(L)
        ? this.pipeline(`att${f16}${sQkv}`, () => K.attention(f16, D, sQkv, maxL))
        : this.pipeline(`attlong${f16}${sQkv}`, () => K.attentionLong(f16, D, sQkv));
      add(att, [L, idsOff, c.width, 0, M, this.off(`${name}.qkv.b`)],
        { 3: 1 / Math.sqrt(D) }, [b.weights, b.p1, b.ids, b.att], [c.heads, seqs, Math.ceil(L / 8)]);
      label = `${name}.out`;
      const sOut = matmul(M, c.width, c.width, b.att, b.p2, `${name}.out.w`);
      residualNorm(M, b.p2, sOut, `${name}.out.b`, x, y, `${name}.norm2`);
      label = `${name}.ff1`;
      const sFf1 = matmul(M, c.width, c.ff, y, b.p1, `${name}.ff1.w`);
      label = `${name}.ff2`;
      const sFf2 = matmul(M, c.ff, c.width, b.p1, b.p2, `${name}.ff2.w`, sFf1, `${name}.ff1.b`, true);
      residualNorm(M, b.p2, sFf2, `${name}.ff2.b`, x, y, next);
    };
    for (let i = 0; i < c.layers; i += 1) {
      layer(`layer${i}`, i + 1 < c.layers ? `layer${i + 1}.norm1` : "head.context_norm", b.xc, b.yc, batch, c.context_len, ctxIds);
    }
    layer("option_layer", "option_layer.norm1", b.xo, b.yo, optSeqs, c.option_len, optIds);
    label = "pool";
    add(this.pipeline(`pool${f16}`, () => K.poolNorm(f16)), [c.option_len, c.width, optIds, this.off("head.option_norm.w"),
      this.off("head.option_norm.b")], { 5: c.eps }, [b.weights, b.xo, b.ids, b.pooled], [optSeqs, 1, 1]);
    label = "head.qkv";
    const sq = matmul(optSeqs, c.width, c.rank, b.pooled, b.pq, "head.q.w");
    const sk = matmul(ctxRows, c.width, c.rank, b.yc, b.p1, "head.k.w");
    const sv = matmul(ctxRows, c.width, c.rank, b.yc, b.p2, "head.v.w");
    label = "head";
    add(this.pipeline(`head${sq}${sk}${sv}`, () => K.head(sq, sk, sv, c.context_len)),
      [c.context_len, c.option_slots, c.rank, ctxIds, maskIds, 0, optSeqs, ctxRows], { 5: 1 / Math.sqrt(c.rank) },
      [b.pq, b.p1, b.p2, b.ids, b.logits], [c.option_slots, batch, 1]);

    const params = this.device.createBuffer({
      size: words.length * PARAM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const blob = new Uint8Array(words.length * PARAM_STRIDE);
    words.forEach((w, i) => blob.set(new Uint8Array(w), i * PARAM_STRIDE));
    this.device.queue.writeBuffer(params, 0, blob);
    const dispatches = specs.map((s, i) => ({
      label: s.label,
      pipeline: s.pipeline,
      groups: s.groups,
      bindGroup: this.device.createBindGroup({
        layout: s.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params, offset: i * PARAM_STRIDE, size: 32 } },
          ...s.buffers.map((buffer, j) => ({ binding: j + 1, resource: { buffer } })),
        ],
      }),
    }));
    const program = { batch, dispatches, params };
    this.programs.set(batch, program);
    return program;
  }

  /**
   * Warm the GPU ahead of a decision. After one to two idle seconds the GPU and the browser's GPU
   * process drop into low-power states, and the next decision pays 10-100 ms for waking them (measured
   * on an M1 Max; the compute itself stays the same). A trivial submit does not wake them; real work
   * does. This re-runs the model on whatever inputs are in its buffers and resolves when the GPU is done.
   * Call it when a decision is coming (for example when the user clicks), a few hundred ms ahead.
   */
  wake(batch = 1): Promise<void> {
    const run = this.busy.then(async () => {
      const program = this.program(batch);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      for (const d of program.dispatches) {
        pass.setPipeline(d.pipeline);
        pass.setBindGroup(0, d.bindGroup);
        pass.dispatchWorkgroups(...d.groups);
      }
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    });
    this.busy = run.catch(() => undefined);
    return run;
  }

  /** Number of GPU dispatches per call (for reporting). */
  dispatchCount(batch = 1): number {
    return this.program(batch).dispatches.length;
  }

  /**
   * Score `batch` positions. contextIds [batch * context_len], optionIds [batch * slots * option_len],
   * optionMask [batch * slots] (non-zero = a real option). Returns logits [batch * slots];
   * masked slots get the lowest float. Calls are serialised.
   */
  score(contextIds: Int32Array, optionIds: Int32Array, optionMask: Int32Array, batch = 1): Promise<Float32Array> {
    const run = this.busy.then(() => this.run(contextIds, optionIds, optionMask, batch));
    this.busy = run.catch(() => undefined);
    return run;
  }

  private async run(contextIds: Int32Array, optionIds: Int32Array, optionMask: Int32Array, batch: number): Promise<Float32Array> {
    const c = this.config;
    const program = this.program(batch);
    const nCtx = batch * c.context_len;
    const nOpt = batch * c.option_slots * c.option_len;
    const nMask = batch * c.option_slots;
    if (contextIds.length !== nCtx || optionIds.length !== nOpt || optionMask.length !== nMask) {
      throw new Error(`input sizes ${contextIds.length}/${optionIds.length}/${optionMask.length}, expected ${nCtx}/${nOpt}/${nMask}`);
    }
    const ids = new Int32Array(nCtx + nOpt + nMask);
    ids.set(contextIds, 0);
    ids.set(optionIds, nCtx);
    ids.set(optionMask, nCtx + nOpt);
    const { device } = this;
    device.queue.writeBuffer(this.buffers.ids, 0, ids);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(this.querySet
      ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }
      : undefined);
    for (const d of program.dispatches) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bindGroup);
      pass.dispatchWorkgroups(...d.groups);
    }
    pass.end();
    const logitBytes = nMask * 4;
    const tsOffset = Math.ceil(logitBytes / 256) * 256;
    encoder.copyBufferToBuffer(this.buffers.logits, 0, this.staging, 0, logitBytes);
    if (this.querySet && this.queryBuffer) {
      encoder.resolveQuerySet(this.querySet, 0, 2, this.queryBuffer, 0);
      encoder.copyBufferToBuffer(this.queryBuffer, 0, this.staging, tsOffset, 16);
    }
    device.queue.submit([encoder.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.staging.getMappedRange(0, logitBytes).slice(0));
    if (this.querySet) {
      const ts = new BigUint64Array(this.staging.getMappedRange(tsOffset, 16));
      this.lastGpuMs = Number(ts[1] - ts[0]) / 1e6;
    }
    this.staging.unmap();
    return out;
  }

  /**
   * Development aid: run one call with every dispatch in its own timed pass and return the GPU time
   * per dispatch label in milliseconds (summed over dispatches sharing a label). Needs gpuTiming.
   */
  async profile(contextIds: Int32Array, optionIds: Int32Array, optionMask: Int32Array, batch = 1): Promise<Record<string, number>> {
    if (!this.querySet) throw new Error("profile() needs gpuTiming and an adapter with timestamp-query");
    await this.score(contextIds, optionIds, optionMask, batch);
    const { device } = this;
    const program = this.program(batch);
    const n = program.dispatches.length;
    const qs = device.createQuerySet({ type: "timestamp", count: 2 * n });
    const resolve = device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 16 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    program.dispatches.forEach((d, i) => {
      const pass = encoder.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bindGroup);
      pass.dispatchWorkgroups(...d.groups);
      pass.end();
    });
    encoder.resolveQuerySet(qs, 0, 2 * n, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, read, 0, 16 * n);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(read.getMappedRange());
    const out: Record<string, number> = {};
    program.dispatches.forEach((d, i) => {
      const key = d.label.replace(/^layer\d+\./, "layer.");
      out[key] = (out[key] ?? 0) + Number(ts[2 * i + 1] - ts[2 * i]) / 1e6;
    });
    read.unmap();
    qs.destroy(); resolve.destroy(); read.destroy();
    return out;
  }

  destroy(): void {
    for (const buffer of Object.values(this.buffers)) buffer.destroy();
    for (const p of this.programs.values()) p.params.destroy();
    this.staging.destroy();
    this.querySet?.destroy();
    this.queryBuffer?.destroy();
    this.device.destroy();
  }
}
