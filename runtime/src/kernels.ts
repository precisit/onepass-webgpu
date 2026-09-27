// WGSL kernels for the one-pass scorer. Activations are always f32; weights are f32 or f16 (WT).
// Every kernel reads its parameters from 8 words of a uniform buffer (see Engine.program).
//
// Matmuls are split along K ("partials"): `matmul` writes S partial products, and whoever reads
// the result next (a row epilogue, attention, or the next matmul's input load) sums them and adds
// the bias. At batch 1 this keeps thousands of threads busy with short loops instead of a few
// hundred threads with long ones.

const header = (f16: boolean) => `${f16 ? "enable f16;\n" : ""}alias WT = ${f16 ? "f16" : "f32"};\n`;

// Layer norm of one row held by a 256-thread workgroup (each thread owns up to 4 values).
const rowNorm = `
var<workgroup> red: array<f32, 256>;

fn wsum(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  let out = red[0];
  workgroupBarrier();
  return out;
}
`;

/** Per row: X = emb[id] + pos; Y = LN(X). One workgroup per row. */
export const embed = (f16: boolean) => `${header(f16)}
struct P { width: u32, seqLen: u32, idsOff: u32, embOff: u32, posOff: u32, lnW: u32, lnB: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> ids: array<i32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${rowNorm}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let m = wg.x + wg.y * 65535u;
  let id = u32(ids[p.idsOff + m]);
  let pos = m % p.seqLen;
  var x = 0.0;
  if (t < p.width) {
    x = f32(W[p.embOff + id * p.width + t]) + f32(W[p.posOff + pos * p.width + t]);
    X[m * p.width + t] = x;
  }
  let mean = wsum(select(0.0, x, t < p.width), t) / f32(p.width);
  let d = select(0.0, x - mean, t < p.width);
  let inv = 1.0 / sqrt(wsum(d * d, t) / f32(p.width) + p.eps);
  if (t < p.width) { Y[m * p.width + t] = d * inv * f32(W[p.lnW + t]) + f32(W[p.lnB + t]); }
}
`;

export interface MatmulVariant {
  RM: number; // rows per workgroup
  KS: number; // K per split
  aSplits: number; // 0: A is a plain [M, K] buffer; n: A = act(sum of n partials + bias)
  aRelu: boolean;
  /** Packed 8-bit weights (4 per u32, [K, N] row-major) with a per-tensor scale and zero point. */
  w8?: "int8" | "uint8";
  /** A plugin weight format: its WGSL defines w4(k, n4) (see WeightFormat in index.ts). */
  plugin?: string;
}

// four 8-bit weights of one u32 (little-endian: columns n..n+3) as floats
const unpack8 = (kind: "int8" | "uint8") => kind === "int8" ? `
fn unpack8(q: u32) -> vec4<f32> {
  let s = bitcast<i32>(q);
  return vec4<f32>(vec4<i32>((s << 24u) >> 24u, (s << 16u) >> 24u, (s << 8u) >> 24u, s >> 24u));
}` : `
fn unpack8(q: u32) -> vec4<f32> {
  return vec4<f32>(f32(q & 255u), f32((q >> 8u) & 255u), f32((q >> 16u) & 255u), f32(q >> 24u));
}`;

/**
 * Partial products: Out[s, m, n] = sum_{k in split s} A[m, k] * W[k, n]. With 8-bit weights, W stays packed
 * in GPU memory (4 per u32) and each thread unpacks its word to floats in registers.
 * Workgroup = 64 threads x 4 columns (vec4) = 256 columns, RM rows, one K split (wg.z).
 */
export const matmul = (f16: boolean, v: MatmulVariant, pluginWgsl = "") => `${header(f16)}
const RM = ${v.RM}u;
const KS = ${v.KS}u;
${v.plugin ? `struct P { M: u32, N: u32, K: u32, qOff: u32, aBias: u32, fOff: u32, x0: u32, x1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Q: array<u32>;
@group(0) @binding(4) var<storage, read> W: array<vec4<WT>>;
fn qbyte(i: u32) -> u32 { let b = p.qOff + i; return (Q[b >> 2u] >> ((b & 3u) * 8u)) & 255u; }
fn fval(i: u32) -> f32 { let j = p.fOff + i; return f32(W[j / 4u][j % 4u]); }
${pluginWgsl}` : `struct P { M: u32, N: u32, K: u32, wOff: u32, aBias: u32, zp: f32, scale: f32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
${v.w8 ? `@group(0) @binding(1) var<storage, read> Q: array<u32>;
${v.aSplits ? "@group(0) @binding(4) var<storage, read> W: array<vec4<WT>>;" : ""}
${unpack8(v.w8)}` : "@group(0) @binding(1) var<storage, read> W: array<vec4<WT>>;"}`}
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<vec4<f32>>;
var<workgroup> at: array<f32, ${v.RM * v.KS}>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let row0 = wg.y * RM;
  let k0 = wg.z * KS;
  for (var e = t; e < RM * KS; e += 64u) {
    let m = row0 + e / KS;
    let k = k0 + e % KS;
    var a = 0.0;
    if (m < p.M) {
${v.aSplits === 0 ? "      a = A[m * p.K + k];" : `      for (var j = 0u; j < ${v.aSplits}u; j += 1u) { a += A[(j * p.M + m) * p.K + k]; }
      a += f32(W[(p.aBias + k) / 4u][k % 4u]);${v.aRelu ? "\n      a = max(a, 0.0);" : ""}`}
    }
    at[e] = a;
  }
  workgroupBarrier();
  let n4 = wg.x * 64u + t;
  if (n4 * 4u >= p.N) { return; }
  let stride = p.N / 4u;
  var acc: array<vec4<f32>, RM>;
  ${v.plugin ? "" : `var wi = ${v.w8 ? "p.wOff" : "p.wOff / 4u"} + k0 * stride + n4;`}
  for (var kk = 0u; kk < KS; kk += 1u) {
    ${v.plugin ? "let w = w4(k0 + kk, n4);" : `let w = ${v.w8 ? "unpack8(Q[wi]) - vec4<f32>(p.zp)" : "vec4<f32>(W[wi])"};
    wi += stride;`}
    for (var r = 0u; r < RM; r += 1u) { acc[r] = fma(vec4<f32>(at[r * KS + kk]), w, acc[r]); }
  }
  let rows = min(RM, p.M - row0);
  for (var r = 0u; r < rows; r += 1u) { Out[(wg.z * p.M + row0 + r) * stride + n4] = acc[r]${v.w8 ? " * p.scale" : ""}; }
}
`;

/** Per row: X += sum_s Part[s] + bias (the residual add); Y = LN(X). One workgroup per row, width <= 256. */
export const residualNorm = (f16: boolean, splits: number) => `${header(f16)}
struct P { M: u32, width: u32, bOff: u32, lnW: u32, lnB: u32, eps: f32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${rowNorm}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let m = wg.x + wg.y * 65535u;
  var x = 0.0;
  if (t < p.width) {
    var y = 0.0;
    for (var j = 0u; j < ${splits}u; j += 1u) { y += Part[(j * p.M + m) * p.width + t]; }
    x = X[m * p.width + t] + (y + f32(W[p.bOff + t]));
    X[m * p.width + t] = x;
  }
  let mean = wsum(select(0.0, x, t < p.width), t) / f32(p.width);
  let d = select(0.0, x - mean, t < p.width);
  let inv = 1.0 / sqrt(wsum(d * d, t) / f32(p.width) + p.eps);
  if (t < p.width) { Y[m * p.width + t] = d * inv * f32(W[p.lnW + t]) + f32(W[p.lnB + t]); }
}
`;

/**
 * Masked multi-head self-attention. q, k, v come from the fused projection's partials (+ bias).
 * Workgroup = (head, sequence, block of 8 queries); 32 threads per query. L <= maxL.
 */
export const attention = (f16: boolean, headDim: number, splits: number, maxL: number) => `${header(f16)}
const D = ${headDim}u;
const LM = ${maxL}u;
const QB = 8u;
struct P { L: u32, idsOff: u32, width: u32, scale: f32, M: u32, bOff: u32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> Part: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
var<workgroup> ks: array<f32, ${maxL * headDim}>;
var<workgroup> vs: array<f32, ${maxL * headDim}>;
var<workgroup> qs: array<f32, ${8 * headDim}>;
var<workgroup> sc: array<f32, ${8 * maxL}>;

fn proj(row: u32, col: u32) -> f32 {
  var a = 0.0;
  for (var j = 0u; j < ${splits}u; j += 1u) { a += Part[(j * p.M + row) * 3u * p.width + col]; }
  return a + f32(W[p.bOff + col]);
}

fn valid(base: u32, l: u32) -> bool { return l == 0u || ids[p.idsOff + base + l] != 0; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let h = wg.x;
  let base = wg.y * p.L;
  let q0 = wg.z * QB;
  for (var e = t; e < p.L * D; e += 256u) {
    let col = h * D + e % D;
    ks[e] = proj(base + e / D, p.width + col);
    vs[e] = proj(base + e / D, 2u * p.width + col);
  }
  for (var e = t; e < QB * D; e += 256u) {
    let qi = q0 + e / D;
    qs[e] = select(0.0, proj(base + qi, h * D + e % D), qi < p.L);
  }
  workgroupBarrier();
  let qi = t / 32u;
  let lane = t % 32u;
  for (var l = lane; l < p.L; l += 32u) {
    var dot = 0.0;
    for (var d = 0u; d < D; d += 1u) { dot += qs[qi * D + d] * ks[l * D + d]; }
    sc[qi * LM + l] = select(-3.0e38, dot * p.scale, valid(base, l));
  }
  workgroupBarrier();
  var mx = -3.0e38;
  for (var l = 0u; l < p.L; l += 1u) { mx = max(mx, sc[qi * LM + l]); }
  workgroupBarrier();
  for (var l = lane; l < p.L; l += 32u) {
    sc[qi * LM + l] = select(0.0, exp(sc[qi * LM + l] - mx), valid(base, l));
  }
  workgroupBarrier();
  let query = q0 + qi;
  if (query >= p.L) { return; }
  var sum = 0.0;
  for (var l = 0u; l < p.L; l += 1u) { sum += sc[qi * LM + l]; }
  for (var d = lane; d < D; d += 32u) {
    var o = 0.0;
    for (var l = 0u; l < p.L; l += 1u) { o += sc[qi * LM + l] * vs[l * D + d]; }
    O[(base + query) * p.width + h * D + d] = o / sum;
  }
}
`;

/** Per option: mean over its non-padding tokens, then LN. One workgroup per option. */
export const poolNorm = (f16: boolean) => `${header(f16)}
struct P { L: u32, width: u32, idsOff: u32, lnW: u32, lnB: u32, eps: f32, p6: u32, p7: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<WT>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${rowNorm}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let s = wg.x + wg.y * 65535u;
  var x = 0.0;
  if (t < p.width) {
    var count = 0.0;
    for (var l = 0u; l < p.L; l += 1u) {
      if (ids[p.idsOff + s * p.L + l] != 0) {
        x += X[(s * p.L + l) * p.width + t];
        count += 1.0;
      }
    }
    x = x / max(count, 1.0);
  }
  let mean = wsum(select(0.0, x, t < p.width), t) / f32(p.width);
  let d = select(0.0, x - mean, t < p.width);
  let inv = 1.0 / sqrt(wsum(d * d, t) / f32(p.width) + p.eps);
  if (t < p.width) { Y[s * p.width + t] = d * inv * f32(W[p.lnW + t]) + f32(W[p.lnB + t]); }
}
`;

/**
 * Scoring head: each option's query attends over the context; logit = q . attended * scale.
 * q, k, v are read as sums of their matmul partials. One workgroup per (option, position).
 */
export const head = (sq: number, sk: number, sv: number) => `
struct P { Lc: u32, slots: u32, rank: u32, ctxOff: u32, maskOff: u32, scale: f32, Mq: u32, Mc: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;
var<workgroup> qv: array<f32, 1024>;
var<workgroup> sc: array<f32, 64>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) t: u32) {
  let n = wg.x;
  let b = wg.y;
  let qrow = b * p.slots + n;
  let krow = b * p.Lc;
  for (var r = t; r < p.rank; r += 256u) {
    var a = 0.0;
    for (var j = 0u; j < ${sq}u; j += 1u) { a += q[(j * p.Mq + qrow) * p.rank + r]; }
    qv[r] = a;
  }
  workgroupBarrier();
  // scores: 4 threads per context position
  let l = t / 4u;
  let part = t % 4u;
  var dot = 0.0;
  if (l < p.Lc) {
    for (var r = part; r < p.rank; r += 4u) {
      var kk = 0.0;
      for (var j = 0u; j < ${sk}u; j += 1u) { kk += k[(j * p.Mc + krow + l) * p.rank + r]; }
      dot += qv[r] * kk;
    }
  }
  red[t] = dot;
  workgroupBarrier();
  if (part == 0u && l < p.Lc) {
    sc[l] = (red[t] + red[t + 1u] + red[t + 2u] + red[t + 3u]) * p.scale;
  }
  workgroupBarrier();
  var mx = -3.0e38;
  var sum = 0.0;
  for (var i = 0u; i < p.Lc; i += 1u) { if (ids[p.ctxOff + krow + i] != 0) { mx = max(mx, sc[i]); } }
  for (var i = 0u; i < p.Lc; i += 1u) { if (ids[p.ctxOff + krow + i] != 0) { sum += exp(sc[i] - mx); } }
  var acc = 0.0;
  for (var r = t; r < p.rank; r += 256u) {
    var att = 0.0;
    for (var i = 0u; i < p.Lc; i += 1u) {
      if (ids[p.ctxOff + krow + i] == 0) { continue; }
      var vv = 0.0;
      for (var j = 0u; j < ${sv}u; j += 1u) { vv += v[(j * p.Mc + krow + i) * p.rank + r]; }
      att += (exp(sc[i] - mx) / sum) * vv;
    }
    acc += qv[r] * att;
  }
  workgroupBarrier();
  red[t] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  if (t == 0u) {
    logits[qrow] = select(-3.4028234663852886e38, red[0] * p.scale, ids[p.maskOff + qrow] != 0);
  }
}
`;
