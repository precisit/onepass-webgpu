// Minimal ONNX reader: the initializers of a ModelProto, nothing else.
// Protobuf wire format: ModelProto.graph = 7, GraphProto.initializer = 5,
// TensorProto: dims = 1, data_type = 2, float_data = 4, name = 8, raw_data = 9, data_location = 14.

export interface Initializer {
  name: string;
  dims: number[];
  dataType: number; // 1 = float32, 2 = uint8, 3 = int8, 10 = float16, 7 = int64
  bytes: Uint8Array; // little-endian raw data (a view into the file, not a copy)
}

class Reader {
  pos: number;
  constructor(readonly buf: Uint8Array, start = 0, readonly end = buf.length) {
    this.pos = start;
  }
  varint(): number {
    let result = 0;
    let scale = 1;
    for (;;) {
      const b = this.buf[this.pos++];
      result += (b & 0x7f) * scale;
      if (b < 0x80) return result;
      scale *= 128;
    }
  }
  skip(wire: number): void {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) {
      const len = this.varint();
      this.pos += len;
    }
    else if (wire === 5) this.pos += 4;
    else throw new Error(`onnx: unsupported wire type ${wire}`);
  }
}

function fields(buf: Uint8Array, start: number, end: number, visit: (field: number, wire: number, r: Reader) => void): void {
  const r = new Reader(buf, start, end);
  while (r.pos < end) {
    const key = r.varint();
    const field = Math.floor(key / 8);
    const wire = key & 7;
    const before = r.pos;
    visit(field, wire, r);
    if (r.pos === before) r.skip(wire);
  }
}

function tensor(buf: Uint8Array, start: number, end: number): Initializer {
  const out: Initializer = { name: "", dims: [], dataType: 0, bytes: new Uint8Array(0) };
  let floats: number[] | null = null;
  fields(buf, start, end, (field, wire, r) => {
    if (field === 1 && wire === 0) out.dims.push(r.varint());
    else if (field === 1 && wire === 2) {
      const stop = r.varint() + r.pos;
      while (r.pos < stop) out.dims.push(r.varint());
    } else if (field === 2) out.dataType = r.varint();
    else if (field === 8) {
      const len = r.varint();
      out.name = new TextDecoder().decode(buf.subarray(r.pos, r.pos + len));
      r.pos += len;
    } else if (field === 9) {
      const len = r.varint();
      out.bytes = buf.subarray(r.pos, r.pos + len);
      r.pos += len;
    } else if (field === 4 && wire === 2) {
      const len = r.varint();
      floats = Array.from(new Float32Array(buf.slice(r.pos, r.pos + len).buffer));
      r.pos += len;
    } else if (field === 14 && r.varint() === 1) {
      throw new Error("onnx: external data is not supported yet");
    }
  });
  if (floats) out.bytes = new Uint8Array(new Float32Array(floats).buffer);
  return out;
}

/** All initializers of an ONNX file, by name. The byte views alias `file`. */
export function readInitializers(file: ArrayBuffer | Uint8Array): Map<string, Initializer> {
  const buf = file instanceof Uint8Array ? file : new Uint8Array(file);
  const found = new Map<string, Initializer>();
  fields(buf, 0, buf.length, (field, wire, r) => {
    if (field !== 7 || wire !== 2) return;
    const len = r.varint();
    const graphEnd = r.pos + len;
    fields(buf, r.pos, graphEnd, (gField, gWire, g) => {
      if (gField !== 5 || gWire !== 2) return;
      const tLen = g.varint();
      const t = tensor(buf, g.pos, g.pos + tLen);
      found.set(t.name, t);
      g.pos += tLen;
    });
    r.pos = graphEnd;
  });
  return found;
}

/** A float32 initializer as a Float32Array, optionally transposed from [rows, cols] to [cols, rows]. */
export function floatTensor(init: Initializer, transpose = false): Float32Array {
  if (init.dataType !== 1) throw new Error(`onnx: ${init.name} is data type ${init.dataType}, expected float32`);
  const src = new Float32Array(init.bytes.byteLength / 4);
  new Uint8Array(src.buffer).set(init.bytes);
  if (!transpose) return src;
  const [rows, cols] = init.dims;
  const out = new Float32Array(src.length);
  for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) out[c * rows + r] = src[r * cols + c];
  return out;
}

/** A per-tensor quantized initializer (int8 or uint8) as floats: (q - zeroPoint) * scale. */
export function dequantTensor(init: Initializer, scale: number, zeroPoint: number): Float32Array {
  if (init.dataType !== 2 && init.dataType !== 3) throw new Error(`onnx: ${init.name} is not int8 or uint8`);
  const q = init.dataType === 3
    ? new Int8Array(init.bytes.buffer, init.bytes.byteOffset, init.bytes.byteLength)
    : init.bytes;
  const out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i += 1) out[i] = Math.fround(Math.fround(q[i] - zeroPoint) * scale);
  return out;
}
