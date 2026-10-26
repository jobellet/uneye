// formats.js - readers and writers of the eye-trace file formats of the web labeler, for trials of DIFFERENT durations.
//
// Read:  .mat (MATLAB v5, uncompressed or compressed; v7.3 = HDF5), .npz (numpy), .h5 / .hdf5 (HDF5). CSV stays in app.js.
// Write: .mat (v5), .npz, .h5 (the labels, with the positions, in the same layout).
// No dependency, except HDF5 which needs h5wasm (web/h5wasm.js, loaded on demand by the page; pass the module in Node).
// Works in the browser (global UnEyeFormats) and in Node >= 18 (module.exports), the same code for both.
//
// Layouts understood when reading (the first that matches wins; names are not case sensitive):
//   trials  cell array (MAT) / group of datasets trial_* (npz, h5) of (T_k x 2) or (2 x T_k) matrices [x y]
//   X, Y    two cell arrays of vectors, or two 2-D arrays trials x time (rows padded with NaN: the padding is cut)
//   xy      (sum T_k x 2) + trial_offset (N+1) or trial_lengths (N): trial k = rows trial_offset[k] : trial_offset[k+1]   <- what this page writes
//   eye     the same as xy: the eye traces of the mocol export (mocol_aMUA_eye_sessionNN.h5, trial_offset)
//   x, y    one trial (vectors)
// Optional: names (one string per trial), fs (sampling rate, Hz), labels (same layout as the positions: int8, 0 fixation, 1 saccade,
// 2 blink, -1 not reviewed; a cell array in MAT) which are loaded as the labels of the page.
// Optional machine detection (same layout as labels; never treated as reviewed labels): pred_labels (int8, 1 saccade, 0 none, 2 blink) and
// prob (float32, the network's P(saccade) before the threshold; NaN allowed) -> out.predLabels [Int8Array], out.prob [Float32Array];
// HDF5 attributes threshold / merge_ms / min_dur_ms of the post-processing -> out.probParams.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.UnEyeFormats = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const dec = new TextDecoder();
  const enc = new TextEncoder();

  // ---------------------------------------------------------------- helpers
  async function inflate(bytes, format) {                 // format: 'deflate' (zlib, MAT) or 'deflate-raw' (zip)
    if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot decompress the file (DecompressionStream is missing): save it uncompressed.');
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(format));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  const CRC_T = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1); t[n] = c >>> 0; }
    return t;
  })();
  function crc32(u8) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < u8.length; i++) c = CRC_T[(c ^ u8[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }

  function concatBytes(parts) {
    let n = 0; for (const p of parts) n += p.length;
    const out = new Uint8Array(n); let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  // a typed array of the same numbers, always a fresh aligned copy (views into a zip / mat buffer can start at any byte)
  function copyAs(Type, u8, offset, count) {
    const out = new Type(count);
    new Uint8Array(out.buffer).set(u8.subarray(offset, offset + count * Type.BYTES_PER_ELEMENT));
    return out;
  }

  // ---------------------------------------------------------------- ZIP (npz)
  async function readZip(buf) {
    const u8 = new Uint8Array(buf), dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let eocd = -1;
    for (let i = u8.length - 22; i >= Math.max(0, u8.length - 65557); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
    if (eocd < 0) throw new Error('Not a zip / npz file.');
    let n = dv.getUint16(eocd + 10, true), pos = dv.getUint32(eocd + 16, true);
    const files = {};
    for (let k = 0; k < n; k++) {
      if (dv.getUint32(pos, true) !== 0x02014b50) throw new Error('Corrupt zip directory.');
      const method = dv.getUint16(pos + 10, true);
      let csize = dv.getUint32(pos + 20, true), usize = dv.getUint32(pos + 24, true);
      const nl = dv.getUint16(pos + 28, true), el = dv.getUint16(pos + 30, true), cl = dv.getUint16(pos + 32, true);
      let lho = dv.getUint32(pos + 42, true);
      const name = dec.decode(u8.subarray(pos + 46, pos + 46 + nl));
      if (csize === 0xFFFFFFFF || usize === 0xFFFFFFFF || lho === 0xFFFFFFFF) {         // zip64 extra field
        let e = pos + 46 + nl; const end = e + el;
        while (e + 4 <= end) {
          const id = dv.getUint16(e, true), sz = dv.getUint16(e + 2, true);
          if (id === 1) {
            let q = e + 4;
            if (usize === 0xFFFFFFFF) { usize = Number(dv.getBigUint64(q, true)); q += 8; }
            if (csize === 0xFFFFFFFF) { csize = Number(dv.getBigUint64(q, true)); q += 8; }
            if (lho === 0xFFFFFFFF) { lho = Number(dv.getBigUint64(q, true)); q += 8; }
          }
          e += 4 + sz;
        }
      }
      const lnl = dv.getUint16(lho + 26, true), lel = dv.getUint16(lho + 28, true);
      const start = lho + 30 + lnl + lel;
      const raw = u8.subarray(start, start + csize);
      files[name] = method === 0 ? raw.slice() : (method === 8 ? await inflate(raw, 'deflate-raw') : (() => { throw new Error('Unsupported zip compression ' + method); })());
      pos += 46 + nl + el + cl;
    }
    return files;
  }

  function buildZip(entries) {                            // entries: [{name, data: Uint8Array}], stored (no compression)
    const parts = [], central = []; let offset = 0;
    for (const e of entries) {
      const name = enc.encode(e.name), crc = crc32(e.data), size = e.data.length;
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(8, 0, true);
      lh.setUint32(14, crc, true); lh.setUint32(18, size, true); lh.setUint32(22, size, true); lh.setUint16(26, name.length, true);
      parts.push(new Uint8Array(lh.buffer), name, e.data);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true);
      ch.setUint32(16, crc, true); ch.setUint32(20, size, true); ch.setUint32(24, size, true); ch.setUint16(28, name.length, true); ch.setUint32(42, offset, true);
      central.push(new Uint8Array(ch.buffer), name);
      offset += 30 + name.length + size;
    }
    const cd = concatBytes(central), end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, entries.length, true); end.setUint16(10, entries.length, true);
    end.setUint32(12, cd.length, true); end.setUint32(16, offset, true);
    return concatBytes([...parts, cd, new Uint8Array(end.buffer)]);
  }

  // ---------------------------------------------------------------- NPY
  const NPY_TYPES = { '<f8': Float64Array, '<f4': Float32Array, '<i1': Int8Array, '|i1': Int8Array, '<i2': Int16Array, '<i4': Int32Array,
                      '<u1': Uint8Array, '|u1': Uint8Array, '<u2': Uint16Array, '<u4': Uint32Array, '|b1': Uint8Array };

  // -> {shape, data} data: TypedArray in C order, or an Array of strings
  function parseNpy(u8) {
    if (u8[0] !== 0x93 || dec.decode(u8.subarray(1, 6)) !== 'NUMPY') throw new Error('Not an npy array.');
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength), major = u8[6];
    const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true), hstart = major === 1 ? 10 : 12;
    const header = dec.decode(u8.subarray(hstart, hstart + hlen));
    const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
    const fortran = /'fortran_order':\s*True/.test(header);
    const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)[1].split(',').map(s => s.trim()).filter(s => s.length).map(Number));
    const count = shape.reduce((a, b) => a * b, 1), start = hstart + hlen;
    let data;
    if (NPY_TYPES[descr]) data = copyAs(NPY_TYPES[descr], u8, start, count);
    else if (descr === '<i8' || descr === '<u8') {
      const big = copyAs(descr === '<i8' ? BigInt64Array : BigUint64Array, u8, start, count); data = new Float64Array(count);
      for (let i = 0; i < count; i++) data[i] = Number(big[i]);
    } else if (/^<U(\d+)$/.test(descr)) {
      const L = Number(/^<U(\d+)$/.exec(descr)[1]); data = [];
      const cp = copyAs(Uint32Array, u8, start, count * L);
      for (let i = 0; i < count; i++) { let s = ''; for (let j = 0; j < L; j++) { const c = cp[i * L + j]; if (c) s += String.fromCodePoint(c); } data.push(s); }
    } else if (/^\|S(\d+)$/.test(descr)) {
      const L = Number(/^\|S(\d+)$/.exec(descr)[1]); data = [];
      for (let i = 0; i < count; i++) data.push(dec.decode(u8.subarray(start + i * L, start + (i + 1) * L)).replace(/\0+$/, ''));
    } else throw new Error('Unsupported array type ' + descr + ' (object arrays / pickles cannot be read: save with plain numeric arrays).');
    if (fortran && shape.length === 2) data = transpose(data, shape[0], shape[1], true);
    return { shape, data, isBool: descr === '|b1' };
  }

  // 2-D re-ordering: fromF=true: data is column-major (r x c) -> row-major
  function transpose(data, r, c, fromF) {
    const out = new (data.constructor)(data.length);
    if (fromF) { for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i * c + j] = data[j * r + i]; }
    else { for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = data[i * c + j]; }
    return out;
  }

  async function readNpz(buf) {
    const files = await readZip(buf), vars = {};
    for (const name of Object.keys(files)) {
      if (!name.endsWith('.npy')) continue;
      const a = parseNpy(files[name]);
      vars[name.slice(0, -4)] = { shape: a.shape, data: a.data, isBool: !!a.isBool };
    }
    return vars;
  }

  // numpy .npz: arrays = [{name, type: 'f4'|'f8'|'i1'|'i4', shape, data: TypedArray (C order)} | {name, strings: [...]}]
  function buildNpz(arrays) {
    const NP = { f4: ['<f4', Float32Array], f8: ['<f8', Float64Array], i1: ['|i1', Int8Array], i4: ['<i4', Int32Array], u1: ['|u1', Uint8Array] };
    const entries = arrays.map(a => {
      let descr, body, shape;
      if (a.strings) {
        const L = Math.max(1, ...a.strings.map(s => [...s].length)); shape = [a.strings.length]; descr = '<U' + L;
        const u32 = new Uint32Array(a.strings.length * L);
        a.strings.forEach((s, i) => { [...s].forEach((ch, j) => { u32[i * L + j] = ch.codePointAt(0); }); });
        body = new Uint8Array(u32.buffer);
      } else {
        const [d, T] = NP[a.type]; descr = d; shape = a.shape;
        const typed = a.data instanceof T ? a.data : T.from(a.data);
        body = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
      }
      const shapeStr = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
      let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
      const pad = 64 - ((10 + header.length + 1) % 64); header += ' '.repeat(pad % 64) + '\n';
      const h = new Uint8Array(10 + header.length); h.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 255, header.length >> 8]); h.set(enc.encode(header), 10);
      return { name: a.name + '.npy', data: concatBytes([h, body]) };
    });
    return buildZip(entries);
  }

  // ---------------------------------------------------------------- MATLAB v5
  const MI = { INT8: 1, UINT8: 2, INT16: 3, UINT16: 4, INT32: 5, UINT32: 6, SINGLE: 7, DOUBLE: 9, INT64: 12, UINT64: 13, MATRIX: 14, COMPRESSED: 15, UTF8: 16, UTF16: 17 };
  const MX = { CELL: 1, STRUCT: 2, OBJECT: 3, CHAR: 4, SPARSE: 5, DOUBLE: 6, SINGLE: 7, INT8: 8, UINT8: 9, INT16: 10, UINT16: 11, INT32: 12, UINT32: 13, INT64: 14, UINT64: 15 };

  async function readMat5(buf) {
    const u8 = new Uint8Array(buf);
    if (dec.decode(u8.subarray(0, 19)) === 'MATLAB 7.3 MAT-file') throw Object.assign(new Error('MATLAB v7.3 file (HDF5)'), { hdf5: true });
    if (u8.length < 128 || (dec.decode(u8.subarray(126, 128)) !== 'IM' && dec.decode(u8.subarray(126, 128)) !== 'MI')) throw new Error('Not a MATLAB v5 file.');
    const little = dec.decode(u8.subarray(126, 128)) === 'IM';
    if (!little) throw new Error('Big-endian MATLAB files are not supported.');
    const vars = {};
    await parseElements(u8, 128, u8.length, vars, null);
    return vars;
  }

  // reads the data elements of buf[start, end): top level (vars) or inside a cell (list)
  async function parseElements(u8, start, end, vars, list, keepNulls) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let pos = start;
    while (pos + 8 <= end) {
      let type = dv.getUint32(pos, true), size = dv.getUint32(pos + 4, true), body = pos + 8;
      if ((type >>> 16) !== 0) { size = type >>> 16; type = type & 0xFFFF; body = pos + 4; }      // small data element format
      const next = type === MI.COMPRESSED ? body + size : body + ((size + 7) & ~7);       // compressed elements are not padded to 8 bytes
      if (type === MI.COMPRESSED) {
        const inner = await inflate(u8.subarray(body, body + size), 'deflate');
        await parseElements(inner, 0, inner.length, vars, list, keepNulls);
      } else if (type === MI.MATRIX) {
        const m = await parseMatrix(u8, body, body + size);
        if (m) { if (list) list.push(m); else vars[m.name] = m; }
        else if (list && keepNulls) list.push({ cls: 'skip', dims: [0, 0] });
      }
      pos = next > pos ? next : end;
    }
  }

  function readTag(u8, pos) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    let type = dv.getUint32(pos, true), size = dv.getUint32(pos + 4, true), body = pos + 8;
    if ((type >>> 16) !== 0) { size = type >>> 16; type = type & 0xFFFF; body = pos + 4; return { type, size, body, next: pos + 8 }; }
    return { type, size, body, next: body + ((size + 7) & ~7) };
  }

  function numbersOf(u8, tag) {                           // numeric data element -> Float64Array
    const T = { [MI.INT8]: Int8Array, [MI.UINT8]: Uint8Array, [MI.INT16]: Int16Array, [MI.UINT16]: Uint16Array, [MI.INT32]: Int32Array, [MI.UINT32]: Uint32Array,
                [MI.SINGLE]: Float32Array, [MI.DOUBLE]: Float64Array }[tag.type];
    if (T) { const a = copyAs(T, u8, tag.body, tag.size / T.BYTES_PER_ELEMENT); return Float64Array.from(a); }
    if (tag.type === MI.INT64 || tag.type === MI.UINT64) { const b = copyAs(tag.type === MI.INT64 ? BigInt64Array : BigUint64Array, u8, tag.body, tag.size / 8); return Float64Array.from(b, Number); }
    throw new Error('Unsupported MATLAB data type ' + tag.type);
  }

  async function parseMatrix(u8, start, end) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const t1 = readTag(u8, start); const cls = dv.getUint32(t1.body, true) & 0xFF;            // array flags
    const t2 = readTag(u8, t1.next); const dims = []; for (let i = 0; i < t2.size / 4; i++) dims.push(dv.getInt32(t2.body + 4 * i, true));
    const t3 = readTag(u8, t2.next); const name = dec.decode(u8.subarray(t3.body, t3.body + t3.size));
    const count = dims.reduce((a, b) => a * b, 1);
    if (cls === MX.CELL) {
      const cells = []; await parseElements(u8, t3.next, end, null, cells);
      return { name, cls: 'cell', dims, cells };
    }
    if (cls === MX.CHAR) {
      const t4 = readTag(u8, t3.next); let s;
      if (t4.type === MI.UINT16 || t4.type === MI.UTF16) { const a = copyAs(Uint16Array, u8, t4.body, t4.size / 2); s = String.fromCharCode.apply(null, a); }
      else s = dec.decode(u8.subarray(t4.body, t4.body + t4.size));
      // column-major rows: one string per row (a row vector is the common case)
      const rows = dims[0], cols = dims[1] || 0, out = [];
      for (let r = 0; r < rows; r++) { let q = ''; for (let c = 0; c < cols; c++) q += s[c * rows + r] || ''; out.push(q.replace(/\s+$/, '')); }
      return { name, cls: 'char', dims, strings: out };
    }
    if (cls >= MX.DOUBLE && cls <= MX.UINT64) {
      const t4 = readTag(u8, t3.next), flags = dv.getUint32(t1.body, true);
      if (flags & 0x0800) return null;                                                       // complex: ignored
      const data = count ? numbersOf(u8, t4) : new Float64Array(0);
      const shape = dims.length === 2 ? dims : dims;
      return { name, cls: 'numeric', dims: shape, data, order: 'F', isBool: Boolean(flags & 0x0200) };
    }
    if (cls === MX.STRUCT) {                                                                 // fields -> {cls: 'struct', dims, fields, elems: [{field: matrix}]}
      const tl = readTag(u8, t3.next), fl = dv.getInt32(tl.body, true);
      const tn = readTag(u8, tl.next), fields = [];
      for (let i = 0; fl > 0 && i < tn.size / fl; i++) fields.push(dec.decode(u8.subarray(tn.body + i * fl, tn.body + (i + 1) * fl)).replace(/\0[\s\S]*$/, ''));
      const mats = []; await parseElements(u8, tn.next, end, null, mats, true);
      const elems = [];
      for (let e = 0; e < count && e < 200; e++) { const o = {}; fields.forEach((f, k) => { o[f] = mats[e * fields.length + k]; }); elems.push(o); }
      return { name, cls: 'struct', dims, fields, elems };
    }
    return null;                                                                             // object, sparse: ignored
  }

  // MATLAB v5, uncompressed. vars: [{name, type: 'f8', dims: [r, c], data: column-major numbers} | {name, cell: [{dims, data}, ...] (column of matrices)} | {name, strings: [...]}]
  function buildMat5(vars) {
    const pad8 = n => (8 - (n % 8)) % 8;
    const tag = (type, bytes) => { const h = new DataView(new ArrayBuffer(8)); h.setUint32(0, type, true); h.setUint32(4, bytes.length, true); return concatBytes([new Uint8Array(h.buffer), bytes, new Uint8Array(pad8(bytes.length))]); };
    const i32 = a => new Uint8Array(Int32Array.from(a).buffer);
    const flagsEl = cls => { const f = new DataView(new ArrayBuffer(8)); f.setUint32(0, cls, true); return tag(MI.UINT32, new Uint8Array(f.buffer)); };
    const matrix = (name, cls, dims, dataEl) => {
      const body = concatBytes([flagsEl(cls), tag(MI.INT32, i32(dims)), tag(MI.INT8, enc.encode(name)), ...dataEl]);
      return tag(MI.MATRIX, body);
    };
    const numeric = (name, dims, data, Type, mi, mx) => {
      const typed = data instanceof Type ? data : Type.from(data);
      return matrix(name, mx, dims, [tag(mi, new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength))]);
    };
    const parts = [];
    for (const v of vars) {
      if (v.cell) {
        const els = v.cell.map(c => numeric('', c.dims, c.data, c.single ? Float32Array : Float64Array, c.single ? MI.SINGLE : MI.DOUBLE, c.single ? MX.SINGLE : MX.DOUBLE));
        parts.push(matrix(v.name, MX.CELL, [v.cell.length, 1], els));
      } else if (v.strings) {
        const els = v.strings.map(s => { const a = Uint16Array.from([...s].map(ch => ch.charCodeAt(0))); return matrix('', MX.CHAR, [1, a.length], [tag(MI.UINT16, new Uint8Array(a.buffer))]); });
        parts.push(matrix(v.name, MX.CELL, [v.strings.length, 1], els));
      } else if (v.type === 'i1') parts.push(numeric(v.name, v.dims, v.data, Int8Array, MI.INT8, MX.INT8));
      else if (v.type === 'i4') parts.push(numeric(v.name, v.dims, v.data, Int32Array, MI.INT32, MX.INT32));
      else parts.push(numeric(v.name, v.dims, v.data, Float64Array, MI.DOUBLE, MX.DOUBLE));
    }
    const head = new Uint8Array(128); head.set(enc.encode('MATLAB 5.0 MAT-file, Platform: web, Created by the U\'n\'Eye web labeler'));
    new DataView(head.buffer).setUint16(124, 0x0100, true); head.set(enc.encode('IM'), 126);
    return concatBytes([head, ...parts]);
  }

  // ---------------------------------------------------------------- HDF5 (h5wasm)
  // h5 = the h5wasm module (browser: global h5wasm; Node: await import('h5wasm/node'))
  // a writable in-memory directory of the h5wasm file system (the root can be read-only, as in the Node build)
  function memDir(FS) {
    if (FS.__uneyeDir) return FS.__uneyeDir;
    let dir = '';
    try { if (FS.filesystems && FS.filesystems.MEMFS) { FS.mkdir('/uneye_mem'); FS.mount(FS.filesystems.MEMFS, {}, '/uneye_mem'); dir = '/uneye_mem/'; } } catch (e) { dir = ''; }
    FS.__uneyeDir = dir;
    return dir;
  }

  const H5_WANTED = /^(xy|eye|x|y|trial_offset|trial_lengths|lengths|names|fs|labels|pred_labels|predlabels|prob|observed|reviewed|trials?.*|x_\d+|y_\d+)$/i;
  const H5_SKIP = /^(a?mua|lfp|ephys|neural|raw_ephys)$/i;

  // opts.all: every numeric dataset (up to 40 M values each), not only the layouts of this page - for the column explorer
  async function readH5(buf, h5, opts) {
    const all = !!(opts && opts.all);
    const Module = await h5.ready, FS = Module.FS, path = memDir(FS) + 'in_' + Math.random().toString(36).slice(2) + '.h5';
    FS.writeFile(path, new Uint8Array(buf));
    const vars = {};
    try {
      const f = new h5.File(path, 'r');
      const walk = (group, prefix) => {
        for (const k of group.keys()) {
          const item = group.get(k);
          if (item && item.type === 'Group') { walk(item, prefix + k + '/'); continue; }
          const shape = item.shape || [];
          const count = shape.reduce((a, b) => a * b, 1);
          const wanted = H5_WANTED.test(k) || H5_WANTED.test(prefix.split('/')[0] || '');
          const small1Dor2Col = !H5_SKIP.test(k) && count <= 4e7 && shape.length <= 2;
          if (!all && !wanted && !small1Dor2Col) continue;
          if (all && count > 4e7) continue;
          let v; try { v = item.value; } catch (e) { continue; }
          const isBoolArr = Array.isArray(v) && v.length > 0 && typeof v[0] === 'boolean';
          if (isBoolArr) v = Uint8Array.from(v, x => x ? 1 : 0);
          const isEnumBool = Boolean(item.metadata && item.metadata.type === 8);
          const isStrArr = Array.isArray(v) && v.length <= 100000 && (v.length === 0 || typeof v[0] === 'string');
          if (all && !ArrayBuffer.isView(v) && typeof v !== 'number' && !isStrArr) continue;
          vars[(prefix + k).replace(/^(trials?[^/]*)\//i, '$1/')] = { shape, data: (v && v.constructor && v.constructor.name === 'BigInt64Array') ? Float64Array.from(v, Number) : v, isBool: isBoolArr || isEnumBool };
        }
      };
      walk(f, '');
      const attrs = f.attrs || {};
      for (const k of Object.keys(attrs)) {
        const val = () => Number(attrs[k].value !== undefined ? attrs[k].value : attrs[k]);
        if (/^fs$/i.test(k) && !vars.fs) vars.fs = { shape: [], data: [val()] };
        else if (/^(threshold|merge_ms|min_dur_ms|savgol_ms)$/i.test(k)) { const v = val(); if (isFinite(v)) vars['@' + k.toLowerCase()] = v; }
      }
      f.close();
    } finally { try { FS.unlink(path); } catch (e) { /* ignore */ } }
    return vars;
  }

  // vars: [{name, type: 'f4'|'i1'|'i4', shape, data} | {name, strings}], attrs: {fs: 1000}
  async function buildH5(vars, attrs, h5) {
    const Module = await h5.ready, FS = Module.FS, path = memDir(FS) + 'out_' + Math.random().toString(36).slice(2) + '.h5';
    const f = new h5.File(path, 'w');
    for (const v of vars) {
      if (v.strings) f.create_dataset({ name: v.name, data: v.strings });
      else f.create_dataset({ name: v.name, data: v.data, shape: v.shape, dtype: { f4: '<f', f8: '<d', i1: '<b', i4: '<i' }[v.type] });
    }
    for (const k of Object.keys(attrs || {})) f.create_attribute(k, attrs[k]);
    f.close();
    const bytes = FS.readFile(path).slice(); FS.unlink(path);
    return bytes;
  }

  // ---------------------------------------------------------------- vars -> trials
  // v: {shape, data} (C order) or MAT {cls, dims, data (F order), cells, strings}. Returns plain objects.
  function toC(v) {
    if (v.cls === 'numeric') { const d = v.dims; if (d.length === 2 && d[0] > 1 && d[1] > 1) return { shape: d, data: transpose(v.data, d[0], d[1], true), isBool: !!v.isBool }; return { shape: d, data: v.data, isBool: !!v.isBool }; }
    return v;
  }
  const lc = vars => { const o = {}; for (const k of Object.keys(vars)) o[k.toLowerCase()] = vars[k]; return o; };
  const strList = v => !v ? null : (v.strings ? v.strings : (v.cells ? v.cells.map(c => (c.strings ? c.strings.join('') : '')) : (Array.isArray(v.data) ? v.data.map(String) : null)));

  function cut(arr, n) { return arr.subarray ? arr.subarray(0, n) : arr.slice(0, n); }
  function trimNaN(x, y) {                                   // cut the trailing padding (both NaN)
    let n = x.length; while (n > 0 && !isFinite(x[n - 1]) && !isFinite(y[n - 1])) n--;
    return [cut(x, n), cut(y, n)];
  }
  function mkTrial(x, y, name) {
    [x, y] = trimNaN(Float32Array.from(x), Float32Array.from(y));
    const valid = new Uint8Array(x.length); for (let i = 0; i < x.length; i++) valid[i] = (isFinite(x[i]) && isFinite(y[i])) ? 1 : 0;
    return { x, y, valid, name };
  }
  // (n x 2) or (2 x n) matrix, row-major data -> [x, y]
  function xyOf(shape, data) {
    const [a, b] = shape;
    if (b === 2) { const n = a, x = new Float64Array(n), y = new Float64Array(n); for (let i = 0; i < n; i++) { x[i] = data[2 * i]; y[i] = data[2 * i + 1]; } return [x, y]; }
    if (a === 2) return [Float64Array.from(data.subarray ? data.subarray(0, b) : data.slice(0, b)), Float64Array.from(data.subarray ? data.subarray(b, 2 * b) : data.slice(b, 2 * b))];
    throw new Error('A trial must be an N x 2 or 2 x N matrix of x and y (found ' + a + ' x ' + b + ').');
  }

  // Inspects a vars map (from readVarsFile / readH5 / readMat5 / readNpz) to list candidate per-sample variables for labels and probability.
  function inspectVars(vars0) {
    const vars = lc(vars0);
    let fs = null;
    if (vars.fs) { const f = toC(vars.fs); const d = f.data; const fv = Number(d && d.length ? d[0] : d); if (isFinite(fv) && fv > 0) fs = fv; }
    const savgolMs = vars['@savgol_ms'] !== undefined ? Number(vars['@savgol_ms']) : null;
    const pp = {}; for (const k of ['threshold', 'merge_ms', 'min_dur_ms']) if (vars['@' + k] !== undefined) pp[k] = vars['@' + k];
    const probParams = Object.keys(pp).length ? pp : null;

    // Determine the expected sample count / layout of per-sample variables
    let layout = null, nSamples = 0, nTrials = 0, shape2D = null;
    const num = k => (vars[k] ? toC(vars[k]) : null);
    if (vars.trials && vars.trials.cells) {
      layout = 'cells'; nTrials = vars.trials.cells.length;
      for (const c of vars.trials.cells) { const s = c.shape || c.dims || []; nSamples += Math.max(s[0] || 0, s[1] || 0); }
    } else if (vars.x && vars.x.cells && vars.y && vars.y.cells) {
      layout = 'cells'; nTrials = vars.x.cells.length;
      for (const c of vars.x.cells) nSamples += (c.data ? c.data.length : 0);
    } else if (vars.xy || vars.eye) {
      const xyv = num(vars.xy ? 'xy' : 'eye');
      if (xyv && xyv.shape && xyv.shape[1] === 2) {
        layout = 'concat'; nSamples = xyv.shape[0];
        const off = num('trial_offset'), len = num('trial_lengths') || num('lengths');
        nTrials = off && off.data ? Math.max(1, off.data.length - 1) : (len && len.data ? len.data.length : 1);
      }
    } else if ((vars.x && vars.y) && ((vars.x.shape || vars.x.dims || []).length === 2) && Math.min(...(vars.x.shape || vars.x.dims)) > 1) {
      const X = num('x');
      layout = '2d'; shape2D = X.shape; nTrials = X.shape[0]; nSamples = X.shape[0] * X.shape[1];
    } else if (vars.x && vars.y && num('x').data) {
      layout = 'concat'; nTrials = 1; nSamples = num('x').data.length;
    }

    const SKIP_KEYS = /^(xy|eye|x|y|trials|trial_offset|trial_lengths|lengths|names|fs|@.*)$/i;
    const candidates = [];
    const summarizeArr = (arr, vIsBool) => {
      if (!arr || !arr.length) return { min: NaN, max: NaN, isInt: false, isBool: !!vIsBool, hasNeg: false };
      let min = Infinity, max = -Infinity, isInt = true, hasNeg = false, finite = 0;
      const step = Math.max(1, Math.floor(arr.length / 5000));
      for (let i = 0; i < arr.length; i += step) {
        const v = arr[i];
        if (v === v && isFinite(v)) {
          finite++;
          if (v < min) min = v;
          if (v > max) max = v;
          if (v !== Math.floor(v)) isInt = false;
          if (v < 0) hasNeg = true;
        }
      }
      const intOk = finite > 0 && isInt;
      const isBool = Boolean(vIsBool || (intOk && min >= 0 && max <= 1));
      return { min: finite ? min : NaN, max: finite ? max : NaN, isInt: intOk, isBool, hasNeg };
    };

    for (const origKey of Object.keys(vars0)) {
      if (SKIP_KEYS.test(origKey)) continue;
      const v = vars0[origKey];
      if (!v) continue;
      if (layout === 'cells' && v.cells && v.cells.length === nTrials && v.cells[0] && toC(v.cells[0]).data) {
        const c0 = toC(v.cells[0]);
        const st = summarizeArr(c0.data, v.isBool || c0.isBool);
        candidates.push({ name: origKey, length: nSamples, ...st });
      } else if (layout === '2d' && shape2D) {
        const c = toC(v);
        if (c && c.shape && c.shape.length === 2 && c.shape[0] === shape2D[0] && c.shape[1] === shape2D[1] && c.data && ArrayBuffer.isView(c.data)) {
          const st = summarizeArr(c.data, v.isBool || c.isBool);
          candidates.push({ name: origKey, length: nSamples, ...st });
        }
      } else {
        const c = toC(v);
        if (!c || !c.data || !ArrayBuffer.isView(c.data)) continue;
        const sh = c.shape || c.dims || [c.data.length];
        const is1D = sh.length === 1 || (sh.length === 2 && (sh[0] === 1 || sh[1] === 1));
        const len = c.data.length;
        if (is1D && ((layout === 'concat' && len === nSamples) || (!layout && len >= 20))) {
          const st = summarizeArr(c.data, v.isBool || c.isBool);
          candidates.push({ name: origKey, length: len, ...st });
        }
      }
    }

    for (const c of candidates) {
      const rng = isFinite(c.min) ? (c.isInt ? `${c.min}…${c.max}` : `${Number(c.min.toFixed(3))}…${Number(c.max.toFixed(3))}`) : 'empty';
      c.summary = `${c.isBool ? 'bool (0/1) · ' : c.isInt ? 'int · ' : ''}${rng} · ${c.length.toLocaleString()} samples`;
    }

    let defaultLabelsVar = '', defaultLabelsRole = 'prelabels', defaultProbVar = '';
    const byLc = new Map(candidates.map(c => [c.name.toLowerCase(), c]));
    if (byLc.has('labels')) {
      defaultLabelsVar = byLc.get('labels').name;
      defaultLabelsRole = 'reviewed';
    } else if (byLc.has('pred_labels')) {
      defaultLabelsVar = byLc.get('pred_labels').name;
      defaultLabelsRole = 'prelabels';
    } else if (byLc.has('predlabels')) {
      defaultLabelsVar = byLc.get('predlabels').name;
      defaultLabelsRole = 'prelabels';
    } else {
      const m = candidates.find(c => /^(saccade_?labels?|label|l|events?|gt|ground_?truth|is_?sacc\w*|saccade_?mask|saccade_?bool|bool_?labels?)$/i.test(c.name) && (c.isBool || (c.isInt && c.min >= -1 && c.max <= 4)));
      if (m) { defaultLabelsVar = m.name; defaultLabelsRole = m.hasNeg ? 'reviewed' : 'prelabels'; }
    }

    if (byLc.has('prob')) {
      defaultProbVar = byLc.get('prob').name;
    } else {
      const p = candidates.find(c => /^(probability|p_?sacc\w*|pred_?prob|net_?prob|saccade_?prob)$/i.test(c.name) && c.min >= -0.001 && c.max <= 1.001);
      if (p) defaultProbVar = p.name;
    }

    return {
      hasStandardTrace: !!layout,
      fs,
      savgolMs: isFinite(savgolMs) ? savgolMs : null,
      probParams,
      nTrials,
      nSamples,
      vars: candidates,
      defaultLabelsVar,
      defaultLabelsRole,
      defaultProbVar,
    };
  }

  // Lists candidate label / boolean variables in vars0 that match an existing trials array (or standalone layout)
  function listLabelVars(vars0, trials) {
    const vars = lc(vars0);
    const nTrials = trials && trials.length ? trials.length : 0;
    const totalSamples = nTrials ? trials.reduce((a, t) => a + t.x.length, 0) : 0;
    const maxTrialLen = nTrials ? Math.max(...trials.map(t => t.x.length)) : 0;
    const offV = vars.trial_offset ? toC(vars.trial_offset) : null;
    const offTotal = (offV && offV.data && offV.data.length >= 2) ? Number(offV.data[offV.data.length - 1]) : 0;
    const SKIP = /^(xy|eye|x|y|trials|trial_offset|trial_lengths|lengths|names|fs|@.*)$/i;
    const out = [];
    for (const origKey of Object.keys(vars0 || {})) {
      if (SKIP.test(origKey)) continue;
      const v = vars0[origKey];
      if (!v) continue;
      let sampleArr = null, length = 0, matches = false;
      if (v.cells && (!nTrials || v.cells.length >= nTrials) && v.cells[0] && toC(v.cells[0]).data) {
        sampleArr = toC(v.cells[0]).data;
        length = v.cells.reduce((a, cl) => a + (toC(cl).data ? toC(cl).data.length : 0), 0);
        matches = true;
      } else {
        const c = toC(v);
        if (!c || !c.data || !ArrayBuffer.isView(c.data)) continue;
        sampleArr = c.data;
        length = c.data.length;
        const sh = c.shape || c.dims || [length];
        const is1D = sh.length === 1 || (sh.length === 2 && (sh[0] === 1 || sh[1] === 1));
        const is2D = sh.length === 2 && sh[0] > 1 && sh[1] > 1;
        if (is1D && (!nTrials || length === totalSamples || (offTotal > 0 && length === offTotal) || (nTrials === 1 && length >= trials[0].x.length))) matches = true;
        else if (is2D && (!nTrials || (sh[0] === nTrials && sh[1] >= maxTrialLen) || (sh[1] === nTrials && sh[0] >= maxTrialLen))) matches = true;
      }
      if (!matches || !sampleArr || !sampleArr.length) continue;
      let min = Infinity, max = -Infinity, isInt = true, hasNeg = false, finite = 0;
      const step = Math.max(1, Math.floor(sampleArr.length / 5000));
      for (let i = 0; i < sampleArr.length; i += step) {
        const val = sampleArr[i];
        if (val === val && isFinite(val)) {
          finite++;
          if (val < min) min = val;
          if (val > max) max = val;
          if (val !== Math.floor(val)) isInt = false;
          if (val < 0) hasNeg = true;
        }
      }
      if (!finite) continue;
      const intOk = isInt;
      const isBool = Boolean(v.isBool || (intOk && min >= 0 && max <= 1));
      const isLabelLike = isBool || (intOk && min >= -1 && max <= 4);
      const rng = intOk ? `${min}…${max}` : `${Number(min.toFixed(3))}…${Number(max.toFixed(3))}`;
      const summary = `${isBool ? 'bool (0/1) · ' : intOk ? 'int · ' : ''}${rng} · ${length.toLocaleString()} samples`;
      out.push({ name: origKey, length, min, max, isInt: intOk, isBool, isLabelLike, hasNeg, summary });
    }
    out.sort((a, b) => (b.isLabelLike ? 1 : 0) - (a.isLabelLike ? 1 : 0) || (b.isBool ? 1 : 0) - (a.isBool ? 1 : 0));
    let defaultVar = '';
    const byLc = new Map(out.map(c => [c.name.toLowerCase(), c]));
    for (const pref of ['labels', 'pred_labels', 'predlabels', 'l', 'is_saccade', 'saccade']) {
      if (byLc.has(pref)) { defaultVar = byLc.get(pref).name; break; }
    }
    if (!defaultVar) {
      const m = out.find(c => c.isLabelLike && !/^(observed|reviewed)$/i.test(c.name)) || out[0];
      if (m) defaultVar = m.name;
    }
    return { vars: out, defaultVar };
  }

  // Extracts a boolean or integer label variable from vars0 into an array of Int8Arrays matching trials
  function extractLabelsFromVars(vars0, trials, varName) {
    if (!vars0 || !trials || !trials.length) return null;
    const vars = lc(vars0);
    let key = varName;
    if (!key) {
      const listed = listLabelVars(vars0, trials);
      key = listed.defaultVar;
    }
    if (!key) return null;
    const v0 = vars[key.toLowerCase()];
    if (!v0) return null;
    const toLabArray = (src, len) => {
      const arr = new Int8Array(len);
      for (let i = 0; i < len; i++) {
        const v = i < src.length ? src[i] : -1;
        if (v === true) arr[i] = 1;
        else if (v === false) arr[i] = 0;
        else if (!(v === v && isFinite(v)) || v < 0) arr[i] = -1;
        else arr[i] = Math.round(v);
      }
      return arr;
    };
    if (v0.cells && v0.cells.length >= trials.length) {
      return trials.map((t, k) => toLabArray(toC(v0.cells[k]).data, t.x.length));
    }
    const L = toC(v0);
    if (!L || !L.data) return null;
    const sh = L.shape || L.dims || [L.data.length];
    if (sh.length === 2 && sh[0] > 1 && sh[1] > 1) {
      const [r, c] = sh;
      if (r === trials.length) {
        return trials.map((t, k) => toLabArray(L.data.subarray(k * c, (k + 1) * c), t.x.length));
      }
      if (c === trials.length) {
        return trials.map((t, k) => {
          const tmp = new Float64Array(t.x.length);
          for (let i = 0; i < t.x.length && i < r; i++) tmp[i] = L.data[i * c + k];
          return toLabArray(tmp, t.x.length);
        });
      }
    }
    let offs = null;
    const offV = vars.trial_offset ? toC(vars.trial_offset) : null;
    const lenV = (vars.trial_lengths || vars.lengths) ? toC(vars.trial_lengths || vars.lengths) : null;
    if (offV && offV.data && offV.data.length >= trials.length + 1 && L.data.length >= offV.data[trials.length]) {
      offs = Array.from(offV.data);
    } else if (lenV && lenV.data && lenV.data.length >= trials.length) {
      offs = [0]; for (let k = 0; k < trials.length; k++) offs.push(offs[offs.length - 1] + Number(lenV.data[k]));
    } else {
      offs = [0]; for (const t of trials) offs.push(offs[offs.length - 1] + t.x.length);
    }
    if (L.data.length < offs[trials.length]) return null;
    return trials.map((t, k) => toLabArray(L.data.subarray(offs[k], offs[k + 1]), t.x.length));
  }

  // returns {trials: [{x, y, valid, name}], fs, labels: [Int8Array|null] or null, predLabels: [Int8Array] or null, prob: [Float32Array] or null, probParams}
  // opts (optional): { labelsVar: string, labelsRole: 'prelabels'|'reviewed', probVar: string }
  function trialsFromVars(vars0, opts) {
    const vars = lc(vars0), out = { trials: [], fs: null, labels: null, predLabels: null, prob: null, probParams: null };
    if (!vars.pred_labels && vars.predlabels) vars.pred_labels = vars.predlabels;
    const names = strList(vars.names);
    if (vars.fs) { const f = toC(vars.fs); const d = f.data; const fv = Number(d && d.length ? d[0] : d); if (isFinite(fv) && fv > 0) out.fs = fv; }
    const num = k => (vars[k.toLowerCase()] ? toC(vars[k.toLowerCase()]) : null);
    const rawVar = k => (k ? vars[k.toLowerCase()] : null);

    const splitByOffsets = (xyv, key) => {
      const off = num('trial_offset'), len = num('trial_lengths') || num('lengths');
      const [n0, c] = xyv.shape;
      if (c !== 2) throw new Error('"' + key + '" must have two columns (x, y).');
      let offs; if (off) offs = Array.from(off.data); else if (len) { offs = [0]; for (const l of len.data) offs.push(offs[offs.length - 1] + l); } else offs = [0, n0];
      const xs = new Float64Array(n0), ys = new Float64Array(n0); for (let i = 0; i < n0; i++) { xs[i] = xyv.data[2 * i]; ys[i] = xyv.data[2 * i + 1]; }
      const trials = [];
      for (let k = 0; k + 1 < offs.length; k++) trials.push(mkTrial(xs.subarray(offs[k], offs[k + 1]), ys.subarray(offs[k], offs[k + 1]), names ? names[k] : undefined));
      return { trials, offs };
    };

    let offs = null, shape2D = null;
    if (vars.trials && vars.trials.cells) {                                       // MAT cell of N x 2 matrices
      out.trials = vars.trials.cells.map((c, k) => { const m = toC(c); const [x, y] = xyOf(m.shape || m.dims, m.data); return mkTrial(x, y, names ? names[k] : undefined); });
    } else if (vars.x && vars.x.cells && vars.y && vars.y.cells) {                // two cell arrays
      out.trials = vars.x.cells.map((c, k) => mkTrial(c.data, vars.y.cells[k].data, names ? names[k] : undefined));
    } else if (vars.xy || vars.eye) {
      const key = vars.xy ? 'xy' : 'eye'; const r = splitByOffsets(num(key), key); out.trials = r.trials; offs = r.offs;
    } else if (vars.trials && vars.trials.shape) {                                // numeric 3-D (n, T, 2) or (n, 2, T)
      const t = num('trials'), [n, a, b] = t.shape;
      if (t.shape.length !== 3) throw new Error('"trials" must be a cell array or a 3-D array.');
      for (let k = 0; k < n; k++) { const sl = t.data.subarray(k * a * b, (k + 1) * a * b); const [x, y] = xyOf([a, b], sl); out.trials.push(mkTrial(x, y, names ? names[k] : undefined)); }
    } else if ((vars.x && vars.y) && ((vars.x.shape || vars.x.dims || []).length === 2) && Math.min(...(vars.x.shape || vars.x.dims)) > 1) {   // 2-D X, Y: trials x time
      const X = num('x'), Y = num('y'), [n, T] = X.shape;
      shape2D = [n, T];
      for (let k = 0; k < n; k++) out.trials.push(mkTrial(X.data.subarray(k * T, (k + 1) * T), Y.data.subarray(k * T, (k + 1) * T), names ? names[k] : undefined));
    } else if (vars.x && vars.y) {                                                // one trial
      out.trials = [mkTrial(num('x').data, num('y').data, names ? names[0] : undefined)];
      offs = [0, out.trials[0].x.length];
    } else {                                                                      // group / dataset names trial_k (T x 2), x_k / y_k
      const keys = Object.keys(vars).filter(k => /^trials?[_/]?\d+$/.test(k)).sort((a, b) => parseInt(a.replace(/\D+/g, '')) - parseInt(b.replace(/\D+/g, '')));
      if (keys.length) out.trials = keys.map((k, i) => { const m = num(k); const [x, y] = xyOf(m.shape, m.data); return mkTrial(x, y, names ? names[i] : k); });
      else {
        const xs = Object.keys(vars).filter(k => /^x_\d+$/.test(k)).sort((a, b) => parseInt(a.slice(2)) - parseInt(b.slice(2)));
        if (!xs.length) throw new Error('No eye traces found. Expected: trials (cell array), X and Y, xy + trial_offset, eye + trial_offset, or trial_0 ... (see the format list in formats.js).');
        out.trials = xs.map((k, i) => mkTrial(num(k).data, num('y_' + k.slice(2)).data, names ? names[i] : k));
      }
    }
    if (!out.trials.length) throw new Error('The file holds no trial.');

    // Extract any per-sample variable (by name) into an array of TypedArrays matching out.trials
    const extractVar = (key, Type) => {
      if (!key) return null;
      const v0 = rawVar(key);
      if (!v0) return null;
      if (v0.cells) return perTrial(v0, null, out.trials, Type);
      const L = toC(v0);
      if (!L || !L.data) return null;
      if (shape2D) {
        const [n, T] = shape2D;
        if (!L.shape || L.shape[0] !== n || L.shape[1] !== T) return null;
        const o = [];
        for (let k = 0; k < n; k++) o.push(Type.from(L.data.subarray(k * T, k * T + out.trials[k].x.length)));
        return o;
      }
      if (offs) return perTrial(L, offs, out.trials, Type);
      return null;
    };

    // Labels / pre-labels selection
    if (opts && opts.labelsVar !== undefined) {
      if (opts.labelsVar && opts.labelsVar !== 'none') {
        const arr = extractVar(opts.labelsVar, Int8Array);
        const role = opts.labelsRole || (/^labels$/i.test(opts.labelsVar) ? 'reviewed' : 'prelabels');
        if (role === 'reviewed') out.labels = arr;
        else out.predLabels = arr;
      }
    } else {
      if (vars.labels) out.labels = extractVar('labels', Int8Array);
      if (vars.pred_labels) out.predLabels = extractVar('pred_labels', Int8Array);
    }

    // Predictive probability selection
    if (opts && opts.probVar !== undefined) {
      if (opts.probVar && opts.probVar !== 'none') {
        out.prob = extractVar(opts.probVar, Float32Array);
      }
    } else {
      if (vars.prob) out.prob = extractVar('prob', Float32Array);
    }

    const pp = {}; for (const k of ['threshold', 'merge_ms', 'min_dur_ms']) if (vars['@' + k] !== undefined) pp[k] = vars['@' + k];
    if (Object.keys(pp).length) out.probParams = pp;
    return out;
  }

  // a per-sample variable (prob, pred_labels) cut like the positions: offsets (offs) of the concatenated layout, or a cell per trial;
  // null when its length does not match the positions
  function perTrial(L, offs, trials, Type) {
    if (!L) return null;
    if (L.cells) { if (L.cells.length < trials.length) return null; return trials.map((t, k) => Type.from(toC(L.cells[k]).data.subarray(0, t.x.length))); }
    if (!offs || L.data.length < offs[offs.length - 1]) return null;
    const out = [];
    for (let k = 0; k + 1 < offs.length; k++) out.push(Type.from(L.data.subarray(offs[k], offs[k + 1]).subarray(0, trials[k].x.length)));
    return out;
  }

  function labelsByOffsets(L, offs, trials) {
    return perTrial(L, offs, trials, Int8Array);
  }

  // ---------------------------------------------------------------- top level
  function extOf(name) { const m = /\.([A-Za-z0-9]+)$/.exec(name || ''); return m ? m[1].toLowerCase() : ''; }

  // name: file name, buf: ArrayBuffer, h5: h5wasm module or an async function returning it (only for HDF5)
  async function readTrialsFile(name, buf, h5, opts) {
    const ext = extOf(name);
    let vars;
    if (ext === 'npz') vars = await readNpz(buf);
    else if (ext === 'mat') {
      try { vars = await readMat5(buf); }
      catch (e) { if (!e.hdf5) throw e; vars = await readH5(buf, typeof h5 === 'function' ? await h5() : h5); }
    } else if (ext === 'h5' || ext === 'hdf5' || ext === 'hdf') vars = await readH5(buf, typeof h5 === 'function' ? await h5() : h5);
    else throw new Error('Unknown file type: ' + name);
    return trialsFromVars(vars, opts);
  }

  // the variables of a .mat / .npz / .h5 file as they are (no layout expected): input of the column explorer (import.js tablesFromVars)
  async function readVarsFile(name, buf, h5) {
    const ext = extOf(name), getH5 = async () => (typeof h5 === 'function' ? await h5() : h5);
    if (ext === 'npz') return readNpz(buf);
    if (ext === 'mat') {
      try { return await readMat5(buf); }
      catch (e) { if (!e.hdf5) throw e; return readH5(buf, await getH5(), { all: true }); }
    }
    if (ext === 'h5' || ext === 'hdf5' || ext === 'hdf') return readH5(buf, await getH5(), { all: true });
    throw new Error('Unknown file type: ' + name);
  }

  // ---------------------------------------------------------------- writers
  // trials: [{x, y}] (raw or smoothed positions), labels: [Int8Array] (-1 = not reviewed), names: [string]|null, fs
  function layoutOf(trials, labels) {
    const offs = [0]; for (const t of trials) offs.push(offs[offs.length - 1] + t.x.length);
    const total = offs[offs.length - 1], xy = new Float32Array(total * 2), lab = labels ? new Int8Array(total) : null;
    trials.forEach((t, k) => { for (let i = 0; i < t.x.length; i++) { xy[2 * (offs[k] + i)] = t.x[i]; xy[2 * (offs[k] + i) + 1] = t.y[i]; if (lab) lab[offs[k] + i] = labels[k][i]; } });
    return { offs, total, xy, lab };
  }

  function writeNpzFile(trials, labels, names, fs) {
    const L = layoutOf(trials, labels);
    const arrays = [{ name: 'xy', type: 'f4', shape: [L.total, 2], data: L.xy }, { name: 'trial_offset', type: 'i4', shape: [L.offs.length], data: Int32Array.from(L.offs) },
                    { name: 'fs', type: 'f8', shape: [], data: Float64Array.of(fs) }];
    if (L.lab) {
      arrays.push({ name: 'labels', type: 'i1', shape: [L.total], data: L.lab });
      const obs = new Uint8Array(L.total);
      for (let i = 0; i < L.total; i++) obs[i] = L.lab[i] >= 0 ? 1 : 0;
      arrays.push({ name: 'observed', type: 'u1', shape: [L.total], data: obs });
      arrays.push({ name: 'reviewed', type: 'u1', shape: [L.total], data: obs });
    }
    if (names && names.length === trials.length) arrays.push({ name: 'names', strings: names });
    return buildNpz(arrays);
  }

  function writeMatFile(trials, labels, names, fs) {
    const vars = [{ name: 'trials', cell: trials.map(t => { const n = t.x.length, d = new Float64Array(2 * n); d.set(t.x, 0); d.set(t.y, n); return { dims: [n, 2], data: d }; }) },   // column-major N x 2
                  { name: 'fs', dims: [1, 1], data: Float64Array.of(fs) }];
    if (labels) {
      vars.push({ name: 'labels', cell: labels.map(l => ({ dims: [l.length, 1], data: Float64Array.from(l) })) });
      vars.push({ name: 'observed', cell: labels.map(l => ({ dims: [l.length, 1], data: Float64Array.from(l, v => v >= 0 ? 1 : 0) })) });
      vars.push({ name: 'reviewed', cell: labels.map(l => ({ dims: [l.length, 1], data: Float64Array.from(l, v => v >= 0 ? 1 : 0) })) });
    }
    if (names && names.length === trials.length) vars.push({ name: 'names', strings: names });
    return buildMat5(vars);
  }

  async function writeH5File(trials, labels, names, fs, h5) {
    const L = layoutOf(trials, labels);
    const vars = [{ name: 'xy', type: 'f4', shape: [L.total, 2], data: L.xy }, { name: 'trial_offset', type: 'i4', shape: [L.offs.length], data: Int32Array.from(L.offs) }];
    if (L.lab) {
      vars.push({ name: 'labels', type: 'i1', shape: [L.total], data: L.lab });
      const obs = new Uint8Array(L.total);
      for (let i = 0; i < L.total; i++) obs[i] = L.lab[i] >= 0 ? 1 : 0;
      vars.push({ name: 'observed', type: 'u1', shape: [L.total], data: obs });
      vars.push({ name: 'reviewed', type: 'u1', shape: [L.total], data: obs });
    }
    if (names && names.length === trials.length) vars.push({ name: 'names', strings: names });
    return buildH5(vars, { fs }, typeof h5 === 'function' ? await h5() : h5);
  }

  return { readTrialsFile, readVarsFile, readNpz, readMat5, readH5, readZip, parseNpy, inspectVars, listLabelVars, extractLabelsFromVars, trialsFromVars, writeNpzFile, writeMatFile, writeH5File, buildNpz, buildMat5, buildH5, buildZip, crc32, extOf };
}));
