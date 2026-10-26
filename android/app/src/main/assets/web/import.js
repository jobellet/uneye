// import.js - reads a text export of eye tracking (CSV / TSV / space separated, ARFF, EyeLink .asc) into a table of named columns,
// finds out which columns are X, Y and time, and turns the chosen columns into trials for the labeler.
//
// Nothing is assumed about the file: the delimiter and the header are detected, every column is kept (numbers as Float64, text as codes
// with a dictionary), and suggest() only proposes a mapping that the explorer window (import_ui.js) lets the user change.
//
// Time: the unit (s, ms, us or a plain sample index) and the sampling rate are measured from the time column (mean step, so that
// timestamps rounded to the millisecond or repeated at 2 kHz give the right rate). A signal whose time stamps are not on a regular
// grid (dropped frames, jitter) is resampled to a uniform grid; stretches without data stay invalid (NaN).
//
// Works in the browser (global UnEyeImport) and in Node >= 18 (module.exports), the same code for both.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.UnEyeImport = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const TYPICAL_RATES = [25, 30, 50, 60, 75, 90, 100, 120, 125, 144, 150, 200, 240, 250, 300, 333, 350, 500, 600, 1000, 1200, 1250, 2000];
  const MAX_DICT = 5000;                         // distinct strings kept per text column
  const PAGE = 1 << 15;

  // ---------------------------------------------------------------- growable columns
  class ColBuilder {
    constructor(name, text) {
      this.name = name; this.n = 0; this.pages = []; this.cur = null;
      this.text = !!text; this.map = text ? new Map() : null; this.labels = text ? [] : null;
    }
    push(v) {
      const o = this.n & (PAGE - 1);
      if (o === 0) { this.cur = new Float64Array(PAGE); this.pages.push(this.cur); }
      this.cur[o] = v; this.n++;
    }
    pushText(s) {
      let c = this.map.get(s);
      if (c === undefined) {
        if (this.labels.length >= MAX_DICT) { this.push(NaN); return; }
        c = this.labels.length; this.labels.push(s); this.map.set(s, c);
      }
      this.push(c);
    }
    finish() {
      const out = new Float64Array(this.n);
      for (let p = 0; p < this.pages.length; p++) out.set(p === this.pages.length - 1 ? this.pages[p].subarray(0, this.n - p * PAGE) : this.pages[p], p * PAGE);
      const col = { name: this.name, data: out };
      if (this.text) col.labels = this.labels;
      return col;
    }
  }

  // ---------------------------------------------------------------- text helpers
  // source: a string or an (async) iterable of string chunks. fn(line) may return false to stop.
  async function eachLine(source, fn) {
    if (typeof source === 'string') source = [source];
    let carry = '', first = true;
    for await (let chunk of source) {
      if (first) { if (chunk.charCodeAt(0) === 0xFEFF) chunk = chunk.slice(1); first = false; }
      const s = carry.length ? carry + chunk : chunk;
      let start = 0;
      for (;;) {
        const nl = s.indexOf('\n', start);
        if (nl < 0) break;
        let end = nl;
        if (end > start && s.charCodeAt(end - 1) === 13) end--;
        if (fn(s.substring(start, end)) === false) return;
        start = nl + 1;
      }
      carry = s.substring(start);
    }
    if (carry.length) fn(carry.charCodeAt(carry.length - 1) === 13 ? carry.slice(0, -1) : carry);
  }

  async function* chunksOfFile(file, size) {
    size = size || (8 << 20);
    const dec = new TextDecoder('utf-8');
    for (let o = 0; o < file.size; o += size) yield dec.decode(await file.slice(o, o + size).arrayBuffer(), { stream: true });
    yield dec.decode();
  }

  function splitQuoted(line, d) {
    const out = []; let cur = '', q = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (q) { if (ch === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += ch; }
      else if (ch === '"') q = true;
      else if (ch === d) { out.push(cur); cur = ''; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  }
  function splitLine(line, d) {
    if (d === 'ws') return line.trim().split(/\s+/);
    if (line.indexOf('"') >= 0) return splitQuoted(line, d);
    return line.split(d);
  }

  const MISSING = /^(|nan|na|n\/a|null|none|\.|-|--|\?|inf|-inf)$/i;
  function parseNum(tok, decComma) {
    if (decComma && tok.indexOf(',') >= 0) tok = tok.replace(',', '.');
    const v = +tok;                                  // '' and ' ' give 0, handled by the caller through MISSING
    return v;
  }
  function numLike(tok, decComma) {
    const t = tok.trim();
    if (MISSING.test(t)) return true;
    return !isNaN(parseNum(t, decComma));
  }
  function numOrNaN(tok, decComma) {
    const t = tok.trim();
    if (t === '' || MISSING.test(t)) return NaN;
    const v = parseNum(t, decComma);
    return v;                                        // NaN when not a number
  }

  const DELIMS = ['\t', ';', ',', '|', 'ws'];
  function chooseDelimiter(lines) {
    const probe = lines.slice(0, 100);
    for (const d of DELIMS) {
      const counts = new Map();
      for (const l of probe) { const k = splitLine(l, d).length; counts.set(k, (counts.get(k) || 0) + 1); }
      let best = 0, bestK = 1;
      for (const [k, c] of counts) if (c > best || (c === best && k > bestK)) { best = c; bestK = k; }
      if (bestK >= 2 && best >= 0.9 * probe.length) return { delimiter: d, nFields: bestK };
    }
    return { delimiter: null, nFields: 1 };
  }

  function uniqueNames(names) {
    const seen = new Map();
    return names.map((n, i) => {
      let b = (n || '').trim().replace(/^["']|["']$/g, '') || ('col' + (i + 1));
      const c = seen.get(b) || 0; seen.set(b, c + 1);
      return c ? b + '_' + (c + 1) : b;
    });
  }

  // decides delimiter, header and decimal comma from the first lines; shared by the explorer and the quick "is this a plain file?" test
  function sniffLines(lines) {
    const { delimiter, nFields } = chooseDelimiter(lines);
    const rows = lines.slice(0, 60).map(l => delimiter ? splitLine(l, delimiter) : [l]);
    let decComma = false;
    if (delimiter && delimiter !== ',') {
      let hit = 0, tot = 0;
      for (const r of rows.slice(1)) for (const f of r) { const t = f.trim(); if (t) { tot++; if (/^-?\d+,\d+([eE][+-]?\d+)?$/.test(t)) hit++; } }
      decComma = tot > 0 && hit > 0.3 * tot;
    }
    const nonNum = (r) => r.reduce((a, f) => a + (numLike(f, decComma) ? 0 : 1), 0);
    let header = false;
    if (rows.length >= 1) {
      const n0 = nonNum(rows[0]);
      const n1 = rows.length > 1 ? nonNum(rows[1]) : 0;
      const ident = rows[0].every(f => /^[A-Za-z_"'][\w .()%\-\/'"]*$/.test(f.trim()));
      header = n0 > 0 && (n0 > n1 || ident);
    }
    const dataRows = header ? rows.slice(1) : rows;
    const textCol = [];
    for (let j = 0; j < nFields; j++) {
      let num = 0, tot = 0;
      for (const r of dataRows) { if (j < r.length) { tot++; if (numLike(r[j], decComma)) num++; } }
      textCol.push(tot > 0 && num < 0.8 * tot);
    }
    const names = uniqueNames(header ? rows[0].slice(0, nFields) : Array.from({ length: nFields }, (_, i) => 'col' + (i + 1)));
    return { delimiter, nFields, header, decComma, textCol, names };
  }

  const isCommentLine = (l) => { const c = l.charCodeAt(0); return c === 35 || c === 37 || (c === 47 && l.charCodeAt(1) === 47); };   // # % //

  // Quick look at the head of a text file: is it the plain numeric layout the labeler has always read (no header, 1-2 columns, or
  // many columns = trials x time)? Those keep the old fast path, everything else goes to the explorer.
  function sniffHead(text, name) {
    const ext = ((name || '').split('.').pop() || '').toLowerCase();
    if (ext === 'arff' || ext === 'asc') return { kind: ext, plain: false };
    if (/^\s*@relation/im.test(text.slice(0, 4000))) return { kind: 'arff', plain: false };
    if (/^MSG\t|^\*\* CONVERTED FROM|EYELINK/m.test(text.slice(0, 8000))) return { kind: 'asc', plain: false };
    const lines = text.split(/\r?\n/).filter(l => l.trim() && !isCommentLine(l)).slice(0, 60);
    if (!lines.length) return { kind: 'delimited', plain: true };
    const s = sniffLines(lines);
    const plain = !s.header && !s.decComma && !s.textCol.some(Boolean) && (s.nFields <= 2 || s.nFields >= 100);
    return { kind: 'delimited', plain, sniff: s };
  }

  // ---------------------------------------------------------------- delimited text
  async function parseDelimited(source, name) {
    const sniffBuf = [];
    let st = null, cols = null, rowCount = 0, sniffed = false;
    const info = { kind: 'delimited', comments: 0 };

    function decide() {
      sniffed = true;
      if (!sniffBuf.length) { cols = []; st = { names: [], nFields: 0, header: false, textCol: [] }; return; }
      st = sniffLines(sniffBuf);
      cols = st.names.map((n, j) => new ColBuilder(n, st.textCol[j]));
      info.delimiter = st.delimiter; info.header = st.header; info.decimalComma = st.decComma;
      const rows = st.header ? sniffBuf.slice(1) : sniffBuf;
      for (const l of rows) addRow(l);
    }
    function addRow(l) {
      const f = st.delimiter ? splitLine(l, st.delimiter) : [l];
      if (f.length === 1 && st.nFields > 1 && !f[0].trim()) return;
      for (let j = 0; j < st.nFields; j++) {
        const c = cols[j], tok = j < f.length ? f[j] : '';
        if (c.text) c.pushText(tok.trim()); else c.push(numOrNaN(tok, st.decComma));
      }
      rowCount++;
    }
    await eachLine(source, (line) => {
      if (!line.trim()) return;
      if (isCommentLine(line)) { info.comments++; return; }
      if (!sniffed) { sniffBuf.push(line); if (sniffBuf.length >= 200) decide(); return; }
      addRow(line);
    });
    if (!sniffed) decide();
    return { kind: 'delimited', name, nRows: rowCount, cols: cols.map(c => c.finish()), info };
  }

  // ---------------------------------------------------------------- ARFF
  function splitArffRow(l) {
    if (l.indexOf("'") < 0 && l.indexOf('"') < 0) return l.split(',');
    const out = []; let cur = '', q = '';
    for (let i = 0; i < l.length; i++) {
      const ch = l[i];
      if (q) { if (ch === q) q = ''; else cur += ch; }
      else if (ch === '"' || ch === "'") q = ch;
      else if (ch === ',') { out.push(cur); cur = ''; }
      else cur += ch;
    }
    out.push(cur);
    return out;
  }
  async function parseArff(source, name) {
    const attrs = []; let inData = false, rowCount = 0, cols = null, relation = '';
    await eachLine(source, (line) => {
      const l = line.trim();
      if (!l || l[0] === '%') return;
      if (!inData) {
        if (/^@relation/i.test(l)) { relation = l.replace(/^@relation\s+/i, '').replace(/^['"]|['"]$/g, ''); return; }
        if (/^@attribute/i.test(l)) {
          const m = /^@attribute\s+(?:'([^']*)'|"([^"]*)"|(\S+))\s+(.*)$/i.exec(l);
          if (!m) return;
          const aname = m[1] !== undefined ? m[1] : (m[2] !== undefined ? m[2] : m[3]);
          const type = m[4].trim();
          if (type[0] === '{') {
            const labels = splitArffRow(type.replace(/^\{|\}$/g, '')).map(s => s.trim());
            attrs.push({ name: aname, nominal: labels });
          } else attrs.push({ name: aname, numeric: /^(integer|numeric|real)/i.test(type) });
          return;
        }
        if (/^@data/i.test(l)) {
          inData = true;
          cols = attrs.map(a => new ColBuilder(a.name, !!a.nominal || a.numeric === false));
          cols.forEach((c, j) => { if (attrs[j].nominal) { c.labels = attrs[j].nominal.slice(); attrs[j].nominal.forEach((s, k) => c.map.set(s, k)); } });
        }
        return;
      }
      if (l[0] === '{') throw new Error('Sparse ARFF files are not supported.');
      const f = splitArffRow(l);
      for (let j = 0; j < cols.length; j++) {
        const c = cols[j], tok = j < f.length ? f[j].trim() : '?';
        if (c.text) { if (tok === '?') c.push(NaN); else { const k = c.map.get(tok); if (k === undefined) c.pushText(tok); else c.push(k); } }
        else c.push(tok === '?' || tok === '' ? NaN : +tok);
      }
      rowCount++;
    });
    if (!cols) throw new Error('This ARFF file has no @data section.');
    return { kind: 'arff', name, nRows: rowCount, cols: cols.map(c => c.finish()), info: { relation, delimiter: ',', header: true } };
  }

  // ---------------------------------------------------------------- EyeLink ASC
  // samples: "time x y pupil [flags]" (one eye) or "time xL yL pL xR yR pR [flags]"; "." = missing. START ... END = one recording block (a trial).
  async function parseAsc(source, name) {
    const info = { rate: null, eyes: 0, blocks: 0, messages: new Map(), events: { fixations: 0, saccades: 0, blinks: 0 }, headerLines: [] };
    let cols = null, timeCol = null, blockCol = null, eyes = 1, inBlock = false, blockIdx = -1, pendingName = null, nSamples = 0;
    const blockNames = [];
    function ensureCols(nv) {
      const base = eyes === 2 ? ['xL', 'yL', 'pupilL', 'xR', 'yR', 'pupilR'] : ['x', 'y', 'pupil'];
      const names = ['time'];
      for (let k = 0; k < nv; k++) names.push(k < base.length ? base[k] : 'extra' + (k - base.length + 1));
      cols = names.map(n => new ColBuilder(n, false));
      blockCol = new ColBuilder('block', true);
      blockCol.labels = blockNames;                       // filled in while reading; the codes are the block numbers
      timeCol = cols[0];
    }
    await eachLine(source, (line) => {
      const c0 = line.charCodeAt(0);
      if (c0 >= 48 && c0 <= 57) {                        // a sample
        if (!inBlock) return;
        let f = line.split('\t');
        if (f.length < 3) f = line.trim().split(/\s+/);
        const t = +f[0];
        const vals = [];
        for (let k = 1; k < f.length; k++) {
          const tk = f[k].trim();
          if (tk === '.') { vals.push(NaN); continue; }
          const v = +tk;
          if (tk === '' || isNaN(v)) break;                // flags such as "..." or "I.C" close the line
          vals.push(v);
        }
        if (!cols) ensureCols(vals.length);
        timeCol.push(t);
        for (let k = 0; k < cols.length - 1; k++) cols[k + 1].push(k < vals.length ? vals[k] : NaN);
        blockCol.push(blockIdx);
        nSamples++;
        return;
      }
      if (line.startsWith('MSG')) {
        const m = /^MSG\s+(-?\d+(?:\.\d+)?)\s+(.*)$/.exec(line);
        if (!m) return;
        const key = m[2].replace(/-?\d+(?:\.\d+)?/g, '#').slice(0, 40);
        info.messages.set(key, (info.messages.get(key) || 0) + 1);
        const tid = /^TRIALID\s+(\S+(?:\s+\S+)*)/.exec(m[2]);
        if (tid) { const nm = 'trial ' + tid[1]; if (inBlock) { if (!blockNames[blockIdx]) blockNames[blockIdx] = nm; } else pendingName = nm; }
        return;
      }
      if (line.startsWith('START')) {
        const u = line.toUpperCase();
        eyes = (u.indexOf('LEFT') >= 0 && u.indexOf('RIGHT') >= 0) ? 2 : 1;
        info.eyes = eyes;
        inBlock = true; blockIdx = info.blocks++;
        blockNames[blockIdx] = pendingName || ''; pendingName = null;
        return;
      }
      if (line.startsWith('END')) { inBlock = false; return; }
      if (line.startsWith('SAMPLES')) { const m = /RATE\s+([\d.]+)/.exec(line); if (m) info.rate = parseFloat(m[1]); return; }
      if (line.startsWith('EFIX')) { info.events.fixations++; return; }
      if (line.startsWith('ESACC')) { info.events.saccades++; return; }
      if (line.startsWith('EBLINK')) { info.events.blinks++; return; }
      if (line.startsWith('**') && info.headerLines.length < 20) info.headerLines.push(line.replace(/^\*+\s*/, ''));
    });
    if (!cols) throw new Error('No samples found in this .asc file (was it converted with samples included?).');
    const out = cols.map(c => c.finish());
    for (let b = 0; b < blockNames.length; b++) if (!blockNames[b]) blockNames[b] = 'block ' + (b + 1);
    const bc = blockCol.finish(); bc.labels = blockNames;
    out.push(bc);
    info.delimiter = 'tab'; info.header = true;
    info.messages = Array.from(info.messages, ([text, count]) => ({ text, count })).sort((a, b) => b.count - a.count).slice(0, 40);
    return { kind: 'asc', name, nRows: nSamples, cols: out, info, timeUnit: 'ms' };
  }

  // ---------------------------------------------------------------- entry points
  function kindOf(name, head) {
    const ext = ((name || '').split('.').pop() || '').toLowerCase();
    if (ext === 'arff') return 'arff';
    if (ext === 'asc') return 'asc';
    if (/^\s*@relation/im.test(head || '')) return 'arff';
    if (/^MSG\t|^\*\* CONVERTED FROM/m.test(head || '')) return 'asc';
    return 'delimited';
  }

  // source: string or async iterable of chunks (chunksOfFile(file) in the browser, fs.createReadStream(path, 'utf8') in Node)
  async function parseSource(source, name, headText) {
    const k = kindOf(name, headText);
    const t = k === 'arff' ? await parseArff(source, name) : k === 'asc' ? await parseAsc(source, name) : await parseDelimited(source, name);
    addDerived(t);
    return t;
  }
  async function parseFile(file) {
    const head = await file.slice(0, 8192).text();
    return parseSource(chunksOfFile(file), file.name, head);
  }
  function parseString(text, name) { return parseSource(text, name, text.slice(0, 8192)); }

  // left / right eye pairs (lx / rx, xL / xR, gaze_left_x / gaze_right_x ...) get a derived "mean of the two eyes" column
  function rightPartner(name) {
    let m = /^l([xy])$/i.exec(name); if (m) return 'r' + m[1];
    m = /^([xy])l$/i.exec(name); if (m) return m[1] + 'r';
    const toks = name.split(/[_\s.\-]+/);
    const i = toks.findIndex(t => /^(l|left)$/i.test(t));
    if (i < 0) return null;
    const t = toks[i];
    toks[i] = t === t.toUpperCase() ? (t.length === 1 ? 'R' : 'RIGHT') : (t[0] === t[0].toUpperCase() ? (t.length === 1 ? 'R' : 'Right') : (t.length === 1 ? 'r' : 'right'));
    return toks.join('_');
  }
  const normKey = (s) => s.toLowerCase().replace(/[_\s.\-]+/g, '_');
  function addDerived(table) {
    const byKey = new Map(table.cols.map((c, i) => [normKey(c.name), i]));
    const add = [];
    table.cols.forEach((c, i) => {
      if (c.labels) return;
      const partner = rightPartner(c.name);
      if (!partner) return;
      const j = byKey.get(normKey(partner));
      if (j === undefined || table.cols[j].labels) return;
      const stripped = normKey(c.name).replace(/(^|_)(l|left)(?=_|$)/, '').replace(/^l(?=[xy]$)/, '').replace(/(?<=^[xy])l$/, '').replace(/^_|_$/g, '');
      if (!/(^|_)[xy]$|^[xy]/.test(stripped) || /pupil/.test(stripped)) return;
      const a = c.data, b = table.cols[j].data, out = new Float64Array(table.nRows);
      for (let k = 0; k < out.length; k++) {
        const u = a[k], v = b[k];
        out[k] = (u === u && v === v) ? (u + v) / 2 : (u === u ? u : v);
      }
      add.push({ name: (stripped || c.name) + ' (mean L,R)', data: out, derived: true });
    });
    for (const c of add) table.cols.push(c);
  }

  // ---------------------------------------------------------------- MAT / NPZ / HDF5 variables -> tables
  // vars: what UnEyeFormats.readVarsFile returns. Every numeric array becomes columns; arrays with the same number of rows form one table
  // (a matrix: its long side is time; names come from a text array of the right length, e.g. ZuCo "colheader"; structs are flattened as
  // ETdata.pos; a cell array of matrices becomes one table with an extra column holding the cell number).
  const RATE_NAME = /(^|[._\/])(fs|srate|samp\w*(freq|rate)|sampling\w*|rate|hz)$/i;

  function leavesOf(vars) {
    const leaves = [], strings = [];
    const walk = (v, name, depth) => {
      if (!v || depth > 6) return;
      if (v.cls === 'struct') {
        const many = v.elems.length > 1;
        v.elems.slice(0, 50).forEach((e, i) => { for (const f of v.fields) walk(e[f], name + (many ? '(' + (i + 1) + ')' : '') + '.' + f, depth + 1); });
      } else if (v.cls === 'cell') {
        const cs = v.cells || [];
        if (cs.length && cs.every(c => c && c.cls === 'char')) strings.push({ name, list: cs.map(c => c.strings.join('')) });
        else if (cs.length && cs.every(c => c && c.cls === 'numeric' && c.dims.length === 2)) leaves.push({ name, cells: cs.slice(0, 20000) });
      } else if (v.cls === 'char') strings.push({ name, list: v.strings });
      else if (v.cls === 'numeric') { if (v.dims.length === 2) leaves.push({ name, dims: v.dims, data: v.data, order: 'F' }); }
      else if (v.strings) strings.push({ name, list: v.strings });
      else if (v.data !== undefined && v.shape) {                           // npz / h5: C order
        if (typeof v.data === 'number') leaves.push({ name, dims: [1, 1], data: Float64Array.of(v.data), order: 'C' });
        else if (ArrayBuffer.isView(v.data) && v.shape.length <= 2) leaves.push({ name, dims: v.shape.length === 2 ? v.shape : [v.shape[0] === undefined ? 1 : v.shape[0], 1], data: v.data, order: 'C' });
        else if (Array.isArray(v.data) && typeof v.data[0] === 'string') strings.push({ name, list: v.data });
      }
    };
    for (const k of Object.keys(vars)) { if (k[0] === '@') continue; walk(vars[k], k, 0); }
    return { leaves, strings };
  }

  // a numeric leaf as {n rows, columns: [Float64Array]}
  function columnsOfMatrix(dims, data, order) {
    const r = dims[0], c = dims[1], at = order === 'F' ? (i, j) => data[j * r + i] : (i, j) => data[i * c + j];
    if (r === 1 || c === 1) return { n: r * c, cols: [Float64Array.from(data)] };
    const out = [];
    if (r >= c) { for (let j = 0; j < c; j++) { const a = new Float64Array(r); for (let i = 0; i < r; i++) a[i] = at(i, j); out.push(a); } return { n: r, cols: out }; }
    for (let j = 0; j < r; j++) { const a = new Float64Array(c); for (let i = 0; i < c; i++) a[i] = at(j, i); out.push(a); }
    return { n: c, cols: out };
  }

  function tablesFromVars(vars, fileName, kind) {
    const { leaves, strings } = leavesOf(vars);
    let rate = null;
    const pp = {};
    for (const k of ['threshold', 'merge_ms', 'min_dur_ms']) if (vars && vars['@' + k] !== undefined) pp[k] = vars['@' + k];
    const probParams = Object.keys(pp).length ? pp : null;
    const groups = new Map();
    const add = (n, name, data, labels, fromOffsets) => { if (!groups.has(n)) groups.set(n, []); groups.get(n).push({ name, data, labels, fromOffsets }); };
    const headerFor = (leafName, k) => {
      const ok = strings.filter(s => s.list.length === k);
      if (!ok.length) return null;
      const stem = leafName.split(/[.(]/)[0].toLowerCase();
      const pref = ok.find(s => /head|colname|columns|labels|names|channels/i.test(s.name)) || ok.find(s => s.name.toLowerCase().indexOf(stem) >= 0) || ok[0];
      return uniqueNames(pref.list);
    };
    const offsetLeaves = [];
    for (const lf of leaves) {
      if (lf.cells) {                                                 // cell array of matrices: one table, the cell number as a column
        const parts = lf.cells.map(cl => columnsOfMatrix(cl.dims, cl.data, 'F')).filter(p => p.n >= 2);
        if (!parts.length) continue;
        const k = parts[0].cols.length;
        if (!parts.every(p => p.cols.length === k)) continue;
        const total = parts.reduce((a, p) => a + p.n, 0), names = headerFor(lf.name, k);
        const idx = new Float64Array(total);
        for (let j = 0; j < k; j++) {
          const col = new Float64Array(total); let o = 0;
          parts.forEach((p, ci) => { col.set(p.cols[j], o); if (j === 0) idx.fill(ci + 1, o, o + p.n); o += p.n; });
          add(total, names ? names[j] : lf.name + '{}(:,' + (j + 1) + ')', col);
        }
        add(total, lf.name + ' (cell number)', idx);
        continue;
      }
      const m = columnsOfMatrix(lf.dims, lf.data, lf.order);
      if (m.n === 1) { if (RATE_NAME.test(lf.name) && m.cols[0][0] >= 10 && m.cols[0][0] <= 5000) rate = m.cols[0][0]; continue; }
      if (m.cols.length === 1 && /^(trial_?offsets?|(trial_?)?lengths)$/i.test(lf.name)) {
        offsetLeaves.push({ name: lf.name, col: m.cols[0] });
        continue;
      }
      const names = m.cols.length > 1 ? headerFor(lf.name, m.cols.length) : null;
      m.cols.forEach((col, j) => add(m.n, names ? names[j] : (m.cols.length === 1 ? lf.name : lf.name + '(:,' + (j + 1) + ')'), col));
    }
    // Convert trial_offset or trial_lengths into a per-sample 'trial' column on the matching table
    for (const ol of offsetLeaves) {
      let offs = null;
      if (/offset/i.test(ol.name)) {
        if (ol.col.length >= 2 && ol.col[0] === 0) offs = ol.col;
      } else {
        offs = new Float64Array(ol.col.length + 1);
        for (let i = 0; i < ol.col.length; i++) offs[i + 1] = offs[i] + ol.col[i];
      }
      const total = offs ? Math.round(offs[offs.length - 1]) : 0;
      if (offs && total >= 20 && groups.has(total)) {
        const nTrials = offs.length - 1;
        const trNamesObj = strings.find(s => s.list.length === nTrials && /name|trial/i.test(s.name)) || strings.find(s => s.list.length === nTrials);
        const trLabels = trNamesObj ? trNamesObj.list : undefined;
        const trCol = new Float64Array(total);
        for (let k = 0; k < nTrials; k++) {
          const s = Math.max(0, Math.min(total, Math.round(offs[k])));
          const e = Math.max(s, Math.min(total, Math.round(offs[k + 1])));
          trCol.fill(trLabels ? k : k + 1, s, e);
        }
        add(total, 'trial', trCol, trLabels, true);
      } else {
        add(ol.col.length, ol.name, ol.col);
      }
    }
    const tables = [];
    for (const [n, cols] of groups) {
      if (n < 20) continue;
      const named = uniqueNames(cols.map(c => c.name));
      const table = {
        kind: kind || 'mat',
        name: fileName,
        title: named.slice(0, 3).join(', ') + (named.length > 3 ? ', …' : ''),
        nRows: n,
        cols: named.map((nm, i) => {
          const c = { name: nm, data: cols[i].data };
          if (cols[i].labels) c.labels = cols[i].labels;
          if (cols[i].fromOffsets) c.fromOffsets = true;
          return c;
        }),
        info: { rate, probParams, delimiter: null, header: true },
      };
      addDerived(table);
      tables.push(table);
    }
    tables.sort((a, b) => Math.min(b.cols.length, 8) * b.nRows - Math.min(a.cols.length, 8) * a.nRows);
    return tables.filter(t => t.cols.length >= 2).concat(tables.filter(t => t.cols.length < 2)).slice(0, 12);
  }

  // ---------------------------------------------------------------- statistics per column
  function describeColumns(table) {
    const n = table.nRows;
    return table.cols.map((c) => {
      const d = c.data;
      let finite = 0, min = Infinity, max = -Infinity, ints = 0, nonDec = 0, steps = 0;
      let prev = NaN;
      const stepSample = [];
      const stride = Math.max(1, Math.floor(n / 20000));
      for (let i = 0; i < n; i++) {
        const v = d[i];
        if (v === v && v !== Infinity && v !== -Infinity) {
          finite++;
          if (v < min) min = v; if (v > max) max = v;
          if (v === Math.floor(v)) ints++;
          if (prev === prev) { steps++; if (v >= prev) nonDec++; if (i % stride === 0 && stepSample.length < 20000) stepSample.push(v - prev); }
          prev = v;
        }
      }
      let distinct = 0;
      if (c.labels) distinct = c.labels.length;
      else { const seen = new Set(); for (let i = 0; i < n && seen.size < 1000; i += stride) seen.add(d[i]); distinct = seen.size; }
      stepSample.sort((p, q) => p - q);
      return {
        name: c.name, n, finite, nanFrac: n ? 1 - finite / n : 1, min: finite ? min : NaN, max: finite ? max : NaN,
        isInt: finite > 0 && ints === finite, monotonic: steps ? nonDec / steps : 0,
        medianStep: stepSample.length ? stepSample[stepSample.length >> 1] : NaN, distinct, nominal: !!c.labels, derived: !!c.derived,
      };
    });
  }

  // ---------------------------------------------------------------- time axis
  function nearestTypical(r) {
    let best = TYPICAL_RATES[0], bd = Infinity;
    for (const t of TYPICAL_RATES) { const dd = Math.abs(Math.log(r / t)); if (dd < bd) { bd = dd; best = t; } }
    return { rate: best, dist: bd };
  }
  const UNIT_SCALE = { s: 1, ms: 1e-3, us: 1e-6 };

  // data: time column; hintUnit/hintRate: from the column name or the file header. Returns the unit, rate, regularity and gaps.
  function analyzeTime(data, n, opts) {
    opts = opts || {};
    const diffs = [];
    const stride = Math.max(1, Math.floor(n / 50000));
    let prev = NaN, back = 0;
    for (let i = 0; i < n; i++) {
      const v = data[i];
      if (v !== v) continue;
      if (prev === prev && i % stride === 0) diffs.push(v - prev);
      if (prev === prev && v < prev) back++;
      prev = v;
    }
    const pos = diffs.filter(d => d > 0).sort((a, b) => a - b);
    if (pos.length < 2) return { ok: false, reason: 'no usable time steps' };
    const med = pos[pos.length >> 1];
    let sum = 0, cnt = 0;
    for (const d of diffs) if (d >= 0 && d <= 1.6 * med) { sum += d; cnt++; }
    const mu = sum / cnt;                                    // mean step of the regular part (repeated stamps count as 0)
    const intIdx = opts.integerSteps && Math.abs(mu - 1) < 1e-9 && med === 1;
    let unit = null, rate = null;
    if (intIdx && opts.indexName) { unit = 'samples'; rate = opts.hintRate || null; }
    else {
      const cand = [];
      for (const u of ['s', 'ms', 'us']) {
        const r = 1 / (mu * UNIT_SCALE[u]);
        if (r >= 10 && r <= 4000) cand.push({ u, r, near: nearestTypical(r).dist });
      }
      if (opts.hintUnit && cand.find(c => c.u === opts.hintUnit)) unit = opts.hintUnit;
      else if (cand.length) unit = cand.sort((a, b) => a.near - b.near)[0].u;
      if (unit) {
        const r = 1 / (mu * UNIT_SCALE[unit]); const nt = nearestTypical(r);
        rate = nt.dist < 0.03 ? nt.rate : Math.round(r * 10) / 10;
      }
    }
    if (!unit) return { ok: false, reason: 'the time steps (' + mu.toPrecision(4) + ') do not look like seconds, milliseconds or microseconds' };
    const out = { ok: true, unit, rate, meanStep: mu, medianStep: med, backwards: back };
    if (unit === 'samples') { out.regular = true; out.gaps = 0; return out; }
    // regularity: residual against a straight line per stretch; stretches are cut at gaps and at steps backwards
    const sc = UNIT_SCALE[unit], gapThr = Math.max(10 * mu, 1 / rate / sc * 5);
    let gaps = 0, maxRes = 0, segStart = -1, segFirstT = 0, segCount = 0, lastT = NaN, lastI = 0;
    const segs = [];
    for (let i = 0; i < n; i++) {
      const v = data[i];
      if (v !== v) continue;
      if (segStart < 0) { segStart = i; segFirstT = v; segCount = 0; }
      else if (v < lastT || v - lastT > gapThr) { segs.push([segStart, lastI]); if (v - lastT > gapThr) gaps++; segStart = i; segFirstT = v; segCount = 0; }
      lastT = v; lastI = i; segCount++;
    }
    if (segStart >= 0) segs.push([segStart, lastI]);
    const quantum = pos[0];
    for (const [a, b] of segs) {
      const len = b - a; if (len < 3) continue;
      const m = (data[b] - data[a]) / len;
      for (let i = a; i <= b; i += (len > 200000 ? 7 : 1)) {
        const v = data[i]; if (v !== v) continue;
        const r = Math.abs(v - data[a] - (i - a) * m);
        if (r > maxRes) maxRes = r;
      }
    }
    out.gaps = gaps; out.maxResidualSamples = maxRes / mu;
    out.regular = maxRes <= Math.max(1.5 * mu, 2 * quantum) && back === 0;
    return out;
  }

  // ---------------------------------------------------------------- suggestions
  const baseName = (s) => s.slice(s.lastIndexOf('/') + 1);              // HDF5 group path: only the last part names the variable
  const normName = (s) => baseName(s).toLowerCase().replace(/[^a-z0-9]/g, '');
  function scoreXY(name, axis) {
    const b = baseName(name);
    if (/^(xy|eye)\(:,\s*1\)$/i.test(b)) return axis === 'x' ? 92 : -100;
    if (/^(xy|eye)\(:,\s*2\)$/i.test(b)) return axis === 'y' ? 92 : -100;
    const n = normName(name);
    const o = axis === 'x' ? 'y' : 'x';
    if (/(^|[a-z])(gt|target|head|tgt)/.test(n) || /^(gt|tgt)/.test(n) || n === axis + 't' || /(target|goal|stim)/.test(n)) return -50;
    if (n.indexOf(o) >= 0 && n.indexOf(axis) < 0) return -100;
    let s = 0;
    if (n === axis) s = 100;
    else if (n === 'gaze' + axis || n === axis + 'gaze' || n === 'gaze' + axis + 'pos') s = 95;
    else if (n === axis + 'raw' || n === 'raw' + axis) s = 85;
    else if (n === axis + 'pos' || n === axis + 'pix' || n === axis + 'px' || n === axis + 'deg' || n === 'eye' + axis) s = 80;
    else if (n === 'l' + axis || n === 'r' + axis || n === axis + 'l' || n === axis + 'r') s = 45;
    else if (n === axis + 'mm') s = 60;
    else if (n.startsWith(axis) && n.length <= 8) s = 50;
    else if (n.endsWith(axis) && n.length <= 14) s = 40;
    else return -1;
    if (/mean/.test(n) && /lr/.test(n)) s += 5;
    if (/fix|sacc|blink/.test(n)) s -= 20;
    return s;
  }
  function timeNameScore(name) {
    const n = normName(name);
    if (n === 'time' || n === 'timestamp' || n === 'timerel' || n === 'tim' || n === 'timems' || n === 'timeus' || n === 'times' || n === 'timestampms') return 95;
    if (/timestamp|^time|time$|^ts$|^tim|^zeit|^temps|^tiempo|^tempo/.test(n)) return 80;
    if (n === 't' || n === 'tms' || n === 'tus' || n === 'tsec') return 70;
    if (n === 'n' || n === 'sample' || n === 'samples' || n === 'index' || n === 'idx' || n === 'frame') return 40;
    return 0;
  }
  function unitHintFromName(name) {
    const l = baseName(name).toLowerCase();
    if (/(^|[^a-z])(us|usec|µs|micro)/.test(l)) return 'us';
    if (/(^|[^a-z])ms($|[^a-z])|msec|milli/.test(l)) return 'ms';
    if (/\(s\)|(^|[_\s])sec($|[^a-z])|_s$/.test(l)) return 's';
    return null;
  }
  function partnerIndex(table, i, from, to) {
    const name = table.cols[i].name;
    const cand = [];
    if (from === 'x' && to === 'y' && /\(:,\s*1\)$/.test(name)) cand.push(name.replace(/\(:,\s*1\)$/, '(:,2)'));
    for (let k = name.length - 1; k >= 0; k--) if (name[k].toLowerCase() === from) { cand.push(name.slice(0, k) + (name[k] === name[k].toUpperCase() ? to.toUpperCase() : to) + name.slice(k + 1)); break; }
    for (const c of cand) { const j = table.cols.findIndex(cc => cc.name === c); if (j >= 0) return j; }
    return -1;
  }

  function analyzeTimeColumn(table, i, stats) {
    stats = stats || describeColumns(table);
    const name = table.cols[i].name, nn = normName(name);
    return analyzeTime(table.cols[i].data, table.nRows, {
      hintUnit: table.timeUnit || unitHintFromName(name), hintRate: table.info && table.info.rate || null,
      integerSteps: stats[i].isInt, indexName: nn === 'n' || nn === 'sample' || nn === 'samples' || nn === 'index' || nn === 'idx' || nn === 'frame',
    });
  }
  // sampling rate (Hz) of a time column with the given mean step, in the given unit; snapped to a usual rate when close
  function rateFor(meanStep, unit) {
    if (!(meanStep > 0) || !UNIT_SCALE[unit]) return null;
    const r = 1 / (meanStep * UNIT_SCALE[unit]), nt = nearestTypical(r);
    return nt.dist < 0.03 ? nt.rate : Math.round(r * 10) / 10;
  }

  function suggest(table, stats) {
    stats = stats || describeColumns(table);
    const num = (i) => !stats[i].nominal && stats[i].finite >= 0.3 * stats[i].n;
    const sug = { x: -1, y: -1, t: -1, valid: null, trial: -1, labels: -1, labelsRole: 'prelabels', prob: -1, notes: [], time: null };
    let best = -Infinity;
    for (let i = 0; i < table.cols.length; i++) {
      if (!num(i)) continue;
      const sx = scoreXY(table.cols[i].name, 'x'); if (sx < 0) continue;
      const j = partnerIndex(table, i, 'x', 'y');
      let sy = j >= 0 && num(j) ? scoreXY(table.cols[j].name, 'y') : -1;
      let yj = j;
      if (sy < 0) {                                  // no mirrored name: best y on its own
        yj = -1; let by = -1;
        for (let k = 0; k < table.cols.length; k++) if (k !== i && num(k)) { const s = scoreXY(table.cols[k].name, 'y'); if (s > by) { by = s; yj = k; } }
        sy = by - 15;
      }
      if (yj < 0) continue;
      const score = sx + sy + (j === yj ? 10 : 0);
      if (score > best) { best = score; sug.x = i; sug.y = yj; }
    }
    if (sug.x < 0) {                                 // names tell nothing: the first two numeric non-monotonic columns
      const free = []; for (let i = 0; i < table.cols.length; i++) if (num(i) && !(stats[i].monotonic > 0.99 && stats[i].isInt)) free.push(i);
      if (free.length >= 2) { sug.x = free[0]; sug.y = free[1]; sug.notes.push('Column names do not say which are X and Y: check the choice.'); }
      else if (table.cols.length >= 2 && num(0) && num(1)) { sug.x = 0; sug.y = 1; sug.notes.push('Column names do not say which are X and Y: check the choice.'); }
    }
    // time
    let tBest = -1, tScore = 0;
    for (let i = 0; i < table.cols.length; i++) {
      if (i === sug.x || i === sug.y || !num(i)) continue;
      const ns = timeNameScore(table.cols[i].name);
      if (ns <= 0 || stats[i].monotonic < 0.98) continue;
      if (ns > tScore) { tScore = ns; tBest = i; }
    }
    if (tBest >= 0) {
      sug.t = tBest;
      sug.time = analyzeTimeColumn(table, tBest, stats);
      if (!sug.time.ok) { sug.notes.push('Time column "' + table.cols[tBest].name + '": ' + sug.time.reason + '.'); sug.t = -1; sug.time = null; }
    }
    // validity
    for (let i = 0; i < table.cols.length; i++) {
      if (i === sug.x || i === sug.y || i === sug.t || stats[i].nominal) continue;
      const n = normName(table.cols[i].name);
      if (n === 'val' && stats[i].isInt && stats[i].max <= 4) { sug.valid = { col: i, op: '=', val: 0 }; break; }
      if (/^(confidence|conf)$/.test(n) && stats[i].max <= 1.0001) { sug.valid = { col: i, op: '>=', val: 0.5 }; break; }
    }
    // trials
    const trialName = /trial|item|block|segment|session|recording|epoch|run|cell number/i;
    let tr = -1, trScore = 0;
    for (let i = 0; i < table.cols.length; i++) {
      if (i === sug.x || i === sug.y || i === sug.t) continue;
      const c = table.cols[i];
      if (!(c.labels || stats[i].isInt) || !trialName.test(c.name)) continue;
      let runs = 1; for (let k = 1; k < table.nRows; k++) if (c.data[k] !== c.data[k - 1]) runs++;
      if (runs < 2 || runs > 20000 || runs > table.nRows / 20) continue;
      const sc = (c.name.toLowerCase() === 'block' ? 90 : /trial_?id$/i.test(c.name) ? 80 : /trial/i.test(c.name) ? 60 : 40) + (c.labels ? 5 : 0);
      if (sc > trScore) { trScore = sc; tr = i; sug.trialRuns = runs; }
    }
    sug.trial = tr;

    // labels / pre-labels (optional)
    let labBest = -1, labScore = 0;
    for (let i = 0; i < table.cols.length; i++) {
      if (i === sug.x || i === sug.y || i === sug.t || i === sug.trial || stats[i].nominal) continue;
      if (!stats[i].isInt || stats[i].min < -1 || stats[i].max > 4) continue;
      const b = baseName(table.cols[i].name).toLowerCase();
      let sc = 0;
      if (b === 'labels' || b === 'pred_labels' || b === 'predlabels') sc = 100;
      else if (/^(saccade_?labels?|label|events?|gt|ground_?truth)$/.test(b)) sc = 80;
      if (sc > labScore) {
        labScore = sc;
        labBest = i;
        sug.labelsRole = (/^labels?$/i.test(b) && (stats[i].min < 0 || table.cols.some(c => /^(observed|reviewed)$/i.test(baseName(c.name))))) ? 'reviewed' : 'prelabels';
      }
    }
    sug.labels = labBest;

    // predictive probability (optional)
    let probBest = -1, probScore = 0;
    for (let i = 0; i < table.cols.length; i++) {
      if (i === sug.x || i === sug.y || i === sug.t || i === sug.trial || i === sug.labels || stats[i].nominal) continue;
      if (stats[i].min < -0.001 || stats[i].max > 1.001) continue;
      const b = baseName(table.cols[i].name).toLowerCase();
      let sc = 0;
      if (b === 'prob') sc = 100;
      else if (/^(probability|p_?sacc\w*|pred_?prob|net_?prob|saccade_?prob)$/.test(b)) sc = 85;
      if (sc > probScore) { probScore = sc; probBest = i; }
    }
    sug.prob = probBest;

    return sug;
  }

  // ---------------------------------------------------------------- columns -> trials
  function validArray(table, m) {
    const n = table.nRows, X = table.cols[m.x].data, Y = table.cols[m.y].data;
    const ok = new Uint8Array(n);
    const r = m.valid && m.valid.col >= 0 ? m.valid : null;
    const V = r ? table.cols[r.col].data : null;
    for (let i = 0; i < n; i++) {
      let good = X[i] === X[i] && Y[i] === Y[i] && Math.abs(X[i]) !== Infinity && Math.abs(Y[i]) !== Infinity;
      if (good && r) {
        const v = V[i];
        switch (r.op) {
          case '=': good = v === r.val; break;
          case '!=': good = v !== r.val && v === v; break;
          case '<': good = v < r.val; break;
          case '<=': good = v <= r.val; break;
          case '>': good = v > r.val; break;
          case '>=': good = v >= r.val; break;
        }
      }
      ok[i] = good ? 1 : 0;
    }
    return ok;
  }

  function trialRanges(table, m, T, sc) {
    const n = table.nRows;
    let cuts = [0];
    const sp = m.split || { mode: 'none' };
    if (sp.mode === 'column' && sp.col >= 0) {
      const c = table.cols[sp.col].data;
      for (let i = 1; i < n; i++) if (c[i] !== c[i - 1] && !(c[i] !== c[i] && c[i - 1] !== c[i - 1])) cuts.push(i);
    } else if (sp.mode === 'gap' && T) {
      const thr = (sp.gapSec || 1) / sc;
      for (let i = 1; i < n; i++) if (T[i] - T[i - 1] > thr) cuts.push(i);
    }
    if (T) {                                          // never keep a step backwards inside a trial
      const s = new Set(cuts);
      for (let i = 1; i < n; i++) if (T[i] < T[i - 1]) s.add(i);
      cuts = Array.from(s).sort((a, b) => a - b);
    }
    cuts.push(n);
    const out = [];
    for (let k = 0; k + 1 < cuts.length; k++) if (cuts[k + 1] > cuts[k]) out.push([cuts[k], cuts[k + 1]]);
    return out;
  }

  // m: { x, y, t (-1 none), unit, fs, valid, split: {mode, col, gapSec}, chunkSec, resample, gapFactor, minSamples, labels (-1 none), labelsRole, prob (-1 none) }
  function buildTrials(table, m) {
    const notes = [];
    const T = (m.t >= 0 && m.unit && m.unit !== 'samples') ? table.cols[m.t].data : null;
    const sc = T ? UNIT_SCALE[m.unit] : 1;
    const fs = m.fs;
    if (!(fs > 0)) throw new Error('The sampling rate is missing.');
    const X = table.cols[m.x].data, Y = table.cols[m.y].data;
    const Lcol = (m.labels !== undefined && m.labels >= 0 && table.cols[m.labels]) ? table.cols[m.labels].data : null;
    const Pcol = (m.prob !== undefined && m.prob >= 0 && table.cols[m.prob]) ? table.cols[m.prob].data : null;
    const isRev = m.labelsRole === 'reviewed';
    const defLab = isRev ? -1 : 0;
    const ok = validArray(table, m);
    const ranges = trialRanges(table, m, T, sc);
    const splitCol = m.split && m.split.mode === 'column' && m.split.col >= 0 ? table.cols[m.split.col] : null;
    const minSamples = m.minSamples || (splitCol && splitCol.fromOffsets ? 1 : 50), gapFactor = m.gapFactor || 3;
    const trials = [];
    const outLabels = Lcol ? [] : null;
    const outProb = Pcol ? [] : null;
    let dropped = 0, resampledAny = false;
    for (const [a, b] of ranges) {
      let x, y, valid, lab = null, pr = null, t0 = T ? T[a] * sc : a / fs, resampled = false;
      if (T && m.resample) {
        const tFirst = T[a] * sc, tLast = T[b - 1] * sc;
        const len = Math.floor((tLast - tFirst) * fs + 1e-9) + 1;
        if (len < 2) { dropped++; continue; }
        x = new Float32Array(len).fill(NaN); y = new Float32Array(len).fill(NaN); valid = new Uint8Array(len);
        if (Lcol) lab = new Int8Array(len).fill(defLab);
        if (Pcol) pr = new Float32Array(len).fill(NaN);
        const maxSpan = gapFactor / fs;
        let j = a;
        for (let i = 0; i < len; i++) {
          const g = tFirst + i / fs;
          while (j + 1 < b && T[j + 1] * sc <= g + 1e-12) j++;
          const tj = T[j] * sc;
          if (Math.abs(tj - g) < 1e-9 && ok[j]) {
            x[i] = X[j]; y[i] = Y[j]; valid[i] = 1;
            if (lab) { const lv = Lcol[j]; lab[i] = (lv === lv && isFinite(lv)) ? Math.round(lv) : defLab; }
            if (pr) { const pv = Pcol[j]; pr[i] = (pv === pv && isFinite(pv)) ? pv : NaN; }
            continue;
          }
          if (j + 1 >= b) continue;
          const tn = T[j + 1] * sc, span = tn - tj;
          if (!ok[j] || !ok[j + 1] || span <= 0 || span > maxSpan) continue;
          const w = (g - tj) / span;
          x[i] = X[j] + w * (X[j + 1] - X[j]); y[i] = Y[j] + w * (Y[j + 1] - Y[j]); valid[i] = 1;
          if (lab) { const nearJ = w < 0.5 ? j : j + 1, lv = Lcol[nearJ]; lab[i] = (lv === lv && isFinite(lv)) ? Math.round(lv) : defLab; }
          if (pr) { const p0 = Pcol[j], p1 = Pcol[j + 1]; pr[i] = (isFinite(p0) && isFinite(p1)) ? p0 + w * (p1 - p0) : NaN; }
        }
        resampled = true; resampledAny = true; t0 = tFirst;
      } else {
        const len = b - a;
        x = new Float32Array(len); y = new Float32Array(len); valid = new Uint8Array(len);
        if (Lcol) lab = new Int8Array(len);
        if (Pcol) pr = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          if (ok[a + i]) { x[i] = X[a + i]; y[i] = Y[a + i]; valid[i] = 1; } else { x[i] = NaN; y[i] = NaN; }
          if (lab) { const lv = Lcol[a + i]; lab[i] = (lv === lv && isFinite(lv)) ? Math.round(lv) : defLab; }
          if (pr) { const pv = Pcol[a + i]; pr[i] = (pv === pv && isFinite(pv)) ? pv : NaN; }
        }
      }
      let good = 0; for (let i = 0; i < valid.length; i++) good += valid[i];
      if (good < minSamples) { dropped++; continue; }
      let name = '';
      if (splitCol) { const v = splitCol.data[a]; name = splitCol.labels ? (splitCol.labels[v] !== undefined ? splitCol.labels[v] : '') : String(v); }
      const base = { name, t0, rows: [a, b], resampled };
      const chunk = m.chunkSec > 0 ? Math.max(Math.round(m.chunkSec * fs), minSamples) : 0;
      if (chunk && x.length > 1.25 * chunk) {
        let k = 0;
        for (let s = 0; s < x.length;) {
          let e = Math.min(s + chunk, x.length);
          if (x.length - e < 0.25 * chunk) e = x.length;
          const v = valid.slice(s, e); let g = 0; for (let i = 0; i < v.length; i++) g += v[i];
          k++;
          if (g >= minSamples) {
            trials.push({ x: x.slice(s, e), y: y.slice(s, e), valid: v, name: (name ? name + ' · ' : '') + 'part ' + k, t0: base.t0 + s / fs, rows: base.rows, resampled });
            if (outLabels) outLabels.push(lab.slice(s, e));
            if (outProb) outProb.push(pr.slice(s, e));
          } else dropped++;
          s = e;
        }
      } else {
        trials.push({ x, y, valid, name, t0: base.t0, rows: base.rows, resampled });
        if (outLabels) outLabels.push(lab);
        if (outProb) outProb.push(pr);
      }
    }
    if (dropped) notes.push(dropped + ' stretch' + (dropped === 1 ? '' : 'es') + ' with fewer than ' + minSamples + ' valid samples skipped.');
    if (resampledAny) notes.push('Resampled to a uniform ' + fs + ' Hz grid; gaps longer than ' + gapFactor + ' samples stay invalid.');
    return {
      trials,
      fs,
      notes,
      labels: (outLabels && isRev) ? outLabels : null,
      predLabels: (outLabels && !isRev) ? outLabels : null,
      prob: outProb,
      probParams: (table.info && table.info.probParams) || null,
    };
  }

  // how many trials buildTrials() will return (before dropping stretches that are too short), without building them
  function countTrials(table, m) {
    const T = (m.t >= 0 && m.unit && m.unit !== 'samples') ? table.cols[m.t].data : null;
    const sc = T ? UNIT_SCALE[m.unit] : 1;
    const ranges = trialRanges(table, m, T, sc);
    const chunk = m.chunkSec > 0 && m.fs > 0 ? Math.max(1, Math.round(m.chunkSec * m.fs)) : 0;
    let n = 0, samples = 0;
    for (const [a, b] of ranges) {
      const len = T && m.resample && m.fs ? Math.floor((T[b - 1] - T[a]) * sc * m.fs) + 1 : b - a;
      samples += len;
      n += chunk && len > 1.25 * chunk ? Math.max(1, Math.floor(len / chunk)) : 1;
    }
    return { trials: n, samples };
  }

  // 99.9th percentile of the speed in deg/s over the first part of the trials: a quick check of the units
  function peakSpeed(trials, fs, degPerUnit) {
    const v = [];
    for (let k = 0; k < Math.min(trials.length, 20) && v.length < 300000; k++) {
      const t = trials[k], n = Math.min(t.x.length, 30000);
      for (let i = 1; i < n; i++) if (t.valid[i] && t.valid[i - 1]) v.push(Math.hypot(t.x[i] - t.x[i - 1], t.y[i] - t.y[i - 1]) * degPerUnit * fs);
    }
    if (v.length < 10) return NaN;
    v.sort((a, b) => a - b);
    return v[Math.min(v.length - 1, Math.floor(v.length * 0.999))];
  }

  // degrees per pixel for a screen of widthPx pixels / widthCm cm seen from distCm cm
  function degPerPixel(widthPx, widthCm, distCm) {
    if (!(widthPx > 0 && widthCm > 0 && distCm > 0)) return NaN;
    return (2 * Math.atan(widthCm / 2 / distCm) * 180 / Math.PI) / widthPx;
  }

  return {
    parseString, parseFile, parseSource, parseDelimited, parseArff, parseAsc, chunksOfFile, sniffHead, sniffLines,
    tablesFromVars, describeColumns, analyzeTime, analyzeTimeColumn, rateFor, suggest, validArray, countTrials, buildTrials, peakSpeed, degPerPixel, UNIT_SCALE, TYPICAL_RATES,
  };
}));
