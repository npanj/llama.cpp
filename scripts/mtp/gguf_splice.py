#!/usr/bin/env python3
"""Build Qwen3.8-Flash-Next UD-iQ4_K_XXS by splicing tensors into unsloth's UD-Q3_K_XL.

The recipe, from docs/Qwen3.8-Flash-Next.md:
  - base is unsloth UD-Q3_K_XL
  - every IQ4_NL blk.N.ffn_down_exps is replaced with the MXFP4 tensor of the same name
    from AtomicChat AD-3.84bpw-IQ4_XS-M64, byte for byte. The five layers unsloth kept at
    Q8_0 are left alone.
  - output.weight is replaced with the Q8_0 one from unsloth UD-Q4_K_XL
  - the IQ4_NL per_layer_token_embd (the PLE table) is KEPT. Matching on type alone would
    swap it too, which is the trap the log records - so we match on tensor NAME.

Nothing is dequantized. Tensor data is copied verbatim; only the type field and the data
offsets in each shard header change, and the header is patched in place so the tokenizer
metadata is never re-serialized.
"""
from __future__ import annotations
import argparse, json, os, re, struct, sys

QUANT = {  # ggml type id -> (name, elements per block, bytes per block)
 0:('F32',1,4), 1:('F16',1,2), 2:('Q4_0',32,18), 3:('Q4_1',32,20), 6:('Q5_0',32,22),
 7:('Q5_1',32,24), 8:('Q8_0',32,34), 10:('Q2_K',256,84), 11:('Q3_K',256,110),
 12:('Q4_K',256,144), 13:('Q5_K',256,176), 14:('Q6_K',256,210), 16:('IQ2_XXS',256,66),
 17:('IQ2_XS',256,74), 18:('IQ3_XXS',256,98), 19:('IQ1_S',256,50), 20:('IQ4_NL',32,18),
 21:('IQ3_S',256,110), 22:('IQ2_S',256,82), 23:('IQ4_XS',256,136), 24:('I8',1,1),
 25:('I16',1,2), 26:('I32',1,4), 27:('I64',1,8), 28:('F64',1,8), 29:('IQ1_M',256,56),
 30:('BF16',1,2), 39:('MXFP4',32,17), 40:('NVFP4',32,18),
}
BY_NAME = {v[0]: (k, v[1], v[2]) for k, v in QUANT.items()}
SCALAR = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}

def tensor_nbytes(dims, type_id):
    n = 1
    for d in dims: n *= d
    _, blk, size = QUANT[type_id]
    if n % blk: raise ValueError(f'{n} elements is not a multiple of block {blk}')
    return n // blk * size

class Header:
    """A parsed GGUF header that remembers where each tensor's type and offset live."""
    def __init__(self, path, buf):
        self.path, self.buf = path, buf
        self.o = 0
        if self._raw(4) != b'GGUF': raise ValueError(f'{path}: not a GGUF file')
        self.version = self._u32()
        n_tensors, n_kv = self._u64(), self._u64()
        self.kv = {}
        for _ in range(n_kv):
            k = self._str(); self.kv[k] = self._value(self._u32())
        self.tensors = []
        for _ in range(n_tensors):
            name = self._str(); nd = self._u32()
            dims = [self._u64() for _ in range(nd)]
            type_pos = self.o; type_id = self._u32()
            off_pos = self.o;  rel = self._u64()
            self.tensors.append({'name': name, 'dims': dims, 'type': type_id,
                                 'rel': rel, 'type_pos': type_pos, 'off_pos': off_pos,
                                 'nbytes': tensor_nbytes(dims, type_id)})
        self.align = self.kv.get('general.alignment', 32)
        self.header_end = self.o
        self.data_start = (self.o + self.align - 1) // self.align * self.align
        for t in self.tensors:
            t['file_offset'] = self.data_start + t['rel']

    def _raw(self, n):
        if self.o + n > len(self.buf): raise EOFError(f'{self.path}: header truncated')
        v = self.buf[self.o:self.o+n]; self.o += n; return v
    def _u32(self): return struct.unpack('<I', self._raw(4))[0]
    def _u64(self): return struct.unpack('<Q', self._raw(8))[0]
    def _str(self): return self._raw(self._u64()).decode('utf-8', 'replace')
    def _value(self, t):
        if t == 8: return self._str()
        if t == 9:
            et, n = self._u32(), self._u64()
            if et in (8, 9): return [self._value(et) for _ in range(n)]
            self._raw(n * struct.calcsize(SCALAR[et])); return f'<array {n} x t{et}>'
        f = SCALAR[t]; return struct.unpack('<'+f, self._raw(struct.calcsize(f)))[0]

def read_header(path, probe=1 << 20):
    """Grow the read until the whole header parses. Shard 1 carries the vocab."""
    size = os.path.getsize(path)
    while True:
        with open(path, 'rb') as f: buf = f.read(min(probe, size))
        try: return Header(path, buf)
        except EOFError:
            if probe >= size: raise
            probe *= 4

def shard_paths(first):
    m = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', first)
    if not m: return [first]
    total = int(m.group(2)); stem = first[:m.start()]
    return [f'{stem}-{i:05d}-of-{total:05d}.gguf' for i in range(1, total + 1)]

def index_model(first):
    out = {}
    for p in shard_paths(first):
        h = read_header(p)
        for t in h.tensors:
            out[t['name']] = dict(t, path=p)
    return out

def copy_range(src, src_off, n, dst, chunk=32 << 20):
    with open(src, 'rb') as f:
        f.seek(src_off)
        left = n
        while left:
            b = f.read(min(chunk, left))
            if not b: raise EOFError(f'{src}: short read at {src_off}, {left} bytes left')
            dst.write(b); left -= len(b)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True, help='first shard of unsloth UD-Q3_K_XL')
    ap.add_argument('--expert-donor', required=True, help='first shard of the MXFP4 donor')
    ap.add_argument('--output-weight', help='raw Q8_0 output.weight blob (optional)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-name', default='Qwen3.8-Flash-Next-UD-iQ4_K_XXS')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and sizes only')
    a = ap.parse_args()

    base_shards = shard_paths(a.base)
    print(f'base   : {len(base_shards)} shards, {a.base}')
    donor = index_model(a.expert_donor)
    print(f'donor  : {len(donor)} tensors indexed from {a.expert_donor}')

    subs, plan_rows = {}, []
    headers = [read_header(p) for p in base_shards]
    total_before = total_after = 0

    for h in headers:
        for t in h.tensors:
            total_before += t['nbytes']
            name, tname = t['name'], QUANT[t['type']][0]
            new = None
            if re.fullmatch(r'blk\.\d+\.ffn_down_exps\.weight', name) and tname == 'IQ4_NL':
                d = donor.get(name)
                if d is None: sys.exit(f'donor is missing {name}')
                if d['dims'] != t['dims']: sys.exit(f'{name}: dims {d["dims"]} != {t["dims"]}')
                if QUANT[d['type']][0] != 'MXFP4':
                    sys.exit(f'{name}: donor is {QUANT[d["type"]][0]}, expected MXFP4')
                new = {'src': d['path'], 'off': d['file_offset'],
                       'type': d['type'], 'nbytes': d['nbytes'], 'why': 'MXFP4 from donor'}
            elif name == 'output.weight' and a.output_weight:
                tid, _, _ = BY_NAME['Q8_0']
                want = tensor_nbytes(t['dims'], tid)
                have = os.path.getsize(a.output_weight)
                if want != have: sys.exit(f'output.weight blob is {have} bytes, expected {want}')
                new = {'src': a.output_weight, 'off': 0, 'type': tid,
                       'nbytes': want, 'why': 'Q8_0 from UD-Q4_K_XL'}
            if new:
                subs[name] = new
                total_after += new['nbytes']
                plan_rows.append((name, tname, QUANT[new['type']][0],
                                  t['nbytes'], new['nbytes'], new['why']))
            else:
                total_after += t['nbytes']

    ple = [n for n in subs if 'per_layer_token_embd' in n]
    if ple: sys.exit(f'refusing to touch the PLE table: {ple}')

    n_exp = sum(1 for n in subs if 'ffn_down_exps' in n)
    print(f'\nplan   : {len(subs)} substitutions ({n_exp} expert down-projections'
          f'{", output.weight" if "output.weight" in subs else ""})')
    for name, old, newt, ob, nb, why in plan_rows[:3] + (plan_rows[-2:] if len(plan_rows) > 5 else []):
        print(f'         {name:38} {old:7} -> {newt:6} {(nb-ob)/2**20:+9.1f} MiB  ({why})')
    if len(plan_rows) > 5: print(f'         ... {len(plan_rows)-5} more')
    print(f'\nsize   : {total_before/2**30:.2f} GiB -> {total_after/2**30:.2f} GiB '
          f'({(total_after-total_before)/2**30:+.2f} GiB of tensor data)')
    if a.dry_run: return

    os.makedirs(a.out_dir, exist_ok=True)
    n_shard = len(base_shards)
    for i, (src, h) in enumerate(zip(base_shards, headers), start=1):
        dst = os.path.join(a.out_dir, f'{a.out_name}-{i:05d}-of-{n_shard:05d}.gguf')
        if not h.tensors:                       # metadata-only shard: copy verbatim
            print(f'\nshard {i}/{n_shard}: {os.path.getsize(src)/2**20:.1f} MiB, metadata only - copying')
            with open(dst, 'wb') as out: copy_range(src, 0, os.path.getsize(src), out)
            continue
        hdr = bytearray(h.buf[:h.data_start])
        rel, layout = 0, []
        for t in h.tensors:
            s = subs.get(t['name'])
            type_id = s['type'] if s else t['type']
            nbytes  = s['nbytes'] if s else t['nbytes']
            struct.pack_into('<I', hdr, t['type_pos'], type_id)
            struct.pack_into('<Q', hdr, t['off_pos'], rel)
            layout.append((t['name'], s['src'] if s else src,
                           s['off'] if s else t['file_offset'], nbytes, rel))
            rel += (nbytes + h.align - 1) // h.align * h.align
        print(f'\nshard {i}/{n_shard}: {len(h.tensors)} tensors, '
              f'{rel/2**30:.2f} GiB of data -> {os.path.basename(dst)}')
        with open(dst, 'wb') as out:
            out.write(hdr)
            for k, (name, sp, so, nb, want_rel) in enumerate(layout):
                at = h.data_start + want_rel
                if out.tell() < at: out.write(b'\0' * (at - out.tell()))
                assert out.tell() == at, f'{name}: at {out.tell()}, expected {at}'
                copy_range(sp, so, nb, out)
                if k % 50 == 0 or k == len(layout) - 1:
                    print(f'  {k+1:4}/{len(layout)}  {out.tell()/2**30:7.2f} GiB  {name}',
                          flush=True)
    print('\ndone')

if __name__ == '__main__':
    main()
