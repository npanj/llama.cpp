#!/usr/bin/env python3
"""Rebuild a Qwen3.8-Flash-Next MTP sidecar for the mihailescu2m fork's tensor names.

Published sidecars follow the unsloth/upstream naming. This fork wants:

    blk.N.nextn.eh_proj   [2*n_embd, n_embd]  ->  blk.N.nextn.fc_embd   [n_embd, n_embd]
                                              +   blk.N.nextn.fc_hidden [n_embd, n_embd]
    output_hc_norm/down/up                    ->  blk.N.nextn.hc_norm/hc_down/hc_up

The split is byte exact, not a requantization: a Q4_K row of 5120 elements is 20
super-blocks of 256, so cutting at 2560 lands on a super-block boundary. Rows are
contiguous along ne0, so the two halves INTERLEAVE in the file - each row must be cut
separately, a single contiguous slice would be wrong.

Which half is which cannot be read off the file. HF concatenates [embedding, hidden],
so the first half is taken as fc_embd; --swap-halves builds the other assignment. A
wrong choice does not fail loudly - draft acceptance collapses to near zero - so measure
acceptance before trusting it.
"""
from __future__ import annotations
import argparse, importlib.util, os, struct, sys

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('gs', os.path.join(_here, 'gguf_splice.py'))
gs = importlib.util.module_from_spec(spec); spec.loader.exec_module(gs)
Q, BY_NAME = gs.QUANT, gs.BY_NAME

def kv_block(h):
    """Bytes of the metadata block, verbatim: everything between the counts and infos."""
    r = gs.Header.__new__(gs.Header)
    r.path, r.buf, r.o = h.path, h.buf, 0
    r._raw(4); r._u32(); r._u64(); n_kv = r._u64()
    start = r.o
    for _ in range(n_kv):
        r._str(); r._value(r._u32())
    return h.buf[start:r.o], n_kv

def info_bytes(name, dims, type_id, offset):
    b = bytearray()
    n = name.encode('utf-8')
    b += struct.pack('<Q', len(n)) + n
    b += struct.pack('<I', len(dims))
    for d in dims: b += struct.pack('<Q', d)
    b += struct.pack('<I', type_id) + struct.pack('<Q', offset)
    return bytes(b)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--swap-halves', action='store_true',
                    help='assign the FIRST half of eh_proj to fc_hidden instead')
    a = ap.parse_args()

    h = gs.read_header(a.src)
    kvb, n_kv = kv_block(h)
    src_by_name = {t['name']: t for t in h.tensors}

    eh = [t for t in h.tensors if t['name'].endswith('.nextn.eh_proj.weight')]
    if len(eh) != 1: sys.exit(f'expected exactly one nextn.eh_proj, found {len(eh)}')
    eh = eh[0]
    bid = eh['name'].split('.')[1]
    tname = Q[eh['type']][0]
    blk, bsz = Q[eh['type']][1], Q[eh['type']][2]

    n_in, n_rows = eh['dims'][0], eh['dims'][1]
    half = n_in // 2
    if n_in != 2 * half:        sys.exit(f'eh_proj ne0={n_in} is not even')
    if half % blk:              sys.exit(f'{tname}: half {half} is not a multiple of block {blk}')
    row_bytes  = n_in // blk * bsz
    half_bytes = half  // blk * bsz
    print(f'{eh["name"]}: {tname} {eh["dims"]}')
    print(f'  row = {row_bytes} bytes = {n_in//blk} super-blocks; cutting at {half} elements '
          f'({half//blk} super-blocks, {half_bytes} bytes) - block aligned, byte exact')

    first, second = ('fc_hidden', 'fc_embd') if a.swap_halves else ('fc_embd', 'fc_hidden')
    print(f'  first half -> nextn.{first},  second half -> nextn.{second}'
          + ('   [SWAPPED]' if a.swap_halves else ''))

    # Three naming conventions seen in the wild for the head's own mixer:
    #   dzannotti / older unsloth : output_hc_{norm,down,up}
    #   unsloth MTP dir (2026-09) : blk.N.nextn.hc_head_{norm,down,up}
    #   this fork wants           : blk.N.nextn.hc_{norm,down,up}
    RENAME = {}
    for suf in ('norm', 'down', 'up'):
        want = f'blk.{bid}.nextn.hc_{suf}.weight'
        for cand in (f'output_hc_{suf}.weight', f'blk.{bid}.nextn.hc_head_{suf}.weight'):
            if cand in src_by_name:
                RENAME[cand] = want

    plan = []   # (out_name, dims, type_id, nbytes, source spec)
    for t in h.tensors:
        if t is eh:
            for lbl in (first, second):
                plan.append((f'blk.{bid}.nextn.{lbl}.weight', [half, n_rows], t['type'],
                             half_bytes * n_rows,
                             ('split', t, 0 if lbl == first else half_bytes)))
        else:
            plan.append((RENAME.get(t['name'], t['name']), t['dims'], t['type'],
                         t['nbytes'], ('copy', t, 0)))

    if len(RENAME) != 3:
        sys.exit(f'could not locate all three hc mixer tensors; found {sorted(RENAME)}')
    for old, new in RENAME.items():
        print(f'  rename {old} -> {new}')
    # a "shared" sidecar omits output.weight/token_embd and borrows the target's
    for t in ('output.weight', 'token_embd.weight'):
        if t not in src_by_name:
            print(f'  note: {t} absent - shared-head layout, will use the target\'s')

    need = {'fc_embd', 'fc_hidden', 'enorm', 'hnorm', 'hc_norm', 'hc_down', 'hc_up'}
    have = {n.split('.nextn.')[1].replace('.weight', '') for n, *_ in plan if '.nextn.' in n}
    missing = need - have
    if missing: sys.exit(f'the fork requires nextn.{{{",".join(sorted(missing))}}} - not produced')
    print(f'\n{len(h.tensors)} tensors in -> {len(plan)} out; '
          f'nextn set complete: {sorted(have)}')

    align = h.align
    infos, rel = bytearray(), 0
    layout = []
    for name, dims, tid, nbytes, src in plan:
        infos += info_bytes(name, dims, tid, rel)
        layout.append((name, nbytes, src, rel))
        rel += (nbytes + align - 1) // align * align

    head = bytearray(b'GGUF')
    head += struct.pack('<I', h.version)
    head += struct.pack('<Q', len(plan)) + struct.pack('<Q', n_kv)
    head += kvb + infos
    data_start = (len(head) + align - 1) // align * align
    head += b'\0' * (data_start - len(head))

    with open(a.src, 'rb') as fin, open(a.out, 'wb') as out:
        out.write(head)
        for name, nbytes, (kind, t, off), want in layout:
            at = data_start + want
            if out.tell() < at: out.write(b'\0' * (at - out.tell()))
            assert out.tell() == at, name
            if kind == 'copy':
                fin.seek(t['file_offset'])
                left = nbytes
                while left:
                    b = fin.read(min(32 << 20, left)); out.write(b); left -= len(b)
            else:
                # interleaved: take this half out of every row
                for r in range(n_rows):
                    fin.seek(t['file_offset'] + r * row_bytes + off)
                    out.write(fin.read(half_bytes))
    print(f'\nwrote {a.out}  ({os.path.getsize(a.out)/2**30:.2f} GiB)')

if __name__ == '__main__':
    main()
