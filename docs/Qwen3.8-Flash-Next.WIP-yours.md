# Qwen3.8-Flash-Next (`qwen4exp`)

> Research log. Measured on **M1 Max, 64 GB, ~400 GB/s**, streaming experts from SSD.
> Conditions are quoted with every figure — in this log more than most, because several
> measurements here were later found to have been taken under conditions that invalidated them.

> [!IMPORTANT]
> **The machine changed.** Everything above the "Measured on M5 Pro" section below was taken on an
> M1 Max. This laptop now reports **Apple M5 Pro, 64 GB**, which runs this model at roughly 2x the
> M1 Max figures. Do not compare a new measurement against an old one without checking which
> machine it came from.

Where DeepSeek is a streaming problem, Qwen3.8-Flash-Next is a **graph-shape** problem. It decodes
fast enough that the bottleneck moved off the disk entirely and onto GPU work the graph did not
need to be doing: needless copies, broken kernel fusions, and an indexer that materialised a table
proportional to context x chunk size on every layer.

Three architectural features make it unusual, and all three cost real work to support:

- **Hyper-connections** — four parallel residual streams instead of one, mixed by a low-rank gate.
  Roughly 30% of decode GPU time, and the source of several fusion-breaking reshapes.
- **A 26.82 GiB PLE n-gram table** — far larger than RAM, touched a few rows at a time.
- **Gated DeltaNet on 36 of 48 layers**, with 12 full-attention layers carrying a sparse indexer.

---

## At a glance

| | |
|---|---|
| Layers | 48 — 36 SSM (gated DeltaNet) + 12 full attention |
| Experts | 512 per layer, 10 active per token |
| Attention | indexer `top_k = 2048`, `compress_ratio = 4`, head dim 256, GQA 48q:2kv |
| Hyper-connections | `hc = 4` streams |
| PLE table | 26.82 GiB, 90-byte rows — streamed, never resident |
| Checkpoint | `UD-iQ4_K_XXS`, 82.90 GiB, 3 shards (custom, see below) |
| Speculation | **native MTP head** (`mtp-…-Q4_0.gguf`, 2.2 GiB), n-max 4 |
| KV | F16 |

```bash
llama-server -m <first shard> \
  -ngl 99 --moe-stream --moe-stream-cache 32 --moe-stream-io-threads 8 \
  -c 131072 -b 4096 -ub 4096 -cms 4096 -np 1 \
  -md <mtp-…-Q4_0.gguf> \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 --spec-draft-ngl 99 \
  --spec-max-prompt 32768
```

`LLAMA_MOE_STREAM_LOOKAHEAD=1`. **Cache 32 with MTP, 36 without it.** The MTP head is resident on
top of the cache, and cache 36 + MTP *pages* — 1.1 GiB of swap and 5.3M swapouts, which invalidates
any measurement taken there. Cache 32 + MTP is clean.

The cache size is not free, which an earlier version of this log got wrong. See
[Cache size: 32 vs 36](#cache-size-32-vs-36).

---

## The checkpoint

`UD-iQ4_K_XXS` is a custom splice, built here rather than downloaded:

- base is unsloth's `UD-Q3_K_XL`;
- 43 of the 48 `ffn_down_exps` are replaced with **MXFP4** taken byte-for-byte from AtomicChat's
  `AD-3.84bpw-IQ4_XS-M64`, because MXFP4 is the format this fork has a hand-optimised Metal GEMV
  for. The five that unsloth deliberately kept at Q8_0 (layers 2, 4, 30, 46, 47) are left alone;
- `output.weight` is taken from `UD-Q4_K_XL` at Q8_0;
- the `IQ4_NL` PLE table is kept. AtomicChat ships it at `Q5_1`, which is 8.9 GiB larger, and that
  table is streamed per row - the last place to spend bytes.

The result is **0.9 GiB smaller than the Q3_K_XL it came from** despite the bigger output tensor.
Only the tensors named above differ from the base; 1180 of 1224 are untouched.

A trap worth recording: `--splice-type IQ4_NL` looks like the obvious selector for the
down-projections, and it also matches `per_layer_token_embd` — the PLE table is IQ4_NL too. With a
full donor that selector silently swaps the PLE table as well.

---

## Performance

### Where it started

Cache 36, ctx 65536, no speculation, 2 reps x 3 prompts:

| | t/s |
|---|---:|
| prefill (mean) | 121.10 |
| decode (128-tok mean) | 9.548 |

Both figures are from before any of the work below. Prefill reached 167.10 t/s partway through,
after the graph-shape fixes but before block-level top-k.

### Where it is now

`UD-iQ4_K_XXS`, cache 32, `-b 4096 -ub 4096 -fa 1`, llama-bench (`-r 2`, `-r 1` at 128k), no
speculation. Every cell verified swap-clean:

| context | pp t/s | tg t/s |
|---:|---:|---:|
| 4096 | 160.64 | 10.91 |
| 8192 | 159.11 | 10.82 |
| 16384 | 156.00 | 10.67 |
| 32768 | 150.81 | 10.02 |
| 65536 | 144.37 | 9.19 |
| 131072 | 132.79 | 8.03 |

Prefill falls 17.3% and decode 26.4% across a 32x context range.

### With MTP

Same model and cache, llama-server, one run per context, MTP as the only variable. Decode is the
*post-prefill* rate in both arms, so the two are directly comparable:

| context | pp no-MTP | pp MTP | tg no-MTP | tg MTP | decode gain |
|---:|---:|---:|---:|---:|---:|
| 4,061 | 159.14 | 150.11 | 10.44 | 17.55 | **+68%** |
| 8,307 | 157.03 | 146.91 | 10.70 | 20.47 | **+91%** |
| 16,649 | 155.63 | 147.25 | 10.32 | 19.34 | **+87%** |
| 33,339 | 151.43 | 141.51 | 10.35 | 19.12 | **+85%** |
| 66,505 | 142.62 | 131.97 | 9.38 | 16.57 | **+77%** |
| 129,983 | 131.45 | 119.37 | 8.60 | 12.99 | **+51%** |

**MTP is worth far more than the +39% this log used to record, and it still pays at 128k.** The
earlier figure was measured before the `t_h_nextn` export was fixed, when the head was running on a
hidden state that never reached it - see [Native MTP head](#native-mtp-head).

Prefill costs a steady 5.7-9.2%. End-to-end break-even, from these numbers:

| prompt | extra prefill | decode saved | repays after |
|---:|---:|---:|---:|
| 33k | 15.4 s | 0.0443 s/token | **~350 generated tokens** |
| 130k | 100.2 s | 0.0393 s/token | **~2,550 generated tokens** |

The 130k figure is close to the 2,369 measured previously, so the case for `--spec-max-prompt`
stands. The 32k break-even moved from 119 to ~350 tokens, so the threshold itself is worth
re-deriving rather than inherited.

### Cache size: 32 vs 36

An earlier version of this log said cache 32 was free, on the grounds that decode measured
identical at 32 and 36. **That was a decode-only measurement, and prefill was never tested.**
Interleaved A/B/A, two passes, same session, swap flat throughout:

| context | cache 36 | cache 32 | delta |
|---:|---:|---:|---:|
| 4096 | 186.46 / 186.06 | 160.57 / 160.53 | **-13.8%** |
| 8192 | 185.06 / 184.83 | 159.00 / 158.87 | **-14.1%** |
| 16384 | 180.87 / 181.22 | 156.03 / 156.05 | **-13.8%** |

**Cache 32 costs ~14% prefill and ~2% decode.** The penalty is flat across context, which is the
signature of a per-ubatch cost: every prefill ubatch sweeps the whole expert set, so 11% less
residency means ~14% more compulsory re-reads on each sweep. Prefill here is expert-I/O-bound.

That does not make 36 the right choice, because **cache 36 + MTP pages** (1.1 GiB swap, 5.3M
swapouts). The real choice is *cache 32 with MTP* against *cache 36 without it*, and MTP's +51-91%
decode dwarfs 14% of prefill for anything but a very long prompt with a very short answer.

### Prefill is a curve, not a number

The single most important result in this log. Block-level indexer top-k did not make prefill
uniformly faster — it flattened its **slope**:

| | ms/token |
|---|---|
| cell-level (before) | `5.828 + 3.864e-5·n` |
| block-level (after) | `6.026 + 6.668e-6·n` |

A **5.8x smaller quadratic coefficient**, at the cost of ~3% on the constant. Worth +14% at 32k,
**+71% at 160k**. Quoting this change as a single percentage is meaningless: measured at 32k it
looks like +18.5%, because at 32k the quadratic term is only 18% of the total.

---

## What was added, and why

### The indexer — the prefill story

| Change | Rationale | Measured |
|---|---|---|
| **Block-level top-k** | the budget is whole blocks anyway, so selecting over blocks skips materialising the `[n_kv, n_tps]` cell table entirely | **5.8x flatter prefill curve** |
| **Permute-free scoring** | putting `q` on the left lands the head index in dim 0, so `sum_rows` reduces heads directly — removing a permute+cont of ~640 MB per layer per 4096-token chunk | **+1.2% @ 32k, +1.4% @ 52k** |

Block-level selection is also a marginally *better* cut: all `r` cells of a block share a score, so
the cell-level top-k split the boundary block arbitrarily. Causality is unaffected — the causal
mask is still applied after selection.

### The graph — the decode story

| Change | Rationale | Measured |
|---|---|---|
| **RMS_NORM→MUL adjacency** | Metal's fusion check is a *tensor-identity* test; a reshape between the two silently broke it and spilled the norm to memory | part of +20.7% |
| **Hyper-connection collapse** | `ggml_cont` before the stream adds bought nothing — ADD needs contiguous *rows*, which a strided view already has — and blocked chain fusion | part of +20.7% |
| **`ggml_top_k` for MoE routing** | `argsort_top_k` fully sorted all 512 expert scores to view the first 10 | part of +20.7% |
| **Graph reuse / shared QSA input** | the whole graph was rebuilt and re-recorded every token | the largest share of +20.7% |

Combined A/B (cache 36, ctx 65536, no spec): **decode 9.548 → 11.522 t/s, +20.7%**; prefill flat.
Stream stats were unchanged between arms — miss 0.6–0.7%, stall 1.1%, GPU 98.7% — so the gain is
GPU work removed, not I/O.

### Native MTP head

The model ships a NextN block the loader read as part of the trunk and the graph could not run.
Supporting it needs five tensors beyond the shared NextN set, its own graph, and a hidden-state
export from the trunk.

**+51% to +91% decode** at n-max 4, depending on context — see [With MTP](#with-mtp).
This log previously recorded +39%; that was measured before the `t_h_nextn` export below was
fixed, i.e. while the head was still partly starved of the state it needs.

The head is fed the **wide** hyper-connection stream (`hc*n_embd`) from before the final mixer —
not the collapsed output — because its `hnorm` and `fc_hidden` are `[hc*n_embd]`. Three separate
things must hold or acceptance silently collapses to ~1% rather than failing loudly:

1. the trunk must set `res->t_h_nextn` at all (upstream `qwen4exp` sets only `t_embd`);
2. it must be the wide pre-mixer stream;
3. it needs an explicit `ggml_build_forward_expand` — it is a bare reshape view with no consumer,
   so otherwise the scheduler never assigns it a backend and it is never computed.

**The tell:** acceptance identical to the digit (8 accepted / 834 generated) across two different
drafter quantizations. Drafts that do not change when the drafter's weights change are not coming
from the drafter.

### PLE table streaming

26.82 GiB of 90-byte rows. Upstream keeps it off the resident set with `TENSOR_READ_LAZY`, which is
**mmap-only by construction** — rows arrive as page faults against a mapping the size of the whole
table. When expert streaming is on, the table is handed to the servicer instead: `set_input` reads
just this ubatch's rows with buffered parallel `pread`s into a compact tensor, and `get_rows`
dequantizes that exactly as it would the full table. Repeated rows are read once; reads are sorted
by file offset. Upstream's lazy path remains the fallback.

Deliberately **buffered, not `O_DIRECT`** — at 90 bytes a row the page cache genuinely helps, which
is the opposite of the finding for multi-MiB expert slabs.

> **Re-measured on M5 Pro: buffered stands.** An earlier entry here claimed uncached reads were
> worth +7% prefill and +21% decode. That was measured against a broken read path and is withdrawn.
> On a correct build the two are level. See
> [PLE reads: buffered vs uncached](#ple-reads-buffered-vs-uncached).

### Correctness fixes

| Fix | Consequence if missing |
|---|---|
| **Recurrent-state rollback** | conv history and delta-net groups left zeroed/stale, corrupting generation after any rollback |
| **Indexer cache after sequence copies** | cached indexer keys are raw, so a pending update must not rope-shift them |
| **CPU-reference GQA head mapping** | `h % n_kv_head` instead of `h / (n_head/n_kv_head)` — made a *correct* kernel look broken on every real GQA shape |
| **`--spec-max-prompt`** | a draft head is an extra layer over the whole prompt; past a length it cannot pay back |

---

## Measured on M5 Pro

New machine, and a different checkpoint: `Qwen3.8-Flash-Next-Q4_0-Q8out` (93.82 GiB, 3 shards) with
the `Q4_K_M` MTP head, not the `UD-iQ4_K_XXS` / `Q4_0` pair the rest of this log uses. Both changed
at once, so **none of these numbers are a same-conditions A/B against anything above.** They are a
new baseline.

Baseline, `llama-bench -p 4096 -n 128 -b 4096 -ub 4096 -fa 1 -r 2`, cache 36, no speculation:
**pp 337.20, tg 21.78**. Roughly 2x the M1 Max figures for the same command shape.

### Command

```bash
LLAMA_MOE_STREAM_LOOKAHEAD=1 \
llama-server -m <first shard> \
  -ngl 99 --moe-stream --moe-stream-cache 36 --moe-stream-io-threads 8 \
  -c 98304 -b 4096 -ub 4096 -cms 4096 -np 1 -fa 1 \
  -md <mtp-...gguf> \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 --spec-draft-ngl 99 \
  --spec-max-prompt 0
```

Four things differ from the M1 Max command at the top of this log:

- **`--moe-stream-cache 36`, not 32.** Cache 36 with MTP is clean here up to about 98k context.
- **`-c 98304`, not 131072.** 131072 with cache 36 and MTP hits a GPU OOM. Above ~98k, drop to
  cache 32 and the full 131072 works.
- **Do not set `LLAMA_MOE_STREAM_PLE_DIRECT`.** Measured level with the buffered default, so it
  buys nothing.
- **`--spec-max-prompt 0` (off), not 32768.** The flag counts the whole prompt and ignores the
  prompt cache, so in a running chat it disables MTP permanently once history crosses the limit -
  measured on live traffic, that cost 52 seconds on a single request. Only set a limit for one-shot
  long prompts. See
  [MTP across context](#mtp-across-context-and-where---spec-max-prompt-belongs).

Observed with this command at 96k: decode **25.77 t/s**, draft acceptance **0.90** (99 accepted /
110 generated, mean len 4).

### MTP across context, and where `--spec-max-prompt` belongs

**`--spec-max-prompt 32768` is too generous on this machine. For ordinary answer lengths the
threshold belongs at about 16k.**

Two passes, arm order reversed, one server load per arm with every context measured inside it.
Identical prompt files across arms, `cache_prompt` off, 256 tokens decoded, first request discarded.
Cache 36, `-c 98304`.

| ctx | prompt_n | pp off | pp on | prefill cost | tg off | tg on | decode gain | acc | repays after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4k | 4,619 | 344.91 | 318.61 | -7.6% | 18.71 | 26.95 | +44.1% | 0.550 | **68 tok** |
| 8k | 9,042 | 344.53 | 319.51 | -7.3% | 18.34 | 24.15 | +31.6% | 0.503 | **157** |
| 16k | 18,154 | 332.51 | 306.68 | -7.8% | 18.03 | 25.21 | +39.9% | 0.569 | **291** |
| 32k | 36,492 | 318.14 | 287.19 | -9.7% | 17.14 | 20.93 | +22.0% | 0.474 | **1,174** |
| 64k | 72,565 | 294.41 | 263.11 | -10.6% | 15.29 | 19.52 | +27.7% | 0.484 | **2,067** |
| 88k | 87,804 | 284.24 | 251.39 | -11.6% | 14.29 | 16.77 | +17.4% | 0.450 | **3,900** |

"Repays after" is end-to-end break-even: extra prefill seconds divided by seconds saved per
generated token. Below that answer length, turning MTP on is a net loss.

Pick the threshold from the answers you actually get back:

| typical answer | MTP pays up to | flag |
|---|---|---|
| ~256 tokens | 8k | `--spec-max-prompt 9042` |
| ~512-1024 tokens | 16k | `--spec-max-prompt 18154` |
| ~2048 tokens | 32k | `--spec-max-prompt 36492` |

**16384 is the sensible default** - it covers answers down to ~300 tokens and is where the curve
turns sharply: break-even quadruples between 16k and 32k (291 -> 1,174 tokens).

Three things differ from the M1 Max numbers higher up this log:

- **The decode gain is much smaller.** +44% at 4k here against +68% there, and +17% at 88k against
  +51% at 130k. MTP still pays, but not by the margin recorded.
- **The 32k break-even is ~1,174 tokens, not ~350.** The old figure is superseded.
- **Decode without MTP is far better** (18.7 vs 10.4 t/s at 4k), which is most of why the *relative*
  gain shrank. The head has less headroom to recover.

Reproducibility, since these are ratios of two measurements: the MTP arm repeated to under 1% at
every context across both passes, acceptance matching to three decimals. The no-MTP arm varied up to
7% on decode, which is the dominant error term. Swap growth was ruled out as a cause - the 32k
decode dip reproduced identically at 3,445 MB and 2,765 MB of swap.

Run it with `~/models/bin/mtp-baseline.sh`.

#### On live traffic the limit is a trap

The synthetic table above measures **cold** prefill: every request pays for the whole prompt. A real
chat does not. `can_speculate()` compares `task->n_tokens()` - the entire prompt - against the limit,
with no knowledge of how much is already cached, so once a conversation crosses the threshold MTP
switches off for the rest of the session.

Caught on a real session running `--spec-max-prompt 16384`:

| task | context | drafted | decode |
|---:|---:|---|---:|
| 0 | 402 | 0.590 | 25.78 t/s |
| 2 | 13,052 | 0.628 | 23.59 |
| 106 | 13,824 | 0.683 | 26.54 |
| 195 | 17,205 | **no** | **15.40** |
| 660 | 21,065 | **no** | **14.56** |
| 1023 | 27,488 | **no** | **16.17** |

The cutoff lands exactly at the limit, and decode drops ~40%. Task 1023 generated 3,928 tokens in
243 s; at the 20.9 t/s measured for that context it would have taken ~188 s, against roughly 2.4 s
of extra prefill on its 7,080 uncached tokens. **The limit spent 52 seconds to save 2.**

Note also that acceptance on real prose ran **0.59-0.68**, well above the 0.45-0.57 the synthetic
keyword prompts produced. MTP is worth more on real traffic than the table above suggests.

**So: `--spec-max-prompt 0` for interactive use.** A limit only makes sense for one-shot long
prompts with short answers, where the cache never helps - and there the break-even table applies.
Fixing this properly means counting uncached tokens rather than the whole prompt.

### Rebuilding the MTP draft head

The head is **not** produced by this repo's converter. `conversion/qwen4exp.py` sets
`supports_mtp_export = False`, and commit 542888199 ("qwen4exp: native MTP head support") changed
only C++, so the tree can read a head format it cannot write. Nothing on HuggingFace ships that
format either: every published MTP sidecar uses upstream naming.

It is built instead by rewriting an upstream sidecar, with `~/models/bin/mtp_sidecar.py`:

```bash
D=~/models/qwen38-flash-next-mtp
curl -fL --retry 5 -C - -o $D/mtp-src-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
python3 ~/models/bin/mtp_sidecar.py \
  --src $D/mtp-src-Q4_K_M.gguf \
  --out $D/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
```

Two renames and one split, byte exact - no requantization:

| upstream | this fork |
|---|---|
| `blk.48.nextn.eh_proj` `[5120, 2560]` | `nextn.fc_embd` + `nextn.fc_hidden`, `[2560, 2560]` each |
| `blk.48.nextn.hc_head_norm/down/up` | `blk.48.nextn.hc_norm/hc_down/hc_up` |

**Never pass `--swap-halves`.** Which half of `eh_proj` is `fc_embd` cannot be read off the file, and
the wrong choice does not error - it silently collapses drafting. Measured both ways: default gives
**0.50 acceptance, mean len 3.0**; swapped gives **0.00000, 0 accepted of 1186**. Always confirm
acceptance is near 0.50 after rebuilding, not just that the server started.

### Context vs cache size: where the config stops working

One short generation per arm, `vm.swapusage` sampled around each, everything but `-c` and the cache
budget held fixed, MTP on:

| ctx | cache | result |
|---:|---:|---|
| 32768 | 36 | clean, generates |
| 65536 | 36 | clean, generates |
| 98304 | 36 | clean, generates |
| 131072 | 36 | **GPU OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), 3.4 GB swapped |
| 131072 | 32 | clean, generates |

Raising `-c` alone takes the configuration from working to not working, so the full-context KV cache
really does compete with the expert cache. Metal does not quietly skip committing it.

RSS grows about 0.5 GiB per 32k of context, consistent with 24 KiB/token
(12 attention layers x 2 kv heads x 256 dim x 2 for K+V x 2 bytes).

### Cache 36 vs 32, interleaved

`-p 4096 -n 128 -r 2`, A/B/A/B, both pass-1 arms swap-clean:

| pass | cache | pp4096 | tg128 |
|---:|---:|---:|---:|
| 1 | 36 | 337.20 | 21.78 |
| 1 | 32 | 291.06 | 20.61 |
| 2 | 36 | 329.49 | 21.06 |
| 2 | 32 | 283.41 | 18.83 |

**Cache 32 costs ~14% prefill**, reproducing the M1 Max finding on different hardware and a
different checkpoint. The decode cost is noisier here than the ~2% recorded earlier.

**Practical rule: cache 36 up to ~98k context, cache 32 above it.** Dynamic KV laddering would only
help a session that starts short and later needs more than ~98k; it is not worth the invasiveness.

> [!IMPORTANT]
> **Superseded 2026-09-03: the default is now cache 32 at every context.** The rule above is
> correct on speed and was overruled on memory. Cache 36 does not leave the machine enough RAM to
> survive a coding session running alongside the model. See the next section for the numbers.

### What the model actually costs in memory

**Measured 2026-09-03 on this machine (M5 Pro, 64 GB), server running, `vm_stat` wired pages
against a 2.42 GiB idle baseline.** Every earlier entry in this log measured *speed* at a given
cache size; none of them recorded what the configuration costs in memory, which is why the machine
kept hanging.

| cache | model wired | peak under 3.7k prefill | free | new swapouts |
|---:|---:|---:|---:|---:|
| 36 | **51.40 GiB** | - | 0.06 GiB | swap already engaged, 9,428 |
| 32 | **46.96 GiB** | **47.13 GiB** (48,258 MiB) | 0.06 GiB | **zero** |

**Cache 36 wires 51.4 of the 64 GiB, and that was measured with omp NOT running** - only an editor,
Drive and a terminal alongside. That is the state the machine hung in on 2026-09-03, when a 3h13m
omp session against this server froze the UI for 32 s at a stretch while using 121 ms of CPU.

Dropping to cache 32 frees 4.44 GiB and the prefill test produced **zero new swapouts**. Note that
`free` reads 0.06 GiB in both arms and is *not* the signal to watch - macOS fills free memory with
cache. Swapouts and the compressor are the real indicators.

**`iogpu.wired_limit_mb` is the safety cap, and it was set backwards.** The launch script raised it
to 57344 (56 GiB), which leaves macOS ~8 GiB and cannot be reclaimed because wired memory never
pages out. It is now 50176 (49 GiB), about 1.9 GiB over the measured cache-32 peak. The point of a
*lower* cap is the failure mode: past it Metal raises a GPU OOM the server reports and survives,
instead of the whole laptop going unresponsive. 49152 (48 GiB) was tried and rejected - only
894 MiB over the peak, and it is *below* what cache 36 needs, so it would not load at all.

**The wave-cap tuning survives the cache reduction.** Slots per layer go 289 -> 257 of 512, and the
`LLAMA_MOE_STREAM_WAVE_CAP=200` optimum still applies and still yields 3 waves on a large batch
(`wave cap = 200 of 257 slots, 3 waves, partition ON`). The small-batch default moves 139 -> 123
with the wave count unchanged at 4. **This does not extend downward:** the cap is clamped to
`[n_expert_used, n_slots - n_expert_used]`, so below roughly a 26 GiB cache the ceiling falls under
200 and the tuning would be silently clamped away.

**Quality is untouched, exactly.** The expert cache decides which weights are already resident
versus re-read from disk. Same weights, same arithmetic, same outputs - unlike `--cache-reuse N`,
which is genuinely approximate and stays off.

**`--cache-ram` was never set and defaulted to 8192 MiB.** That is a host-RAM store of saved
prompt/KV states, separate from the live slot's KV and from `cache_prompt` prefix reuse. In the
3h13m session it appeared six times and **all six were evictions**, discarding entries of 4.87 and
4.67 GiB, while 77 of 82 slot selections hit the live slot by prefix similarity. It was holding
multiple GiB to serve nothing. Now 512.

### PLE reads: buffered vs uncached

**Verdict: keep buffered. Uncached reads make no measurable difference.**

`LLAMA_MOE_STREAM_PLE_DIRECT` opens the PLE file with `F_NOCACHE` and reads uncached. Two
interleaved pairs on a fixed build, order reversed for the second pair, 11196-token prefill with
`cache_prompt` off, 256 tokens decoded, first request of each arm discarded as cold:

| arm | order | pp | tg |
|---|---|---:|---:|
| buffered | first | 313.37 | 24.03 |
| direct | second | 317.98 | 22.91 |
| direct | first | 321.84 | 22.96 |
| buffered | second | 319.21 | 22.65 |

Means: **pp 316.3 -> 319.9 (+1.1%), tg 23.34 -> 22.94 (-1.7%)**. Both gaps sit inside the buffered
arm's own run-to-run spread (2.7% on prefill, 5.9% on decode across four loads), so neither is a
real effect. Swap fell slightly during every arm, buffered included.

#### The earlier +7%/+21% result was an artifact

The first version of this measurement was invalid, for two independent reasons.

**The read path was broken.** A direct read is block-aligned: it rounds the offset down to a 4096
byte boundary and reads a whole block. The PLE branch handed it `w.dst`, the **90-byte** row
destination, so every uncached row read wrote up to 8 KB into a 90-byte buffer and then returned
the row at `staging + head`, an address the caller never reads. `w.dst` got the aligned block start
instead of the row. The model was computing on wrong PLE values while overrunning its own buffer.
The expert path had this right all along, and its comment says so: the direct path keeps a staging
buffer because block-aligned reads "would scribble outside the slot". The fix bounces the row
through that same staging buffer and copies `row_size` bytes out.

Under a real prefill the bug is not subtle. It fails outright:

```
decode() failed: PLE row streaming: a previous model read failed
```

**The benchmark measured nothing.** The harness sent the same short prompt three times with the
prompt cache on, so requests 2 and 3 prefilled 4 tokens. `prompt_per_second` over 4 tokens is noise,
and 4 tokens is few enough PLE rows that the overflow stayed under the crash threshold. That is why
the broken path looked fast rather than looking broken.

Two smaller harness bugs sat underneath: a `local path` in the zsh driver shadowed `PATH` (zsh ties
the two), and piping response JSON through zsh's `echo` expanded the `\n` escapes inside string
values, so every parse failed into a `|| echo 0` fallback.

**Caveat:** these arms ran with 1.5-1.9 GB of swap already in use, not from a cold boot. That is
not the confound it would have been for the old claim: the buffered arm scored the same at 198 MB
of swap as at 1.9 GB, so the page-cache-competition mechanism the old entry proposed does not show
up at this scale either.

### The MoE dispatch threshold is correct

Metal takes the grouped matrix path for `MUL_MAT_ID` only at batch >= 32 (`ne21_mm_id_min`); an MTP
verify batch of 5 falls to the per-token vector path. The vector path does not amortise the weight
read at all - `test-backend-ops perf`, one shape:

| n | us/run | us/token | TFLOPS |
|---:|---:|---:|---:|
| 1 | 74.93 | 74.93 | 1.34 |
| 2 | 144.66 | 72.33 | 1.39 |
| 4 | 287.98 | 72.00 | 1.40 |
| 8 | 572.37 | 71.55 | 1.41 |

Only 4.5% per-token improvement from n=1 to n=8. But **forcing the matrix path at small batch is far
worse**: with the threshold dropped to 2, the same n=8 case runs 1252 us and 1453 us against 600 us
and 572 us, i.e. **2.1x to 2.5x slower**, 0.64/0.55 TFLOPS against 1.34/1.41.

So the threshold is right, and cross-token expert dedup is the only remaining lever below it - worth
about 6% at batch 5, since only ~3 of 50 expert draws from 512 collide. Not pursued.

### Unsloth variants: full audit, and what is worth borrowing

Every published `unsloth/Qwen3.8-Flash-Next-GGUF` variant, read from the GGUF headers themselves
(remote ones over HTTP range requests, `~/models/bin/gguf_probe.py`). "cache fit" is what fraction
of the expert weight the 36 GiB expert cache can hold.

| model | total | experts | dense | PLE | cache fit | expert format |
|---|---:|---:|---:|---:|---:|---|
| UD-IQ3_XXS | 76.3G | 45.3G | 4.21G | 26.8G | 79% | IQ2_S, i-quant |
| UD-Q3_K_XL | 83.8G | 52.0G | 4.98G | 26.8G | 69% | IQ3_XXS, i-quant |
| UD-IQ4_XS | 87.2G | 55.4G | 4.98G | 26.8G | 65% | IQ3_S, i-quant |
| **local Q4_0-Q8out** | **93.8G** | **63.6G** | **3.43G** | **26.8G** | **57%** | **Q4_0** |
| UD-Q4_K_XL | 103.7G | 71.7G | 5.13G | 26.8G | 50% | Q4_K / Q5_1 |
| UD-Q5_K_XL | 147.4G | 91.6G | 5.13G | **50.7G** | 39% | Q5_K |
| UD-Q6_K_XL | 157.5G | 101.7G | 5.13G | **50.7G** | 35% | Q6_K |
| Q8_0 | 175.3G | 119.5G | 5.09G | **50.7G** | 30% | Q8_0 |

**Two facts settle which variants are even candidates.**

- **Q5_K_XL and above double the PLE table, 26.8 -> 50.7 GiB.** That table is streamed per row on
  every token, so it is the worst place in the model to spend bytes. Disqualifying, whatever the
  quality.
- **Everything below the local checkpoint is i-quant** (92-100% of expert weight). Measured here:
  `q4_0` beats `iq3_xxs` by 27% at decode *while reading 47% more bytes*. Smaller and slower.

That leaves **UD-Q4_K_XL** as the only credible whole-model alternative, and it costs 13% more
expert bytes for K-quant experts and a better dense trunk.

The local checkpoint is already at the speed optimum: fastest Metal kernels, highest expert-cache
residency of any non-i-quant option. **Its one real deficit is the dense trunk, 3.43 GiB against
unsloth's 4.98-5.13 GiB.** That is resident memory, not streamed, so closing it costs nothing per
token. This is what makes selective borrowing better than switching models.

#### Corrections to the earlier audit

An earlier version of this section (2026-09-01) got the shape right and three numbers wrong. Deltas
below are measured, and are *increments over the local tensor*, not unsloth's group totals:

| item | earlier claim | measured |
|---|---|---|
| hyper-connections | +630 MB | **+0.30 GiB** (630 MB is unsloth's total for the group) |
| shared experts | +245 MB | **+0.04 GiB** (local already has half at Q8_0) |
| vision projector | 1.1 GiB | 0.84 GiB |
| attention projections | *not listed* | **+0.92 GiB, the largest dense gap** (all 120 `attn_*` at Q8_0 vs mostly Q4_0) |
| `ssm_out` | *not listed* | +0.13 GiB |
| `output.weight` | *not listed* | **local is already better**: Q8_0 vs unsloth's Q6_K. Do not take theirs |

Confirmed exactly: the IQ3_S gate/up finding, dense trunk 4.98 vs 3.43 GiB, Q8_0 down experts on
layers 2/4/30/46/47, `token_embd` +0.30 GiB, and the PLE table at zero cost - both are
28,800,138,240 bytes, dims `[160, 320001536]`, 90-byte rows. `ple.type` and `ple.row_size` are read
from the tensor (`llama-moe-stream.cpp`), so the streaming path is type-generic, and Metal has
`get_rows_iq4_nl`.

#### The plan

Build **one** spliced checkpoint rather than three: each build is ~97 GiB of writes, and only the
PLE swap carries a kernel-speed risk worth isolating.

| step | what | cost |
|---|---|---|
| 1 | Quality harness first: fixed `llama-perplexity` subset, run on the current checkpoint as the baseline | ~1 h |
| 2 | Generalise `gguf_splice.py` to "take these tensor groups from that donor" | - |
| 3 | Build one checkpoint: PLE `IQ4_NL` (+0.00), attention (+0.92), `hc_*` (+0.30), `token_embd` (+0.30), `ssm_out` (+0.13), shexp (+0.04), the 5 Q8_0 down-expert layers (+1.66) | ~97.2 GiB |
| 4 | Speed A/B against the current checkpoint, then perplexity against the step-1 baseline | ~2 h |
| 5 | Bisect only if speed regressed; the PLE swap is the first suspect | - |

Deferred: **UD-Q4_K_XL** (103.7 GiB download) only if step 4 shows quality is still short.

**State the evidence bar before measuring.** Perplexity differences from splices this size may be
smaller than a scoped run can resolve. Going Q4_0 -> Q8_0 on the dense trunk is unambiguously higher
fidelity; the open question is whether it is *observable*. If perplexity cannot separate them, the
honest conclusion is "no measurable harm, theoretically better, free at runtime" - not "improved".

#### Measuring quality: headline perplexity is not sensitive enough

Baseline on the current checkpoint, `wiki.test.raw`, `-c 4096 -b 4096 -ub 4096`, cache 36, 20 chunks:

```
Final estimate: PPL = 4.5794 +/- 0.05253      (13.90 s per pass)
```

**That error bar is +/-1.15%, and these splices should move perplexity by a few tenths of a percent.**
Comparing two headline numbers therefore proves nothing. The test set holds only ~80 chunks at this
size, so even using all of it only reaches +/-0.58%.

**Use a paired per-chunk comparison instead.** Chunk difficulty varies far more than the models do
(the running mean moved 4.29 -> 4.58 within this one run), and that variance is identical for both
models, so it cancels in the difference. `llama-perplexity` prints a running mean after each chunk,
from which each chunk's own NLL follows exactly:

```
NLL_i = i*ln(running_i) - (i-1)*ln(running_{i-1})
```

Run both checkpoints over the same chunks, difference them per chunk, and test whether the mean
difference is distinguishable from zero. Same compute, far more resolution. `~/models/bin/ppl_pair.py`
does the extraction and the test.

#### Result: the borrowed checkpoint is 18% better on perplexity

Built with `~/models/bin/gguf_splice_groups.py --groups attn,hc,token_embd,ssm_out,shexp,ple,down5`:
691 tensors replaced, 93.82 -> 97.37 GiB. Both checkpoints over the same 40 chunks:

```
headline PPL  baseline : 5.2777 +/- 0.04479
headline PPL  borrowed : 4.3290 +/- 0.03399      (-17.98%)

paired mean NLL diff   : +0.19815 +/- 0.02512 (1 s.e.)
  t = +7.89 over 39 d.o.f.,  better on 40/40 chunks
```

**40 of 40 chunks, t = 7.9.** Not noise. The borrowed model also generates coherent, accurate prose,
so this is not a broken model producing a flattering loss.

#### Bisection: the PLE table contributes nothing; the dense trunk carries all of it

The obvious explanation was the PLE table - 26.82 GiB, ~29% of the model, read on every token at
every layer, `Q4_0` symmetric with no zero-point against a fitted `IQ4_NL` codebook at the same
bytes. **That explanation is wrong.** Same paired test, same 40 chunks:

| variant | PPL | vs baseline | t | chunks better |
|---|---:|---:|---:|---:|
| baseline | 5.2777 | - | - | - |
| PLE swap only | 5.1708 | -2.03% | 1.98 | 26/40 |
| everything EXCEPT the PLE swap | 4.3148 | **-18.24%** | 7.80 | **40/40** |
| both | 4.3290 | -17.98% | 7.89 | 40/40 |

**The 3.55 GiB dense trunk carries the entire effect.** The PLE swap does not clear significance on
its own, and adding it to the trunk changes nothing (-18.24% -> -17.98%, inside noise).

**So drop the PLE swap.** It buys nothing measurable and it was the only component carrying a
speed risk (`IQ4_NL` `get_rows` on the streamed path). The recipe to adopt is:

```
--groups attn,hc,token_embd,ssm_out,shexp,down5      # +3.55 GiB, -18.24% perplexity
```

Still unattributed inside the trunk. The two candidates by prior are `down5` (1.86 GiB, the five
layers unsloth protects for high activation kurtosis) and `token_embd` (Q4_0 symmetric on a 248k
vocab - the same failure mode wrongly attributed to the PLE table above). If one dominates, the
recipe could shrink well below 3.55 GiB.

#### Speed: 18% better quality costs 3.3% decode

Same harness as the tuning sweep - real ~24k prompt, two passes with order reversed, tuned MTP flags
(n-max 3, p-min 0.3) held constant, only `-m` differs:

| | prefill | decode | draft acceptance |
|---|---:|---:|---:|
| baseline | 278.46 | 26.64 | 0.751 |
| v2 (`attn,hc,token_embd,ssm_out,shexp,down5`) | 278.02 | 25.77 | **0.871** |
| delta | -0.2% | **-3.3%** | **+16.0%** |

v2 repeated to 0.1% on decode across both passes; the baseline's own spread was 2.5%, so the 3.3%
cost is real but small. **Prefill is unchanged.**

**Draft acceptance rose 0.751 -> 0.871 with an unchanged draft head.** The head is the same
`Q4_K_M` sidecar; a higher-fidelity target simply agrees with its drafts more often. That is a
second, independent confirmation that the trunk splice improved the model - acceptance is measured
by token agreement, not by loss.

**Verdict: adopt.** -18.24% perplexity for -3.3% decode and +3.55 GiB is a good trade.

#### Attribution: hyper-connections carry the quality, but perplexity is the wrong target

Each group spliced on its own, same paired 40-chunk test against the same baseline:

| group | size | perplexity | t | chunks better | per GiB |
|---|---:|---:|---:|---:|---:|
| **hc** | +0.30 | **-13.14%** | 6.53 | **40/40** | **-43.8%** |
| **attn** | +0.92 | **-8.80%** | 6.01 | 38/40 | -9.6% |
| token_embd | +0.30 | -2.41% | 2.76 | 25/40 | -8.0% |
| down5 | +1.86 | -1.53% | 2.23 | 23/40 | -0.8% |
| ssm_out | +0.13 | -0.37% | 0.30 | 27/40 | - |
| shexp | +0.04 | +0.01% | -0.01 | 22/40 | - |

**Hyper-connections are the single biggest lever in the model: -13% perplexity for 0.30 GiB.** They
are the 4-stream residual mixer, run on every token at every layer, and the local checkpoint had
them at Q4_0. The chunk-win column is the robustness check: 40/40 and 38/40 are unambiguous, while
`token_embd` and `down5` clear t>2 while winning barely half the chunks.

**But minimising on perplexity is a trap.** Combined builds, all statistically identical on
perplexity, differ five-fold in decode:

| build | groups | size | perplexity | draft acceptance | decode |
|---|---|---:|---:|---:|---:|
| v2 | all six | +3.55G | -18.24% | 0.871 | -3.3% |
| **v3** | no down5 | **+1.69G** | -17.79% | 0.817 | **-2.7%** |
| v4 | attn,hc | +1.22G | -17.70% | not measured | - |
| v5 | attn,hc,token_embd | +1.52G | -17.52% | **0.710** | **-14.0%** |

**Decode tracks draft acceptance, not size and not perplexity.** `ssm_out` and `shexp` contribute
nothing measurable to perplexity and are worth ~11 points of decode, because they lift acceptance
from 0.71 to 0.82. v5's acceptance is *below* the unspliced baseline's 0.751.

The likely mechanism: the MTP sidecar is derived from unsloth's own quantisation, so the draft head
agrees best with a target whose numerics resemble the model it came from. The closer the trunk gets
to unsloth's precision profile, the more drafts survive verification.

**Adopt v3: `--groups attn,hc,token_embd,ssm_out,shexp`, +1.69 GiB, -17.79% perplexity, -2.7%
decode.** Do not minimise the group list on perplexity alone - v4 and v5 look equivalent on quality
and are not, and v4's acceptance was never measured.

## Negative results

| Idea | Result |
|---|---|
| **MTP + n-gram stacked** | **net negative** — they compete for the same accepted tokens |
| **MTP at long context** | **inverts.** At 32k the head costs 3.7 s of prefill and repays after 119 generated tokens; at 128k it costs 57.7 s and repays after 2369. Hence the length gate. |
| **union-8 for QSA** | works, knee at kv≈20k — but block-topk got there first, and the payoff is small |
| **union-8 for MTP** | does not apply |
| **`hc_up` matmul at K=320** | a wash, not a win |
| **Upstream `5ea1b124` fa-vec tunings** | **rejected** — no F16 entry matches head dim 256/256; the one matching shape exists only for quantised KV, which is out |
| **Deeper drafts** | the expert GEMV n-curve is linear with a cliff at 32; depth does not amortise the weight read |

---

## Retractions

**"union-8 gives +4.3% / +6.1%."** Voided — measured on a **broken kernel**.

**"The dk256 union failure is a head-dimension problem."** Also wrong. The root cause was the
**CPU reference's** GQA head mapping; the Metal kernel had been correct all along.

**"Long-context decode collapses."** Wrong. The measurements behind it were 1–8 token generations
that hit EOS immediately. Measured properly with `ignore_eos`, decode at 122k is 7.13 (no spec) /
8.63 (MTP).

**"Block-topk is worth +18.5%."** Under-reported — measured at 32k, where the quadratic is only 18%
of prefill. At 128k it is worth far more.

**Upstream's `edb6dec1c` "enable recurrent state rollback" looks wrong**: it adds `QWEN4EXP` to the
whitelist, but their tree has neither the ring-bank conv writes nor the delta-net `n_written < K`
clamp, both prerequisites. Two upstreamable bug reports stand from this log — that one, and the
CPU-reference GQA mapping.

---

## Sweeps

| Sweep | Outcome |
|---|---|
| **ubatch / batch** | keep **4096** |
| **Cache size** | 36 is ~14% faster on prefill, ~2% on decode — but 36 + MTP pages. Use **32 with MTP** |
| **MTP quantization** | `Q4_0` (2.20 GiB); Q8_0 (3.85 GiB) gives similar acceptance (0.55 vs 0.51 mean) for 1.75x the footprint |
| **MTP n-max** | **4**; deeper drafts do not pay |
| **I/O threads with MTP** | keep **8** |
| **Speculation type x cache** | MTP alone beats MTP+n-gram and n-gram alone |

---

## Measurement discipline

- **The first request understates decode by ~25%.** Cold 12.20 → warm 16.49 / 16.26 t/s. Several
  single-shot figures in this log's history were cold and therefore pessimistic. Multi-arm sweeps
  issuing 3+ requests per arm are unaffected; one-shot verification runs are not.
- **That ~25% is a COLD-START effect, not a warm-vs-post-prefill one.** Measured directly, warm
  decode runs only 2-16% above post-prefill decode at the same depth. The two were conflated here
  previously. Post-prefill is the honest number for agentic use, where every turn re-prefills.
- **"It loaded" is not "it fits".** Cache 36 + MTP loads and runs, and pages while doing it. Sample
  `vm.swapusage` around every arm; a rising figure voids the measurement.
- **Warm decode measured once per context carries ~15% variance** — in one sweep the 8k warm figure
  came out *below* its own post-prefill figure, which is physically impossible. Prefer post-prefill
  for A/B work, or repeat the warm arm.
- **A replay trap can fake +33%.** n-gram speculation replaying its own prior output is not a
  measurement of anything.
- **Percentages on a curve are meaningless without the length.** See prefill, above.
- **Check the harness before the model.** A readiness loop waiting for the wrong log string looks
  exactly like a hung server. The line is `llama_server: listening on http://`.
- **Audit 3-way patch applies for silently reverted hunks** — conflict markers show only part of the
  damage; patch *context* can overwrite untouched work.

---

### Decode levers at 44.8k: speculation is already optimal

Two levers, measured at 44,808 tokens on a prompt built from recorded omp sessions - the length
this box actually works at, not the 24k the earlier sweep used. Two passes, order preserved,
v3 + the Q4_K_M head throughout.

| config | prefill | decode | vs base | acceptance |
|---|---:|---:|---:|---:|
| base (n-max 3, p-min 0.3, cache 36, ctx 98304) | 270.82 | 23.48 | - | 0.868 |
| **cache 42, ctx 65536** | 269.29 | **24.00** | **+2.2%** | 0.844 |
| n-max 4 | 266.65 | 22.84 | -2.7% | 0.771 |
| p-min 0.5 | 269.69 | 22.21 | -5.4% | 0.901 |
| n-max 2 | 270.23 | 21.55 | -8.3% | 0.903 |

**`n-max 3` is bracketed on both sides and `p-min 0.3` beats 0.5.** The hypothesis behind the sweep -
that acceptance falls at long context and should favour shallower drafts - is wrong: acceptance at
44.8k is 0.844 against 0.850 at 24k. It barely moves. The 24k tuning transfers.

Note the shape of the losing arms: **`n-max 2` and `p-min 0.5` both reach ~0.90 acceptance and are
the two slowest.** Acceptance is not the objective. Tokens per unit time is, and a shallower or more
timid draft produces fewer tokens per verify step.

The cache arm is the only win, and its mechanism is visible in the numbers: **acceptance is
identical (0.844 vs 0.844) and prefill is flat**, so nothing about speculation changed - only how
much expert weight fits in cache. It is small, though: +2.2% against a baseline whose own two passes
differed by 1.8%. Also note the baseline's acceptance swung 0.844 -> 0.892 between passes, so small
acceptance differences between configs mean nothing.

**Not adopted.** Dropping ctx to 65536 forces omp's contextWindow to ~64,512 and compaction to
31,744 - below the ~38k sessions actually reach. One extra compaction costs minutes; 2.2% of decode
saves seconds.

### A bigger draft head is a loss

`Q8_0` head (3.85 GiB) against the shipped `Q4_K_M` (2.59 GiB), same target, 24k prompt:

| head | size | prefill | decode | acceptance |
|---|---:|---:|---:|---:|
| Q4_K_M | 2.59 GiB | 280.75 | **26.16** | 0.850 |
| Q8_0 | 3.85 GiB | 280.58 | 24.80 | 0.888 |
| | +49% | -0.1% | **-5.2%** | +4.5% |

Speculative decoding is lossless - the target verifies every drafted token - so a better head can
only ever be a speed change. It was a negative one. **The head runs `n-max` times per verify step
while the target runs once**, so its cost carries a 3x multiplier, and at 0.850 acceptance only 16%
of headroom remained in tokens-per-step.

Solving the cost model from the two points: **3.18 ms per GiB per draft pass; the head is ~18% of a
decode step.** Break-even acceptance for a *smaller* head: 2.26 GiB needs 0.813, 1.78 GiB needs
0.783. `Q5_K_M` and `Q6_K` heads would be **larger** than `Q4_K_M`, so they move the wrong way, and
`BF16` (7.24 GiB) is far worse. An F32 head is meaningless - unsloth's source is BF16, so it would
be an upcast carrying no extra information.

### Where decode time actually goes (GGML_METAL_KPROF, 2026-09-03)

First op-level profile of this model. `GGML_METAL_KPROF=1`, v3 + Q4_K_M head, n-max 3, 64 decoded
tokens, 850 decode-graph executions attributed:

| op | ms | share |
|---|---:|---:|
| MUL_MAT | 876.3 | 43.2% |
| MUL_MAT_ID (MoE experts) | 405.1 | 20.0% |
| **CPY** | **330.7** | **16.3%** |
| GATED_DELTA_NET | 56.2 | 2.8% |
| CONT / MUL / ADD / GET_ROWS | ~190 | ~9% |

Top individual nodes:

```
14.9%  303.4 ms  CPY  cache_s_l2 (view) (copy of (view))
 7.9%  160.2 ms  MUL_MAT_ID  ffn_moe_down-1
 6.8%  137.1 ms  MUL_MAT_ID  ffn_moe_gate-1
 6.0%  120.9 ms  MUL_MAT  node_469
```

**The single most expensive node in a decode step is a state copy, not a matmul.** It is the
recurrent conv-state write-back in `build_conv_state` (`src/models/qwen4exp.cpp:1548`). With
speculation on, `n_rs_seq = draft.n_max` (`common/common.h:404`), and the graph writes
`K = n_rs_seq + 1` banks per SSM layer per step:

```c
const int64_t K = (int64_t) cparams.n_rs_seq + 1;
for (int64_t t = 1; t <= K; ++t)
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
```

At n-max 3 that is **4 conv-state copies per layer per step across 36 SSM layers**.

Two consequences:

- **The depth sweep was measuring two costs, not one.** Deeper drafts run the head more times *and*
  add a bank per layer per step. That is why n-max 4 lost as clearly as it did.
- **The DeltaNet computation is 2.8%; the copying around it is 16.3%.** Any plan to port a faster
  GDN kernel (e.g. from Perplexity's Lily) is chasing the wrong 3%.

#### Closed: the CPY share is a profiling artifact

Chased to the end, and **there is nothing to fix here.** Three corrections, each overturning the
last:

1. **16.3% was share within one subgraph.** A 48-layer model does not fit in 167 nodes; decode spans
   several graphs. Across all of them `CPY` is 11.5%, of which `cache_s*` is 10.4%.
2. **It is not the `K = n_rs_seq + 1` conv ring.** Profiling at n-max 3 (K=4) and n-max 1 (K=2) gave
   5.42 and 5.11 ms per token - **6% apart for double the banks.** The real node is the "extra
   states" copy in `build_rs` (`src/llama-graph.cpp:3830`).
3. **That copy is ~0.24% of decode, not 10%.** Sized against measured hardware:

| | |
|---|---|
| one SSM state row | 786,432 elements = 3.0 MiB fp32 |
| traffic per decode step | 216 MiB (36 layers, read + write) |
| measured bandwidth ceiling | **668 GB/s** (`test-backend-ops perf -o ADD`, 24 MB buffer) |
| so the copy costs | **0.34 ms/step = 0.094 ms/token** |
| against a 40 ms/token step | **0.24%** |
| KPROF claimed | 5.42 ms/token, 10.4% |

**KPROF charges roughly 500 us to every node it samples.** A 3.0 MiB copy is ~9 us of real work, so
it appears ~60x more expensive than it is. Large `MUL_MAT` nodes carry enough real work to swamp the
overhead, small nodes do not - so **KPROF systematically over-weights cheap ops**. Treat its output
as a node inventory, not a cost model. `test-backend-ops perf` gives trustworthy per-op numbers.

The lesson generalises: `MUL_MAT` at 48% and `MUL_MAT_ID` at 18.9% are the only entries in that
table large enough to trust, and they say what was already known - this model is dominated by dense
matmul and MoE dispatch, both already investigated.

### Streaming toggles: two real wins, both lossless

Ten env toggles had never been measured here. Reading them first was cheaper than sweeping them:
`LRU` self-documents as "+11% misses" and was skipped on that basis, and the `WAVE_CAP` comment
named the open question directly - whether the masked GEMM passes are "FLOP-expensive or
bandwidth-cheap". They are expensive.

**`LLAMA_MOE_STREAM_WAVE_CAP=200`.** Default is `(n_slots - n_expert_used)/2` = 139 of 289 slots,
4 waves. Every wave re-runs the expert GEMM over the whole token set and masks the rest away.
44.8k context, two passes each:

| cap | target waves | prefill | decode |
|---:|---:|---:|---:|
| 139 (default) | 4 | 268.30 | 22.43 |
| **200** | **3** | **278.39 (+3.8%)** | **23.51 (+4.8%)** |
| 279 (ceiling) | 2 | 271.78 (+1.3%) | 23.48 (+4.7%) |

**200 is an interior optimum** - past it the lost preload overlap costs more than the saved masked
passes, exactly the trade the code comment describes. The cap arms repeated to 0.1-0.4% against the
default's 1.4-1.6%.

**`LLAMA_MOE_STREAM_PARTITION=1`.** Ships disabled. Gives each (token, expert) pair to exactly one
wave instead of masking, removing the waste rather than reducing it. At 44.8k with cap 200:

| | prefill | decode | acceptance |
|---|---:|---:|---:|
| masked (default) | 278.27 | 23.53 | 0.844 |
| **partition** | **297.87 (+7.0%)** | **24.06 (+2.3%)** | 0.844 |

**Paired 40-chunk perplexity: bit-identical.** All 40 chunks matched exactly, PPL 4.3387 both ways.
It changes how pairs are assigned to waves, not what is computed - so this is free speed with no
quality question to answer.

Together, roughly **+11% prefill and +7% decode** over stock, both adopted in
`~/models/bin/qwen-q40-server.sh` (`WAVE_CAP=`/`PARTITION=` back them out).

## Backlog, in priority order

### 1. KV cache quantization - frees memory without giving up context

`-ctk q8_0 -ctv q8_0`. KV is `f16` by default; halving it frees memory for the expert cache **at the
same context length**, which is what the cache-42 arm achieved only by cutting context. At ~85 KB per
token, 98,304 context is roughly 8 GB, so this could fund `--moe-stream-cache 40` with ctx intact.

**Not lossless** - it changes stored attention keys and values. Needs paired perplexity first, then a
speed A/B at cache 40 / ctx 98304. Unknown to check in the same run: this model is 36 SSM layers and
only 12 attention layers, so KV may be a smaller share of memory than 85 KB/token implies, in which
case the freed memory never materialises. ~90 min.

### 2. The remaining streaming toggles

`WAVE_CAP` and `PARTITION` are done and adopted (above). `LRU` is skipped - its own comment records
+11% misses. Left, all lossless:

- **`HOT_DECAY`** - eviction halves route-hotness counters every 64 remap calls; the comment says
  that constant "has never been measured" and that miss rate is now the lever that matters.
- **`PAIR_SLACK`** - default 50% slack on per-wave pair capacity. Now more interesting than before,
  since partitioning made the pair lists the thing that is actually sized.
- **`SPEC_MAX`** - prefetch queue depth, default `n_io_threads * 8` = 64.
- `GPU_SLOT` (modes 0-3, experimental), `NO_PRELOAD` / `NO_HASH_PREFETCH` / `NO_ZEROCOPY` (disable
  switches for existing optimizations - useful for attribution, not as wins).

### 3. A smaller draft head - lossless, modest

Only *below* `Q4_K_M` can win. A `Q3_K_M` head (~2.26 GiB, built with `llama-quantize` from the BF16
source) must hold acceptance above **0.813**; `shared-Q4_K_M` (1.78 GiB) needs 0.783 but omits
`token_embd`/`output` and may not load. Expect a few percent at best. ~40 min.

### 4. cache 40 / ctx 81920 - lossless, small

The middle setting the cache-42 arm suggests: keeps omp compaction near 49k, above the ~38k working
range, while recovering part of the +2.2%. ~50 min.

### 5. The MTP head runs dense despite shipping sparse weights

A code change rather than a measurement, and the head is ~18% of a decode step, so making it sparse
is worth real time. Unscoped.

### 6. F16 `256/256` flash-attention vector tunings

Upstream never tuned this shape. Code + tuning generation, unscoped.

---

## Housekeeping

- **Nothing is committed.** The PLE buffer-overflow fix in `src/llama-moe-stream.cpp`, both doc
  rewrites, and this whole log are uncommitted.
- **Decide whether to keep `LLAMA_MOE_STREAM_PLE_DIRECT`.** Correct now, off by default, buys
  nothing measurable. The alternative is deleting it.
- `~/models/qwen38-flash-next-v2` (97 GiB) is superseded by v3 and can be deleted.
- Re-baseline the pre-M5-Pro half of this log, or mark it clearly as historical. Every figure above
  the "Measured on M5 Pro" section is M1 Max on a checkpoint that no longer exists.

## Closed

- **`--spec-max-prompt`**: set to 0. The limit is cache-blind and disabled MTP mid-conversation.
- **Draft depth and p-min**: `n-max 3`, `p-min 0.3`, bracketed at both 24k and 44.8k.
- **io-threads and `-ub`**: 8 and 4096 already optimal; `-ub 8192` GPU-OOMs at cache 36.
- **A bigger draft head**: measured -5.2%. Q5/Q6/BF16/F32 all ruled out by the cost model.
- **Per-block bitmap for union-8**: lifted the ceiling 4x, no throughput gain at reachable contexts.
- **MXFP4 vs block-level top-k attribution**: blocked, the `UD-iQ4_K_XXS` checkpoint was deleted.
