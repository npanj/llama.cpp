# Qwen3.8-Flash-Next (`qwen4exp`)

> Research log. Conditions are quoted with every figure — in this log more than most, because
> several measurements here were later found to have been taken under conditions that invalidated
> them. See [Measurement discipline](#measurement-discipline).

> [!IMPORTANT]
> **The machine changed.** Figures carried over from before 2026-09-03 were taken on an
> **M1 Max, 64 GB, ~400 GB/s**. This laptop now reports **Apple M5 Pro, 64 GB**, which runs this
> model at roughly 2x the M1 Max figures. Never compare a measurement against an older one without
> checking which machine it came from.

Where DeepSeek is a streaming problem, Qwen3.8-Flash-Next is a **graph-shape** problem. It decodes
fast enough that the bottleneck moved off the disk entirely and onto GPU work the graph did not need
to be doing: needless copies, broken kernel fusions, and an indexer that materialised a table
proportional to context × chunk size on every layer.

Three architectural features make it unusual, and all three cost real work to support:

- **Hyper-connections** — four parallel residual streams instead of one, mixed by a low-rank gate.
  Roughly 30% of decode GPU time, and the source of several fusion-breaking reshapes.
- **A 26.82 GiB PLE table** — far larger than RAM, touched a few rows at a time.
- **Gated DeltaNet on 36 of 48 layers**, with 12 full-attention layers carrying a sparse indexer.

---

## At a glance

| | |
|---|---|
| Layers | 48 — 36 SSM (gated DeltaNet) + 12 full attention |
| Experts | 512 per layer, 10 active per token |
| Attention | 24 heads, 2 KV heads, `key_length = 256` |
| Indexer | 4 heads, `key_length = 128`, `top_k = 2048`, **`compress_ratio = 4`** on attention layers |
| Hyper-connections | `hc = 4` streams |
| Expert shape | `ffn_gate/up_exps` 2560 → 640, `ffn_down_exps` 640 → 2560 |
| PLE table | 26.82 GiB, 90-byte rows — streamed, never resident |
| Checkpoint | **`Q4_0-Q8out-v3`, 95.5 GiB, 3 shards** (custom, see below). `UD-iQ4_K_XXS` at 82.89 GiB is the smaller alternative |
| Speculation | **native MTP head**, shared (`mtp-shared-Q4_K_M.gguf`, 1.776 GiB), `n-max 3`, `p-min 0.3` |
| KV | F16 |

**Do not launch this by hand. The recipe is `~/models/bin/qwen-q40-server.sh`,** which is the
authoritative configuration and carries the measurement behind every setting in its comments. It is
outside this repo, so what follows is a mirror, not the source of truth.

```bash
~/models/bin/qwen-q40-server.sh          # cache 34, ctx 98304, MTP n-max 3 / p-min 0.3
```

Which expands to roughly:

```bash
LLAMA_MOE_STREAM_LOOKAHEAD=1 LLAMA_MOE_STREAM_WAVE_CAP=200 LLAMA_MOE_STREAM_PARTITION=1 \
llama-server -m <first shard> -md <mtp-shared head> \
  -ngl 99 --moe-stream --moe-stream-cache 34 --moe-stream-io-threads 8 \
  -c 98304 -b 4096 -ub 4096 -cms 4096 -np 1 -fa on \
  --cache-reuse 0 --cache-ram 512 \
  --jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 --spec-draft-ngl 99 \
  --spec-max-prompt 0
```

**Cache 34, and the wired limit moves with it.** Wired memory cannot be swapped, so
`iogpu.wired_limit_mb` is the safety cap on the whole machine, not a tuning knob — past it Metal
raises a GPU OOM the server reports and survives, instead of the laptop going unresponsive. The
script derives it: cache ≤ 32 → 50176 MiB, cache ≥ 33 → 55296 MiB. **Setting one without the other
is how you get a GPU OOM at load.**

| cache | model wires | verdict |
|---:|---:|---|
| 32 | 46.96 GiB, peak 47.13 under prefill | conservative; zero new swapouts across the test |
| 34 | 51.03 GiB total wired (2026-09-04) | current default; holds up in live use |
| 36 | 51.40 GiB, free memory 0.06 GiB | fastest (~14% prefill) and **the state the machine hung in** |

Watch `sysctl -n vm.swapusage`, not free RAM, which macOS keeps near zero by design. Note that free
memory is ~0.06 GiB at cache 34 as well; what separates it from 36 is that the machine stays
responsive, and that is one steady-state sample, not a peak-under-prefill measurement.

---

## The checkpoint

Two custom splices exist. **`Q4_0-Q8out-v3` is the default**; `UD-iQ4_K_XXS` is the smaller one and
is what the kernel survey below was built to justify.

### `Q4_0-Q8out-v3` — the default, chosen for quality

bartowski's `Q4_0` with this project's `Q8_0` `output.weight` spliced in, then **unsloth's
`UD-IQ4_XS` spliced into five resident tensor groups**: `attn`, `hc`, `token_embd`, `ssm_out`,
`shexp`. Built 2026-09-02 with `gguf_splice_groups.py --groups attn,hc,token_embd,ssm_out,shexp`.

**It was chosen over `UD-iQ4_K_XXS` for quality, not speed.** Its gate/up experts are `Q4_0` at
4.5 bpw against `IQ3_XXS` at 3.06 bpw, and that is 55% of expert weight. On speed the two are level:
26.5 t/s against 26.0, each at its own best draft depth (measured 2026-08-31).

What the five-group splice buys, against the unspliced `Q4_0-Q8out`, paired 40-chunk perplexity plus
an interleaved speed A/B:

| | unspliced | v3 | change |
|---|---:|---:|---|
| perplexity | 5.2777 | 4.3148 | **−17.79%**, t=7.80, better on 40/40 chunks |
| draft acceptance | 0.751 | 0.817 | +0.066 |
| decode | — | — | −2.7% |
| prefill | — | — | unchanged |
| on disk | — | — | +1.69 GiB |

Hyper-connections alone are −13.14% perplexity for +0.30 GiB; attention adds −8.80% for +0.92 GiB.

**Do not trim the group list on perplexity alone.** `attn,hc,token_embd` measures the same −17.5%,
but draft acceptance falls to 0.710 and decode drops 14%: `ssm_out` and `shexp` are worthless for
perplexity and worth about 11 points of decode.

### `UD-iQ4_K_XXS` — the smaller alternative

`UD-iQ4_K_XXS` is a custom splice, built here rather than downloaded: unsloth's `UD-Q3_K_XL` with 43
of its 48 `ffn_down_exps` replaced byte-for-byte with **MXFP4** from AtomicChat's
`AD-3.84bpw-IQ4_XS-M64`, plus `output.weight` at Q8_0. The five down-projections unsloth deliberately
keeps at Q8_0 (layers 2, 4, 30, 46, 47) are left alone, and the `IQ4_NL` PLE table is kept — the
AtomicChat file ships it at `Q5_1`, 8.9 GiB larger, and that table is streamed per row.

| pattern | n | UD-iQ4_K_XXS (ours) | GiB | UD-Q3_K_XL (base) | GiB |
|---|---:|---|---:|---|---:|
| `per_layer_token_embd` | 1 | IQ4_NL | 26.822 | IQ4_NL | 26.822 |
| `blk.N.ffn_gate_exps` | 48 | IQ3_XXS+IQ4_XS | 14.471 | IQ3_XXS+IQ4_XS | 14.471 |
| `blk.N.ffn_up_exps` | 48 | IQ3_XXS+IQ4_XS | 14.471 | IQ3_XXS+IQ4_XS | 14.471 |
| `blk.N.ffn_down_exps` | 48 | MXFP4+Q8_0 | 21.997 | IQ4_NL+Q8_0 | 23.047 |
| `output.weight` | 1 | Q8_0 | 0.629 | Q6_K | 0.486 |
| `token_embd` | 1 | Q8_0 | 0.629 | Q8_0 | 0.629 |
| `shared experts (*_shexp)` | 192 | Q8_0+F32 | 0.234 | Q8_0+F32 | 0.234 |
| `indexer.*` | 48 | F32+BF16 | 0.037 | F32+BF16 | 0.037 |
| `ple_*` | 6 | F32+Q8_0 | 0.033 | F32+Q8_0 | 0.033 |
| `ssm_* / conv` | 252 | F32+Q8_0 | 0.599 | F32+Q8_0 | 0.599 |
| `hc_* (hyper-conn)` | 387 | Q8_0+F32 | 0.647 | Q8_0+F32 | 0.647 |
| `attn (q/k/v/o/kv)` | 120 | Q8_0 | 2.086 | Q8_0 | 2.086 |
| `routing / norms` | 72 | F32 | 0.234 | F32 | 0.234 |
| **total** | **1224** | | **82.890** | | **83.796** |

**Only two rows differ**, and the result is **0.9 GiB smaller** than the base it came from despite
the larger output tensor. 1180 of 1224 tensors are untouched.

A trap worth recording: `--splice-type IQ4_NL` looks like the obvious selector for the
down-projections, and it also matches `per_layer_token_embd` — the PLE table is IQ4_NL too. With a
full donor, that selector silently swaps the PLE table as well.

### Why MXFP4 — the kernel survey

Every quant format measured at **this model's own expert-GEMM shape**, ranked by time per *effective*
bit-per-weight. Effective BPW includes the block scales, so it is what actually crosses the bus.
Generated by `test-backend-ops perf -o MUL_MAT`; the cases live in the perf suite.

#### Prompt processing — 512 rows per expert (`m=640, k=2560, n=512`)

| Rank | Type | Tensor BPW | Effective BPW | Time | Time / effective BPW |
|---:|---|---:|---:|---:|---:|
| 1 | F16 | 16 | 16 | 0.221 ms | 0.0138 ms |
| 2 | BF16 | 16 | 16 | 0.260 ms | 0.0163 ms |
| 3 | Q8_0 | 8 | 8.5 | 0.241 ms | 0.0283 ms |
| 4 | Q6_K | 6 | 6.5625 | 0.285 ms | 0.0434 ms |
| 5 | Q5_1 | 5 | 6 | 0.281 ms | 0.0468 ms |
| 6 | Q4_1 | 4 | 5 | 0.238 ms | 0.0476 ms |
| 7 | Q4_0 | 4 | 4.5 | 0.238 ms | 0.0528 ms |
| 8 | Q5_K | 5 | 5.5 | 0.307 ms | 0.0558 ms |
| 9 | IQ4_NL | 4 | 4.5 | 0.254 ms | 0.0565 ms |
| 10 | MXFP4 | 4 | 4.25 | 0.244 ms | 0.0574 ms |
| 11 | Q4_K | 4 | 4.5 | 0.279 ms | 0.0620 ms |
| 12 | IQ4_XS | 4 | 4.25 | 0.270 ms | 0.0635 ms |
| 13 | Q3_K | 3 | 3.4375 | 0.293 ms | 0.0852 ms |
| 14 | IQ3_S | 3 | 3.4375 | 0.296 ms | 0.0862 ms |
| 15 | IQ3_XXS | 3 | 3.0625 | 0.287 ms | 0.0936 ms |

#### Token generation — one row per expert (`m=640, k=2560, n=1`)

| Rank | Type | Tensor BPW | Effective BPW | Time | Time / effective BPW |
|---:|---|---:|---:|---:|---:|
| 1 | F16 | 16 | 16 | 9.34 us | 0.58 us |
| 2 | BF16 | 16 | 16 | 9.64 us | 0.60 us |
| 3 | Q8_0 | 8 | 8.5 | 7.44 us | 0.88 us |
| 4 | Q6_K | 6 | 6.5625 | 8.17 us | 1.24 us |
| 5 | Q4_1 | 4 | 5 | 6.86 us | 1.37 us |
| 6 | Q4_0 | 4 | 4.5 | 7.06 us | 1.57 us |
| 7 | Q5_1 | 5 | 6 | 9.48 us | 1.58 us |
| 8 | Q4_K | 4 | 4.5 | 7.82 us | 1.74 us |
| 9 | Q5_K | 5 | 5.5 | 9.59 us | 1.74 us |
| 10 | MXFP4 | 4 | 4.25 | 8.09 us | 1.90 us |
| 11 | IQ4_XS | 4 | 4.25 | 8.14 us | 1.92 us |
| 12 | IQ4_NL | 4 | 4.5 | 8.70 us | 1.93 us |
| 13 | Q3_K | 3 | 3.4375 | 10.41 us | 3.03 us |
| 14 | IQ3_S | 3 | 3.4375 | 13.22 us | 3.85 us |
| 15 | IQ3_XXS | 3 | 3.0625 | 22.24 us | 7.26 us |

**The ranking does not follow bit width.** `IQ3_XXS` is the worst format per effective bit in both
regimes despite reading the fewest bytes — its kernel is occupancy-bound on this GPU, and at `n=1`
it is slower in absolute terms than `F16`, which reads five times as many bytes. That is the entire
argument for the splice: `MXFP4` and `IQ4_XS` cost the same 4.25 effective bits, but MXFP4 has a
hand-optimised Metal GEMV here, and both are far better placed than the `IQ3_XXS` they sit beside.

The `ffn_gate`/`ffn_up` pair remains `IQ3_XXS` because it is the bulk of the file and the size
budget has to come from somewhere; `ffn_down` is where the format change buys the most per GiB spent.

---

## The serving configuration

Settings that are not defaults, each with the measurement that chose it. These lived only in
`~/models/bin/qwen-q40-server.sh` until 2026-09-04; that script is still the source of truth and
carries the longer version of each note.

| Setting | Default | Here | Why |
|---|---|---|---|
| `LLAMA_MOE_STREAM_LOOKAHEAD` | `n_expert_used` = **10** | **1** | The one-layer-ahead prefetch defaults to the routing width. At 10 it spends drive bandwidth on experts the layer will not use. The in-code table (measured on DSV4-Flash) shows over-wide lookahead going *worse than off* — top-16 dropped 7.3 → 5.0 t/s because ~1000 speculative loads per 2 s starve the demand loads the GPU is blocked on |
| `LLAMA_MOE_STREAM_WAVE_CAP` | 139 (4 waves) | **200** (3 waves) | pp 268.30 → 278.39, tg 22.43 → 23.51 at 44.8k. An interior optimum: at 279 (2 waves) the lost preload overlap costs more than the saved masked passes. Lossless — only GEMM scheduling changes. See [WAVE_CAP forces off a safety net](#wave_cap-forces-off-a-safety-net-2026-09-04) |
| `LLAMA_MOE_STREAM_PARTITION` | off | **on** | pp 278.27 → 297.87, tg 23.53 → 24.06. Paired 40-chunk perplexity **bit-identical** with it off, all 40 chunks matching (4.3387 both). It changes how pairs are assigned to waves, not what is computed |
| `--cache-ram` | 8192 MiB | **512** | The server's store of saved prompt/KV states in plain host RAM — the same pool macOS competes for. Over a 3h13m session, 77 of 82 slot selections hit the live slot by prefix similarity and only 5 by LRU; the prompt cache appears 6 times and **all 6 are evictions**, throwing away 4.87 and 4.67 GiB. It was holding multiple GiB to serve nothing |
| `--cache-reuse` | 256 | **0** | The caching win comes entirely from exact-prefix reuse, which `cache_prompt` already gives you: turn 1 prefilled 1253 tokens in 7.1 s, turn 2 prefilled 27 in 0.77 s — identical at 256 and at 0. `--cache-reuse N` shifts KV chunks past a divergence, which is **approximate**: output can differ where two tokens were nearly tied. Not a risk worth taking for an unmeasured gain |
| `--moe-stream-io-threads` | auto | **8** | 8 / 12 / 16 are indistinguishable |
| `-ub` | 512 | **4096** | The single largest prefill parameter. 8192 GPU-OOMs at cache 36; 2048 costs 11.6% prefill |
| Google Drive | running | **quit for the run** | The repo lives in a Drive sync target, so Drive re-uploads `build/` while the model streams experts off the same SSD. Measured at ~1.9 GB resident and 61% CPU. The script quits it and restores on exit |

---

## What was added, and why

### The indexer — the prefill story

| Change | Rationale |
|---|---|
| **Block-level top-k** | the budget is whole blocks anyway, so selecting over blocks skips materialising the `[n_kv, n_tokens]` cell table entirely — this flattens the prefill *curve*, not just its value |
| **Selections mapped through `blk_cells`** | a block id is not a cell base: `set_input_qsa` hands out compacted ids and records members in a table, and the two coincide only when every live cell sits at its own position |
| **Permute-free scoring** | upstream now sums the indexer heads by slices, which removes the same large `permute+cont` this fork removed by putting `q` on the left |
| **F16 indexer keys** | quantising raw index keys changes which blocks survive a discrete top-k, and the 128-wide indexer head cannot tile a 256-block type |
| **Per-query sparse FA** | gathers each query's selected K/V rows directly instead of walking a dense masked row space |

Block-level selection is also a marginally *better* cut: all `r` cells of a block share a score, so a
cell-level top-k split the boundary block arbitrarily. Causality is unaffected — the causal mask is
still applied after selection.

### The graph — the decode story

| Change | Rationale |
|---|---|
| **`RMS_NORM→MUL` adjacency** | Metal's fusion check is a *tensor-identity* test; a reshape between the two silently broke it and spilled the norm to memory |
| **Hyper-connection collapse** | `ggml_cont` before the stream adds bought nothing — ADD needs contiguous *rows*, which a strided view already has — and blocked chain fusion |
| **`ggml_top_k` for MoE routing** | `argsort_top_k` fully sorted all 512 expert scores to view the first 10 |
| **Graph reuse / shared QSA input** | the whole graph was rebuilt and re-recorded every token |

These are GPU work removed, not I/O: stream statistics are unchanged across the change.

### Native MTP head

The model ships a NextN block that upstream's loader reads as part of the trunk and the graph cannot
run. Supporting it needs its own tensors (`fc_embd`, `fc_hidden`, `enorm`, `hnorm`, and a
hyper-connection norm), its own graph, and a hidden-state export from the trunk.

The head is fed the **wide** hyper-connection stream (`hc*n_embd`) from before the final mixer — not
the collapsed output — because its `hnorm` and `fc_hidden` are `[hc*n_embd]`. Three separate things
must hold or acceptance silently collapses to ~1% rather than failing loudly:

1. the trunk must set `res->t_h_nextn` at all;
2. it must be the wide pre-mixer stream, not the mean-pooled one;
3. it needs an explicit `ggml_build_forward_expand` — it is a bare reshape view with no consumer, so
   otherwise the scheduler never assigns it a backend and it is never computed.

**The tell for all three:** acceptance identical to the digit across two different drafter
quantisations. Drafts that do not change when the drafter's weights change are not coming from the
drafter.

An `mtp-` sidecar carries only the NextN block plus embeddings, so the trunk tensors and the PLE
table must be optional when one is loaded.

### PLE table streaming

26.82 GiB of 90-byte rows. Upstream keeps it off the resident set with `TENSOR_READ_LAZY`, which is
**mmap-only by construction** — rows arrive as page faults against a mapping the size of the whole
table. When expert streaming is on, the table is handed to the servicer instead: `set_input` reads
just this ubatch's rows with buffered parallel `pread`s into a compact tensor, and `get_rows`
dequantises that exactly as it would the full table. Repeated rows are read once; reads are sorted by
file offset. Upstream's lazy path remains the fallback.

Deliberately **buffered, not `O_DIRECT`** — at 90 bytes a row the page cache genuinely helps, which
is the opposite of the finding for multi-MiB expert slabs.

### Correctness fixes

| Fix | Consequence if missing |
|---|---|
| **Recurrent-state rollback** | conv history and delta-net groups left zeroed/stale, corrupting generation after any rollback |
| **Indexer cache after sequence copies** | cached indexer keys are raw, so a pending update must not rope-shift them |
| **Spare-block members in `set_input_qsa`** | the unpooled tail — the newest tokens — drops out of the selection entirely |
| **`--spec-max-prompt`** | a draft head is an extra layer over the whole prompt; past a length it cannot pay back |
| **No V cache for the indexer** | the indexer scores keys only; the cache was allocating a V side nothing reads. Presenting its private hparams as MLA-shaped takes `llama_kv_cache`'s existing `has_v = !is_mla` path. Frees 12 × `n_ctx` × 128 B — 288 MiB at ctx 98304, 384 MiB at 131072 (upstream `#28330`) |

---

## Negative results

| Idea | Result |
|---|---|
| **union-8 for the indexer path** | negative below ~96k context and only about +1% above it; the per-query sparse path is the retained design |
| **MTP + n-gram stacked** | net negative — they compete for the same accepted tokens |
| **Deeper MTP drafts** | n-max 2 beats 4. Deeper drafts are accepted less often and the rejected-token verification outweighs the longer runs; depth 1 over-corrects |
| **MTP draft depth as a memory lever** | it is not one — depth 4 → 1 frees only 338 MiB |
| **q8_0 attention KV** | correct, and halves the attention cache, but has not beaten F16 in the Qwen-shaped sparse kernel |
| **`hc_up` matmul at K=320** | a wash, not a win |
| **Upstream `5ea1b124` fa-vec tunings** | rejected — no F16 entry matches head dim 256/256 |
| **Swapping upstream's head-slice sum for our permute-free scoring** | upstream `#28023` already removes the same copy a different way; replacing it is a swap, not a gain |
| **MTP at long context** | **inverts.** At 32k the head costs 3.7 s of prefill and repays after 119 generated tokens; at 128k it costs 57.7 s and repays after 2369 |
| **union-8 for MTP** | does not apply |
| **Deeper drafts** | the expert GEMV n-curve is linear with a cliff at 32; depth does not amortise the weight read |
| **Upstream's `n_kv_max` sparse-FA hint for QSA** (`#27970`/`#28098`) | neutral to within 1% at 32k and 128k. QSA's selection is scattered — top-k picks ~512 four-cell blocks across the whole cache, so nearly every block the kernel could skip still holds a selected cell. **On by default; `LLAMA_QWEN4EXP_SPARSE_FA=0` disables.** See the caveat below |
| **Upstream `#28213` gather-based QSA decode** | 5% slower at 32k, 6% at 128k. It sets `blk_bias = false` to get the per-cell bias, which turns **block top-k off** — that optimisation is worth more than the gather, and the per-query sparse FA above already covers the idea. Not adopted |

**Standing rule: no output-altering optimizations.** Quality is not tradeable for single-digit
percentage gains.

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

**"MTP n-max 4."** Then "n-max 2 beats 4". Both superseded: a 22-arm sweep on a real ~24k prompt
settled on **n-max 3, p-min 0.3**, bracketed again at 44.8k, and re-confirmed 2026-09-04 with the
n-gram chain in front. The three figures come from different machines and checkpoints.

**"Chaining n-gram in front of MTP roughly doubles decode."** Measured on the raw `/completion`
endpoint, which bypasses the chat template. Under `--jinja` — what the server actually runs — it is
+9.9% on code and **−6.6% on reasoning**. See [Speculation stack](#speculation-stack-2026-09-04).

**Upstream's `edb6dec1c` "enable recurrent state rollback" looks wrong**: it adds `QWEN4EXP` to the
whitelist, but their tree has neither the ring-bank conv writes nor the delta-net `n_written < K`
clamp, both prerequisites.

**"Sparse FA for QSA is neutral, and off by default behind `LLAMA_QWEN4EXP_SPARSE_FA`."** Both
halves were wrong, and this entry is the reason the gate now exists.

The flag was never real — no such env var existed anywhere in the tree, on this branch or the one
before it. Sparse FA has been **on unconditionally** since `39b1d9ae8`, which replaced the disabled
call with `top_k->ne[0]`.

The "neutral to within 1%" number is worse than stale. It was measured against **this fork's own
sparse kernels**, and the mainline rebase dropped those as duplicates of `#27970`/`#28098`. The
hint now reaches mainline's kernels, which are different code. Nobody has measured it in that form.

The gate is now implemented for real (`qwen4exp_sparse_fa()`), still **on by default** so the
behaviour does not change under anyone silently, with `LLAMA_QWEN4EXP_SPARSE_FA=0` to A/B it
without a rebuild. **Re-measure before quoting the neutral figure again.**

The general lesson is the one this log keeps relearning: a measurement is only as portable as the
code underneath it, and a rebase can move that code without touching the line that records the
result.

---

## Speculation stack (2026-09-04)

Does chaining `ngram-mod` in front of the MTP head pay? It depends entirely on the endpoint, which
is why the first answer was wrong. Both tables: v3 checkpoint, cache 32, ctx 32768, `np 1`,
n-max 3 / p-min 0.3, one warmup discarded, arm order reversed between passes, swap flat throughout.

**Raw `/completion`, no chat template:**

| traffic | chain off (A/B) | chain on (A/B) | delta | acceptance |
|---|---|---|---|---|
| code | 29.77 / 28.37 | 61.95 / 62.25 | +113% | 1.000 both |
| prose | 19.91 / 20.06 | 21.21 / 21.06 | +5.8% | 0.572 both |
| reason | 25.07 / 24.58 | 26.34 / 25.87 | +5.1% | 0.770 both |

**`/v1/chat/completions` with `--jinja --reasoning-format deepseek`, i.e. the real serving flags:**

| traffic | chain off (A/B) | chain on (A/B) | delta | acceptance off → on |
|---|---|---|---|---|
| code | 28.71 / 28.33 | 31.39 / 31.31 | +9.9% | 0.685 → 0.626 |
| prose | 20.34 / 19.74 | 20.28 / 20.27 | +1.2% | 0.512 → 0.468 |
| reason | 27.75 / 26.48 | 25.79 / 24.85 | **−6.6%** | 0.780 → 0.613 |

Thinking is on by default under that template, so 800–1200 characters of reasoning precede every
answer, and reasoning text drafts like prose. The +113% depended on acceptance being 1.000 — the
drafter was copying the prompt back. With thinking on it cannot, and reasoning regresses. This is
the mechanism behind the older "MTP + n-gram stacked is net negative" entry, and it says *when*:
the chain only pays where the output repeats the input.

`NGRAM=1` in the server script turns it on; the default is off.

**n-max / p-min re-confirmed with the chain on.** Code saturates at ~62 t/s in every config, so
prose and reason decide it. Prose prefers n2, reasoning prefers n4, and they pull about equally
hard — mean of the two: n2 23.48, **n3 23.62**, n4 23.41, n3/p0 23.58, all within 1%. No change.

---

## Real traffic (2026-09-04)

Two `pi` sessions on the `nitin/mainline` build (33 fork commits replayed onto upstream
`8f83678fd`), served through `--jinja --reasoning-format deepseek`, cache 32, ctx 98304,
n-max 3 / p-min 0.3, n-gram chain off. 9 requests, 117,249 prompt tokens, 24,610 generated.

| prompt | pp t/s | generated | tg t/s | acceptance | mean draft |
|---:|---:|---:|---:|---:|---:|
| 54,443 | 251.1 | 226 | 18.86 | 0.760 | 3.25 |
| 54,722 | 254.6 | 177 | 19.71 | 0.738 | 3.11 |
| 477 | 145.6 | 739 | 18.62 | 0.660 | 2.93 |
| 372 | 85.2 | 722 | 22.57 | 0.667 | 2.86 |
| 7,123 | 235.2 | 11,638 | 18.94 | 0.483 | 2.37 |
| 18 | 28.0 | 10,002 | 21.99 | 0.832 | 3.49 |
| 32 | 28.4 | 365 | 17.12 | 0.808 | 3.39 |
| 34 | 32.0 | 145 | 16.01 | 0.711 | 3.04 |
| 28 | 32.9 | 596 | 15.79 | 0.517 | 2.52 |

**Weighted by tokens produced: 20.16 t/s decode, 0.647 acceptance.** Weight by tokens, not by
request - the two 10k+ generations are the session, and a simple mean (18.85 / 0.686)
over-weights sub-100-token replies whose rate is dominated by fixed overhead. Sub-40-token
prompts show pp 28-33 t/s for the same reason; those are not prefill measurements.

Health: no errors, no aborts, swap flat at ~970 MB throughout. The mainline rebase served two
full real sessions cleanly - better evidence for the switch than the benchmark A/B, which was
contaminated by swap growth.

MTP is worth about +40% here. Un-drafted decode brackets this depth at ~15.5 t/s at 32k and
~9.2 at 131k, so 54k without a draft head would be roughly 13-14 against the 18.9-19.7 measured.

Acceptance ranged 0.483 to 0.832 - the code-vs-reasoning split, in the wild. The two big
generations sat at opposite ends (0.483 over 11,638 tokens, 0.832 over 10,002) yet differed by
only 14% in throughput, which is why adaptive draft depth looked attractive and still lost.

Note the session ran at 0.647, well above the 0.46-0.51 this log records for real traffic.
One session; direction only.

## Open bugs

**`nan` logits on the BLAS backend, from block top-k.** `test-llama-archs -a qwen4exp`, Accelerate
row: `NMSE nan`, roundtrip FAIL. Deterministic across seeds 1–6. Metal and pure CPU are clean, and
the full 606-row suite has no other failure, so the Mac serving path is unaffected.

`LLAMA_QWEN4EXP_BLOCK_TOPK=0` clears it. It survives `LLAMA_QSA_GATHER=0` and
`LLAMA_QWEN4EXP_INDEXER_F16=0`, so it is not the `blk_cells` mapping bug and not the key type.

Ruled out: BLAS claiming an op it does not implement (its `supports_op` is correct — only
`NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE` and a gated `MUL_MAT`); `ggml_top_k` returning a view (it
returns a fresh contiguous I32 tensor); an all-masked attention row (the CPU kernel skips masked
cells and guards `S == 0`, yielding 0 rather than `nan`).

What is odd: block top-k changes no matmul's shape, so BLAS takes the same matmuls either way, yet
its presence is required to trigger the fault. That points at graph splitting or allocation — BLAS
and CPU share a buffer type, so sched can alias buffers across splits. Needs a debug build with a
graph-eval callback; `llama-eval-callback` cannot help because the synthesised model has no
tokenizer, and `GGML_SCHED_DEBUG=2` is swallowed by the test's logging.

---

## WAVE_CAP forces off a safety net (2026-09-04)

The server script exports `LLAMA_MOE_STREAM_WAVE_CAP=200`. Every start now logs:

    LLAMA_MOE_STREAM_WAVE_CAP=200 forces N waves, below the 600 pairs/wave floor
    - the static chunk bound may abort; unset it to let the planner pick

This is not cosmetic. Reading `llama-graph.cpp` around the wave planner:

- **200 does not mean 200.** It is clamped to `[n_expert_used, n_slots - n_expert_used]`,
  i.e. 86 of 96 slots here. So the setting means "use the widest wave the cache allows".
- **Forcing it disables the widening.** The planner's corrective branch is guarded by
  `!cap_forced`, so with the env var set, the code that keeps mean pairs-per-wave above the
  floor never runs.
- **The floor exists because of a measured abort.** The comment records it: mean 614 pairs
  per wave held at +20% imbalance, mean 308 blew past +43%, and "at cap 30 the tail came
  +50.2% over, aborting between two cap values that both worked". The hazard is the small
  TAIL ubatch of a prefill, whose hot-expert pair counts do not shrink with the ubatch.

So the risk is a hard abort on an unlucky prefill tail, not a slowdown. It did not fire across
117k prompt tokens of real traffic today, but it is disabled protection against a failure this
log has already seen once.

NOT YET MEASURED: whether 200 actually beats unset. It was adopted before pair partitioning
became the default, and partitioning is what sizes the pair lists the floor is about. Since it
clamps to the maximum anyway, "unset" may cost nothing. Measure before deciding.

## Sweeps

| Sweep | Outcome |
|---|---|
| **ubatch / batch** | keep **4096** |
| **Cache size** | 36 is ~14% faster on prefill, ~2% on decode — but 36 + MTP pages. Use **32 with MTP** |
| **MTP quantization** | `Q4_0` (2.20 GiB); Q8_0 (3.85 GiB) gives similar acceptance (0.55 vs 0.51 mean) for 1.75x the footprint |
| **MTP n-max / p-min** | **3 / 0.3**, bracketed at 24k and 44.8k, re-confirmed with the n-gram chain in front |
| **I/O threads with MTP** | keep **8** |
| **Speculation type × cache** | MTP alone beats n-gram alone; the chain only pays without the chat template |

---

## Measurement discipline

- **Measure under the flags the thing actually runs with.** The n-gram chain was measured on
  `/completion` and recommended for a server that passes `--jinja`. The raw endpoint bypasses the
  template, so those numbers described a configuration that never runs. Cost: one wrong default.
- **Run A/B arms as ABA.** This box speeds up as caches warm — four identical runs at d32768 gave
  13.55 → 14.36 → 15.37 → 15.35 t/s. A single on-then-off pair therefore reads as a loss for
  whichever ran first. Run the feature arm on both sides of the control and check the two agree.
- **One run per arm is not a measurement.** Spread at d32768 is ±1.4 t/s on a mean of ~15, about 9%
  — wider than most effects worth chasing.
- **The first request understates decode by ~25%.** Cold 12.20 → warm 16.49 / 16.26 t/s. Multi-arm
  sweeps issuing 3+ requests per arm are unaffected; one-shot verification runs are not.
- **That ~25% is a COLD-START effect, not warm-vs-post-prefill.** Warm decode runs only 2–16% above
  post-prefill at the same depth. Post-prefill is the honest number for agentic use, where every
  turn re-prefills.
- **"It loaded" is not "it fits".** Cache 36 + MTP loads, runs, and pages while doing it. Sample
  `sysctl -n vm.swapusage` around every arm; a rising figure voids the measurement. Watch swap, not
  free RAM, which macOS keeps near zero by design.
- **Warm decode measured once per context carries ~15% variance** — in one sweep the 8k warm figure
  came out *below* its own post-prefill figure, which is physically impossible.
- **A replay trap can fake +33%.** n-gram speculation replaying its own prior output measures
  nothing. Acceptance of exactly 1.000 is the tell that the drafter is copying.
- **Percentages on a curve are meaningless without the length.**
- **Check the harness before the model.** A readiness loop waiting for the wrong log string looks
  exactly like a hung server. The line is `llama_server: listening on http://`. Tags used as
  filenames must not contain `/`, or every server "dies" instantly with an unwritable log.
- **Audit 3-way patch applies for silently reverted hunks** — conflict markers show only part of
  the damage; patch *context* can overwrite untouched work.
- **`git stash pop` leaves merged files staged.** A later bare `git commit` absorbs them. Stage by
  path.

---

## Backlog, in priority order

### 1. KV cache quantization — frees memory without giving up context

`-ctk q8_0 -ctv q8_0`. KV is `f16` by default; halving it frees memory for the expert cache **at the
same context length**. At ~85 KB per token, 98,304 context is roughly 8 GB, so this could fund
`--moe-stream-cache 40` with ctx intact.

**Not lossless** — it changes stored attention keys and values. Needs paired perplexity first, then
a speed A/B at cache 40 / ctx 98304. Check in the same run: this model is 36 SSM layers and only 12
attention layers, so KV may be a smaller share of memory than 85 KB/token implies, in which case the
freed memory never materialises. Note the indexer V cache is already gone (above), so part of this
has been banked. ~90 min.

### 2. Adaptive MTP draft depth (upstream `#27210`) — **ported, unmeasured**

Prose wants n2, reasoning wants n4, and a fixed n-max cannot have both — measured twice now. This
climbs and drops the depth per request, so one setting can serve both. The most principled fix for
the split above.

**The port is done and on this branch**, not pending: `--spec-type draft-mtp-adaptive`. It reached
`nitin/mainline` late, having been written on `claude/adaptive-mtp` before the rebase and left
behind by it — an earlier version of this entry said it "will not apply cleanly", written when the
port did not exist yet. Both unit tests pass (`test-speculative-adaptive`, `test-arg-parser`).

Two fork-specific conflicts were resolved on the way in. `common/speculative.cpp` — upstream widens
its `spec_mtp` test to include the adaptive type, while this fork carries a `spec_block` clause for
DFLASH/DSPARK that upstream does not have; both kept. And this fork has a **fourth** `DRAFT_MTP`
test upstream does not, in the `-fit` branch of `common_init_from_params`, which decides whether the
draft context is built as `LLAMA_CONTEXT_TYPE_MTP`. With only the non-adaptive type tested,
`--spec-type draft-mtp-adaptive` left `ctx_type` unset and the MTP sidecar — which carries no trunk
tensors — was loaded as a full model and segfaulted.

**What is left is the measurement, not the code.** It has never been run against real traffic here.
The bar it has to clear is the fixed `n-max 3 / p-min 0.3`, whose own sweep put prose and reasoning
within 1% of each other, so the win it is chasing is small and the acceptance range in live traffic
(0.483–0.832) is wider than anything the sweep saw.

### 3. The remaining streaming toggles

`WAVE_CAP` and `PARTITION` are done and adopted. `LRU` is skipped — its own comment records +11%
misses. Left, all lossless: **`HOT_DECAY`** (halves route-hotness every 64 remaps; the constant "has
never been measured"), **`PAIR_SLACK`** (50% slack on per-wave pair capacity, more interesting now
that partitioning sizes the pair lists), **`SPEC_MAX`** (prefetch queue depth, default 64).

### 4. A smaller draft head — lossless, modest

Only *below* `Q4_K_M` can win. A `Q3_K_M` head (~2.26 GiB) must hold acceptance above **0.813**;
`shared-Q4_K_M` (1.78 GiB) needs 0.783 but omits `token_embd`/`output` and may not load. A few
percent at best. ~40 min.

### 5. The MTP head runs dense despite shipping sparse weights

A code change rather than a measurement, and the head is ~18% of a decode step. Unscoped.

### 6. F16 `256/256` flash-attention vector tunings

Upstream never tuned this shape. Code plus tuning generation. Unscoped.

---

## Housekeeping

- **The long-form measurement detail lives in `Qwen3.8-Flash-Next.WIP-yours.md`** (1052 lines,
  uncommitted). This log is the condensed spine; that file still holds the only copy of the
  per-context performance sweeps, the M5 Pro run book, the PLE buffered-vs-uncached study, the
  `GGML_METAL_KPROF` decode breakdown, and the Unsloth variant audit. Fold in or delete
  deliberately — do not lose it by accident.
- **Re-baseline the pre-M5-Pro half of this log, or mark it historical.** Figures older than
  2026-09-03 are M1 Max, several on a checkpoint that no longer exists.
- **Decide whether to keep `LLAMA_MOE_STREAM_PLE_DIRECT`.** Correct, off by default, buys nothing
  measurable. The alternative is deleting it.
- `~/models/qwen38-flash-next-v2` (97 GiB) is superseded by v3 and can be deleted.
- The PLE buffer-overflow fix in `src/llama-moe-stream.cpp` is still uncommitted.

## Closed

- **`--spec-max-prompt`**: set to 0. The limit is cache-blind and disabled MTP mid-conversation.
- **Draft depth and p-min**: `n-max 3`, `p-min 0.3`, bracketed at 24k and 44.8k, re-confirmed 09-04.
- **io-threads and `-ub`**: 8 and 4096 already optimal; `-ub 8192` GPU-OOMs at cache 36.
- **A bigger draft head**: measured −5.2%. Q5/Q6/BF16/F32 all ruled out by the cost model.
- **Per-block bitmap for union-8**: lifted the ceiling 4x, no throughput gain at reachable contexts.
- **MXFP4 vs block-level top-k attribution**: blocked, the `UD-iQ4_K_XXS` checkpoint was deleted.
- **n-gram chained in front of MTP**: measured on both endpoints; opt-in, off by default.
- **Indexer V cache**: removed, 288 MiB at ctx 98304.
