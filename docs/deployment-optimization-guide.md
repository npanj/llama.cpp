# llama.cpp inference: a first-principles optimization guide

This document derives llama.cpp deployment tuning from two physical quantities -
bytes moved and FLOPs executed - and then reads every relevant flag, default and
env var directly out of the source tree. It is the quantitative companion to
[`architecture-and-tuning.md`](architecture-and-tuning.md) (structure and flag
reference) and deepens the models introduced in the
[inference tutorial](tutorials/llama-cpp-inference/tutorial.html). Where this guide
states a default or a name, it was grepped from
[`common/arg.cpp`](../common/arg.cpp), [`common/common.h`](../common/common.h),
[`ggml/src`](../ggml/src) or [`tools/`](../tools); where it states a number, the
number is recomputed here or verified by
[`tutorials/llama-cpp-inference/verify-numbers.json`](tutorials/llama-cpp-inference/verify-numbers.json)
(145 checks, all passing).

Conventions: `GiB = 1024^3`. Model shape for hand-worked examples: 8B dense,
`n_layer=32, n_embd=4096, n_head=32, n_head_kv=8, head_dim=128, n_ff=14336,
n_vocab=128000`, quantization `q4_k` (4.5 bits per weight, `GGML_TYPE_Q4_K`).
Hardware figures in tables are *illustrative* bandwidth/compute assumptions,
not device specifications.

---

## 1. Roofline: what actually costs time

Token generation is a loop of two very different phases.

**Prefill** (prompt processing) runs `n_prompt` tokens at once. Each token in a
chunk costs the full forward-pass compute, `2 x P` FLOPs per token (one
multiply-add per weight). Because a byte of `q4_k` weight holds `8/bpw` params
that each do `2*T` MACs for a chunk of `T` tokens, arithmetic intensity grows
linearly with tokens per batch:

```
intensity(prefill) = 16 * T / bpw       [FLOPs per weight byte read]
```

**Decode** (generation) runs 1 token per sequence. Every layer's weights must be
read again for a single token's worth of compute:

```
intensity(decode) = 16 / bpw            [FLOPs per weight byte read]
                  = 3.56 at q4_k (16/4.5)
                  = 1.0   at f16
```

A machine can sustain `BW` bytes/s or `FLOPS` FLOP/s. Their ratio is the
**machine balance**:

```
balance = FLOPS / BW                    [FLOPs that must amortize each byte]
```

The illustrative 100 GiB/s, 10 TFLOP/s device has `balance = 93 FLOP/byte`.
Decode, at 3.56 FLOP/byte, sits ~26x below it - and the gap is wider on every
faster device - so **decode is memory-bandwidth-bound on every device class in the table
below**, and the gap widens as machines get faster:

```
t_decode/token ~ bytes_read_per_token / BW_effective
```

The **crossover** chunk size where prefill stops being bandwidth-bound is:

```
T_crossover = balance / intensity(decode) = balance * bpw / 16
```

For the example device that is ~26 tokens (`93/3.56`; verified as the
`crossover` check in `verify-numbers.json`); on a 279 FLOP/B data-center GPU
(1000 GiB/s, 300 TFLOP/s) ~78; on a 19 FLOP/B CPU, ~5. Implications:

- Prefill on a GPU with the default `n_ubatch=512`
  ([`common/common.h:462`](../common/common.h)) is far above the crossover: it
  is compute-bound, so quantizing weights buys little prefill *throughput*.
  (It still decides whether the model fits at all, and on CUDA the MMQ integer
  path can beat dequant-to-cuBLAS even here - measure, do not assume.)
- CPU-only decode sits near the roofline knee, so CPU prefill is compute-bound
  even at small batches, and CPU tuning is about *bytes*, not threads
  (section 8).

Worked decode budget (8B example, all values in
[`verify-numbers.json`](tutorials/llama-cpp-inference/verify-numbers.json)):

```
p_attn  = 2*n_embd^2 + 2*n_embd*n_head_kv*head_dim =  41,943,040  /layer
          (Wq 4096x4096, Wo 4096x4096, Wk 4096x1024, Wv 4096x1024)
p_ffn   = 3*n_embd*n_ff                            = 176,160,768  /layer
p_layer =                                            218,103,808
p_blocks= n_layer * p_layer                        = 6,979,321,856
p_embd  = n_vocab * n_embd                         =   524,288,000 (each)
p_total = p_blocks + 2*p_embd                      = 8,027,897,856  (~8.03B)

bytes_resident = p_total * 0.5625 = 4,515,692,544 B = 4.21 GiB  (VRAM footprint)
p_read         = p_blocks + p_embd (output head; the
                 token-embedding table is a per-token row
                 lookup, ~2.3 KiB, not a full read)   = 7,503,609,856
bytes_read     = p_read * 0.5625 = 4,220,780,544 B = 3.93 GiB  per token

t_decode @100 GB/s = 39.3 ms  ->  25.4 tok/s
t_decode @ 25 GiB/s = 157.2 ms ->   6.4 tok/s
t_decode @400 GiB/s =   9.8 ms -> 101.8 tok/s
t_decode @500 GiB/s =   7.9 ms -> 127.2 tok/s
```

The decode column of the hardware table below is fully determined by
`BW/bytes_read`:

| device class          | BW (GiB/s) | FLOPS (TFLOP/s) | balance (FLOP/B) | prefill-bound by | decode tok/s (this 8B q4_k) |
|-----------------------|------------|-----------------|------------------|------------------|------------------------------|
| laptop iGPU / DDR5    | 100        | 10              | 93               | compute (ubatch 512) | 25.4                       |
| Apple silicon (SoC)   | 400        | 30              | 70               | compute          | 101.8                        |
| consumer GPU          | 500        | 100             | 186              | compute          | 127.2                        |
| data-center GPU       | 1000       | 300             | 279              | compute          | 254.4                        |
| CPU with BLAS         | 100        | 2               | 19               | compute (early)  | ~25 ceiling                  |

Every optimization in the rest of this document is one of three moves:
read fewer bytes per token, move the reads to a wider pipe, or stop reading
bytes one token at a time (batching, speculation, cache reuse).

---

## 2. Per-token byte budget

For one decode step of one sequence:

```
bytes/token = bytes_weights + bytes_kv + bytes_overhead

bytes_weights = bpw/8 * (sum over layers placed on the reading device of
                         active params/layer)
              + bpw/8 * (embedding or lm_head bytes, unless weights are tied)

bytes_kv      = 2 * n_layers * n_head_kv * head_dim * type_size * ctx
                ^ one K element + one V element per (layer, kv-head, dim, pos)
```

`bytes_overhead` covers logits (`2*n_vocab` at f32 read once per token),
activations, sampler state - real but second-order, and the reason measured
tok/s sits a few percent under the roofline estimate.

The KV formula makes the architectural levers explicit:

- **GQA shrinks KV linearly in `n_head_kv`.** The 8B example: `2*32*8*128*2 =
  131,072 B = 128 KiB/token` at f16. An MHA model with the same hidden size
  (`n_head_kv = n_head = 32`) reads 512 KiB/token - 4x more - so long-context
  decode collapses 4x sooner.
- KV reads grow *linearly* with context while weight reads do not. KV overtakes
  the per-token weight read at `ctx* = bytes_weights / bytes_kv_per_token`:
  ~32,200 tokens here at f16 KV, ~60,600 at `q8_0`. Past that point context
  management matters more than weight quantization.
- Decode **attention compute** also grows only linearly with context - one new
  query against `ctx` cached keys: `4 * n_layer * n_head * head_dim * ctx`
  FLOPs/token, ~17 GFLOP at ctx 32k for the 8B example, ~1.7 ms at 10 TFLOP/s
  (~4% on top of the 39 ms memory term). The *quadratic* `ctx^2` cost belongs
  to **prefill** (all-vs-all attention: `2*n_layer*n_head*head_dim*ctx^2`
  causal, ~281 TFLOP at 32k here - comparable to the 491 TFLOP of weight
  matmuls) and to any code path that materializes the score matrix. Flash
  attention removes the `ctx^2` memory traffic and keeps the compute tiled,
  which is why `-fa on` is the highest-leverage long-context flag
  (section 5).

Model-level byte budget examples (f16 KV, Q4_K weights, 100 GiB/s):

| model (dense, GQA)    | weights/token | KV/token | ctx  | total | tok/s |
|-----------------------|---------------|----------|------|-------|-------|
| 8B   (above)          | 3.93 GiB      | 128 KiB  | 8k   | 4.93 GiB | 20.3 |
| 8B   (above)          | 3.93 GiB      | 128 KiB  | 32k  | 7.93 GiB | 12.6 |
| 8B-MoE (8 experts, top-2) | 1.72 GiB  | 128 KiB  | 0    | 1.72 GiB | 58.3 |

The MoE row is the whole story of modern serving economics, and it is stated
here for the *same* 8B shape with its dense FFN replaced by 8 experts of width
1792 routed top-2 - identical parameter count, identical 4.21 GiB resident
footprint, but only 1.72 GiB read per token, so **2.29x** the decode rate. That
derivation, and the `-ncmoe` versus `-ngl` comparison that follows from it, are
in [`tutorials/llama-cpp-inference/tutorial.html`](tutorials/llama-cpp-inference/tutorial.html)
section 6 and are checked in `verify-numbers.json`. Real MoE releases differ in
shape (Qwen3-30B-A3B, for instance, routes 8 of 128 experts over 48 layers at
`n_embd` 2048, so its active-byte figure must be recomputed from its own
config, not borrowed from this row) but the economics are the ones derived
here. See section 6.

---

## 3. Weight quantization economics

Decode speed scales inversely with bits-per-weight. The block formats are
defined in [`ggml/src/ggml-common.h`](../ggml/src/ggml-common.h); their
cost and benefit:

| type      | bpw     | block bytes / elems | read/token (8B ex.) | decode tok/s @100 GiB/s | relative | note |
|-----------|---------|---------------------|---------------------|-------------------------|----------|------|
| f16       | 16      | -                   | 13.98 GiB           | 7.15                    | 1.00x    | baseline |
| q8_0      | 8.5     | 34 / 32             | 7.43 GiB            | 13.5                    | 1.88x    | near-lossless |
| q6_K      | 6.5625  | 210 / 256           | 5.73 GiB            | 17.4                    | 2.44x    | |
| q5_K      | 5.5     | 176 / 256           | 4.80 GiB            | 20.8                    | 2.91x    | |
| q4_K      | 4.5     | 144 / 256           | 3.93 GiB            | 25.4                    | 3.56x    | sweet spot |
| iq4_XS    | 4.25    | 136 / 256           | 3.71 GiB            | 26.9                    | 3.76x    | |
| q3_K      | 3.4375  | 110 / 256           | 3.00 GiB            | 33.3                    | 4.65x    | quality drops |
| iq3_XXS   | 3.0625  | 98 / 256            | 2.68 GiB            | 37.4                    | 5.22x    | |
| q2_K      | 2.625   | 84 / 256            | 2.29 GiB            | 43.6                    | 6.10x    | usually too lossy |

Every `bpw` is the block size divided by the elements it covers, taken from the
`static_assert`s in `ggml-common.h`; the `relative` column is exactly `16/bpw`,
which is the whole content of `intensity = 16/bpw`.

(Per-block layout, counted from the `static_assert`s in `ggml-common.h`:
`block_q8_0 = 2 B f16 scale + 32 B int8 quants = 34 B / 32 elems = 8.5 bpw`;
`block_q4_0 = 2 B scale + 16 B nibbles = 18 B / 32 = 4.5 bpw`; the k-quants use
256-element superblocks, e.g. `block_q4_K = 4 + 12 + 128 = 144 B / 256 = 4.5 bpw`
and `block_q6_K = 2 + 16 + 192 = 210 B / 256 = 6.5625 bpw`.)

The economics:

- Gain applies to **decode only**. Prefill is compute-bound (section 1) and
  quantized GEMMs add dequant or integer-path overhead; expect no prefill win,
  sometimes a loss. Benchmark both: `llama-bench -p N -n 0` and `-p 1 -n M`
  (section 9).
- Cost is measured, not guessed: `llama-perplexity -f test.txt` on in-domain
  text, plus `--hellaswag`, plus `--kl-divergence-base <f16 gguf>` which reports
  KL against the fp16 reference instead of absolute PPL - the right metric for a
  quantization decision.
- VRAM budget: `bytes_resident = p_total * bpw/8` (4.20 GiB above vs 14.95 GiB
  f16). When full offload does not fit, prefer degrading the *least sensitive*
  bytes first rather than the whole tensor: `-ot <pattern>=<type>` per-tensor
  buffer-type overrides ([`common/arg.cpp:2784`](../common/arg.cpp)),
  `-ncmoe`/`--n-cpu-moe` and `-ncffn`/`--n-cpu-ffn`
  ([`common/arg.cpp:2797,2807`](../common/arg.cpp)) to keep the last N MoE
  expert or dense FFN layers on CPU (they are read sparsely or amortize
  better), `-ctk`/`-ctv` for KV. See
  [`architecture-and-tuning.md`](architecture-and-tuning.md) for the full flag
  reference.
- Hybrid CPU+GPU decode (partial `-ngl`) interleaves a slow-pipe read with the
  fast pipe and pays PCIe for activations; measured throughput tends toward
  the *slowest* segment, not the average. Benchmark `-ngl 0`, `-ngl 99` (full
  offload), and anything between with `llama-bench -ngl 0,12,99`.

---

## 4. KV cache and context economics

Defaults: `cache_type_k = cache_type_v = GGML_TYPE_F16`
([`common/common.h:343-344`](../common/common.h)); KV types accepted by
`-ctk/-ctv` are `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`
([`common/arg.cpp:305-315`](../common/arg.cpp)).

Sizing: `bytes_kv_slot = 2*n_layer*n_head_kv*head_dim*type_size*ctx`, so the
VRAM cost of `n_parallel` server slots scales as `n_parallel x ctx x
bytes/token` (see `--parallel` and per-slot KV in
[`tools/server/README.md`](../tools/server/README.md)).

The same roofline math as weights says KV quantization is cheap but bounded:
at 8k context KV is only ~20% of the per-token read, so `-ctk q8_0 -ctv q8_0`
buys ~10% decode speed;
its real value is that it *delays* the KV-crossover point (32k -> ~60k in the
8B example) and shrinks the resident KV so more slots fit. `q4_0` halves the
bytes but perturbs attention measurably - re-check perplexity before shipping.

Flash attention: `-fa on|off|auto` (default `auto`,
[`common/arg.cpp:2575`](../common/arg.cpp)) keeps attention scores in SRAM.
A quantized **V** cache requires it: `llama_context` errors with "quantized V
cache requires flash_attn to be enabled" (`src/llama-context.cpp:3757`) unless
FA is on or auto-resolves to on. A quantized K cache alone does not. On CUDA it
also gates the faster FA paths; on `tensor` split mode it is mandatory
([`docs/multi-gpu.md`](multi-gpu.md)).

Prompt and context reuse - the only optimization that removes *whole* weight
reads, by not running them again:

- `-cram, --cache-ram N` (default 8192 MiB; `-1` = no limit, `0` disables,
  [`common/arg.cpp:1719`](../common/arg.cpp),
  [`common/common.h:648`](../common/common.h)) keeps the prompt cache in RAM
  across requests.
- `--cache-reuse N` (default 0 = off) reuses cached tokens from a divergence
  point, but only for matches in chunks of >= N tokens; `--keep` pins tokens
  from the prompt front so shared prefixes survive eviction.
- `-ctxcp, --ctx-checkpoints N` (default 32,
  [`common/arg.cpp:1700`](../common/arg.cpp),
  [`common/common.h:645`](../common/common.h)) stores partial-context snapshots
  so a resumed slot rewinds instead of re-prefilling;
  `-cms, --checkpoint-min-step` (default 8192,
  [`common/common.h:647`](../common/common.h)) spaces them.
- KV placement knobs: `--no-kv-offload` (`-nkvo`) keeps KV on CPU;
  `--cache-idle-slots`/`--no-cache-idle-slots` (default enabled, requires
  cache-ram, [`tools/server/README.md`](../tools/server/README.md)) frees idle
  slots' cache space on new tasks; `-kvu/--kv-unified` shares one KV buffer
  across slots.

Economics of reuse: a cache hit removes `bytes_weights + bytes_kv` for every
reused token, i.e. ~40 ms x tokens on the example device - reuse dominates every
weight-level optimization when your traffic shares prefixes (system prompts,
multi-turn agents). Measure hit rates through `--slots` and `--metrics`
(section 9).

---

## 5. Batching and concurrency

Two different knobs that are easy to confuse:

- `-ub, --ubatch-size` (default **512**,
  [`common/common.h:462`](../common/common.h)) - tokens per graph execution.
  This is the *roofline knob* for prefill: intensity is `16*ubatch/bpw` (section
  1), so prefill stays bandwidth-bound below `T_crossover = balance*bpw/16`
  tokens - about 26 on the example device, not `balance/2`. On high-balance devices (data
  center GPU, balance 300+) raising `-ub` toward 2048 raises prefill throughput
  until compute saturates; `GGML_CUDA_GRAPHS` (build) then trims per-chunk
  launch overhead.
- `-b, --batch-size` (default **2048**,
  [`common/common.h:461`](../common/common.h)) - max tokens per prompt chunk;
  it caps prompt splitting, not decode.

Decode batching: `N` concurrent sequences on one slot set read the weights
**once** per step, so throughput scales ~`N x` until KV reads (`N x ctx x
128 KiB`) or compute dominate:

```
bytes/step(N) = bytes_weights + N * bytes_kv + N * overhead
tok/s(N)      ~ N * BW / bytes/step(N)
```

Server-side: `-np, --parallel N` (default -1 = auto,
[`tools/server/README.md`](../tools/server/README.md)); per-slot KV is the
cost, and each slot adds its own per-token attention read linear in its `ctx`,
so `N` and `ctx` jointly set the crossover to bandwidth- then compute-bound.
`--cont-batching` (on by default) fills freed slot space with waiting prompts;
`-sps, --slot-prompt-similarity` ([`common/arg.cpp:3861`](../common/arg.cpp))
controls prompt-similarity routing between slots, not raw speed. Use
`llama-batched-bench -np 1,4,16 -ngl 99 -n 256 -d 128` to measure the scaling
curve instead of guessing.

---

## 6. Speculative decoding economics

Types (exact strings, [`common/speculative.cpp:38-49`](../common/speculative.cpp),
flag `--spec-type`, env `LLAMA_ARG_SPEC_TYPE`): `none`, `draft-simple`,
`draft-eagle3`, `draft-mtp`, `draft-mtp-adaptive`, `draft-dflash`,
`draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`,
`ngram-cache`. Draft budget: `--spec-draft-n-max` (default 3,
[`common/common.h:327`](../common/common.h)), `--spec-draft-n-min` (default 0,
[`common/common.h:328`](../common/common.h)), `--spec-draft-p-min` (default
0.0, [`common/common.h:332`](../common/common.h)); `-md`
(`--spec-draft-model`/`--model-draft`,
[`common/arg.cpp:4306`](../common/arg.cpp)) points at a draft model.

Model it with three quantities: `t` = target decode time (= bytes/token / BW,
section 1), `alpha*t` = draft token time (`alpha` = draft cost ratio:
`alpha ~ P_draft/P_target` for a model draft; `alpha ~ 0` for ngram drafts),
`p` = per-token acceptance probability, `k` = draft length per cycle.

Expected accepted tokens per cycle (geometric run + the always-accepted verify
token):

```
E(p, k) = (1 - p^(k+1)) / (1 - p)   = 1 + p + p^2 + ... + p^k

speedup = E / (1 + k*alpha)
win  <=>  p + p^2 + ... + p^k > k*alpha   [equivalently E > 1 + k*alpha]
```

Two consequences that kill most speculative-decoding deployments before you
start: **acceptance alone is not enough** - the draft cost `k*alpha` is paid in
full every cycle, and a draft as expensive as the target (`alpha >= 1`) can
never strictly win since `p + ... + p^k < k`. The break-even acceptance solves
`p + p^2 + ... + p^k = k*alpha`: at `k = 3` that is `p ~ 0.24` for
`alpha = 0.1`, but already `p ~ 0.45` for `alpha = 0.25`.

Measured at the 8B example (`t = 39.3 ms`, `alpha = 0.1`, `k = 3`):

```
E(0.7, 3) = 2.533 tokens/cycle
cycle     = (1 + 3*0.1) * 39.3 = 51.1 ms
speedup   = 2.533 / 1.3 = 1.95x   ->  20.2 ms/token, 49.6 tok/s
```

| p     | E (k=3) | speedup (alpha=0.1) | speedup (alpha=0.25) | speedup (alpha=1, same-size draft) |
|-------|---------|---------------------|----------------------|------------------------------------|
| 0.5   | 1.875   | 1.44x               | 1.07x                | 0.47x                              |
| 0.6   | 2.176   | 1.67x               | 1.24x                | 0.54x                              |
| 0.7   | 2.533   | 1.95x               | 1.45x                | 0.63x                              |
| 0.8   | 2.952   | 2.27x               | 1.69x                | 0.74x                              |
| 0.9   | 3.439   | 2.65x               | 1.97x                | 0.86x                              |

(Every cell is `(1-p^(k+1))/((1-p)(1+k*alpha))` - the alpha=1 column is `E/4`,
all below 1, as the win condition predicts. The tutorial's
`verify-numbers.json` checks the `k=3`, `alpha=0.1` column.)

Choosing a method by economics, not name:

- **ngram types** (`ngram-simple`, `ngram-mod`, `ngram-cache`, ...): `alpha
  ~ 0` - the draft is a hash lookup, essentially free. Acceptance is low on
  novel text but the break-even is trivially met: they win on boilerplate,
  code edit loops, and agent traffic, and cost ~nothing when they miss.
  This is the default first experiment.
- **`draft-eagle3` / `draft-dflash` / `draft-dspark`**: tiny trained draft
  heads (`alpha ~ 0.05-0.2`) with high `p` - the best speedups, at the cost of
  a separate draft GGUF (`-md`).
- **`draft-mtp`**: native multi-token-prediction heads shipped with the model
  (Qwen3-Next class); no extra file, moderate `p`. `draft-mtp-adaptive` adapts
  the draft length to observed acceptance.
- **`draft-simple`** (small full model): `alpha` approaches 1 unless the draft
  is much smaller; solve `p + p^2 + ... + p^k = k*alpha` numerically for the
  break-even `p*` (at `k = 3`: `p* ~ 0.24` for `alpha = 0.1`, `p* ~ 0.45` for
  `alpha = 0.25`) before believing any benchmark.

Cross-links: [`docs/speculative.md`](speculative.md),
[`architecture-and-tuning.md` section 7.5](architecture-and-tuning.md).

---

## 7. Backend and build-time levers

Principles first, flags second.

1. **Kernel-launch overhead only matters when the pipe is idle.** Decode
   launches many small kernels; at `t_decode ~ 40 ms/token` across thousands of
   kernels, per-launch gaps are real. `GGML_CUDA_GRAPHS` (build) or
   `GGML_CUDA_ENABLE_UNIFIED_MEMORY`-adjacent runtime knobs capture the graph
   once per shape; CUDA graphs are shape-keyed, so odd `-ub`/`-b` combinations
   that change shapes per call defeat them. On CPU the threadpool amortizes
   launches differently; `-sm tensor` *requires* `-fa`
   ([`docs/multi-gpu.md`](multi-gpu.md)) partly to keep the graph simple.
2. **Dequant cost vs integer path on CUDA.** `GGML_CUDA_FA_ALL_QUANTS` (build,
   default `OFF`, [`docs/build.md`](../docs/build.md)) trades ~4 GiB compile
   memory and binary size for faster FA on exotic quant types; the CMake defines
   `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` force the
   dequant-to-bf16 cuBLAS path vs the native int4 matmul (MMQ) path. MMQ wins on
   small-batch (memory-bound, decode); cuBLAS bf16 wins at large prefill batch.
   Leave both unset (auto) unless a `llama-bench -p 1 -n 128` vs `-p 2048 -n 0`
   split says otherwise.
3. **Build for the silicon you have.** `GGML_NATIVE` (default ON unless
   cross-compiling, [`ggml/CMakeLists.txt:105-110,123`](../ggml/CMakeLists.txt))
   pins `-march` to the build host - fine locally, wrong for distribution; for
   a fleet build use `GGML_CPU_ALL_VARIANTS=ON` + `GGML_BACKEND_DL=ON`
   ([`ggml/CMakeLists.txt:86,183`](../ggml/CMakeLists.txt)) and let the runtime
   pick the best variant; `--list-devices`
   ([`common/arg.cpp:2776`](../common/arg.cpp)) prints what the server sees.
   `GGML_LLAMAFILE` (default OFF, [`ggml/CMakeLists.txt:113-114,197`](../ggml/CMakeLists.txt))
   is the tiled CPU matmul path; `GGML_BLAS` rarely beats it on CPU - measure.
   `GGML_CPU_REPACK` (default ON, [`ggml/CMakeLists.txt:152`](../ggml/CMakeLists.txt))
   trades model-load time for decode-time repacked weights;
   `GGML_CPU_KLEIDIAI` (default OFF, [`ggml/CMakeLists.txt:153`](../ggml/CMakeLists.txt))
   enables Arm SME kernels.
4. **Runtime env vars are backend-local and undocumented by design.** The
   names are the `getenv` strings in each backend file; grep them as ground
   truth, e.g. `grep -rhoE 'getenv\("GGML_VK_[A-Z0-9_]+"\)'
   ggml/src/ggml-vulkan | sort -u` for the Vulkan set (async, coopmat, transfer
   queue, memory-priority knobs), `ggml/src/ggml-metal` for Metal (graph,
   residency, fusion, `GGML_METAL_NCB`,
   [`ggml/src/ggml-metal/ggml-metal.cpp:583`](../ggml/src/ggml-metal/ggml-metal.cpp)),
   `ggml/src/ggml-cpu/kleidiai` for Arm (`GGML_KLEIDIAI_SME`,
   `GGML_TOTAL_THREADS`,
   [`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:309-310`](../ggml/src/ggml-cpu/kleidiai/kleidiai.cpp)).
   Treat all of them as diagnostics, not tuning: flip one, re-measure, keep or
   revert.
5. **Diagnostics that tell you which regime you are in**: `GGML_SCHED_DEBUG=2`
   (env var, [`ggml/src/ggml-backend.cpp:1879`](../ggml/src/ggml-backend.cpp))
   prints per-cgraph scheduling decisions including copy-splitting;
   `--fit-print` / `-fitp` ([`common/arg.cpp:2956`](../common/arg.cpp)) prints
   the computed VRAM budget (`common_fit_params`,
   [`common/common.cpp:1325`](../common/common.cpp); pair with `--fit-target`
   per-GPU MiB lists and `--fit-ctx`,
   [`common/arg.cpp:2995`](../common/arg.cpp)). Use them to confirm that a
   knob did what the model in section 1 predicts.

---

## 8. CPU and NUMA

Decode on CPU is bandwidth-bound too, but "bandwidth" means DDR/HBM and thread
coordination overheads the GPU hides:

- `-t N` / `-tb N` (batch threads): set to **physical** cores for prefill;
  more than physical cores helps little on memory-bound decode and hurts on
  SMT. CPU math-thread auto-detection lives in
  [`common/common.cpp:202`](../common/common.cpp)
  (`common_cpu_get_num_math`, used at [`common/common.cpp:296`](../common/common.cpp));
  `GGML_DEFAULT_N_THREADS = 4`
  ([`ggml/include/ggml.h:232`](../ggml/include/ggml.h)) is the library default
  when no pool is configured.
- `--numa distribute|isolate|numactl` (those three, exactly - `common/arg.cpp`):
  `distribute` spreads threads over all nodes, `isolate` keeps them on the node
  execution started on, `numactl` follows the map `numactl` supplied.
  Cross-socket reads are the classic unexplained 2x decode loss on servers.
  Drop the page cache before switching strategy.
- `--cpu-mask`/`--cpu-strict <0|1>` reserves cores (avoids GPU-daemon and
  scheduler interference); `--prio N` sets thread priority; `--poll` busy-waits
  instead of sleeping - trade CPU burn for lower inter-token latency at low QPS.
- `GGML_CPU_HBM` (build option,
  [`ggml/src/ggml-cpu/CMakeLists.txt:90`](../ggml/src/ggml-cpu/CMakeLists.txt))
  marks HBM as the CPU device memory; `GGML_CPU_REPACK` (build) +
  `--repack`/`--no-repack` runtime (default enabled,
  [`common/arg.cpp:2427`](../common/arg.cpp)) pre-packs weights once at load -
  worth the slower load for long-lived servers, wrong for one-shot CLIs.
  `--no-host` bypasses the host buffer so extra (quantized) buffers can be
  used ([`common/arg.cpp:2435`](../common/arg.cpp)).

---

## 9. Measurement and verification

Rule: model first, measure second. Every number in this document is reproducible
with the tools below; every tuning claim you make should come with one of these
commands attached.

- **Throughput curves** - `llama-bench`:
  `-m MODEL -p 512,2048,4096 -n 128,512 -d 0` (`-d` is
  `--n-depth` context depth, [`tools/llama-bench/llama-bench.cpp:618`](../tools/llama-bench/llama-bench.cpp);
  keep `-d 0` for clean prefill/decode splits), `-ngl 0,20,99`, `-fa 0,1`,
  `-ub 128,512`, `-t`, `-r 3` repetitions, `-o jsonl` for diffing. Use `-n 0`
  to isolate prefill and `-p 1 -n 128` to isolate decode - the roofline
  predicts the decode number within ~10%, so a large gap is a finding.
- **Concurrency** - `llama-batched-bench`: `-np 1,2,4,8 -n 256 -d 128 -ngl 99`
  gives the `tok/s(N)` curve of section 5; the knee is your `-np`.
- **Quality of a quant/KV type** - `llama-perplexity -m Q -f in-domain.txt`,
  `--hellaswag`, and `--kl-divergence-base F16.gguf` + `--kl-divergence`
  against the f16 reference (logits file, [`common/arg.cpp:2510-2522`](../common/arg.cpp)).
- **Op-level sanity** - `test-backend-ops` (catches a backend path silently
  falling back), `test-autorelease`, `test-tokenizer-0`.
- **Server counters** - `--metrics` (off by default) exposes Prometheus
  `/metrics` including tokens-per-second and cache hits; `--slots` (on by
  default) exposes `/slots` per-slot state; `-to` read/write timeout (default
  3600 s). Diff `/metrics` before/after each knob.
- **Arithmetic** - re-verify every closed-form estimate in this document with
  the tutorial evaluator:
  `python3` over
  [`tutorials/llama-cpp-inference/verify-numbers.json`](tutorials/llama-cpp-inference/verify-numbers.json)
  (145 checks, 0 fails at time of writing; covers the byte budget, the KV and
  weight crossovers, the quantization table, `E(p,k)` at `k=3` for `p=0.7` and
  `p=0.4`, and the `-ncmoe` versus `-ngl` comparison. Numbers in this document
  that are *not* in that spec - the multi-device table in section 1 and the
  alpha=0.25 and alpha=1 columns in section 6 - are closed forms you can
  re-derive from the formulas given beside them).

---

## 10. Symptom -> cause -> lever

| symptom                                | roofline reading                          | first levers |
|----------------------------------------|-------------------------------------------|--------------|
| decode far below `BW/bytes_read`       | pipe idle: launches, PCIe, CPU fallback   | `-ngl 99`, graphs build flag, `-fa`, check `-sm`/`-ot` split |
| prefill slow, GPU util low             | below `T_crossover = balance*bpw/16` tokens/step | raise `-ub` then `-b`, `-fa on` |
| long contexts fall off a cliff         | KV read overtakes weights; KV reads grow linear in ctx, prefill attn compute quadratically | `-fa on`, `-ctk q8_0 -ctv q8_0`, raise `-cram`, `-ctxcp` |
| repeated prefixes re-prefill           | cache miss, whole budget re-read          | `-cram`, `--cache-reuse 256`, `--keep`, `--cache-idle-slots` |
| `-ngl 99` OOM                          | `p_total*bpw + slots*KV` > VRAM           | `--fit-target MiB0,MiB1`, `-ncmoe`, `-ncffn`, `-ot`, `-ctv q8_0` |
| many users, low per-user tok/s         | weight reads amortized fine, per-token KV reads `N*ctx*bytes/tok` bind | lower per-slot `-c`, tune `-np` with `llama-batched-bench`, `-fa` |
| speculative speedup < 1                | `p + p^2 + ... + p^k <= k*alpha`          | shorter `--spec-draft-n-max`, raise `--spec-draft-p-min`, cheaper draft (ngram/eagle3) |
| CPU decode ~2x below RAM bandwidth     | NUMA cross-socket, SMT, sleep latency     | `--numa isolate`, `-t` physical cores, `--poll`, `--cpu-mask` |
| quant quality complaints               | wrong metric                              | `--kl-divergence-base` vs f16, per-tensor `-ot`, `q8_0` KV not weights |

## 11. Anti-patterns

- Tuning `-b` while confused about decode: `-b`/`-ub` are prefill knobs; decode
  batching is `-np`/slots.
- Quantizing KV to `q4_0` on a benchmark and shipping it without a perplexity
  or KL check; KV errors corrupt *attention*, which reads them every token.
- Speculative decoding against a same-size draft (`alpha ~ 1`): no `p` wins.
- `--cpu-moe`/`--n-cpu-moe` on a GPU server (it is for CPU-resident experts;
  see `-ncmoe` for GPU servers).
- Benching with `-d > 0` and quoting "decode" numbers that mix a prefill pass.
- Citing env-var names from memory or docs written before a rename: grep the
  `getenv` call sites in `ggml/src/` and `common/`, then verify `GGML_SCHED_DEBUG=2`
  shows the behavior you expected.
- Reading `-ngl`-hybrid throughput as the average of the two pipes; it is the
  slowest segment plus PCIe.

---

## See also

- [`architecture-and-tuning.md`](architecture-and-tuning.md) - full flag, env
  and build reference with source line anchors
- [`tutorials/llama-cpp-inference/tutorial.html`](tutorials/llama-cpp-inference/tutorial.html) -
  tutorial-level walkthrough; `verify-numbers.json` is the arithmetic source of
  truth for this document
- [`docs/speculative.md`](speculative.md), [`docs/multi-gpu.md`](multi-gpu.md),
  [`docs/build.md`](build.md)
- [`tools/server/README.md`](../tools/server/README.md) - endpoints, metrics,
  slots, API-key handling
- [`ggml-common.h`](../ggml/src/ggml-common.h) - block-format byte layouts
  behind every bpw figure here
