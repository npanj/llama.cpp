# llama.cpp: architecture and tuning levers

**What this is.** A developer's map of how llama.cpp is built, where the time and memory actually go, and every knob you can turn to make inference faster or fit in less memory. Written so you can both *deploy well* and *change the code*.

**How to read it.** Sections 1-5 are the architecture. Section 6 onward is the tuning reference. Each section stands on its own - you can jump straight to "I need more tokens/sec" without reading the rest.

**A note on this copy.** This is a fork (`nitin/mainline`). Features marked **[fork]** do not exist upstream. Everything else is upstream llama.cpp.

---

## Words we can't avoid

| Term | What it actually means |
|---|---|
| **ggml** | The tensor library underneath. Knows nothing about language models: just tensors, operations, and a graph of them. |
| **tensor** | An n-dimensional array plus its shape, stride, data type, and a pointer to bytes. Up to 4 dimensions. |
| **graph** (`ggml_cgraph`) | The list of operations for one forward pass, in the order they must run. Built fresh (or reused) on every step. |
| **backend** | One compute device plus the code to run ops on it. CPU, CUDA, Metal, Vulkan, etc. |
| **buffer type** (`buft`) | A kind of memory a backend can allocate: GPU memory, plain host RAM, pinned host RAM, a repacked-weights pool. |
| **scheduler** (`ggml_backend_sched`) | Decides which backend runs each op, inserts the copies between them, and allocates the scratch memory. |
| **quantization** | Storing weights in fewer bits than float32. Q4_K means roughly 4.5 bits per weight. Smaller file, less memory traffic, a little accuracy lost. |
| **GGUF** | The file format. One file holds metadata (architecture, hyperparameters, tokenizer) plus every tensor. |
| **KV cache** | The stored keys and values for every token seen so far, so attention does not recompute them. Usually the second-largest memory consumer after the weights. |
| **prefill / prompt processing** | Running the whole prompt through the model at once. Wide batches, compute-bound. |
| **decode / generation** | Producing one token at a time. Narrow batches, memory-bandwidth-bound. |
| **batch / ubatch** | A *batch* is what you hand to `llama_decode`. A *ubatch* ("micro-batch") is the slice actually pushed through the graph at once. One batch becomes several ubatches. |
| **slot** | In the server, one conversation's worth of state: its tokens, its KV cache range, its sampler. |
| **speculative decoding** | Guess several tokens cheaply, then check them all in one pass of the big model. Wrong guesses are thrown away. Correctness is unchanged. |

---

## 1. The 30-second model

llama.cpp is four layers. Each one is replaceable without touching the others, and that is the whole design.

```
  tools/            llama-server, llama-cli, llama-bench, llama-quantize, ...
                    argument parsing, HTTP, chat templates, sampling policy
  ------------------------------------------------------------------------
  common/           the "batteries": arg.cpp (every CLI flag), sampling,
                    speculative decoding, chat templates, downloads
  ------------------------------------------------------------------------
  src/ (libllama)   knows what a transformer is: model loading, KV cache,
                    batching, per-architecture graph building, tokenizer
  ------------------------------------------------------------------------
  ggml/ (libggml)   knows only tensors: graph build, backend registry,
                    scheduler, allocator, and one kernel set per device
```

Data flows down and results flow back up:

```
  GGUF file -> llama_model (tensors placed in backend buffers)
                    |
  tokens -> llama_batch -> ubatch -> build graph -> scheduler splits graph
                    |                                      |
                    |                             per-backend kernels
                    |                                      |
              KV cache read/write  <-----------------  results
                    |
              logits -> sampler -> next token
```

**The single most useful thing to internalise:** the graph is rebuilt (or reused) for *every* ubatch, and the scheduler re-decides device placement each time. There is no compiled, frozen execution plan. That is why placement flags like `-ot` and `-ngl` work at all, and it is also where a chunk of the per-token overhead lives.

---

## 2. ggml: tensors, graphs, backends

### 2.1 The tensor

`struct ggml_tensor` in `ggml/include/ggml.h`. The fields that matter:

| Field | Meaning |
|---|---|
| `type` | One of 40-odd data types: `F32`, `F16`, `BF16`, `Q4_K`, `Q8_0`, `MXFP4`, ... |
| `ne[4]` | Number of elements per dimension. |
| `nb[4]` | Stride in **bytes** per dimension. This is what makes views free. |
| `op`, `src[]` | The operation that produced it, and its inputs. A tensor *is* a graph node. |
| `buffer`, `data` | Which backend buffer holds the bytes, and where. |
| `view_src` | If non-null, this tensor is a window into another one and owns no memory. |

Quantized types are **block-structured**. A `Q4_K` block covers 256 weights and carries its own scales. This matters for two reasons: row sizes must be multiples of the block size, and dequantization happens inside the matmul kernel, never as a separate pass.

### 2.2 The graph

Building a graph allocates no data. `ggml_mul_mat(ctx, a, b)` just creates a node recording "multiply these two". Actual work happens once, at `ggml_backend_sched_graph_compute`.

Two consequences:

- **Shapes are known before any memory is touched.** This is what makes the "measure then allocate" path possible (`ggml_backend_sched_reserve`) and what powers `--fit`.
- **The graph is disposable.** Rebuilding it each step costs real CPU time at small batch sizes, which is why graph reuse and CUDA/Metal graph capture exist (section 7.8).

### 2.3 Backends: the three-level hierarchy

```
  reg     (ggml_backend_reg_t)     one shared library / one backend family: "CUDA", "Metal", "CPU"
   |
  device  (ggml_backend_dev_t)     one physical device: CUDA0, CUDA1, Metal0
   |
  backend (ggml_backend_t)         one execution stream on that device
```

A device advertises what it can do through three predicates, all in `ggml/include/ggml-backend.h`:

| Predicate | Question it answers |
|---|---|
| `supports_op` | Can I run this exact op, with these shapes and types, at all? |
| `supports_buft` | Can I read tensors that live in this kind of memory? |
| `offload_op` | This op's weights are in host RAM. Is it *still* worth copying them to me? |

`offload_op` is the quiet one. It is why a large prefill can be fast even with most layers on CPU: the batch is wide enough that shipping the weights to the GPU pays for itself. `GGML_OP_OFFLOAD_MIN_BATCH` sets the threshold.

Backends can be built as shared libraries (`GGML_BACKEND_DL=ON`) and discovered at runtime via `GGML_BACKEND_PATH`. That is how one binary ships with AVX2, AVX512 and AVX-VNNI CPU variants and picks the right one on start.

### 2.4 The scheduler: where placement is actually decided

`ggml/src/ggml-backend.cpp`, function `ggml_backend_sched_backend_id_from_cur`. For each node, in order:

1. **Already allocated?** Use the backend that owns its buffer. (cause `1.dst`)
2. **A view?** Follow `view_src`. (cause `1.vsrc`)
3. **A graph input?** Last backend, which is assumed to be the CPU. (cause `1.inp`)
4. **Has a weight input?** Run it where that weight lives - *unless* `op_offload` is on, the weight is in host memory, and a higher-priority backend says `offload_op` is worth it. (causes `1.wgt%d` / `1.off`)
5. Otherwise, inherit from the sources and let the assignment expand outward.

Then it groups consecutive same-backend nodes into **splits**. Every boundary between splits is a device-to-device copy.

**This is the number-one thing to look at when performance is mysteriously bad.** Set `GGML_SCHED_DEBUG=2` and read the split list. If your layer boundaries are producing dozens of splits per token instead of a handful, you are paying for copies, not compute.

The number of splits in the last graph is available programmatically via `ggml_backend_sched_get_n_splits`.

### 2.5 The allocator

`ggml/src/ggml-alloc.c`. Intermediate tensors are packed into one large arena per backend, with lifetimes computed from the graph so dead tensors' space is reused. The size is decided once by a measurement pass over a worst-case graph, then never grown at runtime (unless the graph shape changes, which triggers a re-reserve).

`GGML_SCHED_NO_REALLOC=ON` at build time makes any reallocation an error - useful when you are hunting for a graph shape that unexpectedly changes and causes a hitch.

---

## 3. libllama: from GGUF to logits

### 3.1 Model loading

`src/llama-model-loader.cpp` and `src/llama-model.cpp`.

1. Read GGUF metadata. This alone determines the architecture, hyperparameters, RoPE settings and tokenizer. (`--override-kv` patches it here.)
2. Decide a **buffer type per tensor**. This is the placement decision, and it obeys, in order: `--override-tensor` patterns, then `-ncmoe`/`-ncffn`, then `-ngl` plus `--tensor-split`.
3. Allocate the buffers and read the bytes in - by `mmap`, by plain read, or by direct I/O, per `--load-mode`.

Per-architecture graph code lives in `src/models/` - one file per architecture, 152 of them in this tree. `src/llama-arch.cpp` maps GGUF architecture names and tensor names onto the enums those files use.

**To add a model architecture**, you touch exactly: `llama-arch.{h,cpp}` (name and tensor mapping), `src/models/<yourarch>.cpp` (graph), `llama-model.cpp` (load and dispatch), and the Python conversion script. Nothing in ggml.

### 3.2 The context and its memory

`llama_context` (`src/llama-context.cpp`) owns everything mutable: the KV cache, the compute scheduler, the output buffers, the sampler chains.

The KV cache is not one class. It is a family, picked by what the architecture needs:

| Class | Used for |
|---|---|
| `llama_kv_cache` | Standard full attention. |
| `llama_kv_cache_iswa` | Interleaved sliding-window attention (Gemma 2/3, etc.): a small window cache plus a full one. |
| `llama_memory_recurrent` | Mamba / RWKV style state. Fixed size, no growth with context. |
| `llama_memory_hybrid` | Models that mix attention layers and recurrent layers. |
| `llama_kv_cache_dsa`, `-dsv4`, `-msa` | **[fork]** DeepSeek sparse-attention variants. |

**KV cache size, the formula that matters:**

```
  bytes = n_ctx * n_layer * n_head_kv * head_dim * 2 (K and V) * bytes_per_element
```

With `-ctk q8_0 -ctv q8_0` that last term goes from 2 bytes to roughly 1.06. On a long-context deployment this is often a bigger win than anything you do to the weights.

**Unified vs per-sequence KV** (`--kv-unified` / `--no-kv-unified`): unified means all sequences share one buffer and one attention mask. Good when your parallel requests share a long prefix (same system prompt). Bad when they are unrelated, because every sequence's attention still walks the full mask.

### 3.3 Batch to ubatch

`src/llama-batch.cpp`. You submit up to `n_batch` tokens. The batch allocator splits them into ubatches of at most `n_ubatch` tokens, grouped so that each ubatch is legal for the memory module (same sequence layout, equal split where required).

- `n_batch` (default 2048) bounds **one call** to `llama_decode`.
- `n_ubatch` (default 512) bounds **one graph execution**, and therefore peak compute-buffer size.

Bigger `n_ubatch` = better GPU utilisation on prefill, more scratch memory. This is a direct speed-for-memory trade.

### 3.4 Graph building and reuse

`src/llama-graph.cpp` holds the shared pieces every architecture composes: attention with all its variants, RoPE, MoE routing, normalisation, the input tensors.

Since the graph is rebuilt per ubatch, llama.cpp tries hard to **reuse** it. Each graph-input object implements `can_reuse(params)`. If the new ubatch has the same shape and the same memory context, the graph is kept and only the input tensor *contents* are refreshed. At single-token decode this saves a measurable slice of per-token CPU.

Set `LLAMA_GRAPH_REUSE_DISABLE=1` to turn it off and measure what it is worth on your setup.

### 3.5 The decode step, end to end

`llama_context::decode` in `src/llama-context.cpp`:

```
  validate batch -> balloc->init (split into ubatches)
        -> sched_reserve (ensure compute buffers are big enough)
        -> memory_update (apply pending KV shifts / copies / defrag)
        -> for each ubatch:
             memory->init_batch  (find KV slots; may fail and force a retry)
             build or reuse graph
             set input tensors (tokens, positions, attention mask, ...)
             ggml_backend_sched_graph_compute
             copy out logits / embeddings
        -> reorder outputs back into submission order
```

Two failure paths worth knowing: `init_batch` returning "no room" triggers a defrag-and-retry, and a graph whose shape changed forces a scheduler re-reserve. Both show up as latency spikes, not errors.

---

## 4. The server: slots and continuous batching

`tools/server/server-context.cpp`, function `update_slots()`. This is the scheduling core, and it runs in a loop.

**Slots.** `-np N` creates N slots. Each is one conversation: its own token list, its own KV range, its own sampler. A slot moves through:

```
  IDLE -> STARTED -> PROCESSING_PROMPT -> DONE_PROMPT -> GENERATING -> IDLE
```

**Continuous batching** is what happens next. On each iteration, `update_slots` builds *one* `llama_batch` containing:

- one token from every slot that is generating, plus
- as many prompt tokens as still fit under `n_batch` from slots that are still prefilling.

One `llama_decode` call then serves all of them. A slot that finishes does not stall the others, and a new request starts prefilling in the same pass as everyone else's generation. This is the single biggest reason server throughput beats running N copies of `llama-cli`.

**Prompt caching.** Three mechanisms, often confused:

| Mechanism | Flag | What it does |
|---|---|---|
| Slot prompt reuse | `--cache-prompt` (on by default) | A slot keeps its KV. A follow-up turn on the same slot only prefills the new tokens. |
| KV-shift reuse | `--cache-reuse N` | If the new prompt shares a prefix but diverges, shift the still-valid KV instead of recomputing it. N is the minimum chunk worth shifting. |
| Cross-slot RAM cache | `--cache-ram MiB` (default 8192) | Evicted slot states are held in host RAM and restored later, so a returning conversation skips prefill entirely. |

**Context checkpoints** (`-ctxcp`, default 32; `--checkpoint-min-step`, default 8192) snapshot KV state periodically so a rollback does not mean a full re-prefill. Especially valuable with sliding-window models, where the old KV is simply gone.

---

## 5. Where the time actually goes

Two regimes, and almost every tuning decision follows from which one you are in.

| | Prefill (prompt) | Decode (generation) |
|---|---|---|
| Batch width | hundreds to thousands of tokens | 1, or a few with speculation |
| Bound by | **compute** (matmul throughput) | **memory bandwidth** (reading the weights) |
| Helped by | bigger `-ub`, flash attention, more GPU layers | smaller weights, fewer bytes per token, speculation |
| Hurt by | small ubatch, CPU fallback on hot ops | per-token CPU overhead, graph rebuilds, scheduler splits |

**The decode arithmetic.** To produce one token, a dense model reads every weight once. A 7B model at Q4_K is about 4 GB. On a device with 400 GB/s usable bandwidth, the floor is 4/400 = 10 ms per token, i.e. ~100 tok/s. You cannot beat that by tuning; you beat it by reading fewer bytes (smaller quant, MoE, speculation which amortises one read across several tokens).

**Why MoE changes the picture.** An MoE model reads only the routed experts, so bytes-per-token is far below total parameters. That is exactly why `-ncmoe` works: expert weights are large but *cold*, so putting them in CPU RAM costs less than you would expect, while the hot dense path stays on the GPU.

---

## 6. Tuning levers: memory placement

This is where the biggest wins are on any machine where the model does not comfortably fit.

| Lever | What it does | When to reach for it |
|---|---|---|
| `-ngl N` | Put the first N layers on GPU. Default is auto (`-1`). | The blunt instrument. Start here, then stop using it. |
| `-ncmoe N` | Keep MoE expert weights of the first N layers on CPU, everything else on GPU. | **The best single flag for MoE models that do not fit.** Far better than lowering `-ngl`. |
| `-ncffn N` | Same idea for dense FFN weights. | Dense models that nearly fit. |
| `-ot REGEX=BUFTYPE` | Per-tensor placement by regex. `-ot 'blk\.(1[0-9])\.ffn_.*_exps=CPU'` | When you know exactly which tensors are cold. `-ncmoe` is a shorthand for a common case of this. |
| `--cpu-moe` | All expert weights on CPU. | Extreme memory pressure. |
| `-sm layer\|row\|none\|tensor` | Multi-GPU split strategy. | `layer` (default) pipelines; `row` parallelises each matmul across GPUs; `tensor` is experimental tensor parallelism. |
| `-ts a,b,c` | Proportion of the model per GPU. | Mismatched GPU sizes. |
| `-dev CUDA0,CUDA1` | Restrict which devices are used at all. | Leaving a GPU free for something else. |
| `--main-gpu N` | Which GPU holds scratch and small tensors. | With `-sm none`, or when one GPU has more headroom. |
| `-lm mmap\|mlock\|mmap+mlock\|none` | How weights are read. | `mlock` prevents swapping; `mmap` gives fast start and shared pages across processes. |
| `--direct-io` | Bypass the page cache on load. | Huge models where page cache churn hurts more than it helps. |
| `-lzm on\|auto\|off` | Read certain tensors (e.g. per-layer embeddings) from disk on demand. | Gemma 3n-style models with enormous embedding tables. |
| `--no-repack` | Disable runtime weight repacking into CPU-friendly layouts. | Debugging, or when repack memory cost is not worth it. |
| `--no-host` | Skip host buffers so "extra" buffer types (AMX, repack pools) can be used. | Rare; measure. |

**`--fit` is the automation of all of the above.** `common_fit_params` in `common/fit.cpp` loads the model with `no_alloc`, projects per-device memory, then reduces context and rewrites `-ngl` / `-ot` until it fits under a margin. `llama-fit-params` prints the arguments it would have used, so you can inspect and freeze them:

```bash
./build/bin/llama-fit-params --model model.gguf | tee args.txt
cat args.txt | xargs ./build/bin/llama-server --model model.gguf
```

Use `--fit-print` to see the projection without running, and `--fit-ctx` / `--fit-target` to steer it.

### MoE expert streaming **[fork]**

`--moe-stream` pages routed expert weights from the GGUF on demand into a small per-layer cache, instead of holding them resident.

| Flag | Meaning |
|---|---|
| `--moe-stream` | Turn it on. |
| `--moe-stream-cache N` | Cache slots per streamed layer (0 = auto from a byte budget). |
| `--moe-stream-io-threads N` | I/O threads for expert reads. |
| `--moe-stream-direct` | `O_DIRECT` reads, bypassing the page cache. |

Tuning env vars: `LLAMA_MOE_STREAM_LOOKAHEAD` (prefetch depth), `LLAMA_MOE_STREAM_LRU`, `LLAMA_MOE_STREAM_HOTNESS`, `LLAMA_MOE_STREAM_PARTITION`, `LLAMA_MOE_STREAM_WAVE_CAP`, `LLAMA_MOE_STREAM_STATS_MS` (periodic hit-rate stats - start here). Implementation is in `src/llama-moe-stream.cpp`.

---

## 7. Tuning levers: everything else

### 7.1 Batch shape

| Lever | Default | Effect |
|---|---|---|
| `-b N` | 2048 | Max tokens per `llama_decode` call. In the server, also the prefill chunk size. |
| `-ub N` | 512 | Max tokens per graph execution. **Raise for prefill throughput, lower to cut compute-buffer memory.** |
| `-np N` | 1 (server: auto) | Number of slots / parallel sequences. |
| `-c N` | 0 (from model) | Total context. In the server this is *shared* across slots unless `--kv-unified-per-slot` is set. |

Rule of thumb: on a GPU with headroom, `-ub 2048 -b 2048` often buys 1.5-2x on prompt processing. On a memory-tight setup, dropping `-ub` to 128 or 256 reclaims real megabytes.

### 7.2 KV cache

| Lever | Effect |
|---|---|
| `-ctk TYPE` / `-ctv TYPE` | Quantize the K and V caches. `q8_0` is the safe default; `q4_0` halves it again with a visible quality cost. |
| `--no-kv-offload` | Keep the KV cache in host RAM. Frees VRAM, costs bandwidth. |
| `--swa-full` | Keep a full-size cache for sliding-window models instead of just the window. Uses much more memory; needed for some reuse paths. |
| `--kv-unified` / `--no-kv-unified` | One shared KV buffer vs one per sequence. Unified wins on shared prefixes, loses on unrelated requests. |
| `--kv-unified-per-slot N` | Cap context per slot while still using a unified buffer. |
| `-ctxcp N`, `-cms N` | Context checkpoints and their minimum spacing. |
| `--context-shift` | Slide the window when context fills instead of stopping. Drops the oldest tokens. |

**`-ctv` needs flash attention.** V-cache quantization is only supported on the flash-attention path in most backends. If `-ctv q8_0` silently does nothing, check that `-fa` resolved to `on`.

### 7.3 Attention

| Lever | Effect |
|---|---|
| `-fa on\|off\|auto` | Flash attention. Default `auto`: enabled when the backend supports every required op at the model's head size. |
| `--attention causal\|non-causal` | For embedding models. |
| `--no-op-offload` | Stop the scheduler from shipping host-weight ops to the GPU. |

`-fa` is close to free when supported: less memory, fewer passes over the KV. The one reason to force it `off` is to check whether a numerical problem comes from the flash kernel.

### 7.4 Threading (CPU backend)

| Lever | Effect |
|---|---|
| `-t N` | Threads for generation. **Physical cores, not logical.** More threads than cores usually makes decode slower. |
| `-tb N` | Threads for batch/prefill. Can be higher than `-t`, since prefill is compute-bound. |
| `--cpu-mask HEX`, `--cpu-range LO-HI` | Pin to specific cores. Matters on big.LITTLE and on NUMA. |
| `--cpu-strict 0\|1` | Do not let the OS move threads off the mask. |
| `--poll 0..100` | Busy-wait level between graph nodes. Default 50. High = lower latency, more power. 0 = sleep, better for shared machines. |
| `--prio 0..3` | Process/thread scheduling priority. |
| `--numa distribute\|isolate\|numactl` | NUMA strategy. Drop the page cache before switching. |

On Apple silicon, `-t` should usually be the performance-core count (e.g. 6 or 8), not the total.

### 7.5 Speculative decoding

The lever with the largest headroom on single-stream latency. `docs/speculative.md` has the background; `--spec-type` selects the method:

| Type | What drafts the tokens |
|---|---|
| `draft-simple` | A separate small model (`-md`). Classic. |
| `draft-eagle3` | An EAGLE-3 head. |
| `draft-mtp` | The model's own multi-token-prediction head. |
| `draft-mtp-adaptive` **[fork]** | MTP with draft depth adapted to the recent acceptance rate. |
| `draft-dflash`, `draft-dspark` **[fork]** | DFlash / DFlash + Markov head. |
| `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` | Self-speculation from n-grams in the context. **No second model needed.** |

Core knobs: `--spec-draft-n-max` (how many tokens to draft, default 3), `--spec-draft-n-min`, `--spec-draft-p-min` (stop drafting when the draft model's confidence falls below this), `--spec-max-prompt` (disable speculation for very long prompts, where the draft cost stops paying).

The n-gram variants deserve a look before you reach for a draft model: they cost almost nothing, and on repetitive or code-heavy workloads the acceptance rate is high.

### 7.6 Sampling

Sampling is a **chain** of `llama_sampler` objects (`src/llama-sampler.cpp`), applied in the order given by `--samplers`. Available: `top_k`, `top_p`, `min_p`, `typical_p`, `temp`, `dynatemp`, `xtc`, `top_n_sigma`, `mirostat`, `dry`, penalties, grammar, logit bias.

Two performance notes:

- **`--top-k` bounds the work of every later sampler.** With a 150k-token vocabulary, an unbounded chain sorts the whole thing every token. `--top-k 40` (the default) keeps it cheap.
- **`--backend-sampling`** runs the chain on the GPU, so the logits never cross the bus. Worth measuring on large-vocabulary models.

Constrained output (`--grammar`, `--json-schema`) filters logits every token. A tight grammar can *increase* throughput by shrinking the candidate set; a pathological one can dominate the step.

### 7.7 Build-time options

These are decided by CMake and cannot be changed at runtime.

| Option | Effect |
|---|---|
| `GGML_NATIVE` | Build for this exact CPU. On by default; turn **off** for portable binaries. |
| `GGML_CPU_ALL_VARIANTS` | Build every CPU ISA variant and dispatch at runtime (needs `GGML_BACKEND_DL`). |
| `GGML_AVX512`, `GGML_AVX_VNNI`, `GGML_AMX_INT8`, ... | Explicit ISA enables. |
| `GGML_CPU_REPACK` | Runtime repacking of Q4_0 into blocked layouts. On by default. |
| `GGML_CPU_KLEIDIAI` | KleidiAI ARM kernels. |
| `GGML_LLAMAFILE` | tinyBLAS sgemm kernels. |
| `GGML_BLAS` + `GGML_BLAS_VENDOR` | External BLAS for large prefill matmuls. |
| `GGML_CUDA_FORCE_MMQ` | Always use the quantized matmul kernels instead of cuBLAS. |
| `GGML_CUDA_FA_ALL_QUANTS` | Compile flash attention for every KV quant combination. Long build, more flexibility. |
| `GGML_CUDA_GRAPHS` | CUDA graph capture. Cuts per-token launch overhead. |
| `GGML_SCHED_MAX_COPIES` | Input copies for pipeline parallelism across GPUs. Default 4. |
| `GGML_LTO` | Link-time optimization. |

### 7.8 Backend environment variables

Runtime escape hatches, mostly for bisecting a performance or correctness problem. The full list is discoverable with:

```bash
grep -rhoE 'getenv\("[A-Z_0-9]+"\)' ggml/src src common | sort -u
```

The ones you will actually use:

**Diagnostics (all backends)**

| Variable | Use |
|---|---|
| `GGML_SCHED_DEBUG=2` | Print the graph split list with placement causes. **Start every performance investigation here.** |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | Measure what graph reuse is buying. |
| `LLAMA_KV_CACHE_DEBUG=1` | KV cell allocation trace. |
| `LLAMA_BATCH_DEBUG=1` | Batch and ubatch splitting trace. |

**CUDA**

`GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY` (spill to host instead of OOM), `GGML_CUDA_NO_PINNED`, `GGML_CUDA_P2P`, `GGML_CUDA_PDL`, `GGML_CUDA_DEVICES`.

**Metal**

`GGML_METAL_FUSION_DISABLE`, `GGML_METAL_GRAPH_OPTIMIZE_DISABLE`, `GGML_METAL_CONCURRENCY_DISABLE`, `GGML_METAL_NCB` (command buffers), `GGML_METAL_MMID_MIN_TOKENS` and `GGML_METAL_MMID_GROUPED` (MoE indirect matmul), `GGML_METAL_CAPTURE_COMPUTE` (GPU trace capture), `GGML_METAL_BF16_DISABLE`.

**Vulkan**

`GGML_VK_DISABLE_COOPMAT` / `COOPMAT2`, `GGML_VK_FORCE_MMVQ`, `GGML_VK_MAX_NODES_PER_SUBMIT`, `GGML_VK_PERF_LOGGER`, `GGML_VK_VISIBLE_DEVICES`, `GGML_VK_PREFER_HOST_MEMORY`.

**CPU**

`GGML_CPU_DISABLE_FUSION`, `GGML_KLEIDIAI_SME`, `GGML_TOTAL_THREADS`.

**The pattern:** every `_DISABLE_` variable exists so you can prove an optimization is or is not responsible for what you are seeing. Use them to bisect, not to deploy.

---

## 8. How to know if it worked

Never tune without measuring. llama.cpp ships the instruments.

### `llama-bench` - the standard measurement

```bash
./build/bin/llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -fa 1
```

- `-p N` prompt-processing throughput at N tokens (prefill regime).
- `-n N` generation throughput for N tokens (decode regime).
- `-d N` run with N tokens already in the KV cache. **Use this.** Decode speed at depth 0 is not the number your users experience.
- Comma-separate any value to sweep: `-ub 128,256,512,1024`.
- `-r N` repetitions; `-o json|csv|md` output format.

### `llama-batched-bench` - throughput under concurrency

Sweeps prompt length x generation length x parallel sequences. This is the tool for choosing `-np`, `-b` and `-ub` for a server deployment, because it measures the thing a server actually does.

### `llama-perplexity` - did quality survive?

```bash
./build/bin/llama-perplexity -m model.gguf -f wiki.test.raw
```

Run it before and after a quantization or KV-type change. Also does `--hellaswag`, `--winogrande`, `--multiple-choice`, and `--kl-divergence` (which compares against a reference model's logits - the sharpest signal that a quantization hurt).

### Built-in accounting

- `--fit-print` - projected per-device memory before loading anything.
- The memory breakdown table printed at startup, per device: model / context / compute.
- `/metrics` on the server (`--metrics`) - Prometheus counters for tokens, queue depth, slot occupancy.
- `/slots` on the server (`--slots`) - live per-slot state.
- `--perf` / the per-run timing summary: prompt eval time, eval time, and the per-token split.

### `test-backend-ops` - correctness and per-op speed

```bash
./build/bin/test-backend-ops test -o MUL_MAT      # numerics vs CPU reference
./build/bin/test-backend-ops perf -o MUL_MAT      # per-op throughput
```

**This is the tool for kernel work.** It compares every backend against the CPU reference on the exact shapes the models use, and `perf` mode gives you a microbenchmark without a model in the loop.

---

## 9. Playbooks

**Model does not fit in VRAM, and it is MoE.**
`-ncmoe N` and bisect N downward from the layer count until it fits. Do not touch `-ngl`. Then `-ctk q8_0 -ctv q8_0 -fa on`. Then, if still short, `--moe-stream` **[fork]**.

**Model does not fit, and it is dense.**
`-ctk q8_0 -ctv q8_0 -fa on` first (cheapest quality cost). Then `-ncffn N`. Then a smaller quant of the weights. Lower `-ngl` last - it is the bluntest option.

**Prompt processing is slow.**
Raise `-ub` (512 -> 1024 -> 2048) and `-b` to match. Confirm `-fa` is on. Check `GGML_SCHED_DEBUG=2` for split count. On CPU, raise `-tb`.

**Generation is slow, single user.**
You are bandwidth-bound. Smaller quant, or speculative decoding. Start with `--spec-type ngram-map-k4v` (free), then a draft model. Check `-t` equals physical core count. Confirm CUDA/Metal graphs are not disabled.

**Server throughput is low with many users.**
Raise `-np`. Ensure `--cont-batching` (default on). Raise `-b` so prefill chunks are large enough to batch with generation. Use `--kv-unified` if requests share a system prompt, `--no-kv-unified` if they do not. Size `--cache-ram` to your conversation reuse rate.

**Long context is slow or OOMs.**
`-ctk`/`-ctv` quantization, `-fa on`. For sliding-window models leave `--swa-full` off. Raise `-ctxcp` so rollbacks do not re-prefill.

**Something regressed and you do not know what.**
`GGML_SCHED_DEBUG=2` for placement. Then bisect with the `_DISABLE_` env vars: fusion, graph optimize, graph reuse, graph capture. Then `test-backend-ops perf` on the suspect op.

---

## 10. Improving llama.cpp: where the code is

| What you want to change | Where to work | How to verify |
|---|---|---|
| A faster kernel | `ggml/src/ggml-<backend>/` | `test-backend-ops test -o OP` then `perf -o OP` |
| A new op | `ggml.h` + `ggml.c` (shape/meta), CPU reference in `ggml-cpu/ops.cpp`, then each backend, then a case in `tests/test-backend-ops.cpp` | same |
| Op fusion | Each backend's graph-compute loop, using `ggml_can_fuse` / `ggml_can_fuse_subgraph` from `ggml-impl.h` | A/B with `GGML_<BACKEND>_DISABLE_FUSION` |
| Placement / scheduling | `ggml/src/ggml-backend.cpp` | `GGML_SCHED_DEBUG=2`, split count |
| Memory allocation | `ggml/src/ggml-alloc.c` | The startup memory breakdown |
| A new model architecture | `llama-arch.{h,cpp}`, `src/models/<arch>.cpp`, `llama-model.cpp`, the convert script | Perplexity vs the reference implementation |
| KV cache behaviour | `src/llama-kv-cache*.cpp`, `src/llama-memory*.cpp` | `LLAMA_KV_CACHE_DEBUG=1`, long-context perplexity |
| Batching / graph reuse | `src/llama-batch.cpp`, `src/llama-context.cpp`, `src/llama-graph.cpp` | `llama-bench -d`, `LLAMA_BATCH_DEBUG=1` |
| Server scheduling | `tools/server/server-context.cpp` (`update_slots`) | `llama-batched-bench`, `/metrics` |
| Sampling | `src/llama-sampler.cpp` | `tests/test-sampling.cpp` |
| A quantization type | `ggml-common.h` (block layout), `ggml-quants.c` (reference), per-backend kernels, `llama-quant.cpp` (selection) | Perplexity and KL divergence vs F16 |
| A new backend | A directory under `ggml/src/`, implementing the `ggml_backend_*_i` interfaces | `test-backend-ops test -b YOURBACKEND` |

### The kernel-work loop

1. `test-backend-ops perf -o <OP>` to get a baseline at the shapes that matter.
2. Change the kernel.
3. `test-backend-ops test -o <OP>` - numerics against the CPU reference. Non-negotiable.
4. `test-backend-ops perf -o <OP>` again.
5. `llama-bench` end to end, because a faster kernel that changes the graph split pattern can be a net loss.

**[fork]** `tools/tuning/` sweeps a Metal kernel's config grid on the machine it runs on and emits pasteable rows for `ggml-metal-tuning.cpp`. It handles thermal throttling by re-measuring a baseline anchor every four candidates and discarding drifted measurements. See `tools/tuning/README.md`.

### Before you upstream anything

Read `AGENTS.md` and `CONTRIBUTING.md` in this repo. The short version: llama.cpp is deliberately kept simple, every merged line is maintained forever by a small team, and a simpler change that does 90% of the job beats a complex one that does 100%. Discuss in an issue before building.

---

## 11. What this fork adds

For orientation when reading code that is not upstream:

| Area | Files |
|---|---|
| MoE expert streaming from SSD | `src/llama-moe-stream.{h,cpp}`, `--moe-stream*`, `LLAMA_MOE_STREAM_*` |
| DeepSeek sparse attention (v3.2 / v4) | `src/llama-kv-cache-dsa*.cpp`, `-dsv4.cpp`, `-msa.cpp`, `LLAMA_DSV4_*` |
| Qwen4-experimental sparse attention and MTP | `src/models/qwen*`, `LLAMA_QWEN4EXP_*` |
| Adaptive MTP draft depth | `common/speculative-adaptive.h`, `--spec-type draft-mtp-adaptive` |
| DFlash / DSpark speculation | `src/models/dflash.cpp`, `--spec-type draft-dflash\|draft-dspark` |
| Metal kernel tuning and profiling | `ggml/src/ggml-metal/ggml-metal-tuning.cpp`, `tools/tuning/`, `GGML_METAL_*` |
| Params fitting | `common/fit.cpp`, `tools/fit-params/` (note: `--fit` also exists upstream) |

---

## Further reading in this repo

- `docs/build.md` - build options per platform
- `docs/multi-gpu.md` - multi-GPU splitting in depth
- `docs/speculative.md` - speculative decoding background
- `docs/ops.md` - which backend supports which op (generated)
- `tools/server/README.md` - every server flag and endpoint
- `tools/server/README-dev.md` - server internals
- `tools/fit-params/README.md` - the fitting algorithm
- `tools/tuning/README.md` - **[fork]** Metal kernel tuning
