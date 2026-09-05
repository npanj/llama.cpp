# FreeToken: what is worth borrowing

> Review of FlashML-org/FreeToken as a source of ideas for the `qwen4exp` work here, and as a
> possible benchmark target on Apple Silicon. Claims verified against their source, their pull
> request queue and this tree on 2026-09-01. The original draft of this file was a third-party
> review; everything below is what survived checking.

---

## Summary

**Do not benchmark Flash-Next through FreeToken.** It pins the whole expert bank in host RAM and
never reads experts from disk at inference. Their stated floor is 96 GB of host RAM, 128 GB
recommended. This machine has 64 GB. Their macOS pull request is a wrapper that spawns
`llama-server` or `mlx_lm.server`, so it would measure this tree anyway.

Of five techniques proposed for porting, **one led to a real win here** (and the win was the
opposite of what the proposal said), one names a real gap with a small prize, and three do not hold.

---

## What FreeToken is

Six weeks old at time of review (created 2026-07-20). 11k stars, 1023 forks. Of 143 pull requests,
**15 are merged, and 12 of those 15 are by one maintainer**. Outside contributions merge at roughly
2%.

That matters for reading any analysis of this project: four of the five pull requests originally
cited as "FreeToken's design" are unmerged proposals, not shipped behaviour.

### Apple Silicon status

Their roadmap (issue #79, maintainer-authored) lists *"macOS: a native Metal engine on Apple Silicon
Macs"* with no owner and no linked branch. A maintainer told issue #9 on 2026-08-21 that Apple
Silicon was "our first priority"; eleven days later no commit on `main` mentions Metal.

PR #65 is the only Apple pull request. It has merge conflicts, has not moved since 2026-08-24, spans
+4311 lines over 36 files, and no maintainer has commented on it. Its author states plainly that the
Metal backend "uses MLX/llama.cpp" and FreeToken adds serving and API compatibility.

### The architectural difference that settles it

`python/freetoken/moe/offload_cache.py` contains no disk I/O at all: no `pread`, no `open`, no
`io_uring`, no `O_DIRECT`. It moves bytes from pinned host RAM into VRAM. `O_DIRECT` appears only in
`expert_banks.py`, the load-time path that fills those host banks once and holds them for the
process lifetime.

| | FreeToken | here |
|---|---|---|
| Expert tier | pinned host RAM, never disk at inference | SSD, streamed per layer |
| Transfer | host to device over PCIe | none, unified memory |
| Host RAM floor | 96 GB min, 128 GB recommended | 64 GB works |

They also have **no merged speculative decoding** (MTP/DFlash/DSpark are roadmap items; #69, #70,
#71, #258 are open) and no merged GGUF support.

---

## The five proposals

| Proposal | Source | Verdict |
|---|---|---|
| KV / expert-cache laddering | PR #300 (open) | premise **confirmed by measurement**, but the static fix captures the whole win. See below |
| Expert dedup across the batch | PRs #46/47/49/52 (open) | real gap here, small prize. Their PRs are x86 CPU kernels; #52 is Intel AMX |
| Fused top-k + softmax router | PR #319 (merged) | **not an optimisation.** The PR deletes a dependency branch so everything routes through their existing Triton router. No benchmark in it |
| Double-buffered prefetch, D2D gather | `offload_cache.py` | **does not apply.** "D2D gather instead of re-transferring from host" avoids a PCIe copy. Unified memory has no such copy |
| PLE table streaming | PR #311 (merged) | already done here in `6e54c007f`, but **their read-path finding beat ours.** See below |

---

## What the checking produced

### 1. KV allocation really does steal memory, and the static fix is enough

Measured on this machine (M5 Pro, 64 GB) with `Qwen3.8-Flash-Next-Q4_0-Q8out` (93.82 GiB) and the
Q4_K_M MTP head. One short generation per arm, `vm.swapusage` sampled around each:

| ctx | cache | MTP | result |
|---:|---:|---|---|
| 32768 | 36 | yes | clean, generates |
| 65536 | 36 | yes | clean, generates |
| 98304 | 36 | yes | clean, generates |
| 131072 | 36 | yes | **GPU OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), 3.4 GB swapped |
| 131072 | 32 | yes | clean, generates |

Raising `-c` alone, with everything else fixed, takes the configuration from working to not working.
So the full-context KV cache does compete with the expert cache, and Metal does not quietly avoid
committing it.

What cache 36 buys, interleaved A/B/A/B, `-p 4096 -n 128 -r 2`, both arms swap-clean in pass 1:

| pass | cache | pp4096 t/s | tg128 t/s |
|---:|---:|---:|---:|
| 1 | 36 | 337.20 | 21.78 |
| 1 | 32 | 291.06 | 20.61 |
| 2 | 36 | 329.49 | 21.06 |
| 2 | 32 | 283.41 | 18.83 |

**Cache 32 costs about 14% of prefill.** That independently reproduces the -13.8%/-14.1%/-13.8%
recorded in the research log on different hardware and a different checkpoint.

**Conclusion: use `--moe-stream-cache 36` for any session under about 98k context.** Dynamic
laddering would only add value for a session that starts short and later needs more than ~98k, and
the implementation is invasive: the KV cache is one flat tensor per layer with `kv_size` baked into
about twenty stride calculations (`src/llama-kv-cache.cpp:234`), and the expert cache is allocated
at model load, before the context exists (`src/llama-moe-stream.cpp:229`). Not worth it.

### 2. The PLE read path: our note was right, and the first re-test was wrong

FreeToken's merged #311 streams the PLE table with `O_DIRECT` and states *"No RAM cache: an on/off
A/B showed zero decode difference."* This tree deliberately went the other way, and
`llama-moe-stream.h` recorded "tiny PLE rows benefit from the page cache".

`LLAMA_MOE_STREAM_PLE_DIRECT` now toggles it. **Measured on a correct build, the two are level**,
which matches FreeToken's own "zero decode difference" more closely than it matches any win:

| arm | order | pp t/s | tg t/s |
|---|---|---:|---:|
| buffered | first | 313.37 | 24.03 |
| direct | second | 317.98 | 22.91 |
| direct | first | 321.84 | 22.96 |
| buffered | second | 319.21 | 22.65 |

Means: pp +1.1% for direct, tg -1.7%. Both sit inside the buffered arm's own spread across four
loads (2.7% prefill, 5.9% decode). No effect.

An earlier version of this section reported +7% prefill and +21% decode for direct reads. **That is
withdrawn.** It was measured against a read path that overran its destination buffer by up to 8 KB
per row and delivered the wrong bytes, on a harness whose prompts were 4 tokens long after prompt
cache reuse. Details in the research log under "PLE reads: buffered vs uncached".

The useful residue is the bug, not the benchmark: the direct read is block-aligned and needs a
staging buffer with head/tail slack, which the expert path always had and the PLE path did not.

### 3. The batch-32 dispatch cliff is real, and it is a compute-shape problem

Metal's `MUL_MAT_ID` takes the grouped matrix path only at batch >= 32
(`ne21_mm_id_min`, `ggml/src/ggml-metal/ggml-metal-ops.cpp`). Below that it dispatches
`n_expert_used * n_tokens` independent vector products. An MTP verify batch of 5 lands on the wrong
side.

`test-backend-ops perf -o MUL_MAT_ID`, one shape, this machine:

| n | us/run | us/token | TFLOPS |
|---:|---:|---:|---:|
| 1 | 74.93 | 74.93 | 1.34 |
| 2 | 144.66 | 72.33 | 1.39 |
| 3 | 216.07 | 72.02 | 1.40 |
| 4 | 287.98 | 72.00 | 1.40 |
| 6 | 430.32 | 71.72 | 1.40 |
| 8 | 572.37 | 71.55 | 1.41 |

Perfectly linear: **4.5% per-token improvement from n=1 to n=8**, throughput flat. Adding tokens to
the batch buys nothing below the threshold.

Dedup is not the lever. At 5 tokens x 10 experts drawn from 512, only about 3 of 50 draws collide,
so perfect dedup is worth roughly 6%. The bigger question is whether the threshold itself is right
on this hardware.

---

## Reference points

Their published Flash-Next numbers, for scale. Not comparable hardware: a 5090 with the whole expert
bank resident, against unified memory streaming ~94 GiB off SSD.

| setup | pp t/s | tg t/s | source |
|---|---:|---:|---|
| RTX 5090 32 GB + 96 GB RAM, PLE mmap | 3045 @32k | 44.75 @0k / 39.02 @128k | PR #279 |
| RTX 5070 Ti 16 GB + 125 GiB RAM | - | 18.69 | issue #214 |
| here, M5 Pro 64 GB, cache 36, no spec | 337 @4k | 21.78 | this review |

The decode gap is mostly hardware. The prefill gap is the interesting one, and it points where the
research log already pointed: prefill here is expert-I/O-bound, and they never touch disk for
experts.

---

## Not doing

- Benchmarking against FreeToken. It cannot load this model on 64 GB.
- Porting the fused router or the D2D gather. One cites a PR containing no optimisation, the other
  solves a problem unified memory does not have.
- Tracking their pull request queue. Watch issue #79 and the `qwen4_exp` commits on `main`.
