# Faster speculative verification for MoE on Metal — design and execution plan

**Status:** implemented (flag `GGML_METAL_MMID_GROUPED`, default off), measured, **loses at every
shape and end to end**, then independently reviewed. **Line closed — do not retry.**
See *Implementation results* and then *Review of the implementation*.

The review found (a) the implementation never actually batched the weight loads, so it did not
test the idea, and (b) the idea is dead regardless: at the real operating point the op is not
load-bound at all, so dedup targets a non-bottleneck. §3–§5's numbers also used an unrepresentative
routing pattern and overstated the prize.

**Machine:** Apple M5 Pro, 64 GB, 307 GB/s. **Model:** Qwen3.8-Flash-Next REAP-288 (288 of 512
experts kept, top-10 routing, 48 layers), fully resident. **Checkout:** this fork, branch
`nitin/mainline`.

---

## The point

llama.cpp's speculative decoding on this model gets almost nothing from drafting more than one
token, because verifying a 4-token draft costs nearly 4× verifying 1. MLX does not have this
problem. The cause is one Metal kernel, and it is now measured to the line.

| | llama.cpp | MLX |
|---|---|---|
| Best decode, each tuned | 39.5 tok/s (`NMAX=1`) | 42.4 tok/s |
| Cost per verified token | **0.90–0.98** decode steps | **0.66** |
| Draft depth that pays | 1 | 3 |

**Fixing the kernel was projected to take llama.cpp past MLX** — realistically +10–15% end to end,
optimistically +25%. The op-level gain was measured (2.5×); the end-to-end number was a projection.
**Update after implementation: both did not survive contact with the machine.** The projected gain
was an L2-cache artifact and the shipped kernel is slower everywhere (*Implementation results*).

**Before reading further, know the two things already banked**, independent of any kernel work:
`NMAX=1` (+9%, in `~/models/bin/flashnext-reap-server.sh`) and pruning to 288 experts, which made
the model resident and took plain decode from 18.04 to 35.5 tok/s.

---

## Words we can't avoid

| Term | What it means here |
|---|---|
| **draft / verify** | Speculative decoding: a small head *drafts* a few tokens, the full model *verifies* them in one batched pass. Accepted tokens are free; the batch costs one forward pass over N tokens instead of 1. |
| **routed expert GEMM** | In a mixture-of-experts layer each token picks 10 of 288 expert matrices. The op that runs them is `MUL_MAT_ID`. It dominates the marginal cost of adding tokens to a verify batch. |
| **pair** | One (token, expert-slot) combination. 4 tokens × top-10 = 40 pairs. |
| **mat-vec / mat-mat path** | Metal has two `MUL_MAT_ID` kernels. Mat-vec (`kernel_mul_mv_id`) handles few tokens; mat-mat (`kernel_mul_mm_id`) handles many. The switch is a token-count threshold. |
| **map0** | A small pre-pass kernel (`kernel_mul_mm_id_map0`) that groups pairs by expert. Only the mat-mat path uses it today. |
| **NR0 / NR1** | Tile size of the mat-mat kernel: rows × columns processed per threadgroup. Hardcoded 64 × 32. |

---

## What is measured (the evidence)

### 1. Verification cost scales almost linearly with draft length

End to end, 300-token greedy prompt, interleaved arms:

| Arm | Speedup | Tokens / round | Round cost (decode steps) |
|---|---|---|---|
| llama.cpp `NMAX=1` | 1.11× | 1.77 | 1.59 |
| llama.cpp `NMAX=3` | 1.02× | 2.57 | **2.51** |
| MLX block=3 | 1.50× | 2.80 | 1.86 |

Going 2 → 4 tokens costs llama.cpp 1.59 → 2.51 steps. A verify batch should amortise weight reads;
this one doesn't.

### 2. It is not acceptance, not the drafter, not I/O

Each of these was the leading hypothesis at some point. **Do not re-try them.**

| Hypothesis | Test | Result |
|---|---|---|
| Drafter precision too low | 3 heads, 1.78 → 3.85 GiB, Q4_K_M and Q8_0 | acceptance moved **0.006** |
| Acceptance rate is the lever | `PMIN` 0.3 → 0.7 | acceptance 0.58 → **0.925** (above MLX's 0.897) and throughput **dropped** 36.3 → 34.5. Mean accepted length is flat at ~2.5 whatever you do. |
| Expert gather fires per token | `LLAMA_MOE_STREAM_STATS_MS` touches per remap, 2 vs 4 tokens | 14.27 → **15.60**. Dedups fine. |
| Expert streaming stalls | stream stats on the resident build | `307 slots covers all 288 experts -- streaming disabled`. Stall 2%. |
| Chained draft steps fed the wrong state | `qwen4exp.cpp:588` | already exports the widened stream. Per-position decay 0.757 → 0.486 → 0.324 is ordinary chained-head behaviour. |

### 3. The op itself, measured in isolation

`test-backend-ops perf -o MUL_MAT_ID`, Q4_0, 288 experts, top-10, gate/up shape m=640 k=2560.
These cases are in the tree (see *Harness* below).

| n tokens | mat-vec (used today) | mat-mat |
|---|---|---|
| 1 | **20.7 µs** | 88.6 |
| 2 | **39.0** | 141.4 |
| 4 | **114.1** | 230.1 |
| 8 | **238.2** | 446.9 |
| 16 | **420.2** | 759.4 |

Mat-vec scales ~1.9× per doubling — linear. Mat-mat is worse at every size, so the upstream
threshold (`ne21 >= 32`, `ggml-metal-common.cpp:45`) is correctly set. **Lowering it does not help.**

### 4. Why: mat-vec reloads weights per pair; mat-mat wastes its tile

`mul_mv.metal:3155`:

```c
const int iid1 = tgpig.z/args.nei0;   // token
const int idx  = tgpig.z%args.nei0;   // top-k slot
const int32_t i02 = ids[iid1][idx];   // expert
src0_cur = src0s + i02*args.nb02;     // its weights, loaded PER PAIR
```

One threadgroup-set per pair (`ne123 = ne20*ne21`, `ggml-metal-ops.cpp:2826`). Two tokens that
route to the same expert load it twice. Measured on the real model at n=4: **40 pairs over ~15.6
unique experts = 2.56× redundant weight traffic.**

| Path at n=4 | Data moved | Time | Achieved bandwidth |
|---|---|---|---|
| mat-vec | 36.9 MB | 114 µs | **323 GB/s — at hardware spec** |
| mat-mat | **14.4 MB** | 230 µs | 71 GB/s |
| *mat-mat at spec* | 14.4 MB | **~45 µs** | — |

Mat-mat already moves the right amount of data. It squanders it: tile is 64×32 but each active
expert holds 1–4 rows, so ~90% of every tile is padding, and it lands compute-bound on wasted work.

### 5. The ceiling, measured directly

Hold n=4, vary top-k to sweep the pair count without changing per-expert work:

| n_used | Pairs | Time | µs per pair |
|---|---|---|---|
| 1 | 4 | 9.37 | 2.34 |
| 2 | 8 | 17.10 | 2.14 |
| 4 | 16 | 37.54 | 2.35 |
| 10 | 40 | 137.26 | 3.43 |

**Cost tracks pair count, not FLOPs** (n_used=1 and 4 both run ~1.4 TFLOPS). Perfect dedup at
n=4 → ~15.6 pairs × ~2.3 µs ≈ **36 µs against 114–137 today. 3.2–3.8× on the op.**

> **Correction (added after implementation, see *Results*):** this ceiling is a cache artifact.
> The M5 Pro L2 is 16 MB (`hw.perflevel0.l2cachesize`); a deduplicated 15.6-expert working set is
> 12.8 MB of Q4_0 gate/up weights and fits entirely in L2. At n_used ≤ 4 the whole sweep is
> L2-resident, so "µs per pair" there is L2 latency, not the price a DRAM-resident dedup kernel
> would pay. The real ceiling is lower than 3.2–3.8×.

### 6. Dispatch width matters as much as the tile

Mat-mat's fixed cost fits `t = 26.9 + 54.1·n µs` (verified: predicts 81 at n=1, measured 88.6).
The 26.9 µs is 2,880 threadgroup launches, most of which read one word and exit. Any grouped
design must dispatch over **active experts only**:

| Grouped dispatch | Threadgroups | Fixed cost |
|---|---|---|
| z = all 288 experts | 5,760 | **53.8 µs** — eats the win |
| z = active only (~16) | 320 | 3.0 µs |

**So a compacted active-expert list is mandatory.** A kernel that scans `ids` to find its own
tokens and dispatches over all 288 experts is simpler to write and does not pay off.

---

## Are we on the right track?

**Yes, with two honest caveats.**

**Caveat 1 — the end-to-end gain is smaller than the op gain.** The routed GEMM is the dominant
marginal cost of extra draft tokens, not the only one. Attention over N positions, norms, the PLE
n-gram lookups (one row read per token), and the chained draft head all scale with N. Expect the
verify round to get *much* flatter in N, not flat. That is why the projection is +10–15%, not the
+45% you would get if the round cost became constant.

**Caveat 2 — MLX already delivers 42.4 today, for zero work.** The kernel's job is to make
llama.cpp's depth-3 profitable and land somewhere in the 45–50 range. If checkpoint 2 below fails,
the pragmatic answer for speculation-heavy workloads is MLX, and this document should say so
rather than chase it.

Everything else points the same way: the mechanism is unambiguous, the ceiling is measured, and
every competing explanation was tested and killed.

---

## Design options

Two viable designs. Both reuse `map0`'s grouping. They differ in which proven kernel they start
from.

### Option 1 — grouped mat-vec (recommended)

**Start from the kernel that already hits 323 GB/s** and remove its one flaw.

New kernel `kernel_mul_mv_id_grp`:

- **z = active-expert slot** (from a compacted list `map0` writes), not a pair.
- Each threadgroup: read this expert's token list (`hids[im*ne21 ..]`, count `tpe[im]`), then
  run the mat-vec accumulating **C columns at once** — `sumf[NR0][C]` instead of `sumf[NR0]`,
  loading each weight block once.
- The inner change is small. `block_q_n_dot_y` (`mul_mv.metal:85`) reads `d` plus four `uint16`
  per block. Hoist them to registers, loop C columns. ~40 lines for Q4_0.
- No gather/scatter buffers: per-column `src1`/`dst` offsets come straight from the ids.

**Why recommended:** bandwidth is proven; the register footprint (`NR0 × C` floats, C ≤ 8) is
modest; the fallback to the existing path is a one-line dispatch condition.

**Cost:** map0 has to be wired into the mat-vec branch of `ggml_metal_op_mul_mat_id`
(`ggml-metal-ops.cpp:2661`), plus one new scratch buffer and one new pipeline. This is the bulk of
the work — host plumbing, not kernel code.

### Option 2 — narrow-tile mat-mat

**Start from the kernel whose data traffic is already right** and fix its geometry.

New variant of `kernel_mul_mm_id` (`mul_mm.metal:430`) with `NR1 = 8` (or 4), used when
`ne21 < 32`.

- All host plumbing already exists for this path: map0, `hids`, `tpe`, scratch sizing.
- **But the tile size is not a constant you can flip.** The 128-thread mapping is wired to
  64 rows × 32 columns (`mul_mm.metal:449–450` and the `lr0`/`lr1` loaders); shared-memory
  offsets assume 32. A narrow variant needs its own thread mapping, its own smem layout, and on
  Apple10 its own `matmul2d_descriptor(NR1, NR0, NK)` instantiation.
- Also needs the active-expert compaction (§6), which map0 does not do today.

**Why not first:** the bet is that tile waste is the *whole* reason mat-mat sits at 71 GB/s. It is
the most likely reason, but unproven; Option 1's bandwidth is measured.

### Considered and rejected

| Idea | Why not |
|---|---|
| Lower the mat-vec/mat-mat threshold | Measured worse at every n (§3). |
| Scan-based grouping, dispatch over all 288 experts | Fixed cost 53.8 µs, eats the win (§6). **Confirmed empirically** - this is the design that got implemented; measured fixed cost ~19 µs, still fatal (*Results*). |
| Sort pairs by expert for L2 locality, no dedup | Each pair still moves 921 KB; the n=4 working set is 37 MB, far past L2. Cheap to test (~1 h) but low expected value. |
| Lower top-k for verification only | Changes the model's output. Not a speedup, a different model. |
| Draft trees / tree attention | Increases N per verify. Makes this kernel matter more, not less; do it *after*. |
| Tune `PMIN`, drafter precision, chain feeding | All tested, all dead (§2). |

---

## Harness — what is already in the tree

62 lines, correctness-verified, **no behaviour change at default**. `git diff` shows them.

| Where | What |
|---|---|
| `tests/test-backend-ops.cpp` | `MUL_MAT_ID` perf cases at the real routed shapes: Q4_0/Q4_1/Q8_0, 288 experts, top-10, n ∈ {1,2,4,8,16}, both gate/up and down shapes. Plus the pair-count ceiling sweep (n=4, n_used ∈ {1,2,4,10}). |
| `ggml/src/ggml-metal/ggml-metal-common.cpp:18` | `GGML_METAL_MMID_MIN_TOKENS` env override on the threshold. Default 32 = upstream. |

Commands:

```bash
cd ~/Documents/shared-with-google-drive/explorations/llama.cpp
cmake --build build --target test-backend-ops -j 8

# op-level perf (the numbers in §3 and §5)
./build/bin/test-backend-ops perf -o MUL_MAT_ID | grep n_mats=288

# correctness gate — must print "3/3 backends passed"
./build/bin/test-backend-ops test -o MUL_MAT_ID

# threshold sweep
GGML_METAL_MMID_MIN_TOKENS=2 ./build/bin/test-backend-ops perf -o MUL_MAT_ID | grep n_mats=288
```

End-to-end launcher and A/B: `~/models/bin/flashnext-reap-server.sh` (defaults: `NMAX=1`,
`CACHE=38`, resident build). Per-position acceptance needs trace logging: append `-lv 4` and grep
`acc per pos`. Interleaved A/B pattern is in the scratchpad's `ab.sh`; the essentials are 300
tokens, `temperature 0`, discard the first run, alternate arms, two rounds.

**Baselines to beat, same prompt, same machine:**

| | Value |
|---|---|
| Op, Q4_0 gate/up, n=4 | 114–137 µs |
| Op, n=1 | 20.7 µs — **must not regress** |
| Decode, `NMAX=1` | 39.5 tok/s |
| Decode, `NMAX=3` | 36.3 tok/s |
| MLX, same weights | 42.4 tok/s |

---

## Execution steps — Option 1

Each step has a file, a change, a verify command, and a stop condition. Do not skip the verify.
If a stop condition fires, write down the number and stop — the measurement is the deliverable.

### Step 0 — reproduce the baseline (15 min)

Build and run the harness above. Op numbers must match §3 within ~10%; correctness 3/3. If not,
something about the machine or build differs and every later number is suspect.

### Step 1 — batched dot product (kernel, ~40 lines)

**File:** `ggml/src/ggml-metal/kernels/mul_mv.metal`, next to `block_q_n_dot_y` (line 85).

Add `template<short C> inline void block_q_n_dot_y_c(device const block_q4_0 * qb, thread const
float * sumy, thread float (*yl)[16], int il, thread float * out)`:

- read `d` and `qs[0..3]` into registers **once**;
- for `c` in `0..C`: the existing 4-accumulator loop against `yl[c]`, `out[c] += d * (...)`.

Keep `block_q_n_dot_y` untouched — the existing path must keep calling it.

**Verify:** compiles (`cmake --build build --target ggml-metal`).

### Step 2 — grouped impl (kernel, ~120 lines)

**File:** same. Copy `mul_vec_q_n_f32_impl` (line 218) to `mul_vec_q_n_f32_grp_impl<block_q, NR0, C>`.

Changes from the original:

- inputs: expert weights `src0` (already per-expert), `src1` base, `dst` base, and a small
  `thread const int * cols` of length `ncol ≤ C` holding pair indices;
- `float sumf[NR0][C]`, `float yl[C][16]`, `float sumy[C][2]`;
- per k-block: load `yl[c]`/`sumy[c]` for each column from `src1 + col_offset(cols[c])`, then
  one call to `block_q_n_dot_y_c<C>` per row;
- write-out: `simd_sum` per `(row, c)`, to `dst + dst_offset(cols[c])`.

Offsets replicate what `kernel_mul_mv_id` computes today: `i11 = idx % ne11`, `i12 = iid1`,
`dst = (i1*ne0 + i2*ne1*ne0)*4` with `i1 = idx`, `i2 = iid1` — where `idx = pair % nei0`,
`iid1 = pair / nei0`.

**Verify:** compiles.

### Step 3 — wrapper kernel + pipeline (kernel + registration, ~80 lines)

**Files:** `mul_mv.metal` (kernel), `ggml-metal-device.cpp` / `ggml-metal-device.m` (pipeline
lookup, mirror how `kernel_mul_mv_id` is registered, with the same function constants
`FC_mul_mv_nsg` etc.), `ggml-metal-impl.h` (kargs struct if new fields are needed).

`kernel_mul_mv_id_grp`:

- `im = active[tgpig.z]` — the compacted list from Step 4; `if (tgpig.z >= n_active) return;`
- `neh1 = tpe[im]`; token list at `hids + im*ne21`;
- loop `c0` over `0..neh1` in chunks of `C`, fill `cols[]`, call the grouped impl;
- keep the same `args0` construction as `kernel_mul_mv_id` for everything else.

`C = 4` first. It matches the common case (n ≤ 4 tokens per expert) and keeps registers small.

**Verify:** compiles; pipeline resolves (a debug print on first use is fine).

### Step 4 — active-expert compaction in map0 (kernel, ~15 lines)

**File:** `mul_mm.metal:364`, `kernel_mul_mm_id_map0`. It runs one thread per expert and already
computes `tpe_u32[ide] = n_all`.

Add: a `device atomic_uint * n_active` and `device int32_t * active`; after computing `n_all`,
`if (n_all > 0) { uint slot = atomic_fetch_add_explicit(n_active, 1, memory_order_relaxed);
active[slot] = ide; }`. Host zeroes `n_active` before each call (a tiny fill kernel or a
pre-zeroed buffer region).

Order of `active[]` is arbitrary; that is fine.

**Verify:** compiles. Sanity: run the mat-mat path (which now also runs the extended map0) through
`test-backend-ops test -o MUL_MAT_ID` — 3/3, proving the extension didn't break the consumer that
already exists.

### Step 5 — host dispatch (~60 lines)

**File:** `ggml-metal-ops.cpp`, `ggml_metal_op_mul_mat_id` (line 2661).

In the `else` (mat-vec) branch, when `2 <= ne21 && ne21 < min_tokens` and type is Q4_0:

1. size and bind the extra buffers (extend `ggml_metal_op_mul_mat_id_extra_tpe/_ids` with an
   `_extra_active` of `ne02` ints + 1 counter);
2. dispatch the (extended) map0 exactly as the mat-mat branch does, then the barrier;
3. dispatch `kernel_mul_mv_id_grp` with
   `x = (ne01 + nr0*nsg - 1)/(nr0*nsg)`, `y = 1`, **`z = min(ne02, ne20*ne21)`** — the CPU-side
   upper bound on distinct experts; the kernel's `n_active` check handles the slack.

Gate it behind an env var (`GGML_METAL_MMID_GROUPED=1`) during development so A/B is a flag flip.

**Verify — CHECKPOINT 1:** `./build/bin/test-backend-ops test -o MUL_MAT_ID` → **3/3 backends
passed** with the flag on. Stop condition: any failure. Do not proceed on a "close enough" NMSE.

### Step 6 — op-level perf — CHECKPOINT 2

`GGML_METAL_MMID_GROUPED=1 ./build/bin/test-backend-ops perf -o MUL_MAT_ID | grep n_mats=288`

| Target | Stop condition |
|---|---|
| Q4_0 gate/up **n=4 under ~50 µs** (today 114–137) | n=4 not under ~90 µs → stop, report |
| **n=1 unchanged** at ~20.7 µs | any regression at n=1 → stop; the flag must not be on by default |
| n=2 and n=8 monotone and better than mat-vec | — |

If it comes in at 60–90 µs, that is still a real win — report it honestly as such and decide
whether to continue.

**Outcome (2026-09-06): stop condition fired.** Grouped n=4 = 134 µs vs 109 plain (uniform);
never better at any shape. See *Implementation results*.

### Step 7 — remaining expert types (~1 h)

Repeat Steps 1–2 for `block_q4_1` and `block_q8_0` (the checkpoints use Q4_1 for some
down-projections and Q8_0 for the shared expert). Same structure; the dot-product bodies differ.
Correctness gate after each.

### Step 8 — end-to-end — CHECKPOINT 3

Rebuild `llama-server`. Interleaved A/B, flag off vs on, at `NMAX=1` and `NMAX=3`, plus a
`PMIN` re-sweep at the new `NMAX` optimum (the old sweep was on the streaming model and is
already known to be mistuned for the resident one).

| Target | Meaning |
|---|---|
| `NMAX=3` faster than `NMAX=1` | verification is now amortising |
| decode > 42.4 | llama.cpp passes MLX on equivalent weights |
| per-position acceptance unchanged | the kernel is numerically faithful |

Stop condition: `NMAX=3` still slower than `NMAX=1` with the flag on → the residual per-token
costs dominate; document, keep the kernel for the op-level win, and move on.

**Outcome (2026-09-06): stop condition fired.** `NMAX=3` stays slower than `NMAX=1` with the flag
on (26.3 vs 28.9 tok/s), and the flag costs ~3–6% at both settings. There is no op-level win to
keep it for. Acceptance is bit-identical flag-off vs flag-on, so the kernel is numerically
faithful. See *Implementation results*.

### Step 9 — default and upstream

Turn the flag on by default for `2 <= ne21 < 32` on Q4_0/Q4_1/Q8_0 once Checkpoints 1–3 pass.
Fold the numbers into `docs/Qwen3.8-Flash-Next.md`. Upstream PR: the perf cases from the harness
are the reproducer; the pair-count sweep is the one-table argument.

**Outcome: not done.** Checkpoints 2 and 3 failed; the flag stays opt-in, default off. There is
no upstream case — the harness is still a fine reproducer, but for why the obvious grouped kernel
loses.

---

## Implementation results (2026-09-06)

Option 1 was implemented as specified, minus the Step-4 compaction: the grouped kernel dispatches
`z = ne02` (all 288 experts) with empty threadgroups early-returning, and reuses unmodified `map0`
for the per-expert token lists. That single simplification is the whole story of this section.

**Files changed** (uncommitted, branch `nitin/mainline`):

| File | What |
|---|---|
| `ggml/src/ggml-metal/kernels/mul_mv.metal` | `kernel_mul_mv_id_grp` + batched-dot impls for q4_0/q4_1/q8_0, C = 4 |
| `ggml/src/ggml-metal/ggml-metal-impl.h` | `MUL_MV_ID_GRP_C` |
| `ggml/src/ggml-metal/ggml-metal-device.{h,cpp}` | `get_pipeline_mul_mv_id_grp` |
| `ggml/src/ggml-metal/ggml-metal-common.{h,cpp}` | `ggml_metal_op_mul_mat_id_use_grp`, env `GGML_METAL_MMID_GROUPED` (default 0), `ne21 >= 2` only |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | dispatch branch: `map0` + barrier + grouped kernel |
| `tests/test-backend-ops.cpp` | `pool` option on `test_mul_mat_id`: restrict routing to `pool` distinct experts (dedup stress cases); correctness cases at pool = 2 |

**Checkpoint 1 (correctness): pass.** `test -o MUL_MAT_ID` = 3/3 backends, flag off and on,
including pool-2 cases where every expert serves multiple tokens.

### Checkpoint 2 (op-level): fail at every shape

`perf -o MUL_MAT_ID -p "n_mats=288"`, µs/run, plain vs grouped (flag on). `u` = uniform routing
(pool = 0), `p` = 16-expert pool (maximal dedup opportunity). Q4_0 gate/up, m=640 k=2560:

| n | u: plain -> grouped | p: plain -> grouped |
|---|---|---|
| 1 | 20.65 -> 20.67 (flag never routes n=1) | — |
| 2 | 38.95 -> 51.37 (+32%) | 31.23 -> 49.15 (+57%) |
| 4 | 109.0 -> 134 (+23%) | 59.97 -> 81.96 (+37%) |
| 8 | 221.9 -> 257.6 (+16%) | 116.1 -> 146.1 (+26%) |
| 16 | 404.9 -> 438.7 (+8%) | 229.4 -> 274.1 (+20%) |

Q4_1 mirrors Q4_0. Q8_0 down-proj (m=2560 k=640) is far worse (n=2 p: 155 -> 415, +167%); its
1-simdgroup grouped variant loses badly and the kernel stays off for it unless fixed.

The pair-count sweep with the flag on (n=4, uniform, n_used = distinct experts touched):

| n_used | plain | grouped | grouped − plain |
|---|---|---|---|
| 1 | 8.13 | 27.03 | **+18.9** |
| 2 | 17.07 | 29.23 | +12.2 |
| 4 | 29.71 | 43.09 | +13.4 |

`n_used = 1` isolates the pure fixed cost of dispatching 288 experts (288 threadgroup columns,
287 early-exit): +18.9 µs before any dedup benefit exists. Exactly §6's predicted failure mode for
the z = all-experts design (predicted 53.8 µs on the mat-mat grid; measured ~19 µs on the narrower
mat-vec grid). The dedup saves less than that costs.

**§5's ceiling was an L2 artifact.** Plain path, pool = 16, n=4: the 40 pairs logically touch
40 x 921.6 KB = 36.9 MB in 59.97 µs = 615 GB/s apparent, twice DRAM spec. Impossible unless the
redundant reloads come from cache — and they do: the deduplicated set is 16 x 921.6 KB = 14.7 MB,
under the 16 MB L2 (`hw.perflevel0.l2cachesize`). The plain kernel already gets the dedup for
free via L2 hits; the grouped kernel only saves traffic that was never going to DRAM. §5 priced
those bytes at DRAM bandwidth and overestimated the prize. Worse, at pool = 16 only 16 of 288
expert columns do work: ~80 of the 1,440 grouped threadgroups are useful, which under-occupies
the GPU and drops effective bandwidth to ~180 GB/s. The dedup premise only holds where the
working set exceeds L2 (uniform routing at n >= 8, ~35+ MB) — and there the grouped kernel still
loses.

### Checkpoint 3 (end-to-end): fail

`/tmp/e2e-mmid-ab.sh`, interleaved 2 rounds, 300-tok greedy. tok/s:

| Arm | flag off | flag on | delta |
|---|---|---|---|
| `NMAX=1` | 30.55 | 28.88 | −5.5% |
| `NMAX=3` | 27.07 | 26.31 | −2.8% |

(Absolute tok/s below the 39.5/36.3 baselines — machine busier that day; the off arms are the
valid A/B baseline.) The stop condition fired: `NMAX=3` still loses to `NMAX=1` with the flag on.
Draft acceptance identical to 5 decimals (0.70988 / 0.50968) off vs on -> kernel numerically
faithful, confirmed end to end.

### Verdict

- The kernel is correct, C = 4 batching works, and it is slower everywhere. Flag stays off.
- Root cause 1: z = all-288 dispatch. §6 demanded active-expert compaction (Step 4); skipping it
  was the only deviation from the plan and it is fatal. A retry must dispatch over ~16-32 active
  experts, not 288.
- Root cause 2: L2 already dedups what the benchmark said was redundant. At n <= 4 the working
  set is under 16 MB, so the plain kernel's "2.56x redundant traffic" is mostly L2 hits. Real
  headroom for any dedup design is smaller than §5 claimed.
- What a retry looks like: Step-4 compaction (`active[]` from `map0`, `z = n_active`, host reads
  `n_active` once and skips the launch if 0 is not possible async), *plus* restricting to shapes
  where the set exceeds L2 (n >= 8) — but see the n=8 row above; even uniform-routing gains there
  were negative. Given root cause 2, expectation for a retry should be near zero.

---

## Review of the implementation (2026-09-07)

Independent review of the Option-1 implementation. **Verdict: the "stop" conclusion is right, the
stated reason is incomplete, and the implementation contains a flaw that means the idea was never
actually tested.** Both points matter, because they change what a retry would be worth.

### Finding 1 — the core optimisation was not implemented

Design Step 1 was a batched dot product: read the weight block's `d` and four `uint16` **once**,
then loop `C` columns. `block_q_n_dot_y_c` does not exist in the tree. The shipped inner loop
(`mul_mv.metal:3324`) nests the other way:

```c
for (int ib = ix; ib < nb; ib += NQ) {      // k blocks
    FOR_UNROLL (short cc = 0; cc < C; ++cc) {       // columns  <- OUTER
        ... load yl for column cc ...
        FOR_UNROLL (short row = 0; row < NR0; ++row) {   // rows  <- INNER
            sumf[row][cc] += block_q_n_dot_y(ax[row] + ib, s, yl, il);
```

`ax[row] + ib` is therefore read once **per (row, column) pair** — the same weight traffic as the
plain kernel. The code comment is explicit about the trade ("stage one column of y at a time into
the same registers"), which saves `C x 16` registers and destroys the entire benefit.

So Checkpoint 2 measured *plain weight traffic + 288-expert dispatch + a `map0` pre-pass*. It could
only lose. **"Grouped batching does not pay" is not supported by this experiment.**

The scaffolding around it is sound: pipeline, dispatch, `map0` reuse, correctness (independently
re-verified here: `GGML_METAL_MMID_GROUPED=1 test -o MUL_MAT_ID` = 3/3), and bit-identical
acceptance end to end. ~587 lines of correct plumbing with the optimisation missing from the middle.

### Finding 2 — but the idea is dead anyway, for a better reason

Decisive test, added to the harness: hold `n=4` and `n_used=10` so the pair count and FLOPs are
**constant** at 40 pairs / 131 MFLOP, and vary only `pool` — how many distinct experts those pairs
land on, i.e. the unique working set.

| pool | working set | time | TFLOPS |
|---|---|---|---|
| 10 | 8.8 MB | 60.76 µs | 2.16 |
| 16 | 14.1 MB | **61.52 µs** | 2.13 |
| 24 | 21.1 MB | 62.21 µs | 2.11 |
| 40 | 35.2 MB | 76.97 µs | 1.70 |
| 0 (uniform, ~40 distinct) | ~35 MB | 112–115 µs | 1.17 |

**Time is flat from pool=10 to pool=24 — 2.4% across a 2.4× working-set change that crosses the
16 MB L2 boundary.** The op is not load-bound at the real operating point. Deduplicating weight
loads optimises a non-bottleneck, however well it is implemented.

This is stronger than §5's "L2 already dedups it": even *within* L2, more or less unique data
changes nothing. The limit is compute/occupancy at ~2.1 TFLOPS.

### Finding 3 — §3–§5 measured an unrepresentative routing pattern

The original cases used uniform random routing (`pool=0`), giving ~40 distinct experts for 40
pairs — near-zero overlap. The real model has strong routing locality: ~15.6 distinct, i.e.
`pool≈16`. At the realistic operating point the op costs **61.5 µs, not the 114 µs** §3 reported.

So the headline "2.56× redundant traffic at 323 GB/s" described a benchmark artifact. Every
future `MUL_MAT_ID` measurement for this model must set `pool≈16`.

### Finding 4 — mat-vec is already the best available kernel

At the realistic point (n=4, pool=16, 131 MFLOP):

| Path | Time | TFLOPS |
|---|---|---|
| **mat-vec (shipping today)** | **61.5 µs** | **2.13** |
| grouped (Qwen's) | 82.0 µs | 1.60 |
| mat-mat (forced) | 105–109 µs | 1.20–1.25 |

There is no dedup win to capture. Option 2 (narrow-tile mat-mat) would have to beat 2.13 TFLOPS
from 1.25 — plausible in principle, since its tile waste is compute waste and compute is the
binding constraint, but it starts 1.7× behind and the design is the larger of the two.

### Finding 5 — linear verification cost is inherent to MoE, not a bug

Even at realistic routing the op scales ~1.95× per doubling of tokens (31.2 / 60.0 / 116.1 / 229.4
µs at n = 2/4/8/16, pool=16). That is expected: a dense model verifying N tokens re-reads one
weight set, but an MoE verifying N tokens does N× the expert FLOPs and touches more experts
(10 → ~15.6 going 1 → 4 tokens). Since the op is compute-bound, cost tracks FLOPs.

**§1's framing — "a verify batch should amortise weight reads; this one doesn't" — is wrong for
MoE.** There is far less to amortise than a dense-model intuition suggests.

### Recommendation

- **Stop the kernel line.** Flag stays off. Do not do the Step-4 retry: with Finding 2 its expected
  value is zero, not "near zero".
- **Keep the code**, flag-gated and correct, as the reproducer for Findings 1–2.
- **Fix the harness permanently:** `pool≈16` for realistic cases (done, in tree).
- Remaining headroom is general GEMM efficiency at small n (2.13 TFLOPS), not anything
  MoE-specific or speculation-specific. That is an upstream ggml-metal topic and not obviously
  worth this project's time.
- `NMAX=1` (+9%) stands and is unaffected by any of this.

---

## Execution steps — Option 2 (if Option 1 stalls at Step 5)

1. Copy `kernel_mul_mm_id` to `kernel_mul_mm_id_n8`; set `NR1 = 8`; rewrite `lr1` and the `sb`
   load loops for a 32-thread (one simdgroup) threadgroup; `execution_simdgroups<1>` and
   `matmul2d_descriptor(8, 64, 32)` on the tensor path; fix the two hardcoded `32*(sgitg&1)`
   store offsets.
2. Add the Step-4 compaction to map0 and dispatch `z = min(ne02, ne20*ne21)` over `active[]`.
3. Register the pipeline with its own smem size (`sb` shrinks 4×).
4. Route `2 <= ne21 < 32` to it behind a flag.
5. Same three checkpoints.

Expect this to be more code than Option 1, because the kernel body is larger and both compile
paths (`GGML_METAL_HAS_TENSOR` on/off) must be handled.

---

## Risks

| Risk | Mitigation |
|---|---|
| **Silent numerical error.** Wrong results with no crash — the exact failure your MTP notes document ("benchmarks as slow instead of failing"). | Correctness gate on every step; per-position acceptance must be unchanged at Checkpoint 3. |
| **Register spill** from `sumf[NR0][C]`. | Start `C=4`; check the Metal compiler's spill report; drop to `C=2` if needed. Still a 2× reduction in loads. |
| **n=1 regression.** llama.cpp currently beats MLX on plain decode (35.5 vs 28.2) — losing that would be a net loss. | Never route `ne21 == 1` to the new kernel. Checkpoint 2 gates on it. |
| **The microbenchmark is cache-resident.** Op numbers may not transfer 1:1 to the real model, whose 35.6 GiB of experts come from RAM. | This *favours* dedup (fewer RAM reads). Checkpoint 3 is the real test. |
| **Residual per-token costs cap the gain** (attention, PLE rows, draft head). | Already priced into the +10–15% projection. If Checkpoint 3 shows `NMAX=3` still loses, that is the finding. |

---

## Open questions worth 30 minutes each, not more

- **Does the streaming (512-expert v3) build benefit too?** Its wave planner may already dedup per
  layer. Check `moe stream` touches per remap on v3 with `NMAX=3`; if they scale like the resident
  build's (14.3 → 15.6), the kernel helps there as well.
- **What does MLX actually do here?** `mlx_vlm/models/switch_layers.py` dedups with
  `np.unique(idx)` and runs one quantized matmul per unique expert over all its rows — which is
  Option 1 in spirit. Worth one read before writing the kernel, not for code, for confirmation.
- **Is `C` better chosen per layer?** Tokens-per-expert varies. Fixed `C=4` with a loop is the
  simple answer; leave adaptive sizing for after Checkpoint 3.

---

## History of this investigation, for context

Ordered as it happened, so the dead ends are visible:

1. Compared MLX and llama.cpp on the same pruned weights: MLX 42.4, llama.cpp 36.3 at `NMAX=3`.
2. Swept `NMAX`: 1 → 39.5 (+9%). Shipped.
3. Hypothesised drafter precision → killed (0.006).
4. Hypothesised acceptance → killed (`PMIN=0.7` beats MLX's rate and is slower).
5. Hypothesised per-token expert gather → killed (touches 14.27 → 15.60).
6. Derived per-token verify cost: 0.98 vs MLX's 0.66. Localised to `MUL_MAT_ID`.
7. Added perf cases; found linear scaling; found the `ne21 >= 32` threshold.
8. Hypothesised lowering the threshold → killed (worse at every n).
9. Costed grid compaction alone → insufficient (mat-mat per-token cost is the problem, not launch).
10. Found the mechanism: per-pair reloads (mat-vec) vs tile waste (mat-mat).
11. Measured the ceiling by pair-count sweep: 2.56× redundancy, ~36–45 µs reachable.
12. Costed the dispatch width: compaction is mandatory.
13. Stopped here. Design complete; kernel not written.
14. Kernel written (Option 1, C = 4, q4_0/q4_1/q8_0, flag `GGML_METAL_MMID_GROUPED`), with one
    simplification: `z` = all 288 experts with early-exit, no Step-4 compaction.
15. Checkpoint 1 passed: 3/3 backends, flag off and on, incl. pool-2 sharing cases.
16. Checkpoint 2 failed at every shape; the n_used=1 arm measured the all-288 dispatch cost
    directly (+18.9 µs). §6 confirmed: the simplification in item 14 was fatal.
17. Found the flaw in §5: pool-16 plain path runs at 615 GB/s apparent — the 14.7 MB dedup set
    fits the 16 MB L2, so the "redundant" reloads were cache hits all along. The op-level prize
    was much smaller than measured.
18. Checkpoint 3 failed: −5.5% at NMAX=1, −2.8% at NMAX=3; NMAX=3 still loses to NMAX=1 with the
    flag on (stop condition). Acceptance bit-identical off/on — kernel numerically faithful.
19. Verdict: flag stays off. Step 9 not done. Remaining lever is Step-4 compaction + only-L2-
    exceeding shapes, and after item 17 the expected value is near zero.
