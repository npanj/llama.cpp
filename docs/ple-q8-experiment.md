# Does a higher-precision PLE table improve quality?

**Status:** CLOSED — bar not cleared. Bar set 2026-09-18 **before** any measurement; result below.

**The question.** The PLE table is 26.82 GiB, 28% of the checkpoint, and sits at `Q4_0` while every
other dense component is `Q8_0`. It is the largest remaining precision gap in the file. Does giving
it more bits improve quality enough to pay for itself?

---

## What is already known

A PLE swap was tested once before and **failed** (see
[`Qwen3.8-Flash-Next.md`](Qwen3.8-Flash-Next.md), "Bisection: the PLE table contributes nothing").
That test swapped `Q4_0` for `IQ4_NL` at **identical byte size** - a fitted codebook against a
symmetric one, so strictly better fidelity for free:

| variant | PPL | vs baseline | t | chunks better |
|---|---:|---:|---:|---:|
| baseline | 5.2777 | - | - | - |
| PLE swap only | 5.1708 | -2.03% | 1.98 | 26/40 |
| everything except PLE | 4.3148 | -18.24% | 7.80 | 40/40 |
| both | 4.3290 | -17.98% | 7.89 | 40/40 |

The swap did not clear significance on its own, and adding it to the dense trunk moved the result
the wrong way (inside noise). The recorded conclusion was to drop it.

**What that settles:** a better 4-bit codebook at the same size buys nothing measurable.

**What it does not settle:** whether more *bits* help. `IQ4_NL` is still 4 bits. This experiment
tests 8.

---

## The variable

Exactly one tensor changes: `per_layer_token_embd.weight`.

| | format | bytes/row | table | checkpoint |
|---|---|---:|---:|---:|
| baseline (v3) | `Q4_0` | 90 | 26.82 GiB | 95.5 GiB |
| candidate | `Q8_0` | 170 | 50.66 GiB | ~119.3 GiB |

Donor: `bartowski/Qwen3.8-Flash-Next-GGUF`, `Q8_0` build, shard 2 of 6. Verified by parsing that
shard's header before downloading: `per_layer_token_embd.weight`, `Q8_0`, 170 bytes/row.

Rows are 160 elements, so only 32-element block formats apply. `Q5_K`/`Q6_K` use 256-element
superblocks and cannot be used here - which is why bartowski's own `Q5_K_M` build falls back to
`Q5_1` for this tensor.

Build command:

```bash
gguf_splice_groups.py --groups ple   # base = v3, donor = bartowski Q8_0 shard 2
```

---

## ⚖️ The bar, set before measuring

**Quality.** Paired per-chunk NLL over the same 40 chunks of `wiki.test.raw`, same chunk order,
via `ppl_pair.py`. To adopt, Q8_0 PLE must clear **both**:

- perplexity better than baseline by **more than 2%**, and
- **t > 3** over 39 d.o.f.

Rationale for 2%: the previous PLE swap measured −2.03% at t=1.98 and was judged not to clear. A
result at or below that is the same non-result with more bits, not a new finding.

**Speed.** Interleaved A/B, real ~24k prompt, two passes with order reversed, MTP flags held at
n-max 3 / p-min 0.3, only `-m` differs.

Expected cost, stated in advance: rows go 90 → 170 bytes, so every PLE gather reads **89% more**.
This lands on the exact path PR #29030 fixed. A prefill regression here is predicted, not a surprise.

**Adoption rule.** Even if quality clears the bar, adoption also requires prefill regression
**under 10%**. +23.8 GiB on disk for a sub-2% quality gain and a large prefill loss is not a trade
worth shipping.

**If the bar is not cleared:** record it and close the line permanently. Two failed PLE experiments
at different precisions is sufficient evidence that the dense trunk, not the PLE table, carries
quality in this model.

---

## Results

40 chunks of `wiki.test.raw`, `-c 4096 -b 4096 -ub 4096`, expert cache 20 GiB, identical flags,
only `-m` differing. Cache lowered from 36 to 20 because the 36 run was killed under memory
pressure; perplexity does not depend on the expert cache, and both arms used the same value, so
the paired comparison is unaffected.

| | PPL | vs baseline | t | chunks better |
|---|---:|---:|---:|---:|
| baseline (`Q4_0` PLE, 90 B/row) | 4.3308 | - | - | - |
| candidate (`Q8_0` PLE, 170 B/row) | 4.3118 | **-0.44%** | **0.99** | **25/40** |

Bar was >2% at t>3. **Not cleared**, and not close.

Sanity check: this baseline (4.3308) sits next to the historical v3 figure (4.3148) measured on a
different day at a different cache size, so the harness agrees with itself.

Speed A/B was not run. Adoption required clearing quality *and* speed; quality failed decisively,
so the speed number cannot change the outcome. The predicted cost stands unmeasured: rows go
90 -> 170 bytes, an 89% larger read per PLE gather.

**Verdict: rejected. Do not adopt. Line closed.**

## Why this is stronger than one null

Two independent PLE experiments at two different precisions now both measure nothing:

| experiment | change | vs baseline | t | chunks better |
|---|---|---:|---:|---:|
| earlier | `Q4_0` -> `IQ4_NL`, same bytes, better codebook | -2.03% | 1.98 | 26/40 |
| this one | `Q4_0` -> `Q8_0`, +23.84 GiB, double the bits | -0.44% | 0.99 | 25/40 |

**Doubling the bits did less than changing the codebook did.** If PLE precision mattered, that
ordering would be hard to explain. Combined with the dense-trunk bisection, which attributed the
entire -18.24% to a 3.55 GiB trunk, the conclusion is that **the PLE table does not carry quality
in this model** - despite being 28% of the checkpoint.

**Do not retry this line.** If a future checkpoint changes the PLE's role in the architecture, that
is a new question; precision alone is answered.
