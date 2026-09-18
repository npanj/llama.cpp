# Does a higher-precision PLE table improve quality?

**Status:** running. Bar set 2026-09-18, **before** any measurement.

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

_To be filled in. Nothing measured yet._

| | PPL | vs baseline | t | chunks better | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|
| baseline (Q4_0 PLE) | | | | | | |
| candidate (Q8_0 PLE) | | | | | | |

**Verdict:** _pending_
