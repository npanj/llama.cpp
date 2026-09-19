# Rebuilding the Qwen3.8-Flash-Next MTP draft head

The MTP draft head is a small model that guesses the next few tokens so the big model can check
several at once. It is worth roughly +50% generation speed on Qwen3.8-Flash-Next.

Why you need these scripts: every MTP sidecar published on HuggingFace uses upstream tensor naming,
and this fork expects different names. The published file will not load here as-is.
`mtp_sidecar.py` converts it.

If you downloaded a ready-made head from the model repo, you do not need any of this. This is for
rebuilding it yourself, or for auditing what was done.

---

## What the conversion actually does

Two renames and one split. Nothing is requantized; tensor data is copied byte for byte.

| upstream / unsloth | this fork |
|---|---|
| `blk.N.nextn.eh_proj` `[2*n_embd, n_embd]` | `nextn.fc_embd` + `nextn.fc_hidden`, `[n_embd, n_embd]` each |
| `blk.N.nextn.hc_head_norm` / `hc_head_down` / `hc_head_up` | `blk.N.nextn.hc_norm` / `hc_down` / `hc_up` |

The split lands on a Q4_K super-block boundary (a 5120-element row is 20 blocks of 256, cut at
2560), which is why it can be exact. Rows interleave in the file, so each row is cut separately. A
single contiguous slice would be silently wrong.

### This naming is fork-specific, and upstream is going the other way

The names in the right-hand column exist only in this fork. Upstream PR
[#28243](https://github.com/ggml-org/llama.cpp/pull/28243) adds shared-MTP support for qwen4exp and
keeps the left-hand column instead: `eh_proj` stays whole at `[2*n_embd, n_embd]`, and the mixer
tensors stay `hc_head_norm` / `hc_head_down` / `hc_head_up`.

So if that PR merges:

- unsloth's published sidecar will load on upstream llama.cpp with no conversion at all
- a head converted by this script will **not** load on upstream, only here
- this fork still will not load unsloth's original, which is why this script exists

Nothing breaks today, and the conversion is still required for this tree. But it is a detour that
upstream is on course to remove, so do not treat these names as standard.

---

## Rebuild it

```bash
D=~/models/qwen38-flash-next-mtp
mkdir -p $D

# 1. Get the upstream sidecar (~2.6 GiB)
curl -fL --retry 5 -C - -o $D/mtp-src-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf

# 2. Convert it to this fork's naming
python3 scripts/mtp/mtp_sidecar.py \
  --src $D/mtp-src-Q4_K_M.gguf \
  --out $D/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
```

Python 3 standard library only, no `pip install` needed.

---

## ⚠️ Then check that it works

**Never pass `--swap-halves`.** Which half of `eh_proj` is `fc_embd` cannot be read off the file.
The wrong choice does not raise an error. It builds fine, the server starts fine, and drafting
silently collapses. Measured both ways:

| | draft acceptance |
|---|---|
| default | **~0.50** (mean draft length 3.0) |
| `--swap-halves` | **0.00000**, 0 accepted out of 1186 |

So after rebuilding, confirm acceptance is near 0.50 in the server logs. **Do not just check that
the server started.** The failure mode here is a model that runs at normal-looking speed while the
draft head contributes nothing.

---

## The "shared" variant

A shared head additionally omits `output.weight` and `token_embd.weight` and borrows the target
model's copies, which are already resident in memory. That saves 0.82 GiB (2.595 to 1.776 GiB) and
measured byte-identical output.

`mtp_sidecar.py` handles a shared-layout source: it prints a note and carries on.

Two constraints:

- A shared head cannot be loaded on its own. It only works as `-md` alongside the target model.
- A sidecar must never tie `output` to `token_embd`. That collapses acceptance to zero.

---

## Credit

The source sidecar is [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
These scripts only rename and split its tensors. All the model weights are unsloth's work, and the
underlying model is Qwen's.

`gguf_splice.py` is included because `mtp_sidecar.py` imports its GGUF header reader. It was
originally written for a different job (splicing quantization formats between checkpoints) and its
module docstring describes that job, not this one.
