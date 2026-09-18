# Running Qwen3.8-Flash-Next V3 on a 64 GB Mac

**The short version.** This model's weights are 95.5 GiB. Your Mac has 64 GB. It still runs, at
roughly 24-27 tokens/sec, because this fork keeps the rarely-used parts of the model on your SSD
and pulls them in only when a token actually needs them.

Stock llama.cpp cannot do this. It will try to load all 95.5 GiB into memory and fail.

---

## Words we can't avoid

| Term | What it means here |
|---|---|
| **MoE** (mixture of experts) | The model is split into many small sub-networks. Each token uses only ~10 of them. The rest sit idle. |
| **expert streaming** | Keeping those idle sub-networks on SSD instead of in RAM, and reading the ~10 you need per token. This is what makes 95.5 GiB fit in 64 GB. |
| **expert cache** | A fixed slice of RAM that holds recently-used experts so you don't re-read them from SSD every token. Bigger = faster, until it starves macOS. |
| **MTP** (multi-token prediction) | A small extra "draft head" that guesses the next few tokens. The big model checks the guesses in one pass. Correct guesses are nearly free speed. |
| **wired memory** | Memory macOS is not allowed to page out to disk. The GPU needs its weights wired. If you wire too much, the whole machine freezes. |
| **prefill** | Reading your prompt, before any answer appears. |
| **decode** | Writing the answer, one token at a time. |

---

## 1. What you need

| | Requirement | Why |
|---|---|---|
| **Machine** | Apple Silicon Mac, **64 GB** unified memory | Tuned and measured here. See §7 for other sizes. |
| **Disk** | ~100 GB free, **internal SSD** | The experts are read from disk on every token. An external USB drive will be far slower. |
| **OS** | macOS with Metal | The fast paths in this fork are Metal-only. |
| **Build tools** | Xcode command line tools, CMake | `xcode-select --install`, `brew install cmake` |

**The disk matters more than you'd think.** Expert streaming reads from SSD continuously while
generating. Internal Apple SSD is what these numbers were measured on.

---

## 2. Build

```bash
git clone https://github.com/npanj/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 --config Release
```

Check it worked:

```bash
./build/bin/llama-server --help | grep moe-stream
```

You should see four `--moe-stream*` flags. If you see nothing, you built stock llama.cpp — check
you cloned this fork and are on the default branch.

---

## 3. Get the model

### 3a. The checkpoint — 95.5 GiB, 3 shards

**<https://huggingface.co/nitinpanj/qwen38-flash-next-v3>**

Download all three shards into one directory. You point llama.cpp at the *first* one and it finds
the rest.

```bash
D=~/models/qwen38-flash-next-v3
mkdir -p $D
for i in 1 2 3; do
  curl -fL --retry 5 -C - -o $D/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-0000$i-of-00003.gguf \
    https://huggingface.co/nitinpanj/qwen38-flash-next-v3/resolve/main/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-0000$i-of-00003.gguf
done
```

<details>
<summary>What "V3" means, if you're curious</summary>

bartowski's Q4_0 with a Q8_0 `output.weight` spliced in, plus unsloth's UD-IQ4_XS spliced into five
tensor groups that stay resident in memory (`attn`, `hc`, `token_embd`, `ssm_out`, `shexp`).

Measured against the unspliced checkpoint, paired 40-chunk perplexity:
**5.2777 → 4.3148, a 17.8% improvement**, better on 40 of 40 chunks, for +1.69 GiB on disk and
−2.7% decode speed. Draft acceptance also rose, 0.751 → 0.817.

The resident groups are not arbitrary. Trimming the list to `attn,hc,token_embd` measures the same
perplexity but drops decode 14% — `ssm_out` and `shexp` are worthless for perplexity and worth
about 11 points of decode.
</details>

### 3b. The MTP draft head — the +50% speed part

This is the small "guesser" model from the table above. **Without it you run at ~18 tokens/sec
instead of ~27.**

**Just download it.** It is in the same repo, 1.9 GiB:

```bash
D=~/models/qwen38-flash-next-mtp
mkdir -p $D
curl -fL --retry 5 -C - -o $D/mtp-shared-Q4_K_M.gguf \
  https://huggingface.co/nitinpanj/qwen38-flash-next-v3/resolve/main/MTP/mtp-shared-Q4_K_M.gguf
```

> **This file only works one way.** It has no token embeddings of its own — it borrows the main
> model's, which is how it stays at 1.9 GiB instead of 2.6. So it must be passed with `-md`
> *alongside* the V3 checkpoint, on this fork. It cannot be loaded on its own, and it will not work
> on stock llama.cpp.

<details>
<summary>Or build your own draft head, if you'd rather not trust a binary</summary>

**You cannot use a HuggingFace MTP sidecar directly.** Every published sidecar uses upstream tensor
naming; this fork expects different names. The file will not load as-is. Converting it takes about
five minutes and Python's standard library:

```bash
D=~/models/qwen38-flash-next-mtp
mkdir -p $D

curl -fL --retry 5 -C - -o $D/mtp-src-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf

python3 scripts/mtp/mtp_sidecar.py \
  --src $D/mtp-src-Q4_K_M.gguf \
  --out $D/mtp-Q4_K_M.gguf
```

Two tensor renames and one split, byte-exact — no requantization. Full details in
[`scripts/mtp/README.md`](../scripts/mtp/README.md).

**Then check draft acceptance is near 0.50 in the server logs.** A bad conversion does not error —
the server starts normally and the draft head silently contributes nothing. If you built your own,
use your output path in place of `mtp-shared-Q4_K_M.gguf` below.
</details>

**Don't want the draft head at all?** Drop the four `--spec-*` flags and `-md` from the command in
§5. Everything still works, at ~18 tokens/sec instead of ~27.

### Where this guide assumes the files live

```
~/models/qwen38-flash-next-v3/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00001-of-00003.gguf
~/models/qwen38-flash-next-mtp/mtp-shared-Q4_K_M.gguf
```

---

## 4. Raise the Metal wired-memory limit

This is the step people skip, and then the model fails to load.

macOS caps how much memory the GPU may wire. The default cap is below what this model needs.

```bash
sudo sysctl iogpu.wired_limit_mb=59392
```

**This resets on reboot.** Run it again after restarting, or add it to a login script.

**What the number means.** 59392 MiB is 58 GiB. It is a *ceiling*, not a reservation — nothing is
consumed until the model loads. Its purpose is failure mode: past the cap, Metal raises an
out-of-memory error the server reports and survives. Without a cap, macOS itself locks up.

Pick the cap to match your expert cache (§5):

| Expert cache | Wired cap |
|---|---|
| 32 GiB | `50176` |
| 33-34 GiB | `55296` |
| 35-36 GiB | `59392` |

**Do not set a cap without setting the matching cache size, or vice versa.** A cap below what the
cache needs fails at load. A cap far above it risks freezing the machine.

---

## 5. Run it

```bash
export LLAMA_MOE_STREAM_LOOKAHEAD=1
export LLAMA_MOE_STREAM_WAVE_CAP=200
export LLAMA_MOE_STREAM_PARTITION=1
export LLAMA_QWEN4EXP_SPARSE_FA=1

./build/bin/llama-server \
  -m ~/models/qwen38-flash-next-v3/Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00001-of-00003.gguf \
  -md ~/models/qwen38-flash-next-mtp/mtp-shared-Q4_K_M.gguf \
  -ngl 99 \
  --moe-stream --moe-stream-cache 36 --moe-stream-io-threads 8 --moe-stream-direct \
  -c 98304 -b 4096 -ub 4096 -cms 512 -np 1 -fa on \
  --cache-reuse 0 --cache-ram 512 \
  --jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 \
  --spec-draft-ngl 99 --spec-max-prompt 0 \
  --host 127.0.0.1 --port 8080
```

Then open <http://127.0.0.1:8080>, or point any OpenAI-compatible client at
`http://127.0.0.1:8080/v1`.

**First load takes a few minutes** — it is reading 95.5 GiB off disk. Later loads are faster while
the file is still in the OS page cache.

### What each unusual flag is doing

| Flag | Why it's there |
|---|---|
| `--moe-stream` | The whole point: experts live on SSD. |
| `--moe-stream-cache 36` | 36 GiB of RAM held for recently-used experts. The single biggest speed knob. |
| `--moe-stream-io-threads 8` | Parallel SSD reads. 8, 12 and 16 measured indistinguishable; 8 is fine. |
| `--moe-stream-direct` | Bypass the OS page cache for expert reads, so it can't evict your model. |
| `-ub 4096` | Prompt-reading chunk size. Dropping to 2048 costs ~12% prefill; 8192 runs out of GPU memory. |
| `--spec-type draft-mtp` | Turn on the draft head. Worth roughly +50% generation speed. |
| `--spec-draft-n-max 3` | Guess 3 tokens ahead. Measured optimum; 4 was slower in every test. |
| `--spec-draft-p-min 0.3` | Skip guessing when the draft head isn't confident. Worth ~+10%. |
| `--spec-max-prompt 0` | **No limit.** See the warning below. |
| `--cache-reuse 0` | Off deliberately — the real caching win comes from exact-prefix reuse, which is already on. |
| `-fa on` | Flash attention. |

> **Don't set `--spec-max-prompt` to a number.** It compares your *whole* prompt against the limit
> with no knowledge of the prompt cache, so in a running chat it switches the draft head off
> permanently once your history crosses it — and never turns it back on. Measured live at a 16384
> limit: generation fell from 25.8 to 15.4 tokens/sec the moment context passed 16k. Only set a
> limit for one-shot long prompts with short answers.

---

## 6. What you should see

Measured on an Apple M5 Pro, 64 GB, with this exact configuration:

| | Speed |
|---|---|
| Prompt reading (4k) | ~367 tokens/sec |
| Generation, draft head **off** | 18.0-18.6 tokens/sec |
| Generation, draft head **on** | ~27.6 tokens/sec |
| Generation at 29k context, real chat traffic | ~20.6 tokens/sec |

Generation slows as context fills — that is expected and is dominated by reading the attention
cache, not by the streamed experts.

---

## 7. If your Mac isn't 64 GB

**Less than 64 GB.** Lower `--moe-stream-cache` and the matching wired cap. The model still loads —
the cache is just RAM for speed, not a requirement. Expect it to be slower. Untested below 64 GB.

**96 GB or 128 GB.** Raise the cache. But note the measured ceiling on *this* machine: at 38 GiB of
cache on a 64 GB Mac, generation **collapsed** from ~24 to ~3.7 tokens/sec as macOS started swapping
the server's own heap. More cache is not monotonically better — raise it in small steps and watch
for swap.

**Not a Mac.** `--moe-stream` is not Metal-specific in principle, but nothing here has been tuned or
measured on CUDA or CPU. You are in unexplored territory.

---

## 8. Troubleshooting

**"Checkpoint not found" / it only loads part of the model.**
Point `-m` at shard `00001-of-00003`. All three shards must be in the same directory.

**It fails at load with a GPU out-of-memory error.**
Your wired cap is too low for your cache size. Match them using the table in §4. If it still fails,
raise the cap in 2048 MiB steps — do not jump straight to the maximum.

**The whole machine freezes or becomes unresponsive.**
Your wired cap is too *high*. macOS has been left too little memory. Drop to `--moe-stream-cache 32`
with cap `50176`, which leaves macOS a ~15 GiB floor. This costs ~14% prompt speed and ~7%
generation speed, and it is the configuration with the most evidence behind it.

**Generation is much slower than the numbers above.**
- Check swap: `sysctl vm.swapusage`. If swap is growing, your cache is too big.
- Check the model is on internal SSD, not an external drive.
- Close other memory-hungry apps. A 64 GB Mac running this model has very little room left.
- If your file sync client (Dropbox, Google Drive) is watching the model directory, pause it.

**The draft head accepts nothing / generation got slower with MTP on.**
First check draft acceptance in the server logs. If it is near **zero**, the conversion in §3b
picked the wrong half of `eh_proj` — rebuild it *without* `--swap-halves`. If acceptance is near
0.50 and speed is still low, make sure you passed the head via `-md` alongside the main model; a
shared-layout head cannot run standalone.

**Generation was fast, then permanently dropped mid-conversation.**
You set `--spec-max-prompt` to a number. Set it to `0`. See §5.

---

## 9. Going deeper

- [`architecture-and-tuning.md`](architecture-and-tuning.md) — how llama.cpp is built, where time
  and memory go, and every knob you can turn. Fork-only features are marked **[fork]**.
- [`deployment-optimization-guide.md`](deployment-optimization-guide.md) — the quantitative
  companion: tuning derived from bytes moved and FLOPs executed.
- [`Qwen3.8-Flash-Next.md`](Qwen3.8-Flash-Next.md) — notes on this model architecture specifically.
- [`moe-spec-verify-kernel.md`](moe-spec-verify-kernel.md) — a research log on a fused Metal verify
  kernel. Conclusion: closed, don't retry. Kept so nobody repeats it.
- [`scripts/mtp/README.md`](../scripts/mtp/README.md) — exactly what the draft-head conversion does,
  and how to verify you got it right.

---

## 10. Credit

This fork builds on [mihailescu2m/llama.cpp](https://github.com/mihailescu2m/llama.cpp), which is
where expert streaming (`--moe-stream`), the phase-aware ubatch work, the SSD context cache and MTP
rejection sampling come from. That work is the reason any of this runs at all.

Upstream is [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

This branch additionally carries six pull requests that were still open upstream when it was cut —
see the fork notes in the [README](../README.md).
