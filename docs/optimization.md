# RTX 3090 performance notes

Measured on the machine this working copy was prepared on.

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GiB, compute capability 8.6 |
| Driver | 615.65.07 (WSL2), CUDA UMD 13.4 |
| OS | WSL2, kernel 6.18.33.2, 20 vCPU, 27 GiB RAM |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Model | `m-a-p/YuE2-3B` (6.77 GiB bf16) + `m-a-p/YuE2-Vae` (0.49 GiB) |

## What limits performance here

YuE2 runs in four stages, and they have very different costs:

1. **Score planning (AR, `cot=full`/`melody`)** — autoregressive, up to 4096 tokens.
2. **Semantic generation (AR)** — autoregressive, up to 9000 tokens. This is the
   dominant cost of a full song.
3. **NAR flow matching** — a fixed 32-step midpoint ODE over the whole song.
4. **VAE decode** — one pass, tiled by default.

Stages 1 and 2 are batch-1 autoregressive decode. Each token requires reading
the full 3 B-parameter checkpoint, so the loop is bound by memory bandwidth, not
FLOPs. On a 3090 (~936 GB/s) the roofline for bf16 3 B weights is roughly
150-160 tok/s; the measured ~96-99 tok/s is about 62 % of that, the rest going to
sampling arithmetic, the vocabulary projection over 184704 tokens, and the
per-step host hand-off.

## Changes made

### 1. Graph options were unreachable (the main fix)

`GraphAR` already implemented two accelerations:

- `attention_backend` — selects the fused variable-length FlashAttention kernel
  instead of a masked SDPA that can fall back to a slow math kernel.
- `fuse_projections` — concatenates Q/K/V and gate/up weights so each layer
  issues one GEMM per group instead of three and two.

Neither was reachable: `generate_tokens()` did not accept them and
`YuE2Pipeline` never passed them, so `GraphAR` always ran with its defaults and
`fuse_projections` was effectively dead outside the test suite.

`sampling.generate_tokens()` now forwards both, `YuE2Pipeline` supplies them from
the active profile, and the generation timing record reports
`fuse_projections` so an artifact shows what actually ran.

### 2. Profiles instead of hard-coded backend flags

`YuE2Pipeline.__init__` used to set the deterministic flags inline. Those
defaults exist so a seeded run reproduces the recorded benchmark coordinates and
they are worth keeping as the default for comparison. They also leave throughput
unused.

`yue2.perf` now owns those flags behind three named profiles:

| Profile | Fusion | Graphs | TF32 | cuDNN autotune | Changes results |
|---|---|---|---|---|---|
| `reference` | no | yes | no | no | no — identical to the pinned release |
| `balanced` | yes | yes | no | no | no — same arithmetic, fewer launches |
| `fast` | yes | yes | yes | yes | **yes** |

`fast` is never selected implicitly. `validated_profile()` downgrades impossible
combinations and explains why.

### 3. Attention kernel is now explicit

With `auto`, `GraphAR` selects `flash` whenever the variable-length
`_flash_attention_forward` schema exposes `seqused_k`, which it does on the
pinned torch 2.10 build. All benchmark runs below confirmed `attention=flash`.
Exposing the choice lets you pin `cudnn` or `sdpa` for comparison or debugging.

## Measured results

Single prompt, `cot=full`, `seed=831001`, 2000 semantic tokens, 80 s of audio,
one run per profile **back to back in a single session**, warm decode measured by
decoding the same latents twice.

| Profile | ABC tok/s | Semantic tok/s | NAR (s) | VAE warm (s) | Peak VRAM |
|---|---|---|---|---|---|
| `reference` | 71.75 | 74.20 | 18.0 | 1.84 | 7.29 GiB |
| `balanced` | 78.57 | 76.23 | 18.0 | 1.79 | 8.95 GiB |
| `fast` | 78.78 | 75.89 | 18.0 | 1.12 | 8.97 GiB |

Reading these honestly:

- **`balanced` buys about +9.5 % on planning and +2.7 % on semantic decode.**
  Both profiles reported `attention=flash`, so the gain comes from projection
  fusion alone. It changes no precision setting: the fused GEMMs compute the same
  products as the separate ones, so results can differ only through
  floating-point reassociation, which the graph parity test bounds at 1e-7.
- **`fast` adds nothing to the AR loop** over `balanced` — the decode is
  bandwidth-bound, and TF32 changes matmul precision, not bytes moved. It does
  cut VAE decode by roughly 40 % (1.79 s → 1.12 s), where the VAE's fp32
  convolutions are compute-bound and tensor cores help.
- **NAR is unaffected** by any profile; it is a fixed 32-step solver.
- **Fusion costs VRAM**: peak rises from 7.29 to ~8.96 GiB because the
  concatenated QKV and gate/up weights are materialised on the device. With
  24 GiB this is not a constraint.

### Measure profiles in one session, never across sessions

An earlier session measured the same three profiles at 92.7 / 98.7 / 99.5 ABC
tok/s — roughly **25 % faster in absolute terms** than the table above, with the
same ordering and similar relative gaps. Re-running the identical `--max-tokens
2000` command later reproduced 76.2 and 76.3 tok/s twice in a row, so this was a
change in the machine's performance state (the display is driven by the same
GPU), not a configuration difference.

Treat the **relative** deltas between profiles as the signal and the absolute
throughput as environment-dependent. Always compare profiles inside one
back-to-back run; each `scripts/benchmark.py` invocation appends a timestamped
JSON record under `benches/` so the conditions of a comparison stay attached to
its numbers.

### A quick run for reference

A short `--quick` probe (400 planner tokens, 600 semantic tokens → 24.0 s audio)
on the `reference` profile completed end to end in **34.6 s**: planning 4.9 s,
semantic 6.6 s, NAR 4.1 s, VAE 15.7 s of which 14.8 s was the one-time decoder
load. Peak VRAM was 6.97 GiB.

The first decode of a process pays the VAE load. Because `YuE2Pipeline.decode()`
moves the decoder back to CPU when it finishes, that cost is paid again on every
subsequent decode unless the decoder stays resident — worth knowing when reading
`vae_seconds` out of `result.json`.

## FP8 is not available on this GPU

`quantization="fp8"` calls `prepare_fp8_ar()`, which requires compute capability
>= 8.9. The RTX 3090 is 8.6, so it raises before doing any work. This is a
hardware limit, not a configuration problem: consumer Ampere has no FP8 tensor
cores. `detect_hardware()["fp8_supported"]` reports this and the UI surfaces it,
so the failure is explained up front rather than at generation time.

## The optional vLLM backend does not start out of the box

`uv sync --extra fast` (or `make fast`) installs `vllm==0.19.0` cleanly and the
AR checkpoint derivation succeeds, but engine construction fails:

```
torch._inductor.exc.InductorError: PermissionError: [Errno 13] Permission denied: 'nvcc'
```

vLLM 0.19.0 defaults to `CompilationMode.VLLM_COMPILE`, and `torch.compile`'s
Inductor backend shells out to `nvcc`, which is not installed here (the wheel
targets CUDA 12.9; this environment has torch cu128 and no CUDA toolkit). Weight
loading and FlashAttention selection both succeed before this point, so the
blocker is the missing compiler, not the model.

Setting `YUE2_VLLM_ENFORCE_EAGER=1` skips compilation and the engine does start.
Measured **back to back** against the torch path:

| Backend (2000 semantic tokens, 80 s audio) | ABC tok/s | Semantic tok/s | NAR (s) | VAE warm (s) |
|---|---|---|---|---|
| torch, `balanced` | 76.26 | 76.18 | 18.0 | 1.72 |
| vLLM, `enforce_eager=1` | 85.13 | 94.79 | 21.0 | 1.77 |

But a single pair of runs is not enough here, because the torch path is not
stable across sessions while vLLM is. Semantic decode throughput:

| Session | torch `balanced` | vLLM eager |
|---|---|---|
| quiet host | 96.4 | 95.7 |
| contended host | 76.2 | 94.8 |

The torch path swings by about 25 %; vLLM holds near 95 tok/s in both sessions.
The likely reason is architectural: the torch path decodes in a Python loop with
a GPU→CPU synchronization per token (`int(next_id.item())`, then `graph.step`),
so it is latency-bound and degrades when the host is busy. vLLM runs its engine
loop in a worker process with its own scheduling and is far less exposed.

So the honest conclusion is:

- **vLLM eager is at least as fast as the torch path, and more consistent.**
  Even with compilation disabled it matched the torch path's best case and beat
  its contended case by ~24 %.
- **On a quiet machine the two are equivalent** for semantic decode, and the
  torch path is ~10 % faster on planning in that state.
- **NAR is consistently ~3 s slower after a vLLM run** (21.0 vs 18.0 s). NAR does
  not use the AR backend at all, so this is the worker process still releasing
  GPU memory. Budget for it if you generate song after song.
- **Compiled vLLM remains untested.** Every number above has vLLM's own fused
  kernels switched off. Installing a CUDA toolkit whose version matches the
  wheel (so `nvcc` is on `PATH`, or `CUDA_HOME` points at one) would remove the
  `enforce_eager` workaround and is the obvious next thing to measure.

For a quiet single-user desktop, `torch` + `balanced` is the simpler choice: no
extra 2 GB of dependencies, no worker process, faster planning, and no
teardown penalty before NAR. Choose `vLLM` + `YUE2_VLLM_ENFORCE_EAGER=1` if the
machine is shared with other work, or if you batch many requests.

## Reproducing

```bash
make bench                                            # the three profiles back to back
uv run python scripts/benchmark.py --quick            # fast wiring check
```

Each run appends a timestamped JSON record under `benches/` and refreshes
`benches/latest.json`, so profiles can be compared across sessions.

The GPU timing tests are separate and marked `slow`:

```bash
uv run pytest tests/ -m slow -s
```

## Caveats

- **Session-to-session variance dominates small differences.** The same command
  measured 96.4 and 76.2 tok/s on different occasions. Only compare profiles
  inside one back-to-back run; see the note above.
- Each profile row above is a **single run**. Differences under about 3 % are
  within run-to-run noise, which puts the semantic gain near the noise floor and
  the planning gain comfortably above it.
- These are local timings, not a reproduction of the published benchmark
  aggregate, and no quality claim is made for any profile. `fast` changes
  floating-point results, so compare audio with the same seed before adopting it.
- The comparison holds the CUDA toolkit, driver, and PyTorch build fixed.
  Separate GPUs or runtime versions will differ.
- The display is driven by the same GPU, so desktop activity is a plausible
  source of the variance. A headless or dedicated compute GPU would be more
  stable, and none of these numbers should be read as a hardware maximum.
