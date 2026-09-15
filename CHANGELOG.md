# Changelog

All notable changes to this working copy are recorded here. This is a fork of
[`multimodal-art-projection/YuE`](https://github.com/multimodal-art-projection/YuE)
prepared for local RTX 3090 inference with a Gradio front end.

## [Unreleased] — anime cue library, loop builder, output housekeeping

### Added

- **An "Anime sounds" tab** with **123 background-music cues across 17 scene
  categories** — morning routine, school life, comedy, suspense, standoff,
  combat, power-up, grief, flashback, romance, horror, traditional Japan,
  sci-fi, ambience beds, and stingers. Each cue carries a plain-language
  `use_when`, a description of how it sounds, the exact prompt, tags, and a
  **Send to generator** button that fills the Create tab (style, 30 s length,
  influence, section plan, and Music-only) and switches to it. No copying.
- **`anime_cues.json`** as the single source of truth, with a documented schema
  and top-level `defaults`. Append a block, press **Reload from disk**, and it
  appears without a restart. Malformed entries are skipped and counted rather
  than breaking the tab, because the file is meant to be hand-edited.
- **`src/yue2/loops.py`** — a loop builder. It beat-tracks the *rendered audio*,
  snaps the loop window so its length is a whole number of bars, folds the ends
  together into a seamless loop, and crossfades an extended track
  (`intro + loop x N + outro`). Emits `loop.wav` and `extended.flac` beside the
  cue, with repeat-count and crossfade controls.
- **Tiered "Clear output"** on the Create tab: clear the on-screen list (files
  untouched), delete audio files while keeping scores and metadata, or delete
  every run. The destructive pair needs a confirmation tick, and each reports
  the count and megabytes freed.
- **A `loop` dependency group** (`librosa`) and tests for the cue library, the
  cue→generator mapping, filtering, and the loop arithmetic.

### Notes

- **The model cannot place a loop point, and this was measured rather than
  assumed.** Asked for `[intro] [verse] [chorus] [verse] [outro]` at 30 s and
  90 BPM, it wrote a single `% verse` marker and chose 87 BPM. The cause is in
  the code: the ABC score reaches the model as prompt tokens only
  (`token_prefixes` → `tokenizer.encode`), and nothing maps its bars to audio
  seconds. Length is the one thing that *is* exact, because it is the token
  budget at 25 frames per second. So the loop is assembled after generation.
- Snapping uses the audio's own tempo, not the score's. The score said 87 BPM
  where the request said 90 — a 3 % error that would drift roughly 0.3 s across
  a 10 s loop.
- A requested 15–25 s window often snaps to something like 14.95–23.59 s,
  because the loop has to be a whole number of bars. The snapped values are
  reported so the sliders can be nudged.
- Seams are clean on sustained and ambient material and rougher on percussive
  cues; the crossfade helps but does not fully hide it on hits.
- librosa brings numba, scikit-learn and friends — noticeably more than the
  ~40 MB first estimated. It is a separate group for that reason.

## [Unreleased] — music-only mode, song library, layout fixes

### Added

- **Music-only (instrumental) generation.** The released checkpoint has no
  instrumental mode: its model card describes songs "with vocals and
  accompaniment", and an empty lyrics field still sings. This adds a
  **Music only (no vocals)** toggle backed by the community
  [instrumental adapter](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras),
  which is a small AR LoRA (~0.6 GB) folded into the base linear weights as
  ``W += scale * (B @ A)``. Turning it on greys out the lyrics box and reveals a
  **section plan** field, because the adapter reads bracketed section tags in
  place of lyrics.
- **`src/yue2/lora.py`** — adapter download, section-plan validation, and the
  weight merge. Keyed by tensor name rather than list order, so an unexpected
  adapter layout raises instead of merging deltas into the wrong projection.
- **An empty lyrics box now means music only.** This is what was asked for and
  it also matches the model's behaviour, rather than failing with a validation
  error while the model sings anyway.
- **A song library on every song tab**, newest first, with a player per entry
  (play/pause), its length, timestamp and style. Rendered dynamically from the
  `runs/` directory, so it survives a restart and always reflects what is
  actually on disk.
- Tests for the section-plan grammar, the LoRA merge arithmetic, the library
  scan, and the new layout.

### Changed

- **The Generate button and the score/timings panels swapped places.** The score
  and timings move to the left column under the inputs; Generate moves to the
  right column, with the song library directly beneath it.

### Fixed

- **Two progress bars, one of them fading.** Gradio paints a bar on every output
  component by default *and* runs the queue indicator, so a handler with four
  outputs showed several. The generation events now use
  `show_progress="minimal"` (a single indicator) and token progress is throttled
  to four repaints a second instead of one per token.
- **`StarletteDeprecationWarning` about `HTTP_422_UNPROCESSABLE_ENTITY` flooded
  the console** on every queue poll. The notice is now filtered at startup.

### Notes

- The instrumental adapter is **CC BY-NC 4.0**, like the base checkpoint, so
  this is a non-commercial path.
- It is a community adapter, not an official release. Generation is verified to
  run and produce audio; the *absence of vocals* was not verified objectively -
  that would need a dedicated vocal classifier - so listen before relying on it.
- It cannot be combined with `quantization="fp8"`, which replaces the plain
  BF16 linears the merge targets. `apply_instrumental` raises rather than
  silently producing a half-adapted model.

## [Unreleased] — user-facing UI redesign

### Added

- **Five tabs instead of one long form.** `Create`, `Cover from a recording`,
  `Edit a song`, `Advanced`, and `Hardware`. The Create tab now shows only style,
  lyrics, song length, prompt influence, planning, render quality, and seed;
  every backend, performance and sampling control moved to `Advanced`.
- **Song length in seconds.** A 10-360 s slider replaces the raw token budget.
  The conversion is exact: YuE2's VAE decodes at 48000 Hz with a 1920x
  downsampling ratio (25 frames per second), and the MERT-v2 backbone behind
  SheetSage2 independently declares `frame_rate: 25.0`. Measured output confirms
  it: 600 tokens decoded to 24.00 s, 2000 to 80.00 s, 2591 to 103.60 s.
- **`src/yue2/duration.py`** — the seconds/tokens arithmetic, plus
  `fit_sampling_bounds()` so a short request lowers the checkpoint's 200-token
  floor instead of failing `Sampling` validation.
- **Prompt influence** on every song tab, which is the pipeline's `cfg_scale`.
- **Reference-audio covers.** `src/yue2/transcribe.py` wraps
  [SheetSage2](https://huggingface.co/m-a-p/SheetSage2): upload a recording, get
  an editable two-voice ABC score back, then render it in a new style. The
  transcriber loads lazily and is released before generation so the two never
  share the 24 GiB card.
- **An Edit tab.** Load a score from a saved run, or send the last Create result
  to it, then change notes, chords or tempo and re-render.
- **`cover` dependency group and extra** — torchaudio, scipy, mir_eval,
  pretty_midi, mido. SheetSage2's published `requirements.txt` pins torch 2.8 and
  transformers 4.45, but the official YuE2 Space loads it on the current
  torch 2.10 / transformers 4.57 stack with only these helpers, and that is what
  is verified here.
- **A custom-override convention.** The two settings that exist on both a simple
  tab and `Advanced` (ODE steps, semantic token budget) default the Advanced
  control to 0, meaning "use the friendly control". A positive value overrides
  it, so there is one source of truth rather than two that disagree.

### Fixed

- **`decode_full` was a control that did nothing.** `YuE2Pipeline.__call__`
  accepted the argument in the UI but never passed it to `decode()`, so the
  "decode in one VAE pass" checkbox had no effect. It is now a real keyword
  argument threaded through to `self.decode(latents, full=decode_full)`.

### Notes

- Generation still runs one request at a time; the queue enforces it.
- The `Create` tab no longer exposes the model path, backend, or sampling knobs.
  `tests/test_app.py` asserts this both ways: the friendly labels must be under
  `Create`, and the technical labels must not be.

## [Unreleased] — uv-managed environment

### Added

- **`uv.lock`** — a full lockfile for the dependency graph (211 packages),
  committed so the environment is reproducible with `uv sync`.
- **`.python-version`** — pins the development interpreter to 3.12. `requires-python`
  stays at `>=3.10` to match the upstream package; the pin only affects local
  development, and uv installs the interpreter into `.uvpython/` on first sync.
- **`[dependency-groups]` in `pyproject.toml`** — `dev` (pytest) and `ui`
  (gradio), both listed in `[tool.uv] default-groups`, so a plain `uv sync`
  produces a checkout that can run the UI and the test suite. `fast` (vLLM)
  remains opt-in.
- **`Makefile`** — `setup`, `sync`, `lock`, `fast`, `app`, `test`, `test-all`,
  `bench`, `download`, `doctor`, and `clean`. It exports the uv variables below,
  so no manual environment setup is needed.
- **`docs/uv.md`** — the uv workflow, including why three environment variables
  are needed here.
- **`.env.example`** — template for the `HF_HOME` setting that keeps the
  Hugging Face cache inside the checkout.

### Changed

- **`README.md`**, **`docs/optimization.md`**, **`docs/generation.md`** — the
  quick-start and reproduction commands now use `make`/`uv run` instead of
  `.venv/bin/python` and `pip`.
- **`src/yue2/fast.py`** — the missing-vLLM error now suggests
  `uv sync --extra fast` as well as the pip equivalent.

### Notes

- **`uv` was already present but only as an ad-hoc installer**, with its cache
  and interpreter directories forced through environment variables on every
  command. The project now carries that configuration itself.
- **`cache-dir` in `[tool.uv]` does not work here.** uv initialises its cache
  before reading project configuration, so it still writes to `~/.cache/uv`.
  `UV_CACHE_DIR` must be a real environment variable. `.env` is likewise read
  only for the subprocess environment, never for uv's own settings — hence the
  `Makefile` exporting all three.
- **`default-extras` is silently ignored** by uv 0.11.32; a named dependency
  group listed in `default-groups` is the working equivalent.
- **`uv sync` removes vLLM** unless the `fast` extra is requested, because it
  reconciles the environment with the lockfile. One test in `tests/test_fast.py`
  consequently skips with `Optional vLLM package is not installed`; the suite
  reports 207 passed / 11 skipped without it and 208 passed / 10 skipped with
  `make fast`. This is expected, not a regression.
- **The three HF-cache test failures** seen earlier (`OSError: Read-only file
  system`) are resolved by `.env` supplying `HF_HOME`, so `make test` now passes
  cleanly.

## [Unreleased] — RTX 3090 local inference + web UI

### Added

- **`src/yue2/perf.py`** — hardware-aware performance profiles.
  - `PerfProfile` dataclass and three named profiles: `reference` (the historical
    backend flags), `balanced` (numerics-preserving: fused projections + explicit
    attention kernel), and `fast` (additionally TF32 + cuDNN autotuning, which
    changes floating-point results).
  - `detect_hardware()` reports device name, VRAM, compute capability, BF16 and
    FP8 support, FlashAttention availability, and a suggested VAE tile size.
  - `validated_profile()` downgrades impossible combinations (for example
    projection fusion without CUDA graphs) and explains why.
  - `apply_profile()` is the single place that sets the process-global
    `torch.backends` flags.
- **`app.py`** — Gradio front end exposing every pipeline option: request fields
  (style, lyrics, cot, seed, cfg scale, id, external ABC), both sampling blocks
  (temperature, top-p, top-k, repetition penalty, penalty window, min/max
  tokens), ODE steps, VAE decode mode, backend, quantization, memory budget,
  offload, hash verification, offline mode, model/VAE revisions, Hugging Face
  token, hub cache directory, and the performance controls. Includes a hardware
  tab, live token progress, ABC score output, and a timings panel.
- **`scripts/benchmark.py`** — per-stage timing harness. Records planning,
  semantic AR, NAR, cold and warm VAE decode, throughput, TTFT, and peak VRAM,
  and appends each run as JSON under `benches/`.
- **`scripts/download_models.py`** — fetches only the reviewed public model
  files for `YuE2-3B` and the VAE into `models/`.
- **`tests/test_perf.py`** — profile contracts and the pipeline→`generate_tokens`
  wiring.
- **`tests/test_benchmark.py`** — benchmark arithmetic plus a `slow`-marked test
  that loads the real checkpoint, measures inference time, and asserts the
  result is plausible (non-silent audio, throughput floor, VRAM ceiling).
- **`docs/optimization.md`** — measured RTX 3090 results and the reasoning
  behind each setting.

### Changed

- **`src/yue2/sampling.py`** — `generate_tokens()` accepts `attention_backend`
  and `fuse_projections` and forwards them to `GraphAR`; the timing record now
  reports `fuse_projections`. Previously `GraphAR` supported both options but
  nothing could reach them.
- **`src/yue2/pipeline.py`** — `YuE2Pipeline` accepts `perf=` (a profile name or
  a `PerfProfile`) and applies it instead of hard-coding the deterministic
  backend flags; `vae_core_frames` can now come from the profile; the effective
  configuration recorded in every artifact includes the active profile.
- **`tests/test_progress_integration.py`** — the `bare_pipe` stub now provides
  the `perf` attribute required by the pipeline contract.
- **`src/yue2/fast.py`** — the vLLM worker honours `YUE2_VLLM_ENFORCE_EAGER=1`,
  which passes `enforce_eager=True` to the engine. vLLM 0.19.0 defaults to
  `CompilationMode.VLLM_COMPILE` and its Inductor backend shells out to `nvcc`;
  without a CUDA toolkit the engine cannot start at all. The default behaviour
  is unchanged.
- **`pyproject.toml`** — added the `ui` extra (`gradio==6.14.0`) and registered
  the `slow` pytest marker.
- **`.gitignore`** — ignore `.uvcache/`, `.uvpython/`, `.hf/`, and `benches/`.

### Fixed

- **Dependency conflict between Gradio and the pinned `transformers`.** Gradio
  6.18+ requires `huggingface-hub>=1.2`, while `transformers==4.57.6` requires
  `huggingface-hub<1.0`; installing the newest Gradio silently upgraded the hub
  and broke `import transformers`. Pinned to `gradio==6.14.0`, the newest release
  that still accepts `huggingface-hub==0.36.2`.
- **`benchmark.py` treated the default `--max-tokens 9000` as an explicit
  override**, so the checkpoint defaults were never used. The default is now
  `None`, meaning "use the checkpoint default".
- **An untouched Gradio text or code component can deliver `None`, not `""`.**
  The generate handler called `.strip()` directly on the ABC score and style
  fields, so submitting the form with an empty score box would raise
  `AttributeError` instead of generating. Every free-text input is now coerced
  before use. Caught by the end-to-end handler test.
- **The UI exited instead of starting when port 7860 was already taken.**
  `app.py` handed a fixed `server_port=7860` to Gradio, which raised
  `OSError: Cannot find empty port in range: 7860-7860`. A second instance, or
  any other process on that port, made the app unusable with a message that did
  not say what to do. The default is now the first free port from 7860 upward,
  with a notice on stderr when it moves; an explicit `--port` still fails loudly
  if that exact port is occupied.

### Notes and findings

- **FP8 is unreachable on an RTX 3090.** `quantization="fp8"` requires compute
  capability >= 8.9 (Ada/Hopper); the 3090 is 8.6, so
  `prepare_fp8_ar()` raises. The UI surfaces this rather than failing late.
- **The RTX 3090 is not the limiting factor for VRAM.** The full BF16 checkpoint
  and KV cache peak around 7.3 GiB, well inside 24 GiB.
- **Projection fusion costs VRAM.** `balanced`/`fast` raise peak usage from
  about 7.3 GiB to about 9.0 GiB because the concatenated QKV and gate/up
  weights are materialised on the device.
