#!/usr/bin/env python
"""Gradio front end for YuE2 song generation.

The interface is split so that a first-time user never meets a backend flag:

* **Create**  -- style, lyrics, length, prompt influence, quality. Nothing else.
* **Cover**   -- upload a recording, transcribe its melody with SheetSage2, then
  render it in a new style.
* **Edit**    -- load or paste a score, change notes/chords/tempo, regenerate.
* **Advanced**-- every backend, performance and sampling control, including the
  raw values behind the friendly names on the other tabs.
* **Hardware**-- what the app detected.

Two settings exist on both a simple tab and the Advanced tab (ODE steps and the
semantic token budget). The Advanced control defaults to 0, meaning "use the
friendly control"; a non-zero value overrides it. That convention keeps a single
source of truth without hiding the raw knob.

Run with::

    uv run app.py --model models/YuE2-3B --vae models/YuE2-Vae
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import socket
import sys
import threading
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Gradio's queue-join route raises a Starlette deprecation warning on every
# poll, which buries real output. Older Gradio/Starlette pairs name the 422
# status constant in a way Starlette now discourages; nothing is wrong, so the
# notice is silenced rather than logged thousands of times.
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr  # noqa: E402

from yue2.duration import (  # noqa: E402
    MAX_SECONDS,
    FRAMES_PER_SECOND,
    fit_sampling_bounds,
    seconds_to_tokens,
    tokens_to_seconds,
)
from yue2.lora import (  # noqa: E402
    DEFAULT_SECTION_PLAN,
    LoRAError,
    apply_instrumental,
    normalise_section_plan,
)
from yue2.perf import (  # noqa: E402
    PROFILE_NAMES,
    PerfProfile,
    detect_hardware,
    profile as named_profile,
)
from yue2.protocol import GenerationConfig, Sampling, SongRequest  # noqa: E402
from yue2.transcribe import Transcriber, TranscriptionError, dependencies_available  # noqa: E402

RUNS_DIR = REPO_ROOT / "runs"
ANIME_CUES_PATH = Path(os.environ.get("YUE2_CUES", REPO_ROOT / "anime_cues.json"))
ALL_CATEGORIES = "All categories"
DEFAULT_PORT = 7860

DEFAULT_STYLE = (
    "English, warm piano pop, expressive female voice, acoustic piano, "
    "rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM"
)
DEFAULT_LYRICS = (
    "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n"
    "Fold the night and leave it here\nMorning has a sky to clear\n\n"
    "[Chorus]\nLet the day come into view\nEvery road begins with you\n"
    "Hold a little room for light\nWe will sing beyond the night"
)

PLANNING_CHOICES = [
    ("Melody + chords (recommended)", "full"),
    ("Melody only (freer arrangement)", "melody"),
    ("No score (fastest, least control)", "off"),
]
QUALITY_CHOICES = [
    ("Draft - 12 steps (fastest)", 12),
    ("Fast - 16 steps", 16),
    ("Balanced - 24 steps", 24),
    ("Best - 32 steps", 32),
]

# Semantic max_tokens is owned by the Song length slider; the planner has its
# own budget because it is a different stage with a different range.
ABC_FIELDS = (
    ("temperature", "Temperature", 0.0, 5.0, 0.05),
    ("top_p", "Top-p", 0.01, 1.0, 0.01),
    ("top_k", "Top-k", 1, 2000, 1),
    ("repetition_penalty", "Repetition penalty", 0.5, 3.0, 0.005),
    ("penalty_window", "Penalty window", 1, 100, 1),
    ("min_tokens", "Min tokens", 0, 4096, 1),
    ("max_tokens", "Max tokens (planner budget)", 1, 4096, 1),
)
SEMANTIC_FIELDS = (
    ("temperature", "Temperature", 0.0, 5.0, 0.05),
    ("top_p", "Top-p", 0.01, 1.0, 0.01),
    ("top_k", "Top-k", 1, 2000, 1),
    ("repetition_penalty", "Repetition penalty", 0.5, 3.0, 0.005),
    ("penalty_window", "Penalty window", 1, 100, 1),
    ("min_tokens", "Min tokens (length floor)", 0, 9000, 1),
)

EXAMPLES = [
    ["Dreamy synth-pop, warm female lead vocal, pulsing bass, shimmering synths, uplifting",
     "[Verse]\nCity windows turn to gold\nEvery streetlight has a story\n"
     "[Chorus]\nStay awake, the night is ours\nWe can dance beneath the stars", 90, 1.0, "full", 16, 42],
    ["Acoustic indie folk, intimate male vocal, fingerpicked guitar, gentle strings",
     "[Verse]\nDust is dancing in the doorway\nSummer settles on the road\n"
     "[Chorus]\nTake me home across the river\nWhere the evening moves so slow", 120, 1.0, "full", 24, 7],
    ["Lo-fi hip hop instrumental, dusty vinyl drums, mellow Rhodes chords, late night study mood",
     "[Instrumental]\n[Verse]\n[Chorus]", 60, 1.2, "full", 12, 99],
]


class PipelineCache:
    """Hold one loaded pipeline; rebuild only when construction inputs change."""

    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._pipe = None

    def get(self, key, factory):
        with self._lock:
            if self._pipe is None or self._key != key:
                self.close()
                self._pipe, self._key = factory(), key
            return self._pipe

    def close(self):
        if self._pipe is not None:
            try:
                self._pipe.close()
            except Exception:  # pragma: no cover - best effort teardown
                pass
        self._pipe, self._key = None, None


CACHE = PipelineCache()
TRANSCRIBER = Transcriber()
GENERATION_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def resolve_token_budget(length_seconds, custom_max_tokens, sem_min_tokens):
    """Pick the semantic token budget and a floor that satisfies ``Sampling``.

    The Advanced tab's override defaults to 0, meaning "use the Song length
    slider". A positive value wins, so there is exactly one source of truth.
    """
    override = int(custom_max_tokens or 0)
    budget = override if override > 0 else seconds_to_tokens(length_seconds)
    return fit_sampling_bounds(int(sem_min_tokens), budget)


def resolve_ode_steps(quality, custom_ode_steps):
    """Render quality sets the ODE steps unless Advanced overrides it."""
    override = int(custom_ode_steps or 0)
    return override if override > 0 else int(quality)


def _sampling_from(values, fields):
    """Build a Sampling from a mapping of field name -> widget value."""
    kwargs = {name: values[name] for name, *_ in fields}
    for name in ("top_k", "penalty_window", "min_tokens", "max_tokens"):
        if name in kwargs:
            kwargs[name] = int(kwargs[name])
    if "min_tokens" in kwargs and "max_tokens" in kwargs and kwargs["min_tokens"] > kwargs["max_tokens"]:
        kwargs["min_tokens"] = kwargs["max_tokens"]
    return Sampling(**kwargs)


def _pipeline_key(kwargs):
    return (
        kwargs["model"], kwargs["vae"], kwargs["device"], kwargs["backend"],
        kwargs["quantization"], kwargs["offload_ar"], kwargs["budget"],
        kwargs["verify_hashes"], kwargs["profile"], kwargs["attention"],
        kwargs["fuse"], kwargs["cuda_graph"], kwargs["tf32"], kwargs["cudnn_bench"],
        kwargs["vae_frames"], kwargs["offline"], kwargs["revision"], kwargs["vae_revision"],
        kwargs["hf_token"], kwargs["cache_dir"], kwargs["music_only"],
    )


def _build_pipeline(kwargs):
    from yue2 import YuE2Pipeline

    base = named_profile(kwargs["profile"])
    performer = PerfProfile(
        name=base.name,
        attention_backend=kwargs["attention"],
        fuse_projections=bool(kwargs["fuse"]),
        cuda_graph=bool(kwargs["cuda_graph"]),
        tf32=bool(kwargs["tf32"]),
        cudnn_benchmark=bool(kwargs["cudnn_bench"]),
        vae_core_frames=int(kwargs["vae_frames"]) or None,
        changes_numerics=bool(kwargs["tf32"] or kwargs["cudnn_bench"]),
    )
    return YuE2Pipeline.from_pretrained(
        kwargs["model"],
        vae=kwargs["vae"],
        revision=kwargs["revision"] or None,
        vae_revision=kwargs["vae_revision"] or None,
        local_files_only=bool(kwargs["offline"]),
        token=kwargs["hf_token"],
        cache_dir=kwargs["cache_dir"],
        device=kwargs["device"],
        memory_budget_gib=float(kwargs["budget"]),
        backend=kwargs["backend"],
        quantization=kwargs["quantization"],
        offload_ar=bool(kwargs["offload_ar"]),
        verify_hashes=bool(kwargs["verify_hashes"]),
        perf=performer,
        progress=False,
    )


def _build_instrumental(kwargs):
    """Build the pipeline and fold the instrumental adapter into its weights."""
    pipe = _build_pipeline(kwargs)
    model, report = apply_instrumental(
        pipe,
        cache_dir=kwargs["cache_dir"],
        local_files_only=bool(kwargs["offline"]),
    )
    pipe._instrumental_report = report
    return pipe


def list_recent_runs(limit=25):
    """Directories under runs/ that hold a saved score from an earlier song."""
    if not RUNS_DIR.is_dir():
        return []
    found = []
    for entry in RUNS_DIR.iterdir():
        if entry.is_dir() and (entry / "score.abc").is_file():
            found.append((entry.stat().st_mtime, entry.name))
    found.sort(reverse=True)
    return [name for _mtime, name in found[:limit]]


def scan_library(limit=60):
    """Every generated song, newest first.

    Reads the runs/ directory rather than an in-memory list so the library
    survives a restart and stays honest about what is actually on disk.
    """
    entries = []
    if not RUNS_DIR.is_dir():
        return entries
    for directory in RUNS_DIR.iterdir():
        audio = directory / "audio.flac"
        if not directory.is_dir() or not audio.is_file():
            continue
        request = {}
        request_file = directory / "request.json"
        if request_file.is_file():
            try:
                request = json.loads(request_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                request = {}
        seconds = None
        result_file = directory / "result.json"
        if result_file.is_file():
            try:
                seconds = json.loads(result_file.read_text(encoding="utf-8")).get("audio_seconds")
            except (OSError, ValueError):
                pass
        style = (request.get("style") or "").strip()
        lyrics = (request.get("lyrics") or "").strip()
        entries.append({
            "name": directory.name,
            "audio": str(audio.resolve()),
            "style": style[:160] or "(no style recorded)",
            "snippet": " ".join(lyrics.split())[:120] or "(instrumental)",
            "seconds": seconds,
            "has_score": (directory / "score.abc").is_file(),
            "mtime": audio.stat().st_mtime,
            "when": datetime.fromtimestamp(audio.stat().st_mtime).strftime("%d %b %H:%M"),
        })
    entries.sort(key=lambda entry: entry["mtime"], reverse=True)
    return entries[:limit]


def load_run(name):
    """Return ``(abc, style, lyrics, status)`` for a saved run directory."""
    if not name:
        return "", "", "", "Pick a saved song first."
    directory = RUNS_DIR / name
    if not directory.is_dir():
        return "", "", "", f"No such run: {name}"
    abc = ""
    if (directory / "score.abc").is_file():
        abc = (directory / "score.abc").read_text(encoding="utf-8")
    style = lyrics = ""
    request_file = directory / "request.json"
    if request_file.is_file():
        try:
            request = json.loads(request_file.read_text(encoding="utf-8"))
            style = request.get("style") or ""
            lyrics = request.get("lyrics") or ""
        except (OSError, ValueError):
            pass
    note = f"Loaded `{name}`."
    if not abc:
        note += " That run used `cot=off`, so it has no score to edit."
    return abc, style, lyrics, note


# --------------------------------------------------------------------------- #
# anime cue library
# --------------------------------------------------------------------------- #

def load_cue_library(path=None):
    """Read the cue JSON, tolerating hand edits.

    A malformed entry is skipped with a count rather than breaking the whole
    tab, because the file is meant to be appended to by hand.
    """
    path = Path(path or ANIME_CUES_PATH)
    if not path.is_file():
        return [], [], {}, f"No cue file at `{path.name}`. Create it, then press Reload."
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [], [], {}, f"Could not read `{path.name}`: {error}"
    if not isinstance(data, dict) or not isinstance(data.get("cues"), list):
        return [], [], {}, f"`{path.name}` needs a top-level `cues` array."
    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    cues, skipped = [], 0
    for entry in data["cues"]:
        if not isinstance(entry, dict) or not str(entry.get("style") or "").strip():
            skipped += 1
            continue
        cue = dict(entry)
        cue.setdefault("title", cue.get("id") or "Untitled cue")
        cue.setdefault("category", "Uncategorised")
        cue.setdefault("use_when", "")
        cue.setdefault("sound", "")
        cue.setdefault("tags", [])
        cue.setdefault("seconds", defaults.get("seconds", 30))
        cue.setdefault("influence", defaults.get("influence", 1.0))
        cue.setdefault("plan", defaults.get("plan") or DEFAULT_SECTION_PLAN)
        cues.append(cue)
    categories = data.get("categories")
    if not isinstance(categories, list) or not categories:
        categories = sorted({cue["category"] for cue in cues})
    note = f"**{len(cues)} cues** in {len(categories)} categories"
    if skipped:
        note += f" - {skipped} skipped (each entry needs at least a `style`)"
    return cues, list(categories), defaults, note


def filter_cues(cues, category=None, query=""):
    """Narrow the library by category and a free-text search."""
    query = (query or "").strip().lower()
    selected = []
    for cue in cues:
        if category and category != ALL_CATEGORIES and cue["category"] != category:
            continue
        if query:
            haystack = " ".join([
                str(cue.get("title", "")), str(cue.get("category", "")),
                str(cue.get("use_when", "")), str(cue.get("sound", "")),
                str(cue.get("style", "")), " ".join(cue.get("tags") or []),
            ]).lower()
            if not all(word in haystack for word in query.split()):
                continue
        selected.append(cue)
    return selected


def reload_cues(category, query):
    """Re-read the cue file and re-apply the current filter."""
    catalogue, categories, _defaults, note = load_cue_library()
    matched = filter_cues(catalogue, category, query)
    if catalogue:
        note += f" · showing **{len(matched)}**"
    return matched, note, gr.update(choices=[ALL_CATEGORIES, *categories])


def cue_to_generator(cue):
    """Fill the Create tab from a cue, and switch to it.

    Returns updates for the Create controls plus a status line and the tab
    selection, so the prompt never has to be copied by hand.
    """
    cue = cue or {}
    style = str(cue.get("style") or "").strip()
    if not style:
        return (gr.update(),) * 6 + (gr.update(), "That cue has no style text.", gr.update())
    seed = cue.get("seed")
    return (
        style,
        int(cue.get("seconds") or 30),
        float(cue.get("influence") or 1.0),
        True,                                     # Music only: anime cues are instrumental
        str(cue.get("plan") or DEFAULT_SECTION_PLAN),
        "",                                       # clear any leftover lyrics
        int(seed) if seed not in (None, "") else gr.update(),
        f"**{cue.get('title', 'Cue')}** loaded into **Create**. Press *Generate song* there.",
        gr.update(selected="create"),
    )


def hardware_report():
    hardware = detect_hardware()
    lines = ["| Setting | Value |", "|---|---|"]
    for key, value in hardware.items():
        lines.append(f"| `{key}` | `{value}` |")
    fp8 = "supported" if hardware.get("fp8_supported") else "NOT supported (needs compute capability >= 8.9)"
    cover = "installed" if dependencies_available() else "not installed (`uv sync --extra cover`)"
    return (
        "\n".join(lines)
        + f"\n\n**FP8 on this device:** {fp8}\n\n"
        + f"**Reference-audio transcription:** {cover}\n"
    )


# --------------------------------------------------------------------------- #
# the single generation handler
# --------------------------------------------------------------------------- #

def generate(
    style, lyrics, length_seconds, prompt_influence, planning, quality, seed,
    music_only, section_plan, abc_text,
    custom_ode_steps, custom_max_tokens, decode_full,
    model, vae, device, backend, quantization, offload_ar, budget,
    profile_name, attention, fuse, cuda_graph, tf32, cudnn_bench, vae_frames,
    verify_hashes, offline, revision, vae_revision, hf_token, cache_dir,
    abc_temperature, abc_top_p, abc_top_k, abc_repetition_penalty,
    abc_penalty_window, abc_min_tokens, abc_max_tokens,
    sem_temperature, sem_top_p, sem_top_k, sem_repetition_penalty,
    sem_penalty_window, sem_min_tokens,
    save_artifacts, progress=gr.Progress(),
):
    """One handler for every tab; each button binds a different component set."""
    started = time.perf_counter()

    def stop(message):
        return None, "", "", message, scan_library()

    # An untouched Gradio text or code component can deliver None as well as "".
    style = (style or "").strip()
    lyrics = lyrics or ""
    abc_text = abc_text or ""
    section_plan = (section_plan or "").strip()
    model = (model or "").strip()
    vae = (vae or "").strip()
    revision = (revision or "").strip()
    vae_revision = (vae_revision or "").strip()
    hf_token = (hf_token or "").strip() or None
    cache_dir = (cache_dir or "").strip() or None

    music_only = bool(music_only)
    if not music_only and not lyrics.strip():
        # The base checkpoint has no instrumental mode: empty lyrics still sing.
        # An empty lyrics box is therefore read as "music only".
        music_only = True
    if music_only:
        try:
            lyrics = normalise_section_plan(section_plan or DEFAULT_SECTION_PLAN)
        except LoRAError as error:
            return None, "", "", f"Section plan problem: {error}", scan_library()
        # The instrumental adapter is trained for cot="full".
        planning = "full"

    if not style:
        return stop("Describe the musical style first.")
    if abc_text.strip() and planning == "off":
        return stop("A supplied score needs planning set to melody or full.")

    # Length: seconds -> semantic token budget. Advanced can override it.
    sem_floor, target_tokens = resolve_token_budget(length_seconds, custom_max_tokens, sem_min_tokens)
    ode_steps = resolve_ode_steps(quality, custom_ode_steps)

    sem_sampling = _sampling_from({
        "temperature": sem_temperature, "top_p": sem_top_p, "top_k": sem_top_k,
        "repetition_penalty": sem_repetition_penalty, "penalty_window": sem_penalty_window,
        "min_tokens": sem_floor, "max_tokens": target_tokens,
    }, SEMANTIC_FIELDS + (("max_tokens",),))
    abc_sampling = _sampling_from({
        "temperature": abc_temperature, "top_p": abc_top_p, "top_k": abc_top_k,
        "repetition_penalty": abc_repetition_penalty, "penalty_window": abc_penalty_window,
        "min_tokens": abc_min_tokens, "max_tokens": abc_max_tokens,
    }, ABC_FIELDS)
    config = GenerationConfig(abc=abc_sampling, semantic=sem_sampling, ode_steps=ode_steps)

    try:
        request = SongRequest(
            style=style, lyrics=lyrics, cot=planning, seed=int(seed),
            abc=abc_text.strip() or None, cfg_scale=prompt_influence, id="song",
        )
    except Exception as error:
        return None, "", "", f"Invalid request: {error}", scan_library()

    kwargs = dict(
        model=model, vae=vae, device=device, backend=backend,
        quantization=quantization, offload_ar=offload_ar, budget=budget,
        verify_hashes=verify_hashes, profile=profile_name, attention=attention,
        fuse=fuse, cuda_graph=cuda_graph, tf32=tf32, cudnn_bench=cudnn_bench,
        vae_frames=vae_frames, offline=offline, revision=revision,
        vae_revision=vae_revision, hf_token=hf_token, cache_dir=cache_dir,
        music_only=music_only,
    )

    try:
        progress(0.0, desc="Loading YuE2 (first run downloads and verifies weights)")
        TRANSCRIBER.close()  # transcription and generation must not share the GPU
        if music_only:
            progress(0.02, desc="Folding in the instrumental adapter")
        builder = _build_instrumental if music_only else _build_pipeline
        pipe = CACHE.get(_pipeline_key(kwargs), lambda: builder(kwargs))
    except Exception as error:
        return (None, "", "",
                f"Could not load the pipeline:\n{error}\n\n{traceback.format_exc(limit=6)}",
                scan_library())

    counters = {"abc": 0, "semantic": 0}

    # Progress updates arrive once per token; repainting on every one makes the
    # bar flicker, so the description is refreshed at most four times a second.
    last_paint = [0.0]

    def on_token(phase, _token):
        counters[phase] = counters.get(phase, 0) + 1
        now = time.perf_counter()
        if now - last_paint[0] < 0.25:
            return
        last_paint[0] = now
        total = counters["abc"] + counters["semantic"]
        progress(min(0.95, total / max(target_tokens, 1) * 0.9),
                 desc=f"score {counters['abc']} / song {counters['semantic']} tokens")

    directory = RUNS_DIR / f"song-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    try:
        with GENERATION_LOCK:
            progress(0.02, desc="Generating")
            pipe.generation_config = config
            result = pipe(
                style=style, lyrics=lyrics, cot=planning, seed=int(seed),
                abc=request.abc, cfg_scale=prompt_influence, id=request.id,
                abc_sampling=abc_sampling, semantic_sampling=sem_sampling,
                decode_full=bool(decode_full), on_token=on_token,
            )
        progress(0.97, desc="Writing audio")
        directory.mkdir(parents=True, exist_ok=True)
        audio_path = directory / "audio.flac"
        result.save(audio_path)
        if save_artifacts:
            result.save_artifacts(directory)
    except Exception as error:
        return None, "", "", f"Generation failed:\n{error}\n\n{traceback.format_exc(limit=8)}", scan_library()

    timing = dict(result.timing)
    timing["wall_seconds"] = round(time.perf_counter() - started, 2)
    timing["requested_seconds"] = round(tokens_to_seconds(target_tokens), 1)
    timing["peak_vram_gib"] = _peak_vram()
    timing["instrumental"] = music_only
    if music_only and getattr(pipe, "_instrumental_report", None):
        timing["instrumental_adapter"] = pipe._instrumental_report
    audio_seconds = len(result.audio) / result.sample_rate
    kind = "instrumental" if music_only else "song"
    status = (
        f"Done in {timing['e2e_seconds']:.0f}s -- **{audio_seconds:.1f}s of {kind}** "
        f"(asked for up to {timing['requested_seconds']:.0f}s), ODE steps {ode_steps}.\n\n"
        f"Artifacts: `{directory.relative_to(REPO_ROOT)}`"
        + ("" if save_artifacts else " (audio and score only; tick *Save full artifacts* in Advanced for the complete record)")
    )
    progress(1.0, desc="Complete")
    return (str(audio_path), result.abc or "", json.dumps(timing, indent=2, default=str),
            status, scan_library())


def _peak_vram():
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 3)
    except Exception:
        pass
    return 0.0


# --------------------------------------------------------------------------- #
# transcription handler
# --------------------------------------------------------------------------- #

def transcribe_reference(audio_path, melody_only, progress=gr.Progress()):
    """Turn an uploaded recording into an editable score plus matched lyrics."""
    if not audio_path:
        return "", "", "Upload a recording first."
    progress(0.1, desc="Loading the transcription model (first run downloads it)")
    try:
        result = TRANSCRIBER.transcribe(audio_path, melody_only=bool(melody_only))
    except TranscriptionError as error:
        return "", "", f"**Transcription failed.** {error}"
    except Exception as error:  # pragma: no cover - defensive
        return "", "", f"**Transcription failed.** {type(error).__name__}: {error}"
    finally:
        # Free the GPU before any generation; the two never run together.
        TRANSCRIBER.close()
    progress(1.0, desc="Transcribed")
    voices = sorted({line.split(":", 1)[1].strip() for line in result["abc"].splitlines()
                     if line.startswith("V:")})
    note = (
        f"Transcribed in {result['seconds']:.0f}s"
        + (f" ({', '.join(voices)})" if voices else "")
        + ". Review the score and align the lyrics with its sections before generating."
    )
    for warning in result["warnings"][:3]:
        note += f"\n\n> {warning}"
    return result["abc"], "", note


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def _song_controls(prefix, *, length=80, influence=1.0, planning="full", quality=16, seed=831001):
    """The friendly controls shared by Create, Cover and Edit.

    "Music only" exists because the base checkpoint has no instrumental mode:
    with a lyrics field it always sings. The toggle swaps the lyrics box for a
    section plan and selects the instrumental adapter at load time.
    """
    controls = {}
    controls[f"{prefix}_style"] = gr.Textbox(
        label="Style", value="", lines=3,
        placeholder="Genre, mood, vocal type, instruments, tempo - e.g. \"warm piano pop, female vocal, 88 BPM\"",
    )
    controls[f"{prefix}_music_only"] = gr.Checkbox(
        value=False, label="Music only (no vocals)",
        info="Uses the instrumental adapter. Leaving the lyrics empty selects this anyway.",
    )
    controls[f"{prefix}_lyrics"] = gr.Textbox(
        label="Lyrics", value="", lines=10,
        placeholder="[Verse]\nYour first lines...\n\n[Chorus]\nYour chorus...",
    )
    controls[f"{prefix}_section_plan"] = gr.Textbox(
        label="Section plan", value=DEFAULT_SECTION_PLAN, lines=7, visible=False,
        placeholder="[instrumental]\n[intro]\n[verse]\n[chorus]\n[outro]",
        info="Bracketed section tags only. The instrumental adapter reads this in place of lyrics.",
    )
    controls[f"{prefix}_docs"] = gr.Markdown(visible=False, value=(
        "Generating **music only**. The base checkpoint always sings, so this loads the "
        "community [instrumental adapter](https://huggingface.co/Mothersuperior/"
        "YuE2-instrumental-cot-full-loras) (~0.6 GB, fetched on first use) and reads the "
        "section plan above instead of lyrics. The adapter is **CC BY-NC 4.0**."
    ))
    with gr.Row():
        controls[f"{prefix}_length"] = gr.Slider(
            10, MAX_SECONDS, value=length, step=5, label="Song length (seconds)",
            info="A maximum: the model may finish earlier when the song ends naturally.",
        )
        controls[f"{prefix}_influence"] = gr.Slider(
            0.0, 20.0, value=influence, step=0.05, label="Prompt influence",
            info="1.0 follows the style text normally; higher clamps harder to it.",
        )
    with gr.Row():
        controls[f"{prefix}_planning"] = gr.Radio(
            PLANNING_CHOICES, value=planning, label="Symbolic planning",
        )
        controls[f"{prefix}_quality"] = gr.Radio(
            QUALITY_CHOICES, value=quality, label="Render quality",
            info="More ODE steps is slower and cleaner.",
        )
    controls[f"{prefix}_seed"] = gr.Number(value=seed, precision=0, label="Seed",
                                           info="Same seed and settings reproduce a song.")

    def _toggle_music_only(on):
        # Grey the lyrics out rather than hiding them, so the switch is obvious.
        return (
            gr.update(interactive=not on),
            gr.update(visible=bool(on)),
            gr.update(visible=not on),
        )

    controls[f"{prefix}_music_only"].change(
        _toggle_music_only,
        inputs=[controls[f"{prefix}_music_only"]],
        outputs=[controls[f"{prefix}_lyrics"], controls[f"{prefix}_section_plan"],
                 controls[f"{prefix}_docs"]],
        show_progress="hidden",
    )
    return controls


def _song_list(entries):
    """Body of every tab's song library. Runs inside ``gr.render``."""
    if not entries:
        gr.Markdown("*Nothing generated yet. Finished songs collect here, newest first.*")
        return
    for entry in entries:
        with gr.Row():
            with gr.Column(scale=2, min_width=170):
                heading = f"**{entry['name'].replace('song-', '')}**"
                if entry.get("seconds"):
                    heading += f" · {entry['seconds']:.0f}s"
                heading += f" · {entry['when']}"
                if entry.get("has_score"):
                    heading += " · score"
                gr.Markdown(heading + f"\n\n{entry['style']}")
            with gr.Column(scale=5):
                gr.Audio(value=entry["audio"], label=entry["snippet"], interactive=False)


def _library_block(state):
    """Register a dynamic song list bound to ``state`` for the current tab."""

    @gr.render(inputs=[state], show_progress="hidden")
    def _view(entries):
        _song_list(entries)

    return _view


def _cue_card(cue, outputs):
    """One anime cue: description, copyable prompt, and a send button."""
    with gr.Row():
        with gr.Column(scale=3):
            heading = f"**{cue['title']}** · *{cue['category']}*"
            if cue.get("seconds"):
                heading += f" · {int(cue['seconds'])}s"
            gr.Markdown(heading)
            if cue.get("use_when"):
                gr.Markdown(cue["use_when"])
            if cue.get("sound"):
                gr.Markdown(f"_{cue['sound']}_")
            gr.Textbox(value=cue["style"], label="Prompt", lines=3, interactive=False,
                       show_label=False, container=False)
        with gr.Column(scale=1, min_width=150):
            if cue.get("tags"):
                gr.Markdown("`" + "` `".join(str(tag) for tag in cue["tags"]) + "`")
            button = gr.Button("Send to generator", size="sm")
            button.click(lambda cue=cue: cue_to_generator(cue), inputs=None,
                         outputs=outputs, show_progress="hidden")


def _cue_block(state, outputs):
    """Register a dynamic anime-cue list bound to ``state`` for the current tab."""

    @gr.render(inputs=[state], show_progress="hidden")
    def _view(cues):
        if not cues:
            gr.Markdown("*No cues match. Try another category or clear the search.*")
            return
        for cue in cues:
            _cue_card(cue, outputs)

    return _view


def _sampling_controls(prefix, defaults, fields):
    """Return ``{f"{prefix}_{field}": component}`` keyed by handler parameter name."""
    controls = {}
    for row in (fields[:4], fields[4:]):
        if not row:
            continue
        with gr.Row():
            for name, label, lo, hi, step in row:
                controls[f"{prefix}_{name}"] = gr.Slider(
                    lo, hi, value=getattr(defaults, name), step=step, label=label,
                    elem_id=f"{prefix}-{name}",
                )
    return controls


def build_ui(args):
    abc_defaults = GenerationConfig().abc
    sem_defaults = GenerationConfig().semantic
    hardware = detect_hardware()
    has_cuda = bool(hardware.get("cuda_available"))
    cover_ready = dependencies_available()

    components = {}   # parameter name -> component, filled as the layout is built

    with gr.Blocks(title="YuE2 - Song Generation", fill_height=True) as demo:
        gr.Markdown(
            "# YuE2 - Song Generation\n"
            "Describe a style, add lyrics, and get a finished song with vocals. "
            "Or upload a recording and reimagine it.\n\n"
            f"Running on **{hardware.get('name')}**"
            + (f" ({hardware.get('total_memory_gib')} GiB)" if has_cuda else " (no CUDA device)")
        )

        with gr.Tabs(selected="create") as main_tabs:
            # ------------------------------------------------------------- #
            with gr.Tab("Create", id="create"):
                with gr.Row():
                    with gr.Column(scale=3):
                        create = _song_controls("create", length=80, quality=16)
                        gr.Examples(
                            examples=EXAMPLES,
                            inputs=[create["create_style"], create["create_lyrics"],
                                    create["create_length"], create["create_influence"],
                                    create["create_planning"], create["create_quality"],
                                    create["create_seed"]],
                            cache_examples=False,
                            label="Try an example",
                        )
                        with gr.Accordion("Score (editable)", open=False):
                            create_score = gr.Code(label="ABC score", language=None, lines=12, wrap_lines=True)
                            send_to_edit = gr.Button("Send this score to the Edit tab")
                        with gr.Accordion("Timings", open=False):
                            create_timings = gr.Code(label="Timings (seconds)", language="json", lines=10)
                    with gr.Column(scale=2):
                        create_audio = gr.Audio(label="Latest result", type="filepath", interactive=False)
                        create_status = gr.Markdown()
                        create_button = gr.Button("Generate song", variant="primary", size="lg")

                        with gr.Accordion("Loop tools - make an extendable cue", open=False):
                            gr.Markdown(
                                "YuE2 cannot place a loop point: the score is prompt text only, so "
                                "sections and tempo come back approximate. This detects the beat grid "
                                "of the *rendered audio*, snaps the loop to whole bars, and writes a "
                                "seamless loop plus an extended track next to the cue."
                            )
                            with gr.Row():
                                loop_start = gr.Slider(0, MAX_SECONDS, value=15, step=0.5,
                                                       label="Loop starts (s)")
                                loop_end = gr.Slider(0, MAX_SECONDS, value=25, step=0.5,
                                                     label="Loop ends (s)")
                            with gr.Row():
                                loop_repeats = gr.Slider(1, 8, value=3, step=1, label="Loop repeats")
                                loop_fade = gr.Slider(0, 300, value=60, step=5, label="Crossfade (ms)")
                                loop_snap = gr.Checkbox(value=True, label="Snap to bar lines")
                            loop_button = gr.Button("Build loop from the latest result")
                            loop_status = gr.Markdown()
                            with gr.Row():
                                loop_audio = gr.Audio(label="Seamless loop", type="filepath",
                                                      interactive=False)
                                loop_extended = gr.Audio(label="Extended track", type="filepath",
                                                         interactive=False)

                        with gr.Accordion("Clear output", open=False):
                            output_status = gr.Markdown(library_status())
                            clear_list_button = gr.Button("Clear the list (keeps every file)")
                            confirm_delete = gr.Checkbox(
                                value=False,
                                label="I understand the buttons below delete files from disk",
                            )
                            with gr.Row():
                                delete_audio_button = gr.Button("Delete audio files (keep scores)")
                                delete_all_button = gr.Button("Delete everything in runs/",
                                                              variant="stop")

                        gr.Markdown("### Your songs\n*Newest first.*")
                        create_library = gr.State(scan_library())
                        _library_block(create_library)

            # ------------------------------------------------------------- #
            with gr.Tab("Cover from a recording"):
                gr.Markdown(
                    "Upload a song you have the rights to use. YuE2 will transcribe its melody "
                    "into an editable score, then render that melody in a style you choose. "
                    "YuE2 itself never takes audio as input - the score is what carries the tune."
                )
                if not cover_ready:
                    gr.Markdown(
                        "> **Transcription is unavailable.** Install the audio helpers with "
                        "`uv sync --extra cover`, then restart the app."
                    )
                with gr.Row():
                    with gr.Column(scale=3):
                        cover_audio_in = gr.Audio(label="Recording to cover", type="filepath",
                                                  sources=["upload"])
                        with gr.Row():
                            cover_melody_only = gr.Checkbox(
                                value=True, label="Melody only",
                                info="Recommended for changing style: it frees the accompaniment.",
                            )
                            transcribe_button = gr.Button("1. Transcribe melody", variant="secondary")
                        transcribe_status = gr.Markdown()
                        cover = _song_controls("cover", length=80, quality=16)
                        cover_abc = gr.Code(
                            label="Transcribed score (review or edit before generating)",
                            language=None, lines=12, wrap_lines=True,
                        )
                        with gr.Accordion("Score used", open=False):
                            cover_score_out = gr.Code(label="ABC score", language=None, lines=12,
                                                      wrap_lines=True)
                        with gr.Accordion("Timings", open=False):
                            cover_timings = gr.Code(label="Timings (seconds)", language="json", lines=10)
                    with gr.Column(scale=2):
                        cover_audio_out = gr.Audio(label="Latest cover", type="filepath", interactive=False)
                        cover_status = gr.Markdown()
                        cover_button = gr.Button("2. Generate cover", variant="primary", size="lg")
                        gr.Markdown("### Your songs\n*Newest first.*")
                        cover_library = gr.State(scan_library())
                        _library_block(cover_library)

            # ------------------------------------------------------------- #
            with gr.Tab("Edit a song"):
                gr.Markdown(
                    "YuE2 writes the composition as readable ABC notation. Change notes, chords, "
                    "tempo or section order here, then render the revised version. The same seed "
                    "keeps the take otherwise close, but an edit always renders a new recording."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            edit_run = gr.Dropdown(choices=list_recent_runs(), value=None,
                                                   label="Load a saved song", scale=4)
                            edit_load = gr.Button("Load", scale=1)
                            edit_refresh = gr.Button("Refresh list", scale=1)
                        edit_load_status = gr.Markdown()
                        edit = _song_controls("edit", length=80, quality=16)
                        gr.Markdown(
                            "*Leave the melody intact and change chords, or rewrite the melody entirely.*"
                        )
                        edit_abc = gr.Code(
                            label="Score (ABC) - edit this", language=None, lines=16, wrap_lines=True,
                        )
                        with gr.Accordion("New score", open=False):
                            edit_score_out = gr.Code(label="ABC score", language=None, lines=12,
                                                     wrap_lines=True)
                        with gr.Accordion("Timings", open=False):
                            edit_timings = gr.Code(label="Timings (seconds)", language="json", lines=10)
                    with gr.Column(scale=2):
                        edit_audio = gr.Audio(label="Latest edit", type="filepath", interactive=False)
                        edit_status = gr.Markdown()
                        edit_button = gr.Button("Render edited version", variant="primary", size="lg")
                        gr.Markdown("### Your songs\n*Newest first.*")
                        edit_library = gr.State(scan_library())
                        _library_block(edit_library)

            # ------------------------------------------------------------- #
            with gr.Tab("Anime sounds", id="anime"):
                cue_catalogue, cue_categories, cue_defaults, cue_note = load_cue_library()
                gr.Markdown(
                    "### Anime background music\n"
                    "Scene-by-scene underscore cues for animation: short, instrumental, and written to "
                    "sit under dialogue. Pick a scene, hit **Send to generator**, then press *Generate "
                    "song* on the Create tab. Every cue is 30 seconds and loads the instrumental adapter.\n\n"
                    f"Editing `{ANIME_CUES_PATH.name}`? Press **Reload from disk** - no restart needed."
                )
                with gr.Row():
                    cue_category = gr.Dropdown(
                        [ALL_CATEGORIES, *cue_categories], value=ALL_CATEGORIES,
                        label="Scene type", scale=4,
                    )
                    cue_search = gr.Textbox(
                        label="Search", scale=4,
                        placeholder="e.g. sad, chase, piano, rain, taiko",
                    )
                    cue_reload = gr.Button("Reload from disk", scale=1)
                cue_status = gr.Markdown(cue_note)
                cue_outputs = [
                    create["create_style"], create["create_length"], create["create_influence"],
                    create["create_music_only"], create["create_section_plan"],
                    create["create_lyrics"], create["create_seed"], cue_status, main_tabs,
                ]
                cue_list = gr.State(filter_cues(cue_catalogue))
                _cue_block(cue_list, cue_outputs)

            # ------------------------------------------------------------- #
            with gr.Tab("Advanced"):
                gr.Markdown(
                    "Everything below overrides the friendly controls. Leave a value at its default "
                    "unless you know why you are changing it."
                )
                with gr.Accordion("Length and rendering overrides", open=True):
                    with gr.Row():
                        custom_max_tokens = gr.Number(
                            value=0, precision=0, label="Custom semantic token budget (0 = use Song length)",
                            info=f"{FRAMES_PER_SECOND} tokens = 1 second. Max 9000 = {MAX_SECONDS}s.",
                        )
                        custom_ode_steps = gr.Number(
                            value=0, precision=0, label="Custom ODE steps (0 = use Render quality)",
                        )
                    decode_full = gr.Checkbox(value=False, label="Decode in one VAE pass",
                                              info="Faster but uses more VRAM; off tiles the decode.")

                with gr.Accordion("Planner sampling (the ABC score stage)", open=False):
                    abc_controls = _sampling_controls("abc", abc_defaults, ABC_FIELDS)
                with gr.Accordion("Song sampling (the audio token stage)", open=False):
                    sem_controls = _sampling_controls("sem", sem_defaults, SEMANTIC_FIELDS)
                    gr.Markdown(
                        "*Max tokens for this stage comes from **Song length** "
                        "(or the custom budget above), so it is not repeated here.*"
                    )

                with gr.Accordion("Backend and performance", open=False):
                    gr.Markdown(
                        "*reference* reproduces the pinned release flags exactly. "
                        "*balanced* fuses QKV/gate-up projections (same arithmetic, fewer launches). "
                        "*fast* additionally enables TF32 and cuDNN autotuning, which **change results**."
                    )
                    with gr.Row():
                        profile_name = gr.Dropdown(PROFILE_NAMES, value=args.profile, label="Performance profile")
                        attention = gr.Dropdown(["auto", "flash", "cudnn", "sdpa"], value="auto",
                                                label="Attention backend")
                        vae_frames = gr.Number(value=1024, precision=0, label="VAE core frames (0 = auto)",
                                               info="Smaller tiles reduce peak VRAM during decode.")
                    with gr.Row():
                        fuse = gr.Checkbox(value=True, label="Fuse projections")
                        cuda_graph = gr.Checkbox(value=True, label="CUDA graphs")
                        tf32 = gr.Checkbox(value=False, label="TF32 matmul")
                        cudnn_bench = gr.Checkbox(value=False, label="cuDNN autotune")
                    with gr.Row():
                        backend = gr.Dropdown(["torch", "torch-eager", "vllm"], value=args.backend,
                                              label="AR backend")
                        quantization = gr.Dropdown(["none", "fp8"], value="none", label="Quantization",
                                                   info="FP8 needs compute capability >= 8.9; unavailable on RTX 3090.")
                        device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
                        budget = gr.Slider(4, 48, value=24, step=1, label="Memory budget (GiB)")
                    with gr.Row():
                        offload_ar = gr.Checkbox(value=False, label="Offload AR")
                        verify_hashes = gr.Checkbox(value=True, label="Verify weight hashes")
                        offline = gr.Checkbox(value=bool(args.offline), label="Offline (local files only)")
                        save_artifacts = gr.Checkbox(value=True, label="Save full artifacts")

                with gr.Accordion("Model sources and output", open=False):
                    with gr.Row():
                        model = gr.Textbox(value=str(args.model), label="Model repo or directory")
                        vae = gr.Textbox(value=str(args.vae), label="VAE repo or directory")
                    with gr.Row():
                        revision = gr.Textbox(value="", label="Model revision (optional)")
                        vae_revision = gr.Textbox(value="", label="VAE revision (optional)")
                    with gr.Row():
                        hf_token = gr.Textbox(value="", label="Hugging Face token (optional)",
                                              type="password",
                                              info="Only needed for gated or private repositories.")
                        cache_dir = gr.Textbox(value="", label="Hub cache directory (optional)",
                                               info="Where a remote repo is downloaded; empty uses HF_HOME.")
                    with gr.Row():
                        unload_button = gr.Button("Unload pipeline (free GPU memory)")
                        transcribe_unload = gr.Button("Unload transcriber (free GPU memory)")

            # ------------------------------------------------------------- #
            with gr.Tab("Hardware"):
                gr.Markdown(hardware_report())
                gr.Markdown(
                    "### RTX 3090 notes\n"
                    "- Compute capability 8.6, BF16 supported, FlashAttention varlen available.\n"
                    "- **FP8 is not available** (requires >= 8.9), so `quantization=none` is the only "
                    "working setting.\n"
                    "- 24 GiB fits the full BF16 3B checkpoint plus KV cache; decode tiling is optional.\n"
                    "- Transcription and generation run one after the other and release the GPU between "
                    "steps, so a cover needs the transcriber's memory only while transcribing.\n"
                )

        # Bind each tab's controls to the one handler. Every mapping must cover
        # every parameter, so a mis-ordered or missing component cannot silently
        # shift the arguments.
        components.update({
            "model": model, "vae": vae, "device": device, "backend": backend,
            "quantization": quantization, "offload_ar": offload_ar, "budget": budget,
            "profile_name": profile_name, "attention": attention, "fuse": fuse,
            "cuda_graph": cuda_graph, "tf32": tf32, "cudnn_bench": cudnn_bench,
            "vae_frames": vae_frames, "verify_hashes": verify_hashes, "offline": offline,
            "revision": revision, "vae_revision": vae_revision, "hf_token": hf_token,
            "cache_dir": cache_dir, "save_artifacts": save_artifacts,
            "custom_ode_steps": custom_ode_steps, "custom_max_tokens": custom_max_tokens,
            "decode_full": decode_full,
            **abc_controls, **sem_controls,
        })

        def mapping(song, abc_component):
            merged = dict(components)
            merged.update({
                "style": song[f"{song['_prefix']}_style"],
                "lyrics": song[f"{song['_prefix']}_lyrics"],
                "music_only": song[f"{song['_prefix']}_music_only"],
                "section_plan": song[f"{song['_prefix']}_section_plan"],
                "length_seconds": song[f"{song['_prefix']}_length"],
                "prompt_influence": song[f"{song['_prefix']}_influence"],
                "planning": song[f"{song['_prefix']}_planning"],
                "quality": song[f"{song['_prefix']}_quality"],
                "seed": song[f"{song['_prefix']}_seed"],
                "abc_text": abc_component,
            })
            parameters = [n for n in inspect.signature(generate).parameters if n != "progress"]
            missing = sorted(set(parameters) - set(merged))
            assert not missing, f"UI is missing controls for: {missing}"
            extra = sorted(set(merged) - set(parameters))
            assert not extra, f"UI has controls with no handler parameter: {extra}"
            return [merged[name] for name in parameters]

        create_song = dict(create, _prefix="create")
        cover_song = dict(cover, _prefix="cover")
        edit_song = dict(edit, _prefix="edit")
        empty_abc = gr.State("")

        handlers = [
            (create_button, mapping(create_song, empty_abc), create_audio, create_score,
             create_timings, create_status, create_library),
            (cover_button, mapping(cover_song, cover_abc), cover_audio_out, cover_score_out,
             cover_timings, cover_status, cover_library),
            (edit_button, mapping(edit_song, edit_abc), edit_audio, edit_score_out,
             edit_timings, edit_status, edit_library),
        ]
        for button, inputs, audio_out, score_out, timing_out, status_out, library_out in handlers:
            # "minimal" keeps a single small indicator. The default also paints a
            # bar over every output component, which is the double bar users see.
            button.click(generate, inputs=inputs,
                         outputs=[audio_out, score_out, timing_out, status_out, library_out],
                         show_progress="minimal")

        transcribe_button.click(
            transcribe_reference,
            inputs=[cover_audio_in, cover_melody_only],
            outputs=[cover_abc, cover["cover_lyrics"], transcribe_status],
            show_progress="minimal",
        )
        edit_load.click(load_run, inputs=[edit_run],
                        outputs=[edit_abc, edit["edit_style"], edit["edit_lyrics"], edit_load_status],
                        show_progress="hidden")
        edit_refresh.click(lambda: gr.update(choices=list_recent_runs()), inputs=None,
                           outputs=[edit_run], show_progress="hidden")
        send_to_edit.click(
            send_score_to_edit,
            inputs=[create_score, create["create_style"], create["create_lyrics"]],
            outputs=[edit_abc, edit["edit_style"], edit["edit_lyrics"], edit_load_status],
            show_progress="hidden",
        )
        unload_button.click(unload, inputs=None, outputs=create_status, show_progress="hidden")
        transcribe_unload.click(unload_transcriber, inputs=None, outputs=transcribe_status)

        # Anime cue browser.
        for trigger in (cue_search.submit, cue_category.change, cue_reload.click):
            trigger(reload_cues, inputs=[cue_category, cue_search],
                    outputs=[cue_list, cue_status, cue_category], show_progress="hidden")

        # Loop builder, driven by whatever is in the latest-result player.
        loop_button.click(
            build_cue_loop,
            inputs=[create_audio, loop_start, loop_end, loop_repeats, loop_fade, loop_snap],
            outputs=[loop_audio, loop_extended, loop_status],
            show_progress="minimal",
        )

        # Output housekeeping. Destructive actions also refresh every tab's list.
        clear_list_button.click(clear_library_list, inputs=None,
                                outputs=[create_library, cover_library, edit_library, output_status],
                                show_progress="hidden")
        delete_audio_button.click(
            delete_audio_files, inputs=[confirm_delete],
            outputs=[create_library, cover_library, edit_library, output_status],
            show_progress="minimal",
        )
        delete_all_button.click(
            delete_all_runs, inputs=[confirm_delete],
            outputs=[create_library, cover_library, edit_library, output_status],
            show_progress="minimal",
        )

    return demo


def unload():
    CACHE.close()
    return "Pipeline unloaded and GPU memory released."


# --------------------------------------------------------------------------- #
# loop builder
# --------------------------------------------------------------------------- #

def build_cue_loop(audio_path, loop_start, loop_end, repeats, fade_ms, snap,
                   progress=gr.Progress()):
    """Turn the latest result into a seamless loop plus an extended track."""
    from yue2 import loops as loop_tools
    import soundfile as sf

    if not audio_path:
        return None, None, "Generate a song first - there is nothing to loop yet."
    source = Path(audio_path)
    if not source.is_file():
        return None, None, f"Could not find `{source}`."
    if not loop_tools.dependencies_available():
        return None, None, ("Loop tools need the `loop` group. Install it with "
                            "`uv sync --extra loop`, then restart the app.")
    try:
        audio, sample_rate = sf.read(source, dtype="float32")
    except Exception as error:
        return None, None, f"Could not read the audio: {error}"

    progress(0.1, desc="Detecting the beat grid")
    try:
        grid = loop_tools.analyse(audio, sample_rate)
    except loop_tools.LoopError as error:
        return None, None, f"Beat detection failed. {error}"
    progress(0.5, desc="Cutting and crossfading")
    try:
        result = loop_tools.build_loop(
            audio, sample_rate, loop_start=float(loop_start), loop_end=float(loop_end),
            repeats=int(repeats), crossfade_ms=float(fade_ms), grid=grid, snap=bool(snap),
        )
    except loop_tools.LoopError as error:
        return None, None, f"Could not build the loop. {error}"

    loop_path = source.parent / "loop.wav"
    extended_path = source.parent / "extended.flac"
    loop_tools.save_wav(loop_path, result.loop, sample_rate)
    loop_tools.save_flac(extended_path, result.extended, sample_rate)

    report = result.report
    snapped = report.get("snapped_to_grid")
    lines = [f"Tempo **{grid.tempo:.1f} BPM** ({grid.bar_seconds:.2f}s per bar)."]
    if snapped:
        lines.append(
            f"Loop snapped from {report['requested']['start']:.1f}-{report['requested']['end']:.1f}s "
            f"to **{snapped['start']:.2f}-{snapped['end']:.2f}s** "
            f"({snapped['bars']} bars, {snapped['loop_seconds']:.2f}s)."
        )
    lines.append(
        f"Extended track **{report['extended_seconds']:.1f}s** "
        f"= {report['intro_seconds']:.1f}s intro + {report['loop_seconds']:.2f}s loop x{repeats} "
        f"+ {report['outro_seconds']:.1f}s outro."
    )
    lines.append(f"Written next to the cue: `{loop_path.name}` and `{extended_path.name}`.")
    progress(1.0, desc="Loops written")
    return str(loop_path), str(extended_path), "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# output housekeeping
# --------------------------------------------------------------------------- #

def library_usage():
    """Disk and count summary for the runs directory."""
    entries = scan_library(limit=10_000)
    total = 0
    for entry in entries:
        try:
            total += Path(entry["audio"]).stat().st_size
        except OSError:
            pass
    mb = total / 2**20
    return entries, mb


def library_status():
    entries, mb = library_usage()
    return f"`runs/` holds **{len(entries)} songs**, about **{mb:.0f} MB** of audio."


def clear_library_list():
    """Empty the on-screen library without touching disk."""
    return [], [], [], "List cleared. The files are untouched - press Reload to bring them back."


def _require_confirmation(confirmed):
    if not confirmed:
        raise gr.Error("Tick the confirmation box first - these buttons delete files from disk.")


def delete_audio_files(confirmed):
    """Remove every generated audio.flac, keeping scores and metadata."""
    _require_confirmation(confirmed)
    removed = freed = 0
    for entry in scan_library(limit=10_000):
        audio = Path(entry["audio"])
        try:
            freed += audio.stat().st_size
            audio.unlink()
            removed += 1
        except OSError:
            pass
    library = scan_library()
    return (library, library, library,
            f"Deleted **{removed}** audio files, freeing **{freed / 2**20:.0f} MB**. "
            f"Scores, timings and metadata were kept.")


def delete_all_runs(confirmed):
    """Remove every run directory."""
    _require_confirmation(confirmed)
    import shutil

    removed = freed = 0
    if RUNS_DIR.is_dir():
        for directory in list(RUNS_DIR.iterdir()):
            if not directory.is_dir():
                continue
            size = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
            try:
                shutil.rmtree(directory)
                removed += 1
                freed += size
            except OSError:
                pass
    return [], [], [], (f"Deleted **{removed}** runs, freeing **{freed / 2**20:.0f} MB**. "
                        f"`runs/` is now empty.")


def send_score_to_edit(score, style, lyrics):
    """Copy the latest Create result into the Edit tab."""
    score = (score or "").strip()
    if not score:
        return (gr.update(), gr.update(), gr.update(),
                "Generate a song with symbolic planning first - `cot=off` produces no score to edit.")
    return score, style or "", lyrics or "", "Score, style and lyrics copied to the Edit tab below."


def unload_transcriber():
    TRANSCRIBER.close()
    return "Transcriber unloaded and GPU memory released."


def port_available(host, port):
    """True if ``port`` can be bound on ``host`` right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, int(port)))
        except OSError:
            return False
    return True


def find_free_port(host, start=DEFAULT_PORT, attempts=20):
    """First bindable port at or after ``start``, or None if all are taken."""
    for candidate in range(start, start + attempts):
        if port_available(host, candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="YuE2 Gradio front end")
    parser.add_argument("--model", default=os.environ.get("YUE2_MODEL", str(REPO_ROOT / "models" / "YuE2-3B")))
    parser.add_argument("--vae", default=os.environ.get("YUE2_VAE", str(REPO_ROOT / "models" / "YuE2-Vae")))
    parser.add_argument("--backend", default="torch", choices=("torch", "torch-eager", "vllm"))
    parser.add_argument("--profile", default="balanced", choices=PROFILE_NAMES)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None,
                        help=f"Port to serve on. Default: the first free port from {DEFAULT_PORT} upward.")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--offline", action="store_true", help="start with 'offline' pre-ticked")
    args = parser.parse_args()

    if args.port is None:
        # Another server (or a previous run) may already hold the default port.
        # Only an explicit --port is treated as a hard requirement.
        port = find_free_port(args.host, DEFAULT_PORT)
        if port is None:
            raise SystemExit(
                f"No free port in {DEFAULT_PORT}-{DEFAULT_PORT + 19} on {args.host}; pass --port."
            )
        if port != DEFAULT_PORT:
            print(f"Port {DEFAULT_PORT} is in use; serving on {port} instead.", file=sys.stderr)
    else:
        port = args.port
        if not port_available(args.host, port):
            raise SystemExit(
                f"Port {port} is already in use. Stop the process holding it, "
                f"or omit --port to let the app choose the next free one."
            )

    build_ui(args).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=port, share=args.share,
        show_error=True, allowed_paths=[str(RUNS_DIR)],
    )


if __name__ == "__main__":
    main()
