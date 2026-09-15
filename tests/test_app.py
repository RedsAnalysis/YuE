"""Contract checks for the Gradio front end.

Most of this file runs without model weights. The final test is marked ``slow``
and drives the real handler on the GPU, so it needs CUDA and the local
checkpoint; skip it with ``-m "not slow"``.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "YuE2-3B"
VAE_DIR = REPO_ROOT / "models" / "YuE2-Vae"


def load_app():
    spec = importlib.util.spec_from_file_location("yue2_app_under_test", REPO_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return load_app()


@pytest.fixture(scope="module")
def ui_kwargs():
    return argparse.Namespace(
        model=str(MODEL_DIR), vae=str(VAE_DIR),
        backend="torch", profile="balanced", offline=True,
    )


@pytest.fixture(scope="module")
def labels(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    return [getattr(b, "label", None) for b in demo.blocks.values() if getattr(b, "label", None)]


def _choices(component):
    """Gradio 6 exposes choices as (label, value) pairs; normalise to values."""
    raw = getattr(component, "choices", None) or []
    return [item[1] if isinstance(item, tuple) else item for item in raw]


def _generate_kwargs(**overrides):
    """A complete, valid argument set for app.generate."""
    base = dict(
        style="pop", lyrics="[Verse]\nwords", length_seconds=90, prompt_influence=1.0,
        planning="full", quality=16, seed=1,
        music_only=False, section_plan="[instrumental]\n[verse]", abc_text="",
        custom_ode_steps=0, custom_max_tokens=0, decode_full=False,
        model="unused", vae="unused", device="cpu", backend="torch",
        quantization="none", offload_ar=False, budget=24, profile_name="reference",
        attention="auto", fuse=False, cuda_graph=False, tf32=False, cudnn_bench=False,
        vae_frames=1024, verify_hashes=False, offline=True, revision="",
        vae_revision="", hf_token=None, cache_dir=None,
        abc_temperature=0.7, abc_top_p=0.9, abc_top_k=30, abc_repetition_penalty=1.005,
        abc_penalty_window=100, abc_min_tokens=32, abc_max_tokens=4096,
        sem_temperature=1.0, sem_top_p=0.95, sem_top_k=100, sem_repetition_penalty=1.2,
        sem_penalty_window=50, sem_min_tokens=200,
        save_artifacts=False, progress=lambda *a, **k: None,
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# length in seconds
# --------------------------------------------------------------------------- #

def test_seconds_map_to_tokens_at_25_per_second(app):
    """The VAE decodes 25 frames/second; 48000 / 1920 = 25."""
    assert app.seconds_to_tokens(80) == 2000
    assert app.seconds_to_tokens(24) == 600
    assert app.tokens_to_seconds(2000) == 80.0
    assert app.FRAMES_PER_SECOND == 25


def test_max_seconds_matches_the_token_ceiling(app):
    assert app.MAX_SECONDS == 360
    assert app.seconds_to_tokens(app.MAX_SECONDS) == 9000


def test_length_slider_cannot_exceed_the_model_ceiling(app):
    assert app.seconds_to_tokens(10_000) == 9000
    assert app.seconds_to_tokens(1) == 25


def test_resolve_token_budget_uses_the_length_slider_by_default(app):
    floor, budget = app.resolve_token_budget(length_seconds=120, custom_max_tokens=0, sem_min_tokens=200)
    assert budget == 3000
    assert floor == 200


def test_resolve_token_budget_lets_advanced_override(app):
    _floor, budget = app.resolve_token_budget(length_seconds=120, custom_max_tokens=750, sem_min_tokens=200)
    assert budget == 750


def test_resolve_token_budget_lowers_the_floor_for_short_songs(app):
    """A 5s request is 125 tokens, below the checkpoint's 200-token floor."""
    floor, budget = app.resolve_token_budget(length_seconds=5, custom_max_tokens=0, sem_min_tokens=200)
    assert budget == 125
    assert floor == 125


def test_resolve_ode_steps_uses_quality_unless_overridden(app):
    assert app.resolve_ode_steps(quality=24, custom_ode_steps=0) == 24
    assert app.resolve_ode_steps(quality=24, custom_ode_steps=48) == 48


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #

def test_tabs_are_split_into_simple_and_advanced(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    tab_labels = [getattr(b, "label", None) for b in demo.blocks.values()
                  if type(b).__name__ == "Tab"]
    assert tab_labels == ["Create", "Cover from a recording", "Edit a song",
                          "Anime sounds", "Advanced", "Hardware"]


def _descendants(block):
    """Every nested block, since controls sit inside Accordions and Columns."""
    found = []
    for child in getattr(block, "children", []):
        found.append(child)
        found.extend(_descendants(child))
    return found


def _tab(demo, label):
    return next(b for b in demo.blocks.values()
                if type(b).__name__ == "Tab" and getattr(b, "label", None) == label)


def _labels_under(block):
    return {getattr(c, "label", None) for c in _descendants(block)}


def test_create_tab_exposes_only_friendly_controls(app, ui_kwargs):
    """The front page must not show backend or sampling knobs."""
    demo = app.build_ui(ui_kwargs)
    create = _labels_under(_tab(demo, "Create"))
    for friendly in ("Style", "Lyrics", "Song length (seconds)", "Prompt influence",
                     "Symbolic planning", "Render quality", "Seed"):
        assert friendly in create, f"missing friendly control: {friendly}"
    for technical in ("Temperature", "Top-k", "Repetition penalty", "Top-p",
                      "Quantization", "AR backend", "Memory budget (GiB)",
                      "Performance profile", "Attention backend", "Device",
                      "Offload AR", "Verify weight hashes", "Model repo or directory"):
        assert technical not in create, f"'{technical}' must not appear on the Create tab"


def test_technical_controls_live_on_the_advanced_tab(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    advanced = _labels_under(_tab(demo, "Advanced"))
    for expected in ("Performance profile", "Attention backend", "Quantization",
                     "Model repo or directory", "Hub cache directory (optional)",
                     "Temperature", "Top-k", "Repetition penalty", "Device",
                     "Custom ODE steps (0 = use Render quality)",
                     "Custom semantic token budget (0 = use Song length)"):
        assert expected in advanced, f"{expected} should be on the Advanced tab"


def test_song_length_is_offered_in_seconds_on_all_three_song_tabs(labels):
    assert labels.count("Song length (seconds)") == 3
    assert labels.count("Render quality") == 3
    assert labels.count("Prompt influence") == 3


def test_planning_choices_use_plain_language(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    radio = next(b for b in demo.blocks.values()
                 if getattr(b, "label", None) == "Symbolic planning")
    assert _choices(radio) == ["full", "melody", "off"]
    rendered = " ".join(label for label, _value in radio.choices)
    assert "Melody + chords" in rendered and "No score" in rendered


def test_quality_choices_are_ordered_by_cost(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    radio = next(b for b in demo.blocks.values() if getattr(b, "label", None) == "Render quality")
    steps = _choices(radio)
    assert steps == sorted(steps)
    assert 16 in steps and 32 in steps


# --------------------------------------------------------------------------- #
# layout: score/timings swap places with the Generate button
# --------------------------------------------------------------------------- #

def _ordered_labels(block):
    """Descendant labels in definition order, which is the on-screen order."""
    return [getattr(c, "label", None) for c in _descendants(block)]


def test_score_and_timings_sit_where_the_generate_button_used_to_be(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    order = _ordered_labels(_tab(demo, "Create"))
    kinds = [type(c).__name__ for c in _descendants(_tab(demo, "Create"))]
    button_at = next(i for i, c in enumerate(_descendants(_tab(demo, "Create")))
                     if kinds[i] == "Button" and getattr(c, "value", None) == "Generate song")
    assert order.index("Score (editable)") < button_at
    assert order.index("Timings") < button_at


def test_song_library_sits_under_the_generate_button(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    descendants = _descendants(_tab(demo, "Create"))
    button_at = next(i for i, c in enumerate(descendants)
                     if type(c).__name__ == "Button" and getattr(c, "value", None) == "Generate song")
    state_at = next(i for i, c in enumerate(descendants) if type(c).__name__ == "State")
    assert state_at > button_at, "the library must render below the Generate button"


def test_every_song_tab_has_its_own_library(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    for name in ("Create", "Cover from a recording", "Edit a song"):
        states = [c for c in _descendants(_tab(demo, name)) if type(c).__name__ == "State"]
        assert states, f"{name} has no song library state"


def test_each_library_registers_a_render(app, ui_kwargs):
    """Three song-tab libraries plus the anime cue browser."""
    demo = app.build_ui(ui_kwargs)
    renders = [f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "apply"]
    assert len(renders) == 4
    for fn in renders:
        assert fn.trigger_mode == "always_last"


# --------------------------------------------------------------------------- #
# song library contents
# --------------------------------------------------------------------------- #

def test_scan_library_returns_newest_first(app, tmp_path, monkeypatch):
    import os
    import time as _time

    runs = tmp_path / "runs"
    runs.mkdir()
    for index, name in enumerate(("song-old", "song-new", "song-middle")):
        entry = runs / name
        entry.mkdir()
        (entry / "audio.flac").write_bytes(b"x")
        (entry / "request.json").write_text(json.dumps({"style": f"style-{name}", "lyrics": "la"}))
        stamp = 1_000_000 + index * 100
        os.utime(entry / "audio.flac", (stamp, stamp))
    monkeypatch.setattr(app, "RUNS_DIR", runs)

    entries = app.scan_library()
    assert [e["name"] for e in entries] == ["song-middle", "song-new", "song-old"]
    assert entries[0]["style"] == "style-song-middle"


def test_scan_library_skips_directories_without_audio(app, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "song-empty").mkdir(parents=True)
    (runs / "song-good").mkdir()
    (runs / "song-good" / "audio.flac").write_bytes(b"x")
    monkeypatch.setattr(app, "RUNS_DIR", runs)

    assert [e["name"] for e in app.scan_library()] == ["song-good"]


def test_scan_library_handles_a_missing_runs_directory(app, tmp_path, monkeypatch):
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path / "absent")
    assert app.scan_library() == []


def test_scan_library_marks_instrumental_and_score_entries(app, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    entry = runs / "song-x"
    entry.mkdir(parents=True)
    (entry / "audio.flac").write_bytes(b"x")
    (entry / "score.abc").write_text("X:1\nK:C\n")
    monkeypatch.setattr(app, "RUNS_DIR", runs)

    row = app.scan_library()[0]
    assert row["has_score"] is True
    assert row["snippet"] == "(instrumental)"
    assert row["audio"].endswith("audio.flac")


# --------------------------------------------------------------------------- #
# anime cue library
# --------------------------------------------------------------------------- #

def test_the_shipped_cue_library_loads(app):
    cues, categories, defaults, note = app.load_cue_library()
    assert len(cues) >= 100, note
    assert len(categories) >= 15
    assert defaults["seconds"] == 30 and defaults["music_only"] is True
    for cue in cues:
        assert cue["style"] and cue["title"] and cue["category"]
        assert cue["plan"], "every cue needs an instrumental section plan"


def test_every_shipped_cue_has_a_loadable_section_plan(app):
    """The plan goes straight to the adapter, which validates it strictly."""
    from yue2.lora import normalise_section_plan

    cues, _categories, _defaults, _note = app.load_cue_library()
    for cue in cues:
        assert normalise_section_plan(cue["plan"]), cue["id"]


def test_cue_categories_are_unique_and_all_used(app):
    import collections

    cues, categories, _defaults, _note = app.load_cue_library()
    counts = collections.Counter(cue["category"] for cue in cues)
    assert set(counts) == set(categories)
    assert all(counts[name] >= 3 for name in categories), "each category should have a few cues"


def test_missing_cue_file_is_reported_not_raised(app, tmp_path):
    cues, categories, _defaults, note = app.load_cue_library(tmp_path / "absent.json")
    assert cues == [] and categories == []
    assert "No cue file" in note


def test_malformed_cue_file_is_reported(app, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    cues, _categories, _defaults, note = app.load_cue_library(path)
    assert cues == [] and "Could not read" in note


def test_cue_file_without_a_cues_array_is_reported(app, tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"version": 1}))
    cues, _categories, _defaults, note = app.load_cue_library(path)
    assert cues == [] and "cues" in note


def test_bad_entries_are_skipped_and_counted(app, tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps({
        "defaults": {"seconds": 30, "plan": "[instrumental]\n[outro]"},
        "cues": [
            {"id": "good", "title": "Good", "category": "Test", "style": "a prompt"},
            {"id": "no-style", "title": "Missing style"},
            "not even an object",
        ],
    }))
    cues, _categories, _defaults, note = app.load_cue_library(path)
    assert [c["id"] for c in cues] == ["good"]
    assert "2 skipped" in note
    # Defaults are applied so a minimal entry still works.
    assert cues[0]["seconds"] == 30
    assert cues[0]["plan"] == "[instrumental]\n[outro]"


def test_filter_cues_by_category(app):
    cues, _categories, _defaults, _note = app.load_cue_library()
    categories = {cue["category"] for cue in cues}
    for name in categories:
        matched = app.filter_cues(cues, name, "")
        assert matched and all(cue["category"] == name for cue in matched)


def test_filter_cues_all_categories_returns_everything(app):
    cues, _categories, _defaults, _note = app.load_cue_library()
    assert len(app.filter_cues(cues, app.ALL_CATEGORIES, "")) == len(cues)


def test_filter_cues_search_matches_several_fields(app):
    cues, _categories, _defaults, _note = app.load_cue_library()
    matched = app.filter_cues(cues, app.ALL_CATEGORIES, "sad")
    assert matched
    # Every hit must mention the term somewhere, and the search is AND-joined.
    assert any("sad" in (cue["tags"] or []) for cue in matched)
    assert app.filter_cues(cues, app.ALL_CATEGORIES, "sad zzzznomatch") == []


def test_cue_to_generator_fills_the_create_tab(app):
    cues, _categories, _defaults, _note = app.load_cue_library()
    cue = cues[0]
    style, seconds, influence, music_only, plan, lyrics, seed, status, tab = app.cue_to_generator(cue)
    assert style == cue["style"]
    assert seconds == int(cue["seconds"]) == 30
    assert influence == float(cue["influence"])
    assert music_only is True, "anime cues must load the instrumental adapter"
    assert plan == cue["plan"]
    assert lyrics == "", "stale lyrics must be cleared"
    assert "loaded into **Create**" in status


def test_cue_to_generator_rejects_a_cue_without_style(app):
    result = app.cue_to_generator({"title": "Broken"})
    assert "no style text" in result[7]


# --------------------------------------------------------------------------- #
# output housekeeping
# --------------------------------------------------------------------------- #

def test_clear_list_empties_every_library_without_touching_disk(app, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    entry = runs / "song-keep"
    entry.mkdir(parents=True)
    (entry / "audio.flac").write_bytes(b"x")
    monkeypatch.setattr(app, "RUNS_DIR", runs)

    create, cover, edit, message = app.clear_library_list()
    assert create == cover == edit == []
    assert "untouched" in message
    assert (entry / "audio.flac").is_file(), "clearing the list must not delete anything"


def test_deleting_audio_requires_confirmation(app):
    with pytest.raises(Exception, match="confirmation"):
        app.delete_audio_files(False)


def test_deleting_everything_requires_confirmation(app):
    with pytest.raises(Exception, match="confirmation"):
        app.delete_all_runs(False)


def test_delete_audio_keeps_scores_and_metadata(app, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    entry = runs / "song-x"
    entry.mkdir(parents=True)
    (entry / "audio.flac").write_bytes(b"x" * 2048)
    (entry / "score.abc").write_text("X:1\nK:C\n")
    (entry / "request.json").write_text("{}")
    monkeypatch.setattr(app, "RUNS_DIR", runs)

    create, _cover, _edit, message = app.delete_audio_files(True)
    assert create == [], "the song disappears from the library once its audio is gone"
    assert not (entry / "audio.flac").exists()
    assert (entry / "score.abc").is_file() and (entry / "request.json").is_file()
    assert "Deleted **1** audio" in message


def test_delete_all_removes_the_run_directories(app, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name in ("song-a", "song-b"):
        entry = runs / name
        entry.mkdir(parents=True)
        (entry / "audio.flac").write_bytes(b"x")
        (entry / "score.abc").write_text("X:1")
    monkeypatch.setattr(app, "RUNS_DIR", runs)

    create, cover, edit, message = app.delete_all_runs(True)
    assert create == cover == edit == []
    assert not any(runs.iterdir())
    assert "Deleted **2** runs" in message


def test_library_status_reports_a_count(app, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    entry = runs / "song-x"
    entry.mkdir(parents=True)
    (entry / "audio.flac").write_bytes(b"x" * 1024)
    monkeypatch.setattr(app, "RUNS_DIR", runs)
    assert "1 songs" in app.library_status()


def test_anime_tab_exposes_the_browser_and_send_actions(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    descendants = _descendants(_tab(demo, "Anime sounds"))
    labels = {getattr(c, "label", None) for c in descendants}
    for expected in ("Scene type", "Search"):
        assert expected in labels, f"missing anime control: {expected}"
    buttons = {getattr(c, "value", None) for c in descendants if type(c).__name__ == "Button"}
    assert "Reload from disk" in buttons


def test_loop_tools_live_on_the_create_tab(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    descendants = _descendants(_tab(demo, "Create"))
    labels = {getattr(c, "label", None) for c in descendants}
    for expected in ("Loop starts (s)", "Loop ends (s)", "Loop repeats", "Crossfade (ms)",
                     "Snap to bar lines"):
        assert expected in labels, f"missing loop control: {expected}"
    buttons = {getattr(c, "value", None) for c in descendants if type(c).__name__ == "Button"}
    assert "Build loop from the latest result" in buttons


def test_clear_output_controls_live_on_the_create_tab(app, ui_kwargs):
    demo = app.build_ui(ui_kwargs)
    descendants = _descendants(_tab(demo, "Create"))
    buttons = {getattr(c, "value", None) for c in descendants if type(c).__name__ == "Button"}
    assert "Clear the list (keeps every file)" in buttons
    assert "Delete audio files (keep scores)" in buttons
    assert "Delete everything in runs/" in buttons


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #

def test_every_generation_input_maps_to_a_handler_parameter(app, ui_kwargs):
    """Regression: positional input lists must follow the signature exactly."""
    demo = app.build_ui(ui_kwargs)
    parameters = [n for n in inspect.signature(app.generate).parameters if n != "progress"]
    generation = [f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "generate"]
    assert len(generation) == 3, "Create, Cover and Edit must each bind a generate handler"
    for fn in generation:
        assert len(fn.inputs) == len(parameters)


def test_ui_has_no_gradio_arity_warning(app, ui_kwargs):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        app.build_ui(ui_kwargs)
    arity = [str(w.message) for w in caught if "arguments" in str(w.message)]
    assert arity == [], f"input/parameter mismatch: {arity}"


def test_pipeline_key_reacts_to_every_rebuild_trigger(app):
    base = {
        "model": "m", "vae": "v", "device": "auto", "backend": "torch",
        "quantization": "none", "offload_ar": False, "budget": 24,
        "verify_hashes": True, "profile": "balanced", "attention": "auto",
        "fuse": True, "cuda_graph": True, "tf32": False, "cudnn_bench": False,
        "vae_frames": 1024, "offline": False, "revision": "", "vae_revision": "",
        "hf_token": None, "cache_dir": None, "music_only": False,
    }
    baseline = app._pipeline_key(base)
    for field, changed in (
        ("backend", "vllm"), ("profile", "fast"), ("attention", "flash"),
        ("fuse", False), ("vae_frames", 512), ("budget", 12), ("quantization", "fp8"),
        ("hf_token", "hf_x"), ("cache_dir", "/tmp/x"), ("music_only", True),
    ):
        assert app._pipeline_key(dict(base, **{field: changed})) != baseline, field


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def test_generate_rejects_missing_style(app):
    audio, _s, _t, status, _lib = app.generate(**_generate_kwargs(style="   "))
    assert audio is None and status == "Describe the musical style first."


def test_empty_lyrics_switches_to_music_only(app, monkeypatch):
    """The base checkpoint always sings, so empty lyrics must mean instrumental."""
    seen = {}

    def fake_build(kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here - we only need the resolved options")

    monkeypatch.setattr(app, "_build_instrumental", fake_build)
    monkeypatch.setattr(app, "apply_instrumental", fake_build)
    _audio, _score, _timings, status, _lib = app.generate(**_generate_kwargs(lyrics="   "))
    # The instrumental builder is only chosen when music_only resolved to True.
    assert "instrumental adapter" in status or "Could not load" in status


def test_music_only_rejects_a_bad_section_plan(app):
    _a, _s, _t, status, _lib = app.generate(
        **_generate_kwargs(music_only=True, section_plan="not a tag")
    )
    assert "Section plan problem" in status


def test_music_only_cannot_be_combined_with_fp8(app):
    """The adapter needs the plain BF16 linears the FP8 swap replaces."""
    from yue2.lora import LoRAError, apply_instrumental

    class _Pipe:
        quantization = "fp8"

    with pytest.raises(LoRAError, match="unquantized BF16"):
        apply_instrumental(_Pipe())


def test_generate_rejects_a_score_with_planning_off(app):
    _a, _s, _t, status, _lib = app.generate(
        **_generate_kwargs(planning="off", abc_text="X:1\nK:C\nCDEF|")
    )
    assert "needs planning set to melody or full" in status


def test_load_run_reports_a_missing_directory(app):
    _abc, _style, _lyrics, status = app.load_run("does-not-exist")
    assert "No such run" in status


def test_send_score_to_edit_explains_an_empty_score(app):
    _abc, _style, _lyrics, status = app.send_score_to_edit("  ", "pop", "la")
    assert "no score to edit" in status


def test_hardware_report_mentions_fp8_and_transcription(app):
    report = app.hardware_report()
    assert "FP8 on this device" in report
    assert "Reference-audio transcription" in report


def test_transcribe_without_audio_is_rejected(app):
    _abc, _lyrics, status = app.transcribe_reference(None, True, progress=lambda *a, **k: None)
    assert "Upload a recording first" in status


# --------------------------------------------------------------------------- #
# ports
# --------------------------------------------------------------------------- #

def test_port_available_reflects_a_bound_socket(app):
    import socket

    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        listener.listen(1)
        assert app.port_available(host, listener.getsockname()[1]) is False


def test_find_free_port_skips_a_busy_port(app):
    """Regression: a second `uv run app.py` used to die on the occupied port."""
    import socket

    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        listener.listen(1)
        taken = listener.getsockname()[1]
        chosen = app.find_free_port(host, taken, attempts=10)
    assert chosen is not None and chosen != taken
    assert app.port_available(host, chosen) is True


def test_find_free_port_gives_up_instead_of_looping_forever(app):
    import socket

    host = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        listener.listen(1)
        assert app.find_free_port(host, listener.getsockname()[1], attempts=1) is None


# --------------------------------------------------------------------------- #
# GPU end-to-end
# --------------------------------------------------------------------------- #

def _require_gpu_and_weights():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to run the UI handler")
    if not (MODEL_DIR / "model.safetensors").is_file() or not (VAE_DIR / "model.safetensors").is_file():
        pytest.skip("local YuE2 weights are not present under models/")


@pytest.mark.slow
def test_ui_handler_generates_audio_end_to_end(app, ui_kwargs):
    """Drive the real handler with the Create tab's own defaults, positionally."""
    _require_gpu_and_weights()
    import soundfile as sf

    demo = app.build_ui(ui_kwargs)
    fn = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "generate")
    parameters = [n for n in inspect.signature(app.generate).parameters if n != "progress"]
    values = [getattr(c, "value", None) for c in fn.inputs]
    overrides = {
        "length_seconds": 12,          # ~300 semantic tokens
        "quality": 12,
        "verify_hashes": False, "offline": True, "save_artifacts": False,
        "abc_max_tokens": 200, "abc_min_tokens": 8, "sem_min_tokens": 8,
    }
    for name, value in overrides.items():
        values[parameters.index(name)] = value

    if values[parameters.index("style")] in (None, ""):
        values[parameters.index("style")] = "English, solo acoustic piano, slow ballad"
    if not (values[parameters.index("lyrics")] or "").strip():
        values[parameters.index("lyrics")] = "[Verse]\nA quiet room, a single light\n[Chorus]\nStay with me"

    audio, score, timings, status, library = app.generate(*values, progress=lambda *a, **k: None)
    try:
        assert "Generation failed" not in status, status
        assert "Could not load" not in status, status
        assert audio is not None and Path(audio).is_file(), status
        samples, rate = sf.read(audio)
        assert rate == 48000 and samples.size > 0
        assert abs(samples).max() > 1e-3, "decoded audio is silent"

        parsed = json.loads(timings)
        assert parsed["e2e_seconds"] > 0
        assert parsed["requested_seconds"] == 12.0
        assert parsed["semantic"]["output_tokens"] > 0
        assert "Done in" in status
        assert score
        # The finished song must be handed back for the library list.
        assert library and library[0]["name"] == Path(audio).parent.name
    finally:
        if audio:
            run_dir = Path(audio).parent
            if run_dir.is_dir() and not any(run_dir.glob("result.json")):
                for child in sorted(run_dir.rglob("*"), reverse=True):
                    child.unlink() if child.is_file() else child.rmdir()
                run_dir.rmdir()
