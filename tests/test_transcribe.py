"""Reference-audio transcription wrapper.

These run without downloading SheetSage2 or touching the GPU: they cover the
error paths and the GPU-release contract, which is what the UI depends on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from yue2 import transcribe


def test_audio_helpers_are_declared_available():
    """The `cover` group is installed by default, so this should be True."""
    assert transcribe.dependencies_available() is True


def test_a_fresh_transcriber_reports_not_loaded():
    subject = transcribe.Transcriber()
    assert subject.loaded is False
    assert subject.load_seconds is None


def test_close_is_safe_when_nothing_was_loaded():
    subject = transcribe.Transcriber()
    subject.close()
    assert subject.loaded is False


def test_missing_audio_file_is_reported_before_any_model_load(tmp_path):
    subject = transcribe.Transcriber()
    with pytest.raises(transcribe.TranscriptionError, match="Audio file not found"):
        subject.transcribe(tmp_path / "absent.wav")
    # The check must happen before loading, so no model was fetched.
    assert subject.loaded is False


def test_transcription_error_is_a_runtime_error():
    assert issubclass(transcribe.TranscriptionError, RuntimeError)


def test_missing_dependencies_produce_an_actionable_message(monkeypatch):
    monkeypatch.setattr(transcribe, "dependencies_available", lambda: False)
    subject = transcribe.Transcriber()
    with pytest.raises(transcribe.TranscriptionError, match="uv sync --extra cover"):
        subject.load()


def test_close_releases_the_model_reference():
    subject = transcribe.Transcriber()
    subject._model = object()          # pretend a model is resident
    assert subject.loaded is True
    subject.close()
    assert subject.loaded is False
    assert subject.load_seconds is None


def test_default_repo_is_the_published_sheet_sage_checkpoint():
    assert transcribe.DEFAULT_REPO == "m-a-p/SheetSage2"


# --------------------------------------------------------------------------- #
# GPU end-to-end (downloads SheetSage2 on first run)
# --------------------------------------------------------------------------- #

def _find_sample_audio():
    """Any previously generated song; the repo ships no audio fixture."""
    repo = Path(__file__).resolve().parents[1]
    candidates = sorted((repo / "benches").glob("audio-*.flac"))
    if not candidates:
        candidates = sorted((repo / "runs").glob("*/audio.flac"))
    return candidates[0] if candidates else None


@pytest.mark.slow
def test_transcription_produces_an_editable_score():
    """The cover workflow depends on SheetSage2 returning usable ABC."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for transcription")
    sample = _find_sample_audio()
    if sample is None:
        pytest.skip("no generated audio to transcribe; run scripts/benchmark.py --save-audio first")

    subject = transcribe.Transcriber()
    try:
        result = subject.transcribe(sample, melody_only=True)
    finally:
        subject.close()

    assert subject.loaded is False, "the transcriber must release the GPU after use"
    abc = result["abc"]
    assert abc.strip()
    # A usable score carries a header and at least one voice.
    assert any(line.startswith("K:") for line in abc.splitlines()), "score has no key"
    assert any(line.startswith("V:") for line in abc.splitlines()), "score has no voice"
    assert result["seconds"] > 0
    assert result["melody_only"] is True

