"""Reference-audio transcription with SheetSage2.

SheetSage2 converts a recording into an editable ABC score. It is what makes the
"cover this song" workflow possible, because YuE2 itself takes text and a score,
never audio as a condition.

Its published ``requirements.txt`` pins torch 2.8 and transformers 4.45, but the
architecture loads unchanged on this package's torch 2.10 / transformers 4.57 as
long as the audio helpers in the ``cover`` extra are installed. Loading happens
lazily and the model can be released explicitly, because transcription and
generation must not hold the GPU at the same time on a 24 GiB card.
"""
from __future__ import annotations

import time
from pathlib import Path

DEFAULT_REPO = "m-a-p/SheetSage2"


class TranscriptionError(RuntimeError):
    """Raised when a reference recording cannot be turned into a score."""


def dependencies_available():
    """True when the optional audio helpers are importable."""
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None
        for name in ("torchaudio", "scipy", "mir_eval", "pretty_midi", "mido")
    )


class Transcriber:
    """Lazily loaded SheetSage2 wrapper with explicit GPU release."""

    def __init__(self, repo=DEFAULT_REPO, device="cuda", cache_dir=None, local_files_only=False):
        self.repo = repo
        self.device = device
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model = None
        self.load_seconds = None

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        if self._model is not None:
            return self._model
        if not dependencies_available():
            raise TranscriptionError(
                "Reference-audio transcription needs the 'cover' extra "
                "(torchaudio, scipy, mir_eval, pretty_midi, mido). "
                "Install it with: uv sync --extra cover"
            )
        try:
            from transformers import AutoModel
        except ImportError as error:  # pragma: no cover - transformers is a hard dep
            raise TranscriptionError(f"transformers is unavailable: {error}") from error

        start = time.perf_counter()
        try:
            model = AutoModel.from_pretrained(
                self.repo,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
        except Exception as error:
            raise TranscriptionError(
                f"Could not load {self.repo}: {type(error).__name__}: {error}"
            ) from error
        model = model.eval()
        if self.device and self.device != "cpu":
            try:
                model = model.to(self.device)
            except Exception as error:
                raise TranscriptionError(f"Could not move the transcriber to {self.device}: {error}") from error
        self._model = model
        self.load_seconds = time.perf_counter() - start
        return model

    def transcribe(self, audio_path, *, melody_only=True, output_dir=None):
        """Return ``{"abc", "warnings", "seconds", "melody_only"}`` for a recording."""
        path = Path(audio_path)
        if not path.is_file():
            raise TranscriptionError(f"Audio file not found: {path}")
        model = self.load()
        start = time.perf_counter()
        try:
            result = model.transcribe(
                str(path),
                output_dir=str(output_dir) if output_dir else None,
                melody_only=bool(melody_only),
            )
        except Exception as error:
            raise TranscriptionError(
                f"Transcription failed: {type(error).__name__}: {error}"
            ) from error
        if not isinstance(result, dict):
            raise TranscriptionError(f"Unexpected transcription result: {type(result).__name__}")
        if result.get("abc_error"):
            raise TranscriptionError(f"Transcription reported an error: {result['abc_error']}")
        abc = (result.get("abc") or "").strip()
        if not abc:
            raise TranscriptionError(
                "SheetSage2 produced no score for this recording. "
                "Try a longer excerpt or a recording with a clearer melody."
            )
        return {
            "abc": abc,
            "warnings": list(result.get("warnings") or []),
            "seconds": time.perf_counter() - start,
            "melody_only": bool(melody_only),
        }

    def close(self):
        """Release the transcriber and its GPU memory."""
        self._model = None
        self.load_seconds = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            pass
