"""Assemble loopable cues from generated audio.

Why this is a post-process and not a generation option: YuE2 conditions on the
ABC score as **prompt tokens only** - ``token_prefixes`` encodes the score into
the prefix and nothing maps its bars to audio seconds. A probe confirmed the
consequences: asked for ``[intro] [verse] [chorus] [verse] [outro]`` at 30 s and
90 BPM, the model wrote a single ``% verse`` marker and chose 87 BPM. Length, by
contrast, is exact, because it is the token budget (25 frames per second).

So the model supplies the material and this module supplies the structure:

1. Beat-track the *rendered audio* - not the score, whose tempo was 3 % off and
   would drift about 0.3 s over a 10 s loop.
2. Snap the loop window so its length is a whole number of bars, which is what
   keeps the meter intact when the loop repeats.
3. Crossfade the seams and emit a seamless loop plus an extended track.

Seams are clean on sustained and ambient material, which is most anime
underscore, and rougher on percussive or hit-driven cues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_BEATS_PER_BAR = 4
ANALYSIS_SR = 22050


class LoopError(RuntimeError):
    """Raised when a loop cannot be built from the supplied audio."""


def dependencies_available():
    import importlib.util

    return importlib.util.find_spec("librosa") is not None


@dataclass
class BeatGrid:
    """Detected rhythmic grid of a rendered cue."""

    tempo: float
    beat_times: np.ndarray
    beats_per_bar: int

    @property
    def bar_seconds(self):
        return 60.0 / self.tempo * self.beats_per_bar

    def to_dict(self):
        return {
            "tempo_bpm": round(self.tempo, 2),
            "beats_detected": int(len(self.beat_times)),
            "beats_per_bar": self.beats_per_bar,
            "bar_seconds": round(self.bar_seconds, 4),
        }


@dataclass
class LoopResult:
    loop: np.ndarray
    extended: np.ndarray
    report: dict


def _mono(audio):
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def analyse(audio, sample_rate, beats_per_bar=DEFAULT_BEATS_PER_BAR):
    """Detect the tempo and beat times of rendered audio."""
    if not dependencies_available():
        raise LoopError(
            "Beat tracking needs the 'loop' group (librosa). Install it with: uv sync --extra loop"
        )
    if isinstance(beats_per_bar, bool) or not isinstance(beats_per_bar, int) or beats_per_bar < 1:
        raise LoopError("beats_per_bar must be a positive integer")
    import librosa

    mono = _mono(audio)
    if mono.size < sample_rate:
        raise LoopError("Audio is shorter than one second; nothing to loop.")
    # Analyse at 22.05 kHz: beat tracking is insensitive to the top octave and
    # this is roughly twice as fast.
    if sample_rate != ANALYSIS_SR:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=ANALYSIS_SR)
    tempo, beats = librosa.beat.beat_track(y=mono, sr=ANALYSIS_SR, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(tempo) or tempo <= 0:
        raise LoopError("Could not detect a tempo in this audio.")
    beats = np.asarray(beats, dtype=float)
    if beats.size < 2:
        raise LoopError("Could not detect a beat grid in this audio.")
    return BeatGrid(tempo=tempo, beat_times=beats, beats_per_bar=beats_per_bar)


def snap_loop(start, end, grid, *, min_seconds=2.0):
    """Snap a loop window to the grid, keeping its length a whole number of bars.

    Both edges move onto the same grid, so the loop length is an exact multiple
    of the bar duration. That is what stops the meter drifting as it repeats -
    snapping the two edges independently would not guarantee it.
    """
    beats = grid.beat_times
    bar = grid.bar_seconds
    start = float(np.clip(start, 0.0, beats[-1]))
    end = float(np.clip(end, 0.0, beats[-1] + bar))
    if end - start < min_seconds:
        raise LoopError(f"Loop window is too short ({end - start:.1f}s); widen it.")

    snapped_start = float(beats[np.argmin(np.abs(beats - start))])
    bars = max(1, int(round((end - start) / bar)))
    snapped_end = snapped_start + bars * bar
    if snapped_end > beats[-1] + bar * 2:
        # Walk back until the window fits inside the audio.
        while bars > 1 and snapped_start + bars * bar > beats[-1] + bar:
            bars -= 1
        snapped_end = snapped_start + bars * bar
    return snapped_start, snapped_end, bars


def _equal_power_fade(n, *, inverse=False):
    """Equal-power ramp; a linear fade would dip in the middle of the join."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    curve = np.sin(t * np.pi / 2) if inverse else np.cos(t * np.pi / 2)
    return curve[:, None]


def crossfade_concat(segments, sample_rate, fade_ms):
    """Join segments, overlapping each pair with an equal-power crossfade."""
    parts = [np.asarray(s, dtype=np.float32) for s in segments if len(s)]
    if not parts:
        raise LoopError("Nothing to join.")
    if len(parts) == 1 or fade_ms <= 0:
        return np.concatenate(parts)
    fade = int(sample_rate * fade_ms / 1000)
    out = parts[0]
    for nxt in parts[1:]:
        n = min(fade, len(out), len(nxt))
        if n <= 1:
            out = np.concatenate([out, nxt])
            continue
        blended = out[-n:] * _equal_power_fade(n) + nxt[:n] * _equal_power_fade(n, inverse=True)
        out = np.concatenate([out[:-n], blended, nxt[n:]])
    return out


def make_seamless(segment, sample_rate, fade_ms):
    """Fold the tail of a segment into its head so it loops without a click.

    The last ``n`` samples are crossfaded with the first ``n`` and the two ends
    are replaced by that single blended region, so the result is ``n`` samples
    shorter. Its final sample is then adjacent to its first in the original
    audio, which is what makes the wrap silent.
    """
    segment = np.asarray(segment, dtype=np.float32)
    fade = int(sample_rate * fade_ms / 1000)
    n = min(fade, len(segment) // 4)
    if n <= 1:
        return segment
    blended = (segment[-n:] * _equal_power_fade(n)
               + segment[:n] * _equal_power_fade(n, inverse=True))
    return np.concatenate([blended, segment[n:-n]])


def build_loop(audio, sample_rate, *, loop_start, loop_end, repeats=3,
               crossfade_ms=60, grid=None, snap=True):
    """Return a ``LoopResult`` with a seamless loop and an extended track."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise LoopError("Expected mono or stereo audio.")
    total = len(audio) / sample_rate
    if loop_end <= loop_start:
        raise LoopError("Loop end must be after loop start.")
    if loop_start < 0 or loop_end > total + 1e-6:
        raise LoopError(f"Loop window {loop_start:.1f}-{loop_end:.1f}s is outside the {total:.1f}s cue.")
    repeats = int(repeats)
    if repeats < 1:
        raise LoopError("Repeat count must be at least 1.")
    crossfade_ms = float(crossfade_ms)
    if crossfade_ms < 0:
        raise LoopError("Crossfade cannot be negative.")

    report = {"requested": {"start": round(loop_start, 3), "end": round(loop_end, 3)},
              "snapped": snap, "repeats": repeats, "crossfade_ms": crossfade_ms,
              "cue_seconds": round(total, 3)}
    if grid is not None:
        report["grid"] = grid.to_dict()

    start, end, bars = float(loop_start), float(loop_end), None
    if snap:
        if grid is None:
            raise LoopError("Snapping needs a beat grid; call analyse() first.")
        start, end, bars = snap_loop(loop_start, loop_end, grid)
        report["snapped_to_grid"] = {"start": round(start, 3), "end": round(end, 3),
                                     "bars": bars, "loop_seconds": round(end - start, 3)}
    report["loop_seconds"] = round(end - start, 3)

    a = int(round(start * sample_rate))
    b = int(round(end * sample_rate))
    if b >= len(audio):
        b = len(audio)
    intro, body, outro = audio[:a], audio[a:b], audio[b:]
    if len(body) < sample_rate // 2:
        raise LoopError("Loop section is under half a second; widen the window.")

    loop = make_seamless(body, sample_rate, crossfade_ms)
    extended = crossfade_concat([intro, *([loop] * repeats), outro], sample_rate, crossfade_ms)

    report["loop_samples"] = int(len(loop))
    report["extended_seconds"] = round(len(extended) / sample_rate, 3)
    report["intro_seconds"] = round(len(intro) / sample_rate, 3)
    report["outro_seconds"] = round(len(outro) / sample_rate, 3)
    return LoopResult(loop=loop, extended=extended, report=report)


def save_wav(path, audio, sample_rate):
    import soundfile as sf

    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    sf.write(str(path), data, sample_rate, subtype="PCM_24")
    return str(path)


def save_flac(path, audio, sample_rate):
    import soundfile as sf

    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    sf.write(str(path), data, sample_rate, subtype="PCM_24")
    return str(path)
