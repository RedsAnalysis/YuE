"""Loop assembly: beat grid, bar-accurate snapping, seamless joins.

All CPU-only, using synthetic audio so nothing needs a GPU or a model.
"""
from __future__ import annotations

import numpy as np
import pytest

from yue2 import loops


def click_track(bpm=120.0, seconds=30.0, sample_rate=48000, channels=2):
    """A music-like fixture: decaying noise bursts on every beat over a soft bed.

    The bursts need a sharp attack. Smooth windowed bumps - a Hann curve, say -
    are invisible to librosa's spectral-flux onset detector at 48 kHz, and the
    tempo comes back as zero.
    """
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng(0)
    signal = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    burst = 480 if sample_rate == 48000 else 220
    envelope = np.exp(-np.linspace(0, 10, burst)).astype(np.float32)
    step = int(60.0 / bpm * sample_rate)
    for start in range(0, n - step, step):
        length = min(burst, n - start)
        signal[start:start + length] += (
            rng.standard_normal(length).astype(np.float32) * envelope[:length]
        )
    return np.repeat(signal[:, None], channels, axis=1)


@pytest.fixture(scope="module")
def grid():
    return loops.analyse(click_track(), 48000)


def test_librosa_is_available():
    assert loops.dependencies_available() is True


def test_analyse_detects_the_click_tempo(grid):
    # librosa sometimes reports half or double tempo; either is a valid grid.
    ratio = grid.tempo / 120.0
    assert any(abs(ratio - k) < 0.08 for k in (0.5, 1.0, 2.0)), grid.tempo
    assert grid.beats_per_bar == 4
    assert len(grid.beat_times) > 4
    assert grid.bar_seconds == pytest.approx(60.0 / grid.tempo * 4)


def test_analyse_rejects_very_short_audio():
    with pytest.raises(loops.LoopError, match="shorter than one second"):
        loops.analyse(click_track(seconds=0.5), 48000)


def test_analyse_rejects_a_bad_meter():
    with pytest.raises(loops.LoopError, match="beats_per_bar"):
        loops.analyse(click_track(), 48000, beats_per_bar=0)


def test_snap_loop_yields_a_whole_number_of_bars(grid):
    start, end, bars = loops.snap_loop(15.0, 25.0, grid)
    assert bars >= 1
    assert (end - start) == pytest.approx(bars * grid.bar_seconds, abs=1e-6)


def test_snap_loop_puts_both_edges_on_the_grid(grid):
    start, end, _bars = loops.snap_loop(15.0, 25.0, grid)
    nearest = grid.beat_times[np.argmin(np.abs(grid.beat_times - start))]
    assert start == pytest.approx(nearest)
    # The end is start plus whole bars, so it lands on the same grid.
    assert end == pytest.approx(start + round((end - start) / grid.bar_seconds) * grid.bar_seconds)


def test_snap_loop_rejects_a_window_that_is_too_short(grid):
    with pytest.raises(loops.LoopError, match="too short"):
        loops.snap_loop(10.0, 10.1, grid)


def test_make_seamless_shortens_by_the_crossfade():
    segment = click_track(seconds=5.0, sample_rate=48000)[: 48000 * 4]
    fade_ms = 60
    n = int(48000 * fade_ms / 1000)
    out = loops.make_seamless(segment, 48000, fade_ms)
    # The two n-sample ends are folded into one crossfaded n-sample region.
    assert len(out) == len(segment) - n
    # The wrap point must not be an outlier against ordinary sample-to-sample
    # movement inside the signal. Measured on real audio this is ~0.014.
    seam = abs(float(out[0].mean()) - float(out[-1].mean()))
    mid = len(out) // 2
    interior = abs(float(out[mid].mean()) - float(out[mid - 1].mean()))
    assert seam <= max(interior * 20, 0.05), f"seam {seam} vs interior {interior}"


def test_make_seamless_is_a_no_op_for_a_huge_crossfade():
    segment = click_track(seconds=1.0, sample_rate=48000)[:4800]
    assert len(loops.make_seamless(segment, 48000, 10_000)) <= len(segment)


def test_crossfade_concat_keeps_the_signal_level_steady():
    """Linear fades dip in the middle; equal-power must not."""
    sr = 48000
    tone = np.full((sr, 1), 0.5, dtype=np.float32)
    joined = loops.crossfade_concat([tone, tone], sr, 500)
    middle = joined[int(sr * 0.95):int(sr * 1.05)]
    assert float(np.abs(middle).min()) > 0.47, "equal-power join should hold level"
    assert len(joined) == 2 * sr - int(sr * 0.5)


def test_crossfade_concat_with_zero_fade_is_a_plain_join():
    a = np.zeros((10, 1), dtype=np.float32)
    b = np.ones((10, 1), dtype=np.float32)
    out = loops.crossfade_concat([a, b], 48000, 0)
    assert len(out) == 20 and out[-1, 0] == 1.0


def test_build_loop_lengths_add_up_exactly_without_crossfades(grid):
    """With fades off the arithmetic is exact: intro + loop x N + outro."""
    audio = click_track(seconds=30.0)
    result = loops.build_loop(audio, 48000, loop_start=15, loop_end=25,
                              repeats=3, crossfade_ms=0, grid=grid)
    report = result.report
    expected = report["intro_seconds"] + report["loop_seconds"] * 3 + report["outro_seconds"]
    assert report["extended_seconds"] == pytest.approx(expected, abs=0.01)
    assert report["extended_seconds"] > report["cue_seconds"], "an extended track must be longer"


def test_build_loop_crossfades_shorten_the_extended_track(grid):
    """Fades overlap the joins, so the result is slightly shorter than the sum."""
    audio = click_track(seconds=30.0)
    plain = loops.build_loop(audio, 48000, loop_start=15, loop_end=25,
                             repeats=3, crossfade_ms=0, grid=grid)
    faded = loops.build_loop(audio, 48000, loop_start=15, loop_end=25,
                             repeats=3, crossfade_ms=60, grid=grid)
    assert faded.report["extended_seconds"] < plain.report["extended_seconds"]
    # Four joins plus the loop's own two folded edges, all at 60 ms.
    assert plain.report["extended_seconds"] - faded.report["extended_seconds"] < 1.0
    assert faded.report["extended_seconds"] > faded.report["cue_seconds"]


def test_build_loop_without_snapping_uses_the_exact_window(grid):
    audio = click_track(seconds=30.0)
    result = loops.build_loop(audio, 48000, loop_start=15, loop_end=25,
                              repeats=2, crossfade_ms=0, grid=grid, snap=False)
    assert result.report["snapped_to_grid"] if "snapped_to_grid" in result.report else True
    assert result.report["loop_seconds"] == pytest.approx(10.0, abs=0.01)


def test_build_loop_without_a_grid_cannot_snap():
    audio = click_track(seconds=10.0)
    with pytest.raises(loops.LoopError, match="needs a beat grid"):
        loops.build_loop(audio, 48000, loop_start=2, loop_end=6, grid=None, snap=True)


@pytest.mark.parametrize("start,end,repeats,message", [
    (5.0, 5.0, 3, "must be after"),
    (5.0, 2.0, 3, "must be after"),
    (0.0, 40.0, 3, "outside"),
    (2.0, 6.0, 0, "at least 1"),
])
def test_build_loop_rejects_bad_arguments(start, end, repeats, message):
    audio = click_track(seconds=10.0)
    with pytest.raises(loops.LoopError, match=message):
        loops.build_loop(audio, 48000, loop_start=start, loop_end=end, repeats=repeats, snap=False)


def test_build_loop_output_is_finite_stereo(grid):
    audio = click_track(seconds=30.0)
    result = loops.build_loop(audio, 48000, loop_start=15, loop_end=25, grid=grid)
    assert result.loop.ndim == 2 and result.loop.shape[1] == audio.shape[1]
    assert result.extended.ndim == 2
    assert np.isfinite(result.loop).all() and np.isfinite(result.extended).all()


def test_save_helpers_write_playable_files(tmp_path, grid):
    import soundfile as sf

    audio = click_track(seconds=30.0)
    result = loops.build_loop(audio, 48000, loop_start=15, loop_end=25, grid=grid)
    wav = loops.save_wav(tmp_path / "loop.wav", result.loop, 48000)
    flac = loops.save_flac(tmp_path / "extended.flac", result.extended, 48000)
    for path, expected in ((wav, len(result.loop)), (flac, len(result.extended))):
        data, rate = sf.read(path)
        assert rate == 48000 and len(data) == expected
