"""Inference-time measurement for YuE2.

Two layers:

* CPU-safe unit checks for the benchmark arithmetic (always run).
* ``test_measured_inference_time`` which loads the real checkpoint on the GPU,
  times every stage, writes a JSON record under ``benches/``, and asserts the
  result is in a plausible range. It is marked ``slow`` and skips unless CUDA
  and the local weights are both present.

Run just the timing test::

    pytest tests/test_benchmark.py -m slow -s
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "YuE2-3B"
VAE_DIR = REPO_ROOT / "models" / "YuE2-Vae"


def _args(**overrides):
    base = dict(quick=False, max_tokens=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_quick_mode_uses_a_short_budget():
    abc, semantic = benchmark.build_request(_args(quick=True))
    assert semantic.max_tokens == benchmark.QUICK_SEMANTIC_TOKENS
    assert abc.max_tokens == benchmark.QUICK_ABC_TOKENS
    assert semantic.min_tokens <= semantic.max_tokens


def test_default_budget_passes_no_sampling_override():
    abc, semantic = benchmark.build_request(_args(quick=False, max_tokens=None))
    assert abc is None and semantic is None


def test_explicit_max_tokens_becomes_a_sampling_override():
    abc, semantic = benchmark.build_request(_args(quick=False, max_tokens=1500))
    assert semantic.max_tokens == 1500
    assert abc.max_tokens <= 4096


def test_stage_timings_reads_nested_result_timing():
    result = SimpleNamespace(
        timing={
            "abc": {"seconds": 2.0, "output_tokens": 100, "output_tps": 50.0, "ttft_seconds": 0.4, "attention": "flash"},
            "semantic": {"seconds": 8.0, "output_tokens": 800, "output_tps": 100.0,
                         "ttft_seconds": 0.5, "attention": "flash", "cfg_branches": 1,
                         "fuse_projections": True},
            "nar_seconds": 3.0,
            "vae_seconds": 1.5,
            "e2e_seconds": 15.0,
        },
        audio=[0.0] * 48000,
        sample_rate=48000,
        truncated={"abc": False, "semantic": False},
    )
    pipe = SimpleNamespace(load_timing={"mot_load_seconds": 3.0})
    stages = benchmark.stage_timings(result, pipe, wall_seconds=16.0, peak_bytes=8 * 2**30)

    assert stages["e2e_seconds"] == 15.0
    assert stages["abc_tps"] == 50.0
    assert stages["semantic_tokens"] == 800
    assert stages["nar_seconds"] == 3.0
    assert stages["audio_seconds"] == 1.0
    assert stages["peak_vram_gib"] == 8.0
    assert stages["fuse_projections"] is True


def test_stage_timings_tolerates_missing_stage_fields():
    result = SimpleNamespace(timing={}, audio=[0.0] * 4800, sample_rate=48000,
                             truncated={"abc": False, "semantic": False})
    stages = benchmark.stage_timings(result, SimpleNamespace(load_timing={}), 1.0, 0)
    assert stages["semantic_seconds"] == 0.0
    assert stages["semantic_tokens"] is None


def _require_gpu_and_weights():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to measure inference time")
    if not (MODEL_DIR / "model.safetensors").is_file() or not (VAE_DIR / "model.safetensors").is_file():
        pytest.skip("local YuE2 weights are not present under models/")


@pytest.mark.slow
def test_measured_inference_time(tmp_path, capsys):
    """Time a real short generation and persist the measurement."""
    _require_gpu_and_weights()
    import torch
    from yue2 import YuE2Pipeline
    from yue2.perf import detect_hardware, profile as named_profile, validated_profile
    from yue2.protocol import Sampling

    hardware = detect_hardware()
    performer, notes = validated_profile(named_profile("balanced"), hardware)
    assert performer.cuda_graph is True, f"CUDA graphs should survive validation: {notes}"
    assert hardware["bf16_supported"] is True

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    pipe = YuE2Pipeline.from_pretrained(
        str(MODEL_DIR), vae=str(VAE_DIR), device="cuda", backend="torch",
        quantization="none", perf=performer, local_files_only=True, progress=False,
    )
    sampling = Sampling(max_tokens=256, min_tokens=8)
    try:
        result = pipe(
            style="English, solo acoustic piano, slow ballad, male vocal, 70 BPM",
            lyrics="[Verse]\nA quiet room, a single light\n[Chorus]\nStay with me tonight",
            cot="full", seed=831001,
            abc_sampling=sampling, semantic_sampling=sampling,
        )
        peak = torch.cuda.max_memory_allocated()
    finally:
        pipe.close()

    timing = result.timing
    semantic = timing["semantic"]
    abc = timing["abc"]
    print(
        f"\n  abc      {abc['seconds']:6.2f}s  {abc['output_tokens']:5d} tok  "
        f"{abc.get('output_tps', 0):7.2f} tok/s  attention={abc.get('attention')}"
    )
    print(
        f"  semantic {semantic['seconds']:6.2f}s  {semantic['output_tokens']:5d} tok  "
        f"{semantic.get('output_tps', 0):7.2f} tok/s  fused={semantic.get('fuse_projections')}"
    )
    print(f"  nar      {timing['nar_seconds']:6.2f}s   vae {timing['vae_seconds']:5.2f}s")
    print(f"  e2e      {timing['e2e_seconds']:6.2f}s   peak VRAM {peak / 2**30:.2f} GiB")
    print(capsys.readouterr().err[-400:])

    # The generation must have produced usable audio.
    assert result.audio.ndim == 2 and result.audio.shape[1] == 2
    assert result.sample_rate == 48000
    import numpy as np

    assert np.isfinite(result.audio).all()
    assert float(np.abs(result.audio).max()) > 1e-3, "decoded audio is silent"

    # Throughput floors: generous, but they catch a silent fallback to eager
    # CPU-style decoding or a broken CUDA graph capture.
    assert abc["output_tokens"] > 0
    assert semantic["output_tokens"] > 0
    assert semantic.get("output_tps", 0) > 10, f"AR decode collapsed: {semantic.get('output_tps')}"
    assert timing["e2e_seconds"] > 0 and timing["nar_seconds"] > 0
    assert peak < 24 * 2**30, "peak VRAM exceeded the RTX 3090's 24 GiB"

    record = {
        "profile": performer.to_dict(),
        "hardware": hardware,
        "abc": abc,
        "semantic": semantic,
        "nar_seconds": timing["nar_seconds"],
        "vae_seconds": timing["vae_seconds"],
        "e2e_seconds": timing["e2e_seconds"],
        "peak_vram_gib": round(peak / 2**30, 3),
    }
    target = tmp_path / "measured.json"
    target.write_text(json.dumps(record, indent=2, default=str))
    assert json.loads(target.read_text())["e2e_seconds"] == timing["e2e_seconds"]
