"""Performance-profile contracts and the plumbing that carries them into GraphAR."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yue2 import perf, pipeline
from yue2.pipeline import YuE2Pipeline
from yue2.protocol import GenerationConfig


def test_builtin_profiles_cover_the_documented_three():
    assert perf.profile_names() == ["reference", "balanced", "fast"]
    reference = perf.profile("reference")
    assert reference.fuse_projections is False
    assert reference.tf32 is False and reference.changes_numerics is False
    balanced = perf.profile("balanced")
    assert balanced.fuse_projections is True and balanced.tf32 is False
    assert balanced.changes_numerics is False
    fast = perf.profile("fast")
    assert fast.tf32 is True and fast.cudnn_benchmark is True
    assert fast.changes_numerics is True


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="profile must be one of"):
        perf.profile("turbo")


@pytest.mark.parametrize("backend", ["auto", "flash", "cudnn", "sdpa"])
def test_attention_backend_accepts_documented_values(backend):
    assert perf.PerfProfile(attention_backend=backend).attention_backend == backend


def test_invalid_attention_backend_and_tile_size_are_rejected():
    with pytest.raises(ValueError, match="attention_backend"):
        perf.PerfProfile(attention_backend="magic")
    with pytest.raises(ValueError, match="vae_core_frames"):
        perf.PerfProfile(vae_core_frames=0)
    with pytest.raises(ValueError, match="vae_core_frames"):
        perf.PerfProfile(vae_core_frames=True)


def test_apply_profile_maps_onto_torch_backend_flags():
    perf.apply_profile(perf.profile("fast"), torch.device("cpu"))
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False

    perf.apply_profile(perf.profile("reference"), torch.device("cpu"))
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_detect_hardware_reports_lists_not_tuples():
    hardware = perf.detect_hardware()
    assert hardware["device_type"] in {"cpu", "cuda", "mps"}
    assert isinstance(hardware["cuda_available"], bool)
    if hardware["device_type"] == "cuda":
        # JSON round-trips are used in artifacts, so capability must be a list.
        assert isinstance(hardware["capability"], list)


def test_validated_profile_flags_fp8_as_unsupported_on_ampere():
    """The RTX 3090 is compute capability 8.6, below the FP8 floor of 8.9."""
    ampere = {"device_type": "cuda", "cuda_available": True, "bf16_supported": True,
              "fp8_supported": False, "capability": [8, 6], "flash_attention": True}
    _, notes = perf.validated_profile(perf.profile("balanced"), ampere, quantization="fp8")
    assert any("8.9" in note for note in notes)


def test_validated_profile_drops_fusion_without_cuda_graphs():
    corrected, notes = perf.validated_profile(
        perf.PerfProfile(name="probe", fuse_projections=True, cuda_graph=False),
        {"device_type": "cpu", "cuda_available": False},
    )
    assert corrected.fuse_projections is False
    assert any("fuse_projections" in note for note in notes)


def test_validated_profile_disables_graphs_without_cuda():
    corrected, notes = perf.validated_profile(
        perf.profile("balanced"), {"device_type": "cpu", "cuda_available": False}
    )
    assert corrected.cuda_graph is False
    assert any("CUDA graphs" in note for note in notes)


def test_default_profile_is_balanced_only_on_cuda():
    assert perf.default_profile_for(torch.device("cuda")).name == "balanced"
    assert perf.default_profile_for(torch.device("cpu")).name == "reference"


def _bare_pipe(performer):
    pipe = object.__new__(YuE2Pipeline)
    pipe.progress, pipe.backend = False, "torch"
    pipe.perf = performer
    pipe.tokenizer = None
    pipe.generation_config = GenerationConfig()
    pipe._load_model = lambda **kwargs: object()
    return pipe


def test_pipeline_forwards_graph_options_into_generate_tokens(monkeypatch):
    """The profile must reach generate_tokens; this was the missing wiring."""
    captured = {}

    def fake_generate(subject, prefix, sampling, seed, phase, **kwargs):
        captured.update(kwargs)
        return [1, 2], {"output_tokens": 2}, False

    monkeypatch.setattr(pipeline, "generate_tokens", fake_generate)
    performer = perf.PerfProfile(name="probe", attention_backend="flash", fuse_projections=True)
    _bare_pipe(performer)._generate([1] * 8, GenerationConfig().semantic, 7, "semantic")

    assert captured["attention_backend"] == "flash"
    assert captured["fuse_projections"] is True
    assert captured["use_cuda_graph"] is True


def test_torch_eager_backend_disables_cuda_graphs(monkeypatch):
    captured = {}

    def fake_generate(subject, prefix, sampling, seed, phase, **kwargs):
        captured.update(kwargs)
        return [1], {"output_tokens": 1}, False

    monkeypatch.setattr(pipeline, "generate_tokens", fake_generate)
    pipe = _bare_pipe(perf.profile("balanced"))
    pipe.backend = "torch-eager"
    pipe._generate([1] * 8, GenerationConfig().semantic, 7, "semantic")

    assert captured["use_cuda_graph"] is False


def test_pipeline_rejects_an_unknown_profile_name():
    with pytest.raises(ValueError, match="profile must be one of"):
        YuE2Pipeline(SimpleNamespace(), SimpleNamespace(), perf="turbo")


def test_effective_config_records_the_profile(tmp_path):
    (tmp_path / "config.json").write_text('{"release_variant": "test"}')
    pipe = _bare_pipe(perf.profile("fast"))
    pipe.quantization, pipe.device = "none", torch.device("cpu")
    pipe.memory_budget_gib = 24
    pipe.model_dir = pipe.vae_dir = tmp_path
    pipe.vae_core_frames, pipe.offload_ar = 1024, False
    pipe.weights, pipe.runtime_sha256 = {}, "deadbeef"
    request = pipeline.SongRequest(style="s", lyrics="l")
    config = pipe.effective_config(request)
    assert config["perf"]["name"] == "fast"
    assert config["perf"]["tf32"] is True
