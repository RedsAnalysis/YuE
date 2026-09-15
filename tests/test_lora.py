"""Instrumental adapter: plan validation and the merge contract.

The merge itself is exercised on the GPU by ``test_app.py``; these cover the
pure logic and the failure paths that must not corrupt base weights.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from yue2 import lora


def test_section_plan_accepts_the_documented_tags():
    plan = lora.normalise_section_plan("[instrumental]\n[intro]\n[verse]\n[chorus]\n[outro]")
    assert plan == "[instrumental]\n[intro]\n[verse]\n[chorus]\n[outro]"


def test_section_plan_accepts_qualified_tags():
    assert lora.normalise_section_plan("[verse 1]\n[chorus]") == "[verse 1]\n[chorus]"


def test_section_plan_tolerates_escaped_newlines_and_blank_lines():
    assert lora.normalise_section_plan("[intro]\\n\\n[verse]") == "[intro]\n[verse]"


@pytest.mark.parametrize("plan,message", [
    ("", "at least one section"),
    ("   ", "at least one section"),
    ("verse", "bracketed tag"),
    ("[unknown]", "Unknown section"),
    ("[verse]\nnot a tag", "bracketed tag"),
])
def test_section_plan_rejects_bad_input(plan, message):
    with pytest.raises(lora.LoRAError, match=message):
        lora.normalise_section_plan(plan)


def test_default_plan_is_itself_valid():
    assert lora.normalise_section_plan(lora.DEFAULT_SECTION_PLAN)


def test_every_allowed_section_is_accepted():
    for section in lora.ALLOWED_SECTIONS:
        assert lora.normalise_section_plan(f"[{section}]")


def test_adapter_locations_are_the_published_ones():
    assert lora.AR_REPO == "Mothersuperior/YuE2-instrumental-cot-full-loras"
    assert lora.AR_FILE == "ar_lora_inst_v3abc.safetensors"
    assert lora.DEFAULT_SCALE == 1.0


def test_download_failure_is_reported_with_the_repo(tmp_path, monkeypatch):
    import huggingface_hub

    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(lora.LoRAError, match="Could not download"):
        lora.download_adapter("a/b", "c.safetensors")


def test_merge_rejects_an_adapter_missing_its_tensors(tmp_path):
    """A wrong-shaped adapter must raise, not silently merge into the model."""
    import torch
    from safetensors.torch import save_file

    path = tmp_path / "bad.safetensors"
    save_file({"unrelated.weight": torch.zeros(2, 2)}, str(path))

    class _Layer:
        class self_attn:
            class q_proj:
                weight = torch.zeros(2, 2)

    class _Model:
        model = type("M", (), {"layers": [_Layer()]})()

    with pytest.raises(lora.LoRAError, match="missing"):
        lora.merge_lora(_Model(), path, "self_attn", "mlp")


def test_merge_folds_the_delta_into_the_base_weight(tmp_path):
    """W += scale * (B @ A) on a tiny model, checked against a hand calculation."""
    import torch
    from safetensors.torch import save_file

    torch.manual_seed(0)
    in_features, out_features, rank = 8, 4, 2
    a = torch.randn(rank, in_features)        # lora_A is [rank, in]
    b = torch.randn(out_features, rank)       # lora_B is [out, rank]
    scale = 0.5

    state = {}
    for block, projections in (("self_attn", lora.ATTN_PROJECTIONS),
                               ("mlp", lora.MLP_PROJECTIONS)):
        for projection in projections:
            state[f"layers.0.{block}.{projection}.lora_A"] = a.clone()
            state[f"layers.0.{block}.{projection}.lora_B"] = b.clone()
    path = tmp_path / "ar.safetensors"
    save_file(state, str(path))

    class _Projections(torch.nn.Module):
        def __init__(self, names):
            super().__init__()
            for name in names:
                setattr(self, name, torch.nn.Linear(in_features, out_features, bias=False))

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Projections(lora.ATTN_PROJECTIONS)
            self.mlp = _Projections(lora.MLP_PROJECTIONS)

    class _Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([_Layer()])

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Backbone()

    model = _Model()
    target = model.model.layers[0].self_attn.q_proj
    before = target.weight.detach().clone()
    merged = lora.merge_lora(model, path, "self_attn", "mlp", scale=scale)

    assert merged == len(lora.ATTN_PROJECTIONS) + len(lora.MLP_PROJECTIONS)
    torch.testing.assert_close(target.weight, before + scale * (b @ a))
    # Every named projection moved by the same delta; nothing else changed.
    assert not torch.equal(model.model.layers[0].mlp.gate_proj.weight, torch.zeros_like(before))


# --------------------------------------------------------------------------- #
# GPU end-to-end (downloads the adapter on first run)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_instrumental_adapter_merges_and_generates():
    """The headline claim: the adapter loads, folds in, and makes audio."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the instrumental adapter")

    repo = Path(__file__).resolve().parents[1]
    model_dir, vae_dir = repo / "models" / "YuE2-3B", repo / "models" / "YuE2-Vae"
    if not (model_dir / "model.safetensors").is_file():
        pytest.skip("local YuE2 weights are not present under models/")

    from yue2 import YuE2Pipeline
    from yue2.protocol import Sampling

    pipe = YuE2Pipeline.from_pretrained(str(model_dir), vae=str(vae_dir), device="cuda",
                                        perf="balanced", local_files_only=True, progress=False)
    try:
        model, report = lora.apply_instrumental(pipe)
        # Real checkpoint: 28 layers x 7 projections per branch.
        assert report["ar_linears"] == 196, report
        assert report["seconds"] >= 0
        assert model is pipe._model

        result = pipe(style="Cinematic orchestral, sweeping strings, no vocals",
                      lyrics=lora.normalise_section_plan(lora.DEFAULT_SECTION_PLAN),
                      cot="full", seed=7,
                      abc_sampling=Sampling(max_tokens=120, min_tokens=8),
                      semantic_sampling=Sampling(max_tokens=250, min_tokens=8))
        assert result.audio.ndim == 2 and result.audio.shape[1] == 2
        assert abs(result.audio).max() > 1e-3, "instrumental output is silent"
        assert len(result.audio) / result.sample_rate > 1
    finally:
        pipe.close()



def test_merge_rejects_a_layer_without_the_named_block(tmp_path):
    import torch
    from safetensors.torch import save_file

    state = {}
    for projection in lora.ATTN_PROJECTIONS:
        state[f"layers.0.self_attn.{projection}.lora_A"] = torch.zeros(2, 8)
        state[f"layers.0.self_attn.{projection}.lora_B"] = torch.zeros(4, 2)
    path = tmp_path / "attn-only.safetensors"
    save_file(state, str(path))

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for projection in lora.ATTN_PROJECTIONS:
                setattr(self, projection, torch.nn.Linear(8, 4, bias=False))

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Block()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = type("M", (), {"layers": [_Layer()]})()

    with pytest.raises(lora.LoRAError, match="has no 'mlp' block"):
        lora.merge_lora(_Model(), path, "self_attn", "mlp")
