"""Instrumental (music-only) generation via a community LoRA.

The released YuE2-3B checkpoint is a **song** model: its model card describes
turning lyrics and a style prompt into "a complete song with vocals and
accompaniment", and an empty lyrics field still produces singing. There is no
instrumental switch in the base checkpoint.

`Mothersuperior/YuE2-instrumental-cot-full-loras` is a community AR-branch LoRA
that adds one. It is trained to read a **section plan** (bracketed tags such as
``[intro]`` / ``[verse]``) in place of lyrics, and its deltas are folded into the
base linear weights as ``W += scale * (B @ A)`` so the rest of the pipeline -
planner, sampler, VAE - is untouched.

Two caveats worth repeating to a user:

* The adapter is **CC BY-NC 4.0**, like the base checkpoint, so this is a
  non-commercial path.
* It is a community adapter, not an official release. Quality is not validated
  here beyond producing audio.
"""
from __future__ import annotations

import time
from pathlib import Path

# Trained defaults documented on the adapter card.
AR_REPO = "Mothersuperior/YuE2-instrumental-cot-full-loras"
AR_FILE = "ar_lora_inst_v3abc.safetensors"
NAR_REPO = "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
NAR_FILE = "nar_lora_joint_v4.safetensors"
DEFAULT_SCALE = 1.0

ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Sections the adapter was trained to read.
ALLOWED_SECTIONS = (
    "instrumental", "intro", "verse", "pre-chorus", "chorus", "bridge", "outro",
)
DEFAULT_SECTION_PLAN = "[instrumental]\n[intro]\n[verse]\n[chorus]\n[verse]\n[chorus]\n[outro]"


class LoRAError(RuntimeError):
    """Raised when the instrumental adapter cannot be fetched or merged."""


def normalise_section_plan(plan):
    """Validate a section plan the way the adapter was trained to read it."""
    lines = [line.strip() for line in (plan or "").replace("\\n", "\n").splitlines() if line.strip()]
    if not lines:
        raise LoRAError("Add at least one section tag, for example [instrumental].")
    for line in lines:
        if not (line.startswith("[") and line.endswith("]")):
            raise LoRAError(f"Every line must be a single bracketed tag. Got: {line!r}")
        name = line[1:-1].strip().split(" ")[0].lower()
        if name not in ALLOWED_SECTIONS:
            raise LoRAError(
                f"Unknown section {name!r}. Use only: " + ", ".join(ALLOWED_SECTIONS)
            )
    return "\n".join(lines)


def download_adapter(repo, filename, *, cache_dir=None, local_files_only=False):
    """Fetch one adapter file, with an actionable error if that is impossible."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - huggingface-hub is a hard dep
        raise LoRAError(f"huggingface_hub is unavailable: {error}") from error
    try:
        return Path(hf_hub_download(repo, filename, cache_dir=cache_dir,
                                    local_files_only=local_files_only))
    except Exception as error:
        raise LoRAError(
            f"Could not download {repo}/{filename}: {type(error).__name__}: {error}"
        ) from error


def merge_lora(model, path, attn_name, mlp_name, *, scale=DEFAULT_SCALE, replace_io=False):
    """Fold ``W += scale * (B @ A)`` into the base weights.

    Keyed by tensor name rather than list order, so an unexpected adapter layout
    fails loudly instead of merging deltas into the wrong projection.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - safetensors is a hard dep
        raise LoRAError(f"safetensors is unavailable: {error}") from error
    import torch

    tensors = load_file(str(path), device="cpu")
    merged = 0
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers):
            for block, projections in ((attn_name, ATTN_PROJECTIONS), (mlp_name, MLP_PROJECTIONS)):
                module = getattr(layer, block, None)
                if module is None:
                    raise LoRAError(f"Layer {index} has no '{block}' block for this adapter")
                for projection in projections:
                    key = f"layers.{index}.{block}.{projection}"
                    missing = [k for k in (f"{key}.lora_A", f"{key}.lora_B") if k not in tensors]
                    if missing:
                        raise LoRAError(f"Adapter is missing {missing[0]}")
                    a = tensors[f"{key}.lora_A"].float()
                    b = tensors[f"{key}.lora_B"].float()
                    linear = getattr(module, projection)
                    delta = (scale * (b @ a)).to(device=linear.weight.device, dtype=linear.weight.dtype)
                    linear.weight.add_(delta)
                    merged += 1
        if replace_io:
            # vae2llm / llm2vae ship as full replacement weights, not deltas.
            for name in ("vae2llm", "llm2vae"):
                target = getattr(model, name, None)
                if target is None:
                    raise LoRAError(f"Model has no '{name}' projection for this adapter")
                state = {k.split(".", 1)[1]: v.to(device=target.weight.device, dtype=target.weight.dtype)
                         for k, v in tensors.items() if k.startswith(name + ".")}
                if not state:
                    raise LoRAError(f"Adapter carries no replacement weights for {name}")
                target.load_state_dict(state)
    return merged


def apply_instrumental(pipe, *, cache_dir=None, local_files_only=False, include_nar=True,
                       scale=DEFAULT_SCALE):
    """Load the base weights into ``pipe`` and fold the instrumental adapters in.

    Returns ``(model, report)``. The pipeline caches the model, so the caller
    must key its own cache on whether this was applied.
    """
    if pipe.quantization != "none":
        raise LoRAError(
            "The instrumental adapter needs the unquantized BF16 linears; "
            f"quantization={pipe.quantization!r} replaces them."
        )
    start = time.perf_counter()
    model = pipe._load_model()
    report = {"ar_linears": 0, "nar_linears": 0, "seconds": 0.0, "scale": scale}

    ar_path = download_adapter(AR_REPO, AR_FILE, cache_dir=cache_dir,
                               local_files_only=local_files_only)
    report["ar_linears"] = merge_lora(model, ar_path, "self_attn", "mlp", scale=scale)

    if include_nar:
        try:
            nar_path = download_adapter(NAR_REPO, NAR_FILE, cache_dir=cache_dir,
                                        local_files_only=local_files_only)
            report["nar_linears"] = merge_lora(model, nar_path, "nar_self_attn", "nar_mlp",
                                               scale=scale, replace_io=True)
        except LoRAError as error:
            # The AR adapter alone still yields instrumental audio; the NAR
            # adapter mostly affects the acoustic decoder.
            report["nar_error"] = str(error)

    report["seconds"] = round(time.perf_counter() - start, 1)
    return model, report
