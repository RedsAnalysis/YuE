"""Hardware-aware performance profiles for YuE2 inference.

The package defaults in :class:`yue2.pipeline.YuE2Pipeline` are deliberately
deterministic: TF32, cuDNN autotuning and reduced-precision reductions are all
disabled so that a seeded run reproduces the recorded coordinates. Those
defaults leave measurable throughput on the table on consumer Ampere parts such
as the RTX 3090 (compute capability 8.6).

This module separates the two concerns:

* ``reference`` -- byte-for-byte the historical backend flags.
* ``balanced`` -- numerics-preserving wins only (fused QKV/gate-up projections
  and an explicit fused attention backend). Every GEMM computes the same
  products; only launch count and kernel selection change.
* ``fast`` -- additionally enables TF32 matmul and cuDNN autotuning. These
  *do* change floating-point results and are therefore opt-in.

Nothing here is applied implicitly; the caller passes a profile to
``YuE2Pipeline(perf=...)``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch

# Compute capability at which the experimental FP8 kernels become usable.
FP8_MIN_CAPABILITY = (8, 9)

PROFILE_NAMES = ("reference", "balanced", "fast")


@dataclass(frozen=True)
class PerfProfile:
    """A named bundle of backend and graph options."""

    name: str = "reference"
    attention_backend: str = "auto"
    fuse_projections: bool = False
    cuda_graph: bool = True
    tf32: bool = False
    cudnn_benchmark: bool = False
    vae_core_frames: int | None = None
    changes_numerics: bool = False

    def __post_init__(self):
        if self.attention_backend not in {"auto", "flash", "cudnn", "sdpa"}:
            raise ValueError("attention_backend must be auto, flash, cudnn, or sdpa")
        if self.vae_core_frames is not None and (
            isinstance(self.vae_core_frames, bool)
            or not isinstance(self.vae_core_frames, int)
            or self.vae_core_frames < 1
        ):
            raise ValueError("vae_core_frames must be a positive integer or None")

    def to_dict(self):
        return asdict(self)


_PROFILES = {
    # Historical backend flags exactly as the pinned release set them.
    "reference": PerfProfile(name="reference"),
    # Same arithmetic, fewer kernel launches, explicit fused attention.
    "balanced": PerfProfile(
        name="balanced",
        attention_backend="auto",
        fuse_projections=True,
        cuda_graph=True,
        changes_numerics=False,
    ),
    # Consumer-Ampere throughput preset; opt-in because results shift.
    "fast": PerfProfile(
        name="fast",
        attention_backend="auto",
        fuse_projections=True,
        cuda_graph=True,
        tf32=True,
        cudnn_benchmark=True,
        changes_numerics=True,
    ),
}


def profile(name):
    """Return a built-in profile by name."""
    if name not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}")
    return _PROFILES[name]


def profile_names():
    return list(PROFILE_NAMES)


def detect_hardware(device=None):
    """Describe the active accelerator without assuming a GPU is present."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    device = torch.device(device)
    report = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        capability = (props.major, props.minor)
        report.update(
            name=props.name,
            index=index,
            total_memory_gib=round(props.total_memory / 2**30, 2),
            capability=list(capability),
            bf16_supported=torch.cuda.is_bf16_supported(),
            fp8_supported=capability >= FP8_MIN_CAPABILITY,
            multi_processor_count=props.multi_processor_count,
        )
        # A 24 GiB card fits the reference VAE tiling; smaller cards need tiles.
        report["suggested_vae_core_frames"] = 512 if props.total_memory < 16 * 2**30 else 1024
    else:
        report.update(name=device.type, bf16_supported=False, fp8_supported=False)
    report["flash_attention"] = _flash_attention_available(device)
    return report


def _flash_attention_available(device):
    if torch.device(device).type != "cuda":
        return False
    if not hasattr(torch.ops.aten, "_flash_attention_forward"):
        return False
    try:
        schema = str(torch.ops.aten._flash_attention_forward.default._schema)
    except (AttributeError, RuntimeError):
        return False
    return "seqused_k" in schema


def apply_profile(performer, device):
    """Set the process-global torch backend flags for ``performer``."""
    if not isinstance(performer, PerfProfile):
        raise TypeError("apply_profile expects a PerfProfile")
    torch.backends.cudnn.benchmark = bool(performer.cudnn_benchmark)
    torch.backends.cudnn.deterministic = not performer.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = bool(performer.tf32)
    torch.backends.cudnn.allow_tf32 = bool(performer.tf32)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("high" if performer.tf32 else "highest")
    return performer


def default_profile_for(device):
    """``balanced`` on CUDA, ``reference`` elsewhere.

    ``fast`` is never selected implicitly because it changes results.
    """
    device = torch.device(device)
    return profile("balanced" if device.type == "cuda" else "reference")


def validated_profile(performer, hardware, quantization="none"):
    """Return ``(profile, list_of_notes)`` after checking it against hardware."""
    if not isinstance(performer, PerfProfile):
        raise TypeError("validated_profile expects a PerfProfile")
    notes = []
    if performer.fuse_projections and not performer.cuda_graph:
        notes.append("fuse_projections is ignored without CUDA graphs")
        performer = replace(performer, fuse_projections=False)
    if performer.cuda_graph and not hardware.get("cuda_available"):
        notes.append("CUDA graphs unavailable on this device; using eager decode")
        performer = replace(performer, cuda_graph=False)
    if hardware.get("device_type") == "cuda" and not hardware.get("bf16_supported", False):
        notes.append("device does not report BF16 support")
    if quantization == "fp8" and not hardware.get("fp8_supported", False):
        notes.append(
            "FP8 requires compute capability >= 8.9; "
            f"this device is {hardware.get('capability')}"
        )
    if performer.attention_backend == "flash" and not hardware.get("flash_attention"):
        notes.append("requested flash attention is unavailable; GraphAR will reject it")
    if performer.changes_numerics:
        notes.append("profile enables TF32/autotuning and changes floating-point results")
    return performer, notes
