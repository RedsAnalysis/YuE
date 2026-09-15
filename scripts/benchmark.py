#!/usr/bin/env python
"""Measure YuE2 inference time end to end on the local GPU.

Reports per-stage timings (planning, semantic AR, NAR flow matching, VAE
decode), throughput, and peak VRAM so that optimisation profiles can be
compared on the same machine. Results are appended as JSON so runs accumulate
into a record instead of overwriting each other.

Examples
--------
Quick wiring check (~1 minute of AR) on one prompt::

    python scripts/benchmark.py --quick

Compare every profile on the same seed::

    python scripts/benchmark.py --profiles reference,balanced,fast
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_STYLE = (
    "English, warm piano pop, expressive female voice, acoustic piano, "
    "rounded bass and light drums, lyrical memorable melody, 88 BPM"
)
DEFAULT_LYRICS = (
    "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n"
    "[Chorus]\nLet the day come into view\nEvery road begins with you"
)


QUICK_SEMANTIC_TOKENS = 600
QUICK_ABC_TOKENS = 400


def build_request(args):
    """Translate --quick/--max-tokens into explicit sampling overrides.

    ``None`` means "use the checkpoint default". A shortened budget produces a
    shorter song; it is a timing probe, not a quality sample.
    """
    from yue2.protocol import Sampling

    abc_sampling = semantic_sampling = None
    if args.quick:
        abc_sampling = Sampling(max_tokens=QUICK_ABC_TOKENS, min_tokens=8)
        semantic_sampling = Sampling(max_tokens=QUICK_SEMANTIC_TOKENS, min_tokens=8)
    elif args.max_tokens:
        abc_sampling = Sampling(max_tokens=min(args.max_tokens, 4096), min_tokens=8)
        semantic_sampling = Sampling(max_tokens=args.max_tokens, min_tokens=8)
    return abc_sampling, semantic_sampling


def stage_timings(result, pipe, wall_seconds, peak_bytes, warm_decode_seconds=None):
    timing = result.timing
    semantic = timing.get("semantic", {})
    abc = timing.get("abc", {})
    nar_seconds = timing.get("nar_seconds", 0.0)
    vae_seconds = timing.get("vae_seconds", 0.0)
    return {
        "e2e_seconds": round(timing.get("e2e_seconds", wall_seconds), 3),
        "wall_seconds": round(wall_seconds, 3),
        "abc_seconds": round(abc.get("seconds", 0.0), 3),
        "abc_tokens": abc.get("output_tokens"),
        "abc_tps": round(abc.get("output_tps", 0.0), 2),
        "abc_ttft_seconds": round(abc.get("ttft_seconds") or 0.0, 3),
        "abc_attention": abc.get("attention"),
        "semantic_seconds": round(semantic.get("seconds", 0.0), 3),
        "semantic_tokens": semantic.get("output_tokens"),
        "semantic_tps": round(semantic.get("output_tps", 0.0), 2),
        "semantic_ttft_seconds": round(semantic.get("ttft_seconds") or 0.0, 3),
        "semantic_attention": semantic.get("attention"),
        "semantic_cfg_branches": semantic.get("cfg_branches"),
        "fuse_projections": semantic.get("fuse_projections"),
        "nar_seconds": round(nar_seconds, 3),
        "vae_seconds": round(vae_seconds, 3),
        # vae_seconds includes a cold decoder load; this is the pure decode.
        "vae_warm_decode_seconds": round(warm_decode_seconds, 3) if warm_decode_seconds is not None else None,
        "vae_load_seconds": round(max(vae_seconds - (warm_decode_seconds or 0.0), 0.0), 3),
        "audio_seconds": round(len(result.audio) / result.sample_rate, 2),
        "sample_rate": result.sample_rate,
        "load_seconds": {k: round(v, 3) for k, v in (pipe.load_timing or {}).items()},
        "peak_vram_gib": round(peak_bytes / 2**30, 3),
        "truncated": result.truncated,
    }


def run_once(args, profile_name):
    import torch
    from yue2 import YuE2Pipeline
    from yue2.perf import detect_hardware, profile as named_profile, validated_profile

    hardware = detect_hardware()
    performer = named_profile(profile_name)
    performer, notes = validated_profile(performer, hardware, quantization="none")
    for note in notes:
        print(f"  ! {note}", file=sys.stderr)

    model_dir = Path(args.model or REPO_ROOT / "models" / "YuE2-3B")
    vae_dir = Path(args.vae or REPO_ROOT / "models" / "YuE2-Vae")
    abc_sampling, semantic_sampling = build_request(args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained(
        str(model_dir) if model_dir.is_dir() else str(model_dir),
        vae=str(vae_dir) if vae_dir.is_dir() else str(vae_dir),
        device=args.device,
        memory_budget_gib=args.budget,
        backend=args.backend,
        quantization=args.quantization,
        perf=performer,
        local_files_only=model_dir.is_dir(),
        progress=args.progress,
    )
    construction_seconds = time.perf_counter() - load_start
    warm_decode = None
    try:
        wall_start = time.perf_counter()
        result = pipe(
            style=args.style,
            lyrics=args.lyrics,
            cot=args.cot,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            abc_sampling=abc_sampling,
            semantic_sampling=semantic_sampling,
        )
        wall = time.perf_counter() - wall_start
        # Decode the same latents again: the first decode paid the decoder load,
        # so this isolates the actual VAE cost for fair profile comparison.
        decode_start = time.perf_counter()
        pipe.decode(result.latents)
        warm_decode = time.perf_counter() - decode_start
    finally:
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        pipe.close()

    record = {
        "profile": performer.to_dict(),
        "profile_notes": notes,
        "backend": args.backend,
        "quantization": args.quantization,
        "device": str(pipe.device),
        "cot": args.cot,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "quick": bool(args.quick),
        "construction_seconds": round(construction_seconds, 3),
        "timings": stage_timings(result, pipe, wall, peak, warm_decode),
    }
    return record, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", default="balanced", help="comma-separated profile names")
    ap.add_argument("--model", default=None)
    ap.add_argument("--vae", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--budget", type=float, default=24)
    ap.add_argument("--backend", default="torch", choices=("torch", "torch-eager", "vllm"))
    ap.add_argument("--quantization", default="none", choices=("none", "fp8"))
    ap.add_argument("--cot", default="full", choices=("full", "melody", "off"))
    ap.add_argument("--seed", type=int, default=831001)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap semantic generation; omit to use the checkpoint default (9000)")
    ap.add_argument("--quick", action="store_true", help="short budgets for a fast wiring check")
    ap.add_argument("--style", default=DEFAULT_STYLE)
    ap.add_argument("--lyrics", default=DEFAULT_LYRICS)
    ap.add_argument("--output", default=str(REPO_ROOT / "benches"))
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--no-progress", dest="progress", action="store_false")
    args = ap.parse_args()

    from yue2.perf import detect_hardware

    hardware = detect_hardware()
    print("=== Hardware ===")
    for key, value in hardware.items():
        print(f"  {key}: {value}")

    records = []
    for name in [p.strip() for p in args.profiles.split(",") if p.strip()]:
        print(f"\n=== Profile: {name} ===", flush=True)
        record, result = run_once(args, name)
        records.append(record)
        t = record["timings"]
        print(
            f"  e2e {t['e2e_seconds']:8.1f}s | abc {t['abc_seconds']:7.1f}s "
            f"({t['abc_tokens']} tok, {t['abc_tps']} tok/s) | "
            f"semantic {t['semantic_seconds']:7.1f}s ({t['semantic_tokens']} tok, {t['semantic_tps']} tok/s)",
            flush=True,
        )
        print(
            f"  nar {t['nar_seconds']:8.1f}s | vae(cold) {t['vae_seconds']:6.1f}s | "
            f"vae(warm) {t['vae_warm_decode_seconds'] or 0:6.2f}s | "
            f"audio {t['audio_seconds']:.1f}s | peak VRAM {t['peak_vram_gib']:.2f} GiB",
            flush=True,
        )
        if args.save_audio:
            out = Path(args.output) / f"audio-{name}.flac"
            out.parent.mkdir(parents=True, exist_ok=True)
            result.save(out)
            print(f"  wrote {out}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor(),
        },
        "hardware": hardware,
        "cli": {k: v for k, v in vars(args).items() if k != "lyrics"},
        "records": records,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output / f"bench-{stamp}.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    latest = output / "latest.json"
    latest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
