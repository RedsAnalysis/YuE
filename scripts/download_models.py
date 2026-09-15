#!/usr/bin/env python
"""Download YuE2-3B and the YuE2 VAE into ./models for fully-offline use.

Only the reviewed public model files are fetched (see yue2.storage.MODEL_FILES);
example audio, wheels and figures are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from yue2.storage import MODEL_FILES, MODEL_LICENSES  # noqa: E402

ALLOW = (
    sorted(MODEL_FILES)
    + ["model-?????-of-?????.safetensors"]
    + ["licenses/" + name for name in sorted(MODEL_LICENSES)]
)

TARGETS = {
    "YuE2-3B": "m-a-p/YuE2-3B",
    "YuE2-Vae": "m-a-p/YuE2-Vae",
}


def human(n: int) -> str:
    return f"{n / 2**30:.2f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(REPO_ROOT / "models"))
    ap.add_argument("--only", choices=sorted(TARGETS), action="append")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf"))
    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    names = args.only or sorted(TARGETS)
    grand = time.perf_counter()
    for name in names:
        repo = TARGETS[name]
        out = dest / name
        print(f"\n=== {repo} -> {out} ===", flush=True)
        start = time.perf_counter()
        if (out / "config.json").is_file() and any(out.glob("*.safetensors")):
            print("  already present, re-verifying via snapshot_download", flush=True)
        snapshot_download(
            repo,
            revision=args.revision,
            local_dir=out,
            allow_patterns=ALLOW,
        )
        total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(
            f"  done in {time.perf_counter() - start:.1f}s, {human(total)} on disk",
            flush=True,
        )
    print(f"\nAll downloads finished in {time.perf_counter() - grand:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
