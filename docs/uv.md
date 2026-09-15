# Working with uv

This checkout is a [uv](https://docs.astral.sh/uv/) project. `uv.lock` pins the
entire dependency graph, `.python-version` pins the interpreter, and
`[dependency-groups]` in `pyproject.toml` decides what a plain `uv sync`
installs.

## Quick start

```bash
make setup     # create .venv from uv.lock, then download the models
make app       # run the Gradio UI (first free port from 7860)
```

The app serves on the first free port from 7860 upward and prints the URL, so a
second instance or a leftover server will not stop it from starting. Pass
`--port N` to require an exact port; if that one is taken the app exits with a
message saying so.

`make help` lists every target. If you prefer to call uv directly, see
[Calling uv directly](#calling-uv-directly) first.

## What a plain `uv sync` gives you

| Group / extra | Contents | Installed by default |
|---|---|---|
| `project.dependencies` | torch, transformers, huggingface-hub, safetensors, tiktoken, numpy, soundfile, accelerate | yes |
| group `dev` | pytest | yes |
| group `ui` | gradio | yes |
| group `cover` | torchaudio, scipy, mir_eval, pretty_midi, mido | yes |
| extra `fast` | vllm, triton | **no** |
| extra `test` | pytest (pip equivalent of the `dev` group) | — |
| extra `ui` | gradio (pip equivalent of the `ui` group) | — |
| extra `cover` | the transcription helpers (pip equivalent of the `cover` group) | — |

`ui` and `cover` are default groups because `app.py` is the main entry point
here and the Cover tab needs the transcription helpers. The optional vLLM
backend is never installed implicitly — it is roughly 2 GB:

```bash
make fast          # or: uv sync --extra fast
```

With the `fast` extra installed the suite reports one extra passing test; without
it that test skips with `Optional vLLM package is not installed`. Nothing else
changes.

## Common commands

```bash
uv sync                        # make .venv match uv.lock
uv lock                        # re-resolve and rewrite uv.lock
uv add <package>               # add a runtime dependency
uv add --group dev <package>   # add a development dependency
uv run python app.py           # run anything inside the environment
uv run pytest tests/           # the test suite
uv tree                        # inspect the resolved graph
```

`uv run` re-syncs automatically before executing, so the environment cannot drift
from the lockfile.

## Calling uv directly

Three variables keep uv state inside the checkout. That makes the project
self-contained and is required wherever `$HOME` is not writable — the DSH
sandbox, for example, mounts a read-only `~/.cache`, and uv would otherwise fail
with `Read-only file system (os error 30)`.

```bash
export UV_CACHE_DIR="$PWD/.uvcache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uvpython"
export UV_ENV_FILE="$PWD/.env"
```

The `Makefile` exports all three for you, which is why `make <target>` needs no
setup.

`.env` holds `HF_HOME` so the Hugging Face cache also lands inside the checkout.
`make setup` writes it; `.env.example` shows the format. Absolute paths are
required because uv does not expand variables inside `.env`.

Note that `.env` is read by uv for the *subprocess* environment only. It does not
configure uv itself, which is why `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` must
be real environment variables rather than entries in `.env`. `cache-dir` in
`[tool.uv]` is likewise not applied early enough to help: uv initialises its
cache before it reads project configuration.

## Interpreter

`requires-python` stays at `>=3.10` to match the upstream package, while
`.python-version` pins 3.12 for development. uv downloads the interpreter into
`.uvpython/` on first sync, so no system Python 3.12 is needed.
