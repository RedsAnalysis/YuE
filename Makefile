# uv-driven tasks for the YuE2 working copy.
#
# The three exported variables below keep all uv state inside the checkout, so
# the project is self-contained and works where $HOME is not writable. Set the
# same three in your shell if you would rather call `uv` directly; see
# docs/uv.md.

SHELL := /bin/bash
.DEFAULT_GOAL := help

export UV_CACHE_DIR := $(CURDIR)/.uvcache
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uvpython
export UV_ENV_FILE := $(CURDIR)/.env

PY := uv run python
MODEL ?= models/YuE2-3B
VAE ?= models/YuE2-Vae

.PHONY: help setup sync lock fast app test test-all bench download doctor clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

.env: ## Create the local env file used by uv run
	@printf 'HF_HOME=%s/.hf\n' "$(CURDIR)" > $@
	@echo "wrote .env (HF_HOME=$(CURDIR)/.hf)"

setup: .env sync download ## Create the environment and fetch the models

sync: .env ## Install/sync the environment from uv.lock
	uv sync

lock: ## Re-resolve dependencies and update uv.lock
	uv lock

fast: ## Also install the optional vLLM backend (the `fast` extra)
	uv sync --extra fast

app: .env ## Run the Gradio UI
	$(PY) app.py --model $(MODEL) --vae $(VAE)

test: .env ## Fast test suite (no GPU)
	$(PY) -m pytest tests/ -m "not slow"

test-all: .env ## Full suite, including the GPU timing tests
	$(PY) -m pytest tests/

bench: .env ## Compare the performance profiles on this machine
	$(PY) scripts/benchmark.py --profiles reference,balanced,fast --max-tokens 2000

download: .env ## Fetch YuE2-3B and the VAE into models/
	$(PY) scripts/download_models.py

doctor: .env ## Report the detected hardware and dependency versions
	$(PY) -m yue2.cli doctor

clean: ## Remove the virtualenv and the uv caches
	rm -rf .venv .uvcache
