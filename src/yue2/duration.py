"""Audio-length arithmetic for the request protocol.

YuE2's semantic stage emits one codec frame per 1/25 second of audio. Two
independent sources agree on that rate:

* ``YuE2VAEConfig`` decodes at ``sample_rate=48000`` with
  ``downsampling_ratio=1920``, so 48000 / 1920 = 25 frames per second.
* The MERT-v2 backbone behind SheetSage2 declares ``"frame_rate": 25.0`` in its
  config, and measurement confirms it: 600 tokens decoded to 24.00 s, 2000 to
  80.00 s, and 2591 to 103.60 s.

So a requested duration converts to a token budget with exact integer
arithmetic, and no empirical fudge factor is involved.
"""
from __future__ import annotations

SAMPLE_RATE = 48000
DOWNSAMPLING_RATIO = 1920
FRAMES_PER_SECOND = SAMPLE_RATE // DOWNSAMPLING_RATIO  # 25

# The semantic stage caps at 9000 tokens; CONTEXT is 24576 but the prefix and
# the generation budget must both fit, and 9000 is the released ceiling.
MAX_TOKENS = 9000
MIN_TOKENS = 1

MAX_SECONDS = MAX_TOKENS // FRAMES_PER_SECOND  # 360


def seconds_to_tokens(seconds, *, maximum=MAX_TOKENS, minimum=MIN_TOKENS):
    """Convert a requested duration to a semantic token budget.

    The result is clamped to ``[minimum, maximum]`` rather than raising, because
    this backs a UI slider whose range may not match the model's limits.
    """
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TypeError("seconds must be a number")
    tokens = int(round(float(seconds) * FRAMES_PER_SECOND))
    return max(int(minimum), min(int(maximum), tokens))


def tokens_to_seconds(tokens):
    """Convert a token count back to its exact audio duration in seconds."""
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise TypeError("tokens must be an integer")
    return tokens / FRAMES_PER_SECOND


def fit_sampling_bounds(min_tokens, max_tokens):
    """Return ``(min_tokens, max_tokens)`` that satisfy ``Sampling`` validation.

    ``Sampling`` requires ``0 <= min_tokens <= max_tokens``. A user who asks for
    a 5-second song produces a ``max_tokens`` below the checkpoint's default
    ``min_tokens`` of 200, so the floor has to come down with it.
    """
    min_tokens = max(0, int(min_tokens))
    max_tokens = max(1, int(max_tokens))
    return min(min_tokens, max_tokens), max_tokens
