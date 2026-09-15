"""Audio-length arithmetic.

The conversion is exact, so these are equality checks rather than tolerances.
"""
from __future__ import annotations

import pytest

from yue2 import duration


def test_frame_rate_follows_from_the_vae_configuration():
    """YuE2VAEConfig decodes at 48000 Hz with a 1920x downsampling ratio."""
    assert duration.SAMPLE_RATE == 48000
    assert duration.DOWNSAMPLING_RATIO == 1920
    assert duration.FRAMES_PER_SECOND == 25


@pytest.mark.parametrize("seconds,tokens", [
    (1, 25), (8, 200), (24, 600), (80, 2000), (90, 2250), (360, 9000),
])
def test_seconds_to_tokens(seconds, tokens):
    assert duration.seconds_to_tokens(seconds) == tokens


@pytest.mark.parametrize("tokens,seconds", [
    (0, 0.0), (25, 1.0), (600, 24.0), (2000, 80.0), (2049, 81.96),
])
def test_tokens_to_seconds(tokens, seconds):
    assert duration.tokens_to_seconds(tokens) == pytest.approx(seconds)


def test_round_trip_is_lossless_on_whole_seconds():
    for seconds in range(1, duration.MAX_SECONDS + 1):
        assert duration.tokens_to_seconds(duration.seconds_to_tokens(seconds)) == seconds


def test_conversion_clamps_to_the_released_ceiling():
    assert duration.MAX_SECONDS == 360
    assert duration.seconds_to_tokens(10_000) == duration.MAX_TOKENS == 9000


def test_conversion_clamps_below_and_rounds():
    assert duration.seconds_to_tokens(0) == 1
    assert duration.seconds_to_tokens(-5) == 1
    assert duration.seconds_to_tokens(3.9) == 98   # 97.5 rounds to 98


def test_seconds_must_be_a_number():
    with pytest.raises(TypeError):
        duration.seconds_to_tokens("80")
    with pytest.raises(TypeError):
        duration.seconds_to_tokens(True)


def test_tokens_must_be_an_integer():
    with pytest.raises(TypeError):
        duration.tokens_to_seconds(80.0)
    with pytest.raises(TypeError):
        duration.tokens_to_seconds(True)


@pytest.mark.parametrize("floor,budget,expected", [
    (200, 3000, (200, 3000)),   # ordinary: floor is below the budget
    (200, 125, (125, 125)),     # 5s request: floor has to come down
    (0, 500, (0, 500)),         # floor disabled
    (-10, 500, (0, 500)),       # floor cannot go negative
    (9000, 100, (100, 100)),    # floor above the budget is clamped
    (50, 0, (1, 1)),            # budget cannot go to zero, and the floor follows it
])
def test_fit_sampling_bounds_satisfies_the_sampling_contract(floor, budget, expected):
    result = duration.fit_sampling_bounds(floor, budget)
    assert result == expected
    assert 0 <= result[0] <= result[1]


def test_fit_sampling_bounds_output_is_accepted_by_sampling():
    from yue2.protocol import Sampling

    for seconds in (1, 5, 30, 120, 400):
        floor, budget = duration.fit_sampling_bounds(200, duration.seconds_to_tokens(seconds))
        sampling = Sampling(min_tokens=floor, max_tokens=budget)
        assert sampling.min_tokens <= sampling.max_tokens
