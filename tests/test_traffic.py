"""Tests for TrafficGenerator — time-of-day load variation."""

import numpy as np
import pytest

from llm_router_env.traffic import TrafficGenerator


def make_generator(seed: int = 0) -> TrafficGenerator:
    return TrafficGenerator(rng=np.random.default_rng(seed))


class TestLoadFactor:
    def test_load_factor_range(self):
        gen = make_generator()
        for t in np.linspace(0.0, 1.0, 20):
            lf = gen.load_factor(t)
            assert 0.0 <= lf <= 1.0, f"load_factor({t}) = {lf} out of [0, 1]"

    def test_load_factor_peak_business_hours(self):
        """Load factor near 9am (0.375) should exceed load at 3am (0.125)."""
        rng = np.random.default_rng(42)
        gen = TrafficGenerator(rng=rng)
        # Suppress noise by averaging many samples
        n = 200
        morning = np.mean([gen.load_factor(0.375) for _ in range(n)])
        night = np.mean([gen.load_factor(0.125) for _ in range(n)])
        assert morning > night, (
            f"Expected higher load at 9am ({morning:.3f}) than at 3am ({night:.3f})"
        )


class TestSampleUsesTimeOfDay:
    def test_sample_complexity_higher_at_peak(self):
        """
        Average complexity sampled during business hours (load ~0.9) should
        exceed average complexity at off-peak hours (load ~0.1).
        """
        n = 500
        rng_peak = np.random.default_rng(0)
        rng_off = np.random.default_rng(0)
        gen_peak = TrafficGenerator(rng=rng_peak)
        gen_off = TrafficGenerator(rng=rng_off)

        # 9am ≈ 0.375 (sinusoid peaks here) → high load
        complexity_peak = np.mean([gen_peak.sample(0.375).complexity for _ in range(n)])
        # 3am ≈ 0.125 (sinusoid trough) → low load
        complexity_off = np.mean([gen_off.sample(0.125).complexity for _ in range(n)])

        assert complexity_peak > complexity_off, (
            f"Expected higher complexity at peak ({complexity_peak:.3f}) "
            f"than off-peak ({complexity_off:.3f})"
        )

    def test_sample_quality_required_higher_at_peak(self):
        """
        Average quality_required during business hours should exceed off-peak.
        """
        n = 500
        rng_peak = np.random.default_rng(1)
        rng_off = np.random.default_rng(1)
        gen_peak = TrafficGenerator(rng=rng_peak)
        gen_off = TrafficGenerator(rng=rng_off)

        quality_peak = np.mean([gen_peak.sample(0.375).quality_required for _ in range(n)])
        quality_off = np.mean([gen_off.sample(0.125).quality_required for _ in range(n)])

        assert quality_peak > quality_off, (
            f"Expected higher quality_required at peak ({quality_peak:.3f}) "
            f"than off-peak ({quality_off:.3f})"
        )

    def test_sample_returns_valid_prompt(self):
        gen = make_generator()
        for t in [0.0, 0.25, 0.375, 0.5, 0.75, 1.0]:
            prompt = gen.sample(t)
            assert 0.0 <= prompt.complexity <= 1.0, f"complexity out of range at t={t}"
            assert 0.0 <= prompt.length <= 1.0, f"length out of range at t={t}"
            assert 0.0 <= prompt.quality_required <= 1.0, f"quality_required out of range at t={t}"
