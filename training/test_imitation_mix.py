"""Unit tests for imitation_mix decay schedule (targets #113).

Verifies the linear decay from imitation_mix_start to imitation_mix_end
is monotone non-increasing and hits 0.0 at the expected version.
"""

from __future__ import annotations

import pytest

from training.config import AsyncConfig


class TestImitationMixDecay:
    def setup_method(self):
        self.cfg = AsyncConfig()

    def test_v0_returns_start(self):
        """Version 0 (before training) should return the start value."""
        assert self.cfg.imitation_mix_for_version(0) == self.cfg.imitation_mix_start

    def test_v1_returns_start(self):
        """Version 1 (first model) should return the start value."""
        assert self.cfg.imitation_mix_for_version(1) == self.cfg.imitation_mix_start

    def test_end_version_returns_end(self):
        """At the decay end version, should return the end value."""
        v_end = self.cfg.imitation_mix_decay_end_version
        assert self.cfg.imitation_mix_for_version(v_end) == self.cfg.imitation_mix_end

    def test_past_end_returns_end(self):
        """Well past the end version, should still return the end value."""
        v_far = self.cfg.imitation_mix_decay_end_version + 100
        assert self.cfg.imitation_mix_for_version(v_far) == self.cfg.imitation_mix_end

    def test_hits_zero(self):
        """With default config (end=0.0), the schedule must reach 0.0."""
        v_end = self.cfg.imitation_mix_decay_end_version
        assert self.cfg.imitation_mix_for_version(v_end) == 0.0

    def test_monotone_non_increasing(self):
        """The schedule must be monotone non-increasing over all versions."""
        prev = self.cfg.imitation_mix_for_version(0)
        for v in range(1, self.cfg.imitation_mix_decay_end_version + 10):
            curr = self.cfg.imitation_mix_for_version(v)
            assert curr <= prev + 1e-9, (
                f"non-monotone at v={v}: {curr} > {prev}"
            )
            prev = curr

    def test_midpoint_is_interpolated(self):
        """At the midpoint of the decay range, the value should be ~halfway."""
        v_end = self.cfg.imitation_mix_decay_end_version
        if v_end <= 2:
            pytest.skip("decay range too short for midpoint test")
        v_mid = (1 + v_end) // 2
        val = self.cfg.imitation_mix_for_version(v_mid)
        start = self.cfg.imitation_mix_start
        end = self.cfg.imitation_mix_end
        expected_mid = (start + end) / 2
        # Allow some slack since v_mid may not be exactly halfway
        assert abs(val - expected_mid) < 0.05, (
            f"v={v_mid}: got {val}, expected ~{expected_mid}"
        )

    def test_all_values_non_negative(self):
        """The mix fraction should never go negative."""
        for v in range(0, self.cfg.imitation_mix_decay_end_version + 20):
            val = self.cfg.imitation_mix_for_version(v)
            assert val >= 0.0, f"negative imitation_mix at v={v}: {val}"

    def test_pinned_values(self):
        """Pin exact expected values at key versions for the default config.
        Default: start=0.3, end=0.0, decay_end_version=5."""
        assert self.cfg.imitation_mix_start == 0.3
        assert self.cfg.imitation_mix_end == 0.0
        assert self.cfg.imitation_mix_decay_end_version == 5

        expected = {
            0: 0.3,
            1: 0.3,
            2: 0.225,  # 0.3 + (0.0 - 0.3) * (2-1)/(5-1) = 0.3 - 0.075 = 0.225
            3: 0.15,   # 0.3 + (0.0 - 0.3) * (3-1)/(5-1) = 0.3 - 0.15 = 0.15
            4: 0.075,  # 0.3 + (0.0 - 0.3) * (4-1)/(5-1) = 0.3 - 0.225 = 0.075
            5: 0.0,
            6: 0.0,
            10: 0.0,
        }
        for v, exp in expected.items():
            got = self.cfg.imitation_mix_for_version(v)
            assert abs(got - exp) < 1e-9, (
                f"v={v}: expected {exp}, got {got}"
            )
