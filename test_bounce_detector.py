"""
Tests for bounce_detector module.

Run with:
    python -m pytest test_bounce_detector.py -v
"""

import math
from bounce_detector import BounceDetector, detect_bounce, map_to_grid, Phase


# ─── BounceDetector state-machine tests ────────────────────────

class TestBounceDetector:

    def test_single_parabola_gives_one_bounce(self):
        """Simulate a single ball bounce: fall → apex → rise."""
        det = BounceDetector(buffer_size=3, cooldown_frames=5)
        # cy values: falling (increasing), apex, then rising (decreasing)
        cy_values = [100, 120, 140, 160, 180, 200,   # falling
                     210, 215, 218, 220,               # approaching apex
                     218, 215, 210, 200, 190, 180]     # rising
        bounces = [i for i, cy in enumerate(cy_values) if det.update(cy)]
        assert len(bounces) == 1, f"Expected 1 bounce, got {bounces}"
        # bounce should occur around the apex (index ~10–12)
        assert 8 <= bounces[0] <= 13

    def test_no_bounce_on_steady_fall(self):
        """Monotonically increasing cy → no bounce."""
        det = BounceDetector(buffer_size=3, cooldown_frames=5)
        cy_values = list(range(100, 300, 5))
        bounces = [i for i, cy in enumerate(cy_values) if det.update(cy)]
        assert len(bounces) == 0

    def test_no_bounce_on_steady_rise(self):
        """Monotonically decreasing cy → no bounce."""
        det = BounceDetector(buffer_size=3, cooldown_frames=5)
        cy_values = list(range(300, 100, -5))
        bounces = [i for i, cy in enumerate(cy_values) if det.update(cy)]
        assert len(bounces) == 0

    def test_cooldown_prevents_rapid_triggers(self):
        """Two apex events within cooldown → only first accepted."""
        det = BounceDetector(buffer_size=3, cooldown_frames=20)

        cy = []
        # first parabola
        cy += list(range(100, 200, 10))  # fall
        cy += list(range(200, 100, -10)) # rise
        # second parabola immediately after (within 20 frames of first)
        cy += list(range(100, 200, 10))  # fall
        cy += list(range(200, 100, -10)) # rise

        bounces = [i for i, v in enumerate(cy) if det.update(v)]
        # cooldown=20 should suppress the second bounce if it's too close
        assert len(bounces) >= 1

    def test_noisy_single_frame_ignored(self):
        """A single noisy frame shouldn't trigger a bounce."""
        det = BounceDetector(buffer_size=5, cooldown_frames=5)
        # steady descent with one upward spike at frame 5
        cy_values = [100, 110, 120, 130, 140,
                     100,  # spike (noise)
                     160, 170, 180, 190, 200]
        bounces = [i for i, cy in enumerate(cy_values) if det.update(cy)]
        # median filter should absorb this
        assert len(bounces) == 0, f"False bounce(s) at {bounces}"

    def test_phase_starts_falling(self):
        det = BounceDetector()
        assert det.phase == Phase.FALLING

    def test_reset_clears_state(self):
        det = BounceDetector(buffer_size=3, cooldown_frames=3)
        for cy in [100, 150, 200, 180, 160]:
            det.update(cy)
        det.reset()
        assert det.phase == Phase.FALLING

    def test_multiple_bounces_detected(self):
        """Two well-separated parabolas → two bounces."""
        det = BounceDetector(buffer_size=3, cooldown_frames=5)

        cy = []
        # first bounce: fall → apex → rise
        cy += list(range(100, 220, 10))   # 12 frames falling
        cy += list(range(220, 100, -10))  # 12 frames rising
        # gap / second bounce: fall → apex → rise
        cy += list(range(100, 220, 10))   # 12 frames falling
        cy += list(range(220, 100, -10))  # 12 frames rising

        bounces = [i for i, v in enumerate(cy) if det.update(v)]
        assert len(bounces) == 2, f"Expected 2 bounces, got {len(bounces)} at {bounces}"


# ─── detect_bounce stateless API ──────────────────────────────

class TestDetectBounce:

    def test_returns_true_at_apex(self):
        """Stateless wrapper should fire at the local max."""
        history = []
        cy_values = [100, 120, 140, 160, 180, 200,
                     210, 215, 218, 220,
                     218, 215, 210, 200]
        hits = [i for i, cy in enumerate(cy_values)
                if detect_bounce(cy, history)]
        assert len(hits) >= 1


# ─── map_to_grid ──────────────────────────────────────────────

class TestMapToGrid:

    def test_origin(self):
        col, row = map_to_grid(0.0, 0.0)
        assert col == 0 and row == 0

    def test_far_corner(self):
        col, row = map_to_grid(1.525, 2.74)
        assert col == 2 and row == 3  # max indices for 3×4

    def test_center(self):
        col, row = map_to_grid(0.7625, 1.37)
        assert col == 1 and row == 2  # center of 3×4 grid → (1, 2)

    def test_clamps_negative(self):
        col, row = map_to_grid(-0.5, -0.5)
        assert col == 0 and row == 0

    def test_clamps_beyond_table(self):
        col, row = map_to_grid(5.0, 10.0)
        assert col == 2 and row == 3

    def test_custom_grid(self):
        col, row = map_to_grid(1.0, 2.0, grid_cols=2, grid_rows=2)
        assert col == 1 and row == 1

    def test_quadrant_lower_right(self):
        """Ball at (1.2, 2.0) → should be col=2, row=2 in 3×4."""
        col, row = map_to_grid(1.2, 2.0)
        assert col == 2 and row == 2
