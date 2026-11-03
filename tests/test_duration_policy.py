import unittest

from duration_policy import choose_duration


class TestChooseDuration(unittest.TestCase):
    def test_natural_pace_when_the_line_fits(self):
        self.assertIsNone(choose_duration(natural=4.7, target=5.0, limit=6.0))

    def test_running_into_the_pause_after_is_fine(self):
        # Longer than the original's speech, but the silence after the line absorbs it
        self.assertIsNone(choose_duration(natural=5.8, target=5.0, limit=6.0))

    def test_short_line_is_slowed_down_within_a_natural_range(self):
        # 3 s of speech for a 5 s slot: slow down, but by at most 12%
        self.assertAlmostEqual(choose_duration(natural=3.0, target=5.0, limit=6.0), 3.0 / 0.88)

    def test_slow_down_stops_at_the_target(self):
        self.assertAlmostEqual(choose_duration(natural=3.95, target=4.4, limit=6.0), 4.4)

    def test_long_line_is_sped_up_at_most_by_the_cap(self):
        self.assertAlmostEqual(choose_duration(natural=6.6, target=5.0, limit=6.0), 6.0)
        self.assertAlmostEqual(choose_duration(natural=9.0, target=5.0, limit=6.0), 9.0 / 1.2)

    def test_padding_is_taken_from_the_budget(self):
        self.assertAlmostEqual(choose_duration(natural=6.0, target=5.0, limit=6.0, pad=0.1), 5.8)

    def test_no_slot_means_natural_pace(self):
        self.assertIsNone(choose_duration(natural=3.0, target=None, limit=None))
