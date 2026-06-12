import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from lib.timeline import keep_intervals, remap, total_duration, words_in_range


def w(text, s, e):
    return {"w": text, "s": s, "e": e}


WORDS = [
    w("hello", 1.0, 1.4),
    w("world", 1.5, 1.9),      # gap 0.1 -> same run
    w("next", 3.0, 3.4),       # gap 1.1 -> new run
    w("sentence", 3.5, 4.0),
]


class TestKeepIntervals:
    def test_empty(self):
        assert keep_intervals([]) == []

    def test_single_run(self):
        out = keep_intervals(WORDS[:2], max_gap=0.5, pad=0.15)
        assert out == [(0.85, 2.05)]

    def test_gap_splits_runs(self):
        out = keep_intervals(WORDS, max_gap=0.5, pad=0.15)
        assert out == [(0.85, 2.05), (2.85, 4.15)]

    def test_pad_clamps_at_zero(self):
        out = keep_intervals([w("hi", 0.05, 0.4)], pad=0.15)
        assert out[0][0] == 0.0

    def test_touching_padded_intervals_merge(self):
        # gap 0.6 > max_gap 0.5 splits runs, but pads (2*0.4) overlap -> merge back
        words = [w("a", 1.0, 2.0), w("b", 2.6, 3.0)]
        out = keep_intervals(words, max_gap=0.5, pad=0.4)
        assert out == [(0.6, 3.4)]


class TestRemap:
    INTERVALS = [(1.0, 2.0), (3.0, 5.0)]

    def test_before_first_interval(self):
        assert remap(0.5, self.INTERVALS) == 0.0

    def test_inside_first(self):
        assert remap(1.5, self.INTERVALS) == 0.5

    def test_in_gap_clamps_to_cut(self):
        assert remap(2.5, self.INTERVALS) == 1.0

    def test_inside_second(self):
        assert remap(4.0, self.INTERVALS) == 2.0

    def test_after_last(self):
        assert remap(99.0, self.INTERVALS) == 3.0

    def test_monotonic(self):
        ts = [i * 0.1 for i in range(60)]
        mapped = [remap(t, self.INTERVALS) for t in ts]
        assert mapped == sorted(mapped)


class TestHelpers:
    def test_words_in_range_snap(self):
        assert [x["w"] for x in words_in_range(WORDS, 1.2, 3.7, snap=0.5)] == [
            "hello", "world", "next", "sentence",
        ]
        assert [x["w"] for x in words_in_range(WORDS, 1.2, 3.7, snap=0.0)] == [
            "world", "next",
        ]

    def test_total_duration(self):
        assert total_duration([(1.0, 2.0), (3.0, 5.0)]) == pytest.approx(3.0)
