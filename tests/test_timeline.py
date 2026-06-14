import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from lib.timeline import (
    keep_intervals,
    midstatement_end,
    remap,
    snap_end,
    total_duration,
    words_in_range,
)


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


# Real Za9UseDTu3E tail: clip ended at 797.0 mid-statement, runs to a pause at 801.82.
ZA9_TAIL = [
    w("kritis.", 796.74, 797.22),
    w("Itu", 797.48, 797.76),
    w("penting", 797.76, 798.06),
    w("bagi.", 798.06, 798.26),
    w("Itu", 798.26, 798.44),
    w("adalah", 798.44, 798.74),
    w("investasi.", 798.74, 799.34),
    w("Penting", 799.72, 800.26),
    w("bagi", 800.26, 800.62),
    w("penguasa", 800.62, 801.10),
    w("itu.", 801.10, 801.30),
    w("Oh", 801.30, 801.40),
    w("gitu", 801.40, 801.62),
    w("ya.", 801.62, 801.82),
    w("Karena", 802.56, 802.78),  # after 0.74s pause -> new run
]


class TestMidstatementEnd:
    def test_none_when_at_pause(self):
        # end after "world" (1.9) -> next word "next" at 3.0 is 1.1s away (> max_gap)
        assert midstatement_end(WORDS, 1.95, max_gap=0.5) is None

    def test_returns_run_end_when_midrun(self):
        # end inside the [next, sentence] run -> pauses at sentence.e = 4.0
        assert midstatement_end(WORDS, 3.2, max_gap=0.5) == pytest.approx(4.0)

    def test_none_after_last_word(self):
        assert midstatement_end(WORDS, 99.0, max_gap=0.5) is None

    def test_respects_max_gap(self):
        # gap next<-world is 1.1s; with max_gap=1.5 the two runs merge
        assert midstatement_end(WORDS, 1.95, max_gap=1.5) == pytest.approx(4.0)

    def test_za9_regression(self):
        # the actual bug: end=797.0 should report the real pause at 801.82
        assert midstatement_end(ZA9_TAIL, 797.0, max_gap=0.5) == pytest.approx(801.82)


class TestSnapEnd:
    def test_unchanged_at_pause(self):
        assert snap_end(WORDS, 1.95, max_gap=0.5) == pytest.approx(1.95)

    def test_extends_to_pause_within_cap(self):
        assert snap_end(WORDS, 3.2, max_gap=0.5, max_extend=6.0) == pytest.approx(4.0)

    def test_za9_regression(self):
        assert snap_end(ZA9_TAIL, 797.0, max_gap=0.5, max_extend=6.0) == pytest.approx(801.82)

    def test_cap_backs_off_to_last_sentence_word(self):
        # cap at 797.0+2.0=799.0; last .?! word within window is "investasi." (798.74-799.34)
        # whose e (799.34) exceeds 799.0 -> prior sentence word "bagi." ends 798.26
        assert snap_end(ZA9_TAIL, 797.0, max_gap=0.5, max_extend=2.0) == pytest.approx(798.26)

    def test_cap_no_sentence_word_returns_cap(self):
        # run of bare tokens (no .?!), cap binds -> return end+max_extend
        run = [w("a", 0.0, 0.2), w("b", 0.4, 0.6), w("c", 0.8, 1.0), w("d", 1.2, 1.4)]
        assert snap_end(run, 0.1, max_gap=0.5, max_extend=0.5) == pytest.approx(0.6)


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
