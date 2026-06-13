import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lib.correct import apply_corrections, is_single_token, merge_split_numbers


def w(text, s, e):
    return {"w": text, "s": s, "e": e}


def transcript(words, seg_text):
    return {
        "language": "id",
        "words": [w(t, i, i + 0.4) for i, t in enumerate(words)],
        "segments": [{"s": 0.0, "e": 9.0, "text": seg_text}],
    }


class TestIsSingleToken:
    def test_single(self):
        assert is_single_token("naval")

    def test_multi_word_rejected(self):
        assert not is_single_token("naval ravikant")

    def test_empty_rejected(self):
        assert not is_single_token("")


class TestApplyCorrections:
    def test_case_insensitive_match_forces_value_casing(self):
        t = transcript(["naval", "NAVAL", "Naval"], "naval NAVAL Naval")
        out, hits = apply_corrections(t, {"naval": "Naval"})
        assert [x["w"] for x in out["words"]] == ["Naval", "Naval", "Naval"]
        assert out["segments"][0]["text"] == "Naval Naval Naval"
        assert hits["naval"] == 6  # 3 words + 3 segment occurrences

    def test_punctuation_preserved(self):
        t = transcript(["naval,", "(naval)"], "naval, and (naval).")
        out, _ = apply_corrections(t, {"naval": "Naval"})
        assert [x["w"] for x in out["words"]] == ["Naval,", "(Naval)"]
        assert out["segments"][0]["text"] == "Naval, and (Naval)."

    def test_no_partial_word_hits(self):
        t = transcript(["navalny", "naval"], "navalny naval")
        out, hits = apply_corrections(t, {"naval": "Naval"})
        assert [x["w"] for x in out["words"]] == ["navalny", "Naval"]
        assert out["segments"][0]["text"] == "navalny Naval"
        assert hits["naval"] == 2

    def test_timestamps_untouched(self):
        t = transcript(["naval"], "naval")
        out, _ = apply_corrections(t, {"naval": "Naval"})
        assert out["words"][0]["s"] == 0.0 and out["words"][0]["e"] == 0.4

    def test_multi_word_key_ignored(self):
        t = transcript(["naval"], "naval ravikant")
        out, hits = apply_corrections(t, {"naval ravikant": "Naval Ravikant"})
        assert hits == {}
        assert out["segments"][0]["text"] == "naval ravikant"

    def test_zero_hit_key_reported(self):
        t = transcript(["hello"], "hello")
        _, hits = apply_corrections(t, {"naval": "Naval"})
        assert hits["naval"] == 0

    def test_input_not_mutated(self):
        t = transcript(["naval"], "naval")
        apply_corrections(t, {"naval": "Naval"})
        assert t["words"][0]["w"] == "naval"
        assert t["segments"][0]["text"] == "naval"

    def test_idempotent(self):
        t = transcript(["naval"], "naval")
        once, _ = apply_corrections(t, {"naval": "Naval"})
        twice, _ = apply_corrections(once, {"naval": "Naval"})
        assert once == twice


class TestMergeSplitNumbers:
    def test_thousands_chain(self):
        t = transcript(["harga", "1", ".500", ".000", "rupiah"], "harga 1.500.000 rupiah")
        out, merges = merge_split_numbers(t)
        assert [x["w"] for x in out["words"]] == ["harga", "1.500.000", "rupiah"]
        assert merges == 2

    def test_decimal_comma(self):
        t = transcript(["sekitar", "9", ",5", "persen"], "sekitar 9,5 persen")
        out, merges = merge_split_numbers(t)
        assert [x["w"] for x in out["words"]] == ["sekitar", "9,5", "persen"]
        assert merges == 1

    def test_merged_token_spans_timestamps(self):
        # words built at s=i, e=i+0.4 -> "1"@1, ".500"@2, ".000"@3
        t = transcript(["x", "1", ".500", ".000"], "x 1.500.000")
        out, _ = merge_split_numbers(t)
        num = out["words"][1]
        assert num["w"] == "1.500.000"
        assert num["s"] == 1.0 and num["e"] == 3.4

    def test_no_merge_when_prev_not_digit(self):
        # sentence period then a leading-dot number-ish token must NOT merge
        t = transcript(["end.", ".5"], "end. .5")
        out, merges = merge_split_numbers(t)
        assert [x["w"] for x in out["words"]] == ["end.", ".5"]
        assert merges == 0

    def test_segment_defensive_join(self):
        t = transcript([], "harga 1 .500 .000 dan 9 ,5")
        out, _ = merge_split_numbers(t)
        assert out["segments"][0]["text"] == "harga 1.500.000 dan 9,5"

    def test_segment_period_not_merged(self):
        t = transcript([], "ada 5. 000 orang")
        out, _ = merge_split_numbers(t)
        assert out["segments"][0]["text"] == "ada 5. 000 orang"

    def test_idempotent(self):
        t = transcript(["1", ".500", ".000"], "1.500.000")
        once, _ = merge_split_numbers(t)
        twice, n = merge_split_numbers(once)
        assert once == twice and n == 0

    def test_input_not_mutated(self):
        t = transcript(["1", ".500"], "1.500")
        merge_split_numbers(t)
        assert [x["w"] for x in t["words"]] == ["1", ".500"]
