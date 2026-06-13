import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from lib.captions import build_ass, clean_caption


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("leverage.", "leverage"),
        ("word,", "word"),
        ("really?", "really"),
        ("stop!", "stop"),
        ("wait;", "wait"),
        ("note:", "note"),
        ("...", ""),
        ("1.500.000", "1.500.000"),
        ("9,5", "9,5"),
        ("He earned 1.500.000, right?", "He earned 1.500.000 right"),
        ("naik 9,5% lho.", "naik 9,5% lho"),
        ("a, b. c?", "a b c"),
    ],
)
def test_clean_caption(raw, expected):
    assert clean_caption(raw) == expected


def w(text, s, e):
    return {"w": text, "s": s, "e": e}


def _dialogue_text(ass: str) -> list[str]:
    return [
        line.split(",", 9)[9]
        for line in ass.splitlines()
        if line.startswith("Dialogue:")
    ]


def test_build_ass_strips_punctuation_keeps_number_dots():
    words = [
        w("Dia", 0.0, 0.3),
        w("dapat", 0.3, 0.6),
        w("1.500.000,", 0.6, 1.0),
        w("kan?", 1.0, 1.3),
    ]
    texts = _dialogue_text(build_ass(words))
    joined = " ".join(texts)
    # number separators survive
    assert "1.500.000" in joined
    # sentence punctuation gone
    assert "?" not in joined
    assert "," not in joined.replace("1.500.000", "")
    # uppercase applied (UPPERCASE default)
    assert joined == joined.upper()


def test_build_ass_chunking_unaffected():
    # period forces a phrase flush; stripping happens after chunking
    words = [
        w("one", 0.0, 0.3),
        w("two.", 0.3, 0.6),
        w("three", 0.6, 0.9),
        w("four", 0.9, 1.2),
    ]
    texts = _dialogue_text(build_ass(words))
    assert texts[0] == "ONE TWO"
    assert "FOUR" in texts[-1]
    assert len(texts) == 2
