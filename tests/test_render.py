import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from render import validate


def clip(summary="A clean insight stated plainly.", start=10.0, end=40.0, id=1):
    return {"id": id, "summary": summary, "start": start, "end": end}


class TestValidate:
    def test_clean_summary_passes(self):
        validate(clip(), duration=100.0)

    def test_missing_summary_raises(self):
        with pytest.raises(ValueError, match="no 'summary'"):
            validate({"id": 1, "start": 10.0, "end": 40.0}, duration=100.0)

    def test_blank_summary_raises(self):
        with pytest.raises(ValueError, match="no 'summary'"):
            validate(clip(summary="   "), duration=100.0)

    @pytest.mark.parametrize("mark", [";", "—", "–", "--"])
    def test_banned_punctuation_raises(self, mark):
        with pytest.raises(ValueError, match="banned"):
            validate(clip(summary=f"first part{mark} second part"), duration=100.0)

    def test_range_outside_video_raises(self):
        with pytest.raises(ValueError, match="outside video"):
            validate(clip(start=10.0, end=200.0), duration=100.0)
