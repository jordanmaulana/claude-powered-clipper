import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from lib import facetrack, ffmpeg
import render


# --- 1b: facetrack proxy -> source coordinate rescale (pure math, no video) ---

def _fake_detect(width, height, fps, n_frames, centers):
    """Return a _detect_centers stand-in producing controlled samples."""
    times = [i / fps for i in range(len(centers))]
    info = {"fps": fps, "width": width, "height": height, "n_frames": n_frames,
            "switch_flags": [False] * len(centers), "n_speakers": 1}
    return lambda *a, **k: (times, list(centers), info)


class TestProxyRescale:
    OUT = (1920, 1080)  # source dims; proxy is 960x540

    def test_tracked_geometry_in_source_pixels(self, monkeypatch):
        monkeypatch.setattr(facetrack, "_detect_centers",
                            _fake_detect(960, 540, 30.0, 30, [480.0] * 12))
        r = facetrack.track(Path("proxy.mp4"), Path("model"), out_dims=self.OUT)

        assert r["mode"] == "tracked"
        assert r["src_width"] == 1920
        assert r["crop_h"] == 1080
        # crop_w = min(even(1080*9/16), even(1920)) = 608, full source height crop
        assert r["crop_w"] == 608
        max_x = 1920 - 608
        assert len(r["x"]) == 30
        for x in r["x"]:
            assert x % 2 == 0                # ffmpeg crop needs even offsets
            assert 0 <= x <= max_x           # clamped to source frame
        # proxy center 480 -> source 960; x ~= 960 - 608/2 = 656
        assert all(abs(x - 656) <= 2 for x in r["x"])

    def test_low_detection_falls_back_to_center(self, monkeypatch):
        centers = [None] * 11 + [480.0]      # detection_rate < MIN_DETECTION_RATE
        monkeypatch.setattr(facetrack, "_detect_centers",
                            _fake_detect(960, 540, 30.0, 30, centers))
        r = facetrack.track(Path("proxy.mp4"), Path("model"), out_dims=self.OUT)

        assert r["mode"] == "center"
        assert r["crop_w"] == 608 and r["src_width"] == 1920
        assert len(set(r["x"])) == 1         # static center crop
        assert r["x"][0] % 2 == 0 and 0 <= r["x"][0] <= 1920 - 608


# --- single-encode end-to-end through render_clip (pad path, no face model) ---

def _args():
    return argparse.Namespace(gap=0.5, pad=0.15, no_captions=True,
                              no_active_speaker=False, debug=False, snap_end=False)


@pytest.mark.skipif(not ffmpeg.FFMPEG or not ffmpeg.FFPROBE, reason="ffmpeg not available")
def test_render_clip_single_encode(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    source = workdir / "source.mp4"
    # Tall 1080x1920 source -> render takes the pad path (no YuNet model needed).
    subprocess.run(
        [ffmpeg.FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=size=1080x1920:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)],
        check=True, capture_output=True,
    )
    meta = {"video_id": "test", "width": 1080, "height": 1920, "fps": 30.0, "duration": 6.0}
    (workdir / "meta.json").write_text(json.dumps(meta))
    words = [{"w": f"w{i}", "s": i * 0.4, "e": i * 0.4 + 0.3} for i in range(13)]  # ~0..5.2s
    outdir = tmp_path / "out"
    outdir.mkdir()
    clip = {"id": 1, "slug": "synthetic", "title": "T", "summary": "S",
            "start": 0.0, "end": 5.0}

    r = render.render_clip(clip, words, meta, workdir, outdir, _args())

    out = r["path"]
    assert out.exists() and out.stat().st_size > 0
    # Single encode reads source directly — no clip_flat intermediate is produced.
    assert not (workdir / "clips" / "01" / "clip_flat.mp4").exists()
    probe = ffmpeg.probe(out)
    assert (probe["width"], probe["height"]) == (1080, 1920)
    assert probe["has_audio"]
