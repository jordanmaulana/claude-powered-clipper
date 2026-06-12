"""ffmpeg/ffprobe subprocess helpers and filtergraph builders."""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def require_binaries() -> None:
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            sys.exit(f"error: {name} not found on PATH (brew install ffmpeg)")


def run(cmd: list[str], desc: str = "") -> None:
    """Run a command, exit with stderr tail on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-15:])
        sys.exit(f"error: {desc or cmd[0]} failed:\n{tail}")


def probe(path: Path) -> dict:
    """Return {width, height, fps, duration, has_audio} for a media file."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"error: ffprobe failed on {path}:\n{proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        sys.exit(f"error: no video stream in {path}")
    num, _, den = video["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration": float(data["format"]["duration"]),
        "has_audio": any(s["codec_type"] == "audio" for s in data["streams"]),
    }


def escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument (ass=, sendcmd=f=)."""
    # Filter args treat \ : ' specially; escape for the filtergraph parser.
    s = str(path)
    s = s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return s


def cut_filtergraph(intervals: list[tuple[float, float]], fps: float) -> str:
    """trim/atrim + concat graph removing everything outside the keep-intervals."""
    parts = []
    for i, (s, e) in enumerate(intervals):
        parts.append(f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}];")
        parts.append(f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}];")
    pads = "".join(f"[v{i}][a{i}]" for i in range(len(intervals)))
    # fps filter normalizes VFR sources to CFR so Pass B frame timestamps are exact
    parts.append(f"{pads}concat=n={len(intervals)}:v=1:a=1[vc][a];")
    parts.append(f"[vc]fps={fps:.6f}[v]")
    return "\n".join(parts)
