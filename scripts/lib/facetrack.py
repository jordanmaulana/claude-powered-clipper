"""YuNet face tracking -> smoothed 9:16 crop path -> ffmpeg sendcmd file.

Operates on the already-cut clip (edited timeline), so no timestamp remapping here.
"""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

SAMPLE_HZ = 5.0
DETECT_WIDTH = 640
SCORE_THRESHOLD = 0.6
STICKINESS = 2.0          # penalty weight: distance from previous center (frame-widths)
CARRY_SECONDS = 2.0       # hold last center this long through misses
RECENTER_SECONDS = 1.0    # then lerp toward frame center over this long
SNAP_JUMP = 0.25          # center jump > this fraction of width = shot cut, snap
DEAD_ZONE = 0.04          # ignore face drift inside this fraction of width
EMA_ALPHA = 0.12          # at SAMPLE_HZ
MAX_PAN_SPEED = 0.35      # fraction of width per second
MIN_DETECTION_RATE = 0.3  # below this -> static center crop


def even(x: float) -> int:
    return int(round(x / 2)) * 2


def _detect_centers(video: Path, model: Path,
                    on_progress: Callable[[float], None] | None = None,
                    ) -> tuple[list[float], list[float | None], dict]:
    """Sample frames at ~SAMPLE_HZ, return (times, center_x or None per sample, video info)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / SAMPLE_HZ))

    scale = DETECT_WIDTH / width
    det_size = (DETECT_WIDTH, even(height * scale))
    detector = cv2.FaceDetectorYN.create(str(model), "", det_size, SCORE_THRESHOLD)

    times: list[float] = []
    centers: list[float | None] = []
    prev_center: float | None = None
    frame_idx = 0
    while True:
        if frame_idx % step:
            if not cap.grab():
                break
            frame_idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, det_size)
        _, faces = detector.detect(small)
        center = None
        if faces is not None and len(faces):
            best, best_score = None, -np.inf
            for f in faces:
                x, y, w, h, conf = f[0], f[1], f[2], f[3], f[-1]
                cx = (x + w / 2) / scale
                area = (w * h) / (det_size[0] * det_size[1])
                dist = abs(cx - prev_center) / width if prev_center is not None else 0.0
                score = area + 0.3 * conf - STICKINESS * dist
                if score > best_score:
                    best, best_score = cx, score
            center = float(best)
            prev_center = center
        times.append(frame_idx / fps)
        centers.append(center)
        frame_idx += 1
        if on_progress and len(times) % 25 == 0:
            on_progress(frame_idx / max(1, n_frames))
    cap.release()
    info = {"fps": fps, "width": width, "height": height, "n_frames": n_frames}
    return times, centers, info


def _fill_misses(centers: list[float | None], width: float) -> list[float]:
    """Carry last center through short miss runs, then ease back to frame center."""
    carry_n = int(CARRY_SECONDS * SAMPLE_HZ)
    recenter_n = max(1, int(RECENTER_SECONDS * SAMPLE_HZ))
    mid = width / 2
    out: list[float] = []
    last, miss_run = mid, 0
    for c in centers:
        if c is not None:
            last, miss_run = c, 0
            out.append(c)
        else:
            miss_run += 1
            if miss_run <= carry_n:
                out.append(last)
            else:
                t = min(1.0, (miss_run - carry_n) / recenter_n)
                out.append(last + (mid - last) * t)
    return out


def _smooth(times: list[float], centers: list[float], width: float) -> list[float]:
    """Shot-cut snap + dead-zone + EMA + pan-speed clamp."""
    target = smooth = centers[0]
    out = [smooth]
    for i in range(1, len(centers)):
        c = centers[i]
        dt = max(1e-3, times[i] - times[i - 1])
        if abs(c - smooth) > SNAP_JUMP * width:
            target = smooth = c  # shot cut: snap, don't pan
        else:
            if abs(c - target) > DEAD_ZONE * width:
                target = c
            step = EMA_ALPHA * (target - smooth)
            max_step = MAX_PAN_SPEED * width * dt
            smooth += float(np.clip(step, -max_step, max_step))
        out.append(smooth)
    return out


def track(video: Path, model: Path,
          on_progress: Callable[[float], None] | None = None) -> dict:
    """Compute per-frame crop x positions for a 9:16 crop of `video`.

    Returns {"mode": "tracked"|"center", "crop_w", "crop_h", "fps", "n_frames",
             "x": [per-frame int], "detection_rate": float}
    """
    times, raw, info = _detect_centers(video, model, on_progress)
    width, height = info["width"], info["height"]
    crop_w = min(even(height * 9 / 16), even(width))
    n_frames = info["n_frames"]
    max_x = width - crop_w

    detection_rate = sum(c is not None for c in raw) / max(1, len(raw))
    base = {
        "crop_w": crop_w, "crop_h": height, "fps": info["fps"],
        "n_frames": n_frames, "detection_rate": round(detection_rate, 3),
    }
    if detection_rate < MIN_DETECTION_RATE or not raw:
        x = even(max_x / 2)
        return {**base, "mode": "center", "x": [x] * n_frames}

    centers = _smooth(times, _fill_misses(raw, width), width)
    frame_times = np.arange(n_frames) / info["fps"]
    interp = np.interp(frame_times, times, centers)
    xs = [min(max(even(c - crop_w / 2), 0), max_x) for c in interp]
    return {**base, "mode": "tracked", "x": xs}


def write_sendcmd(result: dict, path: Path) -> None:
    fps = result["fps"]
    lines = [f"{i / fps:.4f} crop x {x};" for i, x in enumerate(result["x"])]
    path.write_text("\n".join(lines) + "\n")


def write_debug_preview(video: Path, model: Path, result: dict, out: Path) -> None:
    """Render a preview with the smoothed crop window drawn, for tuning."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    crop_w = result["crop_w"]
    for i in range(result["n_frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        x = result["x"][min(i, len(result["x"]) - 1)]
        cv2.rectangle(frame, (x, 0), (x + crop_w, h), (0, 255, 0), 4)
        writer.write(frame)
    cap.release()
    writer.release()
