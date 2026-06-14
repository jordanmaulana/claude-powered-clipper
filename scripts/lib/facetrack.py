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

# --- active-speaker (lip-motion) tracking ---
MAX_TRACKLETS  = 4       # cap faces tracked per sample
LINK_MAX_JUMP  = 0.18    # max center move (frac width) to link det->tracklet (< SNAP_JUMP on purpose)
TRACKLET_TTL   = 3       # samples a tracklet survives unmatched (~0.6s @5Hz)
MOUTH_W_FRAC   = 1.6     # mouth ROI width  = this * mouth-corner distance
MOUTH_H_FRAC   = 0.9     # mouth ROI height = this * mouth-corner distance
MOUTH_PATCH    = 24      # mouth patch resized to PATCH^2 gray for diffing
MOTION_WIN     = 3       # samples to smooth mouth motion (~0.6s)
SPEAKER_MARGIN = 1.30    # challenger must beat active speaker's motion by this ratio
SPEAKER_DWELL  = 5       # samples challenger must lead before switch (~1.0s)
MOTION_FLOOR   = 0.004   # below this normalized motion = ambiguous -> v1 fallback


def even(x: float) -> int:
    return int(round(x / 2)) * 2


def _extract_mouth_patch(small: np.ndarray, f: np.ndarray) -> np.ndarray | None:
    """Crop the mouth ROI from the detection frame, gray + resize to a fixed patch.

    Mouth corners (f[10:12] right, f[12:14] left) are in `small` coords. ROI is
    sized from the corner distance so near/far faces yield comparable patches;
    the fixed MOUTH_PATCH^2 resize makes the later diff scale-invariant. Returns
    None for profile/degenerate faces (corners collapsed) or out-of-frame ROIs.
    """
    rmx, rmy, lmx, lmy = f[10], f[11], f[12], f[13]
    mcx, mcy = (rmx + lmx) / 2, (rmy + lmy) / 2
    mouth_dist = float(np.hypot(lmx - rmx, lmy - rmy))
    if mouth_dist < 2.0:
        return None
    half_w = MOUTH_W_FRAC * mouth_dist / 2
    half_h = MOUTH_H_FRAC * mouth_dist / 2
    H, W = small.shape[:2]
    x0, x1 = int(max(0, mcx - half_w)), int(min(W, mcx + half_w))
    y0, y1 = int(max(0, mcy - half_h)), int(min(H, mcy + half_h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = small[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (MOUTH_PATCH, MOUTH_PATCH))


def _mouth_motion(prev_patch: np.ndarray | None, patch: np.ndarray | None) -> float:
    """Normalized mean abs-diff (0..1) of two fixed-size gray mouth patches."""
    if prev_patch is None or patch is None:
        return 0.0
    return float(cv2.absdiff(prev_patch, patch).mean()) / 255.0


def _best_by_area(faces: list[dict], prev_cx: float | None, width: float) -> dict | None:
    """v1 single-subject scorer: pick the face maximizing area + 0.3*conf,
    penalized by distance from prev_cx (the stickiness gravity well).

    `faces` are dicts with cx (full-res), area (normalized), conf. Returns the
    chosen face dict, or None if `faces` is empty.
    """
    best, best_score = None, -np.inf
    for fc in faces:
        dist = abs(fc["cx"] - prev_cx) / width if prev_cx is not None else 0.0
        score = fc["area"] + 0.3 * fc["conf"] - STICKINESS * dist
        if score > best_score:
            best, best_score = fc, score
    return best


def _link_detections(tracklets: list[dict], dets: list[dict], width: float) -> dict[int, int]:
    """Greedy nearest-center matching of detections to existing tracklets.

    Returns {det_index: tracklet_index}. A pair links only if their center gap
    is within LINK_MAX_JUMP*width; each det and tracklet is used at most once
    (closest pairs first). Unmatched dets/tracklets are the caller's job to
    spawn/age.
    """
    gate = LINK_MAX_JUMP * width
    pairs = sorted(
        (abs(t["cx"] - d["cx"]), di, ti)
        for ti, t in enumerate(tracklets)
        for di, d in enumerate(dets)
    )
    used_t: set[int] = set()
    used_d: set[int] = set()
    matched: dict[int, int] = {}
    for dist, di, ti in pairs:
        if di in used_d or ti in used_t or dist > gate:
            continue
        used_d.add(di)
        used_t.add(ti)
        matched[di] = ti
    return matched


def _update_tracklet(trk: dict, det: dict) -> None:
    """Fold a matched detection into a tracklet: motion vs its previous mouth
    patch, rolled MOTION_WIN window -> smooth_motion, position/area refresh."""
    trk["motion"] = _mouth_motion(trk["prev_patch"], det["patch"])
    trk["prev_patch"] = det["patch"]
    trk["cx"], trk["area"], trk["conf"] = det["cx"], det["area"], det["conf"]
    trk["motion_hist"].append(trk["motion"])
    if len(trk["motion_hist"]) > MOTION_WIN:
        trk["motion_hist"].pop(0)
    trk["smooth_motion"] = float(np.mean(trk["motion_hist"]))
    trk["miss"] = 0


def _select_speaker(tracklets: list[dict], state: dict, width: float,
                    active_speaker: bool = True) -> float | None:
    """Choose the active speaker's center for this sample, with hysteresis.

    Ranks live tracklets by smoothed mouth motion. A challenger must beat the
    current speaker's motion by SPEAKER_MARGIN for SPEAKER_DWELL consecutive
    samples before the crop switches (kills ping-pong). Falls back to v1
    area+stickiness scoring when there is one face or the lip signal is flat
    (< MOTION_FLOOR). Returns None when no face is live (caller treats as a miss).
    """
    live = [t for t in tracklets if t["miss"] == 0]
    if not live:
        return None

    leader = max(live, key=lambda t: t["smooth_motion"])
    best_motion = leader["smooth_motion"]

    # Flat / ambiguous lip signal (or a single face / disabled) -> v1 scorer.
    if not active_speaker or len(live) == 1 or best_motion < MOTION_FLOOR:
        active = next((t for t in live if t["id"] == state["active_id"]), None)
        prev_cx = active["cx"] if active else None
        chosen = _best_by_area(live, prev_cx, width)
        state["active_id"] = chosen["id"]
        state["challenger_id"] = None
        state["challenge_count"] = 0
        return chosen["cx"]

    active = next((t for t in live if t["id"] == state["active_id"]), None)
    if active is None:  # current speaker left frame -> adopt the loudest mouth now
        state["active_id"] = leader["id"]
        state["challenger_id"] = None
        state["challenge_count"] = 0
        return leader["cx"]

    if leader["id"] != active["id"] and best_motion > active["smooth_motion"] * SPEAKER_MARGIN:
        if state["challenger_id"] == leader["id"]:
            state["challenge_count"] += 1
        else:
            state["challenger_id"] = leader["id"]
            state["challenge_count"] = 1
        if state["challenge_count"] >= SPEAKER_DWELL:
            state["active_id"] = leader["id"]
            state["challenger_id"] = None
            state["challenge_count"] = 0
            return leader["cx"]
    else:
        state["challenger_id"] = None
        state["challenge_count"] = 0
    return active["cx"]


def _step(small: np.ndarray, faces, scale: float, det_area: float, width: float,
          tracklets: list[dict], state: dict, next_id: int,
          active_speaker: bool = True, debug_sink: list | None = None,
          ) -> tuple[float | None, int]:
    """Process one sampled frame: build detections, link to tracklets, update
    motion, age/drop stale tracklets, then pick the active speaker's center.

    Mutates `tracklets` and `state` in place. Returns (center_x or None, next_id).
    When `debug_sink` is given, appends one per-sample overlay record to it.
    """
    dets: list[dict] = []
    if faces is not None and len(faces):
        rows = sorted(faces, key=lambda f: float(f[2]) * float(f[3]), reverse=True)
        for f in rows[:MAX_TRACKLETS]:
            x, y, w, h, conf = f[0], f[1], f[2], f[3], f[-1]
            dets.append({
                "cx": (x + w / 2) / scale,
                "area": (w * h) / det_area,
                "conf": float(conf),
                "patch": _extract_mouth_patch(small, f),
                "box": (x / scale, y / scale, w / scale, h / scale),  # full-res
            })

    matched = _link_detections(tracklets, dets, width)
    matched_t = set(matched.values())
    n_orig = len(tracklets)
    # age existing tracklets that got no detection this sample (before spawning new ones)
    for ti in range(n_orig):
        if ti not in matched_t:
            tracklets[ti]["miss"] += 1
    for di, det in enumerate(dets):
        if di in matched:
            _update_tracklet(tracklets[matched[di]], det)
        else:
            tracklets.append({
                "id": next_id, "cx": det["cx"], "prev_patch": det["patch"],
                "motion": 0.0, "motion_hist": [], "smooth_motion": 0.0,
                "miss": 0, "area": det["area"], "conf": det["conf"],
            })
            next_id += 1
    # drop stale tracklets
    tracklets[:] = [t for t in tracklets if t["miss"] <= TRACKLET_TTL]
    if len(tracklets) > MAX_TRACKLETS:
        tracklets.sort(key=lambda t: (t["miss"], -t["area"]))
        del tracklets[MAX_TRACKLETS:]

    center = _select_speaker(tracklets, state, width, active_speaker)
    if debug_sink is not None:
        active = next((t for t in tracklets if t["id"] == state["active_id"]), None)
        debug_sink.append({
            "boxes": [d["box"] for d in dets],
            "chosen_cx": center,
            "chosen_motion": round(active["smooth_motion"], 4) if active else 0.0,
        })
    return center, next_id


def _detect_centers(video: Path, model: Path,
                    on_progress: Callable[[float], None] | None = None,
                    active_speaker: bool = True, debug: bool = False,
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
    det_area = det_size[0] * det_size[1]
    detector = cv2.FaceDetectorYN.create(str(model), "", det_size, SCORE_THRESHOLD)

    times: list[float] = []
    centers: list[float | None] = []
    switches: list[bool] = []
    seen_ids: set[int] = set()
    debug_records: list | None = [] if debug else None
    tracklets: list[dict] = []
    state = {"active_id": None, "challenger_id": None, "challenge_count": 0}
    next_id = 0
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
        prev_active = state["active_id"]
        center, next_id = _step(small, faces, scale, det_area, width,
                                tracklets, state, next_id, active_speaker,
                                debug_records)
        # a speaker switch = active id changed to a different existing subject
        switches.append(prev_active is not None
                        and state["active_id"] != prev_active)
        seen_ids.update(t["id"] for t in tracklets if t["miss"] == 0)
        times.append(frame_idx / fps)
        centers.append(center)
        frame_idx += 1
        if on_progress and len(times) % 25 == 0:
            on_progress(frame_idx / max(1, n_frames))
    cap.release()
    info = {"fps": fps, "width": width, "height": height, "n_frames": n_frames,
            "switch_flags": switches, "n_speakers": len(seen_ids)}
    if debug_records is not None:
        info["debug"] = {"times": times, "records": debug_records}
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


def _smooth(times: list[float], centers: list[float], width: float,
            snap_flags: list[bool] | None = None) -> list[float]:
    """Shot-cut snap + dead-zone + EMA + pan-speed clamp.

    snap_flags[i] True forces an instant snap at sample i (used on speaker
    switches so the crop cuts rather than slow-pans when hosts are close).
    """
    target = smooth = centers[0]
    out = [smooth]
    for i in range(1, len(centers)):
        c = centers[i]
        dt = max(1e-3, times[i] - times[i - 1])
        forced = snap_flags is not None and snap_flags[i]
        if forced or abs(c - smooth) > SNAP_JUMP * width:
            target = smooth = c  # shot cut / speaker switch: snap, don't pan
        else:
            if abs(c - target) > DEAD_ZONE * width:
                target = c
            step = EMA_ALPHA * (target - smooth)
            max_step = MAX_PAN_SPEED * width * dt
            smooth += float(np.clip(step, -max_step, max_step))
        out.append(smooth)
    return out


def track(video: Path, model: Path,
          on_progress: Callable[[float], None] | None = None,
          active_speaker: bool = True, debug: bool = False) -> dict:
    """Compute per-frame crop x positions for a 9:16 crop of `video`.

    With active_speaker=True (default) the crop cuts to whoever is talking via
    lip-motion; False forces the v1 single-subject (area+stickiness) scorer.
    debug=True attaches per-sample overlay data under result["_debug"].

    Returns {"mode": "tracked"|"center", "crop_w", "crop_h", "fps", "n_frames",
             "x": [per-frame int], "detection_rate": float, "n_speakers": int}
    """
    times, raw, info = _detect_centers(video, model, on_progress, active_speaker, debug)
    width, height = info["width"], info["height"]
    crop_w = min(even(height * 9 / 16), even(width))
    n_frames = info["n_frames"]
    max_x = width - crop_w

    detection_rate = sum(c is not None for c in raw) / max(1, len(raw))
    base = {
        "crop_w": crop_w, "crop_h": height, "fps": info["fps"],
        "n_frames": n_frames, "detection_rate": round(detection_rate, 3),
        "n_speakers": info.get("n_speakers", 0),
    }
    if "debug" in info:
        base["_debug"] = info["debug"]
    if detection_rate < MIN_DETECTION_RATE or not raw:
        x = even(max_x / 2)
        return {**base, "mode": "center", "x": [x] * n_frames}

    centers = _smooth(times, _fill_misses(raw, width), width, info.get("switch_flags"))
    frame_times = np.arange(n_frames) / info["fps"]
    interp = np.interp(frame_times, times, centers)
    xs = [min(max(even(c - crop_w / 2), 0), max_x) for c in interp]
    return {**base, "mode": "tracked", "x": xs}


def write_sendcmd(result: dict, path: Path) -> None:
    fps = result["fps"]
    lines = [f"{i / fps:.4f} crop x {x};" for i, x in enumerate(result["x"])]
    path.write_text("\n".join(lines) + "\n")


def write_debug_preview(video: Path, model: Path, result: dict, out: Path) -> None:
    """Render a preview with the crop window drawn, for tuning.

    When result carries "_debug" (track(..., debug=True)) it also overlays every
    detected face thin gray and the chosen active speaker thick green with its
    mouth-motion score, so you can see why the crop picked who it did.
    """
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    crop_w = result["crop_w"]
    dbg = result.get("_debug")
    sample_times = np.asarray(dbg["times"]) if dbg else None
    for i in range(result["n_frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        if dbg is not None and len(sample_times):
            rec = dbg["records"][min(int(np.searchsorted(sample_times, i / fps)),
                                     len(dbg["records"]) - 1)]
            cx = rec["chosen_cx"]
            for bx, by, bw, bh in rec["boxes"]:
                p1, p2 = (int(bx), int(by)), (int(bx + bw), int(by + bh))
                chosen = cx is not None and bx <= cx <= bx + bw
                color = (0, 255, 0) if chosen else (160, 160, 160)
                cv2.rectangle(frame, p1, p2, color, 4 if chosen else 2)
                if chosen:
                    cv2.putText(frame, f"talk {rec['chosen_motion']:.3f}",
                                (int(bx), max(0, int(by) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        x = result["x"][min(i, len(result["x"]) - 1)]
        cv2.rectangle(frame, (x, 0), (x + crop_w, h), (0, 255, 255), 4)
        writer.write(frame)
    cap.release()
    writer.release()
