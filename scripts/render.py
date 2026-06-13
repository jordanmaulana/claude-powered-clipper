"""Render clips.json -> output/<id>/clip_NN_<slug>.mp4 (9:16, captions, silence removed)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import captions, facetrack, ffmpeg, timeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
OUT_W, OUT_H = 1080, 1920


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"error: {path} not found")
    return json.loads(path.read_text())


def validate(clip: dict, duration: float) -> None:
    if not 0 <= clip["start"] < clip["end"] <= duration + 1:
        raise ValueError(f"range {clip['start']}-{clip['end']} outside video (0-{duration:.0f}s)")
    if clip["end"] - clip["start"] > 120:
        print(f"  warn: clip {clip['id']} is {clip['end'] - clip['start']:.0f}s before gap removal (>120s)")


def nearest_speech(words: list[dict], t: float) -> str:
    w = min(words, key=lambda w: abs(w["s"] - t))
    i = words.index(w)
    ctx = " ".join(x["w"] for x in words[max(0, i - 5):i + 6])
    return f"nearest speech at {w['s']:.1f}s: ...{ctx}..."


def render_clip(clip: dict, words: list[dict], meta: dict, workdir: Path,
                outdir: Path, args: argparse.Namespace) -> dict:
    clip_dir = workdir / "clips" / f"{clip['id']:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    source = workdir / "source.mp4"

    clip_words = timeline.words_in_range(words, clip["start"], clip["end"])
    if not clip_words:
        raise ValueError(
            f"no speech in {clip['start']}-{clip['end']}s; {nearest_speech(words, clip['start'])}"
        )

    intervals = timeline.keep_intervals(clip_words, max_gap=args.gap, pad=args.pad)
    (clip_dir / "keep.json").write_text(json.dumps(intervals))
    clip_duration = timeline.total_duration(intervals)

    # Pass A: cut silence gaps -> flat 16:9 intermediate on the edited timeline
    flat = clip_dir / "clip_flat.mp4"
    graph = ffmpeg.cut_filtergraph(intervals, meta["fps"])
    script = clip_dir / "pass_a_filter.txt"
    script.write_text(graph)
    ffmpeg.run_with_progress(
        [ffmpeg.FFMPEG, "-y", "-i", str(source), "-filter_complex_script", str(script),
         "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "14", "-preset", "fast",
         "-c:a", "aac", "-b:a", "192k", str(flat)],
        "Pass A (silence cut)", clip_duration, clip_dir / "pass_a.log",
    )

    # Captions: remap word times onto the edited timeline, then chunk
    filters = []
    if not args.no_captions:
        remapped = [
            {"w": w["w"], "s": timeline.remap(w["s"], intervals), "e": timeline.remap(w["e"], intervals)}
            for w in clip_words
        ]
        ass_path = clip_dir / "captions.ass"
        ass_path.write_text(captions.build_ass(remapped))

    # Framing: face-track 16:9-ish sources; narrow sources scale-and-crop
    src_ratio = meta["width"] / meta["height"]
    if src_ratio <= 9 / 16 + 0.01:
        mode = "pad"
        filters.append(f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                       f"crop={OUT_W}:{OUT_H}")
    else:
        tty = sys.stdout.isatty()
        on_progress = (
            (lambda f: print(f"\r  face tracking: {f * 100:5.1f}%", end="", flush=True))
            if tty else None
        )
        result = facetrack.track(flat, MODEL, on_progress)
        if tty:
            print("\r  face tracking: 100.0%")
        mode = result["mode"]
        x0 = result["x"][0]
        crop = f"crop=w={result['crop_w']}:h={result['crop_h']}:x={x0}:y=0"
        if mode == "tracked":
            cmd_path = clip_dir / "crop.cmd"
            facetrack.write_sendcmd(result, cmd_path)
            filters.append(f"sendcmd=f='{ffmpeg.escape_filter_path(cmd_path)}',{crop}")
        else:
            filters.append(crop)
        filters.append(f"scale={OUT_W}:{OUT_H}:flags=lanczos")
        if args.debug:
            facetrack.write_debug_preview(flat, MODEL, result, clip_dir / "track_debug.mp4")

    if not args.no_captions:
        filters.append(f"ass='{ffmpeg.escape_filter_path(ass_path)}'")

    # Pass B: one encode — dynamic crop, scale, burn captions
    final = outdir / f"clip_{clip['id']:02d}_{clip['slug']}.mp4"
    graph_b = f"[0:v]{','.join(filters)}[v]"
    script_b = clip_dir / "pass_b_filter.txt"
    script_b.write_text(graph_b)
    ffmpeg.run_with_progress(
        [ffmpeg.FFMPEG, "-y", "-i", str(flat), "-filter_complex_script", str(script_b),
         "-map", "[v]", "-map", "0:a", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-c:v", "libx264", "-crf", "19", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(final)],
        "Pass B (frame + captions)", clip_duration, clip_dir / "pass_b.log",
    )
    return {
        "path": final,
        "duration": clip_duration,
        "cuts": len(intervals) - 1,
        "mode": mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--clip", type=int, default=None, help="render only this clip id")
    parser.add_argument("--gap", type=float, default=None, help="max kept inter-word gap (s)")
    parser.add_argument("--pad", type=float, default=None, help="padding around speech runs (s)")
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--debug", action="store_true", help="write face-track preview video")
    args = parser.parse_args()

    meta = load_json(args.workdir / "meta.json")
    transcript = load_json(args.workdir / "transcript.json")
    spec = load_json(args.workdir / "clips.json")
    defaults = spec.get("defaults", {})
    if args.gap is None:
        args.gap = defaults.get("max_gap", 0.5)
    if args.pad is None:
        args.pad = defaults.get("pad", 0.15)

    outdir = ROOT / "output" / meta["video_id"]
    outdir.mkdir(parents=True, exist_ok=True)

    clips = spec["clips"]
    if args.clip is not None:
        clips = [c for c in clips if c["id"] == args.clip]
        if not clips:
            sys.exit(f"error: no clip with id {args.clip} in clips.json")

    results, failures = [], []
    for i, clip in enumerate(clips, 1):
        label = f"clip {clip['id']:02d} ({clip['slug']})"
        print(f"[{i}/{len(clips)}] rendering {label}...")
        try:
            validate(clip, meta["duration"])
            r = render_clip(clip, transcript["words"], meta, args.workdir, outdir, args)
            results.append((clip, r))
            print(f"  done: {r['path'].name}  {r['duration']:.1f}s  cuts={r['cuts']}  mode={r['mode']}")
        except (ValueError, SystemExit) as e:
            failures.append((clip, str(e)))
            print(f"  FAILED: {e}")

    print(f"\n{len(results)} rendered, {len(failures)} failed -> {outdir}")
    for clip, r in results:
        print(f"  {r['path'].name}  |  {clip['title']}  |  {r['duration']:.0f}s")
    if failures:
        for clip, err in failures:
            print(f"  FAILED clip {clip['id']}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
