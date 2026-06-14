"""Render clips.json -> output/<id>/clip_NN_<slug>.mp4 (9:16, captions, silence removed)."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import captions, facetrack, ffmpeg, timeline  # noqa: E402
from lib.io import load_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
OUT_W, OUT_H = 1080, 1920
PROXY_W = 960  # face-detection proxy width (cut + downscaled, throwaway encode)


def validate(clip: dict, duration: float) -> None:
    if not 0 <= clip["start"] < clip["end"] <= duration + 1:
        raise ValueError(f"range {clip['start']}-{clip['end']} outside video (0-{duration:.0f}s)")
    if not clip.get("summary", "").strip():
        raise ValueError(f"clip {clip['id']} has no 'summary' — every clip must state its insight")
    banned = [";", "—", "–", "--"]
    hit = next((b for b in banned if b in clip["summary"]), None)
    if hit:
        raise ValueError(f"clip {clip['id']} summary uses banned '{hit}' — rewrite without it")
    if clip["end"] - clip["start"] > 120:
        print(f"  warn: clip {clip['id']} is {clip['end'] - clip['start']:.0f}s before gap removal (>120s)")


def nearest_speech(words: list[dict], t: float) -> str:
    w = min(words, key=lambda w: abs(w["s"] - t))
    i = words.index(w)
    ctx = " ".join(x["w"] for x in words[max(0, i - 5):i + 6])
    return f"nearest speech at {w['s']:.1f}s: ...{ctx}..."


def render_clip(clip: dict, words: list[dict], meta: dict, workdir: Path,
                outdir: Path, args: argparse.Namespace,
                threads: int = 0, quiet: bool = False) -> dict:
    clip_dir = workdir / "clips" / f"{clip['id']:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    source = workdir / "source.mp4"

    clip_end = clip["end"]
    if args.snap_end:
        snapped = timeline.snap_end(words, clip_end, max_gap=args.gap)
        if snapped > clip_end and not quiet:
            print(f"  snap: clip {clip['id']} end {clip_end:.1f} -> {snapped:.1f}s (next pause)")
        clip_end = snapped
    else:
        pause = timeline.midstatement_end(words, clip_end, max_gap=args.gap)
        if pause is not None and not quiet:
            print(f"  warn: clip {clip['id']} ends mid-statement at {clip_end:.1f}s; "
                  f"speaker pauses at {pause:.1f}s: {nearest_speech(words, pause)} "
                  f"(extend end past it, or pass --snap-end)")

    clip_words = timeline.words_in_range(words, clip["start"], clip_end)
    if not clip_words:
        raise ValueError(
            f"no speech in {clip['start']}-{clip['end']}s; {nearest_speech(words, clip['start'])}"
        )

    intervals = timeline.keep_intervals(clip_words, max_gap=args.gap, pad=args.pad)
    (clip_dir / "keep.json").write_text(json.dumps(intervals))
    clip_duration = timeline.total_duration(intervals)
    cut = ffmpeg.cut_filtergraph(intervals, meta["fps"])  # produces [v] (video) and [a] (audio)

    # Captions: remap word times onto the edited timeline, then chunk
    if not args.no_captions:
        remapped = [
            {"w": w["w"], "s": timeline.remap(w["s"], intervals), "e": timeline.remap(w["e"], intervals)}
            for w in clip_words
        ]
        ass_path = clip_dir / "captions.ass"
        ass_path.write_text(captions.build_ass(remapped))

    # Framing: face-track 16:9-ish sources; narrow sources scale-and-crop
    src_w, src_h = meta["width"], meta["height"]
    src_ratio = src_w / src_h
    video_filters: list[str] = []
    n_speakers = 0
    if src_ratio <= 9 / 16 + 0.01:
        mode = "pad"
        video_filters.append(f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                             f"crop={OUT_W}:{OUT_H}")
    else:
        # Detection proxy: cut + downscaled throwaway encode for face tracking only.
        proxy = clip_dir / "proxy.mp4"
        # anullsink consumes the cut graph's [a] output (proxy is video-only, -an).
        proxy_graph = f"{cut};\n[v]scale={PROXY_W}:-2[vp];[a]anullsink"
        proxy_script = clip_dir / "proxy_filter.txt"
        proxy_script.write_text(proxy_graph)
        ffmpeg.run_with_progress(
            [ffmpeg.FFMPEG, "-y", "-i", str(source), "-filter_complex_script", str(proxy_script),
             "-map", "[vp]", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
             *(["-threads", str(threads)] if threads else []), str(proxy)],
            "proxy (face detect)", clip_duration, clip_dir / "proxy.log", show_progress=not quiet,
        )

        tty = sys.stdout.isatty() and not quiet
        on_progress = (
            (lambda f: print(f"\r  face tracking: {f * 100:5.1f}%", end="", flush=True))
            if tty else None
        )
        # track() detects on the proxy but returns crop geometry in source pixels.
        result = facetrack.track(proxy, MODEL, on_progress,
                                 active_speaker=not args.no_active_speaker,
                                 debug=args.debug, out_dims=(src_w, src_h))
        if tty:
            print("\r  face tracking: 100.0%")
        mode = result["mode"]
        n_speakers = result.get("n_speakers", 0)
        x0 = result["x"][0]
        crop = f"crop=w={result['crop_w']}:h={result['crop_h']}:x={x0}:y=0"
        if mode == "tracked":
            cmd_path = clip_dir / "crop.cmd"
            facetrack.write_sendcmd(result, cmd_path)
            video_filters.append(f"sendcmd=f='{ffmpeg.escape_filter_path(cmd_path)}',{crop}")
        else:
            video_filters.append(crop)
        video_filters.append(f"scale={OUT_W}:{OUT_H}:flags=lanczos")
        if args.debug:
            facetrack.write_debug_preview(proxy, MODEL, result, clip_dir / "track_debug.mp4")

    if not args.no_captions:
        video_filters.append(f"ass='{ffmpeg.escape_filter_path(ass_path)}'")

    # Single encode from source: cut, dynamic crop, scale, burn captions, loudnorm.
    final = outdir / f"clip_{clip['id']:02d}_{clip['slug']}.mp4"
    graph = (f"{cut};\n"
             f"[v]{','.join(video_filters)}[vout];\n"
             f"[a]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
    script = clip_dir / "render_filter.txt"
    script.write_text(graph)
    ffmpeg.run_with_progress(
        [ffmpeg.FFMPEG, "-y", "-i", str(source), "-filter_complex_script", str(script),
         "-map", "[vout]", "-map", "[aout]",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
         *(["-threads", str(threads)] if threads else []), str(final)],
        "encode (frame + captions)", clip_duration, clip_dir / "render.log", show_progress=not quiet,
    )

    # Insight caption: title + the viewer-facing takeaway, beside the mp4
    md_path = final.with_suffix(".md")
    md_path.write_text(f"# {clip['title']}\n\n{clip['summary'].strip()}\n")

    return {
        "path": final,
        "md": md_path,
        "duration": clip_duration,
        "cuts": len(intervals) - 1,
        "mode": mode,
        "n_speakers": n_speakers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--clip", type=int, default=None, help="render only this clip id")
    parser.add_argument("--gap", type=float, default=None, help="max kept inter-word gap (s)")
    parser.add_argument("--pad", type=float, default=None, help="padding around speech runs (s)")
    parser.add_argument("--snap-end", action="store_true",
                        help="auto-extend each clip end to the next natural pause (capped)")
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--no-active-speaker", action="store_true",
                        help="disable lip-motion speaker switching (v1 single-subject crop)")
    parser.add_argument("--debug", action="store_true", help="write face-track preview video")
    parser.add_argument("--jobs", type=int, default=None,
                        help="clips to render concurrently (default: min(4, cores/2))")
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

    cores = os.cpu_count() or 4
    jobs = max(1, min(args.jobs if args.jobs else min(4, cores // 2), len(clips)))
    threads = max(1, cores // jobs) if jobs > 1 else 0  # cap ffmpeg threads per concurrent job
    quiet = jobs > 1

    def _one(idx: int, clip: dict) -> tuple[dict, dict]:
        label = f"clip {clip['id']:02d} ({clip['slug']})"
        print(f"[{idx}/{len(clips)}] rendering {label}...", flush=True)
        validate(clip, meta["duration"])
        r = render_clip(clip, transcript["words"], meta, args.workdir, outdir, args,
                        threads=threads, quiet=quiet)
        spk = f"  speakers={r['n_speakers']}" if r.get("n_speakers", 0) > 1 else ""
        print(f"  done: {r['path'].name}  {r['duration']:.1f}s  cuts={r['cuts']}  "
              f"mode={r['mode']}{spk}", flush=True)
        return clip, r

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_one, i, c): c for i, c in enumerate(clips, 1)}
        for fut in as_completed(futs):
            clip = futs[fut]
            try:
                results.append(fut.result())
            except (ValueError, SystemExit) as e:
                failures.append((clip, str(e)))
                print(f"  FAILED clip {clip['id']}: {e}", flush=True)
    results.sort(key=lambda cr: cr[0]["id"])

    print(f"\n{len(results)} rendered, {len(failures)} failed -> {outdir}")
    for clip, r in results:
        print(f"  {r['path'].name}  |  {clip['title']}  |  {r['duration']:.0f}s")
    if failures:
        for clip, err in failures:
            print(f"  FAILED clip {clip['id']}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
