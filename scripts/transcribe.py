"""Transcribe work/<id>/audio.wav -> transcript.json (word timestamps) + transcript.md"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
BLOCK_WORDS = 50
BLOCK_SECONDS = 60.0


def fmt_ts(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def run_whisper(audio: Path, model: str) -> dict:
    try:
        import mlx_whisper
        return mlx_whisper.transcribe(str(audio), path_or_hf_repo=model, word_timestamps=True)
    except ImportError:
        from faster_whisper import WhisperModel
        wm = WhisperModel("large-v3", compute_type="int8")
        segments, info = wm.transcribe(str(audio), word_timestamps=True)
        return {
            "language": info.language,
            "segments": [
                {
                    "start": seg.start, "end": seg.end, "text": seg.text,
                    "words": [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words or []],
                }
                for seg in segments
            ],
        }


def build_markdown(segments: list[dict], meta: dict) -> str:
    title = meta.get("title", "?")
    channel = meta.get("channel", "?")
    duration = fmt_ts(meta.get("duration", 0))
    lines = [f"# {title} — channel: {channel} — {duration}", ""]

    block_start, block_words = None, []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if block_start is None:
            block_start = seg["start"]
        block_words.extend(text.split())
        if len(block_words) >= BLOCK_WORDS or seg["end"] - block_start >= BLOCK_SECONDS:
            lines.append(f"[{fmt_ts(block_start)}] {' '.join(block_words)}")
            block_start, block_words = None, []
    if block_words:
        lines.append(f"[{fmt_ts(block_start)}] {' '.join(block_words)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    audio = args.workdir / "audio.wav"
    if not audio.exists():
        sys.exit(f"error: {audio} not found — run download.py first")
    out_json = args.workdir / "transcript.json"
    out_md = args.workdir / "transcript.md"

    if out_json.exists() and not args.force:
        print(f"cached: {out_json}")
    else:
        print(f"transcribing with {args.model} (first run downloads model weights)...")
        result = run_whisper(audio, args.model)
        words = [
            {"w": w["word"].strip(), "s": round(float(w["start"]), 3), "e": round(float(w["end"]), 3)}
            for seg in result["segments"]
            for w in seg.get("words", [])
            if w["word"].strip()
        ]
        transcript = {
            "language": result.get("language"),
            "words": words,
            "segments": [
                {"s": round(float(s["start"]), 3), "e": round(float(s["end"]), 3), "text": s["text"].strip()}
                for s in result["segments"]
                if s["text"].strip()
            ],
        }
        out_json.write_text(json.dumps(transcript, ensure_ascii=False))

    transcript = json.loads(out_json.read_text())
    meta = json.loads((args.workdir / "meta.json").read_text())
    out_md.write_text(build_markdown(transcript["segments"], meta))

    print(f"language: {transcript['language']}")
    print(f"words:    {len(transcript['words'])}")
    print(f"read:     {out_md}")
    print(f"next: read transcript.md, write {args.workdir}/clips.json, then run render.py")


if __name__ == "__main__":
    main()
