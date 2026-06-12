"""Download a YouTube video -> work/<video_id>/{source.mp4, audio.wav, meta.json}"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ffmpeg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FORMAT = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*+ba/b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--workdir", type=Path, default=ROOT / "work")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER", default=None,
                        help="use browser cookies for age-restricted videos (e.g. chrome)")
    args = parser.parse_args()

    ffmpeg.require_binaries()
    import yt_dlp

    base_opts = {"quiet": True}
    if args.cookies_from_browser:
        base_opts["cookiesfrombrowser"] = (args.cookies_from_browser,)

    with yt_dlp.YoutubeDL(base_opts) as ydl:
        try:
            info = ydl.extract_info(args.url, download=False)
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if "age" in msg.lower() or "sign in" in msg.lower() or "members" in msg.lower():
                sys.exit(
                    f"error: video requires authentication.\n{msg}\n"
                    f"retry with browser cookies:\n"
                    f"  uv run scripts/download.py '{args.url}' --cookies-from-browser chrome"
                )
            sys.exit(f"error: cannot access video:\n{msg}")

    if info.get("is_live"):
        sys.exit("error: live streams are not supported")

    video_id = info["id"]
    workdir = args.workdir / video_id
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "source.mp4"
    audio = workdir / "audio.wav"

    if source.exists() and not args.force:
        print(f"cached: {source}")
    else:
        opts = {
            **base_opts,
            "format": FORMAT,
            "merge_output_format": "mp4",
            "outtmpl": str(source),
            "quiet": False,
            "noprogress": False,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([args.url])

    if not audio.exists() or args.force:
        ffmpeg.run(
            [ffmpeg.FFMPEG, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(audio)],
            "audio extraction",
        )

    stream = ffmpeg.probe(source)
    meta = {
        "video_id": video_id,
        "url": args.url,
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "duration": stream["duration"],
        "width": stream["width"],
        "height": stream["height"],
        "fps": stream["fps"],
    }
    (workdir / "meta.json").write_text(json.dumps(meta, indent=2))

    mins, secs = divmod(int(stream["duration"]), 60)
    print(f"\nworkdir: {workdir}")
    print(f"title:   {meta['title']}")
    print(f"duration: {mins}:{secs:02d}  ({stream['width']}x{stream['height']} @ {stream['fps']:.2f}fps)")
    print(f"next: uv run scripts/transcribe.py {workdir}")


if __name__ == "__main__":
    main()
