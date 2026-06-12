# YouTube AI Clipper

This repo turns a YouTube URL into N vertical short clips (1080x1920, burned captions,
silence removed, face-tracked crop). There is no app — **Claude Code is the orchestrator**:
you run the scripts below and you make the one creative decision (which segments to clip).

## Setup (once per machine)

```bash
uv sync
uv run scripts/doctor.py   # checks ffmpeg, downloads YuNet face model
```

Requires `ffmpeg`/`ffprobe` on PATH (brew install ffmpeg). First transcription downloads
~1.6 GB whisper weights from HuggingFace — it is not hung.

## Pipeline (follow in order)

```bash
# 1. Download — prints the work/<id> path
uv run scripts/download.py "<youtube-url>"

# 2. Transcribe — writes transcript.json (word timestamps) + transcript.md
uv run scripts/transcribe.py work/<id>
```

**3. Select clips (your job).** Read `work/<id>/transcript.md`. If it is long (>1500 lines),
read in chunks with offset/limit and note candidate moments per chunk before deciding.

Selection criteria:
- self-contained insight or story — no missing context, complete sentences at both ends
- strong hook in the first 3 seconds
- 20–90 s per clip (before silence removal)
- `[m:ss]` markers are block starts (~1 min resolution); interpolate within a block by
  word position to estimate start/end. Times only need to be roughly right — render.py
  snaps to word boundaries (±0.5 s).

Write `work/<id>/clips.json`:

```json
{
  "video_id": "<id>", "version": 1,
  "defaults": {"max_gap": 0.5, "pad": 0.15},
  "clips": [
    {"id": 1, "slug": "kebab-case-max-40-chars", "title": "Posting-ready title",
     "hook": "why this clip works", "start": 312.0, "end": 371.0}
  ]
}
```

```bash
# 4. Render — silence cut, face-tracked 9:16 crop, captions, -> output/<id>/
uv run scripts/render.py work/<id>
```

**5. Report** a table to the user: file, title, duration. Mention any clip that fell back
to `mode=center` (no face found — normal for slides/B-roll).

## Iterating on feedback

- Re-render one clip after editing clips.json: `render.py work/<id> --clip 3`
- Silence aggressiveness: `--gap 0.4` (tighter) / `--pad 0.2` (more breathing room)
- Face-track tuning preview: `--debug` writes `work/<id>/clips/NN/track_debug.mp4`
- All stages cache; `--force` re-runs download/transcribe

## Files

- `work/<id>/meta.json` — title, duration, dimensions, fps
- `work/<id>/transcript.json` — word-level timestamps (machine truth for render)
- `work/<id>/transcript.md` — what you read to pick clips
- `work/<id>/clips.json` — what you write
- `work/<id>/clips/NN/` — per-clip debug: keep.json (cut intervals), captions.ass, crop.cmd
- `output/<id>/clip_NN_<slug>.mp4` — deliverables

## Troubleshooting

- Download fails (age/region/members): retry with `--cookies-from-browser chrome` (ask user first)
- "no speech in range": clips.json times point at music/silence — error lists nearest dialogue
- Captions out of sync: compare clips/NN/keep.json vs captions.ass — both must be edited-timeline
- Two-host podcasts: tracker follows one subject (v1 limitation) — warn the user
