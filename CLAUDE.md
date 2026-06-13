# YouTube AI Clipper

This repo turns a YouTube URL into vertical short clips — one per qualifying segment,
however many that is (1080x1920, burned captions, silence removed, face-tracked crop). There is no app — **Claude Code is the orchestrator**:
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

Transcribe auto-merges Whisper's split numbers (`25.000` would otherwise become `25 .000`
in captions). Transcripts made before this fix are repaired the first time `correct.py` runs.

**3. Proof-read the transcript (your job).** Whisper mishears proper nouns, brand/product
names, technical jargon, code-switched English, and homophones. While reading
`transcript.md`, collect terms it clearly got wrong — **only** where context makes the right
word unambiguous; do not guess, and do not try to add words Whisper dropped (this step does
substitutions only). Write `work/<id>/corrections.json`:

```json
{"video_id": "<id>", "corrections": {"leverege": "leverage", "naval": "Naval"}}
```

Keys are single tokens, case-insensitive; the value sets the final spelling/casing. Then:

```bash
uv run scripts/correct.py work/<id>   # patches transcript.json words+segments, rebuilds .md
```

The report lists per-term hit counts — a key showing **0 hits** is a typo in your map; fix
it and re-run (it always re-applies from `transcript.raw.json`, so it is idempotent). Skip
this step if the transcript reads clean. Re-read the corrected `transcript.md` before
selecting.

**4. Select clips (your job).** Read `work/<id>/transcript.md`. If it is long (>1500 lines),
read in chunks with offset/limit and note candidate moments per chunk before deciding.
Select **every** segment that meets the criteria below — there is no target count; a dense
hour-long video may yield 15+ clips, a thin one only 2–3. Do not stop after finding "enough".

Selection criteria:
- self-contained insight or story — no missing context, complete sentences at both ends
- strong hook in the first 3 seconds
- summarizeable — the clip delivers one clear insight/takeaway you can state in 1–2
  sentences. If you can't state it, drop the clip.
- 20–60 s per clip (before silence removal)
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
     "hook": "why this clip works", "summary": "the one insight the viewer takes away",
     "start": 312.0, "end": 371.0}
  ]
}
```

`hook` is your internal rationale (why the clip works); `summary` is the viewer-facing
insight — render.py writes it to `output/<id>/clip_NN_<slug>.md`. **Every clip needs a
non-empty `summary`** or render fails that clip.

```bash
# 5. Render — silence cut, face-tracked 9:16 crop, captions, -> output/<id>/
uv run scripts/render.py work/<id>
```

**6. Report** a table to the user: file, title, duration. Each clip also gets a
`clip_NN_<slug>.md` caption (title + summary) beside its mp4. Mention any clip that fell
back to `mode=center` (no face found — normal for slides/B-roll).

## Iterating on feedback

- Re-render one clip after editing clips.json: `render.py work/<id> --clip 3`
- Silence aggressiveness: `--gap 0.4` (tighter) / `--pad 0.2` (more breathing room)
- Face-track tuning preview: `--debug` writes `work/<id>/clips/NN/track_debug.mp4`
- All stages cache; `--force` re-runs download/transcribe

## Files

- `work/<id>/meta.json` — title, duration, dimensions, fps
- `work/<id>/transcript.json` — word-level timestamps (machine truth for render)
- `work/<id>/transcript.raw.json` — original Whisper output, kept once correct.py runs
- `work/<id>/transcript.md` — what you read to pick clips
- `work/<id>/corrections.json` — optional {wrong: right} map you write for step 3
- `work/<id>/clips.json` — what you write
- `work/<id>/clips/NN/` — per-clip debug: keep.json (cut intervals), captions.ass, crop.cmd
- `output/<id>/clip_NN_<slug>.mp4` — deliverables
- `output/<id>/clip_NN_<slug>.md` — insight caption (title + summary)

## Troubleshooting

- Download fails (age/region/members): retry with `--cookies-from-browser chrome` (ask user first)
- "no speech in range": clips.json times point at music/silence — error lists nearest dialogue
- Captions out of sync: compare clips/NN/keep.json vs captions.ass — both must be edited-timeline
- Two-host podcasts: tracker follows one subject (v1 limitation) — warn the user
