# aiclipper

YouTube → N vertical short clips (1080x1920), with burned captions, silence removal,
and face-tracked cropping. No UI: open this repo in Claude Code and say

> Clip this https://www.youtube.com/watch?v=... into 10 meaningful insight clips.

Claude downloads, transcribes (local whisper), picks the segments, and renders.

## Setup

```bash
brew install ffmpeg
uv sync
uv run scripts/doctor.py
```

## Manual pipeline

See [CLAUDE.md](CLAUDE.md) — same steps work by hand; clip selection is the only
step that needs a human (or Claude).
