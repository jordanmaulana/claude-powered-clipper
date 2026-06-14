"""Render a transcript's segments into the readable transcript.md."""

BLOCK_WORDS = 50
BLOCK_SECONDS = 60.0


def fmt_ts(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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
            block_start = seg["s"]
        block_words.extend(text.split())
        if len(block_words) >= BLOCK_WORDS or seg["e"] - block_start >= BLOCK_SECONDS:
            lines.append(f"[{fmt_ts(block_start)}] {' '.join(block_words)}")
            block_start, block_words = None, []
    if block_words:
        lines.append(f"[{fmt_ts(block_start)}] {' '.join(block_words)}")
    return "\n".join(lines) + "\n"
