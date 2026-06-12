"""Word timestamps (edited timeline) -> phrase chunks -> styled .ass subtitle file."""

FONT = "Arial Black"
FONT_SIZE = 72
MAX_WORDS = 4
MAX_PHRASE_SECONDS = 1.8
FLUSH_GAP = 0.35          # gap to next word that ends a phrase
LINGER_GAP = 0.3          # extend phrase display into a following gap up to this long
SENTENCE_END = ".?!,;"
UPPERCASE = True

ASS_HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{FONT},{FONT_SIZE},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,2,2,90,90,560,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def chunk_phrases(words: list[dict]) -> list[dict]:
    """Greedy phrase accumulation. Words must be on the edited timeline.

    Returns [{"text": str, "s": float, "e": float}, ...]
    """
    phrases: list[dict] = []
    current: list[dict] = []

    def flush():
        if current:
            phrases.append({
                "text": " ".join(w["w"] for w in current),
                "s": current[0]["s"],
                "e": current[-1]["e"],
            })
            current.clear()

    for i, word in enumerate(words):
        current.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if (
            len(current) >= MAX_WORDS
            or current[-1]["e"] - current[0]["s"] > MAX_PHRASE_SECONDS
            or word["w"].rstrip("\"')")[-1:] in SENTENCE_END
            or (nxt and nxt["s"] - word["e"] > FLUSH_GAP)
        ):
            flush()
    flush()

    # extend display into short gaps so text doesn't flicker off between phrases
    for cur, nxt in zip(phrases, phrases[1:]):
        if 0 < nxt["s"] - cur["e"] < LINGER_GAP:
            cur["e"] = nxt["s"]
    return phrases


def build_ass(words: list[dict]) -> str:
    events = []
    for p in chunk_phrases(words):
        text = p["text"].upper() if UPPERCASE else p["text"]
        text = text.replace("{", "(").replace("}", ")")  # ASS override-tag chars
        events.append(
            f"Dialogue: 0,{ass_time(p['s'])},{ass_time(p['e'])},Cap,,0,0,0,,{text}"
        )
    return ASS_HEADER + "\n".join(events) + "\n"
