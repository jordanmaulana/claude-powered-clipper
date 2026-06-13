"""Apply a word-substitution map to a transcript (words + segments).

Pure functions, no I/O. Corrections fix transcription errors (proper nouns,
brand/technical terms, code-switched English, homophones, casing) by substituting
single tokens. Timestamps are never touched, so captions stay in sync.
"""

import re

# Leading/trailing punctuation stripped off a word token before matching.
_PUNCT = "\"'.,!?;:()[]{}…—-"


def is_single_token(key: str) -> bool:
    """A valid correction key is one whitespace-free token."""
    return bool(key) and not any(c.isspace() for c in key)


def _split_affix(token: str) -> tuple[str, str, str]:
    """Split a token into (leading punct, core, trailing punct)."""
    core = token.strip(_PUNCT)
    if not core:
        return "", token, ""
    start = token.find(core)
    return token[:start], core, token[start + len(core):]


def _correct_token(token: str, lookup: dict[str, str]) -> tuple[str, bool]:
    """Return (token with core normalized, matched?). Case-insensitive whole-word.

    `matched` is True whenever the core is a known key — even if already correctly
    cased — so hit counts reflect occurrences, and a 0-hit key signals a typo.
    """
    lead, core, trail = _split_affix(token)
    repl = lookup.get(core.lower())
    if repl is None:
        return token, False
    return lead + repl + trail, True


def apply_corrections(transcript: dict, mapping: dict[str, str]) -> tuple[dict, dict]:
    """Apply `mapping` (wrong -> right) to transcript words + segments.

    Returns (corrected_transcript, hits) where hits maps each lowercased key to the
    number of substitutions made across words and segments combined. Keys that are not
    single tokens are ignored (they would let words[] and segments[] drift apart).

    The input transcript is not mutated.
    """
    lookup = {k.lower(): v for k, v in mapping.items() if is_single_token(k)}
    hits = {k: 0 for k in lookup}
    if not lookup:
        return {**transcript, "words": list(transcript.get("words", [])),
                "segments": list(transcript.get("segments", []))}, hits

    # words: token-by-token, preserving punctuation and timestamps.
    new_words = []
    for word in transcript.get("words", []):
        token, matched = _correct_token(word["w"], lookup)
        if matched:
            hits[_split_affix(word["w"])[1].lower()] += 1
            word = {**word, "w": token}
        new_words.append(word)

    # segments: regex whole-word replace on free text.
    def _seg_sub(text: str) -> str:
        def repl(m: re.Match) -> str:
            nonlocal_key = m.group(0).lower()
            hits[nonlocal_key] += 1
            return lookup[nonlocal_key]
        # one alternation of all keys, longest first, word-bounded, case-insensitive
        pattern = r"\b(" + "|".join(re.escape(k) for k in sorted(lookup, key=len, reverse=True)) + r")\b"
        return re.sub(pattern, repl, text, flags=re.IGNORECASE)

    new_segments = [{**s, "text": _seg_sub(s["text"])} for s in transcript.get("segments", [])]

    corrected = {**transcript, "words": new_words, "segments": new_segments}
    return corrected, hits


# A word token that continues a number: leading separator + digit (".000", ",5").
_NUM_CONT = re.compile(r"^[.,]\d")
_ENDS_DIGIT = re.compile(r"\d$")
_SEG_SPLIT_NUM = re.compile(r"(\d)\s+([.,]\d)")


def merge_split_numbers(transcript: dict) -> tuple[dict, int]:
    """Glue Whisper's split number tokens back together.

    Indonesian thousands/decimal separators get tokenized apart at the word level:
    `1` `.500` `.000` -> `1.500.000`, `9` `,5` -> `9,5`. Burned captions are built from
    `words`, so they render the broken form; this repairs them. Segments are normalized
    defensively (Whisper's segment text is usually already joined). Idempotent.

    Returns (corrected_transcript, number_of_word_merges). Input is not mutated.
    """
    new_words: list[dict] = []
    merges = 0
    for word in transcript.get("words", []):
        if new_words and _NUM_CONT.match(word["w"]) and _ENDS_DIGIT.search(new_words[-1]["w"]):
            prev = new_words[-1]
            new_words[-1] = {**prev, "w": prev["w"] + word["w"], "e": word["e"]}
            merges += 1
        else:
            new_words.append(word)

    def _join_segment(text: str) -> str:
        prev = None
        while prev != text:
            prev, text = text, _SEG_SPLIT_NUM.sub(r"\1\2", text)
        return text

    new_segments = [{**s, "text": _join_segment(s["text"])} for s in transcript.get("segments", [])]

    corrected = {**transcript, "words": new_words, "segments": new_segments}
    return corrected, merges
