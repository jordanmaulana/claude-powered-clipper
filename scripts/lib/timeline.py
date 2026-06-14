"""Keep-interval computation and original->edited timestamp remapping.

Pure functions over word timestamp lists: [{"w": str, "s": float, "e": float}, ...]
"""


def words_in_range(words: list[dict], start: float, end: float, snap: float = 0.5) -> list[dict]:
    """Words whose span falls inside [start - snap, end + snap]."""
    return [w for w in words if w["s"] >= start - snap and w["e"] <= end + snap]


def keep_intervals(
    words: list[dict],
    max_gap: float = 0.5,
    pad: float = 0.15,
) -> list[tuple[float, float]]:
    """Group consecutive words into kept speech runs, dropping gaps > max_gap.

    Each run becomes [first.s - pad, last.e + pad]; overlapping/touching padded
    intervals are merged. Returns [] if no words.
    """
    if not words:
        return []
    runs = _runs(words, max_gap)
    intervals = [(max(0.0, run[0]["s"] - pad), run[-1]["e"] + pad) for run in runs]
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _runs(words: list[dict], max_gap: float) -> list[list[dict]]:
    """Group words into continuous speech runs, splitting on gaps > max_gap."""
    runs: list[list[dict]] = [[words[0]]]
    for prev, cur in zip(words, words[1:]):
        if cur["s"] - prev["e"] > max_gap:
            runs.append([cur])
        else:
            runs[-1].append(cur)
    return runs


def midstatement_end(words: list[dict], end: float, max_gap: float = 0.5) -> float | None:
    """Time the speaker actually pauses, if `end` lands inside a continuous speech run.

    `end` is mid-statement when a run still has speech after it (run.first <= end < run.last);
    return that run's last word end. Return None when `end` already sits at/after a pause
    (between runs, or past the last word) — nothing to fix.
    """
    if not words:
        return None
    for run in _runs(words, max_gap):
        if run[0]["s"] <= end < run[-1]["e"]:
            return run[-1]["e"]
    return None


def snap_end(
    words: list[dict],
    end: float,
    max_gap: float = 0.5,
    max_extend: float = 6.0,
) -> float:
    """Auto-extend `end` to the next natural pause, capped at end + max_extend.

    If `end` is not mid-statement, return it unchanged. If the pause is within the cap,
    snap to it. Otherwise the run runs longer than the cap: back off to the last
    sentence-final (.?!) word within [end, end + max_extend], or the cap itself if none.
    """
    pause = midstatement_end(words, end, max_gap)
    if pause is None:
        return end
    if pause - end <= max_extend:
        return pause
    limit = end + max_extend
    ends = [w["e"] for w in words if w["s"] > end and w["e"] <= limit and w["w"][-1:] in ".?!"]
    return ends[-1] if ends else limit


def remap(t: float, intervals: list[tuple[float, float]]) -> float:
    """Map an original-timeline time onto the edited (gaps-removed) timeline.

    Times inside a removed gap clamp to the cut point.
    """
    edited = 0.0
    for s, e in intervals:
        if t < s:
            return edited
        if t <= e:
            return edited + (t - s)
        edited += e - s
    return edited


def total_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in intervals)
