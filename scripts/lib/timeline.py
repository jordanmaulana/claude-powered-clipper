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
    runs: list[list[dict]] = [[words[0]]]
    for prev, cur in zip(words, words[1:]):
        if cur["s"] - prev["e"] > max_gap:
            runs.append([cur])
        else:
            runs[-1].append(cur)

    intervals = [(max(0.0, run[0]["s"] - pad), run[-1]["e"] + pad) for run in runs]
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


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
