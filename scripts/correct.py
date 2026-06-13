"""Apply work/<id>/corrections.json to transcript.json -> rebuild transcript.md

Fixes transcription errors the AI spotted while reading transcript.md (proper nouns,
brand/technical terms, code-switched English, homophones, casing). Substitutions only —
timestamps are untouched, so captions stay in sync.

corrections.json format:
  {"video_id": "<id>", "corrections": {"wrong": "Right", "leverege": "leverage"}}

transcript.raw.json is the immutable Whisper output: created on first run, and the map is
always applied to it (not cumulatively), so editing the map and re-running is idempotent.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.correct import apply_corrections, is_single_token, merge_split_numbers
from transcribe import build_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()

    out_json = args.workdir / "transcript.json"
    raw_json = args.workdir / "transcript.raw.json"
    if not out_json.exists() and not raw_json.exists():
        sys.exit(f"error: {out_json} not found — run transcribe.py first")

    # Preserve the original Whisper output once; always correct from it.
    if not raw_json.exists():
        raw_json.write_text(out_json.read_text())
    transcript = json.loads(raw_json.read_text())

    # Split-number merge always runs (deterministic, invisible in transcript.md).
    transcript, merged = merge_split_numbers(transcript)

    # The substitution map is optional.
    corr_path = args.workdir / "corrections.json"
    mapping = {}
    if corr_path.exists():
        mapping = json.loads(corr_path.read_text()).get("corrections", {})
        for k in mapping:
            if not is_single_token(k):
                print(f"skipped multi-word key (single-token only): {k!r}")
    else:
        print(f"no {corr_path} — applying number-merge only")

    corrected, hits = apply_corrections(transcript, mapping)
    out_json.write_text(json.dumps(corrected, ensure_ascii=False))

    meta = json.loads((args.workdir / "meta.json").read_text())
    (args.workdir / "transcript.md").write_text(build_markdown(corrected["segments"], meta))

    print(f"merged {merged} split number(s)")
    total = sum(hits.values())
    if hits:
        print(f"applied {total} substitution(s) across {len(hits)} term(s):")
        for k, n in sorted(hits.items(), key=lambda kv: -kv[1]):
            flag = "  <- 0 hits, check spelling" if n == 0 else ""
            print(f"  {k!r} -> {mapping_value(mapping, k)!r}: {n}{flag}")
    print(f"rebuilt: {args.workdir / 'transcript.md'}")


def mapping_value(mapping: dict, key_lower: str) -> str:
    """Original-cased value for a lowercased hit key."""
    for k, v in mapping.items():
        if k.lower() == key_lower:
            return v
    return key_lower


if __name__ == "__main__":
    main()
