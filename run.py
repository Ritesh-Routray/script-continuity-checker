"""Continuity checker for AI-generated microdrama scripts.

Usage:
  python run.py <folder with episode .txt files> [--out outputs/myshow]
                [--provider gemini|groq|openai|none] [--model NAME] [--episodes 1-20]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from cc.parser import load_show
from cc.report import write_all
from cc.rules import run_rules


def parse_range(s: str | None):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--provider", default=None, help="gemini | groq | openai | none (rules only)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", default=None, help="e.g. 1-20 or 1,5,9")
    ap.add_argument("--no-camera", action="store_true", help="drop CAMERA lines from LLM prompts (cheaper)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-facts", type=int, default=140, help="bible facts sent per prompt (lower for small TPM limits)")
    a = ap.parse_args(argv)

    t0 = time.time()
    folder = Path(a.folder)
    episodes = load_show(folder)
    keep = parse_range(a.episodes)
    if keep:
        episodes = [e for e in episodes if e.number in keep]
    show = folder.name
    out = Path(a.out or Path("outputs") / show.replace(" ", "_")[:60])
    print(f"Loaded {len(episodes)} episodes, {sum(len(e.events) for e in episodes)} timed blocks from {folder}")

    flags = run_rules(episodes)
    print(f"Rules: {len(flags)} flags")

    meta = {"show": show, "episodes": len(episodes), "model": "rules only",
            "generated": dt.datetime.now().isoformat(timespec="seconds")}

    from cc.llm import LLM, LLMError, load_dotenv
    load_dotenv()
    import os
    provider = (a.provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider != "none":
        try:
            from cc.semantic import resolve_speakers, run_bible
            llm = LLM(provider, a.model, use_cache=not a.no_cache)
            print(f"LLM: {llm.name}")
            try:
                id_flags, id_raw = resolve_speakers(llm, episodes)
            except LLMError as e:
                print(f"  speaker identity failed: {e}", file=sys.stderr)
                id_flags, id_raw = [], {"error": str(e)}
            print(f"  speaker identity: {len(id_flags)} groups")
            story_flags, bible, raw = run_bible(llm, episodes, include_camera=not a.no_camera,
                                                max_facts=a.max_facts)
            llm_flags = id_flags + story_flags
            for i, f in enumerate(llm_flags, 1):
                f.id = f"L{i:03d}"
            flags += llm_flags
            out.mkdir(parents=True, exist_ok=True)
            (out / "story_bible.json").write_text(json.dumps(list(bible.facts.values()), indent=1, ensure_ascii=False), encoding="utf-8")
            with open(out / "llm_raw.jsonl", "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"task": "speaker_identity", "response": id_raw}, ensure_ascii=False) + "\n")
                for r in raw:
                    fh.write(json.dumps({"task": "story_bible", **r}, ensure_ascii=False) + "\n")
            meta.update(model=llm.name, llm_calls=llm.calls, cache_hits=llm.cache_hits,
                        tokens_in=llm.tokens_in, tokens_out=llm.tokens_out)
        except LLMError as e:
            print(f"LLM layer skipped: {e}", file=sys.stderr)
            meta["llm_error"] = str(e)

    meta["seconds"] = round(time.time() - t0, 1)
    write_all(flags, out, meta)
    print(f"Wrote {len(flags)} flags to {out.resolve()} (report.html, report.md, flags.json) in {meta['seconds']}s")
    return flags, meta


if __name__ == "__main__":
    main()
