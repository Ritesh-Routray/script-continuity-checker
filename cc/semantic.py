"""LLM-powered checks. The model proposes; code verifies every quote before a flag is trusted."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from .llm import LLM, LLMError
from .parser import Episode, fmt_ts
from .rules import Flag, speaker_stats, _ranges

# ---------------------------------------------------------------- verification helpers
def _norm_words(s: str) -> list[str]:
    s = s.lower().replace("’", "'")
    return re.findall(r"[a-z0-9']+", s)


PREFIX_RE = re.compile(r"^\s*(?:\(\d\d:\d\d(?:\.\d+)?\)\s*)?(?:[A-Z][A-Z .'’()-]{1,40}:\s+)?")


def quote_in(quote: str, episode: Episode) -> tuple[bool, float | None]:
    """Is the model's quote really in this episode? Returns (found, timestamp).
    Accepts an optional '(MM:SS) SPEAKER:' prefix and '...' gaps; every piece of 3+ words must match."""
    quote = PREFIX_RE.sub("", (quote or "").strip().strip('"“”'))
    pieces = [p for p in re.split(r"\.\.\.|…", quote) if len(_norm_words(p)) >= 3]
    if not pieces:
        return False, None
    first_ts = None
    for n, p in enumerate(pieces):
        ok, ts = _piece_in(p, episode)
        if not ok:
            return False, None
        if n == 0:
            first_ts = ts
    return True, first_ts


def _piece_in(quote: str, episode: Episode) -> tuple[bool, float | None]:
    q = _norm_words(quote)
    qs = " ".join(q)
    best, best_ts = 0.0, None
    for e in episode.events:
        t = " ".join(_norm_words(e.text))
        if qs in t:
            return True, e.start
        if len(q) >= 4:
            # tolerate small paraphrase/ellipsis: fuzzy match against same-length windows
            tw = t.split()
            for i in range(0, max(1, len(tw) - len(q) + 1)):
                r = SequenceMatcher(None, qs, " ".join(tw[i:i + len(q)])).ratio()
                if r > best:
                    best, best_ts = r, e.start
    for text in (episode.logline,):
        if qs in " ".join(_norm_words(text)):
            return True, None
    return best >= 0.85, best_ts if best >= 0.85 else None


# ---------------------------------------------------------------- 1. speaker identity
ALIAS_SYSTEM = """You are a script supervisor on an AI-generated vertical drama series.
Speaker tags in the scripts are used downstream to assign each character ONE voice and ONE face.
Your job: decide which speaker tags refer to the SAME character.
Rules:
- Only merge tags when the sample lines / loglines give concrete evidence (a name used in address,
  the logline naming them, an obvious description like "Lady in Gold Gown" being called "sister" by Juliana's
  sister Clarissa). Cite that evidence.
- Generic one-off extras (a guard, a townsman) should NOT be merged with anyone unless evidence is clear.
- Never invent tags. Use tags exactly as given.
Return JSON: {"characters": [{"canonical": "Name", "tags": ["TAG1","TAG2"], "confidence": "high|medium|low",
"evidence": "one sentence quoting the line or logline that proves it"}]}
Include only groups with 2+ tags."""


def resolve_speakers(llm: LLM, episodes: list[Episode]) -> tuple[list[Flag], dict]:
    stats = speaker_stats(episodes)
    by_ep = {e.number: e for e in episodes}
    blocks = []
    for tag, s in sorted(stats.items(), key=lambda kv: -kv[1]["lines"]):
        first_ep = min(s["eps"])
        n_samp = 2 if s["lines"] >= 3 else 1
        samples = " | ".join(f'"{x.text[:80]}"' for x in s["samples"][:n_samp])
        blocks.append(f"- {tag}: {s['lines']} lines, eps {_ranges(sorted(s['eps']))}. "
                      f"Ep{first_ep} logline: {by_ep[first_ep].logline[:150]} Says: {samples}")
    user = "SPEAKER TAGS:\n" + "\n".join(blocks)
    data = llm.json(ALIAS_SYSTEM, user, max_tokens=2000)
    flags = []
    for g in data.get("characters", []):
        tags = [t for t in g.get("tags", []) if t in stats]
        bogus = [t for t in g.get("tags", []) if t not in stats]
        if len(tags) < 2:
            continue
        flags.append(Flag(
            "speaker_identity", "high" if g.get("confidence") == "high" else "medium",
            f"{g.get('canonical', '?')} speaks under {len(tags)} tags: {' / '.join(tags)}",
            f"{g.get('evidence', '')} Lines: " + "; ".join(
                f"{t} {stats[t]['lines']} (eps {_ranges(sorted(stats[t]['eps']))})" for t in tags)
            + (f". Model also named non-existent tags {bogus} (dropped)." if bogus else ""),
            [{"ep": stats[t]["first"].ep, "ts": stats[t]["first"].ts,
              "text": f"{t}: {stats[t]['first'].text[:160]}"} for t in tags],
            source=f"llm:{llm.name}", verified=not bogus))
    return flags, data


# ---------------------------------------------------------------- 2. rolling story bible
BIBLE_SYSTEM = """You are the continuity supervisor for a serialized AI-generated vertical drama.
You get (a) the STORY BIBLE: facts established in earlier episodes, each with an id, and (b) the NEW EPISODE script with timestamps.

Do two things:
1. "contradictions": places where the NEW EPISODE conflicts with the bible. Examples: a character's
   name/title/relationship changes; someone who is dead, imprisoned, or far away appears free and present
   without explanation; a wound, scar, object or costume that should persist disappears or changes; an item
   that was destroyed/lost is back; a character knows something they never learned; timeline impossibilities.
   NOT contradictions: lies or deception by characters, deliberate twists/reveals that the script itself
   presents as surprising, normal story progress (someone gets released, heals over time), or anything you are unsure about.
   Each needs a VERBATIM quote (copied exactly, 4-25 words) from the NEW EPISODE and the bible fact id it conflicts with.
2. "new_facts": up to 12 durable facts this episode establishes that later episodes must respect
   (names, titles, family relationships, who is married/dead/imprisoned/where, injuries, key objects and who holds them,
   secrets and who knows them). Each with a VERBATIM quote (4-25 words) from the new episode.
   If a new fact replaces an older one legitimately (story progress), set "supersedes" to the old id.

Return JSON only:
{"contradictions":[{"severity":"high|medium|low","type":"name|relationship|status|location|injury|object|knowledge|timeline|other",
  "summary":"one sentence","quote":"...","bible_id":"F12","could_be_intentional":true|false}],
 "new_facts":[{"entity":"...","attribute":"...","value":"...","quote":"...","supersedes":null}]}
If there are no contradictions return an empty list. Precision matters more than recall."""


class Bible:
    def __init__(self, max_facts_in_prompt: int = 140):
        self.facts: dict[str, dict] = {}
        self.n = 0
        self.max = max_facts_in_prompt

    def add(self, ep: int, f: dict, ts: str) -> str:
        self.n += 1
        fid = f"F{self.n}"
        sup = f.get("supersedes")
        if sup in self.facts:
            self.facts[sup]["superseded_by"] = fid
        self.facts[fid] = {"id": fid, "ep": ep, "ts": ts, "entity": f.get("entity", ""),
                           "attribute": f.get("attribute", ""), "value": f.get("value", ""),
                           "quote": f.get("quote", "")}
        return fid

    def active(self) -> list[dict]:
        return [f for f in self.facts.values() if "superseded_by" not in f]

    def render(self) -> str:
        act = self.active()[-self.max:]
        if not act:
            return "(empty: this is the first episode)"
        return "\n".join(f"{f['id']} [ep{f['ep']}] {f['entity']} | {f['attribute']} | {f['value']}" for f in act)


def run_bible(llm: LLM, episodes: list[Episode], log=print, include_camera: bool = True,
              max_facts: int = 140):
    bible = Bible(max_facts)
    flags: list[Flag] = []
    raw_log = []
    for ep in episodes:
        user = f"STORY BIBLE:\n{bible.render()}\n\nNEW EPISODE:\n{ep.compact_text(include_camera=include_camera)}"
        try:
            data = llm.json(BIBLE_SYSTEM, user)
        except LLMError as e:
            log(f"  ep {ep.number}: LLM error {e}")
            raw_log.append({"ep": ep.number, "error": str(e)})
            continue
        raw_log.append({"ep": ep.number, "response": data})
        n_c = 0
        for c in data.get("contradictions", []) or []:
            ok, sec = quote_in(c.get("quote", ""), ep)
            ref = bible.facts.get(c.get("bible_id", ""))
            verified = ok and ref is not None
            sev = c.get("severity", "medium")
            if c.get("could_be_intentional"):
                sev = "low"
            ev = [{"ep": ep.number, "ts": fmt_ts(sec) if sec is not None else "", "text": c.get("quote", "")}]
            if ref:
                ev.append({"ep": ref["ep"], "ts": ref["ts"], "text": f"[bible {ref['id']}] {ref['entity']}: "
                                                                      f"{ref['value']} — \"{ref['quote']}\""})
            why = []
            if not ok:
                why.append("quote not found in script")
            if ref is None:
                why.append(f"bible id {c.get('bible_id')} does not exist")
            flags.append(Flag(
                f"story_{c.get('type', 'other')}", sev,
                f"Ep {ep.number}: {c.get('summary', '')[:140]}",
                c.get("summary", "") + (" (model marked this as possibly intentional)" if c.get("could_be_intentional") else "")
                + (f" [UNVERIFIED: {', '.join(why)}]" if why else ""),
                ev, source=f"llm:{llm.name}", verified=verified))
            n_c += 1
        n_fact_ok = 0
        for f in data.get("new_facts", []) or []:
            ok, sec = quote_in(f.get("quote", ""), ep)
            if ok:          # unverifiable facts never enter the bible -> no hallucinated "canon"
                bible.add(ep.number, f, fmt_ts(sec) if sec is not None else "")
                n_fact_ok += 1
        log(f"  ep {ep.number:>3}: +{n_fact_ok} facts, {n_c} contradiction(s)  [bible {len(bible.active())}]")
    return flags, bible, raw_log
