"""Deterministic continuity checks. No model, no API key, fully reproducible."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .parser import Episode, fmt_ts

HONORIFICS = {"LADY", "LORD", "THE", "SIR", "DUKE", "DUCHESS", "DOWAGER", "MR", "MRS",
              "MISS", "MISTER", "MADAM", "YOUNG", "OLD", "KING", "QUEEN", "PRINCE",
              "PRINCESS", "COUNT", "COUNTESS", "BARON", "EARL"}
# words that make a speaker tag a description rather than a name
DESCRIPTOR_WORDS = {"MAN", "WOMAN", "STRANGER", "GROOM", "BRIDE", "NOBLEMAN", "NOBLEWOMAN",
                    "MATRIARCH", "GUARD", "MAID", "THUG", "OFFICIAL", "INFORMANT", "VOICE",
                    "VILLAGER", "TOWNSMAN", "SCOUT", "FIGHTER", "MOTHER", "FATHER", "PRIEST",
                    "JUDGE", "CONSTABLE", "HOUSEKEEPER", "GOWN", "BEARDED", "BLOND", "MYSTERIOUS",
                    "ELDER", "ARISTOCRATIC", "PEASANT", "ROYAL", "HIGH", "LORD", "LADY", "BOY",
                    "GIRL", "CHILD", "SERVANT", "SOLDIER", "OFFICER", "CROWD", "FORMER", "SPY"}

SPEECH_WPS_MAX = 4.5     # words/second; natural fast speech is ~3, dubbing breaks above ~4
TOL = 0.5                # seconds of tolerance for timing checks


@dataclass
class Flag:
    check: str
    severity: str                  # high | medium | low
    title: str
    detail: str
    evidence: list[dict] = field(default_factory=list)   # {ep, ts, text}
    source: str = "rule"
    verified: bool = True          # rule flags are verified by construction
    id: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _ev(ep: int, sec: float | None, text: str) -> dict:
    return {"ep": ep, "ts": fmt_ts(sec) if sec is not None else "", "text": text[:220]}


# ---------------------------------------------------------------- name variants
WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:[-'][A-Z]?[a-z]+)?\b")


def _texts(episodes: list[Episode]):
    for ep in episodes:
        yield ep.number, None, ep.logline
        yield ep.number, None, ep.continuity_notes
        for sc in ep.scenes:
            yield ep.number, sc.start, sc.heading.title()
        for e in ep.events:
            yield ep.number, e.start, e.text   # camera text too: it feeds image/video prompts


def _strip_possessive(w: str) -> str:
    return re.sub(r"(?:'s|’s|s')$", "", w)


def proper_noun_index(episodes: list[Episode], strict: bool = True):
    """Capitalised words that never appear lower-cased anywhere -> likely proper nouns."""
    lower_vocab: set[str] = set()
    mid_sentence: set[str] = set()   # capitalised somewhere other than a sentence start
    occ: dict[str, list[tuple[int, float | None, str]]] = defaultdict(list)
    for ep, sec, text in _texts(episodes):
        lower_vocab.update(re.findall(r"\b[a-z]{3,}\b", text))
        for m in WORD_RE.finditer(text):
            w = _strip_possessive(m.group(0))
            occ[w].append((ep, sec, text))
            if not re.search(r"(^|[.!?…:;—–-])[\s\"“'‘(]*$", text[:m.start()]):
                mid_sentence.add(w)
    return {w: v for w, v in occ.items()
            if w.lower() not in lower_vocab and (w in mid_sentence or not strict) and len(w) >= 4}


def _dl_distance(a: str, b: str) -> int:
    """Damerau-Levenshtein (optimal string alignment) distance."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _similar(a: str, b: str) -> bool:
    if a == b or a.lower() == b.lower():
        return False
    if a.rstrip("s") == b.rstrip("s"):
        return False
    la, lb = a.lower(), b.lower()
    if la[0] != lb[0] and abs(len(la) - len(lb)) > 1:
        return False
    common = len(next((la[:k] for k in range(min(len(la), len(lb)), 0, -1) if la[:k] == lb[:k]), ""))
    ratio = SequenceMatcher(None, la, lb).ratio()
    dist = _dl_distance(la, lb)
    short = min(len(la), len(lb))
    # typo-level edits (Rupert/Rupart, Gerald/Gearld) or same stem (Ashbury/Ashbourne/Ashburn)
    return (dist == 1 and short >= 5) or (dist == 2 and short >= 7) or (common >= 4 and ratio >= 0.7)


def check_name_variants(episodes: list[Episode]) -> list[Flag]:
    idx = proper_noun_index(episodes, strict=False)
    strong = set(proper_noun_index(episodes))
    words = sorted(idx)
    parent = {w: w for w in words}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def weak_ok(w, other):
        # a word only ever seen at a sentence start must be a 1-edit typo and not look like English
        if w in strong:
            return True
        return not re.search(r"(ing|ed|ly|ness|ful)$", w) and _dl_distance(w.lower(), other.lower()) == 1

    for i, a in enumerate(words):
        for b in words[i + 1:]:
            if _similar(a, b) and weak_ok(a, b) and weak_ok(b, a):
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for w in words:
        groups[find(w)].append(w)

    flags = []
    for members in groups.values():
        if len(members) < 2 or not strong.intersection(members):
            continue
        members.sort(key=lambda w: -len(idx[w]))
        major = members[0]
        parts, evidence = [], []
        for w in members:
            first = idx[w][0]
            eps = sorted({o[0] for o in idx[w]})
            parts.append(f"'{w}' x{len(idx[w])} (eps {_ranges(eps)})")
            evidence.append(_ev(first[0], first[1], first[2]))
        minority = sum(len(idx[w]) for w in members[1:])
        sev = "high" if minority <= max(3, 0.25 * len(idx[major])) or len(members) >= 3 else "medium"
        flags.append(Flag(
            "name_variant", sev,
            f"Spelling drift: {' / '.join(members)}",
            "Same proper noun appears to be spelled differently: " + "; ".join(parts)
            + ". If these are one entity, pick one spelling; if they are different, rename one so "
              "viewers (and the image/voice pipeline) don't confuse them.",
            evidence))
    return flags


def _ranges(nums: list[int]) -> str:
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)


# ---------------------------------------------------------------- speaker tags
def speaker_stats(episodes: list[Episode]):
    stats: dict[str, dict] = {}
    for ep in episodes:
        for e in ep.dialogue:
            s = stats.setdefault(e.speaker, {"lines": 0, "eps": set(), "first": e, "samples": []})
            s["lines"] += 1
            s["eps"].add(ep.number)
            if len(s["samples"]) < 3:
                s["samples"].append(e)
    return stats


FEMALE = {"LADY", "DUCHESS", "MRS", "MISS", "MADAM", "QUEEN", "PRINCESS", "COUNTESS"}
MALE = {"LORD", "SIR", "DUKE", "MR", "MISTER", "KING", "PRINCE", "COUNT", "BARON", "EARL"}


def core_name(tag: str) -> str:
    toks = [t for t in re.split(r"\s+", tag.replace("'S", "")) if t]
    gender = ""
    while toks and toks[0] in HONORIFICS:
        gender = gender or ("F" if toks[0] in FEMALE else "M" if toks[0] in MALE else "")
        toks = toks[1:]
    # keep gendered titles apart so LORD X and LADY X (a couple) are not merged
    return (gender + ":" if gender else "") + (" ".join(toks) or tag)


def is_descriptive(tag: str, proper: set[str]) -> bool:
    toks = re.findall(r"[A-Z]+", tag)
    has_name = any(t.title() in proper for t in toks if t not in HONORIFICS)
    return not has_name and any(t in DESCRIPTOR_WORDS for t in toks)


def check_speaker_tags(episodes: list[Episode]) -> list[Flag]:
    stats = speaker_stats(episodes)
    proper = set(proper_noun_index(episodes))
    flags = []

    by_core = defaultdict(list)
    for tag in stats:
        by_core[core_name(tag).split(":")[-1]].append(tag)
    # split groups where both a male- and a female-titled tag exist (e.g. LORD/LADY HARRINGTON)
    for core in list(by_core):
        genders = {core_name(t).split(":")[0] for t in by_core[core] if ":" in core_name(t)}
        if {"F", "M"} <= genders:
            del by_core[core]
    for core, tags in by_core.items():
        if len(tags) < 2:
            continue
        tags.sort(key=lambda t: -stats[t]["lines"])
        flags.append(Flag(
            "speaker_tag_variant", "medium",
            f"One character, {len(tags)} speaker tags: {' / '.join(tags)}",
            "; ".join(f"{t}: {stats[t]['lines']} lines in eps {_ranges(sorted(stats[t]['eps']))}"
                      for t in tags)
            + ". Voice/face assignment keyed on the tag will treat these as different people.",
            [_ev(stats[t]["first"].ep, stats[t]["first"].start, f"{t}: {stats[t]['first'].text}")
             for t in tags]))

    for tag, s in sorted(stats.items(), key=lambda kv: -kv[1]["lines"]):
        if not is_descriptive(tag, proper):
            continue
        n_eps = len(s["eps"])
        if s["lines"] < 8 and n_eps < 3:
            continue   # one-off extras ("TOWNSMAN") are fine
        flags.append(Flag(
            "recurring_unnamed_speaker", "medium" if n_eps >= 3 else "low",
            f"Recurring character tagged only by description: {tag}",
            f"{s['lines']} lines across eps {_ranges(sorted(s['eps']))}. A recurring speaker "
            "without a stable name risks being cast with a different voice/face each time, or "
            "being the same person as a named character under another tag.",
            [_ev(e.ep, e.start, f"{tag}: {e.text}") for e in s["samples"]]))
    return flags


# ---------------------------------------------------------------- timing
def check_timing(episodes: list[Episode]) -> list[Flag]:
    flags = []
    for ep in episodes:
        for w in ep.parse_warnings:
            flags.append(Flag("format", "medium", f"Ep {ep.number}: format problem", w,
                              [_ev(ep.number, None, w)]))
        for i, sc in enumerate(ep.scenes):
            if sc.start is None or sc.end is None:
                flags.append(Flag("timing", "medium", f"Ep {ep.number} scene {sc.number}: missing scene times",
                                  "Scene start or end time missing.", [_ev(ep.number, None, sc.heading)]))
                continue
            if sc.end < sc.start:
                flags.append(Flag("timing", "high", f"Ep {ep.number} scene {sc.number}: ends before it starts",
                                  f"{fmt_ts(sc.start)} -> {fmt_ts(sc.end)}", [_ev(ep.number, sc.start, sc.heading)]))
            if i and ep.scenes[i - 1].end is not None:
                gap = sc.start - ep.scenes[i - 1].end
                if abs(gap) > TOL:
                    flags.append(Flag(
                        "timing", "medium",
                        f"Ep {ep.number}: {'gap' if gap > 0 else 'overlap'} of {abs(gap):.1f}s before scene {sc.number}",
                        f"Scene {ep.scenes[i-1].number} ends {fmt_ts(ep.scenes[i-1].end)}, "
                        f"scene {sc.number} starts {fmt_ts(sc.start)}.",
                        [_ev(ep.number, sc.start, sc.heading)]))
        scene_by_no = {s.number: s for s in ep.scenes}
        last_line_by_speaker: dict[str, object] = {}
        for e in ep.events:
            if e.end < e.start:
                flags.append(Flag("timing", "high", f"Ep {ep.number} {fmt_ts(e.start)}: block ends before it starts",
                                  f"{e.raw_tag}: {fmt_ts(e.start)} -> {fmt_ts(e.end)}", [_ev(ep.number, e.start, e.text)]))
            sc = scene_by_no.get(e.scene)
            if sc and sc.start is not None and sc.end is not None and \
                    (e.start < sc.start - TOL or e.end > sc.end + TOL):
                flags.append(Flag("timing", "medium", f"Ep {ep.number} {fmt_ts(e.start)}: block outside its scene",
                                  f"{e.raw_tag} runs {fmt_ts(e.start)}-{fmt_ts(e.end)} but scene {sc.number} is "
                                  f"{fmt_ts(sc.start)}-{fmt_ts(sc.end)}.", [_ev(ep.number, e.start, e.text)]))
            if e.kind != "DIALOGUE":
                continue
            dur = e.end - e.start
            words = len(e.text.split())
            if dur > 0 and words >= 4 and words / dur > SPEECH_WPS_MAX:
                flags.append(Flag(
                    "speech_rate", "low" if words / dur < 5.5 else "medium",
                    f"Ep {ep.number} {fmt_ts(e.start)}: line too long for its slot ({words / dur:.1f} words/s)",
                    f"{e.speaker} has {words} words in {dur:.1f}s; natural delivery is ~2.5-3 words/s. "
                    f"Needs ~{words / 2.8:.1f}s or a shorter line (and dubbed languages run longer).",
                    [_ev(ep.number, e.start, f"{e.speaker}: {e.text}")]))
            prev = last_line_by_speaker.get(e.speaker)
            if prev is not None and e.start < prev.end - TOL:
                flags.append(Flag(
                    "timing", "medium", f"Ep {ep.number} {fmt_ts(e.start)}: {e.speaker} talks over themself",
                    f"Line starts at {fmt_ts(e.start)} before their previous line ends at {fmt_ts(prev.end)}.",
                    [_ev(ep.number, prev.start, prev.text), _ev(ep.number, e.start, e.text)]))
            last_line_by_speaker[e.speaker] = e
    return flags


# ---------------------------------------------------------------- metadata
PREV_RE = re.compile(r"(?:directly follows|immediately after|(?:picks up|follows|continues)[a-z ]*?from)\s+(?:the\s+)?(?:events of\s+)?Episode\s+(\d+)", re.I)


def check_metadata(episodes: list[Episode]) -> list[Flag]:
    flags = []
    proper = proper_noun_index(episodes)
    nums = [e.number for e in episodes]
    for a, b in zip(nums, nums[1:]):
        if b != a + 1:
            flags.append(Flag("metadata", "medium", f"Episode numbering jumps {a} -> {b}",
                              "Missing or misnumbered episode file.", [_ev(b, None, "")]))
    for ep in episodes:
        m = PREV_RE.search(ep.continuity_notes)
        if ep.number > 1 and m and int(m.group(1)) != ep.number - 1:
            flags.append(Flag("metadata", "medium",
                              f"Ep {ep.number}: continuity notes say it follows episode {m.group(1)}",
                              f"Notes say: \"{ep.continuity_notes[:200]}\" but the previous episode is {ep.number - 1}. "
                              "If notes feed the next episode's generation prompt, the model gets the wrong context.",
                              [_ev(ep.number, None, ep.continuity_notes)]))
        body = " ".join(e.text for e in ep.events) + " " + " ".join(s.heading for s in ep.scenes)
        speakers = " ".join(e.speaker or "" for e in ep.dialogue)
        for w in sorted({_strip_possessive(x) for x in WORD_RE.findall(ep.logline)}):
            if w in proper and w not in body and w.upper() not in speakers:
                flags.append(Flag("logline_mismatch", "low",
                                  f"Ep {ep.number}: logline mentions '{w}', script never does",
                                  "The logline (used for recaps, metadata and the next episode's prompt) "
                                  f"names '{w}' but the name never appears in this episode's script.",
                                  [_ev(ep.number, None, ep.logline)]))
    return flags


def run_rules(episodes: list[Episode]) -> list[Flag]:
    flags = (check_name_variants(episodes) + check_speaker_tags(episodes)
             + check_metadata(episodes) + check_timing(episodes))
    for i, f in enumerate(flags, 1):
        f.id = f"R{i:03d}"
    return flags
