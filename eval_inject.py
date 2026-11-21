"""Quality check: plant known continuity errors in a copy of the show, run the checker, measure recall.

  python eval_inject.py <episodes folder> [--seeds 3] [--provider none|gemini|groq] [--episodes 1-25]

Rule-level errors are planted at random (seeded). Story-level errors are hand-written lines that
contradict established canon of THIS show; they only make sense for the input pack and are used
only when --provider is not 'none'.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tempfile
from pathlib import Path

from cc.parser import load_show, fmt_ts
from run import main as run_main, parse_range

NAMES = ["Juliana", "Alexander", "Catherine", "Rupert", "Gerald", "Clarissa", "Prudence"]
TAGS = {"ALEXANDER": "DUKE ALEXANDER", "CATHERINE": "LADY CATHERINE", "JULIANA": "LADY JULIANA",
        "RUPERT": "LORD RUPERT", "PRUDENCE": "MISS PRUDENCE"}

# (episode, speaker tag, line, why it contradicts canon) — inserted as a new dialogue block
STORY_PLANTS = [
    (12, "JULIANA", "My mother is waiting for me at home; she will know what to do.",
     "Juliana's mother is dead (late mother's gown/ledger since ep 1)"),
    (17, "ALEXANDER", "I have never held a sword in my life, Juliana. I am only a woodcutter.",
     "Alexander is a scarred war veteran who fought off thugs with combat training (ep 3, 16)"),
    (24, "JULIANA", "My sister Clarissa is the kindest person I know. She would never hurt me.",
     "Clarissa stole her fiance and tormented her (eps 1-13)"),
    (36, "ALEXANDER", "Mother, it has been years. You must have heard that I died in a hunting accident.",
     "Alexander was presumed dead in war, not a hunting accident (soldier backstory)"),
    (44, "CATHERINE", "Rupert, my dear brother, you must not let them take you.",
     "Rupert is Catherine's fiance/husband, not her brother (eps 20-42)"),
]


def _typo(name: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(name) - 1)
    ops = [lambda s: s[:i] + s[i] + s[i:],                       # double a letter
           lambda s: s[:i] + s[i + 1] + s[i] + s[i + 2:] if i + 2 <= len(s) - 1 else s[:i] + "e" + s[i + 1:],
           lambda s: s[:i] + ("e" if s[i] != "e" else "a") + s[i + 1:]]
    return rng.choice(ops)(name)


def plant(src: Path, dst: Path, rng: random.Random, n_each: int, story: bool) -> list[dict]:
    shutil.copytree(src, dst)
    files = {int(re.search(r"episode-(\d+)", p.name).group(1)): p for p in dst.glob("*.txt")}
    planted = []

    def edit(ep, fn):
        p = files[ep]
        s = p.read_text(encoding="utf-8")
        new = fn(s)
        if new and new != s:
            p.write_text(new, encoding="utf-8")
            return True
        return False

    eps = sorted(files)
    for _ in range(n_each):
        # 1. spelling drift: misspell one occurrence of a main name inside dialogue
        for _try in range(20):
            ep, name = rng.choice(eps), rng.choice(NAMES)
            s = files[ep].read_text(encoding="utf-8")
            hits = [m for m in re.finditer(rf"\b{name}\b", s) if s.rfind("\n", 0, m.start()) < m.start()
                    and not s[s.rfind("\n", 0, m.start()) + 1:].startswith("**")]
            if hits:
                m = rng.choice(hits)
                bad = _typo(name, rng)
                files[ep].write_text(s[:m.start()] + bad + s[m.end():], encoding="utf-8")
                planted.append({"type": "name_variant", "ep": ep, "key": bad, "note": f"{name}->{bad}"})
                break
        # 2. speaker tag split
        for _try in range(20):
            ep = rng.choice(eps)
            s = files[ep].read_text(encoding="utf-8")
            cands = [t for t in TAGS if re.search(rf"^\*\*{t} \(", s, re.M)]
            if cands:
                t = rng.choice(cands)
                ms = list(re.finditer(rf"^\*\*{t} \(", s, re.M))
                m = rng.choice(ms)
                files[ep].write_text(s[:m.start()] + f"**{TAGS[t]} (" + s[m.end():], encoding="utf-8")
                planted.append({"type": "speaker_tag_variant", "ep": ep, "key": TAGS[t], "note": f"{t}->{TAGS[t]}"})
                break
        # 3. reversed timestamp on one block
        ep = rng.choice(eps)
        s = files[ep].read_text(encoding="utf-8")
        ms = list(re.finditer(r"^Start: \((\d\d:\d\d\.\d\d)\) \| End: \((\d\d:\d\d\.\d\d)\)$", s, re.M))
        m = rng.choice([x for x in ms if x.group(1) != x.group(2)])
        files[ep].write_text(s[:m.start()] + f"Start: ({m.group(2)}) | End: ({m.group(1)})" + s[m.end():], encoding="utf-8")
        planted.append({"type": "timing", "ep": ep, "key": m.group(2), "note": "start/end swapped"})
        # 4. overstuffed line: add a sentence to a short dialogue line
        ep = rng.choice(eps)
        s = files[ep].read_text(encoding="utf-8")
        ms = [x for x in re.finditer(r'^(\*\*[A-Z][^*\n]*\(.*\):\*\*\nStart: \((\d\d:\d\d\.\d\d)\).*\n)"(.+)"$', s, re.M)]
        m = rng.choice(ms)
        added = " And I will not stand here and let you speak to me that way ever again, do you understand me?"
        files[ep].write_text(s[:m.end() - 1] + added + '"' + s[m.end():], encoding="utf-8")
        planted.append({"type": "speech_rate", "ep": ep, "key": m.group(2), "note": "line stuffed"})
    # 5. continuity note points at wrong episode
    ep = rng.choice([e for e in eps if e > 3])
    if edit(ep, lambda s: re.sub(r"(CONTINUITY NOTES: )", rf"\1Directly follows Episode {ep - 3}. ", s, count=1)):
        planted.append({"type": "metadata", "ep": ep, "key": f"episode {ep - 3}", "note": "wrong prev-episode ref"})

    if story:
        for ep, tag, line, why in STORY_PLANTS:
            if ep not in files:
                continue
            s = files[ep].read_text(encoding="utf-8")
            # insert after the first dialogue block in the last scene, reusing its timing slot
            blocks = list(re.finditer(r'^\*\*[A-Z][^*\n]*\(.*\):\*\*\nStart: \((\d\d:\d\d\.\d\d)\) \| End: \((\d\d:\d\d\.\d\d)\)\n.*\n', s, re.M))
            b = blocks[len(blocks) // 2]
            ins = f'\n**{tag} (Quiet):**\nStart: ({b.group(2)}) | End: ({b.group(2)})\n"{line}"\n'
            files[ep].write_text(s[:b.end()] + ins + s[b.end():], encoding="utf-8")
            planted.append({"type": "story", "ep": ep, "key": line, "note": why})
    return planted


def caught(p: dict, flags: list[dict]) -> list[str]:
    hits = []
    for f in flags:
        blob = json.dumps(f, ensure_ascii=False)
        eps = {e["ep"] for e in f["evidence"]}
        t = p["type"]
        if t == "name_variant" and f["check"] == "name_variant" and re.search(rf"'{re.escape(p['key'])}'", f["detail"]):
            hits.append(f["id"])
        elif t == "speaker_tag_variant" and f["check"] in ("speaker_tag_variant", "speaker_identity") and p["key"] in f["title"]:
            hits.append(f["id"])
        elif t == "timing" and f["check"] == "timing" and p["ep"] in eps and p["key"] in blob:
            hits.append(f["id"])
        elif t == "speech_rate" and f["check"] == "speech_rate" and p["ep"] in eps and "stand here and let you" in blob:
            hits.append(f["id"])
        elif t == "metadata" and f["check"] == "metadata" and p["ep"] in eps:
            hits.append(f["id"])
        elif t == "story" and f["check"].startswith("story_") and p["ep"] in eps:
            q = set(re.findall(r"[a-z]+", p["key"].lower()))
            ev = set(re.findall(r"[a-z]+", " ".join(e["text"] for e in f["evidence"] if e["ep"] == p["ep"]).lower()))
            if len(q & ev) >= 0.5 * len(q):
                hits.append(f["id"] + ("" if f["verified"] else "(unverified)"))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=3, help="plants per rule-type per seed")
    ap.add_argument("--provider", default="none")
    ap.add_argument("--model", default=None)
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--out", default="outputs/eval")
    a = ap.parse_args()
    src = Path(a.folder)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    baseline_flags, _ = run_main([str(src), "--provider", "none", "--out", str(out / "baseline")])
    base_titles = {f.title for f in baseline_flags}
    results = []
    keep = parse_range(a.episodes)
    for seed in range(a.seeds):
        rng = random.Random(seed)
        tmp = Path(tempfile.mkdtemp(prefix="cc_eval_")) / src.name
        planted = plant(src, tmp, rng, a.n, story=a.provider != "none")
        if keep:
            planted = [p for p in planted if p["ep"] in keep]
        args = [str(tmp), "--provider", a.provider, "--out", str(out / f"seed{seed}")]
        if a.model:
            args += ["--model", a.model]
        if a.episodes:
            args += ["--episodes", a.episodes]
        flags, meta = run_main(args)
        fl = [f.to_dict() for f in flags]
        for p in planted:
            p["seed"] = seed
            p["caught_by"] = caught(p, fl)
            results.append(p)
        new_flags = [f for f in fl if f["title"] not in base_titles]
        unmatched = [f for f in new_flags if not any(f["id"] in h for p in planted for h in p["caught_by"])]
        results.append({"seed": seed, "type": "_summary", "new_flags": len(new_flags),
                        "new_flags_not_explained_by_plants": [f["title"] for f in unmatched]})
        shutil.rmtree(tmp.parent, ignore_errors=True)

    rows = {}
    for r in results:
        if r["type"] == "_summary":
            continue
        k = r["type"]
        rows.setdefault(k, [0, 0])
        rows[k][1] += 1
        rows[k][0] += bool(r["caught_by"])
    lines = ["# Planted-error recall", "", f"provider: {a.provider} | seeds: {a.seeds}", "",
             "| Error type | Caught / planted | Recall |", "|---|---|---|"]
    tot_c = tot_n = 0
    for k, (c, n) in sorted(rows.items()):
        lines.append(f"| {k} | {c}/{n} | {c / n:.0%} |")
        tot_c += c
        tot_n += n
    lines.append(f"| **all** | {tot_c}/{tot_n} | {tot_c / max(tot_n, 1):.0%} |")
    lines += ["", "## Misses", ""] + [f"- seed {r['seed']} ep {r['ep']} {r['type']}: {r['note']} ({r['key'][:60]})"
                                       for r in results if r["type"] != "_summary" and not r["caught_by"]]
    lines += ["", "## Extra flags on planted copies not explained by a plant (side-effects / false alarms)", ""]
    for r in results:
        if r["type"] == "_summary":
            lines.append(f"- seed {r['seed']}: {len(r['new_flags_not_explained_by_plants'])} -> "
                         + "; ".join(r["new_flags_not_explained_by_plants"][:8]))
    (out / f"recall_{a.provider}.md").write_text("\n".join(lines), encoding="utf-8")
    (out / f"plants_{a.provider}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
