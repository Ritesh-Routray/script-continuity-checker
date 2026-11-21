"""Parse Kuku-style timestamped microdrama screenplays into structured events."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{1,2}):(\d{2}(?:\.\d+)?)"
HEADER_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
SPAN_RE = re.compile(rf"^Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)\s*$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX", "MUSIC", "SFX", "TEXT", "SUPER", "TITLE"}
TAG_MARKERS = re.compile(r"\((?:O\.S\.|V\.O\.|O\.C\.|CONT'D)\)", re.I)


def ts_to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt_ts(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


@dataclass
class Event:
    ep: int
    scene: int
    kind: str              # DIALOGUE | CAMERA | ACTION | SOUND FX | ...
    speaker: str | None    # normalised speaker tag for dialogue
    raw_tag: str           # full header text, e.g. "MATRIARCH (O.S.) (Firm)"
    start: float
    end: float
    text: str
    line_no: int

    @property
    def ts(self) -> str:
        return fmt_ts(self.start)


@dataclass
class Scene:
    number: int
    heading: str
    start: float | None
    end: float | None
    line_no: int


@dataclass
class Episode:
    number: int
    file: str
    logline: str = ""
    continuity_notes: str = ""
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def dialogue(self) -> list[Event]:
        return [e for e in self.events if e.kind == "DIALOGUE"]

    @property
    def duration(self) -> float:
        ends = [s.end for s in self.scenes if s.end is not None]
        return max(ends) if ends else 0.0

    def compact_text(self, include_camera: bool = False) -> str:
        """Token-cheap rendering for LLM prompts: one line per event with timestamp."""
        out = [f"EPISODE {self.number}", f"LOGLINE: {self.logline}"]
        cur = None
        for e in self.events:
            if e.scene != cur:
                cur = e.scene
                sc = next((s for s in self.scenes if s.number == cur), None)
                out.append(f"\n[SCENE {cur}: {sc.heading if sc else ''}]")
            if e.kind == "CAMERA" and not include_camera:
                continue
            who = e.speaker if e.kind == "DIALOGUE" else e.kind
            out.append(f"({e.ts}) {who}: {e.text}")
        return "\n".join(out)


def speaker_from_tag(tag: str) -> str:
    tag = TAG_MARKERS.sub("", tag)
    tag = re.sub(r"\(.*?\)", "", tag)          # emotion / delivery notes
    return re.sub(r"\s+", " ", tag).strip().upper()


def parse_episode(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    m = re.search(r"episode-(\d+)", path.name, re.I)
    ep = Episode(number=int(m.group(1)) if m else -1, file=path.name)
    scene_no = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()
        if low.startswith("episode number:"):
            n = re.search(r"(\d+)", line)
            if n and int(n.group(1)) != ep.number:
                ep.parse_warnings.append(
                    f"Header says episode {n.group(1)} but filename says {ep.number}")
        elif low.startswith("episode logline:"):
            ep.logline = line.split(":", 1)[1].strip()
        elif low.startswith("continuity notes:"):
            ep.continuity_notes = line.split(":", 1)[1].strip()
        elif SCENE_RE.match(line):
            scene_no = int(SCENE_RE.match(line).group(1))
            ep.scenes.append(Scene(scene_no, "", None, None, i + 1))
        elif low.startswith("scene heading:") and ep.scenes:
            ep.scenes[-1].heading = line.split(":", 1)[1].strip()
        elif low.startswith("scene start:") and ep.scenes:
            t = re.search(TS, line)
            ep.scenes[-1].start = ts_to_sec(*t.groups()) if t else None
        elif low.startswith("scene end:") and ep.scenes:
            t = re.search(TS, line)
            ep.scenes[-1].end = ts_to_sec(*t.groups()) if t else None
        elif HEADER_RE.match(line):
            tag = HEADER_RE.match(line).group(1).strip()
            span = SPAN_RE.match(lines[i + 1].strip()) if i + 1 < len(lines) else None
            if not span:
                ep.parse_warnings.append(f"line {i+1}: block '{tag}' has no Start/End line")
                i += 1
                continue
            start = ts_to_sec(span.group(1), span.group(2))
            end = ts_to_sec(span.group(3), span.group(4))
            j = i + 2
            body = []
            while j < len(lines) and lines[j].strip() and not HEADER_RE.match(lines[j].strip()):
                body.append(lines[j].strip())
                j += 1
            text = " ".join(body).strip().strip('"').strip("“”")
            base = speaker_from_tag(tag)
            kind = base if base in NON_DIALOGUE else "DIALOGUE"
            ep.events.append(Event(ep.number, scene_no, kind,
                                   base if kind == "DIALOGUE" else None,
                                   tag, start, end, text, i + 1))
            i = j
            continue
        i += 1
    return ep


def load_show(folder: str | Path) -> list[Episode]:
    files = sorted(Path(folder).glob("*.txt"),
                   key=lambda p: int(re.search(r"episode-(\d+)", p.name, re.I).group(1))
                   if re.search(r"episode-(\d+)", p.name, re.I) else 0)
    if not files:
        raise FileNotFoundError(f"No .txt episode files in {folder}")
    return [parse_episode(f) for f in files]


def to_dict(ep: Episode) -> dict:
    return asdict(ep)
