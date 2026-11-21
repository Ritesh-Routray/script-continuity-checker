"""Write flags as JSON (for machines), Markdown (to read) and an HTML triage sheet (for the editor)."""
from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from .rules import Flag

SEV_ORDER = {"high": 0, "medium": 1, "low": 2}
CHECK_LABEL = {
    "name_variant": "Spelling drift",
    "speaker_tag_variant": "Speaker tag split",
    "speaker_identity": "Same character, different tags (LLM)",
    "recurring_unnamed_speaker": "Recurring unnamed speaker",
    "speech_rate": "Line too long for slot",
    "timing": "Timestamp error",
    "metadata": "Episode metadata",
    "logline_mismatch": "Logline vs script",
    "format": "Format",
}


def label(check: str) -> str:
    if check.startswith("story_"):
        return f"Story continuity: {check[6:]} (LLM)"
    return CHECK_LABEL.get(check, check)


def sort_flags(flags: list[Flag]) -> list[Flag]:
    return sorted(flags, key=lambda f: (not f.verified, SEV_ORDER.get(f.severity, 3),
                                        min((e["ep"] for e in f.evidence), default=0)))


def write_all(flags: list[Flag], out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    flags = sort_flags(flags)
    (out / "flags.json").write_text(json.dumps([f.to_dict() for f in flags], indent=1, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "report.md").write_text(markdown(flags, meta), encoding="utf-8")
    (out / "report.html").write_text(html_sheet(flags, meta), encoding="utf-8")


def markdown(flags: list[Flag], meta: dict) -> str:
    c = Counter((f.severity, f.verified) for f in flags)
    lines = [f"# Continuity report: {meta.get('show', '')}", "",
             f"Episodes: {meta.get('episodes')} | model: {meta.get('model', 'rules only')} | "
             f"flags: {len(flags)} (high {c[('high', True)]}, medium {c[('medium', True)]}, "
             f"low {c[('low', True)]}, unverified LLM claims {sum(v for (s, ok), v in c.items() if not ok)})", ""]
    by_check = Counter(label(f.check) for f in flags)
    lines += ["| Check | Flags |", "|---|---|"] + [f"| {k} | {v} |" for k, v in by_check.most_common()] + [""]
    current = None
    for f in flags:
        group = "Unverified LLM claims (quote not found; review with care)" if not f.verified \
            else f"{f.severity.upper()} severity"
        if group != current:
            lines += ["", f"## {group}", ""]
            current = group
        lines.append(f"### {f.id} · {label(f.check)} · {f.title}")
        lines.append(f"{f.detail}")
        for e in f.evidence:
            lines.append(f"- ep {e['ep']} {e['ts']}: “{e['text']}”")
        lines.append("")
    return "\n".join(lines)


def html_sheet(flags: list[Flag], meta: dict) -> str:
    rows = []
    for f in flags:
        ev = "<br>".join(f"<b>ep {e['ep']} {html.escape(e['ts'])}</b> {html.escape(e['text'])}" for e in f.evidence)
        rows.append(
            f"<tr data-sev='{f.severity}' data-ver='{int(f.verified)}' data-check='{html.escape(label(f.check))}'>"
            f"<td>{f.id}</td><td><span class='sev {f.severity}'>{f.severity}</span>"
            f"{'' if f.verified else '<br><span class=unv>unverified</span>'}</td>"
            f"<td>{html.escape(label(f.check))}<br><small>{html.escape(f.source)}</small></td>"
            f"<td><b>{html.escape(f.title)}</b><br>{html.escape(f.detail)}</td><td class=ev>{ev}</td>"
            f"<td><select data-id='{f.id}'><option></option><option>real – fix</option>"
            f"<option>real – ignore</option><option>false alarm</option></select></td></tr>")
    checks = sorted({label(f.check) for f in flags})
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continuity Report</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1c1a;--mut:#6b675f;--line:#e4e0d8;--hi:#b42318;--md:#b54708;--lo:#475467;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#171614;--fg:#ece9e3;--mut:#a29d93;--line:#34312c;--hi:#f97066;--md:#fdb022;--lo:#98a2b3;--card:#1f1e1b}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}} .mut{{color:var(--mut)}} table{{border-collapse:collapse;width:100%;background:var(--card)}}
td,th{{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}} th{{position:sticky;top:0;background:var(--card)}}
.sev{{font-weight:600;text-transform:uppercase;font-size:11px}} .high{{color:var(--hi)}} .medium{{color:var(--md)}} .low{{color:var(--lo)}}
.unv{{font-size:11px;color:var(--md)}} .ev{{font-size:12.5px;color:var(--mut);max-width:420px}} .bar{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}}
select,button{{font:inherit;padding:4px 6px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}}
.wrap{{overflow-x:auto}}</style></head><body>
<h1>Continuity report — {html.escape(str(meta.get('show', '')))}</h1>
<div class=mut>{meta.get('episodes')} episodes · {html.escape(str(meta.get('model', 'rules only')))} · {len(flags)} flags · generated {html.escape(str(meta.get('generated', '')))}</div>
<div class=bar><select id=fsev><option value="">all severities</option><option>high</option><option>medium</option><option>low</option></select>
<select id=fcheck><option value="">all checks</option>{''.join(f'<option>{html.escape(c)}</option>' for c in checks)}</select>
<label><input type=checkbox id=fver> verified only</label><button id=exp>Export decisions (CSV)</button></div>
<div class=wrap><table><thead><tr><th>ID</th><th>Sev</th><th>Check</th><th>Issue</th><th>Evidence (episode, timestamp)</th><th>Editor decision</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<script>
const K='cc-decisions-'+{json.dumps(str(meta.get('show', '')))};
let d={{}};try{{d=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
document.querySelectorAll('select[data-id]').forEach(s=>{{s.value=d[s.dataset.id]||'';s.onchange=()=>{{d[s.dataset.id]=s.value;try{{localStorage.setItem(K,JSON.stringify(d))}}catch(e){{}}}}}});
function f(){{const sv=fsev.value,ck=fcheck.value,v=fver.checked;document.querySelectorAll('tbody tr').forEach(r=>{{
r.style.display=(!sv||r.dataset.sev==sv)&&(!ck||r.dataset.check==ck)&&(!v||r.dataset.ver=='1')?'':'none'}})}}
fsev.onchange=fcheck.onchange=fver.onchange=f;
exp.onclick=()=>{{let c='id,decision\\n';document.querySelectorAll('select[data-id]').forEach(s=>c+=s.dataset.id+','+(s.value||'')+'\\n');
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([c],{{type:'text/csv'}}));a.download='decisions.csv';a.click()}};
</script></body></html>"""
