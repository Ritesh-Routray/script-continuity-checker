"""Streamlit front end:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from cc.llm import LLM, LLMError, DEFAULT_MODELS, load_dotenv
from cc.parser import load_show
from cc.report import html_sheet, label, sort_flags
from cc.rules import run_rules

load_dotenv()
st.set_page_config(page_title="Script Continuity Checker", layout="wide")
st.title("Script Continuity Checker")
st.caption("Catch name drift, speaker-tag splits, timing errors and story contradictions in AI-written "
           "microdrama scripts before they go to video generation.")

with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", ["none (rules only)", "gemini", "groq", "openai-compatible"], index=1)
    prov = {"none (rules only)": "none", "openai-compatible": "openai"}.get(provider, provider)
    model = st.text_input("Model", DEFAULT_MODELS.get(prov, "")) if prov != "none" else None
    key_var = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}.get(prov)
    if key_var:
        secret = None
        try:
            secret = st.secrets.get(key_var)
        except Exception:
            pass
        key = st.text_input(f"{key_var}", value="", type="password",
                            help="Leave empty to use the key from .env / Streamlit secrets")
        if key or secret:
            os.environ[key_var] = key or secret
    if prov == "openai":
        base = st.text_input("Base URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
        os.environ["OPENAI_BASE_URL"] = base
    ep_range = st.text_input("Episodes (optional)", "", placeholder="e.g. 1-20")
    camera = st.checkbox("Send CAMERA lines to the model", True, help="Better visual continuity, ~40% more tokens")

files = st.file_uploader("Episode scripts (.txt files, or one .zip)", type=["txt", "zip"], accept_multiple_files=True)
local = st.text_input("…or a local folder path", "")

if st.button("Run check", type="primary") and (files or local):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(local) if local and not files else Path(tmp)
        for f in files or []:
            if f.name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(f.read())) as z:
                    for n in z.namelist():
                        if n.endswith(".txt"):
                            (folder / Path(n).name).write_bytes(z.read(n))
            else:
                (folder / f.name).write_bytes(f.read())
        episodes = load_show(folder)
    if ep_range:
        from run import parse_range
        keep = parse_range(ep_range)
        episodes = [e for e in episodes if e.number in keep]
    st.write(f"Loaded **{len(episodes)}** episodes, {sum(len(e.events) for e in episodes)} timed blocks.")
    flags = run_rules(episodes)
    meta = {"show": folder.name if local else "upload", "episodes": len(episodes), "model": "rules only"}
    if prov != "none":
        try:
            from cc.semantic import resolve_speakers, run_bible
            llm = LLM(prov, model)
            prog = st.progress(0.0, text="Speaker identities…")
            idf, _ = resolve_speakers(llm, episodes)
            logs = []

            def log(msg):
                logs.append(msg)
                prog.progress(min(len(logs) / len(episodes), 1.0), text=msg.strip())
            sf, bible, _ = run_bible(llm, episodes, log=log, include_camera=camera)
            for i, f in enumerate(idf + sf, 1):
                f.id = f"L{i:03d}"
            flags += idf + sf
            meta.update(model=llm.name, llm_calls=llm.calls, tokens_in=llm.tokens_in, tokens_out=llm.tokens_out)
            prog.empty()
            with st.expander(f"Story bible ({len(bible.active())} active facts)"):
                st.dataframe(pd.DataFrame(bible.active()), use_container_width=True)
        except LLMError as e:
            st.warning(f"LLM layer skipped: {e}")
    st.session_state["flags"] = sort_flags(flags)
    st.session_state["meta"] = meta

if "flags" in st.session_state:
    flags, meta = st.session_state["flags"], st.session_state["meta"]
    df = pd.DataFrame([{"id": f.id, "severity": f.severity, "verified": f.verified, "check": label(f.check),
                        "issue": f.title, "detail": f.detail,
                        "evidence": " || ".join(f"ep{e['ep']} {e['ts']} {e['text']}" for e in f.evidence)}
                       for f in flags])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flags", len(df))
    c2.metric("High", int((df.severity == "high").sum()))
    c3.metric("Unverified LLM", int((~df.verified).sum()))
    c4.metric("Model", meta.get("model", ""))
    sev = st.multiselect("Severity", ["high", "medium", "low"], ["high", "medium", "low"])
    checks = st.multiselect("Checks", sorted(df.check.unique()), sorted(df.check.unique()))
    view = df[df.severity.isin(sev) & df.check.isin(checks)]
    st.dataframe(view, use_container_width=True, hide_index=True, height=520)
    st.download_button("Download triage sheet (HTML)", html_sheet(flags, meta), "continuity_report.html", "text/html")
    st.download_button("Download flags (JSON)", json.dumps([f.to_dict() for f in flags], indent=1, ensure_ascii=False),
                       "flags.json", "application/json")
