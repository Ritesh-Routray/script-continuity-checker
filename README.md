# Script Continuity Checker (Kuku assignment, Problem 3)

**Who it's for:** the script editor or showrunner on an AI-generated microdrama, at the step between
"LLM wrote the episode scripts" and "we spend money generating video and voice".

**What it does:** it reads every episode script in a show and returns a ranked list of continuity
errors. Each flag carries the episode and timestamp, so an editor can check it in seconds.

| Layer | Checks | Needs a key? |
|---|---|---|
| Rules (`cc/rules.py`) | spelling drift in names and places, one character under several speaker tags, recurring speakers with no name, timestamps that run backwards, gaps or overlaps, lines too long for their time slot, continuity notes that point at the wrong episode, logline names missing from the script | No |
| LLM (`cc/semantic.py`) | (1) which speaker tags are the same person; (2) a rolling **story bible** of names, relationships, deaths, injuries, objects and secrets, with every new episode checked against it | Yes (free tier) |

**Anti-hallucination:** the model has to quote the script word for word. Code then checks each
quote against the real script and checks that every bible fact ID it cites exists. A fact that
fails this check never enters the bible. A contradiction that fails it is shown separately as
*unverified*.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY and/or GROQ_API_KEY (both have free tiers)

# full run (rules + LLM), writes outputs/<show>/report.html, report.md, flags.json, story_bible.json, llm_raw.jsonl
python run.py "path/to/episodes_folder" --provider gemini
python run.py "path/to/episodes_folder" --provider groq --model openai/gpt-oss-120b
python run.py "path/to/episodes_folder" --provider none        # rules only, <1 s, no key

# UI
streamlit run app.py

# quality check: plant known errors in a copy, measure recall
python eval_inject.py "path/to/episodes_folder" --seeds 5                 # rule-level errors
python eval_inject.py "path/to/episodes_folder" --seeds 1 --provider gemini  # + story-level errors
```

**Swapping the model:** set `--provider` / `--model`, or `LLM_PROVIDER` / `LLM_MODEL` in `.env`.
`openai` works with any OpenAI-compatible endpoint (OpenRouter, Ollama, vLLM, an internal gateway)
through `OPENAI_BASE_URL`. Responses are cached in `.llm_cache/`, so a rerun costs nothing.

**Input format:** one `.txt` file per episode, named `episode-NNN-*.txt`, in the screenplay format
of the input pack (`**SPEAKER (Emotion):**` followed by `Start: (MM:SS.ss) | End: (MM:SS.ss)`).
