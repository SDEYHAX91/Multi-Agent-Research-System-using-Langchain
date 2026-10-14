"""
ResearchScribe — Multi-Agent Research System

# OnRender Link:
https://multi-agent-research-system-using-28d2.onrender.com/

Run from the project root:
    streamlit run path/to/app.py
"""

import html
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# pyrefly: ignore [missing-import]
from agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    writer_chain,
)
# pyrefly: ignore [missing-import]
from auth import render_account_sidebar, require_login

APP_NAME = "ResearchScribe"

STEPS = [
    ("01", "Search Agent", "Gathers recent web information"),
    ("02", "Reader Agent", "Scrapes & extracts deep content"),
    ("03", "Writer Chain", "Drafts the full research report"),
    ("04", "Critic Chain", "Reviews & scores the report"),
]

# Shown only if the model can't generate suggestions (rate limit, bad output, ...)
FALLBACK_EXAMPLES = ["LLM agents 2026", "CRISPR gene editing", "Fusion energy progress"]

# History is saved next to this file (one file per account) so it survives refreshes / restarts
HISTORY_DIR = Path(__file__).with_name("history")
LEGACY_HISTORY_FILE = Path(__file__).with_name("research_history.json")  # pre-login, single shared file
ICON_FILE = Path(__file__).resolve().parent / "ICON.png"
if not ICON_FILE.exists() and Path("ICON.png").resolve().exists():
    ICON_FILE = Path("ICON.png").resolve()          
LOGO_FILE = ICON_FILE if ICON_FILE.exists() else Path(__file__).resolve().parent / "logo.png"
MAX_HISTORY = 30

# ------------------------------------------------------------------ #
# Page config + styling
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title=f"{APP_NAME} · Multi-Agent Research",
    page_icon=str(ICON_FILE) if ICON_FILE.exists() else "🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #07090d;
  --panel: #0e1217;
  --panel-2: #131920;
  --border: #232a33;
  --text: #ebe8e1;
  --muted: #8b919b;
  --accent: #19e6b4;     /* cyan-green */
  --accent-2: #22d3ee;   /* cyan */
}

/* ---------- base ---------- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important;}
header[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stSidebar"] {
  min-width: 280px !important;
  transform: none !important;
}
.stApp {
  background:
    radial-gradient(1100px 480px at 50% -8%, rgba(25,230,180,.11), transparent 62%),
    var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
}
.stApp .block-container, .stApp [data-testid="stMainBlockContainer"] {
  max-width: 1400px !important; padding: 3rem 3rem 2rem 3rem !important;
}
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th {color: var(--text);}
.stApp a {color: var(--accent-2);}

/* ---------- hero ---------- */
.eyebrow {
  text-align: center; font-family: 'JetBrains Mono', monospace; font-size: .72rem;
  letter-spacing: .28em; color: var(--accent); text-transform: uppercase;
}
.brand {
  text-align: center; font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
  font-size: clamp(2.4rem, 7vw, 4.8rem) !important; line-height: 1.35 !important; letter-spacing: -.025em;
  margin: .6rem 0 1.4rem 0; padding: 0;
  white-space: nowrap;   /* prevents wrapping to 2 lines */
}
.brand .b1 {color: var(--text) !important;}
.brand .b2 {color: var(--accent) !important;}
.tagline {
  text-align: center; max-width: 780px; margin: 0 auto 2.4rem auto;
  color: var(--muted) !important; font-size: 1.12rem; line-height: 1.7;
}
.fade-line {
  height: 1px; margin: 0 0 2.2rem 0;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}

/* ---------- section titles ---------- */
.section-title {
  font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.5rem;
  margin: 0 0 .4rem 0; padding: 0; color: var(--text);
}

/* ---------- input form ---------- */
[data-testid="stForm"] {border: none !important; padding: 0 !important; background: transparent !important;}
[data-testid="stWidgetLabel"] p {
  font-family: 'JetBrains Mono', monospace !important; font-size: .72rem !important;
  letter-spacing: .14em; text-transform: uppercase; color: var(--accent) !important;
}
.stTextInput div[data-baseweb="input"] {
  background: var(--panel-2) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; transition: all .15s ease;
}
.stTextInput div[data-baseweb="input"]:focus-within {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(25,230,180,.15);
}
.stTextInput input {color: var(--text) !important; font-size: .95rem;}
.stTextInput input::placeholder {color: #5d636d !important;}

/* primary button (cyan-green gradient) */
[data-testid="stFormSubmitButton"] button, button[kind="primary"] {
  width: 100%; height: 3rem; border: none !important; border-radius: 10px !important;
  background: linear-gradient(90deg, #10e6a0 0%, #22d3ee 100%) !important;
  box-shadow: 0 6px 22px rgba(25,230,180,.28);
  transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}
[data-testid="stFormSubmitButton"] button p, button[kind="primary"] p {
  color: #03140f !important; font-weight: 700; font-size: .95rem;
}
[data-testid="stFormSubmitButton"] button:hover, button[kind="primary"]:hover {
  transform: translateY(-1px); filter: brightness(1.06);
  box-shadow: 0 10px 28px rgba(25,230,180,.38);
}

/* example chips */
.try-label {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .16em;
  color: var(--muted) !important; margin: 1.2rem 0 .2rem 0;
}
.st-key-chips {gap: .45rem !important;}
.st-key-chips button {
  min-height: 0 !important; padding: .28rem .75rem !important; border-radius: 8px !important;
  background: var(--panel-2) !important; border: 1px solid var(--border) !important;
}
.st-key-chips button p {font-size: .78rem !important; color: #b4b8bf !important;}
.st-key-chips button:hover {border-color: var(--accent) !important;}
.st-key-chips button:hover p {color: var(--accent) !important;}

/* ---------- pipeline cards ---------- */
.step {
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: .95rem 1.2rem; transition: all .25s ease;
}
.step-head {display: flex; align-items: center; justify-content: space-between;}
.step-left {display: flex; align-items: baseline; gap: .8rem;}
.step-num {font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: var(--accent);}
.step-name {font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; color: var(--text);}
.step-status {
  font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: .14em; color: #5d636d;
}
.step-desc {margin-top: .35rem; font-size: .8rem; color: var(--muted);}
.step.running {
  border-color: var(--accent-2); background: rgba(34,211,238,.05);
  animation: pulse 1.4s ease-in-out infinite;
}
.step.running .step-status {color: var(--accent-2);}
.step.done {border-left: 3px solid var(--accent); background: rgba(25,230,180,.05);}
.step.done .step-status {color: var(--accent);}
.step.error {border-left: 3px solid #ff5c6c; background: rgba(255,92,108,.06);}
.step.error .step-status {color: #ff5c6c;}
@keyframes pulse {
  0%, 100% {box-shadow: 0 0 0 0 rgba(34,211,238,.35);}
  50% {box-shadow: 0 0 0 6px rgba(34,211,238,0);}
}

/* ---------- results ---------- */
[class*="st-key-card_"] {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.3rem 1.6rem;
}
.card-label {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .18em;
  color: var(--accent); text-transform: uppercase;
  padding-bottom: .7rem; margin-bottom: .6rem; border-bottom: 1px solid var(--border);
}
.meta {font-family: 'JetBrains Mono', monospace; font-size: .75rem; color: var(--muted) !important; margin-bottom: 1rem;}
[data-testid="stExpander"] details {
  background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
}
[data-testid="stExpander"] summary p {font-size: .9rem; color: var(--text);}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {font-family: 'Syne', sans-serif; color: var(--text);}
.stMarkdown code {background: var(--panel-2); color: var(--accent-2);}
.stDownloadButton button {
  background: transparent !important; border: 1px solid var(--accent) !important; border-radius: 10px !important;
}
.stDownloadButton button p {color: var(--accent) !important; font-weight: 600;}
.stDownloadButton button:hover {background: rgba(25,230,180,.1) !important;}

.stApp [data-testid="stSidebar"] {background: var(--panel) !important; border-right: 1px solid var(--border);}
.side-brand {
  font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.35rem; letter-spacing: -.02em;
  color: var(--text); margin: .2rem 0 1rem 0;
  white-space: nowrap;
}
.side-brand span {color: var(--accent);}
.side-label {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .18em;
  color: var(--accent); margin: 1.3rem 0 .5rem 0;
}
.side-empty {font-size: .82rem; color: var(--muted) !important; line-height: 1.5;}
.hist-meta {
  font-family: 'JetBrains Mono', monospace; font-size: .64rem; color: #5d636d !important;
  margin: .25rem 0 .6rem .25rem;
  line-height: 1.3;
}
.st-key-newbtn button {
  background: transparent !important; border: 1px solid var(--accent) !important; border-radius: 10px !important;
}
.st-key-newbtn button p {color: var(--accent) !important; font-weight: 600;}
.st-key-newbtn button:hover {background: rgba(25,230,180,.1) !important;}
.st-key-history {gap: .45rem !important;}
.st-key-history [data-testid="stVerticalBlock"] {gap: .45rem !important;}
.st-key-history [data-testid="stHorizontalBlock"] {align-items: center !important;}
.st-key-history button {
  justify-content: flex-start !important; background: var(--panel-2) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important;
  padding: .5rem .75rem !important; min-height: 0 !important;
  width: 100% !important; transition: border-color .15s ease, background .15s ease;
}
.st-key-history [class*="st-key-hist_"] button {
  display: flex !important; flex-direction: column !important;
  align-items: flex-start !important; justify-content: center !important;
  padding: .55rem .75rem !important; min-height: 52px !important;
}
.st-key-history [class*="st-key-hist_"] button:hover {border-color: var(--accent) !important;}
.st-key-history [class*="st-key-hist_"] button div {width: 100% !important; text-align: left !important;}
.st-key-history [class*="st-key-hist_"] button p {
  text-align: left !important; width: 100% !important; margin: 0 !important;
  white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
}
.st-key-history [class*="st-key-hist_"] button p:first-of-type,
.st-key-history [class*="st-key-hist_"] button strong {
  font-size: .84rem !important; font-weight: 600 !important; color: var(--text) !important;
  line-height: 1.25 !important; margin-bottom: .18rem !important; display: block !important;
}
.st-key-history [class*="st-key-hist_"] button p:last-of-type:not(:first-of-type) {
  font-family: 'JetBrains Mono', monospace !important; font-size: .63rem !important;
  font-weight: 400 !important; color: #727883 !important; line-height: 1.2 !important;
  margin: 0 !important; display: block !important;
}
.st-key-history [class*="st-key-del_"] button {
  background: transparent !important; border: none !important; padding: .35rem !important;
  justify-content: center !important; align-items: center !important; min-height: 0 !important; width: auto !important;
}
.st-key-history [class*="st-key-del_"] button p {color: #5d636d !important; font-size: .85rem !important; margin: 0 !important; line-height: 1 !important;}
.st-key-history [class*="st-key-del_"] button:hover p {color: #ff5c6c !important;}
.st-key-clearbtn button {background: transparent !important; border: none !important;}
.st-key-clearbtn button p {color: var(--muted) !important; font-size: .78rem !important;}
.st-key-clearbtn button:hover p {color: #ff5c6c !important;}

/* ---------- confirmation dialog ---------- */
[data-testid="stDialog"] div[role="dialog"] {
  background: var(--panel) !important; border: 1px solid var(--border) !important;
  border-radius: 16px !important; box-shadow: 0 24px 70px rgba(0,0,0,.6);
}
[data-testid="stDialog"] h2, [data-testid="stDialog"] p {color: var(--text) !important;}
[data-testid="stDialog"] h2 {font-family: 'Syne', sans-serif !important; font-weight: 700 !important;}
.dlg-topic {
  font-family: 'JetBrains Mono', monospace; font-size: .82rem; color: var(--accent-2);
  background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px;
  padding: .5rem .75rem; margin: .4rem 0 .2rem 0; word-break: break-word;
}
.dlg-note {font-size: .85rem; color: var(--muted) !important; margin-top: .4rem;}
[data-testid="stDialog"] button {border-radius: 10px !important; height: 2.6rem;}
[data-testid="stDialog"] [class*="st-key-dlg_cancel"] button {
  background: transparent !important; border: 1px solid var(--border) !important;
}
[data-testid="stDialog"] [class*="st-key-dlg_cancel"] button:hover {border-color: var(--accent) !important;}
[data-testid="stDialog"] [class*="st-key-dlg_confirm"] button {
  background: #ff5c6c !important; border: none !important; box-shadow: 0 6px 22px rgba(255,92,108,.28);
}
[data-testid="stDialog"] [class*="st-key-dlg_confirm"] button p {color: #1a0508 !important; font-weight: 700;}
[data-testid="stDialog"] [class*="st-key-dlg_confirm"] button:hover {filter: brightness(1.08);}

.footer {
  text-align: center; margin-top: 3rem; font-family: 'JetBrains Mono', monospace;
  font-size: .68rem; letter-spacing: .06em; color: #5d636d !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Authentication gate — shows login / sign-up and stops here if not signed in
# ------------------------------------------------------------------ #
require_login(APP_NAME)

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def to_text(content) -> str:
    """Normalise LangChain message content (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def fresh_steps():
    return [{"status": "waiting", "secs": None} for _ in STEPS]


def history_file() -> Path:
    return HISTORY_DIR / f"user_{st.session_state.user['id']}.json"


def load_history() -> list:
    path = history_file()
    if not path.exists() and st.session_state.user["id"] == 1 and LEGACY_HISTORY_FILE.exists():
        path = LEGACY_HISTORY_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def save_history(history: list) -> None:
    try:
        HISTORY_DIR.mkdir(exist_ok=True)
        history_file().write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception: 
        pass  


def md_escape(text: str) -> str:
    for ch in ("\\", "*", "_", "`", "[", "]", "#", "~", "<", ">"):
        text = text.replace(ch, "\\" + ch)
    return text


def load_item(item_id: str) -> None:
    for item in st.session_state.history:
        if item["id"] == item_id:
            timings = item.get("timings") or []
            st.session_state.result = item
            st.session_state.topic_input = item["topic"]
            st.session_state.steps = [
                {"status": "done", "secs": timings[i] if i < len(timings) else None}
                for i in range(len(STEPS))
            ]
            break


def delete_item(item_id: str) -> None:
    st.session_state.history = [h for h in st.session_state.history if h["id"] != item_id]
    save_history(st.session_state.history)
    if (st.session_state.result or {}).get("id") == item_id:
        new_research()


def clear_history() -> None:
    st.session_state.history = []
    save_history([])
    new_research()


def new_research() -> None:
    st.session_state.result = None
    st.session_state.steps = fresh_steps()
    st.session_state.topic_input = ""

_dialog = getattr(st, "dialog", None) or st.experimental_dialog


def _finish_dialog_action(fn, *args) -> None:
    """Run the action in a button callback, then flag the dialog to close on rerun."""
    fn(*args)
    st.session_state["_dialog_done"] = True


@_dialog("Delete this research?")
def confirm_delete_dialog(item_id: str) -> None:
    if st.session_state.pop("_dialog_done", False):
        st.rerun()  # action finished -> close the dialog
    item = next((h for h in st.session_state.history if h["id"] == item_id), None)
    topic = item["topic"] if item else "this item"
    st.markdown(f'<div class="dlg-topic">{html.escape(topic)}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dlg-note">This will permanently remove it from your history. '
        "This can't be undone.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(key="dlg_cancel_one"):
            if st.button("Cancel", use_container_width=True):
                st.rerun()
    with c2:
        with st.container(key="dlg_confirm_one"):
            st.button(
                "Delete",
                type="primary",
                use_container_width=True,
                on_click=_finish_dialog_action,
                args=(delete_item, item_id),
            )


@_dialog("Clear all history?")
def confirm_clear_dialog() -> None:
    if st.session_state.pop("_dialog_done", False):
        st.rerun()
    n = len(st.session_state.history)
    st.markdown(
        f'<div class="dlg-note" style="margin-top:0">This will permanently delete all '
        f"<b>{n}</b> saved research run{'s' if n != 1 else ''}. This can't be undone.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(key="dlg_cancel_all"):
            if st.button("Cancel", use_container_width=True):
                st.rerun()
    with c2:
        with st.container(key="dlg_confirm_all"):
            st.button(
                "Clear all",
                type="primary",
                use_container_width=True,
                on_click=_finish_dialog_action,
                args=(clear_history,),
            )


def step_html(i: int, status: str, secs=None) -> str:
    num, name, desc = STEPS[i]
    label = {
        "waiting": "WAITING",
        "running": "● RUNNING",
        "done": "✓ DONE" + (f" · {secs:.1f}s" if secs is not None else ""),
        "error": "✕ FAILED",
    }[status]
    return (
        f'<div class="step {status}">'
        f'<div class="step-head"><div class="step-left">'
        f'<span class="step-num">{num}</span><span class="step-name">{name}</span></div>'
        f'<span class="step-status">{label}</span></div>'
        f'<div class="step-desc">{desc}</div></div>'
    )


def run_pipeline(topic: str, set_step) -> dict:
    """Same 4 steps as pipeline.run_research_pipeline, reporting progress via set_step."""
    state = {"topic": topic}
    timings = {}
    i = 0
    try:
        # 1. Search
        i = 0
        set_step(i, "running")
        t0 = time.time()
        search_result = build_search_agent().invoke(
            {"messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]}
        )
        state["search_result"] = to_text(search_result["messages"][-1].content)
        timings[i] = time.time() - t0
        set_step(i, "done", timings[i])

        # 2. Reader
        i = 1
        set_step(i, "running")
        t0 = time.time()
        reader_result = build_reader_agent().invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}'. "
                        f"Pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_result'][:800]}",
                    )
                ]
            }
        )
        state["scraped_content"] = to_text(reader_result["messages"][-1].content)
        timings[i] = time.time() - t0
        set_step(i, "done", timings[i])

        # 3. Writer
        i = 2
        set_step(i, "running")
        t0 = time.time()
        research_combined = (
            f"SEARCH RESULTS: {state['search_result']} \n\n"
            f"DETAILED SCRAPED CONTENT: {state['scraped_content']}"
        )
        state["report"] = to_text(writer_chain.invoke({"topic": topic, "research": research_combined}))
        timings[i] = time.time() - t0
        set_step(i, "done", timings[i])

        # 4. Critic
        i = 3
        set_step(i, "running")
        t0 = time.time()
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        timings[i] = time.time() - t0
        set_step(i, "done", timings[i])
    except Exception:
        set_step(i, "error")
        raise

    state["total_time"] = sum(timings.values())
    state["timings"] = [timings.get(k) for k in range(len(STEPS))]
    return state


def use_example(text: str):
    st.session_state.topic_input = text


def generate_examples(avoid=None, k: int = 3) -> list:
    """Let the LLM invent k short research topics. Never raises; falls back to static chips."""
    avoid = [a for a in (avoid or []) if a]
    prompt = (
        f"Today is {datetime.now():%B %d, %Y}. Suggest {k} interesting, current topics worth "
        "researching on the web. Make them come from clearly different fields and be a bit "
        "unexpected rather than the obvious headline topics.\n"
        "Each topic: 2-4 words, specific, and usable as a search query.\n"
        f"Do not repeat or closely resemble: {', '.join(avoid) or 'none'}.\n"
        f'Reply with ONLY a JSON array of {k} strings, e.g. ["Topic one", "Topic two", "Topic three"].'
    )
    try:
        # pyrefly: ignore [missing-import]
        from llm_failsafe import get_model

        text = to_text(get_model("search", temperature=1.0).invoke(prompt).content)
        match = re.search(r"\[.*?\]", text, re.DOTALL)  # tolerate code fences / extra prose
        topics = [t.strip() for t in json.loads(match.group(0)) if isinstance(t, str) and t.strip()]
        topics = [t for t in topics if len(t) <= 40][:k]
        if len(topics) == k:
            return topics
    except Exception: 
        pass
    return list(FALLBACK_EXAMPLES)


# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #
st.session_state.setdefault("topic_input", "")
st.session_state.setdefault("steps", fresh_steps())
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", load_history())

# ------------------------------------------------------------------ #
# Hero
# ------------------------------------------------------------------ #
st.markdown(
    f"""<div class="eyebrow">Multi-Agent AI System</div>
<div class="brand"><span class="b1">Research</span><span class="b2">Scribe</span></div>
<p class="tagline">Four specialized AI agents collaborate — searching, scraping, writing, and critiquing — to deliver a polished research report on any topic.</p>
<div class="fade-line"></div>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Input (left) + Pipeline (right)
# ------------------------------------------------------------------ #
left, right = st.columns([1.05, 1], gap="large")

with left:
    with st.form("research_form", border=False):
        topic = st.text_input(
            "Research topic",
            key="topic_input",
            placeholder="e.g. Quantum computing breakthroughs in 2026",
        )
        submitted = st.form_submit_button("Run Research Pipeline  →", type="primary")

    st.markdown('<div class="try-label">TRY →</div>', unsafe_allow_html=True)
    if "examples" not in st.session_state:
        with st.spinner("Finding ideas..."):
            recent = [h["topic"] for h in st.session_state.history[:6]]
            st.session_state.examples = generate_examples(avoid=recent)
    with st.container(key="chips"):
        for i, ex in enumerate(st.session_state.examples):
            st.button(ex, key=f"chip_{i}", on_click=use_example, args=(ex,))

    if submitted and not topic.strip():
        st.warning("Please enter a research topic first.")

with right:
    st.markdown('<h3 class="section-title">Pipeline</h3>', unsafe_allow_html=True)
    slots = [st.empty() for _ in STEPS]


def set_step(i: int, status: str, secs=None):
    st.session_state.steps[i] = {"status": status, "secs": secs}
    slots[i].markdown(step_html(i, status, secs), unsafe_allow_html=True)


# Render current pipeline state
for idx, s in enumerate(st.session_state.steps):
    slots[idx].markdown(step_html(idx, s["status"], s["secs"]), unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------ #
if submitted and topic.strip():
    st.session_state.steps = fresh_steps()
    st.session_state.result = None
    for idx in range(len(STEPS)):
        set_step(idx, "waiting")
    try:
        new_result = run_pipeline(topic.strip(), set_step)
        new_result["id"] = uuid.uuid4().hex[:8]
        new_result["created_at"] = datetime.now().strftime("%d %b, %H:%M")
        st.session_state.result = new_result
        st.session_state.history = [new_result] + st.session_state.history[: MAX_HISTORY - 1]
        save_history(st.session_state.history)
    except Exception as e:  # noqa: BLE001
        st.error(f"Pipeline failed: {e}")

# ------------------------------------------------------------------ #
# Results
# ------------------------------------------------------------------ #
res = st.session_state.result
if res:
    st.markdown('<div class="fade-line" style="margin-top:2.5rem"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Results</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="meta">⏱ {res["total_time"]:.1f}s total &nbsp;·&nbsp; '
        f'📝 {len(res["report"].split()):,} words &nbsp;·&nbsp; 🧭 {res["topic"]}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Search Results (raw)"):
        with st.container(key="card_search"):
            st.markdown('<div class="card-label">Search Agent Output</div>', unsafe_allow_html=True)
            st.markdown(res["search_result"])

    with st.expander("📄 Scraped Content (raw)"):
        with st.container(key="card_scrape"):
            st.markdown('<div class="card-label">Reader Agent Output</div>', unsafe_allow_html=True)
            st.markdown(res["scraped_content"])

    st.write("")
    with st.container(key="card_report"):
        st.markdown('<div class="card-label">📝 Final Research Report</div>', unsafe_allow_html=True)
        st.markdown(res["report"])
        st.download_button(
            "⬇ Download report (.md)",
            data=res["report"],
            file_name=f"report_{res['topic'][:30].strip().replace(' ', '_')}.md",
            mime="text/markdown",
        )

    st.write("")
    with st.container(key="card_critic"):
        st.markdown('<div class="card-label">🧐 Critic Review</div>', unsafe_allow_html=True)
        st.markdown(res["feedback"])

# ------------------------------------------------------------------ #
# Sidebar — history
# ------------------------------------------------------------------ #
with st.sidebar:
    if ICON_FILE.exists():
        st.image(str(ICON_FILE), width=30)
    elif LOGO_FILE.exists():
        st.image(str(LOGO_FILE), width=30)
    else:  # fallback if icon image is missing
        st.markdown('<div class="side-brand">Research<span>Scribe</span></div>', unsafe_allow_html=True)
    with st.container(key="newbtn"):
        st.button("＋  New research", on_click=new_research, use_container_width=True)

    st.markdown('<div class="side-label">HISTORY</div>', unsafe_allow_html=True)
    history = st.session_state.history
    active_id = (st.session_state.result or {}).get("id")

    if not history:
        st.markdown(
            '<div class="side-empty">No research yet. Your finished runs will be saved here.</div>',
            unsafe_allow_html=True,
        )
    else:
        if active_id:
            st.markdown(
                f"<style>.st-key-hist_{active_id} button {{border-color: var(--accent) !important; "
                f"background: rgba(25,230,180,.08) !important;}}</style>",
                unsafe_allow_html=True,
            )
        with st.container(key="history"):
            for item in history:
                iid = item["id"]
                title = item["topic"] if len(item["topic"]) <= 28 else item["topic"][:27] + "…"
                words = len(item.get("report", "").split())
                created = item.get("created_at", "")
                meta_text = f"{created} · {words:,} words" if created else f"{words:,} words"
                c1, c2 = st.columns([5, 1], gap="small", vertical_alignment="center")
                with c1:
                    st.button(
                        f"**{md_escape(title)}**\n\n{meta_text}",
                        key=f"hist_{iid}",
                        on_click=load_item,
                        args=(iid,),
                        use_container_width=True,
                        help=item["topic"],
                        wrap=True,
                    )
                with c2:
                    if st.button("✕", key=f"del_{iid}", help="Delete"):
                        confirm_delete_dialog(iid)
        with st.container(key="clearbtn"):
            if st.button("Clear all history"):
                confirm_clear_dialog()

    render_account_sidebar()  # signed-in user + Log out

st.markdown(
    f'<div class="footer">{APP_NAME} · Powered by LangChain multi-agent pipeline · Built with Streamlit</div>',
    unsafe_allow_html=True,
)
