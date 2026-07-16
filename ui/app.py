"""Streamlit Chat UI — Unified AI Platform Interface.

Premium dark-themed interface with:
- Auto-routing via orchestrator (or manual mode)
- Rich metadata display: intent, sources, tools, guardrail status
- Glassmorphism cards, animated typing indicator, color-coded pills
- Quick-action prompts in the empty state
- File upload for RAG
"""
import json
import os
import re

import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://app:8080")
# Default to settings.default_model — single source of truth.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:4b")
STREAM_URL = f"{API_URL}/v1/chat/stream"


@st.cache_data(ttl=30)
def _fetch_available_models() -> list[str]:
    """Fetch chat-capable model names from the backend.

    Filters out embedding models (nomic-embed, etc.) since they cannot
    be used for text generation — LiteLLM/Ollama will reject them with
    "does not support generate".

    Returns DEFAULT_MODEL as a safe fallback if the request fails.
    """
    try:
        r = requests.get(f"{API_URL}/v1/models", timeout=3)
        r.raise_for_status()
        data = r.json()
        names = [m.get("id", m.get("name")) for m in data.get("data", []) if m.get("id") or m.get("name")]
        # Exclude embedding models — they only support /embeddings, not /chat/completions
        names = [n for n in names if "embed" not in n.lower()]
        return names or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


st.set_page_config(
    page_title="AI Platform",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# PREMIUM CSS — dark glassmorphism, gradients, smooth animations
# ============================================================================
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
    --bg-0: #0a0a14;
    --bg-1: #11111e;
    --bg-2: #1a1a2e;
    --primary: #667eea;
    --primary-bright: #8b9aff;
    --secondary: #764ba2;
    --accent: #f093fb;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
    --info: #3b82f6;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --text-mute: #64748b;
    --border: rgba(255,255,255,0.08);
    --border-bright: rgba(102,126,234,0.3);
    --glass: rgba(30,30,50,0.5);
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
code, pre, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }

.stApp {
    background:
        radial-gradient(at 20% 0%, rgba(102,126,234,0.15) 0%, transparent 50%),
        radial-gradient(at 80% 100%, rgba(118,75,162,0.12) 0%, transparent 50%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
    background-attachment: fixed;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,15,26,0.98) 0%, rgba(20,20,35,0.98) 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text);
}

/* Hero */
.hero {
    padding: 1.5rem 0 1.25rem 0;
    text-align: center;
    background: linear-gradient(180deg, transparent 0%, rgba(102,126,234,0.05) 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    border-radius: 0 0 24px 24px;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 50%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero p { color: var(--text-dim); margin: 0.5rem 0 0 0; font-size: 0.95rem; font-weight: 400; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: var(--glass) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.4rem !important;
    margin: 0.6rem 0 !important;
    transition: all 0.2s ease;
    animation: fadeInUp 0.3s ease;
}
[data-testid="stChatMessage"]:hover { border-color: var(--border-bright) !important; transform: translateY(-1px); }
/* Chat bubble layout alignment is handled via st.columns in Python
   (more reliable than CSS :has() selectors across Streamlit versions). */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

/* ===== Custom chat bubbles (column-based) ===== */
.bubble {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin: 0.6rem 0;
    transition: all 0.2s ease;
    animation: fadeInUp 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.bubble:hover { border-color: var(--border-bright); transform: translateY(-1px); }
.bubble-user {
    background: linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.12) 100%) !important;
    border-color: rgba(102,126,234,0.3) !important;
    border-left: 3px solid var(--primary) !important;
}
.bubble-assistant {
    background: var(--glass) !important;
    border-color: var(--border) !important;
    border-left: 3px solid var(--secondary) !important;
}
.bubble-content { color: var(--text); line-height: 1.55; word-wrap: break-word; }
.bubble-content p { margin: 0 0 0.6rem 0; }
.bubble-content p:last-child { margin-bottom: 0; }
.bubble-content code {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.1rem 0.4rem;
    font-size: 0.85em;
}
.bubble-content pre {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    overflow-x: auto;
}

/* Blinking cursor for streaming text */
.cursor-blink {
    display: inline-block;
    color: var(--primary);
    animation: cursorBlink 1s steps(2) infinite;
    font-weight: 700;
    margin-left: 1px;
}
@keyframes cursorBlink { 50% { opacity: 0; } }

/* Reasoning block (collapsible) */
details.reasoning {
    transition: all 0.2s ease;
}
details.reasoning summary::marker { display: none; }
details.reasoning summary::-webkit-details-marker { display: none; }
details.reasoning summary:hover { color: var(--text) !important; }
details.reasoning pre {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Glass card */
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
    transition: all 0.2s ease;
}
.glass-card:hover { border-color: var(--border-bright); }

/* Section title */
.section-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-mute);
    font-weight: 700;
    margin: 1.1rem 0 0.5rem 0;
    padding-left: 0.2rem;
}

/* Pills */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}
.pill-primary { background: rgba(102,126,234,0.15); color: var(--primary-bright); border-color: rgba(102,126,234,0.3); }
.pill-success { background: rgba(34,197,94,0.15); color: var(--success); border-color: rgba(34,197,94,0.3); }
.pill-warning { background: rgba(245,158,11,0.15); color: var(--warning); border-color: rgba(245,158,11,0.3); }
.pill-error   { background: rgba(239,68,68,0.15); color: var(--error); border-color: rgba(239,68,68,0.3); }
.pill-info    { background: rgba(59,130,246,0.15); color: var(--info); border-color: rgba(59,130,246,0.3); }
.intent-direct_chat { background: rgba(34,197,94,0.2); color: var(--success); border-color: rgba(34,197,94,0.3); }
.intent-rag_query   { background: rgba(59,130,246,0.2); color: var(--info); border-color: rgba(59,130,246,0.3); }
.intent-agent_task  { background: rgba(168,85,247,0.2); color: #c084fc; border-color: rgba(168,85,247,0.3); }
.intent-blocked     { background: rgba(239,68,68,0.2); color: var(--error); border-color: rgba(239,68,68,0.3); }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.1rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(102,126,234,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.35) !important;
    color: white !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Inputs */
.stTextInput input, .stChatInput input {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
}
.stTextInput input:focus, .stChatInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important;
}
.stTextInput label, .stChatInput label { color: var(--text-dim) !important; font-weight: 500; }

/* Selectbox */
.stSelectbox [data-baseweb="select"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px);
}
.stSelectbox [data-baseweb="select"]:hover { border-color: var(--border-bright) !important; }

/* Radio */
.stRadio > label { color: var(--text-dim) !important; font-weight: 500; }
.stRadio [role="radiogroup"] label {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
    margin: 0.25rem 0;
    transition: all 0.2s ease;
}
.stRadio [role="radiogroup"] label:hover {
    border-color: var(--border-bright);
    background: rgba(30,30,50,0.7);
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--glass) !important;
    border: 1px dashed var(--border-bright) !important;
    border-radius: 12px !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.7rem 1rem;
}
[data-testid="stMetricValue"] { color: var(--text); font-weight: 700; }
[data-testid="stMetricLabel"] { color: var(--text-dim); font-size: 0.8rem; }

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-dim) !important; }

/* Code */
.stCodeBlock, code, pre {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(102,126,234,0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(102,126,234,0.5); }

/* Divider */
hr { border-color: var(--border) !important; margin: 1rem 0; }

/* Headings */
h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600; }

/* Quick action cards */
.qa-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin: 0.5rem 0;
    transition: all 0.25s ease;
    cursor: pointer;
    color: var(--text);
    font-size: 0.92rem;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    width: 100%;
}
.qa-card:hover {
    border-color: var(--primary);
    background: rgba(102,126,234,0.1);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(102,126,234,0.15);
}
.qa-icon {
    font-size: 1.5rem;
    width: 2.4rem;
    height: 2.4rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
    border-radius: 10px;
    flex-shrink: 0;
}
.qa-content { flex: 1; }
.qa-title { font-weight: 600; color: var(--text); margin-bottom: 0.15rem; }
.qa-desc { font-size: 0.78rem; color: var(--text-dim); }

/* Source chip */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    margin: 0.25rem 0.3rem 0.25rem 0;
    font-size: 0.8rem;
    color: var(--text-dim);
    transition: all 0.2s ease;
}
.source-chip:hover { border-color: var(--border-bright); color: var(--text); }

/* Typing indicator */
.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.2rem;
    color: var(--text-dim);
    font-size: 0.85rem;
}
.typing-dot {
    width: 7px;
    height: 7px;
    background: var(--primary);
    border-radius: 50%;
    animation: typingBounce 1.2s infinite ease-in-out;
}
.typing-dot:nth-child(2) { animation-delay: 0.15s; }
.typing-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
    animation: pulse 2s infinite;
}
.status-online  { background: var(--success); box-shadow: 0 0 8px rgba(34,197,94,0.6); }
.status-offline { background: var(--error);   box-shadow: 0 0 8px rgba(239,68,68,0.6); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

/* Spinner override */
.stSpinner > div { border-color: var(--primary) !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata_log" not in st.session_state:
    st.session_state.metadata_log = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================================
# HELPERS
# ============================================================================
def _latency_pill(latency_ms: float) -> str:
    """Render latency with traffic-light color."""
    if latency_ms < 5000:
        kind, icon = "success", "⚡"
    elif latency_ms < 15000:
        kind, icon = "info", "⏱"
    else:
        kind, icon = "warning", "🐢"
    return f'<span class="pill pill-{kind}">{icon} {latency_ms:.0f}ms</span>'


def _render_message_metadata(meta: dict) -> str:
    """Render metadata as a row of pills."""
    if not meta:
        return ""
    parts: list[str] = []

    intent = meta.get("intent", "")
    if intent:
        parts.append(f'<span class="pill intent-{intent}">{intent.replace("_", " ")}</span>')

    model = meta.get("model_used", "")
    if model:
        parts.append(f'<span class="pill pill-info">🤖 {model}</span>')

    latency = meta.get("latency_ms", 0)
    if latency:
        parts.append(_latency_pill(latency))

    sources = meta.get("sources", [])
    if sources:
        parts.append(
            f'<span class="pill pill-primary">📎 {len(sources)} source{"s" if len(sources) != 1 else ""}</span>'
        )

    tools = meta.get("tools_used", [])
    if tools:
        parts.append(f'<span class="pill pill-warning">🔧 {", ".join(tools)}</span>')

    # PII guardrail
    pii_check = next(
        (
            c
            for c in meta.get("guardrails", {}).get("input", {}).get("checks", [])
            if c.get("name") == "pii_detection"
        ),
        None,
    )
    if pii_check and pii_check.get("entities"):
        parts.append('<span class="pill pill-warning">🛡️ PII redacted</span>')

    return "".join(parts)


def _render_sources(sources: list) -> str:
    """Render source citations as chips."""
    if not sources:
        return ""
    chips = []
    for s in sources:
        filename = s.get("filename", "unknown")
        score = s.get("relevance_score", 0) or s.get("score", 0)
        chips.append(
            f'<span class="source-chip">📄 {filename} '
            f'<span style="color:var(--text-mute)">· {float(score):.0%}</span></span>'
        )
    return (
        '<div style="margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid var(--border)">'
        '<div class="section-title">Sources</div>'
        + "".join(chips)
        + "</div>"
    )


def _render_chat_bubble(role: str, content_html: str) -> None:
    """Render a chat bubble aligned by role: user on the right, assistant on the left.

    Uses st.columns (reliable across Streamlit versions) instead of CSS hacks
    on st.chat_message, which is not stylable for horizontal alignment.
    """
    if role == "user":
        # spacer left, bubble right
        spacer, bubble = st.columns([1, 4], gap="small")
        with bubble:
            st.markdown(
                f'<div class="bubble bubble-user">{content_html}</div>',
                unsafe_allow_html=True,
            )
    else:
        # bubble left, spacer right
        bubble, spacer = st.columns([4, 1], gap="small")
        with bubble:
            st.markdown(
                f'<div class="bubble bubble-assistant">{content_html}</div>',
                unsafe_allow_html=True,
            )


def _process_prompt(prompt: str, model: str) -> None:
    """Stream a prompt to the orchestrator and render thinking + content live."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    _render_chat_bubble("user", prompt)

    # Reserve a left-aligned column for the assistant bubble
    bubble_col, _ = st.columns([4, 1], gap="small")
    with bubble_col:
        # Live containers — we mutate their content as events arrive
        thinking_box = st.empty()
        content_box = st.empty()
        content_box.markdown(
            '<div class="bubble bubble-assistant">'
            '<div class="typing-indicator">'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '<span style="margin-left:0.5rem">Thinking</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        thinking_text = ""
        content_text = ""
        intent_value = "direct_chat"
        model_used = model
        sources: list = []
        latency_ms = 0.0
        metadata_log: dict = {}

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ][-10:]

        def _render_bubble() -> str:
            """Render current thinking + content as a single bubble HTML."""
            think_html = ""
            if thinking_text.strip():
                # Escape HTML in reasoning
                safe_think = (
                    thinking_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                think_html = (
                    '<details class="reasoning" open style="margin-bottom:0.85rem; '
                    'padding:0.7rem 0.9rem; background:rgba(102,126,234,0.06); '
                    'border:1px solid rgba(102,126,234,0.2); border-radius:10px;">'
                    '<summary style="cursor:pointer; color:var(--text-dim); '
                    'font-size:0.78rem; font-weight:600; user-select:none;">'
                    '🧠 Reasoning</summary>'
                    f'<pre style="margin:0.6rem 0 0 0; white-space:pre-wrap; '
                    f'font-family:JetBrains Mono,monospace; font-size:0.78rem; '
                    f'color:var(--text-dim); line-height:1.5;">{safe_think}</pre>'
                    '</details>'
                )
            # Streamed markdown: just show plain-text progressive content
            safe_content = (
                content_text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            return (
                f'<div class="bubble bubble-assistant">{think_html}'
                f'<div class="bubble-content">{safe_content}'
                '<span class="cursor-blink">▌</span></div></div>'
            )

        def _render_meta() -> str:
            """Render metadata pills + sources."""
            if not (intent_value or latency_ms or sources):
                return ""
            parts: list[str] = []
            if intent_value:
                parts.append(
                    f'<span class="pill intent-{intent_value}">'
                    f'{intent_value.replace("_", " ")}</span>'
                )
            if model_used:
                parts.append(f'<span class="pill pill-info">🤖 {model_used}</span>')
            if latency_ms:
                kind = "success" if latency_ms < 5000 else ("info" if latency_ms < 15000 else "warning")
                icon = "⚡" if latency_ms < 5000 else ("⏱" if latency_ms < 15000 else "🐢")
                parts.append(f'<span class="pill pill-{kind}">{icon} {latency_ms:.0f}ms</span>')
            for s in sources:
                filename = s.get("filename", "unknown")
                score = s.get("relevance_score", 0) or s.get("score", 0)
                parts.append(
                    f'<span class="source-chip">📄 {filename} '
                    f'<span style="color:var(--text-mute)">· {float(score):.0%}</span></span>'
                )
            return (
                '<div style="margin-top:0.85rem; display:flex; flex-wrap:wrap; '
                f'gap:0.4rem;">{"".join(parts)}</div>'
            )

        try:
            with requests.post(
                STREAM_URL,
                json={
                    "message": prompt,
                    "history": history if history else None,
                    "model": model,
                },
                stream=True,
                timeout=300,
            ) as r:
                r.raise_for_status()
                # SSE format: lines beginning with "data: " then blank line
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if not payload:
                        continue
                    try:
                        ev = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    kind = ev.get("event")
                    if kind == "metadata":
                        intent_value = ev.get("intent", intent_value)
                        model_used = ev.get("model_used", model_used)
                    elif kind == "thinking":
                        thinking_text += ev.get("delta", "")
                        thinking_box.markdown(
                            f'<div class="bubble bubble-assistant">'
                            f'<div class="typing-indicator">'
                            f'<div class="typing-dot"></div>'
                            f'<div class="typing-dot"></div>'
                            f'<div class="typing-dot"></div>'
                            f'<span style="margin-left:0.5rem">Reasoning…</span>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                    elif kind == "content":
                        content_text += ev.get("delta", "")
                        content_box.markdown(_render_bubble(), unsafe_allow_html=True)
                    elif kind == "done":
                        content_text = ev.get("answer", content_text)
                        intent_value = ev.get("intent", intent_value)
                        sources = ev.get("sources", sources)
                        latency_ms = ev.get("latency_ms", latency_ms)
                        model_used = ev.get("model_used", model_used)
                        metadata_log = ev
                        # Final render without blinking cursor
                        final_html = _render_bubble().replace(
                            '<span class="cursor-blink">▌</span>', ""
                        )
                        content_box.markdown(final_html, unsafe_allow_html=True)
                        # Render metadata pills below
                        st.markdown(_render_meta(), unsafe_allow_html=True)
                    elif kind == "error":
                        content_box.markdown(
                            f'<div class="bubble bubble-assistant">'
                            f'<div class="bubble-content">**Error:** '
                            f'{ev.get("detail", "unknown")}</div></div>',
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            content_box.markdown(
                f'<div class="bubble bubble-assistant">'
                f'<div class="bubble-content">**Connection error:** '
                f'{str(e)[:200]}</div></div>',
                unsafe_allow_html=True,
            )

        # Excel download button if response contains report link
        report_match = re.search(r"/v1/reports/([\w._-]+\.xlsx)", content_text)
        if report_match:
            report_filename = report_match.group(1)
            try:
                dl_url = f"{API_URL}/v1/reports/{report_filename}"
                dl_resp = requests.get(dl_url, timeout=10)
                if dl_resp.status_code == 200:
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=dl_resp.content,
                        file_name=report_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True,
                    )
            except Exception:
                pass

        # Persist to session state for history re-render
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": content_text,
                "metadata": metadata_log or {
                    "intent": intent_value,
                    "model_used": model_used,
                    "latency_ms": latency_ms,
                },
                "sources": sources,
                "thinking": thinking_text,
            }
        )
        if metadata_log:
            st.session_state.metadata_log.append(metadata_log)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        """
<div style="padding: 0.5rem 0 1rem 0;">
  <div style="font-size:1.1rem; font-weight:700; color:var(--text); display:flex; align-items:center; gap:0.5rem;">
    <span style="font-size:1.35rem">✨</span> AI Platform
  </div>
  <div style="color:var(--text-dim); font-size:0.78rem; margin-top:0.25rem;">
    On-premise · CPU · Production
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
    available_models = _fetch_available_models()
    default_idx = (
        available_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in available_models else 0
    )
    model = st.selectbox(
        "Model", available_models, index=default_idx, label_visibility="collapsed"
    )

    st.markdown('<div class="section-title">Knowledge Base</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload", type=["txt", "md", "pdf", "docx", "xlsx"], label_visibility="collapsed"
    )
    if uploaded and st.button("📥 Ingest", type="primary", use_container_width=True):
        with st.spinner("Ingesting..."):
            try:
                r = requests.post(
                    f"{API_URL}/v1/rag/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=60,
                )
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"✅ {data.get('filename')} → {data.get('chunks', 0)} chunks")
                else:
                    st.error(f"Error: {r.text[:150]}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    try:
        r = requests.get(f"{API_URL}/v1/rag/status", timeout=3)
        docs = r.json().get("documents_indexed", 0)
        st.metric("Documents Indexed", docs)
    except Exception:
        st.metric("Documents Indexed", "—")

    st.markdown('<div class="section-title">System</div>', unsafe_allow_html=True)
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        online = r.status_code == 200
    except Exception:
        online = False

    if online:
        st.markdown(
            """
<div class="glass-card" style="padding:0.55rem 0.85rem;">
  <span class="status-dot status-online"></span>
  <span style="color:var(--text); font-weight:500; font-size:0.88rem;">Backend online</span>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="glass-card" style="padding:0.55rem 0.85rem;">
  <span class="status-dot status-offline"></span>
  <span style="color:var(--text); font-weight:500; font-size:0.88rem;">Backend offline</span>
</div>
""",
            unsafe_allow_html=True,
        )

    # Last response
    if st.session_state.metadata_log:
        st.markdown('<div class="section-title">Last Response</div>', unsafe_allow_html=True)
        meta = st.session_state.metadata_log[-1]
        meta_html = _render_message_metadata(meta)
        if meta_html:
            st.markdown(
                f'<div class="glass-card" style="display:flex; flex-wrap:wrap; gap:0.35rem;">{meta_html}</div>',
                unsafe_allow_html=True,
            )
        sources_html = _render_sources(meta.get("sources", []))
        if sources_html:
            st.markdown(sources_html, unsafe_allow_html=True)

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.metadata_log = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.caption("v1.0 · [API Docs](http://localhost:8080/docs)")


# ============================================================================
# MAIN AREA
# ============================================================================
st.markdown(
    """
<div class="hero">
  <h1>Ask anything, intelligently</h1>
  <p>Chat · RAG · Agent — auto-routed by the orchestrator</p>
</div>
""",
    unsafe_allow_html=True,
)


# Empty state with quick actions
if not st.session_state.messages:
    quick_prompts = [
        ("📊", "Analisis laporan keuangan", "dari dokumen yang sudah di-upload"),
        ("💰", "Hitung rasio keuangan", "Revenue 500M, Net Income 100M, Assets 1B, Equity 600M"),
        ("📥", "Buatkan Excel report", "analisis keuangan lengkap dengan rasio"),
        ("💡", "Explain machine learning", "in simple terms"),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(quick_prompts):
        with cols[i % 2]:
            if st.button(
                f"{icon}  {title}",
                key=f"qa_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = f"{title} {desc}"
                st.rerun()
            st.caption(desc)


# Display chat history
for msg in st.session_state.messages:
    meta_html = ""
    if msg["role"] == "assistant" and msg.get("metadata"):
        meta_html = _render_message_metadata(msg["metadata"])
    sources_html = _render_sources(msg.get("sources", [])) if msg["role"] == "assistant" else ""

    content = msg["content"]
    full_html = f'<div class="bubble-content">{content}</div>'
    if meta_html:
        full_html += (
            f'<div style="margin-top:0.85rem; display:flex; flex-wrap:wrap; gap:0.4rem;">{meta_html}</div>'
        )
    if sources_html:
        full_html += sources_html

    _render_chat_bubble(msg["role"], full_html)


# Process pending quick-action prompt
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    _process_prompt(prompt, model)
elif prompt := st.chat_input("Ask anything…"):
    _process_prompt(prompt, model)
