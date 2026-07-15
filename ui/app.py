"""Streamlit Chat UI — Unified AI Platform Interface.

Premium dark-themed interface with:
- Auto-routing via orchestrator (or manual mode)
- Rich metadata display: intent, sources, tools, guardrail status
- Glassmorphism cards, animated typing indicator, color-coded pills
- Quick-action prompts in the empty state
- File upload for RAG
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://app:8080")
# Default to settings.default_model — single source of truth.
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:1.7b")


@st.cache_data(ttl=30)
def _fetch_available_models() -> list[str]:
    """Fetch list of available model names from the backend.

    Returns DEFAULT_MODEL as a safe fallback if the request fails.
    """
    try:
        r = requests.get(f"{API_URL}/v1/models", timeout=3)
        r.raise_for_status()
        data = r.json()
        names = [m.get("id", m.get("name")) for m in data.get("data", []) if m.get("id") or m.get("name")]
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


def _process_prompt(prompt: str, model: str, force_intent: str | None) -> None:
    """Send a prompt to the orchestrator and update session state."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    _render_chat_bubble("user", prompt)

    typing_html = (
        '<div class="bubble bubble-assistant">'
        '<div class="typing-indicator">'
        '<div class="typing-dot"></div>'
        '<div class="typing-dot"></div>'
        '<div class="typing-dot"></div>'
        '<span style="margin-left:0.5rem">Thinking</span>'
        '</div></div>'
    )
    typing_col, _ = st.columns([4, 1], gap="small")
    with typing_col:
        typing = st.empty()
        typing.markdown(typing_html, unsafe_allow_html=True)

        try:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ][-10:]

            r = requests.post(
                f"{API_URL}/v1/chat",
                json={
                    "message": prompt,
                    "history": history if history else None,
                    "model": model,
                    "force_intent": force_intent,
                },
                timeout=180,
            )

            typing.empty()

            if r.status_code == 200:
                data = r.json()
                answer = data.get("answer", "No response")
                sources = data.get("sources", [])

                st.markdown(
                    f'<div class="bubble-content">{answer}</div>',
                    unsafe_allow_html=True,
                )

                meta_html = _render_message_metadata(data)
                if meta_html:
                    st.markdown(
                        f'<div style="margin-top:0.85rem; display:flex; flex-wrap:wrap; gap:0.4rem;">{meta_html}</div>',
                        unsafe_allow_html=True,
                    )

                sources_html = _render_sources(sources)
                if sources_html:
                    st.markdown(sources_html, unsafe_allow_html=True)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "metadata": data, "sources": sources}
                )
                st.session_state.metadata_log.append(data)
            else:
                error_msg = f"**Error ({r.status_code})**\n\n```\n{r.text[:300]}\n```"
                st.markdown(
                    f'<div class="bubble-content">{error_msg}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "metadata": {}}
                )
        except Exception as e:
            typing.empty()
            error_msg = f"**Connection error**\n\n```\n{str(e)[:200]}\n```"
            st.markdown(
                f'<div class="bubble-content">{error_msg}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "metadata": {}}
            )


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

    st.markdown('<div class="section-title">Routing</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["🧠 Auto", "💬 Chat", "📄 RAG", "🤖 Agent"],
        index=0,
        label_visibility="collapsed",
    )
    force_map = {
        "🧠 Auto": None,
        "💬 Chat": "direct_chat",
        "📄 RAG": "rag_query",
        "🤖 Agent": "agent_task",
    }
    force_intent = force_map[mode]

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
        "Upload", type=["txt", "md", "pdf", "docx"], label_visibility="collapsed"
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
        ("💡", "Explain machine learning", "in simple terms"),
        ("🧮", "Calculate 15 * 37 + 42", "test the agent path"),
        ("✍️", "Write a haiku", "about Python and AI"),
        ("🌍", "Translate to Indonesian", "'Good morning, how are you?'"),
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
    _process_prompt(prompt, model, force_intent)
elif prompt := st.chat_input("Ask anything… (auto-routes to Chat / RAG / Agent)"):
    _process_prompt(prompt, model, force_intent)
