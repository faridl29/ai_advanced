"""Streamlit Chat UI — Unified AI Platform Interface.

Single chat interface that:
- Auto-routes via orchestrator (or manual mode)
- Shows metadata: intent, sources, tools, guardrail status
- Maintains conversation history
- Supports file upload for RAG
"""
import json
import time

import requests
import streamlit as st

API_URL = "http://app:8080"

st.set_page_config(
    page_title="AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS — Premium dark theme
# =============================================================================
st.markdown("""
<style>
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --bg-dark: #0f0f1a;
        --bg-card: rgba(30,30,50,0.7);
        --text: #e2e8f0;
        --text-dim: #94a3b8;
    }

    .stApp {
        background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 50%, #16213e 100%);
    }

    [data-testid="stSidebar"] {
        background: rgba(15,15,26,0.95);
        border-right: 1px solid rgba(102,126,234,0.2);
    }

    .stChatMessage {
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    h1 {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    .metadata-card {
        background: var(--bg-card);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }

    .intent-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .intent-direct_chat { background: rgba(34,197,94,0.2); color: #22c55e; }
    .intent-rag_query { background: rgba(59,130,246,0.2); color: #3b82f6; }
    .intent-agent_task { background: rgba(168,85,247,0.2); color: #a855f7; }
    .intent-blocked { background: rgba(239,68,68,0.2); color: #ef4444; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata_log" not in st.session_state:
    st.session_state.metadata_log = []


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("🤖 AI Platform")
    st.caption("Production-Ready • On-Premise • CPU")
    st.divider()

    # Mode selection
    mode = st.radio(
        "Routing Mode",
        ["🧠 Auto (Orchestrator)", "💬 Force: Chat", "📄 Force: RAG", "🤖 Force: Agent"],
        index=0,
        help="Auto mode lets the AI decide. Force mode overrides.",
    )

    force_map = {
        "🧠 Auto (Orchestrator)": None,
        "💬 Force: Chat": "direct_chat",
        "📄 Force: RAG": "rag_query",
        "🤖 Force: Agent": "agent_task",
    }
    force_intent = force_map[mode]

    st.divider()

    # Model selection
    model = st.selectbox("Model", ["phi3", "qwen2.5", "gemma3", "llama3.2"], index=0)

    st.divider()

    # Document upload (for RAG)
    st.subheader("📄 Knowledge Base")
    uploaded = st.file_uploader("Upload document", type=["txt", "md", "pdf", "docx"])
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
                    st.error(f"Error: {r.text}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # RAG status
    try:
        r = requests.get(f"{API_URL}/v1/rag/status", timeout=3)
        data = r.json()
        docs = data.get("documents_indexed", 0)
        st.metric("Docs Indexed", docs)
    except Exception:
        pass

    st.divider()

    # Health
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ Backend Online")
        else:
            st.error("❌ Backend Error")
    except Exception:
        st.error("❌ Backend Offline")

    # Metadata from last response
    if st.session_state.metadata_log:
        st.divider()
        st.subheader("📊 Last Response")
        meta = st.session_state.metadata_log[-1]

        # Intent badge
        intent = meta.get("intent", "unknown")
        st.markdown(f'<span class="intent-badge intent-{intent}">{intent}</span>', unsafe_allow_html=True)

        # Latency
        st.caption(f"⏱️ {meta.get('latency_ms', 0):.0f}ms | Model: {meta.get('model_used', 'N/A')}")

        # Sources
        sources = meta.get("sources", [])
        if sources:
            st.caption("📎 Sources:")
            for s in sources:
                st.caption(f"  • {s.get('filename', '?')} (score: {s.get('relevance_score', 0):.2f})")

        # Tools
        tools = meta.get("tools_used", [])
        if tools:
            st.caption(f"🔧 Tools: {', '.join(tools)}")

        # Guardrails
        guardrails = meta.get("guardrails", {})
        input_g = guardrails.get("input", {})
        if input_g:
            checks = input_g.get("checks", [])
            pii_check = next((c for c in checks if c.get("name") == "pii_detection"), None)
            if pii_check and pii_check.get("entities"):
                st.caption(f"🛡️ PII detected & redacted")

    # Clear chat
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.metadata_log = []
        st.rerun()

    st.caption("v1.0 • [API Docs](http://localhost:8080/docs)")


# =============================================================================
# MAIN CHAT AREA
# =============================================================================
st.header("🤖 AI Platform")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources inline if available
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.caption(f"• {s.get('filename', '?')}")

# Chat input
if prompt := st.chat_input("Ask anything... (auto-routes to Chat/RAG/Agent)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call orchestrator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            try:
                # Build history for context
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # Exclude current
                ][-10:]  # Last 10 messages

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

                if r.status_code == 200:
                    data = r.json()
                    answer = data.get("answer", "No response")
                    sources = data.get("sources", [])
                    intent = data.get("intent", "unknown")
                    latency = data.get("latency_ms", 0)

                    # Store metadata
                    st.session_state.metadata_log.append(data)
                else:
                    answer = f"Error ({r.status_code}): {r.text[:200]}"
                    sources = []
                    intent = "error"
                    latency = (time.time() - start) * 1000

            except Exception as e:
                answer = f"Connection error: {e}"
                sources = []
                intent = "error"
                latency = (time.time() - start) * 1000

        # Display response
        st.markdown(answer)

        # Show intent badge
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"🏷️ {intent}")
        with col2:
            st.caption(f"⏱️ {latency:.0f}ms")
        with col3:
            if sources:
                st.caption(f"📎 {len(sources)} sources")

        # Show sources if RAG
        if sources:
            with st.expander("📎 View Sources"):
                for s in sources:
                    st.caption(f"• {s.get('filename', '?')} — score: {s.get('relevance_score', 0):.3f}")

    # Store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
