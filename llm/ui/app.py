"""
DevOps AI Assistant – Streamlit chat interface for the RAG pipeline.

Connects to the FastAPI backend at ``/ask`` and renders a persistent
chat conversation with streaming answers and source attribution.
"""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DevOps AI Assistant",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Rounded chat bubbles */
    .stChatMessage { border-radius: 12px; }

    /* Subtle sources expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #888;
    }

    /* Sidebar header area */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("API_KEY", "")
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False


# ── Helpers ──────────────────────────────────────────────────────────────────
def _stream_response(text: str):
    """Stream text preserving markdown — yield line by line with word
    streaming within each line."""
    lines = text.split("\n")
    for line in lines:
        if line.strip() == "":
            yield "\n"
            time.sleep(0.05)
        elif line.startswith("#") or line.startswith("-") or line.startswith("*"):
            for word in line.split():
                yield word + " "
                time.sleep(0.02)
            yield "\n"
        else:
            for word in line.split():
                yield word + " "
                time.sleep(0.02)
            yield "\n"
            time.sleep(0.05)


def _fetch_health() -> dict | None:
    """Call GET /health and return the JSON, or None on failure."""
    try:
        resp = requests.get(f"{st.session_state.api_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _parse_sources(answer: str) -> tuple[str, list[str]]:
    """Split an answer into (body, sources_list).

    Handles both ``Sources:`` and ``**Sources:**`` markers.
    """
    for marker in ["**Sources:**", "Sources:"]:
        if marker in answer:
            parts = answer.split(marker, 1)
            body = parts[0].strip()
            raw = parts[1].strip()
            sources = [s.strip() for s in raw.replace("**", "").split(",")
                       if s.strip()]
            return body, sources
    return answer, []


def _ask_backend(question: str) -> dict:
    """
    POST /ask and return the parsed JSON response.

    Raises requests exceptions on failure so the caller can map them to
    user-friendly chat messages.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if st.session_state.api_key:
        headers["x-api-key"] = st.session_state.api_key

    resp = requests.post(
        f"{st.session_state.api_url}/ask",
        json={"question": question},
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 DevOps AI Assistant")
    st.caption("v1.0.0")
    st.divider()

    # API key
    key_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="key-abc123",
    )
    st.session_state.api_key = key_input
    if st.session_state.api_key:
        st.success("✅ API key set")
    else:
        st.warning("❌ No API key set")

    # API URL
    url_input = st.text_input(
        "API URL",
        value=st.session_state.api_url,
    )
    st.session_state.api_url = url_input.rstrip("/")

    st.divider()

    # Health check
    st.markdown("**Backend Status**")
    health = _fetch_health()
    if health:
        st.markdown("Status: 🟢 Online")
        st.markdown(f"Model: `{health.get('model', 'unknown')}`")
        idx = health.get("index_loaded", False)
        st.markdown(f"Index: {'✅ Loaded' if idx else '❌ Not loaded'}")
    else:
        st.markdown("Status: 🔴 Offline")

    st.divider()

    # Clear conversation
    if st.button("🗑️ Clear conversation", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("DevOps AI Assistant • Open Source")


# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        # Re-render sources expander for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            srcs = msg["sources"]
            with st.expander(f"📂 Sources ({len(srcs)} files)"):
                for src in srcs:
                    st.code(src, language=None)


# ── Chat input ───────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a DevOps question...",
    disabled=st.session_state.is_loading,
):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate assistant response
    st.session_state.is_loading = True
    with st.chat_message("assistant", avatar="🤖"):
        try:
            data = _ask_backend(prompt)
            answer_body, sources = _parse_sources(data["answer"])

            # Stream the answer word-by-word
            streamed = st.write_stream(_stream_response(answer_body))

            # Sources expander
            if sources:
                with st.expander(f"📂 Sources ({len(sources)} files)"):
                    for src in sources:
                        st.code(src, language=None)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_body,
                "sources": sources,
            })

        except requests.exceptions.ConnectionError:
            err = f"❌ Cannot reach the API at {st.session_state.api_url}. Is the server running?"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 401:
                err = "🔑 Invalid or missing API key. Set it in the sidebar."
            elif status == 429:
                err = "⏳ Rate limit hit. Please wait a moment and try again."
            else:
                err = f"❌ API error (HTTP {status}). Please try again."
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

        except requests.exceptions.Timeout:
            err = "⏱️ The request timed out. The model may be under load."
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

        except Exception as exc:
            err = f"❌ Unexpected error: {exc}"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

        finally:
            st.session_state.is_loading = False
