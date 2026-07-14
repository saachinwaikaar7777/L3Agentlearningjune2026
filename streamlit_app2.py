"""
ShopEase — Streamlit chat UI for real multi-user simulation
=============================================================

This reuses the exact same agent, tools, and SqliteSaver checkpointer
built for the M1-M4 case study (see shopease_agent.py), wrapped in an
interactive chat UI so you can simulate *different real users* by:

  - Opening this app in two different browser tabs/windows and typing a
    different "Shopper ID" in each -> proves session isolation (M2)
    with real concurrent Streamlit sessions, not just a scripted test.
  - Closing a tab (or stopping/restarting `streamlit run`) and coming
    back later with the *same* Shopper ID -> proves durability (M3),
    because the cart and conversation live in shopease_sessions.db, not
    in Streamlit's in-memory session state.

One Streamlit server process is shared by every browser tab that
connects to it — exactly like one production server handling many
concurrent shoppers — so this is a faithful simulation, not a mock.

Requirements: everything in requirements.txt (streamlit is already
listed there).

Run:
    streamlit run streamlit_app.py

Then open the printed http://localhost:8501 URL in two separate browser
windows (or one normal + one incognito window, so cookies/session state
don't get shared) to simulate two different shoppers at once.
"""

import sqlite3
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from shopease_agent import (
    TOOLS,
    SYSTEM_PROMPT,
    make_llm,
    CATALOG,
    ORDER_STATUS,
    DISCOUNT_CODES,
    CARTS,
    DB_PATH,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent as create_agent

st.set_page_config(page_title="ShopEase Assistant", page_icon="🛍️", layout="wide")


# ---------------------------------------------------------------------------
# Shared, persistent backend — built ONCE per running Streamlit process and
# reused by every browser tab/user. This mirrors production: one server,
# many concurrent shoppers, isolated by thread_id, durable via SQLite.
# st.cache_resource ensures this only runs once no matter how many users
# connect or how many times Streamlit reruns the script per interaction.
# ---------------------------------------------------------------------------

@st.cache_resource
def get_checkpointer():
    # check_same_thread=False: Streamlit runs each browser session in its
    # own thread within the same process, so the sqlite connection must be
    # safe to share across threads.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


@st.cache_resource
def get_agent():
    return create_agent(make_llm(), TOOLS, prompt=SYSTEM_PROMPT, checkpointer=get_checkpointer())


def get_history(thread_id: str):
    """Pull this shopper's durable message history straight from the checkpointer,
    so a page refresh (or a new tab with the same Shopper ID) shows the real
    persisted conversation rather than anything cached in the browser/session."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_agent().get_state(config)
    return snapshot.values.get("messages", []) if snapshot else []


# ---------------------------------------------------------------------------
# Sidebar — "log in" as a shopper, see the demo catalog, see your live cart
# ---------------------------------------------------------------------------

st.sidebar.title("🛍️ ShopEase")

if "thread_id" not in st.session_state:
    # Each new browser tab starts as its own anonymous shopper by default,
    # so opening the app twice already demonstrates isolation with zero setup.
    st.session_state.thread_id = f"guest-{uuid.uuid4().hex[:8]}"

st.sidebar.subheader("Shopper session")
shopper_id = st.sidebar.text_input(
    "Shopper ID (simulates a real logged-in user)",
    value=st.session_state.thread_id,
    help="Two browser tabs with DIFFERENT IDs = two independent shoppers "
         "(proves M2 isolation). The SAME ID after closing/reopening the "
         "tab = the same shopper, same cart (proves M3 durability).",
)
if shopper_id != st.session_state.thread_id:
    st.session_state.thread_id = shopper_id
    st.rerun()

col1, col2 = st.sidebar.columns(2)
if col1.button("🔀 New session", use_container_width=True):
    st.session_state.thread_id = f"guest-{uuid.uuid4().hex[:8]}"
    st.rerun()
if col2.button("🧹 Clear cart", use_container_width=True):
    CARTS.pop(st.session_state.thread_id, None)
    st.rerun()

st.sidebar.caption(f"Active thread_id: `{st.session_state.thread_id}`")

with st.sidebar.expander("📦 Demo catalog & test data"):
    st.write("**Products:**")
    for name, info in CATALOG.items():
        st.write(f"- {name.title()} — ${info['price']:.2f} — sizes: {info['sizes']}")
    st.write("**Order IDs to try:**", ", ".join(ORDER_STATUS.keys()))
    st.write("**Discount codes to try:**", ", ".join(DISCOUNT_CODES.keys()))

show_tool_calls = st.sidebar.checkbox("Show tool calls (debug view)", value=False)

st.sidebar.divider()
st.sidebar.subheader("🛒 Live cart (server-side truth)")
cart = CARTS.get(st.session_state.thread_id)
if not cart or not cart["items"]:
    st.sidebar.write("Empty")
else:
    subtotal = sum(i["price"] for i in cart["items"])
    for item in cart["items"]:
        size_note = f" (size {item['size']})" if item["size"] else ""
        st.sidebar.write(f"- {item['name'].title()}{size_note} — ${item['price']:.2f}")
    total = subtotal * (1 - cart["discount"])
    if cart["discount"]:
        st.sidebar.write(f"Discount: {cart['discount'] * 100:.0f}%")
    st.sidebar.write(f"**Total: ${total:.2f}**")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("ShopEase Shopping Assistant")
st.caption(
    f"You're chatting as **{st.session_state.thread_id}**. Open this app in "
    "another browser tab (or incognito window) with a different Shopper ID "
    "in the sidebar to prove carts and conversations stay isolated between "
    "different shoppers."
)

thread_id = st.session_state.thread_id
config = {"configurable": {"thread_id": thread_id}}
agent = get_agent()

# Render persisted history (this is what makes "come back later" work: it's
# read from shopease_sessions.db via the checkpointer, not from a Python
# list that would vanish on restart).
for msg in get_history(thread_id):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            with st.chat_message("assistant"):
                st.write(msg.content)
        if show_tool_calls and msg.tool_calls:
            with st.chat_message("assistant"):
                for tc in msg.tool_calls:
                    st.caption(f"🔧 called `{tc['name']}` with {tc['args']}")
    elif isinstance(msg, ToolMessage) and show_tool_calls:
        with st.chat_message("assistant"):
            st.caption(f"↩️ `{msg.name}` returned: {msg.content}")

user_input = st.chat_input("Ask about products, your cart, or an order...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config=config
            )
        st.write(result["messages"][-1].content)
    st.rerun()