"""
app.py

Streamlit front-end for the Custom AI Chatbot with Memory (Project 1,
DecodeLabs Generative AI Internship).

Key architectural concept:
    Streamlit reruns this ENTIRE script top-to-bottom on every user
    interaction (every message sent, every button click). A plain Python
    list defined here would be wiped and recreated on every rerun --
    which would defeat the whole point of "memory."

    The fix is `st.session_state`: a dict-like object Streamlit preserves
    across reruns for as long as the browser session stays open. Our
    conversation_history list lives there instead of as a bare variable.
"""

import streamlit as st

from chat_engine import get_claude_response

st.set_page_config(page_title="Claude Chatbot with Memory", page_icon="🤖")
st.title("🤖 Custom AI Chatbot with Memory")
st.caption("Project 1 — DecodeLabs Generative AI Internship | Powered by Claude")

# --------------------------------------------------------------------------
# 1. Initialize persistent memory (this block only runs ONCE per session,
#    thanks to the "not in st.session_state" guard)
# --------------------------------------------------------------------------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# --------------------------------------------------------------------------
# 2. Re-render the existing conversation on every rerun
#    (Streamlit has no persistent DOM -- we redraw everything, every time)
# --------------------------------------------------------------------------
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------------------------------
# 3. Capture new user input
# --------------------------------------------------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # --- Structural Validation Gate ---
    # Never let an empty/whitespace-only string reach the API: Claude
    # returns a 400 Bad Request for it, which would crash the app if
    # left unguarded.
    cleaned_input = user_input.strip()

    if not cleaned_input:
        st.warning("Please enter a non-empty message.")
    else:
        # Step 1: Append the user's turn to memory and display it
        st.session_state.conversation_history.append(
            {"role": "user", "content": cleaned_input}
        )
        with st.chat_message("user"):
            st.markdown(cleaned_input)

        # Step 2: Send the FULL history to Claude and display the reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = get_claude_response(
                        st.session_state.conversation_history
                    )
                except (ValueError, RuntimeError) as exc:
                    reply = f"⚠️ {exc}"
            st.markdown(reply)

        # Step 3: Append Claude's reply to memory so the NEXT turn has
        # full context of this exchange too.
        st.session_state.conversation_history.append(
            {"role": "assistant", "content": reply}
        )

# --------------------------------------------------------------------------
# 4. Sidebar utility: reset the conversation
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Session Controls")
    st.write(f"Messages in memory: {len(st.session_state.conversation_history)}")
    if st.button("🗑️ Clear conversation"):
        st.session_state.conversation_history = []
        st.rerun()
