"""
GenAI / LLMOps bonus module — Streamlit chatbot instrumented end-to-end with MLflow.

Run it with:
    export OPENAI_API_KEY=sk-...        # never hardcode this; see README.md
    streamlit run app.py

Every turn is traced: build_context() (a toy retrieval step) and chat()
(the chain that calls OpenAI) are both wrapped with @mlflow.trace in
chatbot.py, and mlflow.openai.autolog() below additionally captures the
raw OpenAI call as its own nested span -- so one trace per turn shows the
full path: retrieval -> model call -> answer.

View the traces by running, in another terminal, from this same folder:
    mlflow ui --backend-store-uri sqlite:///genai_mlflow.db
"""

import os

import mlflow
import streamlit as st

import chatbot

# A separate tracking DB from the main workshop notebook's mlflow.db, so the
# two modules' runs/traces don't mix.
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(os.path.dirname(__file__), 'genai_mlflow.db')}")
mlflow.set_experiment("genai-bonus-chatbot")
mlflow.openai.autolog()

st.set_page_config(page_title="MLflow GenAI Bonus — Chatbot", page_icon="💬")
st.title("💬 MLflow-traced chatbot")
st.caption(
    "Every message below is traced end-to-end with MLflow. "
    "Run `mlflow ui --backend-store-uri sqlite:///genai_mlflow.db` in this folder to watch the traces land."
)

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "**`OPENAI_API_KEY` is not set.**\n\n"
        "Set it as an environment variable before launching Streamlit, e.g.\n\n"
        "```bash\nexport OPENAI_API_KEY=sk-...\nstreamlit run app.py\n```\n\n"
        "Never type a real API key into this app, a notebook cell, or a chat with an AI assistant."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": ..., "content": ...}

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask about MLflow...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Pass prior turns only (chatbot.chat appends the new question itself).
                answer = chatbot.chat(question, history=st.session_state.history[:-1])
            except RuntimeError as exc:
                answer = f"⚠️ {exc}"
        st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
