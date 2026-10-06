"""
Shared chatbot logic for the MLflow GenAI / LLMOps bonus module.

Used by both:
  - app.py              the real Streamlit chat UI students run
  - build_notebook.py   the companion analysis notebook, via predict_fn

Keeping the actual chat logic in one module means the exact code path that
gets evaluated in the notebook (tracing, scorers, judges) is the exact code
path that runs in the deployed app -- not a lookalike copy of it.

SECURITY: the OpenAI API key is read ONLY from the OPENAI_API_KEY
environment variable at runtime. It is never hardcoded here, never
accepted as a function argument from a notebook cell, and never written
to disk or committed. See README.md for how to set it before running.
"""

from __future__ import annotations

import os

import mlflow
from mlflow.entities import SpanType
from openai import OpenAI

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful teaching assistant for a university MLflow workshop. "
    "Answer clearly and concisely, in two or three sentences. If a question "
    "is unrelated to MLflow, machine learning, or the workshop, say so "
    "politely and redirect the student back to those topics."
)

# Tiny hardcoded FAQ standing in for a real retrieval step (e.g. a vector
# store). Keeping it as data makes build_context's tracing behavior
# realistic without needing an external index for the workshop.
_FAQ = {
    "mlflow": (
        "MLflow is an open-source platform for managing the ML lifecycle: "
        "tracking, models, the model registry, evaluation, and projects."
    ),
    "tracing": (
        "MLflow Tracing captures the inputs, outputs, and timing of each "
        "step in an LLM application as nested spans, so you can see exactly "
        "what the model saw and produced at every step."
    ),
    "registry": (
        "The MLflow Model Registry versions and stages models (for example "
        "via aliases like @champion) so deployment doesn't mean re-copying "
        "files around by hand."
    ),
    "evaluation": (
        "mlflow.genai.evaluate() runs a set of scorers -- including "
        "LLM-judge scorers like Correctness and Safety -- over a dataset "
        "of inputs/outputs and reports pass/fail plus per-row rationale."
    ),
}

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazily build the OpenAI client, reading the key from the environment only.

    Raises a clear error if OPENAI_API_KEY isn't set, rather than silently
    failing or prompting for a key to be typed in.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set.\n\n"
                "Set it as an environment variable before running, e.g.\n"
                "    export OPENAI_API_KEY=sk-...\n\n"
                "Never paste a real API key into a notebook cell, a Streamlit "
                "widget, or this chat -- only an environment variable."
            )
        _client = OpenAI(api_key=api_key)
    return _client


@mlflow.trace(span_type=SpanType.RETRIEVER)
def build_context(question: str) -> str:
    """Toy 'retrieval' step over a tiny FAQ.

    Stands in for a real vector-store lookup in a RAG app. Traced with
    span_type=RETRIEVER so it shows up as a distinct step from the LLM call
    in the trace viewer -- in a production app this is where a vector DB
    query would go.
    """
    q = question.lower()
    hits = [text for key, text in _FAQ.items() if key in q]
    return "\n".join(hits) if hits else "No specific context found; answer from general knowledge."


@mlflow.trace(span_type=SpanType.CHAIN)
def chat(question: str, history: list[dict] | None = None) -> str:
    """Answer one user turn: build context, then call the chat model.

    `history` is a list of {"role": ..., "content": ...} dicts from prior
    turns (OpenAI message format), oldest first, excluding the current
    question. This is the single function shared by the Streamlit app and
    the notebook's predict_fn, so both are exercising the same code.
    """
    context = build_context(question)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    )

    client = get_client()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content
