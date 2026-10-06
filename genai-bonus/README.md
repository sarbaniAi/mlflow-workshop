# Bonus Module — MLflow for GenAI & LLMOps

**This is an optional, advanced bonus module.** It's not part of the core half-day MLflow seminar (that's in `../notebook/`) — it's for students who want to go further, into MLflow's GenAI side: tracing, agent evaluation, and LLM-judge scorer feedback.

## What's here

| File | What it is |
|---|---|
| [`chatbot.py`](chatbot.py) | Shared chat logic (`build_context()` + `chat()`), instrumented with `@mlflow.trace`. Both `app.py` and the notebook call the *same* functions, so what you evaluate is exactly what runs in the app. |
| [`app.py`](app.py) | A real, runnable Streamlit chatbot UI, instrumented end-to-end with MLflow (tracing + autolog). |
| [`GenAI_LLMOps_Bonus_CHRIST_University.ipynb`](GenAI_LLMOps_Bonus_CHRIST_University.ipynb) | The companion analysis notebook: tracing basics, building an eval dataset, `mlflow.genai.evaluate()` with built-in LLM-judge scorers, reading judge rationale, and writing a custom judge with `make_judge()`. |
| `build_notebook.py` | Generates the `.ipynb` above. Edit this and re-run it to change the notebook. |
| `requirements.txt` | Extra packages this module needs beyond the main workshop (`openai`, `streamlit`). |

The example scenario: an OpenAI-powered chatbot (`gpt-4o-mini`) that answers MLflow questions, with **`gpt-4o` acting as the LLM judge** that scores its answers.

## Setup

```bash
pip install -r requirements.txt
```

### Your OpenAI API key — read this before running anything

This module makes real calls to OpenAI's API, so it needs **your own** `OPENAI_API_KEY`.

- **Never** paste a real API key into a notebook cell that gets saved, into `app.py`, into a GitHub commit, or into a chat with an AI assistant.
- Set it as an environment variable before you start Jupyter or Streamlit:

  ```bash
  export OPENAI_API_KEY=sk-...          # macOS/Linux
  set OPENAI_API_KEY=sk-...             # Windows (Command Prompt)
  ```

- If you forget, the notebook's setup cell falls back to a hidden `getpass()` prompt — it keeps the key only in memory for that session, never written to disk or to the notebook file.
- `app.py` will show a clear on-screen error (not a crash) if the key isn't set.

## Running the chatbot app

```bash
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Chat with it — every turn is traced automatically.

To see the traces it's logging, in a **second terminal, in this same folder**:

```bash
mlflow ui --backend-store-uri sqlite:///genai_mlflow.db
```

Open the **Traces** tab and click into any trace to see the nested spans: the custom `chat`/`build_context` spans plus the auto-instrumented OpenAI call.

## Running the notebook

```bash
jupyter notebook GenAI_LLMOps_Bonus_CHRIST_University.ipynb
```

Run cells top to bottom. Section 0 sets up your API key and a separate MLflow tracking store (`genai_mlflow.db`) from the one `app.py` uses by default — set `MLFLOW_TRACKING_URI`/experiment the same way in both if you want the app's live chats and the notebook's evaluation runs to land in one place.

## What this covers that the main workshop doesn't

The main workshop (`../notebook/`) is classical ML: tracking, model logging, the registry, `mlflow.evaluate()`, and projects — all with open-source MLflow, no API key, no network calls.

This module is the GenAI/LLMOps analogue of that same lifecycle:

- **Tracing** instead of plain param/metric logging — every step's actual input/output, not just final numbers (`mlflow.openai.autolog()` + `@mlflow.trace`)
- **`mlflow.genai.evaluate()`** instead of `mlflow.evaluate()` — scored by LLM judges (`Correctness`, `Guidelines`, `Safety`, `RelevanceToQuery`) since there's rarely one "correct" string for an LLM to match
- **Per-row judge rationale** — *why* a judge scored what it scored, not just a number
- **Custom judges** via `mlflow.genai.judges.make_judge()` — write your own grading criteria in plain English

See the notebook's final section for the full side-by-side comparison with the main workshop's concepts.

## A known limitation

The `mlflow.genai.evaluate()` and judge cells make real calls to the OpenAI API and genuinely need your own working key and network access — they can't be pre-run or cached for you. Everything up to that point (the chatbot itself, its tracing, the dataset construction, the scorer/judge setup) has been tested end-to-end against the real installed packages; only the live LLM calls themselves are left for you to run with your own key.
