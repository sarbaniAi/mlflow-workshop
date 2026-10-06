# MLflow & the ML Lifecycle — CHRIST University Workshop

Materials for the half-day, hands-on MLflow seminar for M.Sc. Data Science PG students at CHRIST (Deemed to be University), Bengaluru (host: Dr. Rajesh Ramachandran, ~90 students).

Agenda: intro to MLflow → experiment tracking → model logging/versioning → the Model Registry → evaluation → MLflow Projects & deployment → a simple end-to-end workflow → industry use cases, best practices & MLOps skills → mini capstone.

## Contents

| Path | What it is |
|---|---|
| [`notebook/MLflow_Workshop_CHRIST_University.ipynb`](notebook/MLflow_Workshop_CHRIST_University.ipynb) | The hands-on notebook (Tracking, Models, Registry, Evaluation, Projects, and a mini capstone). Works locally or in Google Colab with open-source MLflow — no server, no account. Tested end-to-end against MLflow 3.16.1 (`jupyter nbconvert --execute`, 0 errors). |
| `notebook/build_notebook.py` | The script that generates the notebook (`nbformat`-based). Edit this and re-run it to change the notebook rather than hand-editing the `.ipynb`. |
| [`notebook/MLflow_Workshop_CHRIST_University_SOLUTIONS.ipynb`](notebook/MLflow_Workshop_CHRIST_University_SOLUTIONS.ipynb) | **Instructor-only answer key.** Identical to the student notebook except Section 7's four Mini Capstone `# TODO` cells are filled in with a complete, tested solution on the breast-cancer dataset. Don't hand this to students before the capstone. |
| `notebook/build_solution_notebook.py` | Generates the solutions notebook by reusing `build_notebook.py`'s Sections 0-6 verbatim and filling in Section 7. |
| `docs/setup-instructions.md` | Student-facing setup instructions (local Jupyter or Colab), sent out before the session. |
| `slides/` | Source for the slide deck (28 slides) — `deck.json` + one HTML file per slide. |
| [`genai-bonus/`](genai-bonus/) | **Optional bonus module:** MLflow for GenAI/LLMOps — tracing, agent evaluation, and LLM-judge scorers, built around a real OpenAI-powered Streamlit chatbot. Not part of the core half-day agenda. See [`genai-bonus/README.md`](genai-bonus/README.md). |

## Live versions

The deck and setup instructions are also published as live, shareable pages:

- **Slides:** https://claude.ai/artifact/7HhTPhfVTdBPewXJaagmPw
- **Setup instructions (student-facing):** https://claude.ai/artifact/XiMBVFk1xmtjeFnyU9ZwDr

## Running the notebook

See [`docs/setup-instructions.md`](docs/setup-instructions.md) for full student-facing setup steps. Short version:

```bash
pip install mlflow scikit-learn pandas matplotlib jupyter
jupyter notebook notebook/MLflow_Workshop_CHRIST_University.ipynb
```

or open it directly in [Google Colab](https://colab.research.google.com/) — the first code cell installs everything it needs.

## MLflow version notes

The notebook targets MLflow 3.16.1 and works around a few version-specific gotchas worth knowing if you adapt it:

- The old filesystem tracking backend (`./mlruns` alone) is in maintenance mode and doesn't support the Model Registry — the notebook uses `mlflow.set_tracking_uri("sqlite:///mlflow.db")` throughout.
- `mlflow.sklearn.log_model()`'s newer default serialization format (`skops`) rejects tree-based models like `RandomForestClassifier` unless explicitly marked trusted — the notebook passes `serialization_format="pickle"` instead.
- `search_model_versions()` doesn't populate `.aliases`; read aliases from `client.get_registered_model(name).aliases` instead.
