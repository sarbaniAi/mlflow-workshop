import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
# Title & intro
# ---------------------------------------------------------------------------
md("""\
# MLflow & the Machine Learning Lifecycle
### Hands-on Workshop — M.Sc. Data Science, CHRIST (Deemed to be University)

This notebook is the hands-on companion to the seminar. It has seven parts:

| Section | What you'll do | Time |
|---|---|---|
| 0. Setup | Install MLflow, check it works | 5 min |
| 1. Lab 1 — Tracking | Log and compare experiment runs | 45 min |
| 2. Lab 2 — Models | Package a model with a signature | 30 min |
| 3. Lab 3 — Registry | Register and promote a model version | 35 min |
| 4. Evaluation | Score a model with `mlflow.evaluate()` | 15 min |
| 5. Projects & Deployment | Package code, not just the model | 20 min |
| 6. Industry practices | Best practices and MLOps skills to build | reading |
| 7. Mini Capstone | Do all of the above, on your own, on a new dataset | 40 min |

**Run cells top to bottom.** Markdown cells explain *why*; code cells show *how*. Cells marked `# TODO` in Section 7 are for you to complete.

> Works identically in a local Jupyter notebook and in Google Colab — no cloud account, no API keys.
""")

# ---------------------------------------------------------------------------
# Section 0 — Setup
# ---------------------------------------------------------------------------
md("""\
## 0. Setup

We use **open-source MLflow**, running entirely on this machine — no server, no account, no API key.

Metadata (params, metrics, tags, and the Model Registry) is stored in a local SQLite file, `mlflow.db`. Artifacts (models, plots, and other files) are stored in a local folder, `mlruns/`, next to this notebook. Both are created automatically the first time you log something.

> **Why SQLite instead of the plain `mlruns/` file store?** Recent MLflow versions have put the old file-only store into maintenance mode, and — separately — the **Model Registry we use in Lab 3 has always required a database-backed store**. A single `sqlite:///mlflow.db` line gets you both a fully supported setup and registry support, with zero extra installation.

**If you are in Google Colab:** both `mlflow.db` and `mlruns/` live on the temporary Colab VM and are **deleted when the runtime resets**. Two options:
1. Keep the runtime alive for the whole session (default — fine for a single workshop sitting), or
2. At the end, run the "Save your work" cell near the bottom to zip and download both.
""")

code("""\
# Run once. In Colab this takes ~20–30 seconds.
%pip install --quiet mlflow scikit-learn pandas matplotlib
""")

code("""\
import mlflow
import sklearn
print("mlflow      ", mlflow.__version__)
print("scikit-learn", sklearn.__version__)
""")

code("""\
# Metadata -> local SQLite file; artifacts -> local ./mlruns folder.
# Both are created automatically on first use.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
print("Tracking URI:", mlflow.get_tracking_uri())
""")

md("""\
### Viewing runs

You have two options for looking at what you log — use whichever fits where you're running:

- **Local Jupyter / terminal:** open a second terminal in this same folder and run `mlflow ui`, then open **http://localhost:5000** in your browser. This gives you the full Tracking UI shown in the slides (run comparison, parallel-coordinates plot, artifact browser).
- **Anywhere (including Colab):** use `mlflow.search_runs()` to pull runs into a pandas DataFrame and inspect them right here in the notebook. We'll do this after every lab, so the notebook is fully self-contained even without the UI.
""")

# ---------------------------------------------------------------------------
# Section 1 — Tracking
# ---------------------------------------------------------------------------
md("""\
## 1. Lab 1 — Experiment Tracking

**Dataset:** the built-in `wine` dataset from scikit-learn (178 samples, 13 features, 3 classes) — no download required.

**Goal:** train a few classifiers with different hyperparameters, log every attempt as its own MLflow *run*, then compare them without scrolling back through the notebook.
""")

code("""\
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.25, random_state=42, stratify=wine.target
)
print("Train:", X_train.shape, " Test:", X_test.shape, " Classes:", wine.target_names)
""")

code("""\
mlflow.set_experiment("wine-classifier")


def train_and_log(n_estimators, max_depth, run_name=None):
    \"\"\"Train a RandomForestClassifier and log everything about the attempt.\"\"\"
    with mlflow.start_run(run_name=run_name):
        # 1. Log the inputs (params) — fixed for this run
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForestClassifier")

        # 2. Train
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # 3. Log the outputs (metrics)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # 4. Log a file artifact — here, a confusion matrix plot
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, colorbar=False)
        fig.tight_layout()
        fig.savefig("confusion_matrix.png")
        plt.close(fig)
        mlflow.log_artifact("confusion_matrix.png")

        run_id = mlflow.active_run().info.run_id
        print(f"run_id={run_id}  n_estimators={n_estimators:<4} max_depth={str(max_depth):<4} "
              f"accuracy={acc:.4f}  f1_macro={f1:.4f}")
        return run_id, acc


# A small hyperparameter sweep — three attempts, three runs
configs = [
    dict(n_estimators=50, max_depth=3),
    dict(n_estimators=150, max_depth=6),
    dict(n_estimators=300, max_depth=None),
]

results = [train_and_log(**cfg, run_name=f"rf-{i+1}") for i, cfg in enumerate(configs)]
best_run_id, best_acc = max(results, key=lambda r: r[1])
print(f"\\nBest run: {best_run_id}  (accuracy={best_acc:.4f})")
""")

md("""\
### Autologging — the one-line version

For many frameworks (scikit-learn, XGBoost, PyTorch Lightning, and more) MLflow can capture params, metrics and the model itself **without any manual `log_*` calls**, via `mlflow.autolog()`.
""")

code("""\
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="rf-autolog"):
    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    # No log_param / log_metric calls — MLflow captured them automatically.

mlflow.sklearn.autolog(disable=True)  # turn it back off for the rest of the notebook
""")

md("### Comparing runs\n\nPull every run in this experiment into a DataFrame and sort by accuracy — this is the programmatic equivalent of the Tracking UI's comparison table.")

code("""\
runs_df = mlflow.search_runs(
    experiment_names=["wine-classifier"],
    order_by=["metrics.accuracy DESC"],
)
runs_df[["run_id", "params.n_estimators", "params.max_depth", "metrics.accuracy", "metrics.f1_macro"]].head(10)
""")

md("""\
**Try it:** add two or three more configurations to the `configs` list above (different `n_estimators` / `max_depth` values) and re-run. Watch the comparison table grow.

If you have `mlflow ui` running, refresh **http://localhost:5000** now — you'll see every run you just logged, with sortable columns and a parallel-coordinates plot under the "Chart" view.
""")

# ---------------------------------------------------------------------------
# Section 2 — Models
# ---------------------------------------------------------------------------
md("""\
## 2. Lab 2 — Packaging Models

An MLflow **model** is a *self-describing folder*: the fitted model, its dependencies, and — importantly — a **signature** describing exactly what input it expects and what output it returns.

We'll retrain the best configuration, this time logging the model itself with an explicit signature and a small input example.
""")

code("""\
from mlflow.models import infer_signature

with mlflow.start_run(run_name="rf-best-packaged") as run:
    best_cfg = configs[1]  # n_estimators=150, max_depth=6 — adjust if your best run differs
    mlflow.log_params(best_cfg)

    model = RandomForestClassifier(**best_cfg, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)

    # Describe the model's input/output contract from real data
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_train[:5],
        # MLflow's newest default format for scikit-learn models is "skops" (safer
        # deserialization), but it currently rejects tree-based models like this
        # RandomForest unless you explicitly mark the type as trusted. We use the
        # classic "pickle" format here for simplicity — fine for a workshop /
        # trusted-source setting; review the security note MLflow prints below for
        # what to consider before doing this in production.
        serialization_format="pickle",
    )

    packaged_run_id = run.info.run_id

print("Packaged run:", packaged_run_id, " accuracy:", round(acc, 4))
print()
print("Signature MLflow inferred:")
print(signature)
""")

md("### Loading a packaged model back\n\nTwo ways to load it — pick whichever matches what you need:")

code("""\
model_uri = f"runs:/{packaged_run_id}/model"

# 1. Native flavor — returns the original scikit-learn object
sk_model = mlflow.sklearn.load_model(model_uri)
print(type(sk_model))

# 2. pyfunc flavor — a generic .predict() interface every MLflow model shares,
#    regardless of the framework it was trained with. This is what a serving
#    endpoint calls under the hood.
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
print(pyfunc_model.predict(X_test[:5]))
""")

md("**Why the signature matters:** try passing badly-shaped input and see MLflow catch it before it reaches your model — this is exactly what happens at a real serving endpoint.")

code("""\
import numpy as np

bad_input = X_test[:5, :5]  # wrong number of columns on purpose
try:
    pyfunc_model.predict(bad_input)
except Exception as e:
    print("Caught as expected:")
    print(type(e).__name__, "-", str(e)[:200])
""")

# ---------------------------------------------------------------------------
# Section 3 — Registry
# ---------------------------------------------------------------------------
md("""\
## 3. Lab 3 — The Model Registry

Tracking gives you *runs*. The **Model Registry** gives you a *named, versioned model* that downstream code can depend on — with lineage back to the exact run that produced each version.

We'll register the model we packaged above, document it, and promote it with an **alias** (the modern replacement for the old `Staging`/`Production` stages).

> You may see a message like `registering model based on models:/m-... instead` — that's MLflow's newer internal logged-model ID kicking in behind the scenes. Registration still works exactly as described below; you can ignore it.
""")

code("""\
from mlflow import MlflowClient

client = MlflowClient()
model_name = "wine-classifier-rf"

registered = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Registered '{model_name}' as version {registered.version}")
""")

code("""\
# Document the model and this specific version
client.update_registered_model(
    name=model_name,
    description="RandomForest classifier for the sklearn wine dataset. "
                 "Trained in the CHRIST University MLflow workshop.",
)

client.update_model_version(
    name=model_name,
    version=registered.version,
    description=f"n_estimators={best_cfg['n_estimators']}, max_depth={best_cfg['max_depth']}, "
                 f"test accuracy={acc:.4f}",
)

# Promote this version with an alias — serving code will always ask for "@champion",
# so promoting a new version later is just moving the alias, no redeploy required.
client.set_registered_model_alias(name=model_name, alias="champion", version=registered.version)

print(f"models:/{model_name}@champion -> version {registered.version}")
""")

code("""\
# Load by alias, exactly as a serving process would
champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
print(champion_model.predict(X_test[:5]))
""")

code("""\
# Inspect what's registered.
# Note: alias info lives on the registered model (get_registered_model),
# not on each version returned by search_model_versions.
rm = client.get_registered_model(model_name)
print(f"Registered model: {model_name}")
print(f"Aliases -> version: {rm.aliases}")
print()
for mv in client.search_model_versions(f"name='{model_name}'"):
    print(f"version={mv.version}  run_id={mv.run_id}  description={mv.description!r}")
""")

md("""\
**Try it:** go back to Lab 1, train one more configuration, log its model with a signature (like in Lab 2), register it as a **new version** of `wine-classifier-rf`, and move the `@champion` alias to it. This is exactly how a model gets promoted in production — without touching any serving code.
""")

# ---------------------------------------------------------------------------
# Section 4 — Evaluation
# ---------------------------------------------------------------------------
md("""\
## 4. MLflow Evaluation

`mlflow.evaluate()` scores a model against a held-out dataset and automatically logs a standard set of metrics and plots — no need to hand-write accuracy/F1/confusion-matrix code every time.
""")

code("""\
import pandas as pd

eval_df = pd.DataFrame(X_test, columns=wine.feature_names)
eval_df["label"] = y_test

with mlflow.start_run(run_name="rf-evaluation"):
    result = mlflow.evaluate(
        model=f"models:/{model_name}@champion",
        data=eval_df,
        targets="label",
        model_type="classifier",
    )

pd.Series(result.metrics).sort_index()
""")

md("""\
`mlflow.evaluate()` also accepts a `extra_metrics` argument for **custom metric functions** — useful when accuracy alone doesn't capture what matters (a cost-weighted error, a fairness gap, a domain-specific score).

For GenAI / LLM applications, the same `mlflow.evaluate()` framework extends to metrics like answer relevance, faithfulness and toxicity, including LLM-as-judge scoring — same idea, applied to text outputs instead of class labels. Out of scope for today, but worth knowing the same tool covers it.
""")

# ---------------------------------------------------------------------------
# Section 5 — Projects & Deployment
# ---------------------------------------------------------------------------
md("""\
## 5. MLflow Projects & Deployment Concepts

Everything so far has packaged the **model**. MLflow Projects packages the **code** — so "how do I even run this?" has one answer, not five READMEs.

An `MLproject` file (plain YAML, no code) declares the entry point and its parameters:

```yaml
name: wine-mlproject-demo

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 150}
      max_depth: {type: int, default: 5}
    command: "python train.py {n_estimators} {max_depth}"
```

With that file next to a `train.py` script, anyone — a teammate, a CI runner, a scheduled job — reproduces your exact run with one command:

```bash
mlflow run . -P n_estimators=300
```

No "activate my conda env first," no "did you install the right pandas version" — the project declares what it needs, and `mlflow run` handles the rest. Let's build one and actually run it.
""")

code("""\
import os

os.makedirs("wine_project", exist_ok=True)
""")

code("""\
%%writefile wine_project/train.py
import os
import sys

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# mlflow run passes parameters as positional command-line arguments, in the
# order declared in the MLproject file.
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 150
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# MLFLOW_TRACKING_URI is set as an environment variable by the notebook cell
# that launches this script (not hardcoded here), so this run lands in the
# SAME mlflow.db as every lab before it — not a fresh one inside this folder.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_train[:5],
        serialization_format="pickle",
    )

    print(f"Logged run: accuracy={acc:.4f}  n_estimators={n_estimators}  max_depth={max_depth}")
""")

code("""\
%%writefile wine_project/MLproject
name: wine-mlproject-demo

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 150}
      max_depth: {type: int, default: 5}
    command: "python train.py {n_estimators} {max_depth}"
""")

md("""\
Now run it. `--env-manager local` tells MLflow to use *this* Python environment instead of building a fresh conda/virtualenv for the project — the right call here, since Section 0 already installed everything the project needs. We pass `MLFLOW_TRACKING_URI` as an environment variable (rather than hardcoding it inside `train.py`) so the project logs into the exact same `mlflow.db` as Labs 1–3, not a new one scoped to this folder.
""")

code("""\
import os

# Earlier cells called mlflow.set_experiment(), which leaves MLFLOW_EXPERIMENT_ID
# set in THIS process's environment. mlflow run's subprocess would inherit it and
# then refuse to also accept --experiment-name below, so clear it first.
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{os.path.abspath('mlflow.db')}"

!mlflow run wine_project -P n_estimators=150 -P max_depth=5 --env-manager local --experiment-name wine-mlproject-demo
""")

code("""\
# Prove it: query the run `mlflow run` just created, from the exact same
# tracking store every other lab in this notebook uses.
runs = mlflow.search_runs(experiment_names=["wine-mlproject-demo"])
runs[["run_id", "params.n_estimators", "params.max_depth", "metrics.accuracy"]]
""")

md("""\
That's reproducibility end to end: anyone with nothing but the `wine_project/` folder and a Python environment runs the exact same command and gets a logged, comparable result — no "works on my machine."
""")

md("""\
### Deployment, the simplest version

Once a model is registered, turning it into a REST endpoint is one line — no Flask app, no Dockerfile to hand-write:

```bash
mlflow models serve -m models:/wine-classifier-rf@champion -p 5001
```

This spins up a local server; a POST request to `/invocations` with JSON-formatted rows returns predictions, validated against the signature we logged in Lab 2. At organizational scale, this same idea becomes a managed service (autoscaling, authentication, monitoring) — Databricks Model Serving is one example, and most major cloud/ML platforms offer an equivalent.
""")

# ---------------------------------------------------------------------------
# Section 6 — Industry practices
# ---------------------------------------------------------------------------
md("""\
## 6. Industry Best Practices & Building MLOps Skills

A few habits that separate a workshop exercise from a production MLflow setup:

**Do**
- Log a signature and input example on **every** model, not just the ones you plan to serve
- Serve by **alias** (`@champion`), never a hardcoded version number — promotion becomes a metadata change, not a redeploy
- Tag runs like commit messages — owner, dataset version, git commit — so future-you can search by tag

**Avoid**
- Logging raw training data as an artifact — log a small sample or a versioned pointer instead
- An unpinned `conda.yaml` / `requirements.txt` — it breaks silently, months later, on someone else's machine
- Tracking only the runs you remember to log manually — wire MLflow logging into CI/CD instead

### Skills worth building next

You already have the Python and ML fundamentals this workshop assumed. To go from "I used MLflow once" to "I can own an MLOps pipeline," the next layer is usually:

- **Git & CI/CD basics** — training and deployment pipelines live in version control like any other code
- **Containers (Docker)** — the unit most model-serving systems actually deploy
- **Cloud fundamentals** — any one of AWS, Azure or GCP; most MLOps stacks run on one of them
- **Monitoring & observability** — data drift and performance decay don't announce themselves; you have to watch for them
- **Stakeholder communication** — translating a metric into a business decision is a skill in its own right, and often what separates a good ML engineer from a great one
""")

# ---------------------------------------------------------------------------
# Section 7 — Capstone
# ---------------------------------------------------------------------------
md("""\
## 7. Mini Capstone Challenge

This is the "simple end-to-end workflow" the whole notebook has been building toward — **Track → Package → Register → Evaluate**, chained together, on a dataset you haven't seen yet.

**Pair up — with a large room, two sets of eyes catch bugs faster. ~40 minutes.** Do the full workflow yourself, end to end, on a **different dataset** — `load_breast_cancer` (binary classification, 30 features).

### Checklist
- [ ] Load the dataset below and split into train/test
- [ ] Train **at least two** models or hyperparameter settings, logging each as its own run (params, metrics, one artifact)
- [ ] Compare runs with `mlflow.search_runs()`
- [ ] Package the best model with a signature and input example
- [ ] Register it and assign it the `@champion` alias
- [ ] Run `mlflow.evaluate()` against your champion model

### Deliverable
A screenshot (or the printed DataFrame output) of your run comparison, plus your registered model's name, version and alias.
""")

code("""\
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    cancer.data, cancer.target, test_size=0.25, random_state=42, stratify=cancer.target
)
print("Train:", Xc_train.shape, " Test:", Xc_test.shape, " Classes:", cancer.target_names)

mlflow.set_experiment("breast-cancer-capstone")
""")

code("""\
# TODO 1: train at least two models / hyperparameter settings.
# Reuse the train_and_log() pattern from Lab 1, adapted to this dataset —
# or write your own loop. Log params, at least one metric, and one artifact
# for each attempt.

# your code here
""")

code("""\
# TODO 2: compare your runs.
capstone_runs = mlflow.search_runs(
    experiment_names=["breast-cancer-capstone"],
    order_by=["metrics.accuracy DESC"],
)
capstone_runs.head(10)
""")

code("""\
# TODO 3: package your best model with a signature and input example
# (see Lab 2 for the pattern), then register it and set the @champion alias
# (see Lab 3 for the pattern).

# your code here
""")

code("""\
# TODO 4: run mlflow.evaluate() against your registered champion model
# (see Section 4 for the pattern).

# your code here
""")

# ---------------------------------------------------------------------------
# Save work (Colab)
# ---------------------------------------------------------------------------
md("""\
## Save your work (Google Colab only)

Colab's filesystem disappears when the runtime disconnects. Run the cell below to zip `mlflow.db` and the `mlruns/` folder together and download them, so you keep a record of everything you logged today.
""")

code("""\
import shutil
import zipfile
import os

with zipfile.ZipFile("mlflow_backup.zip", "w") as zf:
    zf.write("mlflow.db")
    for root, _, filenames in os.walk("mlruns"):
        for fname in filenames:
            path = os.path.join(root, fname)
            zf.write(path)

try:
    from google.colab import files
    files.download("mlflow_backup.zip")
except ImportError:
    print("Not running in Colab — mlflow_backup.zip was created in this folder.")
""")

md("""\
## Wrap-up

You've now used every core piece of MLflow hands-on, and seen how they chain into one workflow:

- **Tracking** — every run logged, nothing relying on memory
- **Models** — a portable package with a signature every tool understands
- **Registry** — a governed, versioned, aliasable model
- **Evaluation** — metrics computed consistently, the same way every time
- **Projects** — code packaged so anyone can reproduce your run
- **Deployment** — a registered model, one command away from a REST endpoint

**Resources:** [mlflow.org/docs/latest](https://mlflow.org/docs/latest/) · [github.com/mlflow/mlflow](https://github.com/mlflow/mlflow)

Questions afterwards: sarbaniiitb2020@gmail.com
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    "colab": {"provenance": [], "name": "MLflow_Workshop_CHRIST_University.ipynb"},
}

with open("/home/claude/mlflow_workshop/MLflow_Workshop_CHRIST_University.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written with", len(cells), "cells.")
