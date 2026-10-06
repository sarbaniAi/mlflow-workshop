"""
Builds the INSTRUCTOR SOLUTIONS notebook for the Mini Capstone (Section 7).

Rather than duplicating the ~500 lines of Sections 0-6 by hand, this script
reuses the exact cell list from build_notebook.py (so Sections 0-6 are
byte-for-byte identical to what students get) and replaces only the three
blank "# your code here" TODO cells in Section 7 with full, tested
implementations.

Run `python3 build_notebook.py` first if you've changed that file -- this
script re-executes it in-process to get its cell list, it doesn't read the
.ipynb output.
"""

import importlib.util

import nbformat as nbf

# ---------------------------------------------------------------------------
# Load build_notebook.py's cell list without letting it write its own file.
# ---------------------------------------------------------------------------
_real_write = nbf.write
nbf.write = lambda *a, **k: None  # suppress build_notebook.py's own nbf.write call

spec = importlib.util.spec_from_file_location("build_notebook", "build_notebook.py")
build_notebook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_notebook)

nbf.write = _real_write  # restore for our own use below

cells = build_notebook.cells

# ---------------------------------------------------------------------------
# Replace the title cell with a solutions-specific version.
# ---------------------------------------------------------------------------
assert cells[0]["cell_type"] == "markdown" and "Machine Learning Lifecycle" in cells[0]["source"]
cells[0]["source"] = """\
# MLflow & the Machine Learning Lifecycle
### Hands-on Workshop — M.Sc. Data Science, CHRIST (Deemed to be University)

> **INSTRUCTOR SOLUTIONS NOTEBOOK.** Sections 0-6 are identical to the student
> notebook (`MLflow_Workshop_CHRIST_University.ipynb`). Section 7's four
> `# TODO` cells are filled in here with a complete, tested solution — use
> this as an answer key, not something to hand to students before the
> capstone.

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
| 7. Mini Capstone | Do all of the above, on your own, on a new dataset — **solved below** | 40 min |

**Run cells top to bottom.** Markdown cells explain *why*; code cells show *how*.

> Works identically in a local Jupyter notebook and in Google Colab — no cloud account, no API keys.
"""

# ---------------------------------------------------------------------------
# Locate and replace the three blank TODO cells in Section 7.
# ---------------------------------------------------------------------------
def find_cell(marker):
    matches = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and marker in c["source"]]
    assert len(matches) == 1, f"expected exactly one cell matching {marker!r}, found {len(matches)}"
    return matches[0]


# TODO 1 — train at least two models, logging each as its own run.
i = find_cell("# TODO 1: train at least two models")
cells[i]["source"] = """\
# SOLUTION 1: train at least two models / hyperparameter settings, each its own run.
# Same train_and_log() pattern as Lab 1, adapted to this dataset: log params,
# metrics, and one artifact (a confusion matrix) per attempt.

def train_and_log_capstone(n_estimators, max_depth, run_name=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForestClassifier")

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(Xc_train, yc_train)
        preds = model.predict(Xc_test)

        acc = accuracy_score(yc_test, preds)
        f1 = f1_score(yc_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(yc_test, preds, ax=ax, colorbar=False)
        fig.tight_layout()
        fig.savefig("capstone_confusion_matrix.png")
        plt.close(fig)
        mlflow.log_artifact("capstone_confusion_matrix.png")

        run_id = mlflow.active_run().info.run_id
        print(f"run_id={run_id}  n_estimators={n_estimators:<4} max_depth={str(max_depth):<4} "
              f"accuracy={acc:.4f}  f1_macro={f1:.4f}")
        return run_id, acc


# Three attempts, three runs -- same sweep-and-compare pattern as Lab 1.
capstone_configs = [
    dict(n_estimators=50, max_depth=4),
    dict(n_estimators=200, max_depth=8),
    dict(n_estimators=300, max_depth=None),
]

capstone_results = [
    (*train_and_log_capstone(**cfg, run_name=f"cancer-rf-{i + 1}"), cfg)
    for i, cfg in enumerate(capstone_configs)
]
best_capstone_run_id, best_capstone_acc, best_capstone_cfg = max(
    capstone_results, key=lambda r: r[1]
)
print(f"\\nBest run: {best_capstone_run_id}  (accuracy={best_capstone_acc:.4f}, cfg={best_capstone_cfg})")
"""

# TODO 3 — package the best model with a signature/input example, register, set alias.
i = find_cell("# TODO 3: package your best model")
cells[i]["source"] = """\
# SOLUTION 3: package the best model with a signature + input example (Lab 2
# pattern), register it, and promote it with the @champion alias (Lab 3 pattern).

with mlflow.start_run(run_name="cancer-rf-best-packaged") as run:
    mlflow.log_params(best_capstone_cfg)

    model = RandomForestClassifier(**best_capstone_cfg, random_state=42)
    model.fit(Xc_train, yc_train)
    preds = model.predict(Xc_test)
    acc = accuracy_score(yc_test, preds)
    mlflow.log_metric("accuracy", acc)

    signature = infer_signature(Xc_train, model.predict(Xc_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=Xc_train[:5],
        serialization_format="pickle",  # see the note on this in Lab 2
    )

    capstone_packaged_run_id = run.info.run_id

print("Packaged run:", capstone_packaged_run_id, " accuracy:", round(acc, 4))

capstone_model_uri = f"runs:/{capstone_packaged_run_id}/model"
capstone_model_name = "breast-cancer-rf"

registered_capstone = mlflow.register_model(model_uri=capstone_model_uri, name=capstone_model_name)
print(f"Registered '{capstone_model_name}' as version {registered_capstone.version}")

client.update_registered_model(
    name=capstone_model_name,
    description="RandomForest classifier for the sklearn breast cancer dataset "
                "-- mini capstone solution.",
)
client.update_model_version(
    name=capstone_model_name,
    version=registered_capstone.version,
    description=f"n_estimators={best_capstone_cfg['n_estimators']}, "
                f"max_depth={best_capstone_cfg['max_depth']}, test accuracy={acc:.4f}",
)

client.set_registered_model_alias(
    name=capstone_model_name, alias="champion", version=registered_capstone.version
)
print(f"models:/{capstone_model_name}@champion -> version {registered_capstone.version}")
"""

# TODO 4 — run mlflow.evaluate() against the registered champion model.
i = find_cell("# TODO 4: run mlflow.evaluate()")
cells[i]["source"] = """\
# SOLUTION 4: run mlflow.evaluate() against the registered @champion model
# (same pattern as Section 4, applied to the breast cancer test set).

cancer_eval_df = pd.DataFrame(Xc_test, columns=cancer.feature_names)
cancer_eval_df["label"] = yc_test

with mlflow.start_run(run_name="cancer-rf-evaluation"):
    capstone_eval_result = mlflow.evaluate(
        model=f"models:/{capstone_model_name}@champion",
        data=cancer_eval_df,
        targets="label",
        model_type="classifier",
    )

pd.Series(capstone_eval_result.metrics).sort_index()
"""

# ---------------------------------------------------------------------------
# Add a closing "deliverable" cell right after Section 7's evaluate() cell.
# ---------------------------------------------------------------------------
deliverable_md = nbf.v4.new_markdown_cell(
    """\
### Deliverable, assembled

Everything the checklist asked for, pulled into one place:
"""
)
deliverable_code = nbf.v4.new_code_cell(
    """\
print("Run comparison (sorted by accuracy):")
display(capstone_runs.head(10))

print()
print(f"Registered model : {capstone_model_name}")
print(f"Version          : {registered_capstone.version}")
print(f"Alias            : @champion -> version {registered_capstone.version}")
print(f"Test accuracy    : {acc:.4f}")
"""
)

insert_at = find_cell("# SOLUTION 4: run mlflow.evaluate()") + 1
cells[insert_at:insert_at] = [deliverable_md, deliverable_code]

# ---------------------------------------------------------------------------
# Write the solutions notebook.
# ---------------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nbf.write(nb, "MLflow_Workshop_CHRIST_University_SOLUTIONS.ipynb")
print(f"Wrote {len(cells)} cells.")
