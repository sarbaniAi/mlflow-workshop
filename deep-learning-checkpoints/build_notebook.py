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
# Deep Learning Checkpoints with MLflow
### Logging, Comparing, and Registering Checkpoints from a Training Run

> **This is an optional, advanced bonus module** — like `genai-bonus/`, it is
> not part of the core half-day MLflow seminar. The main workshop
> (`../notebook/`) covers classical ML with scikit-learn, where "the model"
> is a single `.fit()` call. This notebook covers the deep-learning case,
> where training runs for many epochs and **which checkpoint you keep
> matters** — the best validation score rarely lands on the last epoch.

MLflow has first-class support for this through `mlflow.tensorflow`'s
**`MlflowModelCheckpointCallback`**, a Keras callback that logs a checkpoint
as an MLflow artifact after every epoch (or just the best one), without any
manual `model.save()` bookkeeping.

| Section | What you'll do |
|---|---|
| 0. Setup | Install TensorFlow/Keras + MLflow |
| 1. Train with automatic checkpointing | One training run, one checkpoint artifact per epoch |
| 2. Inspect the checkpoint artifacts | See what MLflow logged, per epoch |
| 3. Load specific checkpoints back | Prove epoch 0, epoch 2, epoch 4 are genuinely different models |
| 4. Register multiple checkpoints as model versions | The Model Registry side: three checkpoints -> three versions of one registered model |
| 5. Promote the best checkpoint to `@champion` | Same alias pattern as the main workshop's Lab 3 |
| 6. Wrap-up | How this maps onto the registry concepts from the main workshop |

> Runs entirely locally with open-source MLflow and TensorFlow/Keras (CPU) —
> no GPU, no cloud account, no API keys. The dataset (`sklearn.datasets.load_digits`,
> 1,797 small 8x8 handwritten-digit images) is tiny on purpose, so a full
> training run finishes in seconds and the whole notebook runs end to end
> on a laptop.
""")

# ---------------------------------------------------------------------------
# Section 0 — Setup
# ---------------------------------------------------------------------------
md("""\
## 0. Setup
""")

code("""\
%pip install --quiet "tensorflow-cpu>=2.18" mlflow scikit-learn pandas matplotlib
""")

code("""\
import numpy as np
import pandas as pd
import mlflow
from mlflow.tensorflow import MlflowModelCheckpointCallback, load_checkpoint
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow import keras

mlflow.set_tracking_uri("sqlite:///checkpoints_mlflow.db")
mlflow.set_experiment("digits-cnn-checkpoints")

print("mlflow  ", mlflow.__version__)
import tensorflow as tf
print("tensorflow", tf.__version__)
print("keras     ", keras.__version__)
print("GPUs visible:", tf.config.list_physical_devices("GPU"), "(expected: none -- this runs on CPU)")
""")

md("""\
### The dataset

`load_digits()` is scikit-learn's built-in 8x8-pixel handwritten digit
dataset (1,797 images, 10 classes, 0-9) — small enough to train on in
seconds on a CPU, but genuinely a multi-class image classification problem,
so a small CNN is a legitimate (if tiny) deep learning model for it.
""")

code("""\
digits = load_digits()
X = digits.images.reshape(-1, 8, 8, 1).astype("float32") / 16.0  # scale pixel values to [0, 1]
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, " Test:", X_test.shape, " Classes:", np.unique(y))
""")

# ---------------------------------------------------------------------------
# Section 1 — Train with automatic checkpointing
# ---------------------------------------------------------------------------
md("""\
## 1. Train with automatic checkpointing

A small CNN -- two conv layers, then dense layers down to 10 classes.

The key piece is `MlflowModelCheckpointCallback`. With `save_best_only=False`,
it logs a checkpoint of the model **after every epoch** as an MLflow artifact
under `checkpoints/epoch_<N>/`, inside whichever run is active -- one
training run, N checkpoints, all logged automatically as the training loop
runs. (You'll see a one-line "saving as HDF5" notice from Keras per epoch --
that's just the checkpoint file format the callback uses internally; it's
expected and harmless.)
""")

code("""\
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(8, 8, 1)),
        keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()
model.summary()
""")

code("""\
EPOCHS = 8

checkpoint_cb = MlflowModelCheckpointCallback(
    monitor="val_accuracy",
    mode="max",
    save_best_only=False,   # log EVERY epoch's checkpoint, not just the best -- we want
                             # several checkpoints to compare and register below.
    save_weights_only=False,
    save_freq="epoch",
)

with mlflow.start_run(run_name="digits-cnn-training") as run:
    mlflow.log_params({"epochs": EPOCHS, "batch_size": 32, "optimizer": "adam", "model": "small-cnn"})

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=32,
        verbose=2,
        callbacks=[checkpoint_cb],
    )

    # The callback logs the checkpoint *artifacts*; log the per-epoch metrics
    # explicitly too, so they show up as MLflow metric history (the Charts
    # tab in the UI), the same way Lab 1 of the main workshop does.
    for epoch, (acc, val_acc) in enumerate(zip(history.history["accuracy"], history.history["val_accuracy"])):
        mlflow.log_metric("accuracy", acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    training_run_id = run.info.run_id

print("\\nTraining run:", training_run_id)
print("val_accuracy per epoch:", [round(v, 4) for v in history.history["val_accuracy"]])
""")

md("""\
**If you have `mlflow ui` running** (`mlflow ui --backend-store-uri sqlite:///checkpoints_mlflow.db`
in a terminal in this folder), open the run above and look at its
**Artifacts** tab now -- you'll see a `checkpoints/` folder with one
subfolder per epoch, each a complete saved Keras model.
""")

# ---------------------------------------------------------------------------
# Section 2 — Inspect the checkpoint artifacts
# ---------------------------------------------------------------------------
md("""\
## 2. Inspect the checkpoint artifacts

Confirm programmatically what got logged, without leaving the notebook.
""")

code("""\
client = mlflow.MlflowClient()

top_level = client.list_artifacts(training_run_id)
print("Top-level artifacts:", [a.path for a in top_level])

checkpoint_artifacts = client.list_artifacts(training_run_id, "checkpoints")
print("Checkpoint artifacts:")
for a in checkpoint_artifacts:
    print(f"  {a.path}  (dir={a.is_dir})")
""")

code("""\
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["accuracy"], marker="o", label="train accuracy")
ax.plot(history.history["val_accuracy"], marker="o", label="val accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy per epoch -- one checkpoint logged at each point")
ax.legend()
fig.tight_layout()
fig.savefig("accuracy_curve.png")
plt.show()

# Log the plot itself as an artifact on the training run, same pattern as
# the confusion-matrix artifact in the main workshop's Lab 1.
mlflow.log_artifact("accuracy_curve.png", run_id=training_run_id)
""")

md("""\
**Why checkpoint every epoch instead of just the last one?** Look at the
curve above: validation accuracy is rarely monotonic. The epoch with the
lowest training loss is not always the epoch with the best validation
score -- that's the whole reason checkpointing exists as a practice, and
why MLflow logs it as a first-class artifact rather than leaving it to
`model.save()` calls you have to remember to make.
""")

# ---------------------------------------------------------------------------
# Section 3 — Load specific checkpoints back
# ---------------------------------------------------------------------------
md("""\
## 3. Load specific checkpoints back

`mlflow.tensorflow.load_checkpoint(run_id=..., epoch=...)` reconstructs the
Keras model exactly as it was after that epoch. Load back three checkpoints
from different points in training and re-evaluate each on the held-out test
set, to prove they're genuinely different models, not the same weights
three times.
""")

code("""\
checkpoint_epochs_to_compare = [0, 2, 4, EPOCHS - 1]

checkpoint_results = []
for epoch in checkpoint_epochs_to_compare:
    ckpt_model = load_checkpoint(run_id=training_run_id, epoch=epoch)
    loss, acc = ckpt_model.evaluate(X_test, y_test, verbose=0)
    checkpoint_results.append({"epoch": epoch, "test_loss": loss, "test_accuracy": acc})
    print(f"epoch {epoch:>2}:  test_accuracy={acc:.4f}  test_loss={loss:.4f}")

checkpoint_df = pd.DataFrame(checkpoint_results)
checkpoint_df
""")

md("""\
**Notice:** the epoch-0 checkpoint's test accuracy here matches its
val_accuracy from the training log for that same epoch, and so do the
others -- confirming `load_checkpoint()` is reloading the exact weights
from that point in training, not retraining or approximating.
""")

# ---------------------------------------------------------------------------
# Section 4 — Register multiple checkpoints as model versions
# ---------------------------------------------------------------------------
md("""\
## 4. Register multiple checkpoints as model versions

Artifacts are the training-run side of checkpointing. The **Model Registry**
side -- the same one Lab 3 of the main workshop uses -- is where you turn a
specific checkpoint into a named, versioned model that other code can
depend on.

Register three of the checkpoints loaded above as **three separate versions
of the same registered model**, `digits-cnn`. Each version is logged from
its own run (so each has its own lineage back to the exact checkpoint and
metrics it came from), via `mlflow.tensorflow.log_model(..., registered_model_name=...)`.
""")

code("""\
from mlflow.models import infer_signature

model_name = "digits-cnn"
registered_versions = []  # (version, source_epoch, test_accuracy)

for epoch in [0, 2, 4]:
    ckpt_model = load_checkpoint(run_id=training_run_id, epoch=epoch)
    loss, acc = ckpt_model.evaluate(X_test, y_test, verbose=0)

    with mlflow.start_run(run_name=f"digits-cnn-checkpoint-epoch{epoch}") as run:
        mlflow.log_param("source_training_run_id", training_run_id)
        mlflow.log_param("checkpoint_epoch", epoch)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_loss", loss)

        signature = infer_signature(X_test[:5], ckpt_model.predict(X_test[:5], verbose=0))
        model_info = mlflow.tensorflow.log_model(
            ckpt_model,
            name="model",
            signature=signature,
            input_example=X_test[:2],
            registered_model_name=model_name,
        )

    version = model_info.registered_model_version
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Checkpoint from epoch={epoch} of run {training_run_id[:8]}. "
                     f"test_accuracy={acc:.4f}, test_loss={loss:.4f}",
    )
    registered_versions.append((version, epoch, acc))
    print(f"Registered version {version}  <- checkpoint epoch {epoch}  (test_accuracy={acc:.4f})")
""")

md("""\
> You may see `registering model based on models:/m-... instead` -- that's
> MLflow's newer internal logged-model ID appearing behind the scenes,
> exactly as noted in the main workshop's Lab 3. Registration still works
> as described above.
""")

# ---------------------------------------------------------------------------
# Section 5 — Promote the best checkpoint to @champion
# ---------------------------------------------------------------------------
md("""\
## 5. Promote the best checkpoint to `@champion`

Same alias pattern as the main workshop's Lab 3: pick the best-performing
registered version by its logged metric, and move the alias to it rather
than hardcoding a version number anywhere downstream.
""")

code("""\
best_version, best_epoch, best_acc = max(registered_versions, key=lambda t: t[2])

client.set_registered_model_alias(name=model_name, alias="champion", version=best_version)
print(f"models:/{model_name}@champion -> version {best_version} (from checkpoint epoch {best_epoch}, test_accuracy={best_acc:.4f})")
""")

code("""\
# Inspect what's registered -- same pattern as the main workshop's Lab 3.
rm = client.get_registered_model(model_name)
print(f"Registered model: {model_name}")
print(f"Aliases -> version: {rm.aliases}")
print()
for mv in client.search_model_versions(f"name='{model_name}'"):
    print(f"version={mv.version}  run_id={mv.run_id[:8]}...  description={mv.description!r}")
""")

code("""\
# Load by alias, exactly as a serving process would -- it never needs to
# know which checkpoint epoch produced the champion version.
champion_model = mlflow.tensorflow.load_model(f"models:/{model_name}@champion")
preds = champion_model.predict(X_test[:10], verbose=0)
predicted_labels = np.argmax(preds, axis=1)

print("Predicted:", predicted_labels)
print("True:     ", y_test[:10])
print("Match:    ", (predicted_labels == y_test[:10]).sum(), "/ 10")
""")

md("""\
**Try it:** go back to Section 1, train for more epochs, or change the CNN
(add a layer, change the dropout rate), then repeat Section 4 to register a
new checkpoint as **version 4** of `digits-cnn`, and move `@champion` to it
if it's better. This is exactly how a deep learning model gets promoted in
production -- by moving a pointer, never by redeploying code.
""")

# ---------------------------------------------------------------------------
# Section 6 — Wrap-up
# ---------------------------------------------------------------------------
md("""\
## 6. Wrap-up -- how this maps onto the main workshop

| Main workshop (classical ML) | This module (deep learning checkpoints) |
|---|---|
| One `.fit()` call per run | **Many checkpoints per run** -- one training run, one artifact per epoch |
| Manually decide when to log the model | `MlflowModelCheckpointCallback` logs every epoch (or just the best) automatically |
| Register the one model you trained | **Register several checkpoints** from the same run as separate versions, then compare |
| `client.set_registered_model_alias(..., "champion", ...)` | Identical API, same alias pattern -- the registry doesn't care whether the model came from one `.fit()` or epoch 6 of 8 |
| Load `models:/name@champion` to serve | Identical -- serving code never needs to know which epoch, or even which framework, produced the champion version |

The underlying idea carries over exactly: **the Model Registry's job is to
turn "a bunch of candidate models" into "the one we're using," regardless
of how those candidates were produced.** For classical ML, each candidate
was a different hyperparameter setting. Here, each candidate is a different
epoch of the same training run. Same registry, same alias mechanism, same
promotion workflow.
""")

nb["cells"] = cells
nbf.write(nb, "DeepLearning_MLflow_Checkpoints.ipynb")
print(f"Wrote {len(cells)} cells.")
