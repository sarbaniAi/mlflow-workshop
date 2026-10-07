# Bonus Module — Deep Learning Checkpoints with MLflow

**This is an optional, advanced bonus module.** It's not part of the core half-day MLflow seminar (that's in `../notebook/`) — it's for students who want to see how MLflow's tracking + registry story plays out for deep learning, where a single training run produces *many* candidate checkpoints instead of one `.fit()` result.

## What's here

| File | What it is |
|---|---|
| [`DeepLearning_MLflow_Checkpoints.ipynb`](DeepLearning_MLflow_Checkpoints.ipynb) | The notebook: train a small CNN with automatic per-epoch checkpointing, inspect the checkpoint artifacts, reload specific checkpoints and prove they differ, register three of them as separate versions of one model, and promote the best to `@champion`. |
| `build_notebook.py` | Generates the `.ipynb` above. Edit this and re-run it to change the notebook. |
| `requirements.txt` | `mlflow`, `tensorflow-cpu`, `scikit-learn`, `pandas`, `matplotlib`. |

No API key, no GPU, no cloud account — everything runs locally with open-source MLflow and TensorFlow/Keras on CPU.

## What it covers

- **`mlflow.tensorflow.MlflowModelCheckpointCallback`** — a Keras callback that logs a model checkpoint as an MLflow artifact after every epoch (or just the best one), with zero manual `model.save()` bookkeeping.
- **`mlflow.tensorflow.load_checkpoint(run_id=..., epoch=...)`** — reload the exact model from any logged epoch, to compare checkpoints against each other.
- **Registering multiple checkpoints as model versions** — three checkpoints from the *same* training run, registered as three separate versions of *one* registered model (`digits-cnn`), each with its own description and lineage back to its source run and epoch.
- **The same `@champion` alias pattern** from the main workshop's Lab 3, applied to picking the best checkpoint instead of the best hyperparameter setting.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook DeepLearning_MLflow_Checkpoints.ipynb
```

Run cells top to bottom — it trains in seconds on CPU (the dataset is scikit-learn's built-in 8x8 handwritten-digit images, 1,797 samples).

To browse the checkpoint artifacts and the registered versions in the MLflow UI, in a second terminal in this same folder:

```bash
mlflow ui --backend-store-uri sqlite:///checkpoints_mlflow.db
```

Open a run from the "digits-cnn-checkpoints" experiment and look at its **Artifacts** tab — you'll see a `checkpoints/` folder with one subfolder per epoch. The **Models** tab shows `digits-cnn` with its three registered versions and the `@champion` alias.

## Tested

Run end to end via `jupyter nbconvert --execute`: 0 errors across 26 cells. Checked the actual numbers, not just that it didn't crash — three checkpoints (epochs 0, 2, 4) were registered as versions 1/2/3 with distinct, increasing test accuracies, the best one (in that run, epoch 4 at ~0.96) was promoted to `@champion`, and inference through the `@champion` alias matched 9/10 true labels on a held-out batch.

## A note on the warnings you'll see

Keras prints a one-line "saving as HDF5... this format is considered legacy" notice on every checkpoint save, and TensorFlow occasionally logs a "tf.function retracing" notice when switching between differently-shaped inputs across the several models loaded in Section 4. Both are expected and harmless for this notebook's purposes — they don't affect correctness, just verbosity.
