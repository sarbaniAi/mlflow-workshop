# MLflow Workshop — Student Setup Instructions

## Welcome

You're set to attend the **MLflow and the Machine Learning Lifecycle** seminar, run for the M.Sc. Data Science PG students at CHRIST (Deemed to be University). It's a half-day, hands-on session — you'll spend most of it writing and running code, not just watching slides.

MLflow works the same everywhere, but a few minutes of setup beforehand means we spend the whole session building instead of debugging installs. Please complete **one** of the two options below (local Jupyter or Google Colab) before you arrive.

## Before you arrive — checklist

- [ ] Pick **one**: local Jupyter (Option A) or Google Colab (Option B) — no need to do both
- [ ] Local option: Python 3.9 or later installed
- [ ] Colab option: a Google account you can sign into
- [ ] Run the verification cell in "Verifying your setup" and confirm it prints `Setup OK`
- [ ] Bring a laptop with a charger — the whole session is hands-on

Estimated setup time: **10–15 minutes**.

## Option A — Local Jupyter (recommended if you already use Python locally)

**1. Check Python is 3.9 or later**

```bash
python3 --version
```

**2. Create a virtual environment** (keeps this workshop's packages separate from anything else on your machine)

```bash
python3 -m venv mlflow-workshop-env

# macOS / Linux
source mlflow-workshop-env/bin/activate

# Windows (Command Prompt)
mlflow-workshop-env\Scripts\activate.bat
```

**3. Install the packages we'll use**

```bash
pip install mlflow scikit-learn pandas matplotlib jupyter
```

**4. Launch Jupyter**

```bash
jupyter notebook
```

This opens Jupyter in your browser. Keep the terminal window open for the whole workshop — closing it shuts down the notebook.

## Option B — Google Colab (no local install needed)

Nothing to install ahead of time. During the workshop, the very first code cell of the notebook does the install for you:

```python
%pip install --quiet mlflow scikit-learn pandas matplotlib
```

One thing to know: Colab's storage is temporary. If your runtime disconnects or resets, anything MLflow logged is lost unless you download it first — the notebook has a "Save your work" cell at the end for exactly this.

## Verifying your setup

Run this in a fresh notebook cell (local Jupyter or Colab, after the install step):

```python
import mlflow
import sklearn
import pandas
import matplotlib

print("mlflow      ", mlflow.__version__)
print("scikit-learn", sklearn.__version__)
print("Setup OK")
```

If it prints **"Setup OK"** with no errors, you're ready — delete this test cell and see you at the workshop.

## Troubleshooting

**`pip install` fails or is very slow** — switch to the Colab option; it needs no local install.

**`python` isn't recognized, or an old version shows** — on Windows try `py --version`; on Mac/Linux try `python3 --version`. Use whichever command works in the steps above.

**Corporate/college Wi-Fi blocks `pip install`** — try a personal hotspot, or switch to Colab, which only needs a browser.

**"Module not found" errors after activating the virtual environment** — make sure you see `(mlflow-workshop-env)` at the start of your terminal prompt before running `pip install`. If not, re-run the activate command from step 2.

**You already have an older MLflow installed and something looks different from the notebook** — upgrade it:

```bash
pip install --upgrade mlflow
```

Still stuck? See the contact section below — reach out before the day so we can sort it out in advance.

## What to bring, and what to expect

- A laptop with a charger, and your setup verified beforehand (see checklist above)
- No prior MLflow experience needed — basic Python and a first course in ML (models, train/test splits) is enough
- The session is hands-on throughout: expect to be writing and running code for most of the session, working through guided labs and a short capstone challenge at the end
- Everything runs locally on your own machine (or Colab) — no cloud account or credentials to arrange in advance

## Questions

Setup trouble, or anything else before the day: **sarbaniiitb2020@gmail.com**
