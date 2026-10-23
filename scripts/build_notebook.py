"""Generate notebooks/capstone.ipynb from the cell list below.

Keeping the notebook as code makes it reviewable in diffs and keeps it in
sync with the package API. Run:  python scripts/build_notebook.py
"""

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    ("md", """# Cat, Dog, or Neither — Capstone notebook

This notebook drives the `catdog` package. All real logic lives in the package, so the CLI, the web app and this notebook
behave identically.

| Section | What it does | Runs on assessment day? |
|---|---|---|
| 0. Setup | install, import, choose `TRAIN` | ✅ |
| 1. Look at the data | grids, class lists, exclusions | only if `TRAIN` |
| 2. Approach B (baseline + evidence) | two-class model + threshold study | only if `TRAIN` |
| 3. Approach A (shipped model) | three-class model + tuned threshold | only if `TRAIN` |
| 4. Own held-out set | confusion matrix on *your* photos | ✅ (fast) |
| 5. Reload proof | loads `model.keras` in a fresh process | ✅ |
| **6. ASSESSMENT CELL** | **change one line — the folder — and run** | ✅ |

**Runtime → Change runtime type → GPU** before training. With `TRAIN = False`, *Restart and run all* never retrains."""),
    ("code", """# ---- 0. Setup -------------------------------------------------------------
TRAIN = False          # True only when you want to (re)train. Assessment day: False.
REPO_URL = "https://github.com/<your-org>/<your-repo>.git"   # <- your class repository
PROJECT = "/content/catdog"                                  # where the project lives in Colab

import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not os.path.exists(PROJECT):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, PROJECT], check=True)
if not IN_COLAB:
    PROJECT = os.path.abspath("..")  # running locally from notebooks/
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow-datasets", "pillow-heif"], check=True)

import tensorflow as tf
print("TensorFlow", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU") or "none")
print("Project:", PROJECT)"""),
    ("code", """# Optional: keep downloads + trained model on Google Drive so a runtime reset loses nothing.
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["TFDS_DATA_DIR"] = "/content/drive/MyDrive/catdog/tfds\""""),
    ("md", """## 1. Look at the data
*Do not skip looking at your own data.* We read every Caltech-101 class name and decide what to exclude; the decisions
live in `catdog/data/exclusions.py` with a reason for each."""),
    ("code", """from catdog.data.exclusions import CALTECH101_EXCLUDE, CALTECH101_KEEP_NOTES, IMAGENETTE_EXCLUDE
if TRAIN:
    import tensorflow_datasets as tfds
    _, info = tfds.load("caltech101", split="train", with_info=True)
    names = info.features["label"].names
    print(f"Caltech-101 has {len(names)} classes:\\n", ", ".join(names))
print("\\nEXCLUDED from 'neither':")
for k, why in {**CALTECH101_EXCLUDE, **IMAGENETTE_EXCLUDE}.items():
    print(f"  {k:<20} {why}")
print("\\nKept on purpose:")
for k, why in CALTECH101_KEEP_NOTES.items():
    print(f"  {k:<20} {why}")"""),
    ("code", """import matplotlib.pyplot as plt
from catdog.config import LABELS
from catdog.data.sources import load_source

def show_grid(source, n=12, title=None):
    ds = load_source(source, seed=1, limit=n)
    fig, axes = plt.subplots(2, n // 2, figsize=(2 * n // 2, 4.4))
    for ax, (img, y) in zip(axes.flat, ds):
        ax.imshow(img.numpy()); ax.set_title(LABELS[int(y)], fontsize=9); ax.axis("off")
    fig.suptitle(title or source); plt.tight_layout(); plt.show()

if TRAIN:
    show_grid("oxford_iiit_pet", title="Oxford-IIIT Pet (species label)")
    show_grid("caltech101", title="Caltech-101 negatives — notice the plain backgrounds")
    show_grid("imagenette", title="Imagenette negatives — real scenes, clutter")"""),
    ("md", """### Start small
Fifty images per source, one epoch. A bug found here costs a minute, not an evening."""),
    ("code", """from catdog.config import TrainConfig
from catdog.modeling.train import train
if TRAIN:
    smoke = train(TrainConfig(limit=50, head_epochs=1, finetune_epochs=0), out_dir="artifacts/smoke")"""),
    ("md", """## 2. Approach B — two classes + a threshold (the evidence)
Train cats-vs-dogs only, then ask: does "top probability below *t* ⇒ neither" actually separate pets from everything else?
The study reports, for every threshold, how many pets survive and how many negatives are caught."""),
    ("code", """import json, numpy as np
if TRAIN:
    b = train(TrainConfig(finetune_epochs=0), two_class=True)
b_path = "artifacts/reports/approach_b.json"
if os.path.exists(b_path):
    b = json.load(open(b_path)); s = b["threshold_study"]
    sweep = s["sweep"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot([r["threshold"] for r in sweep], [r["pet_recall"] for r in sweep], label="pets kept as cat/dog")
    ax.plot([r["threshold"] for r in sweep], [r["neither_recall"] for r in sweep], label="negatives caught as neither")
    ax.axvline(s["best"]["threshold"], ls="--", c="gray"); ax.set_xlabel("threshold on max probability"); ax.legend(); plt.show()
    print(f"Cat-vs-dog accuracy: {b['pet_accuracy']:.1%} | AUROC pets-vs-negatives: {s['auroc']:.3f}")
    print(f"Best threshold {s['best']['threshold']:.2f}: keeps {s['best']['pet_recall']:.1%} of pets, catches {s['best']['neither_recall']:.1%} of negatives")
    print(f"{s['negatives_confident_over_90']:.1%} of negatives were called cat/dog with >90% confidence")"""),
    ("md", """**Write your conclusion here** (this is marked): e.g. *"With B, the best threshold still sent N of every 100 negatives
into the cat/dog piles, many at >90% confidence — confident and wrong. So we moved to A."*"""),
    ("md", """## 3. Approach A — three classes (the shipped model)
`neither` is a trained class built from Caltech-101 (minus pet-like classes), Imagenette (minus its dog class) and
**your own photos** in `data/own/{cat,dog,neither}/`. A safety-net threshold is then tuned on validation data."""),
    ("code", """from catdog.evaluation.metrics import EvaluationReport, ClassReport, format_report
from IPython.display import Image as IPImage, display
if TRAIN:
    meta = train(TrainConfig())
meta = json.load(open("artifacts/model_meta.json"))
print("Train counts per source:", json.dumps(meta["train_counts"], indent=1))
v = meta["validation"]["tuned"]
print(format_report(EvaluationReport(**{**v, "per_class": [ClassReport(**c) for c in v["per_class"]]})))
if os.path.exists("artifacts/reports/validation_confusion.png"):
    display(IPImage("artifacts/reports/validation_confusion.png", width=420))"""),
    ("md", """## 4. Your own held-out set — the number that matters
Put photos the model has **never** trained on in `data/holdout/{cat,dog,neither}/` (at least 20, from your phone, friends,
anywhere but the training data). This is the matrix you bring on the day."""),
    ("code", """!python -m catdog evaluate data/holdout
if os.path.exists("artifacts/reports/holdout_confusion.png"):
    display(IPImage("artifacts/reports/holdout_confusion.png", width=420))"""),
    ("md", """## 5. Reload proof
A separate Python process loads `model.keras` from disk — nothing from this notebook's memory is used."""),
    ("code", """!python -m catdog info | head -n 20"""),
    ("md", """---
## 6. 🎯 ASSESSMENT CELL
Change **one line** — the folder — and run. Handles any size, shape, PNG/JPEG/HEIC, portrait/landscape; unreadable files
become an `error` row instead of crashing."""),
    ("code", """FOLDER = "/content/test_images"   # <- the only line you change

import sys; sys.path.insert(0, PROJECT)
from catdog.inference import predict_folder_table
predictions = predict_folder_table(FOLDER)"""),
    ("md", """## 7. Break it on purpose
List the photos that defeat it, and why you think they do (lighting, a cat that is tiny in the frame, a goat, a fox,
a dog-shaped toy…). That list is part of your presentation.

| photo | true | predicted | why we think it failed |
|---|---|---|---|
| | | | |"""),
]


def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "cells": cells,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "notebooks" / "capstone.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
