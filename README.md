# Cat, Dog, or Neither

Sorts a folder of photographs into three piles — **dog**, **cat** and **neither** — and is honest when a photo is
neither. Tech4Youth AI Engineering capstone.

```
filename        prediction  confidence
test_01.jpg     dog              94.2%
test_02.jpg     neither          71.8%
test_03.jpg     cat              88.0%
```

## What's in here

| Path | What it is |
|---|---|
| `notebooks/capstone.ipynb` | Colab notebook: look at data → Approach B evidence → train Approach A → hold-out matrix → **assessment cell** |
| `catdog/` | The Python package: data, model, training, inference, metrics, API |
| `web/` | Web interface (Classify · Evaluate · Model card) |
| `artifacts/` | `model.keras` + `model_meta.json` + `reports/` (the saved model the brief requires) |
| `data/own/{cat,dog,neither}/` | Your own photos, mixed into training |
| `data/holdout/{cat,dog,neither}/` | Your held-out photos — never trained on |
| `docs/ARCHITECTURE.md` | System design, data flow, decisions |
| `tests/` | pytest suite |

## Quick start

### Train (Colab, GPU)
1. In Google Drive, put your photos in `MyDrive/catdog/own/neither/` (30–40+ photos of your surroundings) and
   `MyDrive/catdog/holdout/{cat,dog,neither}/` (20+ photos the model never trains on). The notebook creates the
   folders on its first run if they don't exist yet.
2. Open `notebooks/capstone.ipynb` in Colab (File → Open notebook → GitHub), **Runtime → Change runtime type → T4 GPU**.
   `REPO_URL` already points at this repository. If the repo is private, add a Colab secret named `GITHUB_TOKEN`.
3. Set `TRAIN = True`, then **Runtime → Run all**. Allow Google Drive access when asked. Downloads ≈ 1.1 GB the first
   time; training takes roughly 30–60 minutes.
4. Section 3b saves the model to `MyDrive/catdog/artifacts/` and downloads `catdog_artifacts.zip`. Unzip it into this
   project's `artifacts/` folder and commit `model.keras`, `model_meta.json` and `reports/`.
5. Set `TRAIN = False`. From now on *Restart and run all* only loads the model (from the repo or from Drive).

### Run locally
```bash
make install                                  # .venv + dependencies
make test                                     # 30-ish tests, ~20 s
make serve                                    # http://127.0.0.1:8000
python -m catdog predict path/to/folder       # the table
python -m catdog evaluate data/holdout        # your confusion matrix
```

### Assessment day
Notebook section 6, change one line:
```python
FOLDER = "/content/test_images"
```
or in a terminal: `python -m catdog predict /path/to/test_images`, or drop the folder onto the web page and name
sub-folders `cat/ dog/ neither/` to get the confusion matrix built for you.

## Decisions (and the evidence for them)

**"Neither" approach — A (three classes) with a tuned threshold safety net.**
Approach B (cats vs dogs + "low confidence ⇒ neither") is implemented as an experiment, not just dismissed:
`python -m catdog train --two-class` writes `artifacts/reports/approach_b.json` with a full threshold sweep.
Measured on the validation set (1,104 pets, 909 negatives):

> *With B, the cat-vs-dog model itself was 99.5% accurate, but the best threshold (0.91) kept only 90.9% of pets
> (about 100 real cats and dogs pushed into "neither") and still let 5.3% of negatives through. Worse, 7.6% of the
> negatives (69 of 909 photos of things that are not pets) were called cat or dog with **more than 90% confidence** —
> confident and wrong, so no threshold can catch them. So we moved to A.*

| | Approach B (2 classes + threshold) | Approach A (3 classes + safety net) — shipped |
|---|---|---|
| Validation result | balanced accuracy 92.8% at its best threshold | **2,006 / 2,013 correct = 99.7%** |
| Negatives called cat/dog | 5.3% at best threshold; 7.6% at >90% confidence | 1 of 909 (0.1%) |
| Threshold | 0.91 on the top probability | 0.34 on max(cat, dog) — tuned for macro-F1 |

Approach A validation matrix (`artifacts/reports/validation.txt`):

```
                        PREDICTED
                     cat      dog  neither
            cat      375        2        1   = 378
TRUE        dog        3      723        0   = 726
        neither        1        0      908   = 909
```

These validation photos come from the **same sources** as training, so 99.7% is an upper bound, not what to
expect on phone photos — see the hold-out matrix below.

**Data.**
| Source | Role | Kept | Excluded (why) |
|---|---|---|---|
| Oxford-IIIT Pet | cat / dog | 1,993 cat + 4,252 dog trained (378 + 726 validation) | — |
| Caltech-101 | neither | 2,550 trained (450 validation), sampled | `dalmatian` (dog), `snoopy`/`garfield` (cartoon pets), `wild_cat` (looks domestic), `background_google` (unlabelled clutter) |
| Imagenette | neither | 2,550 trained (450 validation), sampled | `n02102040` English springer (dog) |
| Extra "neither" photos | neither | 51 trained (9 validation) | Freely licensed photos of Nigerian streets, markets, food, rooms and farm animals from Openverse; every photo's creator and licence is listed in `catdog_photos/SOURCES.csv` |

**Totals:** 11,396 training images (1,993 cat · 4,252 dog · 5,151 neither) and 2,013 validation images. Training
took about 7 minutes on a Colab T4 GPU (TensorFlow 2.21, Keras 3.13).

Kept on purpose: roosters, llamas, leopards, cougars, people — real "neither" cases that look a bit like pets. Full
reasoning: `catdog/data/exclusions.py`.

**Why diverse negatives:** Caltech-101 is mostly objects on plain backgrounds; alone, it teaches "neither = plain
background". Imagenette adds cluttered real scenes and the extra photos add Nigerian surroundings.

**Pretrained, not from scratch:** EfficientNetV2-B0 on ImageNet. A few thousand images cannot teach edges and fur from
zero; transfer learning generalises far better to phone photos.

## Known limits (say these before you're asked)

* Validation numbers come from the *same sources* as training; expect lower accuracy on real phone photos. The number
  that matters is the hold-out matrix in `artifacts/reports/holdout.json`.
* **Confidence is not calibrated.** "94%" is the softmax score, not "right 94% of the time".
* **Wolves, foxes, jackals** are likely to be called *dog*; **big wild cats** close-up may be called *cat*. Ideally they
  are "neither"; there are few such hard negatives in training.
* **Cat and dog in one photo** → one label only (single softmax). Multi-label would need sigmoid outputs + binary
  cross-entropy.
* Tiny animals in a large scene, heavy darkness, and toy/cartoon/printed animals are likely failure modes — test them.
* "Neither" is defined by what was in the negative set; an unusual object no source covered can still be pulled toward cat/dog.

## Your own measurements

Run `python -m catdog evaluate data/holdout` and paste the matrix here:

```
(paste here)
```
