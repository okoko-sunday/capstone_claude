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
1. Push this folder to the class repository and set `REPO_URL` in the first notebook cell.
2. Open `notebooks/capstone.ipynb` in Colab, **Runtime → Change runtime type → GPU**.
3. Set `TRAIN = True`, run all. Downloads ≈ 1.1 GB (Pets 774 MB, Caltech 131 MB, Imagenette 326 MB) the first time.
4. Download `artifacts/model.keras`, `artifacts/model_meta.json` and `artifacts/reports/` and commit them.
5. Set `TRAIN = False` again. From now on *Restart and run all* only loads the model.

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
`python -m catdog train --two-class` writes `artifacts/reports/approach_b.json` with a full threshold sweep. Fill in
your measured numbers here after running it:

> *With B, the best threshold (__) kept __% of pets but caught only __% of negatives; __% of negatives were called
> cat/dog with >90% confidence. So we moved to A.*

**Data.**
| Source | Role | Kept | Excluded (why) |
|---|---|---|---|
| Oxford-IIIT Pet | cat / dog | all 7,349, `species` label | — |
| Caltech-101 | neither | ≤3,000 sampled | `dalmatian` (dog), `snoopy`/`garfield` (cartoon pets), `wild_cat` (looks domestic), `background_google` (unlabelled clutter) |
| Imagenette | neither | ≤3,000 sampled | `n02102040` English springer (dog) |
| Own photos | all three | all | — |

Kept on purpose: roosters, llamas, leopards, cougars, people — real "neither" cases that look a bit like pets. Full
reasoning: `catdog/data/exclusions.py`.

**Why diverse negatives:** Caltech-101 is mostly objects on plain backgrounds; alone, it teaches "neither = plain
background". Imagenette adds cluttered real scenes and your own photos add Nigerian surroundings.

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
