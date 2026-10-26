# Assessment guide — Cat, Dog, or Neither

Everything to present, run and defend on assessment day, with the real numbers from training on
26 October 2026. Read it top to bottom once, then use the sections as a cheat sheet.

---

## 1. The 5-minute presentation (say it roughly like this)

**① What I built — 30 s**
> "A system that sorts any folder of photos into three piles: dog, cat, or neither. You give it a folder, it prints
> one row per photo — filename, prediction, confidence. It runs in Colab, from the command line, or in a web app."

**② The data — 1 min**
> "Cats and dogs come from Oxford-IIIT Pet: 1,993 cats and 4,252 dogs for training. 'Neither' comes from Caltech-101
> and Imagenette — 2,550 each — plus 60 photos of Nigerian streets, markets, food, rooms and farm animals.
> In total 11,396 training photos and 2,013 for validation.
>
> I printed Caltech-101's class list and removed five classes: **dalmatian**, because it's a dog breed — labelling it
> 'neither' would teach the model that dogs aren't dogs; **snoopy** and **garfield**, cartoon pets; **wild_cat**,
> because they look like house cats; and **background_google**, random clutter that contains pets. From Imagenette I
> removed the **English springer spaniel** class, also a dog.
>
> I deliberately *kept* roosters, llamas, leopards, cougars and people — real 'neither' things that look a bit like
> pets, so the model learns that 'animal' doesn't mean 'pet'."

**③ The hard part: "neither" — 1½ min**
> "A model trained on only cats and dogs has no way to say 'I don't know' — softmax forces its scores to add up to
> 100%, so a plate of jollof rice still comes out as 'dog, 96%'.
>
> I tried **Approach B** first: train cats-vs-dogs, then call anything below a confidence threshold 'neither'. The
> cat-vs-dog part was 99.5% accurate, but even at the best threshold, 0.91, it threw away about 100 real pets and
> still let 5.3% of non-pets through. Worse, **69 of 909 non-pet photos were called cat or dog with more than 90%
> confidence** — confident and wrong, so no threshold can catch them.
>
> So I moved to **Approach A**: three classes, with 'neither' as a real trained class, plus a safety-net threshold
> of 0.34 tuned on validation data. On validation it got 2,006 of 2,013 right, and only 1 of 909 non-pets was
> called a pet."

**④ The model — 30 s**
> "I started from **EfficientNetV2-B0**, pretrained on ImageNet, and replaced its last layer with a 3-output softmax
> layer: cat, dog, neither. Pretrained because 11,000 photos can't teach a network to see from scratch. Training
> took about 7 minutes on a Colab GPU: 6 epochs training the new layer, then 4 epochs fine-tuning the top 40 layers."

**⑤ Results and limits — 1½ min**
> "On validation it scored 99.7%, but those photos come from the same collections as training, so that number is
> too optimistic. The number that matters is my **hold-out set**: 39 photos the model never saw. It got **38 of 39,
> 97.4%**.
>
> The one mistake was a **goat**, called 'dog' at 56%. A goat is four-legged, furry, dog-sized, usually photographed
> outdoors — exactly what most dog photos look like. Its 'neither' score was 38%, so the model was unsure, but 56%
> is above my 0.34 threshold, so the safety net didn't catch it. The fix is more goats, sheep and other farm animals
> in the 'neither' training set.
>
> Limits I know about: the confidence isn't calibrated — 94% doesn't mean right 94% of the time. A photo with a cat
> *and* a dog gets only one label. Wolves and foxes would probably be called dog. And my hold-out photos are web
> photos, so on real phone photos — dim rooms, clutter — I expect it to do worse than 97%."

---

## 2. Your numbers at a glance

| What | Number |
|---|---|
| Training photos | 11,396 — cat 1,993 · dog 4,252 · neither 5,151 |
| ↳ "neither" sources | Caltech-101 2,550 · Imagenette 2,550 · Nigerian photos 51 (+9 in validation) |
| Validation photos | 2,013 — cat 378 · dog 726 · neither 909 |
| Backbone | EfficientNetV2-B0, pretrained on ImageNet |
| Input | 224 × 224, letterboxed (padded, never stretched or cropped) |
| Training | 6 epochs (new head) + 4 epochs (fine-tune top 40 layers) · about 7 minutes on a T4 GPU |
| Safety-net threshold | 0.34 on max(cat, dog) — below it, the answer becomes "neither" |
| Validation accuracy (Approach A) | **2,006 / 2,013 = 99.7%** |
| Hold-out accuracy (the real test) | **38 / 39 = 97.4%** |
| Approach B, best threshold 0.91 | kept 90.9% of pets, caught 94.7% of negatives |
| Approach B, confident and wrong | **7.6% of negatives (69 of 909) called cat/dog at >90% confidence** |

---

## 3. Your confusion matrices

**Hold-out set — 39 photos never trained on (bring this one):**
```
                 PREDICTED
             cat   dog  neither
TRUE  cat     12     0     0    = 12
      dog      0    12     0    = 12
  neither      0     1    14    = 15
```

| Class | Precision (of everything it *called* X, how much was X?) | Recall (of the real X, how many did it find?) |
|---|---|---|
| cat | 12/12 = 100% | 12/12 = 100% |
| dog | 12/13 = 92.3% (the goat was called dog) | 12/12 = 100% |
| neither | 14/14 = 100% | 14/15 = 93.3% (missed the goat) |

**Validation set — 2,013 photos from the training sources:**
```
                 PREDICTED
             cat   dog  neither
TRUE  cat    375     2     1    = 378
      dog      3   723     0    = 726
  neither      1     0   908    = 909
```

---

## 4. On the day: drawing the matrix for the 15 photos

1. Draw the 3 × 3 grid. **Rows = what the photo really is. Columns = what the system said.**
2. For each photo, put a tally in row *true*, column *predicted*.
3. **Accuracy** = diagonal total ÷ 15.
4. **Recall for dog** = dog-row diagonal ÷ 5 (of the 5 real dogs, how many found?). Read *across* the row.
5. **Precision for dog** = dog-column diagonal ÷ column total (of everything called dog, how much really was?). Read *down* the column.

Example: if the diagonal is 5, 4 and 4, accuracy = 13/15 = 86.7%. Each mistake costs 6.7 points.

Then explain each wrong cell the way you explained the goat: *what the photo is → why it looks like the wrong class
→ what you'd change.*

---

## 5. Questions you'll be asked — and your answers

**How many images did you train on, and how many of each class?**
11,396: 1,993 cats, 4,252 dogs, 5,151 neither. Plus 2,013 for validation.

**Where did your "neither" images come from, and what did you leave out?**
Caltech-101 and Imagenette, 2,550 each, plus 60 photos of Nigerian surroundings. Left out: dalmatian (a dog),
snoopy and garfield (cartoon pets), wild_cat (looks like a house cat), background_google (clutter with pets in it),
and Imagenette's English springer spaniel (a dog). Kept on purpose: roosters, llamas, leopards, cougars, people.

**Did you train from scratch or start from a pretrained model? Why?**
Pretrained: EfficientNetV2-B0 trained on ImageNet's 1.2 million photos. It already knows edges, fur, eyes and
shapes; I only replaced the last layer with my 3 classes and fine-tuned the top. 11,000 photos are far too few to
learn vision from zero, and transfer learning works much better on new kinds of photos.

**What's your accuracy on your own held-out set, and why is today's number different?**
38/39 = 97.4%. Today's photos are new, taken on a different camera, in different light, probably ordinary Nigerian
surroundings — harder than my web photos. And with only 15 photos, one mistake moves accuracy by 6.7 points.

**Show me a photograph your system gets wrong. Why?**
The goat (`neither_008.jpg`): called dog at 56%. Four legs, fur, dog-sized, outdoors — it matches what most dog
photos look like, and there are only a few goats among 5,151 "neither" training photos. The model was unsure (38%
neither), but 56% is above my 0.34 threshold, so the safety net didn't catch it. Fix: add more goats, sheep and farm
animals to the "neither" training set.

**Your system says 94% — what does that mean? Is it right 94% of the time?**
No. It's the softmax score: how strongly the model leans towards that class compared with the other two. It's not
a calibrated probability of being correct. Models can be confident and wrong — my Approach B results show exactly
that: 69 non-pet photos were called pets at over 90% confidence.

**If I gave you a photo of a wolf or a fox, what would it say? What should it say?**
It would probably say **dog**: they look very dog-like and there are almost none in my training data. It *should*
say **neither**, because a wolf isn't a pet dog. To fix it I'd add wolves, foxes and jackals to the "neither" set.

**Why did you choose Approach A over B?**
Because I measured B and it failed in the way the brief warns about: at its best threshold it lost about 100 pets
and still let 5.3% of non-pets through, and 7.6% of non-pets were called pets at over 90% confidence, which no
threshold can catch. A learns "neither" as a real class: 1 of 909 non-pets wrong on validation.

**Which of these mistakes would matter most, and why?**
It depends what the system is for. If it alerts someone that a dog is loose on a property, calling an empty yard or
a goat "dog" is the worst mistake: false alarms make people ignore or switch off the system. Missing a real dog is
bad but rarer. That's why I keep the safety net: when the model isn't confident it's a pet, it says "neither". For
that use I'd raise the threshold, accepting that a few real dogs get missed in exchange for fewer false alarms.

**What would you need to deploy this for real?**
Many more photos from the real setting (phone cameras, Nigerian homes and streets, dim light, farm animals);
calibrated confidence so the percentage means something; a clear "unsure" answer for low confidence; monitoring of
real mistakes and regular retraining. The serving part already exists: a FastAPI web service with a web interface.

**What happens if a photo has a cat and a dog?**
It gives one label, usually the bigger or clearer animal, often with lower confidence because the scores are split.
Softmax makes the classes compete for one 100%. To return both, I'd switch the last layer to sigmoid (an independent
yes/no per class) and train with binary cross-entropy on photos labelled with both animals.

**What's softmax?**
It turns the model's three raw scores into percentages that add up to 100%, so the classes compete and only one wins.

**What's the threshold for?**
A safety net. If the strongest of the cat and dog scores is below 0.34, the answer becomes "neither" even if cat or
dog was the top score. It was tuned automatically on validation data for the best balance across all three classes.

---

## 6. Known limits — say these *before* you're asked

1. **Confidence isn't calibrated.** "94%" is a score, not "right 94% of the time".
2. **One label per photo.** A cat and a dog together gets only one answer.
3. **Dog-like animals.** Goats (my one hold-out mistake), wolves, foxes and jackals can be called dog; big wild cats
   close up could be called cat.
4. **My test photos were web photos.** Bright and well framed; real phone photos in dim, cluttered rooms will be harder.
5. **Small animals in a big scene, very dark photos, toys, cartoons and printed pictures** are likely failure modes.
6. **Validation's 99.7% is optimistic** because it comes from the same collections as training.

---

## 7. Break it on purpose — your list so far

| Photo | True | Predicted | Why we think it failed |
|---|---|---|---|
| `holdout/neither/neither_008.jpg` (a goat) | neither | dog (56.3%) | Four-legged, furry, dog-sized, outdoors; few goats in training; above the 0.34 threshold |

Worth adding before the day: test 10–20 of **your own phone photos** (dim room, a goat or chicken nearby, a cluttered
table, a toy or a printed picture of a dog) and add any failures to this table.

---

## 8. Assessment-day runbook

**Before you leave home**
- Charge the laptop; make sure you have data or Wi-Fi for Colab.
- Open the Colab notebook once, with `TRAIN = False`, and **Runtime → Run all**. It should reach the end, with only
  the expected `Not a folder: /content/test_images` error in section 6.
- Optional: `make serve` on the laptop, open http://127.0.0.1:8000, and check the green dot.

**When you get the 15 photos — Colab**
1. Copy them into one folder on your laptop.
2. In Colab: folder icon on the left → right-click → **New folder** → `test_images` → drag the photos into it
   (or set `UPLOAD = True` in the cell above section 6 and pick the files).
3. In section 6, keep `FOLDER = "/content/test_images"` and press **Shift + Enter**.
4. Read out the table: filename, prediction, confidence.

**Or — the web app**
1. Drop the folder onto the glowing upload box (or click **Choose folder**).
2. Click the true label (cat / dog / neither) on each card.
3. Open **Evaluation**: confusion matrix, accuracy, precision and recall, and the mistakes, ready to explain.

**At the board**
Draw the matrix by hand from the table first, compute accuracy, recall and precision, then explain each wrong cell.
Use the web app's Evaluation page only to double-check your arithmetic.

---

## 9. Files to point to if asked

| Question | File |
|---|---|
| Where is the saved model? | `artifacts/model.keras` (+ `artifacts/model_meta.json`) |
| Where are the excluded classes and reasons? | `catdog/data/exclusions.py` |
| Where is the last layer / softmax? | `catdog/modeling/model.py`, the `Dense(3, activation="softmax")` line |
| Where is the threshold rule? | `catdog/inference/predictor.py`, function `decide` |
| Approach B evidence | `artifacts/reports/approach_b.json` |
| Validation matrix | `artifacts/reports/validation.txt` |
| Where did the extra photos come from? | `catdog_photos/SOURCES.csv` (creator and licence for each) |
