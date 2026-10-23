PY ?= .venv/bin/python
FOLDER ?= data/holdout

.PHONY: help install test serve predict evaluate smoke train train-b

help:          ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:       ## Create .venv and install everything
	python3 -m venv .venv && $(PY) -m pip install -r requirements-dev.txt

test:          ## Run the test suite
	$(PY) -m pytest

serve:         ## Start the web interface on http://127.0.0.1:8000
	$(PY) -m catdog serve

predict:       ## Print the table for a folder:  make predict FOLDER=path/to/photos
	$(PY) -m catdog predict $(FOLDER) --recursive

evaluate:      ## Confusion matrix on data/holdout/{cat,dog,neither}
	$(PY) -m catdog evaluate $(FOLDER)

smoke:         ## 50 images per source, 1 epoch: does the pipeline run end to end?
	$(PY) -m catdog train --limit 50 --head-epochs 1 --finetune-epochs 0 --model-dir artifacts/smoke

train:         ## Full Approach-A training (use a GPU / Colab)
	$(PY) -m catdog train

train-b:       ## Approach-B experiment: two classes + threshold study
	$(PY) -m catdog train --two-class
