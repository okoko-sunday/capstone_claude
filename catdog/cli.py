"""Command line entry point.

    python -m catdog predict  path/to/folder          # the assessment-day table
    python -m catdog evaluate data/holdout            # confusion matrix on labelled folders
    python -m catdog train    [--two-class] [--limit 50]
    python -m catdog serve    [--port 8000]           # web interface
    python -m catdog info                              # what model is loaded
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from catdog.config import ARTIFACTS_DIR, HOLDOUT_DIR, LABELS, TrainConfig

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def cmd_predict(args) -> int:
    from catdog.inference.predictor import Predictor, format_table

    predictor = Predictor.load(args.model_dir)
    predictions = predictor.predict_folder(args.folder, recursive=args.recursive)
    print(format_table(predictions))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "prediction", "confidence", *(f"p_{label}" for label in LABELS), "decided_by"])
            for p in predictions:
                writer.writerow(
                    [p.filename, p.label, f"{p.confidence:.4f}", *(f"{p.probabilities.get(label, 0):.4f}" for label in LABELS), p.decided_by]
                )
        print(f"Saved {args.csv}")
    return 0


def cmd_evaluate(args) -> int:
    from catdog.evaluation.metrics import evaluate, format_report, plot_confusion_matrix
    from catdog.inference.image_io import load_labelled_folder
    from catdog.inference.predictor import Predictor

    root = Path(args.folder)
    items = load_labelled_folder(root)
    if not items:
        print(f"No images in {root}/{{cat,dog,neither}}/", file=sys.stderr)
        return 1
    predictor = Predictor.load(args.model_dir)
    predictions = predictor.predict_sources([(str(p.relative_to(root)), p) for p, _ in items])
    y_true = [LABELS[y] for _, y in items]
    report = evaluate(y_true, [p.label for p in predictions])
    print(format_report(report))

    wrong = [(p, t) for p, t in zip(predictions, y_true) if p.label != t]
    if wrong:
        print("\nMistakes (write these down — they are part of your presentation):")
        for p, t in wrong:
            print(f"  {p.filename:<40} true={t:<8} predicted={p.label:<8} {p.confidence:.1%}  probs={_fmt(p.probabilities)}")

    reports = Path(args.model_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "holdout.json").write_text(
        json.dumps({"report": report.to_dict(), "predictions": [p.to_dict() for p in predictions], "y_true": y_true}, indent=2)
    )
    try:
        plot_confusion_matrix(report, reports / "holdout_confusion.png", "Own held-out set")
    except ImportError:
        pass
    print(f"\nSaved {reports / 'holdout.json'}")
    return 0


def cmd_train(args) -> int:
    from catdog.modeling.train import train

    cfg = TrainConfig(limit=args.limit, batch_size=args.batch_size, backbone=args.backbone)
    if args.sources is not None:
        cfg.sources = args.sources
    if args.head_epochs is not None:
        cfg.head_epochs = args.head_epochs
    if args.finetune_epochs is not None:
        cfg.finetune_epochs = args.finetune_epochs
    train(cfg, out_dir=args.model_dir, two_class=args.two_class, pretrained=not args.no_pretrained, tfds_dir=args.tfds_dir)
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    os.environ["CATDOG_MODEL_DIR"] = str(args.model_dir)
    uvicorn.run("catdog.server.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_info(args) -> int:
    from catdog.inference.predictor import Predictor

    info = Predictor.load(args.model_dir).info()
    for key in ("history", "val_animal_prob_hist"):
        info.pop(key, None)
    print(json.dumps(info, indent=2))
    return 0


def _fmt(probs: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.2f}" for k, v in probs.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="catdog", description="Cat, Dog, or Neither")
    parser.add_argument("--model-dir", type=Path, default=ARTIFACTS_DIR, help="folder holding model.keras")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="print the prediction table for a folder of images")
    p.add_argument("folder", type=Path)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--csv", type=Path, help="also save the table as CSV")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="confusion matrix on a labelled folder: <folder>/{cat,dog,neither}/")
    p.add_argument("folder", type=Path, nargs="?", default=HOLDOUT_DIR)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="train a model (use Colab with a GPU for the full run)")
    p.add_argument("--two-class", action="store_true", help="Approach B experiment: cats vs dogs + threshold study")
    p.add_argument("--limit", type=int, help="images per source — start with 50 to test the pipeline")
    p.add_argument("--sources", nargs="*", help="default: oxford_iiit_pet caltech101 imagenette own")
    p.add_argument("--backbone", default="efficientnetv2b0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int)
    p.add_argument("--finetune-epochs", type=int)
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    p.add_argument("--tfds-dir", help="where tensorflow_datasets stores downloads")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("serve", help="start the web interface")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("info", help="show the loaded model's metadata")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
