"""Train a YOLO11 model.

Examples
--------
# Train on the built-in COCO8 sample dataset
python src/train.py --data coco8.yaml --model yolo11n.pt --epochs 50

# Train from a YAML config (overrides win over config defaults)
python src/train.py --config configs/default.yaml --data data/dataset.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a script from the repo root without packaging install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

DEFAULTS = {
    "model": "yolo11n.pt",
    "data": "coco8.yaml",
    "epochs": 100,
    "imgsz": 640,
    "batch": -1,  # -1 => auto batch size
    "device": None,  # None => auto, or "0" / "cpu" / "mps"
    "workers": 8,
    "project": "runs/detect",
    "name": "train",
    "patience": 50,
    "pretrained": True,
    "optimizer": "auto",
    "seed": 0,
}


def load_config(path: str) -> dict:
    """Load a YAML config file into a dict (empty dict if unavailable)."""
    if yaml is None or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("train", data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO11 object detector.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--model", type=str, help="Pretrained model, e.g. yolo11n.pt")
    parser.add_argument("--data", type=str, help="Dataset YAML, e.g. coco8.yaml or data/dataset.yaml")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, help="Training image size.")
    parser.add_argument("--batch", type=int, help="Batch size (-1 for auto).")
    parser.add_argument("--device", type=str, help="cuda:0 / cpu / mps (default: auto).")
    parser.add_argument("--workers", type=int, help="Number of dataloader workers.")
    parser.add_argument("--name", type=str, help="Experiment name under project/.")
    parser.add_argument("--project", type=str, help="Directory for logged outputs.")
    parser.add_argument("--resume", action="store_true", help="Resume last interrupted training.")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()

    # Precedence: CLI args > config file > built-in defaults
    settings = dict(DEFAULTS)
    if args.config:
        settings.update(load_config(args.config))
    for key, value in vars(args).items():
        if key in settings and value is not None and key != "config":
            settings[key] = value

    print("[train] resolved settings:")
    for k, v in settings.items():
        print(f"    {k}: {v}")

    model = YOLO(settings["model"])
    if settings["pretrained"] and settings["model"].endswith(".pt"):
        model.load(settings["model"])  # load pretrained weights

    train_kwargs = dict(
        data=settings["data"],
        epochs=settings["epochs"],
        imgsz=settings["imgsz"],
        batch=settings["batch"],
        workers=settings["workers"],
        project=settings["project"],
        name=settings["name"],
        patience=settings["patience"],
        optimizer=settings["optimizer"],
        seed=settings["seed"],
        resume=args.resume,
    )
    if settings.get("device"):
        train_kwargs["device"] = settings["device"]

    results = model.train(**train_kwargs)
    print(f"[train] finished. Best weights saved under {settings['project']}/{settings['name']}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
