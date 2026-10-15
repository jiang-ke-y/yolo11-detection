"""Validate a YOLO11 model and report mAP metrics.

Examples
--------
python src/evaluate.py --weights runs/detect/train/weights/best.pt --data coco8.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 validation / evaluation.")
    parser.add_argument("--weights", "--model", dest="weights", type=str, default="yolo11n.pt")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="Dataset YAML to validate on.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold for metrics.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.weights)

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        device=args.device,
        split=args.split,
    )

    print("\n================= Evaluation Summary =================")
    print(f"  mAP50-95 (all classes): {results.box.map:.4f}")
    print(f"  mAP50               (all classes): {results.box.map50:.4f}")
    print(f"  mAP75               (all classes): {results.box.map75:.4f}")
    print("  -----------------------------------------------")
    try:
        for i, ap50 in enumerate(results.box.maps):
            name = results.names[i] if i < len(results.names) else f"class_{i}"
            print(f"    {name:<18} mAP50-95={ap50:.4f}")
    except Exception:
        pass
    print("======================================================")
    return results


if __name__ == "__main__":
    main()
