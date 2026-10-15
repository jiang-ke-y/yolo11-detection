"""Run YOLO11 inference on images, videos, or folders.

Examples
--------
python src/predict.py --weights runs/detect/train/weights/best.pt --source assets/sample.jpg
python src/predict.py --model yolo11n.pt --source ./images --conf 0.4 --save-txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 inference.")
    parser.add_argument("--weights", "--model", dest="weights", type=str, default="yolo11n.pt",
                        help="Path to .pt weights (trained best.pt or a pretrained name).")
    parser.add_argument("--source", type=str, required=True,
                        help="Image / video / folder / URL to run detection on.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu / mps (default auto).")
    parser.add_argument("--save-txt", action="store_true", help="Save detections as YOLO .txt labels.")
    parser.add_argument("--nosave", action="store_true", help="Do not write annotated media to disk.")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Filter to these class ids.")
    parser.add_argument("--line-thickness", type=int, default=2, help="Box line thickness.")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=not args.nosave,
        save_txt=args.save_txt,
        classes=args.classes,
        line_width=args.line_thickness,
    )

    # Console summary of detections per image
    for r in results:
        names = r.names
        boxes = r.boxes
        print(f"\n=== {r.path} ===")
        if boxes is None or len(boxes) == 0:
            print("  no detections")
            continue
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            print(f"  {names[int(cls_id)]:<15} conf={conf:.2f}")

    print(f"\n[predict] processed {len(results)} item(s). "
          f"Annotated outputs saved under runs/detect/predict (unless --nosave).")


if __name__ == "__main__":
    main()
