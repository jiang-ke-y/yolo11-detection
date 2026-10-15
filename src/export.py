"""Export a trained YOLO11 model to a deployment format.

Examples
--------
python src/export.py --weights runs/detect/train/weights/best.pt --format onnx
python src/export.py --weights yolo11n.pt --format engine --half
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUPPORTED = ["onnx", "engine", "openvino", "coreml", "tflite", "torchscript", "pb", "saved_model"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO11 weights for deployment.")
    parser.add_argument("--weights", "--model", dest="weights", type=str, required=True)
    parser.add_argument("--format", type=str, default="onnx", choices=SUPPORTED)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 export (GPU only).")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes (ONNX).")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph.")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version.")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.weights)

    export_kwargs = dict(format=args.format, imgsz=args.imgsz, half=args.half)
    if args.format == "onnx":
        export_kwargs.update(dynamic=args.dynamic, simplify=args.simplify)
        if args.opset:
            export_kwargs["opset"] = args.opset

    path = model.export(**export_kwargs)
    print(f"[export] exported to: {path}")


if __name__ == "__main__":
    main()
