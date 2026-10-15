# YOLO11 Object Detection

A clean, ready-to-run project scaffold for object detection with **YOLO11** (the latest generation of the YOLO family, powered by [`ultralytics`](https://github.com/ultralytics/ultralytics)). It covers the full workflow: environment setup, dataset preparation, training, inference, and evaluation.

## Features

- Train YOLO11 on the built-in COCO8 sample dataset or on your own custom dataset
- Run inference on images, videos, and folders with a single command
- Evaluate trained models with mAP50 / mAP50-95 metrics
- Export models to ONNX / TensorRT / OpenVINO for deployment
- Configurable training hyper-parameters via a YAML config file
- Google Colab notebook for zero-setup experimentation

## Project Layout

```
yolo11-detection/
├── src/                # Source code (train / predict / evaluate / export)
│   ├── __init__.py
│   ├── train.py        # Training entry point
│   ├── predict.py      # Inference entry point
│   ├── evaluate.py     # Metrics / validation
│   └── export.py       # Model export for deployment
├── configs/            # YAML training configurations
│   └── default.yaml
├── data/               # Dataset location (raw + annotated), git-ignored
│   └── .gitkeep
├── runs/               # Training outputs & weights, git-ignored
│   └── .gitkeep
├── notebooks/          # Colab / Jupyter notebooks
├── assets/             # Sample images and result screenshots
├── requirements.txt    # Python dependencies
├── colab.ipynb         # Google Colab quick-start notebook
└── README.md
```

## Installation

Requires Python 3.9+.

```bash
# Clone the repository
git clone https://github.com/jiang-ke-y/yolo11-detection.git
cd yolo11-detection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

On first run, `ultralytics` automatically downloads the pretrained `yolo11n.pt` weights.

## Quick Start

### 1. Train

Train on the bundled COCO8 sample dataset (auto-downloaded):

```bash
python src/train.py --data coco8.yaml --model yolo11n.pt --epochs 50 --imgsz 640
```

To train on your own dataset, edit `data/dataset.yaml` (see below) and run:

```bash
python src/train.py --config configs/default.yaml --data data/dataset.yaml
```

### 2. Inference

Run detection on an image and save the annotated output:

```bash
python src/predict.py --weights runs/detect/train/weights/best.pt --source assets/sample.jpg
```

You can also pass a folder or a video file as `--source`.

### 3. Evaluate

```bash
python src/evaluate.py --weights runs/detect/train/weights/best.pt --data coco8.yaml
```

### 4. Export

Export the trained model for deployment:

```bash
python src/export.py --weights runs/detect/train/weights/best.pt --format onnx
```

Supported formats: `onnx`, `engine` (TensorRT), `openvino`, `coreml`, `tflite`.

## Custom Dataset

Organize your dataset in the YOLO label format and describe it with a `data/dataset.yaml`:

```yaml
path: ./data/dataset      # dataset root
train: images/train        # training images
val:   images/val          # validation images

names:
  0: person
  1: car
  2: dog
```

Labels are one `.txt` per image, each line: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1). Use tools such as [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg) for annotation.

## Available Models

YOLO11 comes in five scales. Swap `--model` to any of them:

| Model | Parameters | mAP50-95 (COCO) | Use case |
|-------|-----------:|----------------:|----------|
| yolo11n | 2.6M  | 39.5 | Edge / mobile |
| yolo11s | 9.4M  | 47.0 | Balanced |
| yolo11m | 20.1M | 51.5 | General purpose |
| yolo11l | 25.3M | 53.4 | High accuracy |
| yolo11x | 56.9M | 54.7 | Maximum accuracy |

## Google Colab

Prefer no local setup? Open [`colab.ipynb`](colab.ipynb) in Google Colab and run the cells directly on a free GPU.

## References

- Ultralytics YOLO11 docs: https://docs.ultralytics.com/models/yolo11/
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics

## License

This project is released under the [MIT License](LICENSE). Note that pretrained Ultralytics weights are AGPL-3.0 licensed — review their terms before commercial use.

## Acknowledgements

Built on top of the excellent [Ultralytics](https://ultralytics.com) framework.
