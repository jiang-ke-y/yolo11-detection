# Convenience targets for the YOLO11 detection project.
# Requires Python + `pip install -r requirements.txt` (or run `make install`).

PY ?= python

.PHONY: install train predict evaluate export clean

install:
	$(PY) -m pip install -r requirements.txt

train:
	$(PY) src/train.py --config configs/default.yaml

predict:
	$(PY) src/predict.py --weights yolo11n.pt --source assets/sample.jpg

evaluate:
	$(PY) src/evaluate.py --weights yolo11n.pt --data coco8.yaml

export:
	$(PY) src/export.py --weights runs/detect/train/weights/best.pt --format onnx

clean:
	rm -rf runs/detect/* .pytest_cache **/__pycache__
