# Traffic Sign Recognition

A complete traffic sign recognition system with a CNN classifier for 43 classes, object localization, and a web interface for image upload and live webcam analysis.

## Features

- **CNN Classification** - Custom `TrafficSignNet` for 43 traffic sign classes
- **Object Localization** - Automatic bounding box detection using HSV color segmentation and edge fallback
- **Image Upload** - Upload or drag and drop an image for instant classification
- **Live Webcam** - Real-time traffic sign detection with continuous frame analysis
- **Class ID Display** - Each prediction shows the label, confidence score, and class ID
- **Modern UI** - Glassmorphic web interface with upload and webcam modes

## Project Structure

```text
traffic-sign-recognition/
|-- Dataset/
|   |-- Train/
|   |-- Test/
|   |-- labels.csv
|-- traffic_signs/
|   |-- __init__.py
|   |-- data.py
|   |-- model.py
|   |-- inference.py
|   |-- localization.py
|-- templates/
|-- static/
|-- models/
|   |-- traffic_recognition_best_model.pt
|-- analyze_dataset.py
|-- train.py
|-- predict.py
|-- app.py
```

## Dataset

- `labels.csv` contains 43 label names
- `Test.csv` matches the extracted `Test/` images
- `Train.csv` does not match the extracted training image filenames exactly
- The training pipeline uses the real class folders inside `Dataset/Train/`
- Obvious `- Copie` duplicates are ignored automatically

## Quick Start

### 1. Analyze the dataset

```bash
python analyze_dataset.py
```

### 2. Train the model

```bash
python train.py --epochs 30 --batch-size 256 --eval-batch-size 256 --patience 6
```

Training artifacts are written to `artifacts/`.

- `best_model.pt` - Best training checkpoint
- `metrics.json` - Final evaluation metrics
- `history.json` - Per-epoch training history
- `classification_report.json` - Per-class precision/recall/F1
- `dataset_report.json` - Dataset audit report

## Final Training Run

The final validated training run was executed in Google Colab on GPU with:

- requested epochs: `30`
- executed epochs before early stopping: `24`
- best epoch: `18`
- batch size: `256`
- evaluation batch size: `256`
- early stopping patience: `6`

Main command used:

```bash
python train.py \
  --dataset-dir /content/traffic-sign-recognition \
  --output-dir /content/traffic-sign-recognition/artifacts \
  --epochs 30 \
  --batch-size 256 \
  --eval-batch-size 256 \
  --patience 6 \
  --num-workers 2
```

## Final Metrics

- validation accuracy: `100.00%`
- validation macro F1: `100.00%`
- test accuracy: `98.80%`
- test macro F1: `98.28%`
- best epoch: `18`
- total training time: `1446.58 s` (`24 min 07 s`)

### 3. Predict one image

```bash
python predict.py --image Dataset/Test/00000.png
```

By default, prediction uses:

```text
models/traffic_recognition_best_model.pt
```

### 4. Launch the app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

- The app and CLI use `models/traffic_recognition_best_model.pt` by default
- The model trains on all 43 class folders found in `Dataset/Train/`
- The final Colab run stopped automatically with early stopping after epoch `24`
- On Windows, `--num-workers 0` is the safest default if needed
