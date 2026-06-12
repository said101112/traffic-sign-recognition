# Traffic Sign Recognition

A complete traffic sign recognition system with a CNN classifier (43 classes), object localization, and a rich web interface featuring both image upload and live webcam analysis.

## Features

- **CNN Classification** – Custom `TrafficSignNet` trained for 10 epochs on 43 traffic sign classes
- **Object Localization** – Automatic bounding box detection using HSV color segmentation + edge fallback
- **Image Upload** – Upload or drag & drop an image (including from the web) for instant classification
- **Live Webcam** – Real-time traffic sign detection via webcam with continuous frame analysis
- **Class ID Display** – Each prediction shows the label, confidence score, and class ID
- **Modern UI** – Premium glassmorphic design with tab-based mode switching

## Project Structure

```
traffic-recongnision/
|-- Dataset/
|   |-- Train/          # Training images (43 class folders)
|   |-- Test/           # Test images
|   |-- labels.csv      # 43 class label names
|-- traffic_signs/
|   |-- __init__.py
|   |-- data.py         # Dataset loading and preprocessing
|   |-- model.py        # TrafficSignNet CNN architecture
|   |-- inference.py    # Model loading and prediction
|   |-- localization.py # Bounding box detection (HSV + Canny)
|-- templates/
|   |-- landing.html    # Landing page
|   |-- app_page.html   # Main application page
|   |-- components/     # Reusable HTML components
|-- static/
|   |-- style.css       # Premium CSS styling
|   |-- app.js          # Webcam capture, drag-drop, live analysis
|-- analyze_dataset.py  # Dataset audit script
|-- train.py            # Training pipeline
|-- predict.py          # CLI single image prediction
|-- app.py              # Flask server (upload + webcam API)
```

## Dataset

- `labels.csv` contains `43` label names.
- `Test.csv` matches the extracted `Test/` images correctly.
- `Train.csv` does **not** match the extracted training images (references `.png` while actual files are `.jpg`).
- The training pipeline uses the real class folders inside `Dataset/Train/` as the source of truth.
- Obvious `- Copie` duplicate files are automatically ignored.

## Quick Start

### 1. Analyze the dataset

```bash
python analyze_dataset.py
```

### 2. Train the model

```bash
python train.py --epochs 10 --batch-size 192 --eval-batch-size 256
```

Artifacts are saved to `artifacts/`:
- `best_model.pt` – Best model checkpoint
- `metrics.json` – Final evaluation metrics
- `history.json` – Per-epoch training history
- `classification_report.json` – Per-class precision/recall/F1
- `dataset_report.json` – Dataset audit report

### 3. Predict a single image

```bash
python predict.py --image Dataset/Test/00000.png --top-k 3
```

### 4. Launch the web interface

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Web Interface

The interface provides two modes accessible via tabs in the header:

- **Image Upload** – Select a file, drag & drop from your computer, or drag an image directly from any website. The model returns the predicted class with a bounding box drawn on the image.
- **Webcam Live** – Start your webcam for continuous real-time analysis. Each frame is sent to the server, classified, and returned with an annotated bounding box overlay.

Both modes display the top-3 predictions with confidence scores and class IDs.

## Notes

- The model trains on all `43` existing class folders found in `Dataset/Train/`.
- On Windows, `--num-workers 0` is the safest default if you hit DataLoader issues.
- Webcam mode requests 1080p resolution for best detection accuracy.
