# Traffic Sign Recognition

This project trains a traffic sign classifier from the extracted `Dataset/` folder in this repository.

## What is in the dataset

- `labels.csv` contains `43` label names.
- `Test.csv` matches the extracted `Test/` images correctly.
- `Train.csv` does **not** match the extracted training images, because it points to `.png` files while the extracted training folders contain `.jpg` files with different names.
- The training pipeline in this repo uses the real class folders inside `Dataset/Train/` as the source of truth.

## Files

- `analyze_dataset.py`: audits the dataset and writes `artifacts/dataset_report.json`.
- `train.py`: trains the PyTorch traffic sign model and evaluates it.
- `predict.py`: runs inference on one image using the saved checkpoint.
- `traffic_signs/`: shared dataset and model code.

## Run the dataset analysis

```bash
python analyze_dataset.py
```

## Train the model

```bash
python train.py --epochs 8 --batch-size 128
```

Artifacts are saved to `artifacts/`:

- `best_model.pt`
- `metrics.json`
- `history.json`
- `classification_report.json`
- `dataset_report.json`

## Predict a single image

```bash
python predict.py --image Dataset/Test/00000.png
```

## Launch the upload interface

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser, upload an image and submit the form to see the predicted label and confidence scores.

## Notes

- The model trains on all `43` existing class folders found in `Dataset/Train/`.
- The loader automatically ignores obvious `- Copie` duplicate files.
- On Windows, `--num-workers 0` is the safest default if you hit DataLoader issues.
