# Wildfire Detection System

Wildfire image classification project built with Flask and TensorFlow for detecting fire in satellite or aerial imagery. The application supports web-based inference, batch uploads, heatmap visualization, and a simple training pipeline for a custom CNN model.

## Overview

This project is a binary image classification system that predicts whether an input image belongs to:

- `Fire`
- `No Fire`

The app is designed as a practical student or portfolio project rather than a full production wildfire monitoring platform. It combines:

- a custom CNN training pipeline
- a Flask dashboard for image upload and prediction
- heatmap generation for visual interpretation
- a fallback demo mode when a trained model is not available

## Key Features

- Single-image and multi-image wildfire prediction
- Heatmap output for visual explanation
- Prediction confidence and fire probability
- Prediction history in the dashboard
- Drag-and-drop upload interface
- Demo fallback mode if `model/wildfire_model.h5` is missing
- Local training pipeline for a binary wildfire classifier

## Model Summary

The training code uses a custom convolutional neural network defined in `modeling.py`.

Architecture highlights:

- input size: `224 x 224 x 3`
- data augmentation:
  - random horizontal flip
  - random rotation
  - random zoom
  - custom random brightness layer
- convolution blocks:
  - `Conv2D(32) + MaxPooling`
  - `Conv2D(64) + MaxPooling`
  - `Conv2D(128) + MaxPooling`
- classification head:
  - batch normalization
  - flatten
  - dropout
  - dense layer with ReLU
  - sigmoid output for binary prediction

Training is configured with:

- loss: binary cross-entropy
- optimizer: Adam
- metrics: accuracy, precision, recall

## Dataset

The repository contains two dataset-related folders, and they serve different purposes.

### 1. `dataset/` - training input folder

This is the folder the training script reads by default.

Expected structure:

```text
dataset/
|-- fire/
`-- no_fire/
```

Right now, this folder is mostly a starter layout:

- `dataset/fire/` contains `.gitkeep`
- `dataset/no_fire/` contains `.gitkeep`
- `dataset/README.md` explains the expected structure

That means the project is ready for training, but you still need to place your final binary training images into these folders before running `train.py`.
## How the Project Works

### Training flow

1. Read images from `dataset/fire` and `dataset/no_fire`
2. Validate files and skip unreadable images
3. Split data into:
   - 80% train
   - 10% validation
   - 10% test
4. Resize images to `224 x 224`
5. Normalize pixel values to `[0, 1]`
6. Train the CNN
7. Save the trained model to `model/wildfire_model.h5`

### Inference flow

1. Upload one or more images
2. Load the trained model if available
3. If no trained model exists, switch to demo mode
4. Predict `Fire` or `No Fire`
5. Return:
   - label
   - confidence
   - fire probability
   - heatmap image

### Heatmap behavior

- In trained mode, the project uses Grad-CAM
- In demo mode, the project generates a deterministic heatmap using warm-color cues

## Project Structure

```text
wild/
|-- app.py
|-- config.py
|-- evaluate.py
|-- modeling.py
|-- predict.py
|-- train.py
|-- utils.py
|-- templates/
|   `-- index.html
|-- static/
|   |-- styles.css
|   |-- uploads/
|   `-- outputs/
|-- dataset/
|   |-- fire/
|   |-- no_fire/
|   `-- README.md
|-- datasets/
|   `-- small/
|-- model/
|-- logs/
`-- README.md
```

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- NumPy
- OpenCV
- Pillow
- scikit-learn
- Matplotlib

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Web App

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

Health endpoint:

```text
http://localhost:5000/health
```

## Train the Model

Before training, add your final binary images to:

- `dataset/fire`
- `dataset/no_fire`

Then run:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --dataset dataset --epochs 20 --batch-size 32 --model-path model/wildfire_model.h5
```

## Run CLI Prediction

```bash
python predict.py path/to/image.jpg
```

## Dashboard Output

The dashboard currently focuses on prediction workflow rather than training charts. It shows:

- uploaded image preview
- predicted class
- confidence and fire probability
- heatmap output
- batch inference results
- prediction history


<img width="1381" height="1139" alt="image" src="https://github.com/user-attachments/assets/35f2908a-bbb2-42a1-90e3-baa8b5b80959" />


<img width="1372" height="1147" alt="image" src="https://github.com/user-attachments/assets/fd7f6e8a-be82-4dab-966f-5d16662febfc" />



## Demo Mode

If `model/wildfire_model.h5` is missing, the app still runs in demo mode.

In demo mode:

- uploads still work
- a heuristic score is used instead of CNN inference
- heatmaps are still generated

This is useful for UI demos, but it should not be treated as a real trained model result.

## Limitations

- The default training folder is empty until you add images manually
- The included sample collection is not wired directly into the training script
- The system performs binary classification only
- `start_fire` is not currently handled as a separate class
- Prediction quality depends heavily on dataset quality and class balance
- Demo mode is only a fallback and is not a substitute for training
