# Wildfire Detection System

Wildfire image classification project built with Flask and TensorFlow for detecting fire in satellite or aerial imagery. The application supports web-based inference, batch uploads, heatmap visualization, and a simple training pipeline for a custom CNN model.

## 1) Project Objective

Build an end-to-end binary classifier:

- Input: satellite image
- Output: `Fire` or `No Fire`
- Confidence + fire probability
- Visual explanation heatmap

The system supports two prediction modes:

- `trained_model`: uses `model/wildfire_model.h5`
- `demo`: if model is missing, uses a heuristic fallback

## 2) Algorithms Used

### 2.1 Core ML Algorithm (Supervised Learning)

- Problem type: Binary image classification
- Model: Custom CNN (`wildfire_cnn`)
- Loss: Binary Cross-Entropy
- Optimizer: Adam (`LEARNING_RATE=0.001` by default)
- Metrics: Accuracy, Precision, Recall
- Decision threshold: `PREDICTION_THRESHOLD=0.5`

### 2.2 CNN Architecture

From `modeling.py` (`build_cnn_model`):

1. Data augmentation:
   - Random horizontal flip
   - Random rotation
   - Random zoom
   - Custom `RandomBrightness` layer
2. Feature extraction:
   - `Conv2D(32) + MaxPool`
   - `Conv2D(64) + MaxPool`
   - `Conv2D(128) + MaxPool`
3. Classification head:
   - BatchNormalization
   - Flatten
   - Dropout(0.5)
   - Dense(128, ReLU)
   - Dropout(0.3)
   - Dense(1, Sigmoid)

### 2.3 Training Strategy

From `train.py`:

- Data split: `80% train / 10% validation / 10% test`
- Stratified split used when class counts allow
- TensorFlow `tf.data` pipeline with map, batch, prefetch
- Callbacks:
  - EarlyStopping (`patience=5`, restore best weights)
  - ModelCheckpoint (save best by validation loss)

### 2.4 Explainability Algorithm

From `predict.py`:

- Trained mode explanation: **Grad-CAM** on last convolution layer
- Demo mode explanation: deterministic warm-color heatmap based on color cues

### 2.5 Demo Fallback Heuristic (When No Model File)

A weighted fire-likelihood score is computed from:

- Warm-channel intensity (`R - 0.5G - 0.25B`)
- HSV-based fire-color ratio
- Ember ratio
- Smoke-like low-saturation ratio
- Brightness ratio

Then score is clipped to `[0.02, 0.98]` and used as pseudo-probability.

## 3) File-Wise Functionalities

### 3.1 Core Python Files

| File | Main Role | Key Functions / Classes |
|---|---|---|
| `config.py` | Central constants and paths | Dataset/model/static/log paths, hyperparameters, thresholds |
| `train.py` | End-to-end model training | `collect_labeled_dataset`, `split_dataset`, `make_dataset`, `train`, `parse_args` |
| `modeling.py` | CNN architecture and model loading | `RandomBrightness`, `build_cnn_model`, `load_trained_model` |
| `evaluate.py` | Training/evaluation artifacts | `plot_training_curves`, `save_confusion_matrix`, `save_classification_report`, `evaluate_model` |
| `predict.py` | Inference + heatmaps | `WildfirePredictor`, `generate_gradcam`, `generate_demo_heatmap`, `predict_image` |
| `app.py` | Flask web app + routes | `index`, `predict`, `health`, lazy `get_predictor` loader |
| `utils.py` | Shared utility layer | image validation, preprocessing, logging, history persistence, safe filenames |

### 3.2 Frontend / UI Files

| File | Role |
|---|---|
| `templates/index.html` | Upload form, result panels, batch results, artifacts, history, theme toggle |
| `static/styles.css` | Responsive dashboard styling, dark/light theme, cards/grid/layout |

### 3.3 Deployment / Environment Files

| File | Role |
|---|---|
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Containerized deployment (Gunicorn + Flask) |
| `.dockerignore` | Excludes transient files from Docker build context |
| `.gitignore` | Excludes model files, logs, generated outputs, virtual env |

### 3.4 Data / Artifacts / Support

| Path | Role |
|---|---|
| `dataset/fire`, `dataset/no_fire` | Training classes |
| `model/wildfire_model.h5` | Trained model output |
| `static/uploads` | Uploaded image storage |
| `static/outputs` | Heatmaps, graphs, report files, prediction history |
| `logs/wildfire_app.log` | Runtime logs |
| `notebooks/EDA.ipynb` | Exploratory notebook |

## 4) Code Working Process (End-to-End Flow)

### 4.1 Training Workflow

1. Read class folders (`dataset/fire`, `dataset/no_fire`)
2. Validate image integrity (corrupted files skipped)
3. Split into train/val/test
4. Build tf.data pipelines (resize to `224x224`, normalize to `[0,1]`)
5. Build CNN (`build_cnn_model`)
6. Train with early stopping + best-checkpoint saving
7. Save final model at `model/wildfire_model.h5`
8. Generate artifacts:
   - `training_accuracy.png`
   - `training_loss.png`
   - `confusion_matrix.png`
   - `classification_report.txt`

### 4.2 Inference Workflow (CLI/Web)

1. Load predictor (`WildfirePredictor`)
2. If model exists: run CNN inference
3. If model missing: switch to demo heuristic mode
4. Compute final label using threshold (`>=0.5 => Fire`)
5. Produce heatmap:
   - Trained mode: Grad-CAM
   - Demo mode: warm-color heatmap
6. Return label, confidence, probability, mode, heatmap path

### 4.3 Flask Web Workflow

1. `GET /` loads dashboard, history, artifact images
2. `POST /predict` accepts one/multiple images
3. Validate extensions and save files with unique safe names
4. Run predictor per file
5. Append each result to history JSON
6. Render result cards + heatmaps + training artifacts
7. `GET /health` returns API health and model readiness

## 5) How to Run (Complete)

### 5.1 Prerequisites

- Python 3.10+ recommended
- `pip`

### 5.2 Local Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.3 Prepare Dataset

Place data like:

```text
dataset/
  fire/
    fire_001.jpg
    ...
  no_fire/
    no_fire_001.jpg
    ...
```

Important:

- Keep both classes populated
- At least 10 valid images total are required by split logic

### 5.4 Train the Model

```bash
python train.py
```

Optional arguments:

```bash
python train.py --dataset dataset --epochs 20 --batch-size 32 --model-path model/wildfire_model.h5
```

### 5.5 Run the Web App

```bash
python app.py
```

Open in browser:

```text
http://localhost:5000
```

Health check:

```text
http://localhost:5000/health
```

### 5.6 Run Single-Image Prediction (CLI)

```bash
python predict.py path/to/image.jpg
```

With custom model path:

```bash
python predict.py path/to/image.jpg --model-path model/wildfire_model.h5
```

### 5.7 Run with Docker

Build:

```bash
docker build -t wildfire-app .
```

Run:

```bash
docker run -p 5000:5000 wildfire-app
```

Then open:

```text
http://localhost:5000
```

## 6) Runtime Outputs

After use/training, key outputs include:

- `model/wildfire_model.h5`
- `static/outputs/training_accuracy.png`
- `static/outputs/training_loss.png`
- `static/outputs/confusion_matrix.png`
- `static/outputs/classification_report.txt`
- `static/outputs/prediction_history.json`
- `static/outputs/gradcam_*.jpg` or `heatmap_demo_*.jpg`
- `logs/wildfire_app.log`

## 7) Error Handling and Troubleshooting

- Unsupported upload format:
  - Use `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`
- Model file missing:
  - App auto-switches to demo mode
  - Train model with `python train.py` for real CNN predictions
- Training errors from data:
  - Ensure both class folders have valid images
  - Remove corrupted files
- Empty/weak results:
  - Increase dataset size and class balance
  - Train more epochs or tune threshold/hyperparameters

```text
### Project Structure
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
- scikit-learn
- Matplotlib

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

## 8) Current Features

- Prediction result visualization
- Accuracy graph (static image)
- Heatmap output
- Multiple image upload
- Prediction history
- Health endpoint
- Demo fallback mode

## Limitations

- The default training folder is empty until you add images manually
- The included sample collection is not wired directly into the training script
- The system performs binary classification only
- `start_fire` is not currently handled as a separate class
- Prediction quality depends heavily on dataset quality and class balance
- Demo mode is only a fallback and is not a substitute for training
