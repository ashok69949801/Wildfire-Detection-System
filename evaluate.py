"""Evaluation helpers for training curves, reports, and confusion matrices."""

from __future__ import annotations

from pathlib import Path
import os

from config import BASE_DIR, CLASS_NAMES, OUTPUT_FOLDER

MPL_CACHE_DIR = BASE_DIR / ".matplotlib"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_curves(history: dict, output_dir: Path = OUTPUT_FOLDER) -> dict[str, Path]:
    """Save accuracy and loss graphs used by the dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = output_dir / "training_accuracy.png"
    loss_path = output_dir / "training_loss.png"

    epochs = range(1, len(history.get("accuracy", [])) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history.get("accuracy", []), label="Train accuracy", linewidth=2)
    plt.plot(epochs, history.get("val_accuracy", []), label="Validation accuracy", linewidth=2)
    plt.title("Wildfire CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history.get("loss", []), label="Train loss", linewidth=2)
    plt.plot(epochs, history.get("val_loss", []), label="Validation loss", linewidth=2)
    plt.title("Wildfire CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary crossentropy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()

    return {"accuracy": accuracy_path, "loss": loss_path}


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path = OUTPUT_FOLDER) -> Path:
    """Save a confusion matrix image with stable labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    path = output_dir / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Reds")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path = OUTPUT_FOLDER) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classification_report.txt"
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        digits=4,
        zero_division=0,
    )
    report_path.write_text(report, encoding="utf-8")
    return report_path


def evaluate_model(model, test_ds, history: dict | None = None, output_dir: Path = OUTPUT_FOLDER) -> dict[str, Path]:
    """Evaluate model on test data and persist visual artifacts."""
    artifacts: dict[str, Path] = {}
    if history:
        artifacts.update(plot_training_curves(history, output_dir))

    y_true_batches = []
    for _, labels in test_ds:
        y_true_batches.append(labels.numpy().astype(int).reshape(-1))
    y_true = np.concatenate(y_true_batches)

    y_prob = model.predict(test_ds).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    artifacts["confusion_matrix"] = save_confusion_matrix(y_true, y_pred, output_dir)
    artifacts["classification_report"] = save_classification_report(y_true, y_pred, output_dir)
    return artifacts
