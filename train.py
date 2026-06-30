"""Train the wildfire CNN from dataset/fire and dataset/no_fire images."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATASET_DIR,
    EPOCHS,
    IMAGE_SIZE,
    MODEL_PATH,
    OUTPUT_FOLDER,
    SEED,
)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from evaluate import evaluate_model
from modeling import build_cnn_model
from utils import collect_image_files, ensure_directories, setup_logging, summarize_skipped, validate_image


logger = setup_logging("wildfire.train")


def collect_labeled_dataset(dataset_dir: Path) -> tuple[list[str], list[int], list[Path]]:
    """Load image paths from class folders and skip corrupted files."""
    class_dirs = {
        1: dataset_dir / "fire",
        0: dataset_dir / "no_fire",
    }

    image_paths: list[str] = []
    labels: list[int] = []
    skipped: list[Path] = []

    for label, directory in class_dirs.items():
        for image_path in collect_image_files(directory):
            if validate_image(image_path):
                image_paths.append(str(image_path))
                labels.append(label)
            else:
                skipped.append(image_path)

    return image_paths, labels, skipped


def _stratify_if_possible(labels: list[int]):
    counts = {label: labels.count(label) for label in set(labels)}
    return labels if len(counts) == 2 and min(counts.values()) >= 2 else None


def split_dataset(
    image_paths: list[str],
    labels: list[int],
    seed: int = SEED,
) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    """Split data into 80% train, 10% validation, and 10% test."""
    if len(image_paths) < 10:
        raise ValueError("Add at least 10 valid images so 80/10/10 splits are meaningful.")
    if len(set(labels)) < 2:
        raise ValueError("Both dataset/fire and dataset/no_fire must contain valid images.")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_if_possible(labels),
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_if_possible(temp_labels),
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def make_dataset(
    image_paths: list[str],
    labels: list[int],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Create a normalized tf.data pipeline from image paths."""

    def load_and_preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.float32)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=SEED, reshuffle_each_iteration=True)
    return (
        ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def train(args: argparse.Namespace) -> None:
    ensure_directories()
    tf.keras.utils.set_random_seed(SEED)

    logger.info("Loading images from %s", args.dataset)
    image_paths, labels, skipped = collect_labeled_dataset(args.dataset)
    logger.info("Valid images: %s | Fire: %s | No Fire: %s", len(image_paths), labels.count(1), labels.count(0))
    logger.info(summarize_skipped(skipped))

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(image_paths, labels)
    logger.info(
        "Split sizes | train=%s validation=%s test=%s",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )

    train_ds = make_dataset(train_paths, train_labels, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, args.batch_size)
    test_ds = make_dataset(test_paths, test_labels, args.batch_size)

    model = build_cnn_model()
    model.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Persist the restored best weights at the expected inference path.
    model.save(args.model_path)
    logger.info("Saved model to %s", args.model_path)

    artifacts = evaluate_model(model, test_ds, history.history, OUTPUT_FOLDER)
    for name, path in artifacts.items():
        logger.info("Saved %s artifact to %s", name, path)

    test_metrics = model.evaluate(test_ds, verbose=0)
    logger.info("Test metrics: %s", dict(zip(model.metrics_names, np.round(test_metrics, 4))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN wildfire detector.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR, help="Dataset root with fire/no_fire folders.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Output model path.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
