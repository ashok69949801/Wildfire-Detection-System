"""Inference and Grad-CAM visualization for wildfire predictions."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from config import CLASS_NAMES, MODEL_PATH, OUTPUT_FOLDER, PREDICTION_THRESHOLD

import cv2
import numpy as np

from utils import preprocess_image


def _import_tensorflow():
    import tensorflow as tf  # pylint: disable=import-outside-toplevel

    return tf


def _load_trained_model(model_path: Path):
    from modeling import load_trained_model  # pylint: disable=import-outside-toplevel

    return load_trained_model(model_path)


class WildfirePredictor:
    """Lazy model wrapper used by Flask and command-line inference."""

    def __init__(self, model_path: str | Path = MODEL_PATH, allow_demo_fallback: bool = True):
        self.model_path = Path(model_path)
        self.tf = None
        self.model = None
        self.demo_mode = False

        if self.model_path.exists():
            self.tf = _import_tensorflow()
            self.model = _load_trained_model(self.model_path)
        elif allow_demo_fallback:
            self.demo_mode = True
        else:
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run `python train.py` after adding images to dataset/fire "
                "and dataset/no_fire."
            )

    def predict(self, image_path: str | Path, make_heatmap: bool = True) -> dict:
        if self.demo_mode:
            fire_probability = self._heuristic_fire_probability(image_path)
        else:
            if self.model is None:
                raise RuntimeError("Model is not loaded.")
            batch = preprocess_image(image_path)
            fire_probability = float(self.model.predict(batch, verbose=0)[0][0])

        predicted_index = int(fire_probability >= PREDICTION_THRESHOLD)
        label = CLASS_NAMES[predicted_index]
        confidence = fire_probability if predicted_index == 1 else 1.0 - fire_probability

        heatmap_path = None
        if make_heatmap:
            heatmap_path = (
                self.generate_demo_heatmap(image_path)
                if self.demo_mode
                else self.generate_gradcam(image_path, predicted_index)
            )

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "fire_probability": round(fire_probability * 100, 2),
            "heatmap_path": str(heatmap_path) if heatmap_path else None,
            "prediction_mode": "demo" if self.demo_mode else "trained_model",
        }

    def _last_conv_layer_name(self) -> str:
        if self.model is None or self.tf is None:
            raise ValueError("Grad-CAM requires a loaded trained model.")
        tf = self.tf
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if isinstance(layer, tf.keras.Model):
                for nested in reversed(layer.layers):
                    if isinstance(nested, tf.keras.layers.Conv2D):
                        return nested.name
        raise ValueError("No Conv2D layer found for Grad-CAM.")

    def generate_gradcam(self, image_path: str | Path, predicted_index: int, alpha: float = 0.42) -> Path:
        """Create and save a Grad-CAM overlay for the predicted class."""
        if self.model is None or self.tf is None:
            raise ValueError("Grad-CAM requires a loaded trained model.")
        tf = self.tf
        batch = preprocess_image(image_path)
        last_conv_layer_name = self._last_conv_layer_name()
        last_conv_layer = self.model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[last_conv_layer.output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(batch)
            class_score = predictions[:, 0] if predicted_index == 1 else 1.0 - predictions[:, 0]

        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError("Could not read uploaded image for Grad-CAM rendering.")

        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_FOLDER / f"gradcam_{uuid4().hex[:12]}.jpg"
        cv2.imwrite(str(output_path), overlay)
        return output_path

    def _heuristic_fire_probability(self, image_path: str | Path) -> float:
        """Fallback score used when no trained model is available."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Could not read uploaded image for inference.")

        image_float = image.astype(np.float32) / 255.0
        b_channel, g_channel, r_channel = cv2.split(image_float)
        warm_map = np.clip(r_channel - (0.5 * g_channel) - (0.25 * b_channel), 0.0, 1.0)
        warm_mean = float(warm_map.mean())

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_channel, saturation_channel, value_channel = cv2.split(hsv)

        fire_color_ratio = float(
            (
                ((hue_channel <= 25) | (hue_channel >= 170))
                & (saturation_channel >= 90)
                & (value_channel >= 110)
            ).mean()
        )
        ember_ratio = float(
            (
                (r_channel > 0.60)
                & (r_channel > g_channel + 0.10)
                & (r_channel > b_channel + 0.14)
            ).mean()
        )
        smoke_ratio = float(((saturation_channel <= 70) & (value_channel >= 80)).mean())
        bright_ratio = float((value_channel >= 170).mean())

        score = (
            0.50 * warm_mean
            + 1.90 * fire_color_ratio
            + 1.40 * ember_ratio
            + 0.25 * smoke_ratio
            + 0.20 * bright_ratio
        )
        return float(np.clip(0.04 + (1.10 * score), 0.02, 0.98))

    def generate_demo_heatmap(self, image_path: str | Path, alpha: float = 0.42) -> Path:
        """Create a deterministic heatmap when running without a trained model."""
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError("Could not read uploaded image for heatmap rendering.")

        image_float = original.astype(np.float32) / 255.0
        b_channel, g_channel, r_channel = cv2.split(image_float)
        warm_map = np.clip(r_channel - (0.5 * g_channel) - (0.25 * b_channel), 0.0, 1.0)
        warm_map = cv2.GaussianBlur(warm_map, ksize=(0, 0), sigmaX=11, sigmaY=11)

        normalized = cv2.normalize(warm_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(np.uint8(normalized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)

        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_FOLDER / f"heatmap_demo_{uuid4().hex[:12]}.jpg"
        cv2.imwrite(str(output_path), overlay)
        return output_path


def predict_image(image_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict:
    return WildfirePredictor(model_path).predict(image_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict wildfire presence in a satellite image.")
    parser.add_argument("image", type=Path, help="Path to image file.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to trained .h5 model.")
    args = parser.parse_args()

    result = predict_image(args.image, args.model_path)
    print(result)
