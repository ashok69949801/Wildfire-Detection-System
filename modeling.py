"""CNN architecture and augmentation layers for wildfire detection."""

from __future__ import annotations

from config import INPUT_SHAPE, LEARNING_RATE

import tensorflow as tf
import keras
from keras import layers, models, optimizers


@keras.utils.register_keras_serializable(package="Wildfire")
class RandomBrightness(layers.Layer):
    """Training-only brightness augmentation compatible with saved Keras models."""

    def __init__(self, max_delta: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = max_delta

    def call(self, inputs, training=None):
        if training is None:
            training = False

        def augment():
            return tf.clip_by_value(
                tf.image.random_brightness(inputs, max_delta=self.max_delta),
                0.0,
                1.0,
            )

        return tf.cond(tf.cast(training, tf.bool), augment, lambda: inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"max_delta": self.max_delta})
        return config


def build_cnn_model(input_shape: tuple[int, int, int] = INPUT_SHAPE) -> keras.Model:
    """Build and compile the requested CNN classifier."""
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
            layers.RandomRotation(20 / 360, name="aug_rotation"),
            layers.RandomZoom(0.2, name="aug_zoom"),
            RandomBrightness(0.15, name="aug_brightness"),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=input_shape, name="satellite_image")
    x = data_augmentation(inputs)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="fire_conv_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="fire_conv_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="fire_conv_3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_3")(x)

    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fire_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="wildfire_cnn")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def load_trained_model(model_path):
    """Load the saved model with custom augmentation layer support."""
    return models.load_model(
        model_path,
        custom_objects={"RandomBrightness": RandomBrightness},
    )
