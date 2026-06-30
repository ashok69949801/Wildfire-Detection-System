"""Flask web application for wildfire detection."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for

from config import MAX_CONTENT_LENGTH, OUTPUT_FOLDER, UPLOAD_FOLDER
from predict import WildfirePredictor
from utils import (
    allowed_file,
    append_prediction_history,
    ensure_directories,
    load_prediction_history,
    make_unique_filename,
    setup_logging,
)


ensure_directories()
logger = setup_logging("wildfire.flask")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_predictor: WildfirePredictor | None = None


def get_predictor() -> WildfirePredictor:
    global _predictor
    if _predictor is None:
        _predictor = WildfirePredictor()
        if _predictor.demo_mode:
            logger.warning("Model file not found. Using demo fallback predictor.")
        else:
            logger.info("Loaded wildfire model for inference.")
    return _predictor


def static_url_for(path: str | Path | None) -> str | None:
    if not path:
        return None
    static_relative = Path(path).relative_to(OUTPUT_FOLDER.parent).as_posix()
    return url_for("static", filename=static_relative)


@app.route("/")
def index():
    return render_template(
        "index.html",
        history=load_prediction_history(),
        demo_mode=get_predictor().demo_mode,
    )


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_files = request.files.getlist("images")
    if not uploaded_files:
        single_file = request.files.get("image")
        uploaded_files = [single_file] if single_file is not None else []

    uploaded_files = [item for item in uploaded_files if item is not None and item.filename]
    if not uploaded_files:
        return render_template(
            "index.html",
            error="Upload one or more satellite images before running detection.",
            history=load_prediction_history(),
            demo_mode=get_predictor().demo_mode,
        ), 400

    invalid_files = [item.filename for item in uploaded_files if not allowed_file(item.filename)]
    if invalid_files:
        return render_template(
            "index.html",
            error=(
                "Unsupported file type for: "
                + ", ".join(invalid_files)
                + ". Use PNG, JPG, JPEG, BMP, TIF, or TIFF."
            ),
            history=load_prediction_history(),
            demo_mode=get_predictor().demo_mode,
        ), 400

    results = []
    uploaded_images = []
    heatmap_images = []

    for uploaded_file in uploaded_files:
        filename = make_unique_filename(uploaded_file.filename)
        upload_path = UPLOAD_FOLDER / filename
        uploaded_file.save(upload_path)
        logger.info("Saved upload to %s", upload_path)

        uploaded_url = url_for("static", filename=f"uploads/{filename}")
        try:
            result = get_predictor().predict(upload_path, make_heatmap=True)
        except Exception as exc:
            logger.exception("Prediction failed.")
            return render_template(
                "index.html",
                error=str(exc),
                uploaded_image=uploaded_url,
                history=load_prediction_history(),
                demo_mode=get_predictor().demo_mode,
            ), 500

        heatmap_url = static_url_for(result.get("heatmap_path"))
        row = {
            "filename": uploaded_file.filename,
            "label": result["label"],
            "confidence": result["confidence"],
            "fire_probability": result["fire_probability"],
            "uploaded_image": uploaded_url,
            "heatmap_image": heatmap_url,
            "prediction_mode": result.get("prediction_mode"),
        }
        results.append(row)
        uploaded_images.append(uploaded_url)
        if heatmap_url:
            heatmap_images.append(heatmap_url)

        append_prediction_history(
            {
                "label": result["label"],
                "confidence": result["confidence"],
                "fire_probability": result["fire_probability"],
                "uploaded_image": uploaded_url,
                "heatmap_image": heatmap_url,
                "prediction_mode": result.get("prediction_mode"),
            }
        )

    primary_result = results[0]

    return render_template(
        "index.html",
        result=primary_result,
        results=results,
        uploaded_image=primary_result["uploaded_image"],
        heatmap_image=primary_result["heatmap_image"],
        uploaded_images=uploaded_images,
        heatmap_images=heatmap_images,
        history=load_prediction_history(),
        demo_mode=get_predictor().demo_mode,
    )


@app.route("/health")
def health():
    model_ready = False
    try:
        get_predictor()
        model_ready = True
    except Exception:
        model_ready = False
    return jsonify({"status": "ok", "model_ready": model_ready})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
