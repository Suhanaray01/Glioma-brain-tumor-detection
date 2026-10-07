from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from glioma_hc.config import SETTINGS
from glioma_hc.hybrid_camera.quality import assess_capture_quality
from glioma_hc.inference import InferenceUnavailable, InvalidImage, predict_images

app = FastAPI(title="GliomaScan-HC")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    model_dir = SETTINGS.models_dir
    model_available = model_dir.exists() and any(model_dir.glob("*.keras"))
    note = (
        "A model checkpoint is available; predictions require a completed compatible training run."
        if model_available
        else "Weights are required for real predictions."
    )
    return {"status": "ok", "models_loaded": model_available, "note": note}


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    model_dir = SETTINGS.models_dir
    model_count = len(list(model_dir.glob("*.keras"))) if model_dir.exists() else 0
    summary_path = SETTINGS.reports_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {
        "status": "ok",
        "models_loaded": model_count,
        "missing_weights": model_count == 0,
        "training_summary": summary.get("status", "not_run"),
    }


@app.get("/api/models")
def api_models() -> dict[str, Any]:
    summary_path = SETTINGS.reports_dir / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"models": []}
    return {
        "models": summary.get("models", []),
        "saved_validation_metrics": [],
        "ensemble_members": [],
    }


@app.post("/api/camera/quality")
async def camera_quality(file: UploadFile = File(default=None)) -> dict[str, Any]:
    if file is None:
        return {"sharpness": 0.0, "brightness": 0.0, "glare_pct": 0.0, "roi_found": False, "ok": False, "tips": ["No file supplied."]}
    content = await file.read()
    if not content:
        return {"sharpness": 0.0, "brightness": 0.0, "glare_pct": 0.0, "roi_found": False, "ok": False, "tips": ["Empty upload."]}
    import numpy as np
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(content)).convert("RGB")
    arr = np.array(image)
    return assess_capture_quality(arr)


@app.post("/api/predict")
async def predict(
    files: list[UploadFile] = File(default_factory=list),
    sequence: str = Form("UNSPECIFIED"),
    source: str = Form("upload"),
    explain: str = Form("gradcam"),
    aggregation: str = Form("max"),
) -> dict[str, Any]:
    if sequence.upper() != "UNSPECIFIED":
        raise HTTPException(
            status_code=422,
            detail="The trained dataset does not verify MRI sequence; submit with sequence UNSPECIFIED.",
        )
    image_files = []
    for file in files:
        content = await file.read()
        image_files.append((file.filename or "uploaded image", content))
    try:
        return await run_in_threadpool(predict_images, image_files, aggregation, explain, sequence)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/metrics")
def api_metrics() -> dict[str, Any]:
    results_path = SETTINGS.reports_dir / "test_results.json"
    paper_path = SETTINGS.reports_dir / "paper_reference.json"
    evaluation = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    paper_reference = json.loads(paper_path.read_text(encoding="utf-8")) if paper_path.exists() else {}
    return {
        "results": evaluation.get("metrics", {}),
        "evaluation": evaluation,
        "paper_reference": paper_reference,
        "figures": [],
    }


@app.post("/api/report")
def api_report() -> dict[str, Any]:
    return {"status": "not_implemented", "message": "PDF report generation can be added when report data exists."}


app.mount("/", StaticFiles(directory=SETTINGS.project_root / "frontend", html=True), name="frontend")
