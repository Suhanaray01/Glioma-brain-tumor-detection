import io
import json

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


client = TestClient(app)


def _image_upload():
    image_bytes = io.BytesIO()
    Image.new("RGB", (12, 12)).save(image_bytes, format="PNG")
    return ("files", ("slice.png", image_bytes.getvalue(), "image/png"))


def test_health_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models_loaded"] is False


def test_frontend_root_serves_application():
    response = client.get("/")

    assert response.status_code == 200
    assert "GliomaScan-HC" in response.text


def test_predict_requires_trained_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", tmp_path / "models")
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", tmp_path / "reports")
    response = client.post("/api/predict", files=[_image_upload()])
    assert response.status_code == 503
    assert "non-smoke training run" in response.json()["detail"]


def test_predict_rejects_unverified_sequence_claim():
    response = client.post(
        "/api/predict",
        files=[_image_upload()],
        data={"sequence": "T2FLAIR"},
    )

    assert response.status_code == 422
    assert "does not verify MRI sequence" in response.json()["detail"]


def test_predict_does_not_return_fixture_when_weights_exist(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", tmp_path)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", tmp_path / "reports")
    (tmp_path / "checkpoint.keras").touch()

    response = client.post("/api/predict", files=[_image_upload()])

    assert response.status_code == 503
    assert "non-smoke training run" in response.json()["detail"]


def test_predict_returns_probabilities_from_recorded_model(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()
    model_path = models_dir / "checkpoint.keras"
    model_path.touch()
    (reports_dir / "training_summary.json").write_text(json.dumps({
        "status": "ok",
        "quick": False,
        "models": [{"name": "inception_v3", "status": "trained", "model_path": str(model_path)}],
    }), encoding="utf-8")
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", models_dir)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", reports_dir)

    class TestModel:
        name = "inception_v3"
        input_shape = (None, 8, 8, 3)
        output_shape = (None, 2)

        def predict(self, batch, verbose=0):
            return np.tile([0.25, 0.75], (len(batch), 1))

    monkeypatch.setattr("glioma_hc.inference._load_keras_model", lambda path: TestModel())
    image_bytes = io.BytesIO()
    Image.new("RGB", (12, 12)).save(image_bytes, format="PNG")

    response = client.post(
        "/api/predict",
        files=[("files", ("slice.png", image_bytes.getvalue(), "image/png"))],
    )

    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "glioma"
    assert result["glioma_probability"] == 0.75
    assert result["models"][0]["probs"] == [0.25, 0.75]
    assert result["slices"][0]["sequence"] == "UNSPECIFIED"
    assert "MRI sequence was not specified by the dataset." in result["warnings"]
    assert result["explain"]["status"] == "unavailable"


def test_predict_hybrid_cam_returns_fused_overlay_and_overlap(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir()
    reports_dir.mkdir()
    model_path = models_dir / "checkpoint.keras"
    model_path.touch()
    (reports_dir / "training_summary.json").write_text(json.dumps({
        "status": "ok",
        "quick": False,
        "models": [{"name": "inception_v3", "status": "trained", "model_path": str(model_path)}],
    }), encoding="utf-8")
    monkeypatch.setattr("backend.main.SETTINGS.models_dir", models_dir)
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", reports_dir)

    class TestModel:
        name = "inception_v3"
        input_shape = (None, 8, 8, 3)
        output_shape = (None, 2)

        def predict(self, batch, verbose=0):
            return np.tile([0.25, 0.75], (len(batch), 1))

    monkeypatch.setattr("glioma_hc.inference._load_keras_model", lambda path: TestModel())
    monkeypatch.setattr("glioma_hc.inference.make_gradcam_overlay", lambda *args, **kwargs: "gradcam")
    monkeypatch.setattr(
        "glioma_hc.inference.make_hybrid_cam",
        lambda *args, **kwargs: {"overlay": "hybrid-cam", "attribution_overlap": 0.4},
    )

    response = client.post(
        "/api/predict",
        files=[_image_upload()],
        data={"explain": "hybrid_cam"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["glioma_probability"] == 0.75
    assert result["explain"]["status"] == "generated"
    assert result["explain"]["hybrid_cam_overlay"] == "hybrid-cam"
    assert result["explain"]["attribution_overlap"] == 0.4


def test_metrics_endpoint_reads_saved_test_results(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.main.SETTINGS.reports_dir", tmp_path)
    (tmp_path / "test_results.json").write_text(json.dumps({
        "dataset": "test-dataset",
        "metrics": {"accuracy": 0.75},
    }), encoding="utf-8")

    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.json()["results"] == {"accuracy": 0.75}
    assert response.json()["evaluation"]["dataset"] == "test-dataset"
