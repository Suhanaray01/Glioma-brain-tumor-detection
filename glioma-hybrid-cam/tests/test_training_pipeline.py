from pathlib import Path

import pytest

from glioma_hc.training import train_all


def test_quick_training_run_creates_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "tf", None)
    result = train_all.run_training(
        ["inception_v3"],
        preset="fast",
        quick=True,
        project_root=tmp_path,
    )

    assert result["status"] == "skipped"
    assert (tmp_path / "reports" / "training_summary.json").exists()
    assert "models" in result
    assert result["training_data_dir"] == str(tmp_path / "data" / "smoke" / "processed")
    assert not (tmp_path / "data" / "processed").exists()


def test_image_dataset_uses_configured_label_order(tmp_path):
    pytest.importorskip("tensorflow")
    from PIL import Image
    import numpy as np

    from glioma_hc.config import SETTINGS
    from glioma_hc.training.data_loader import build_image_dataset

    for label, pixel_value in (("normal", 0), ("glioma", 255)):
        class_dir = tmp_path / label
        class_dir.mkdir()
        Image.new("RGB", (8, 8), color=(pixel_value,) * 3).save(class_dir / "sample.png")

    dataset = build_image_dataset(
        tmp_path,
        image_size=(8, 8),
        batch_size=2,
        model_name="inception_v3",
    )

    images, labels = next(iter(dataset))
    pixel_means = images.numpy().mean(axis=(1, 2, 3))
    label_values = labels.numpy()
    assert label_values[np.argmin(pixel_means)] == SETTINGS.label_order.index("normal")
    assert label_values[np.argmax(pixel_means)] == SETTINGS.label_order.index("glioma")
    assert float(pixel_means.min()) == -1.0
    assert float(pixel_means.max()) == 1.0


def test_attention_cnn_input_is_scaled_to_unit_range():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from glioma_hc.training.data_loader import preprocess_model_input

    images = tf.constant([[[[0.0, 127.5, 255.0]]]])
    processed = preprocess_model_input(images, "attention_cnn")

    assert processed.numpy().tolist() == [[[[0.0, 0.5, 1.0]]]]


def test_multiscale_cbam_input_is_scaled_to_unit_range():
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from glioma_hc.training.data_loader import preprocess_model_input

    images = tf.constant([[[[0.0, 127.5, 255.0]]]])
    processed = preprocess_model_input(images, "multiscale_cbam_cnn")

    assert processed.numpy().tolist() == [[[[0.0, 0.5, 1.0]]]]


def test_real_training_skips_without_raw_source_data(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "tf", None)
    for label in ("normal", "glioma"):
        class_dir = tmp_path / "data" / "processed" / "train" / label
        class_dir.mkdir(parents=True)
        (class_dir / f"{label}_train_0.png").touch()

    result = train_all.run_training(
        ["inception_v3"],
        preset="paper",
        quick=False,
        project_root=tmp_path,
    )

    assert result["status"] == "skipped"
    assert "data/raw" in result["note"]
    assert result["models"] == []


def test_attention_model_summary_does_not_claim_frozen_backbone(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "tf", None)

    result = train_all.run_training(
        ["attention_cnn"],
        preset="attention",
        quick=True,
        project_root=tmp_path,
    )

    assert result["training_strategy"] == "attention_cnn_from_scratch"
