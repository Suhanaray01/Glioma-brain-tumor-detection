from glioma_hc.config import SETTINGS
from glioma_hc.dataset_builder.build_dataset import paper_split_indices, build_synthetic_paper_split
from glioma_hc.dataset_builder.import_kaggle_mri import import_dataset


def test_paper_split_matches_expected_label_order():
    labels = [0, 0, 0, 1, 1, 1, 1]
    train_idx, test_idx = paper_split_indices(labels, seed=42)
    assert len(train_idx) + len(test_idx) == len(labels)
    assert SETTINGS.label_order == ["normal", "glioma"]


def test_build_synthetic_paper_split_counts():
    summary = build_synthetic_paper_split()
    assert summary["train"] == 40
    assert summary["test"] == 10
    assert summary["counts"]["normal"] == 20
    assert summary["counts"]["glioma"] == 20


def test_kaggle_import_preserves_test_split_and_excludes_other_tumors(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "prepared"
    for split_name, count in (("Training", 5), ("Testing", 2)):
        for class_name in ("glioma", "notumor", "meningioma", "pituitary"):
            class_dir = source_dir / split_name / class_name
            class_dir.mkdir(parents=True)
            for index in range(count):
                (class_dir / f"image_{index}.jpg").write_bytes(b"image")

    manifest = import_dataset(source_dir, output_dir, seed=17)

    assert manifest["counts"] == {
        "train": {"normal": 4, "glioma": 4},
        "val": {"normal": 1, "glioma": 1},
        "test": {"normal": 2, "glioma": 2},
    }
    assert manifest["negative_class_note"].startswith("The source notumor class")
    assert "does not specify MRI sequences" in manifest["modality_note"]
    assert not (output_dir / "test" / "meningioma").exists()
