# Glioma-brain-tumor-detection
# GliomaScan-HC

GliomaScan-HC is a research-oriented glioma classification pipeline built around the paper:
Pilaoon, Hamamoto, Maneerat, "Glioma Brain Tumor Classification Using Convolution Neural Network and Majority Voting", IEEE Access, 2025.

This repository implements the paper's essential workflow with a practical software extension: a hybrid camera ingest path for MRI screenshots, photos, DICOM, and NIfTI slices. The hybrid camera is an extension beyond the paper and is explicitly not a clinical device.

## Assumptions
- Only axial T2-FLAIR slices are used for classification.
- Training data must be placed under `data/raw/` by the user; no dataset is auto-downloaded.
- The project is built for Python 3.10/3.11 and TensorFlow 2.15.x.
- When real weights are missing, the API returns HTTP 503 and explains the training command.
- Batch sizes and training hyperparameters follow the paper reproduction defaults unless a fast preset is chosen.

## Project purpose
- Reproduce the paper's CNN ensemble approach on glioma vs normal classification using axial FLAIR images.
- Add a hybrid ingest workflow for camera, upload, DICOM, and NIfTI inputs.
- Provide a local FastAPI backend and a lightweight HTML/JS frontend.
- Keep the code open, typed, and testable.

## Directory layout

```text
glioma-hybrid-cam/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── glioma_hc/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing/
│   │   └── pipeline.py
│   ├── hybrid_camera/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── quality.py
│   │   ├── ingest.py
│   │   ├── study.py
│   │   └── camera_sim.py
│   ├── dataset_builder/
│   │   ├── __init__.py
│   │   ├── build_dataset.py
│   │   ├── extract_slices.py
│   │   └── make_smoke_data.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model_zoo.py
│   │   ├── train_all.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── benchmark.py
│   │   ├── robustness.py
│   │   ├── paper_comparison.py
│   │   └── stats_tests.py
│   ├── explainability/
│   │   ├── gradcam.py
│   │   ├── lime_explainer.py
│   │   └── hybrid_cam.py
│   └── serving/
│       ├── __init__.py
│       ├── inference.py
│       └── schemas.py
├── backend/
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── assets/
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── scripts/
│   └── run_all_quick.py
├── notebooks/
│   └── train_on_colab.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_camera_quality.py
│   ├── test_split_and_augmentation.py
│   ├── test_api.py
│   └── test_model_zoo.py
└── pytest.ini
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick smoke check

```bash
python -m pytest -q
```

## Typical workflow

```bash
python -m glioma_hc.dataset_builder.make_smoke_data
python -m glioma_hc.training.train_all --models inception_v3 --preset fast --quick
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Medical disclaimer

This project is for research and educational use only. It does not replace professional medical interpretation and is not a clinically validated diagnostic device.

## Citation

Pilaoon, S., Hamamoto, K., Maneerat, S. (2025). Glioma Brain Tumor Classification Using Convolution Neural Network and Majority Voting. IEEE Access.

## License

MIT. The project is inspired by and built with attribution to the original reference repository. The paper and associated research code remain the basis for this adapted pipeline.
