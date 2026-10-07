from glioma_hc.evaluation.evaluate import compute_metrics


def test_compute_metrics_uses_binary_positive_probability():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])

    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["auc"] == 1.0