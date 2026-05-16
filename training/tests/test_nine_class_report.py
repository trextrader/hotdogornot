from rfconnectorai.eval.nine_class_report import build_report


def test_report_per_class_and_pairs():
    class_names = ["2.4mm-M", "2.92mm-M", "SMA-F"]
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 2]  # one 2.4mm-M -> 2.92mm-M confusion

    rep = build_report(y_true, y_pred, class_names)

    assert rep["overall_accuracy"] == 5 / 6
    assert rep["confusion"][0][1] == 1
    assert rep["per_class"]["2.92mm-M"]["recall"] == 1.0
    assert rep["per_class"]["2.4mm-M"]["recall"] == 0.5
    assert "2.4mm-M -> 2.92mm-M" in rep["notable_confusions"]
