"""Guards the single-source-of-truth invariant for the 9 classes.

classes.yaml, the labeler CANONICAL_CLASSES, and the flat classifier's
default class list must all agree on the same ordered 9 names. If this
test fails, a class list drifted and training/labeling/eval will silently
disagree on label indices.
"""
from pathlib import Path

from rfconnectorai.data.classes import class_names

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "classes.yaml"

EXPECTED = [
    "1.85mm-M", "1.85mm-F",
    "2.4mm-M", "2.4mm-F",
    "2.92mm-M", "2.92mm-F",
    "3.5mm-M", "3.5mm-F",
    "SMA-F",
]


def test_classes_yaml_matches_expected_order():
    assert class_names(CONFIG) == EXPECTED


def test_canonical_classes_matches_yaml():
    import ast

    page = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "pages" / "1_Training_Data.py"
    )
    tree = ast.parse(page.read_text(encoding="utf-8"), filename=str(page))
    canonical = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CANONICAL_CLASSES"
            for t in node.targets
        ):
            canonical = ast.literal_eval(node.value)
            break
    assert canonical is not None, "CANONICAL_CLASSES not found in page source"
    assert canonical == class_names(CONFIG)


def test_classifier_default_classes_match_yaml():
    from rfconnectorai.classifier.train import default_class_names

    assert default_class_names() == class_names(CONFIG)
