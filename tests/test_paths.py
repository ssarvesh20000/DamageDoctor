"""paths.py constants."""
from pathlib import Path

import paths

ROOT = Path(__file__).resolve().parents[1]


def test_paths_are_relative_paths():
    for name in ["CLASSIFIER_CKPT", "CLASSIFIER_COREML", "DETECTOR_CKPT",
                 "DETECTOR_COREML", "DETECTOR_V1_CKPT", "DETECTOR_V1_COREML"]:
        p = getattr(paths, name)
        assert isinstance(p, Path), name
        assert not p.is_absolute(), name


def test_production_detector_is_v7():
    assert paths.DETECTOR_CKPT == Path("runs/kaggle_cardd_v7_detect/best.pt")
    assert paths.DETECTOR_V1_CKPT == Path("runs/detect/runs/detect/cardd_v1/weights/best.pt")
    assert paths.CLASSIFIER_CKPT == Path("models/binary_mobilenetv3_best.pt")


def test_class_names():
    assert paths.CLASS_NAMES == {0: "dent", 1: "scratch", 2: "crack",
                                 3: "glass shatter", 4: "lamp broken", 5: "tire flat"}


def test_class_names_match_kaggle_kernels():
    """Every detector training kernel writes the same names into its data.yaml."""
    import re
    for f in sorted((ROOT / "kaggle" / "kernels").glob("train_detect_v*/train.py")):
        src = f.read_text()
        for k, v in paths.CLASS_NAMES.items():
            assert re.search(rf'{k}:\s*"{v}"', src), (f, k, v)
