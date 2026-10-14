"""build_dataset.py split / 25kUndmged / safety-guard logic and make_manifest.py,
on tiny synthetic images in tmp dirs (never touches data/)."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_counter = [0]


def make_img(path: Path):
    """Write a 4x4 JPEG; a counter after the end marker makes every md5 distinct
    (lossy JPEG can map different colors to identical bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _counter[0] += 1
    n = _counter[0]
    Image.new("RGB", (4, 4), (n % 256, 0, 7)).save(path, format="JPEG")
    with open(path, "ab") as f:
        f.write(f"#{n}".encode())


@pytest.fixture
def raw(tmp_path):
    r = tmp_path / "raw"
    for i in range(30):
        make_img(r / "cardd_damage" / f"d{i:03d}.jpg")
    for i in range(12):
        make_img(r / "clean_cars" / f"c{i:03d}.jpg")
    for i in range(5):
        make_img(r / "clean_cars_extra" / f"e{i:03d}.jpg")
    for i in range(8):
        make_img(r / "external_datasets" / "cars_dataset_2" / f"x{i:03d}.jpg")
    for i in range(5):
        make_img(r / "harder_negatives" / f"h{i:03d}.jpeg")
    u = tmp_path / "25k"
    for split, n in [("train", 9), ("val", 3), ("test", 4)]:
        for i in range(n):
            sub = "car" if i % 2 else "vehicle car car_model x"
            make_img(u / split / sub / f"{split.upper()}{i:02d}.jpg")
    return tmp_path


def build(base: Path, out="processed", ev="eval", *extra):
    cmd = [sys.executable, str(ROOT / "src" / "build_dataset.py"),
           "--raw_dir", str(base / "raw"), "--undmged_dir", str(base / "25k"),
           "--out_dir", str(base / out), "--eval_dir", str(base / ev), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)


def files(d: Path):
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


def test_split_sizes_and_25k_keeps_own_split(raw):
    r = build(raw)
    assert r.returncode == 0, r.stderr
    out = raw / "processed"
    # balanced portion: min(30 damage, 30 no_damage) = 30 per class; 10% val, 10% test
    for split, n in [("train", 24), ("val", 3), ("test", 3)]:
        assert len(list((out / split / "damage").iterdir())) == n
        nd = sorted(p.name for p in (out / split / "no_damage").iterdir())
        balanced = [x for x in nd if x.startswith("no_damage_")]
        extra = [x for x in nd if not x.startswith("no_damage_")]
        assert len(balanced) == n
        # 25k images keep their own split and original filenames (flattened)
        assert extra == sorted(p.name for p in (raw / "25k" / split).rglob("*.jpg"))


def test_rebuild_is_deterministic(raw):
    assert build(raw, "p1", "e1").returncode == 0
    assert build(raw, "p2", "e2").returncode == 0
    assert files(raw / "p1") == files(raw / "p2")
    for f in files(raw / "p1"):
        assert (raw / "p1" / f).read_bytes() == (raw / "p2" / f).read_bytes()


def test_no_undmged_flag(raw):
    assert build(raw, "p", "e", "--no_undmged").returncode == 0
    nd = [p.name for p in (raw / "p").rglob("no_damage/*")]
    assert nd and all(n.startswith("no_damage_") for n in nd)
    assert not (raw / "e" / "undmged_25k").exists()


def test_refuses_existing_out_dir_without_force(raw):
    out = raw / "processed"
    (out / "keep").mkdir(parents=True)
    (out / "keep" / "precious.txt").write_text("x")
    r = build(raw)
    assert r.returncode == 1
    assert "REFUSING" in r.stdout
    assert (out / "keep" / "precious.txt").exists()
    assert not (raw / "eval").exists()  # checked both before touching anything


def test_refuses_existing_eval_dir_without_force(raw):
    (raw / "eval").mkdir()
    r = build(raw)
    assert r.returncode == 1
    assert not (raw / "processed").exists()


def test_force_replaces(raw):
    (raw / "processed" / "junk").mkdir(parents=True)
    r = build(raw, "processed", "eval", "--force")
    assert r.returncode == 0, r.stderr
    assert "DELETING" in r.stdout
    assert not (raw / "processed" / "junk").exists()


def test_eval_by_source_is_test_split_only(raw):
    assert build(raw).returncode == 0
    ev, out = raw / "eval", raw / "processed"
    u = sorted(p.name for p in (ev / "undmged_25k" / "no_damage").iterdir())
    assert u == sorted(p.name for p in (raw / "25k" / "test").rglob("*.jpg"))
    # every per-source eval image is byte-identical to a no_damage TEST image
    test_bytes = {p.read_bytes() for p in (out / "test" / "no_damage").iterdir()}
    train_bytes = {p.read_bytes() for p in (out / "train" / "no_damage").iterdir()}
    for p in ev.rglob("*.jp*g"):
        assert p.read_bytes() in test_bytes
        assert p.read_bytes() not in train_bytes


def test_duplicate_names_inside_25k_split_is_an_error(raw):
    make_img(raw / "25k" / "train" / "other" / "TRAIN00.jpg")
    r = build(raw)
    assert r.returncode != 0
    assert "duplicate filenames" in (r.stdout + r.stderr)


def test_manifest_matches_build(raw, tmp_path):
    assert build(raw).returncode == 0
    m = tmp_path / "m.csv"
    r = subprocess.run([sys.executable, str(ROOT / "src" / "make_manifest.py"),
                        "--data_dir", str(raw / "processed"), "--out", str(m),
                        "--no_sources"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(m)))
    assert len(rows) == len(files(raw / "processed"))
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert {r["class"] for r in rows} == {"damage", "no_damage"}
    assert all(len(r["md5"]) == 32 for r in rows)


def test_manifest_source_lookup(tmp_path, monkeypatch):
    import make_manifest

    src = tmp_path / "srcA"
    make_img(src / "a.jpg")
    make_img(src / "b.jpg")
    data = tmp_path / "data" / "test" / "no_damage"
    data.mkdir(parents=True)
    (data / "copy_of_a.jpg").write_bytes((src / "a.jpg").read_bytes())
    make_img(data / "orphan.jpg")
    monkeypatch.setattr(make_manifest, "RAW_SOURCES", {"srcA": src})
    out = tmp_path / "m.csv"
    monkeypatch.setattr(sys, "argv", ["x", "--data_dir", str(tmp_path / "data"), "--out", str(out)])
    make_manifest.main()
    got = {r["filename"]: r["source"] for r in csv.DictReader(open(out))}
    assert got == {"copy_of_a.jpg": "srcA", "orphan.jpg": "UNKNOWN"}
