"""CSV parsing in compare_ios_benchmark.py (no model runs)."""
import pytest

cib = pytest.importorskip("compare_ios_benchmark")

HEADER = "filename,state,detection_count,top_label,top_confidence,all_labels\n"


def test_parse_all_labels():
    assert cib.parse_all_labels("dent:87|scratch:41") == [("dent", 0.87), ("scratch", 0.41)]
    assert cib.parse_all_labels("glass shatter:99") == [("glass shatter", 0.99)]
    assert cib.parse_all_labels("") == []
    assert cib.parse_all_labels(None) == []


def test_load_ios_csv(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text(HEADER
                 + "000012.jpg,damage,2,tire flat,0.99993354,tire flat:99|dent:40\n"
                 + "000015.jpg,no_damage,0,,,\n"
                 + "000016.jpg,load_failed,0,,,")  # app writes no trailing newline
    rows = cib.load_ios_csv(p)
    assert set(rows) == {"000012.jpg", "000015.jpg", "000016.jpg"}
    a = rows["000012.jpg"]
    assert a["state"] == "damage" and a["count"] == 2
    assert a["top_label"] == "tire flat"
    assert a["top_conf"] == pytest.approx(0.99993354)
    assert a["labels"] == [("tire flat", 0.99), ("dent", 0.40)]
    b = rows["000015.jpg"]
    assert b["state"] == "no_damage" and b["count"] == 0 and b["top_conf"] == 0.0
    assert b["labels"] == []
    assert rows["000016.jpg"]["state"] not in cib.VALID_IOS_STATES


def test_app_thresholds_match_coreml_export():
    assert cib.APP_CONF == 0.25
    assert cib.APP_IOU == 0.7
