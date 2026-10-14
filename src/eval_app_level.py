"""
Image-level detector false positives and app-level decision policies, from the
CSVs written by score_images.py (no model runs here).

"Damage" for the app = at least one detector box with conf >= threshold.

Report sections:
  3b  detector-only image-level FP rate on held-out clean sets at 0.25/0.5/0.7,
      plus FP box/image counts per class
  3c  policies on test (positives = CarDD test, negatives = held-out no_damage):
        P1 detector >= 0.25
        P2 classifier says damage (sigmoid < 0.5) AND detector >= 0.25
        P3 detector >= t, t picked on VAL (CarDD val vs 25kUndmged val):
           a) max F1   b) max recall with FP rate <= 2%
      Thresholds are picked on val only and then reported on test.

Usage (from project root):
    python src/eval_app_level.py --det runs/overnight2/eval/det_v7.csv \
        --clf runs/overnight2/eval/clf_scores.csv --name v7
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from paths import CLASS_NAMES

NEG_TEST = ["u25k_test", "ho_clean_cars", "ho_clean_cars_extra", "ho_cars_dataset_2"]


def load_max(path: Path):
    """{(set, path): max_conf}"""
    with open(path) as f:
        return {(r["set"], r["path"]): float(r["max_conf"]) for r in csv.DictReader(f)}


def load_boxes(path: Path):
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["set"], r["path"])].append((int(r["cls"]), float(r["conf"])))
    return out


def load_clf(path: Path):
    """{path: p_damage}; same image may appear in several sets, score is identical."""
    with open(path) as f:
        return {r["path"]: float(r["p_damage"]) for r in csv.DictReader(f)}


def rates(pos_flags, neg_flags):
    tp, fn = sum(pos_flags), len(pos_flags) - sum(pos_flags)
    fp, tn = sum(neg_flags), len(neg_flags) - sum(neg_flags)
    rec = tp / len(pos_flags) if pos_flags else float("nan")
    fpr = fp / len(neg_flags) if neg_flags else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, rec=rec, fpr=fpr, prec=prec,
                f1=2 * prec * rec / (prec + rec) if tp else 0.0)


def fmt(name, r, npos, nneg):
    return (f"  {name:44s} recall {r['tp']}/{npos} = {r['rec']:.4f}   "
            f"FP {r['fp']}/{nneg} = {r['fpr']:.4%}   precision {r['tp']}/{r['tp'] + r['fp']} = {r['prec']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", type=Path, required=True)
    ap.add_argument("--clf", type=Path, required=True)
    ap.add_argument("--name", default="v7")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="Text file of image paths to drop from negatives "
                         "(e.g. detector training backgrounds)")
    ap.add_argument("--clf_train_paths", type=Path, default=None,
                    help="Text file of CarDD test paths the classifier trained on; "
                         "P2 is also reported without them")
    args = ap.parse_args()

    mx = load_max(args.det)
    boxes = load_boxes(args.det.with_suffix(".boxes.csv"))
    clf = load_clf(args.clf)
    excl = set(Path(args.exclude).read_text().splitlines()) if args.exclude else set()
    by_set = defaultdict(list)
    for (s, p), v in mx.items():
        if p in excl and not s.startswith("cardd"):
            continue
        by_set[s].append((p, v))

    print(f"=== Detector {args.name}: {args.det} ===")
    if excl:
        print(f"(excluded from negatives: {sum(1 for s, p in mx if p in excl and not s.startswith('cardd'))} "
              f"set entries listed in {args.exclude})")

    # ── 3b ──
    print("\n3b. Image-level FP rate (any box >= t) on clean sets")
    for s in ["u25k_test", "u25k_val", "ho_clean_cars", "ho_clean_cars_extra",
              "ho_cars_dataset_2", "ho_harder_negatives", "harder_neg_all"]:
        if s not in by_set:
            continue
        v = [c for _, c in by_set[s]]
        cells = "  ".join(f"@{t}: {sum(c >= t for c in v)}/{len(v)} = {sum(c >= t for c in v) / len(v):.4%}"
                          for t in (0.25, 0.5, 0.7))
        print(f"  {s:22s} {cells}")
    print("\n  FP per class on u25k_test (boxes / images with >=1 box of that class)")
    for t in (0.25, 0.5, 0.7):
        nb, ni = Counter(), Counter()
        for p, _ in by_set["u25k_test"]:
            ks = [k for k, c in boxes.get(("u25k_test", p), []) if c >= t]
            nb.update(ks)
            ni.update(set(ks))
        print(f"   @{t}: " + ", ".join(f"{CLASS_NAMES[k]} {nb[k]}/{ni[k]}" for k in CLASS_NAMES))

    # ── 3c ──
    def flags(entries, t, gate=False):
        return [(c >= t) and (not gate or clf[p] > 0.5) for p, c in entries]

    pos = by_set["cardd_test"]
    neg = [e for s in NEG_TEST for e in by_set[s]]
    hn = by_set.get("harder_neg_all", [])
    vpos, vneg = by_set["cardd_val"], by_set["u25k_val"]
    print(f"\n3c. App-level policies. test positives = CarDD test (n={len(pos)}); "
          f"test negatives = {' + '.join(f'{s} {len(by_set[s])}' for s in NEG_TEST)} = {len(neg)}")
    print(f"    val (threshold picking only): positives CarDD val n={len(vpos)}, negatives u25k_val n={len(vneg)}")

    # pick thresholds on val
    cands = sorted({round(c, 3) for _, c in vpos + vneg if c >= 0.05} | {0.25})
    best_f1, best_fp2 = None, None
    for t in cands:
        r = rates(flags(vpos, t), flags(vneg, t))
        if best_f1 is None or r["f1"] > best_f1[1]["f1"]:
            best_f1 = (t, r)
        if r["fpr"] <= 0.02 and (best_fp2 is None or r["rec"] > best_fp2[1]["rec"]
                                 or (r["rec"] == best_fp2[1]["rec"] and t < best_fp2[0])):
            best_fp2 = (t, r)
    print(f"    val max-F1 threshold t={best_f1[0]}: " + fmt("", best_f1[1], len(vpos), len(vneg)).strip())
    print(f"    val FP<=2% threshold t={best_fp2[0]}: " + fmt("", best_fp2[1], len(vpos), len(vneg)).strip())
    r = rates(flags(vpos, 0.25), flags(vneg, 0.25))
    print(f"    (val at 0.25: {fmt('', r, len(vpos), len(vneg)).strip()})")

    print("\n  TEST:")
    pol = [("P1 detector >= 0.25", 0.25, False),
           ("P2 classifier gate AND detector >= 0.25", 0.25, True),
           (f"P3a detector >= {best_f1[0]} (val max-F1)", best_f1[0], False),
           (f"P3b detector >= {best_fp2[0]} (val FP<=2%)", best_fp2[0], False)]
    for name, t, g in pol:
        print(fmt(name, rates(flags(pos, t, g), flags(neg, t, g)), len(pos), len(neg)))
    print("\n  Per negative source (FP count / n):")
    for name, t, g in pol:
        cells = ", ".join(f"{s} {sum(flags(by_set[s], t, g))}/{len(by_set[s])}" for s in NEG_TEST)
        hcell = f"harder_neg_all {sum(flags(hn, t, g))}/{len(hn)}" if hn else ""
        print(f"   {name:44s} {cells}; {hcell} (harder_negatives: v7 TRAINED on all 165)")
    print("\n  Classifier alone (sigmoid < 0.5) for reference:")
    r = rates([clf[p] > 0.5 for p, _ in pos], [clf[p] > 0.5 for p, _ in neg])
    print(fmt("classifier only", r, len(pos), len(neg)))

    if args.clf_train_paths:
        seen = set(Path(args.clf_train_paths).read_text().splitlines())
        pos2 = [e for e in pos if e[0] not in seen]
        print(f"\n  Positives restricted to CarDD test images NOT in classifier train (n={len(pos2)}):")
        for name, t, g in pol[:2]:
            print(fmt(name, rates(flags(pos2, t, g), flags(neg, t, g)), len(pos2), len(neg)))
        r = rates([clf[p] > 0.5 for p, _ in pos2], [clf[p] > 0.5 for p, _ in neg])
        print(fmt("classifier only", r, len(pos2), len(neg)))


if __name__ == "__main__":
    main()
