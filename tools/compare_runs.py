"""
对比所有实验结果，打印排序表格。

Usage
-----
# 所有 runs 按 Event F0.5 降序
python tools/compare_runs.py

# 只看 esa 相关的实验
python tools/compare_runs.py --filter esa

# 按 Affiliation F0.5 排序
python tools/compare_runs.py --sort affil_f

# 显示完整超参
python tools/compare_runs.py --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COLS_BRIEF = [
    ("run",        "Run",          36),
    ("dataset",    "Data",          5),
    ("d_model",    "d",             4),
    ("n_layers",   "L",             2),
    ("top_k",      "K",             2),
    ("point_f1",   "F1",            6),
    ("point_f05",  "F0.5",          6),
    ("point_auc",  "AUC",           6),
    ("event_f",    "Ev-F0.5",       8),
    ("affil_f",    "Af-F0.5",       8),
    ("threshold",  "Thr",           7),
]

COLS_VERBOSE = COLS_BRIEF + [
    ("epochs",        "Ep",  3),
    ("window_size",   "W",   4),
    ("batch_size",    "B",   4),
    ("learning_rate", "lr",  8),
]


def load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    config_path  = run_dir / "config.yaml"

    if not metrics_path.exists():
        return None  # training might still be running

    try:
        import yaml
        with open(metrics_path) as f:
            m = json.load(f)
        config: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
    except Exception:
        return None

    res = m.get("results", {})
    pw  = res.get("point",       {})
    ev  = res.get("event",       {})
    af  = res.get("affiliation", {})

    row = {
        "run":          run_dir.name,
        "dataset":      config.get("dataset", "?"),
        "d_model":      config.get("d_model",  "?"),
        "n_layers":     config.get("n_layers", "?"),
        "top_k":        config.get("top_k",    "?"),
        "epochs":       config.get("epochs",   "?"),
        "window_size":  config.get("window_size", "?"),
        "batch_size":   config.get("batch_size",  "?"),
        "learning_rate":config.get("learning_rate", "?"),
        "point_f1":     pw.get("f1",        float("nan")),
        "point_f05":    pw.get("f05",       float("nan")),
        "point_auc":    pw.get("auc",       float("nan")),
        "event_f":      ev.get("f_score",   float("nan")),
        "affil_f":      af.get("f_score",   float("nan")),
        "threshold":    m.get("threshold",  float("nan")),
    }
    return row


def fmt(val, width: int) -> str:
    if isinstance(val, float):
        if val != val:  # nan
            s = "  —  "
        else:
            s = f"{val:.4f}"
    else:
        s = str(val)
    return s[:width].ljust(width)


def print_table(rows: list[dict], cols: list, sort_key: str) -> None:
    # Sort
    def _key(r):
        v = r.get(sort_key, float("nan"))
        return -v if isinstance(v, float) else 0  # descending

    rows = sorted(rows, key=_key)

    # Header
    header = "  ".join(label.ljust(w) for _, label, w in cols)
    sep    = "  ".join("-" * w        for _, _, w    in cols)
    print(header)
    print(sep)
    for row in rows:
        line = "  ".join(fmt(row.get(key, "?"), w) for key, _, w in cols)
        print(line)

    print(f"\n{len(rows)} run(s) found.")
    if rows:
        best = rows[0]
        print(f"\n★  Best by {sort_key}: {best['run']}")
        print(f"   Event F0.5={best['event_f']:.4f}   Affil F0.5={best['affil_f']:.4f}"
              f"   F1={best['point_f1']:.4f}   AUC={best['point_auc']:.4f}")
        print(f"\n   PSTG baseline → Event F0.5=0.917   Affil F0.5=0.892")


def main():
    p = argparse.ArgumentParser(description="Compare STMAD experiment runs")
    p.add_argument("--runs_dir", default="runs",  help="Root directory of all runs")
    p.add_argument("--filter",   default=None,    help="Show only runs whose name contains this string")
    p.add_argument("--sort",     default="event_f",
                   choices=["event_f", "affil_f", "point_f1", "point_f05", "point_auc"],
                   help="Metric to sort by (descending)")
    p.add_argument("--verbose",  action="store_true", help="Show extra hyperparameter columns")
    args = p.parse_args()

    runs_root = Path(args.runs_dir)
    if not runs_root.exists():
        print(f"No runs directory found at '{runs_root}'. Run a training first.")
        return

    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    if args.filter:
        run_dirs = [d for d in run_dirs if args.filter in d.name]

    rows = []
    for d in run_dirs:
        row = load_run(d)
        if row:
            rows.append(row)
        else:
            # Still training or crashed
            print(f"  [skip] {d.name}  (no metrics.json yet)")

    if not rows:
        print("No completed runs found.")
        return

    cols = COLS_VERBOSE if args.verbose else COLS_BRIEF
    print_table(rows, cols, sort_key=args.sort)


if __name__ == "__main__":
    main()
