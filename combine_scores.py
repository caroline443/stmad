"""
分数融合：MTA + PSTG-MA 加权集成

原理：
  s_combined = α × norm(s_MTA) + (1-α) × norm(s_MA)

  MTA 分数（raw_smoothed.npy）：重建误差，Event 精度高
  PSTG-MA 分数（combined_scores.npy）：双信号预测残差，Affil 精度高
  加权融合后有望两者兼得

用法：
  python combine_scores.py \\
    --mta_dir  outputs_mta/eval_20260628_151233 \\
    --ma_dir   outputs_ma/eval_20260627_111502 \\
    --alpha    0.5   # MTA 权重（0=纯MA，1=纯MTA）
"""

import argparse
import json
from pathlib import Path

import numpy as np

from anomaly.detector import smooth_residuals, threshold_signal
from utils.metrics import event_wise_metrics, affiliation_metrics, extract_events


# ─────────────────────────────────────────────────────────────────────────────

def normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def evaluate(y_true, anomaly_scores, label=""):
    y_pred = (anomaly_scores > 0).astype(np.int32)

    # 标准2：duration≥2
    y_filt = np.zeros_like(y_true)
    for s, e in extract_events(y_true):
        if e - s + 1 >= 2:
            y_filt[s:e+1] = 1

    ew = event_wise_metrics(y_filt, y_pred)
    af = affiliation_metrics(y_filt, y_pred)
    n  = len(extract_events(y_filt))

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  事件数（duration≥2）：{n}")
    print(f"  Event-wise  P={ew['precision']:.4f}  R={ew['recall']:.4f}  "
          f"F0.5={ew['f0.5']:.4f}")
    print(f"  Affiliation P={af['precision']:.4f}  R={af['recall']:.4f}  "
          f"F0.5={af['f0.5']:.4f}")
    return ew, af


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mta_dir",   required=True,
                   help="MTA eval 目录（需含 raw_smoothed.npy, y_true.npy）")
    p.add_argument("--ma_dir",    required=True,
                   help="PSTG-MA eval 目录（需含 combined_scores.npy）")
    p.add_argument("--alpha",     type=float, default=0.5,
                   help="MTA 权重（default 0.5；1=纯MTA，0=纯MA）")
    p.add_argument("--alphas",    type=float, nargs="+", default=None,
                   help="扫描多个 alpha（覆盖 --alpha）")
    p.add_argument("--pot_alpha", type=float, default=4e-3)
    p.add_argument("--pot_q0",    type=float, default=0.98)
    p.add_argument("--min_peak_z",type=float, default=1.5)
    p.add_argument("--smooth_window", type=int, default=105)
    return p.parse_args()


def main():
    args   = parse_args()
    mta_d  = Path(args.mta_dir)
    ma_d   = Path(args.ma_dir)

    # ── 加载 MTA 分数（已平滑）────────────────────────────────────────────────
    mta_smooth = np.load(mta_d / "raw_smoothed.npy").astype(np.float64)
    y_true     = np.load(mta_d / "y_true.npy").astype(np.int32)
    print(f"MTA  分数长度：{len(mta_smooth):,}  范围：[{mta_smooth.min():.4f}, {mta_smooth.max():.4f}]")

    # ── 加载 PSTG-MA 分数 ─────────────────────────────────────────────────────
    # combined_scores.npy 是原始双信号融合残差（未平滑），需要先平滑
    ma_raw = np.load(ma_d / "combined_scores.npy").astype(np.float64)
    print(f"MA   分数长度：{len(ma_raw):,}  范围：[{ma_raw.min():.4f}, {ma_raw.max():.4f}]")

    # 平滑 MA（与 MTA 一致）
    ma_smooth = smooth_residuals(ma_raw, args.smooth_window).astype(np.float64)

    # ── 长度对齐（取较短的那段）─────────────────────────────────────────────────
    T = min(len(mta_smooth), len(ma_smooth), len(y_true))
    mta_s = mta_smooth[:T]
    ma_s  = ma_smooth[:T]
    y     = y_true[:T]
    print(f"对齐后长度：{T:,}")

    # ── 归一化到 [0,1] ─────────────────────────────────────────────────────────
    mta_n = normalize(mta_s)
    ma_n  = normalize(ma_s)

    # ── 单独基线评估 ───────────────────────────────────────────────────────────
    print("\n=== 单模型基线 ===")
    for name, sig in [("MTA (raw_smoothed)", mta_s),
                      ("PSTG-MA (combined_scores, smoothed)", ma_s)]:
        scores = threshold_signal(sig, method="pot",
                                  pot_q0=args.pot_q0,
                                  pot_alpha=args.pot_alpha,
                                  min_peak_z=args.min_peak_z)
        evaluate(y, scores, label=name)

    # ── Alpha 扫描 ─────────────────────────────────────────────────────────────
    alphas = args.alphas if args.alphas else [args.alpha]

    print("\n=== 融合结果（α=MTA权重，1-α=MA权重）===")
    best_ev = best_af = best_sum = 0
    best_alpha_ev = best_alpha_af = best_alpha_sum = None

    for alpha in alphas:
        combined = alpha * mta_n + (1 - alpha) * ma_n
        scores   = threshold_signal(combined.astype(np.float32),
                                    method="pot",
                                    pot_q0=args.pot_q0,
                                    pot_alpha=args.pot_alpha,
                                    min_peak_z=args.min_peak_z)
        ew, af = evaluate(y, scores,
                          label=f"Fusion α={alpha:.2f} (MTA={alpha:.0%}, MA={1-alpha:.0%})")

        if ew['f0.5'] > best_ev:
            best_ev, best_alpha_ev = ew['f0.5'], alpha
        if af['f0.5'] > best_af:
            best_af, best_alpha_af = af['f0.5'], alpha
        if ew['f0.5'] + af['f0.5'] > best_sum:
            best_sum, best_alpha_sum = ew['f0.5'] + af['f0.5'], alpha

    if len(alphas) > 1:
        print(f"\n{'='*55}")
        print(f"  最优 Event F0.5：{best_ev:.4f}  (α={best_alpha_ev})")
        print(f"  最优 Affil F0.5：{best_af:.4f}  (α={best_alpha_af})")
        print(f"  最优 Ev+Af 之和：{best_sum:.4f}  (α={best_alpha_sum})")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
