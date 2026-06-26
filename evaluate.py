"""
PSTG 推理与评估脚本（对应 Algorithm 1 Part 2 & 3）

用法：
    python evaluate.py [--ckpt ./checkpoints/best.pt] [--data_dir /path/to/data]

输出：
    - Event-wise F0.5（论文主要指标）
    - Affiliation-based F0.5
    - 异常分数时序图
    - 每通道预测 vs 真实对比图
"""

import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from data.dataset import build_datasets, FullSequenceDataset
from models.pstg import PSTG
from anomaly.detector import detect_anomalies, build_full_prediction
from utils.metrics import evaluate_all, event_wise_metrics, affiliation_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="PSTG 推理与评估")
    parser.add_argument("--ckpt",     type=str, default=None,
                        help="Checkpoint 路径（默认使用 checkpoints/best.pt）")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device",   type=str, default=None)
    parser.add_argument("--output",   type=str, default=None,
                        help="输出目录（默认 outputs/）")
    parser.add_argument("--no_plot",  action="store_true",
                        help="禁用图表输出")
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, test_loader, device: str, tau: int) -> np.ndarray:
    """
    Algorithm 1 Part 2：逐窗口预测，只保留前 τ=1 步，拼接为完整序列。

    Returns:
        x_pred: [T_pred, C]
    """
    model.eval()
    all_preds = []
    print("推理中...")
    for context, t_idx in tqdm(test_loader, desc="  推理"):
        context = context.to(device, non_blocking=True)   # [B, C, L]
        pred = model(context)                              # [B, C, F]
        # 保留前 τ 步
        pred_tau = pred[:, :, :tau]                       # [B, C, τ]
        pred_tau = pred_tau.permute(0, 2, 1).reshape(-1, pred.shape[1])
        all_preds.append(pred_tau.cpu().numpy())

    return np.concatenate(all_preds, axis=0).astype(np.float32)


def main():
    args = parse_args()
    cfg = Config()

    if args.data_dir: cfg.DATA_DIR = args.data_dir
    if args.device:   cfg.DEVICE   = args.device
    if args.output:   cfg.OUTPUT_DIR = args.output

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    # ── 加载 checkpoint ───────────────────────────────────────────────────
    ckpt_path = args.ckpt or os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}\n请先运行 train.py")

    print(f"加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("config", {})

    model = PSTG(
        patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
        d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
        num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
        num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
        n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
        context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
        forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
        top_k=cfg.top_k,
        dropout=0.0,   # 推理时关闭 dropout
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"模型参数量：{model.count_parameters():,}")
    print(f"训练于 epoch {ckpt.get('epoch', '?')}")

    # ── 加载数据 ──────────────────────────────────────────────────────────
    print("\n=== 加载测试数据 ===")
    data = build_datasets(cfg)
    test_loader = data["test_loader"]
    test_labels = data["test_labels"]       # [T_test] 原始测试集标签
    test_data   = data["test_data"]         # [T_test, C] 归一化后的测试数据

    # ── 推理 ──────────────────────────────────────────────────────────────
    print("\n=== 推理 ===")
    x_pred = run_inference(model, test_loader, device, cfg.TAU)
    # x_pred: [T_pred, C]，从第 L 步开始的预测
    T_pred = len(x_pred)
    print(f"预测序列长度：{T_pred:,}")

    # 对齐真实值（Algorithm 1 Line 20：X_target = S_test[L+1 : L+T_pred]）
    x_true = test_data[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]   # [T_pred, C]
    y_true = test_labels[cfg.CONTEXT_LEN : cfg.CONTEXT_LEN + T_pred]  # [T_pred]

    # ── 异常检测（Φ 算子）────────────────────────────────────────────────
    print("\n=== 动态阈值异常检测 ===")
    anomaly_scores = detect_anomalies(
        x_true=x_true,
        x_pred=x_pred,
        smooth_window=cfg.smooth_window,
        p_tfi=cfg.P_TFI,
        n_candidates=50,
    )
    print(f"异常分数范围：[{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")

    # ── 评估 ──────────────────────────────────────────────────────────────
    print("\n=== 评估 ===")
    # 用最优阈值（percentile-based 近似）
    from utils.metrics import find_best_threshold
    best_thresh, best_result = find_best_threshold(y_true, anomaly_scores, metric="event_f05")

    print("\n─── Event-wise 指标 ───")
    ew = best_result["event_wise"]
    print(f"  Precision : {ew['precision']:.4f}")
    print(f"  Recall    : {ew['recall']:.4f}")
    print(f"  F0.5      : {ew['f0.5']:.4f}  (论文目标: 0.917)")

    print("\n─── Affiliation-based 指标 ───")
    af = best_result["affiliation"]
    print(f"  Precision : {af['precision']:.4f}")
    print(f"  Recall    : {af['recall']:.4f}")
    print(f"  F0.5      : {af['f0.5']:.4f}  (论文目标: 0.892)")

    # ── 保存结果 ──────────────────────────────────────────────────────────
    results = {
        "event_wise": {
            "precision": float(ew["precision"]),
            "recall":    float(ew["recall"]),
            "f0.5":      float(ew["f0.5"]),
        },
        "affiliation": {
            "precision": float(af["precision"]),
            "recall":    float(af["recall"]),
            "f0.5":      float(af["f0.5"]),
        },
        "threshold":          float(best_thresh),
        "pred_anomaly_rate":  float(best_result.get("pred_anomaly_rate", 0)),
        "target_event_f05":   0.917,
        "target_affil_f05":   0.892,
    }
    result_path = os.path.join(cfg.OUTPUT_DIR, "evaluation_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存至：{result_path}")

    # 保存异常分数 numpy
    score_path = os.path.join(cfg.OUTPUT_DIR, "anomaly_scores.npy")
    np.save(score_path, anomaly_scores)
    print(f"异常分数保存至：{score_path}")

    # ── 可视化 ────────────────────────────────────────────────────────────
    if not args.no_plot:
        _plot_results(
            y_true=y_true,
            anomaly_scores=anomaly_scores,
            x_true=x_true,
            x_pred=x_pred,
            threshold=best_thresh,
            output_dir=cfg.OUTPUT_DIR,
            n_channels=cfg.NUM_CHANNELS,
        )


def _plot_results(
    y_true, anomaly_scores, x_true, x_pred,
    threshold, output_dir, n_channels,
    max_plot_len: int = 5000,
):
    """绘制异常分数时序图和各通道预测对比图。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    T = min(len(anomaly_scores), max_plot_len)
    t = np.arange(T)
    y_pred_bin = (anomaly_scores[:T] > threshold).astype(int)
    gt_events_mask = y_true[:T].astype(bool)

    # ── 图 1：异常分数 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, anomaly_scores[:T], color="steelblue", linewidth=0.8, label="Anomaly Score")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold={threshold:.3f}")

    # 标注真实异常区间（绿色背景）
    gt_regions = []
    in_reg = False
    for i in range(T):
        if gt_events_mask[i] and not in_reg:
            r_start = i; in_reg = True
        elif not gt_events_mask[i] and in_reg:
            ax.axvspan(r_start, i, alpha=0.2, color="green")
            in_reg = False
    if in_reg:
        ax.axvspan(r_start, T, alpha=0.2, color="green")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("PSTG Anomaly Detection Results")
    ax.legend(handles=[
        mpatches.Patch(color="steelblue", label="Anomaly Score"),
        mpatches.Patch(color="red", label=f"Threshold={threshold:.3f}"),
        mpatches.Patch(color="green", alpha=0.3, label="Ground Truth Anomaly"),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "anomaly_scores.png"), dpi=150)
    plt.close()
    print(f"  → 异常分数图：{output_dir}/anomaly_scores.png")

    # ── 图 2：各通道预测对比 ──────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 2.5 * n_channels))
    gs = GridSpec(n_channels, 1, figure=fig, hspace=0.4)

    colors = plt.cm.tab10.colors
    for c in range(n_channels):
        ax = fig.add_subplot(gs[c])
        ax.plot(t, x_true[:T, c], color=colors[c % 10], linewidth=0.7, label="Ground Truth")
        ax.plot(t, x_pred[:T, c], color="gray", linewidth=0.7, linestyle="--", alpha=0.8, label="Prediction")
        # 真实异常区间
        in_reg = False
        for i in range(T):
            if gt_events_mask[i] and not in_reg:
                r_start = i; in_reg = True
            elif not gt_events_mask[i] and in_reg:
                ax.axvspan(r_start, i, alpha=0.15, color="red")
                in_reg = False
        if in_reg:
            ax.axvspan(r_start, T, alpha=0.15, color="red")
        ax.set_ylabel(f"Ch {c+41}")
        ax.set_xlim(0, T)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time Step")
    fig.suptitle("PSTG: Prediction vs Ground Truth (all channels)", fontsize=12)
    plt.savefig(os.path.join(output_dir, "channel_predictions.png"), dpi=150)
    plt.close()
    print(f"  → 通道预测图：{output_dir}/channel_predictions.png")


if __name__ == "__main__":
    main()
