"""
PSTG 可解释性可视化脚本（对应论文 Section 4.7，Figure 3）

绘制两层 PSTG 网络的 4 组矩阵：
  (a) 第一层邻接矩阵（空间依赖）
  (b) 第一层聚合注意力矩阵（时空相关）
  (c) 第二层邻接矩阵（剪枝后连接）
  (d) 第二层聚合注意力矩阵（关键时空链路）

同时计算并打印每层的 Shannon Entropy H（公式 30）。

用法：
    python visualize.py [--ckpt ./checkpoints/best.pt] [--n_samples 100]
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from data.dataset import build_datasets
from models.pstg import PSTG


def parse_args():
    parser = argparse.ArgumentParser(description="PSTG 可视化")
    parser.add_argument("--ckpt",     type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device",   type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=100,
                        help="用于平均的样本数量（从验证集随机取）")
    parser.add_argument("--output",   type=str, default=None)
    return parser.parse_args()


def shannon_entropy(A: np.ndarray) -> float:
    """
    公式 30：H(i) = -Σ_j p_ij log(p_ij)，对所有节点 i 取均值。

    Args:
        A: [n, n] 归一化的邻接/注意力矩阵（每行为概率分布）
    """
    # 避免 log(0)
    eps = 1e-9
    A = np.clip(A, eps, 1.0)
    h_per_row = -np.sum(A * np.log(A), axis=-1)   # [n]
    return float(np.mean(h_per_row))


@torch.no_grad()
def collect_matrices(
    model: PSTG,
    loader,
    device: str,
    n_samples: int,
) -> list:
    """
    收集 n_samples 个批次的邻接矩阵和注意力矩阵，返回每层的平均值。

    Returns:
        adj_list: n_L 个 [n, n] numpy 数组（平均邻接矩阵，对所有头取均值）
        attn_list: n_L 个 [n, n] numpy 数组（平均注意力矩阵）
    """
    model.eval()
    # 我们需要同时拿到邻接矩阵（来自 DynamicGraphLearner）和注意力矩阵（来自 StructureGuidedGraphAttention）
    # 通过 hook 捕获中间变量

    n_layers = model.num_layers
    adj_accum  = [None] * n_layers   # 累计邻接矩阵
    attn_accum = [None] * n_layers   # 累计注意力矩阵
    counts = [0] * n_layers

    # 注册 forward hook 到每一层的 graph_learner 和 graph_attn
    handles = []
    layer_adj = {}
    layer_attn = {}

    def make_adj_hook(layer_idx):
        def hook(module, input, output):
            # output: A_final [B, H, n, n]，对头取均值
            A = output.detach().cpu().float().mean(dim=1)   # [B, n, n]
            A_mean = A.mean(dim=0).numpy()                   # [n, n]
            layer_adj[layer_idx] = A_mean
        return hook

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # output: [B, n, D]，我们需要 alpha
            # 注意：此处 hook 在 graph_attn 的 forward 内部难以直接拿到 alpha
            # 改用 graph_attn.attn_last 存储（需要稍微修改）
            pass
        return hook

    # 更简洁：直接修改 forward 使其返回 adj + attn
    # 这里我们通过重新调用 model forward with return_adj=True 来拿邻接矩阵
    # 注意力矩阵通过在 StructureGuidedGraphAttention 中添加 hook 捕获

    # ── 捕获注意力权重的 Hook ────────────────────────────────────────────
    # 给每层的 graph_attn 注册 hook，捕获 alpha（注意力系数）
    # 需要在 forward 中把 alpha 存到模块属性
    for li, layer in enumerate(model.graph_layers):
        # Monkey-patch：让 graph_attn 保存最后一次的 alpha
        orig_forward = layer.graph_attn.forward

        def patched_forward(z, A_final, _orig=orig_forward, _li=li):
            B, n, D = z.shape
            H = model.graph_layers[_li].graph_attn.H
            head_dim = model.graph_layers[_li].graph_attn.head_dim

            Q = model.graph_layers[_li].graph_attn.W_Q(z)
            K = model.graph_layers[_li].graph_attn.W_K(z)
            V = model.graph_layers[_li].graph_attn.W_V(z)

            def split_heads(t):
                return t.reshape(B, n, H, head_dim).permute(0, 2, 1, 3)

            Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
            import torch.nn.functional as F
            Q_i = Q.unsqueeze(-2)
            K_j = K.unsqueeze(-3)
            QK_cat = torch.cat([Q_i.expand(-1,-1,n,n,-1), K_j.expand(-1,-1,n,n,-1)], dim=-1)
            raw_score = (model.graph_layers[_li].graph_attn.w_A * QK_cat).sum(dim=-1)
            raw_score = model.graph_layers[_li].graph_attn.leaky_relu(raw_score)
            e = A_final * raw_score
            mask = (A_final == 0)
            e = e.masked_fill(mask, float("-inf"))
            alpha = F.softmax(e, dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            # 保存 alpha：取头均值，取批均值
            layer_attn[_li] = alpha.detach().cpu().float().mean(dim=0).mean(dim=0).numpy()  # [n, n]

            # 继续正常 forward
            M = torch.matmul(alpha, V)
            M = M.permute(0, 2, 1, 3).reshape(B, n, D)
            M = model.graph_layers[_li].graph_attn.W_O(M)
            out = model.graph_layers[_li].graph_attn.layer_norm(z + M)
            return out

        layer.graph_attn.forward = patched_forward

    # ── 采样推理 ──────────────────────────────────────────────────────────
    sample_count = 0
    for context, _ in tqdm(loader, desc="收集矩阵"):
        if sample_count >= n_samples:
            break
        context = context.to(device)

        # 使用 return_adj=True 拿邻接矩阵
        _, adj_list = model(context, return_adj=True)

        for li in range(n_layers):
            # adj_list[li]: [B, H, n, n]
            adj_mean = adj_list[li].float().mean(dim=0).mean(dim=0).numpy()   # [n, n]
            if adj_accum[li] is None:
                adj_accum[li] = adj_mean
                attn_accum[li] = layer_attn.get(li, np.zeros_like(adj_mean))
            else:
                adj_accum[li] += adj_mean
                attn_accum[li] += layer_attn.get(li, np.zeros_like(adj_mean))
            counts[li] += 1

        sample_count += context.shape[0]

    # 平均
    for li in range(n_layers):
        if counts[li] > 0:
            adj_accum[li]  /= counts[li]
            attn_accum[li] /= counts[li]

    return adj_accum, attn_accum


def plot_matrix_grid(
    adj_list: list,
    attn_list: list,
    n_channels: int,
    n_patches: int,
    output_dir: str,
):
    """
    绘制 2×2 矩阵网格（对应论文 Figure 3）：
    (a) 第一层邻接矩阵
    (b) 第一层注意力矩阵
    (c) 第二层邻接矩阵
    (d) 第二层注意力矩阵
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    n = adj_list[0].shape[0]   # = C * N = 60

    titles = [
        "(a) Layer 1: Adjacency Matrix\n(Spatial Dependency)",
        "(b) Layer 1: Attention Matrix\n(Spatiotemporal Dependency)",
        "(c) Layer 2: Adjacency Matrix\n(Integrated & Pruned)",
        "(d) Layer 2: Attention Matrix\n(Selective Critical Links)",
    ]
    matrices = [adj_list[0], attn_list[0], adj_list[1], attn_list[1]]
    colors = ["Blues", "Oranges", "Blues", "Oranges"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, mat, title, cmap in zip(axes, matrices, titles, colors):
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Source Node")
        ax.set_ylabel("Target Node")
        # 标记通道边界
        for c in range(1, n_channels):
            ax.axhline(c * n_patches - 0.5, color="black", linewidth=0.4, alpha=0.5)
            ax.axvline(c * n_patches - 0.5, color="black", linewidth=0.4, alpha=0.5)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle(
        f"PSTG Spatiotemporal Graph Visualization\n"
        f"(n={n} nodes = {n_channels} channels × {n_patches} patches)",
        fontsize=11
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, "graph_matrices.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n图矩阵热力图保存至：{out_path}")


def print_entropy_analysis(adj_list: list, attn_list: list):
    """
    公式 30：计算每层邻接矩阵和注意力矩阵的 Shannon Entropy。
    对应论文 Section 4.7 的量化分析。
    """
    print("\n─── Shannon Entropy 分析（公式 30）───")
    print(f"{'层':>4}  {'邻接矩阵 H':>12}  {'注意力矩阵 H':>14}")
    print("-" * 36)
    for li, (adj, attn) in enumerate(zip(adj_list, attn_list)):
        h_adj  = shannon_entropy(adj)
        h_attn = shannon_entropy(attn)
        print(f"  L{li+1}  {h_adj:12.4f}  {h_attn:14.4f}")

    print("\n论文报告值：")
    print("  邻接矩阵 H: L1 ≈ 2.94 → L2 ≈ 3.43（逐渐增大，扩展感受野）")
    print("  注意力矩阵 H: L1 ≈ 3.95 → L2 ≈ 3.54（逐渐减小，聚焦关键连接）")


def main():
    args = parse_args()
    cfg = Config()

    if args.data_dir: cfg.DATA_DIR   = args.data_dir
    if args.device:   cfg.DEVICE     = args.device
    if args.output:   cfg.OUTPUT_DIR = args.output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    # ── 加载模型 ──────────────────────────────────────────────────────────
    ckpt_path = args.ckpt or os.path.join(cfg.CHECKPOINT_DIR, "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_path}")

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
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    # ── 加载数据（用验证集）────────────────────────────────────────────────
    print("加载验证集...")
    data = build_datasets(cfg)
    val_loader = data["val_loader"]

    # ── 收集矩阵 ──────────────────────────────────────────────────────────
    print(f"采样 {args.n_samples} 个样本...")
    adj_list, attn_list = collect_matrices(model, val_loader, device, args.n_samples)

    # ── Shannon Entropy 分析 ──────────────────────────────────────────────
    print_entropy_analysis(adj_list, attn_list)

    # ── 绘图 ──────────────────────────────────────────────────────────────
    plot_matrix_grid(
        adj_list=adj_list,
        attn_list=attn_list,
        n_channels=cfg.NUM_CHANNELS,
        n_patches=cfg.NUM_PATCHES,
        output_dir=cfg.OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
