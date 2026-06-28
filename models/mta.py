"""
MTA: Masked Temporal Autoencoder for Spacecraft Anomaly Detection

新范式（相对于 PSTG 的预测范式）：
  PSTG：X[t-L:t] → 预测 X[t:t+F]，用预测误差检测异常
  MTA ：X[t-L:t]（随机掩码部分 patch）→ 重建被掩码的 patch，用重建误差检测异常

架构：
  Encoder（与 PSTG 完全相同）：
    MultiScalePatchEmbedding → SpatioTemporalGraphLayer × 2
  Decoder（新增，轻量 MLP）：
    [B, C, N, D] → [B, C, N, p_main]

训练：
  随机掩码 mask_ratio 比例的时间 patch，替换为可学习 mask_token，
  仅在掩码位置计算重建损失（MSE + Freq + Shape）

推理（异常检测）：
  不掩码，直接重建所有 patch，
  重建误差 = 异常分数，后接相同的 smooth + POT 流程
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embedding import MultiScalePatchEmbedding
from .graph_module import SpatioTemporalGraphLayer


# ─────────────────────────────────────────────────────────────────────────────
#  解码器
# ─────────────────────────────────────────────────────────────────────────────

class PatchDecoder(nn.Module):
    """
    轻量 MLP 解码器：节点表示 → 原始 patch 值。

    结构：Linear(D → D//2) → GELU → LayerNorm → Linear(D//2 → p_main)

    输入：H ∈ R^(B, C, N, D)
    输出：X̂_patches ∈ R^(B, C, N, p_main)
    """

    def __init__(self, d_model: int, patch_main: int):
        super().__init__()
        hidden = max(d_model // 2, patch_main * 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, patch_main),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., D] → [..., p_main]"""
        return self.net(h)


# ─────────────────────────────────────────────────────────────────────────────
#  掩码重建损失
# ─────────────────────────────────────────────────────────────────────────────

class MTALoss(nn.Module):
    """
    掩码重建复合损失（仅对掩码位置计算）：
      L = L_MSE + λ1·L_freq + λ2·L_shape

    与 PSTGLoss 的区别：
      - 目标是当前窗口内的 patch（重建），而非未来步（预测）
      - 损失只对 mask=True 的 patch 计算（未掩码位置为 0 贡献）
      - mask=None 时退化为全 patch 损失（供推理诊断调用）
    """

    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(
        self,
        pred:   torch.Tensor,            # [B, C, N, p]
        target: torch.Tensor,            # [B, C, N, p]
        mask:   Optional[torch.Tensor],  # [B, N] bool，True=掩码位置
    ) -> Tuple[torch.Tensor, Tuple[float, float, float]]:

        B, C, N, p = pred.shape

        # 构造 4D 权重掩码
        if mask is not None:
            # [B, N] → [B, C, N, p]
            w = mask.unsqueeze(1).unsqueeze(-1).float().expand(B, C, N, p)
            n_w = w.sum().clamp(min=1.0)
        else:
            w = torch.ones(B, C, N, p, device=pred.device)
            n_w = torch.tensor(float(B * C * N * p), device=pred.device)

        # ── L_MSE ────────────────────────────────────────────────────────────
        mse = ((pred - target) ** 2 * w).sum() / n_w

        # ── L_freq：对 patch 时间轴做 FFT ─────────────────────────────────────
        pred_fft = torch.fft.rfft(pred, dim=-1)   # [B, C, N, p//2+1]
        tgt_fft  = torch.fft.rfft(target, dim=-1)
        diff_fft = pred_fft - tgt_fft
        freq_err = diff_fft.real ** 2 + diff_fft.imag ** 2  # [B, C, N, p//2+1]

        p_freq = freq_err.shape[-1]
        if mask is not None:
            w_freq = mask.unsqueeze(1).unsqueeze(-1).float().expand(B, C, N, p_freq)
        else:
            w_freq = torch.ones(B, C, N, p_freq, device=pred.device)
        n_freq = w_freq.sum().clamp(min=1.0)
        freq_loss = (freq_err * w_freq).sum() / n_freq

        # ── L_shape：patch 内相邻差分 ─────────────────────────────────────────
        pred_grad = pred[..., 1:] - pred[..., :-1]      # [B, C, N, p-1]
        tgt_grad  = target[..., 1:] - target[..., :-1]
        shape_err = (pred_grad - tgt_grad) ** 2

        p_shape = shape_err.shape[-1]
        if mask is not None:
            w_shape = mask.unsqueeze(1).unsqueeze(-1).float().expand(B, C, N, p_shape)
        else:
            w_shape = torch.ones(B, C, N, p_shape, device=pred.device)
        n_shape = w_shape.sum().clamp(min=1.0)
        shape_loss = (shape_err * w_shape).sum() / n_shape

        total = mse + self.lambda1 * freq_loss + self.lambda2 * shape_loss
        return total, (mse.item(), freq_loss.item(), shape_loss.item())


# ─────────────────────────────────────────────────────────────────────────────
#  主模型：MTA
# ─────────────────────────────────────────────────────────────────────────────

class MTA(nn.Module):
    """
    Masked Temporal Autoencoder（掩码时间自编码器）

    Args:
        patch_sizes:  多尺度 patch 尺寸，默认 [25, 50, 125]
        patch_main:   主 patch 尺寸（重建目标分辨率），默认 25
        d_model:      嵌入维度，默认 512
        num_heads:    图注意力头数，默认 4
        num_layers:   Progressive 图层数，默认 2
        top_k:        图稀疏化 top-k，默认 6
        n_channels:   输入通道数，默认 6
        context_len:  输入序列长度，默认 250
        mask_ratio:   训练时掩码比例，默认 0.4（40% 时间 patch）
        dropout:      Dropout 率，默认 0.1
    """

    def __init__(
        self,
        patch_sizes: list = None,
        patch_main:  int   = 25,
        d_model:     int   = 512,
        num_heads:   int   = 4,
        num_layers:  int   = 2,
        top_k:       int   = 6,
        n_channels:  int   = 6,
        context_len: int   = 250,
        mask_ratio:  float = 0.4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        self.n_channels  = n_channels
        self.d_model     = d_model
        self.patch_main  = patch_main
        self.n_patches   = context_len // patch_main   # N = 10
        self.mask_ratio  = mask_ratio

        # 主尺度 stride（与 ScalePatchEmbedding 中的计算保持一致）
        if self.n_patches > 1:
            self._stride_main = (context_len - patch_main) // (self.n_patches - 1)
        else:
            self._stride_main = context_len
        self._stride_main = max(1, self._stride_main)

        # ── 编码器（与 PSTG 完全相同的结构）────────────────────────────────────
        self.patch_embed = MultiScalePatchEmbedding(
            patch_sizes=patch_sizes,
            patch_main=patch_main,
            d_model=d_model,
            context_len=context_len,
            n_channels=n_channels,
        )
        self.graph_layers = nn.ModuleList([
            SpatioTemporalGraphLayer(
                d_model=d_model,
                num_heads=num_heads,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── 掩码令牌（可学习，所有位置共享）─────────────────────────────────────
        # shape [1, 1, 1, D]，broadcast 到 [B, C, N, D]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # ── 解码器──────────────────────────────────────────────────────────────
        self.decoder = PatchDecoder(d_model, patch_main)

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────
    #  内部工具方法
    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self):
        """Xavier 初始化所有线性层（与 PSTG 一致）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract_target_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        从原始输入中切取目标 patch 值（使用与 ScalePatchEmbedding 一致的切分方式）。

        Args:
            x: [B, C, L]
        Returns:
            patches: [B, C, N, p_main]
        """
        B, C, L = x.shape
        patches = []
        for i in range(self.n_patches):
            start = i * self._stride_main
            end   = start + self.patch_main
            if end <= L:
                p = x[:, :, start:end]
            else:
                p = F.pad(x[:, :, start:], (0, end - L))
            patches.append(p)
        return torch.stack(patches, dim=2)   # [B, C, N, p_main]

    def _generate_mask(self, B: int, device: torch.device) -> torch.Tensor:
        """
        每样本独立生成随机时间掩码（相同掩码位置跨所有通道共享）。
        向量化实现：对随机噪声排序，取前 n_mask 个位置。

        Returns:
            mask: [B, N] bool，True = 被掩码（需重建）
        """
        N = self.n_patches
        n_mask = max(1, round(N * self.mask_ratio))
        # [B, N] 随机噪声 → 排序 → 前 n_mask 列的 index 即掩码位置
        noise       = torch.rand(B, N, device=device)
        ids_shuffle = noise.argsort(dim=-1)                # [B, N]
        mask        = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)    # 前 n_mask 个位置置 True
        return mask

    # ──────────────────────────────────────────────────────────────────────────
    #  前向传播
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:    torch.Tensor,                   # [B, C, L]
        mask: Optional[torch.Tensor] = None,  # [B, N] bool（None → 推理，不掩码）
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            recon:  [B, C, N, p_main]  重建的 patch 值
            mask:   [B, N] bool | None  本次使用的掩码
            target: [B, C, N, p_main]  目标 patch（原始输入切取）
        """
        B, C, L = x.shape

        # 1. 在 embedding 之前提取目标（保留原始归一化值）
        target = self._extract_target_patches(x)           # [B, C, N, p_main]

        # 2. 多尺度 Patch 嵌入（编码器第一步）
        h = self.patch_embed(x)                            # [B, n, D]，n = C×N

        # 3. 生成并应用掩码（仅训练模式；推理时 mask 为 None）
        if mask is None and self.training:
            mask = self._generate_mask(B, x.device)

        if mask is not None:
            # 将 h 从 [B, n, D] 还原为 [B, C, N, D]
            h_3d = h.reshape(B, C, self.n_patches, self.d_model)
            # mask [B, N] → [B, 1, N, 1] → broadcast 到 [B, C, N, D]
            mask_flag = mask.unsqueeze(1).unsqueeze(-1)
            mask_tok  = self.mask_token.expand(B, C, self.n_patches, self.d_model)
            h_3d = torch.where(mask_flag.expand_as(h_3d), mask_tok, h_3d)
            h = h_3d.reshape(B, C * self.n_patches, self.d_model)

        # 4. 渐进时空图推理（编码器第二步）
        for layer in self.graph_layers:
            h, _ = layer(h)                                # [B, n, D]

        # 5. 解码：[B, n, D] → [B, C, N, D] → [B, C, N, p_main]
        h_3d  = h.reshape(B, C, self.n_patches, self.d_model)
        recon = self.decoder(h_3d)                         # [B, C, N, p_main]

        return recon, mask, target

    # ──────────────────────────────────────────────────────────────────────────
    #  推理：异常分数
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理模式（无掩码）：计算每个输入窗口的重建误差作为异常分数。

        异常分数 = max over channels(mean reconstruction error over patches)
          → 对任何单通道的异常均敏感，平均时间 patch 减少随机扰动

        Args:
            x: [B, C, L]
        Returns:
            score: [B]
        """
        self.eval()
        recon, _, target = self.forward(x, mask=None)
        # [B, C, N, p_main] → patch 级平均误差 [B, C, N]
        patch_err = (recon - target).abs().mean(dim=-1)
        # max(channel) × max(patch)：捕捉最差通道最差时间 patch → [B]
        score = patch_err.max(dim=1).values.max(dim=-1)
        return score

    # ──────────────────────────────────────────────────────────────────────────
    #  工厂方法
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg):
        """从 ConfigMTA 构建 MTA 模型"""
        return cls(
            patch_sizes=cfg.PATCH_SIZES,
            patch_main=cfg.PATCH_MAIN,
            d_model=cfg.D_MODEL,
            num_heads=cfg.NUM_HEADS,
            num_layers=cfg.NUM_LAYERS,
            top_k=cfg.top_k,
            n_channels=cfg.NUM_CHANNELS,
            context_len=cfg.CONTEXT_LEN,
            mask_ratio=cfg.MASK_RATIO,
            dropout=cfg.P_DROPOUT,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
