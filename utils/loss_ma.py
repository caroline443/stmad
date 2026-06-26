"""
PSTG-MA 训练损失（公式设计）

总损失 = L_pred + λ_mem·L_mem + λ_ent·L_ent

L_pred（不变）：MSE + λ1·Freq + λ2·Shape

L_mem（记忆重构损失）：
  ||H^[nL] - z_hat||^2_F （鼓励记忆库存储正常模式）
  使用"正态性权重"：低预测误差的样本权重更高
  → 记忆库重点学习正常样本的模式，自然无法编码异常

L_ent（熵正则化）：
  对低预测误差（正常）样本，最小化其寻址熵
  → 正常样本用少数几个记忆槽就够了（寻址集中）
  → 这与异常样本的高熵形成对比，增强推理时的可分性

分阶段训练：
  warmup 阶段（前 N epoch）：只用 L_pred，让 encoder 先收敛
  full 阶段：L_pred + L_mem + L_ent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import PSTGLoss


class PSTGMALoss(nn.Module):
    """
    PSTG-MA 完整训练损失

    Args:
        lambda1:       频域损失权重（继承自 PSTG）
        lambda2:       形态损失权重（继承自 PSTG）
        lambda_mem:    记忆重构损失权重
        lambda_ent:    熵正则化权重
        warmup_epochs: 前 N 轮只用预测损失（等 encoder 稳定后再激活记忆库）
    """

    def __init__(
        self,
        lambda1:       float = 0.1,
        lambda2:       float = 0.1,
        lambda_mem:    float = 0.1,
        lambda_ent:    float = 0.02,
        warmup_epochs: int   = 10,
    ):
        super().__init__()
        self.pred_loss   = PSTGLoss(lambda1=lambda1, lambda2=lambda2)
        self.lambda_mem  = lambda_mem
        self.lambda_ent  = lambda_ent
        self.warmup_epochs = warmup_epochs

    def forward(
        self,
        pred:        torch.Tensor,   # [B, C, F] 模型预测
        target:      torch.Tensor,   # [B, C, F] 真实值
        mem_outputs: dict,           # MemoryBank 的输出
        h_graph:     torch.Tensor,   # [B, n, D] H^[nL]（用于记忆重构 loss）
        epoch:       int = 999,      # 当前训练轮次（控制 warmup）
    ) -> tuple:
        """
        Returns:
            total_loss : scalar
            detail     : dict，各项损失的标量值（用于打印）
        """
        # ── 1. 预测损失（与 PSTG 完全相同）──────────────────────────────────
        loss_pred, (mse, freq, shape) = self.pred_loss(pred, target)

        if epoch <= self.warmup_epochs:
            # Warmup 阶段：只用预测损失，不激活记忆库
            return loss_pred, {
                "pred": loss_pred.item(), "mse": mse,
                "mem": 0.0, "ent": 0.0,
                "warmup": True,
            }

        # ── 2. 记忆重构损失 ──────────────────────────────────────────────────
        z_hat = mem_outputs["z_hat"]      # [B, n, D]（记忆重构）
        entropy = mem_outputs["entropy"]  # [B]

        # 正态性权重：当前 batch 中预测误差越低的样本权重越高
        # 这些样本更可能是"正常的"，记忆库应该重点学会重构它们
        with torch.no_grad():
            per_sample_pred_err = ((pred - target) ** 2).mean(dim=(1, 2))  # [B]
            # 归一化到 [0, 1]，然后取反（低误差 → 高权重）
            err_norm = (per_sample_pred_err - per_sample_pred_err.min()) / \
                       (per_sample_pred_err.max() - per_sample_pred_err.min() + 1e-9)
            normality_weight = (1.0 - err_norm).detach()   # [B]

        # 记忆重构误差（逐节点），聚合到样本级
        recon_err = ((h_graph.detach() - z_hat) ** 2).mean(dim=-1).mean(dim=-1)  # [B]

        # 用正态性权重加权：高权重样本（正常）的重构误差被更多惩罚
        # → 记忆库优先学好正常样本，对异常样本的重构自然差
        loss_mem = (normality_weight * recon_err).mean()

        # ── 3. 熵正则化 ──────────────────────────────────────────────────────
        # 鼓励正常样本的寻址权重集中（低熵）
        # 高熵 = 均匀分布 = 找不到好原型 = 可能是异常
        # 正常样本（normality_weight 高）应有低熵
        loss_ent = (normality_weight * entropy).mean()

        # ── 总损失 ────────────────────────────────────────────────────────────
        total = loss_pred + self.lambda_mem * loss_mem + self.lambda_ent * loss_ent

        return total, {
            "pred":    loss_pred.item(),
            "mse":     mse,
            "freq":    freq,
            "shape":   shape,
            "mem":     loss_mem.item(),
            "ent":     loss_ent.item(),
            "warmup":  False,
        }
