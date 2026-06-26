"""
PSTG-MA v2 训练损失

总损失 = L_pred + λ_mem·L_mem + λ_ent·L_ent

L_pred：主预测损失（MSE + Freq + Shape，同 PSTG）

L_mem（记忆引导预测损失，v2）：
  ||x_future - x̂_mem_future||^2
  用正态性权重加权：低主预测误差的样本权重高
  → 记忆库优先学好正常模式，对异常预测自然差
  → 误差在数据空间，与主残差同量级，信号强

L_ent（熵正则化）：
  对正常样本（低主预测误差）最小化寻址熵
  → 训练记忆库对正常样本集中寻址（低熵）
  → 推理时异常样本寻址分散（高熵）形成对比
"""

import torch
import torch.nn as nn
from .loss import PSTGLoss


class PSTGMALoss(nn.Module):
    def __init__(
        self,
        lambda1:       float = 0.1,
        lambda2:       float = 0.1,
        lambda_mem:    float = 0.3,   # v2：数据空间误差信号更强，权重可以大一些
        lambda_ent:    float = 0.02,
        warmup_epochs: int   = 10,
    ):
        super().__init__()
        self.pred_loss     = PSTGLoss(lambda1=lambda1, lambda2=lambda2)
        self.lambda_mem    = lambda_mem
        self.lambda_ent    = lambda_ent
        self.warmup_epochs = warmup_epochs

    def forward(
        self,
        pred:        torch.Tensor,   # [B, C, F] 主预测 x̂
        pred_mem:    torch.Tensor,   # [B, C, F] 记忆引导预测 x̂_mem（v2）
        target:      torch.Tensor,   # [B, C, F] 真实值
        mem_outputs: dict,           # MemoryBank 输出
        epoch:       int = 999,
    ) -> tuple:
        # ── 1. 主预测损失（同 PSTG）──────────────────────────────────────
        loss_pred, (mse, freq, shape) = self.pred_loss(pred, target)

        if epoch <= self.warmup_epochs:
            return loss_pred, {
                "pred": loss_pred.item(), "mse": mse,
                "mem": 0.0, "ent": 0.0, "warmup": True,
            }

        # ── 2. 记忆引导预测损失（数据空间，v2 核心）─────────────────────
        # 正态性权重：主预测误差低的样本更可能是正常的，权重更高
        with torch.no_grad():
            per_sample_err = ((pred - target) ** 2).mean(dim=(1, 2))   # [B]
            err_norm       = (per_sample_err - per_sample_err.min()) / \
                             (per_sample_err.max() - per_sample_err.min() + 1e-9)
            w_normal = (1.0 - err_norm).detach()    # [B] 低误差→高权重

        # 记忆预测误差（逐样本 MSE）
        mem_pred_err = ((pred_mem - target) ** 2).mean(dim=(1, 2))     # [B]
        # 加权：正常样本的记忆预测误差被更多惩罚
        # → 记忆库学好正常模式；无法学好的异常→大误差→被检测
        loss_mem = (w_normal * mem_pred_err).mean()

        # ── 3. 熵正则化 ──────────────────────────────────────────────────
        entropy  = mem_outputs["entropy"]    # [B]
        loss_ent = (w_normal * entropy).mean()

        # ── 总损失 ────────────────────────────────────────────────────────
        total = loss_pred + self.lambda_mem * loss_mem + self.lambda_ent * loss_ent

        return total, {
            "pred":  loss_pred.item(),
            "mse":   mse, "freq": freq, "shape": shape,
            "mem":   loss_mem.item(),
            "ent":   loss_ent.item(),
            "warmup": False,
        }
