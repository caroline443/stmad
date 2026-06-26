"""
记忆库模块（Memory Bank）

核心思想：
  用 K 个可学习的记忆槽存储训练集的正常模式原型。
  - 正常样本：寻址权重集中（少数槽被激活）→ 重构误差低
  - 异常样本：寻址权重分散（找不到匹配原型）→ 重构误差高

设计参考：
  - MemAE (Gong et al., ICCV 2019): hard shrinkage 稀疏化
  - MNAD (Park et al., CVPR 2020): 记忆寻址的异常检测

与 PSTG 的接口：
  输入 z = H^[nL] ∈ R^(B×n×D)（graph reasoning 的最终隐状态）
  输出 z_hat（记忆重构）、mem_error（重构误差）、entropy（寻址熵）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MemoryBank(nn.Module):
    """
    可学习记忆库：K 个槽，每槽维度 D，通过余弦相似度软寻址。

    正向传播：
      1. 计算输入 z 与每个记忆槽的余弦相似度
      2. Softmax 得到寻址权重 w（训练时做 hard shrinkage 稀疏化）
      3. 用 w 对记忆槽做加权求和得到重构 z_hat
      4. 计算重构误差和寻址熵（两者均可作为推理时的异常信号）

    Args:
        num_slots:    记忆槽数量 K（默认 200）
        slot_dim:     每槽维度 D（与 d_model 相同）
        temperature:  软寻址温度 τ（越小越稀疏，默认 0.1）
        shrink_thresh: hard shrinkage 阈值（默认 1/K，即均匀分布时的期望值）
    """

    def __init__(
        self,
        num_slots:    int   = 200,
        slot_dim:     int   = 512,
        temperature:  float = 0.1,
        shrink_thresh: float = None,
    ):
        super().__init__()
        self.K = num_slots
        self.D = slot_dim
        self.temperature = temperature
        self.shrink_thresh = shrink_thresh if shrink_thresh is not None else 1.0 / num_slots

        # 可学习记忆槽，初始化为单位球面上的随机向量
        memory = torch.randn(num_slots, slot_dim)
        nn.init.kaiming_uniform_(memory, a=math.sqrt(5))
        self.memory = nn.Parameter(memory)   # [K, D]

    # ── 内部工具 ────────────────────────────────────────────────────────────

    def _hard_shrink(self, w: torch.Tensor) -> torch.Tensor:
        """
        Hard shrinkage（MemAE）：将小于阈值的权重归零，然后重新归一化。
        增强稀疏性，防止记忆槽被所有样本平均使用（否则丧失判别力）。

        w: [..., K]
        """
        w = F.relu(w - self.shrink_thresh)
        w_sum = w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return w / w_sum  # 重归一化

    # ── 前向传播 ─────────────────────────────────────────────────────────────

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: [B, n, D]  graph reasoning 的输出节点特征

        Returns:
            z_hat     : [B, n, D]   记忆重构
            w         : [B, n, K]   寻址权重（可用于可视化）
            mem_error : [B]         每个样本的平均重构误差（推理用）
            entropy   : [B]         每个样本的平均寻址熵（推理用）
        """
        B, n, D = z.shape

        # 1. L2 归一化：余弦相似度计算
        z_norm = F.normalize(z, dim=-1)                     # [B, n, D]
        M_norm = F.normalize(self.memory, dim=-1)           # [K, D]

        # 2. 相似度 → 软寻址权重
        sim = torch.matmul(z_norm, M_norm.T) / self.temperature  # [B, n, K]
        w   = F.softmax(sim, dim=-1)                              # [B, n, K]

        # 3. Hard shrinkage（仅训练阶段）
        if self.training:
            w = self._hard_shrink(w)

        # 4. 记忆重构
        z_hat = torch.matmul(w, self.memory)   # [B, n, D]

        # 5. 重构误差：逐节点 MSE → 平均到样本级
        mem_error = ((z.detach() - z_hat) ** 2).mean(dim=-1).mean(dim=-1)  # [B]

        # 6. 寻址熵：越高 → 越找不到匹配原型 → 越可能是异常
        entropy = -(w * (w + 1e-9).log()).sum(dim=-1).mean(dim=-1)  # [B]

        return {
            "z_hat":     z_hat,       # 用于计算训练 loss
            "w":         w,           # 用于可视化
            "mem_error": mem_error,   # 推理时作为异常分数的第二信号
            "entropy":   entropy,     # 推理时/训练正则化
        }

    @torch.no_grad()
    def get_memory_matrix(self) -> torch.Tensor:
        """返回归一化后的记忆槽矩阵，用于可视化。[K, D]"""
        return F.normalize(self.memory, dim=-1)
