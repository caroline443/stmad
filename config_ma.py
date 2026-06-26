"""
PSTG-MA 超参数配置（继承 PSTG Config，新增记忆库相关参数）
"""

from config import Config


class ConfigMA(Config):
    """
    PSTG-MA 配置。
    所有 PSTG 参数不变，新增记忆库 + 训练策略参数。
    """

    # ── 记忆库参数 ────────────────────────────────────────────────────────
    NUM_MEMORY_SLOTS    = 200      # K：记忆槽数量
    MEMORY_TEMPERATURE  = 0.1     # 软寻址温度（越小越稀疏）
    MEMORY_SHRINK_THRESH = None   # hard shrinkage 阈值（None → 1/K = 0.005）

    # ── 损失权重（在 PSTG 的 λ1=λ2=0.1 基础上新增）─────────────────────
    LAMBDA_MEM  = 0.1    # 记忆重构损失权重
    LAMBDA_ENT  = 0.02   # 熵正则化权重

    # ── 分阶段训练策略 ───────────────────────────────────────────────────
    WARMUP_EPOCHS = 10   # 前 N 轮只用预测损失（Warmup）
    # 建议：如果从 PSTG checkpoint 热启动，可以把 WARMUP_EPOCHS 设为 0

    # ── 双信号融合权重 ───────────────────────────────────────────────────
    ALPHA_PRED = 0.6     # 预测残差的权重（1-ALPHA_PRED = 记忆误差权重）

    # ── 训练总轮次（建议比 PSTG 少，因为可以热启动）────────────────────
    NUM_EPOCHS    = 70
    TRAIN_STRIDE  = 50
