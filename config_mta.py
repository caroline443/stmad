"""
MTA 配置文件

继承 Config 基础配置，新增掩码重建相关参数。
与 PSTG 使用相同的编码器结构（patch_sizes / d_model / num_heads / num_layers），
只有 MTA 特有参数（mask_ratio）和输出路径不同。
"""

from config import Config


class ConfigMTA(Config):

    # ── MTA 特有参数 ─────────────────────────────────────────────────────────

    # 训练时随机掩码的时间 patch 比例
    # N=10 个 patch，mask_ratio=0.4 → 每次掩码 4 个 patch
    # 可调范围：[0.3, 0.6]（太低 → 任务太简单；太高 → 信号太少）
    MASK_RATIO: float = 0.4

    # ── 输出目录（与 PSTG 隔离，避免 checkpoint 混淆）─────────────────────────
    CHECKPOINT_DIR: str = "checkpoints_mta"
    OUTPUT_DIR:     str = "outputs_mta"

    MODEL_NAME: str = "mta"

    # ── 训练参数（与 PSTG 保持一致以便公平比较）──────────────────────────────
    # 其余参数全部继承自 Config（PATCH_SIZES, D_MODEL, NUM_HEADS, NUM_LAYERS 等）
