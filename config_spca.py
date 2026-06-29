"""
SpCA 配置文件

SpCA 是独立于 PSTG 的新架构，参数设计上：
  - D_MODEL=256（PSTG 用 512，SpCA 因无图计算，256 已足够）
  - 三个频段：低(<10% Nyquist) / 中(10-40%) / 高(40-100%)
  - 每频段 1 层跨通道注意力 + 融合后 2 层全局注意力
"""

from config import Config


class ConfigSpCA(Config):

    # ── 频段设计 ─────────────────────────────────────────────────────────────
    N_BANDS:      int   = 3              # 频段数（低/中/高）
    BAND_SPLITS:  tuple = (0.1, 0.4)    # 相对于 Nyquist 的分割点
    N_PATCHES:        int  = 0       # 0=v1线性, >0=v2时序注意力
    # 组件消融开关
    USE_SPECTRAL:     bool = True    # False → 去掉频域分解
    USE_CHANNEL_ATTN: bool = True    # False → 去掉跨通道注意力

    # ── 模型结构（独立于 PSTG 参数）──────────────────────────────────────────
    D_MODEL:         int = 192           # 嵌入维度（PSTG: 512；SpCA无图故192已足够，参数量≈PSTG）
    NUM_HEADS:       int = 4             # 注意力头数
    N_LAYERS_BAND:   int = 1            # 每频段注意力层数
    N_LAYERS_GLOBAL: int = 2            # 融合后全局注意力层数
    P_DROPOUT:     float = 0.1

    # ── 训练（与 PSTG 保持一致，便于公平比较）────────────────────────────────
    # LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, BATCH_SIZE, TRAIN_STRIDE 全部继承

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    CHECKPOINT_DIR: str = "checkpoints_spca"
    OUTPUT_DIR:     str = "outputs_spca"
    MODEL_NAME:     str = "spca"
