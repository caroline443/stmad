"""
SMAP / MSL 数据集配置
"""

import os
from config import Config


class ConfigSMAP(Config):
    """SMAP 数据集（55 通道，N=55，异常率 ~12.8%）"""

    DATASET_NAME  = "smap"
    DATA_DIR      = "/root/autodl-tmp/data/AT/SMAP"  # 可通过 --data_dir 覆盖
    NUM_CHANNELS  = 55
    CHANNELS      = list(range(55))   # 占位，smap_msl_loader 不用

    # 较大通道数 → 适当缩小 D 和 batch 防止 OOM
    D_MODEL     = 256
    BATCH_SIZE  = 32
    B_S         = 32
    TRAIN_STRIDE = 100    # SMAP 训练集 135K，stride=100 → 约 1350 样本

    # SMAP 异常率高（12.8%），threshold 不需要像 ESA-AD 那么保守
    P_TFI = 0.21

    # 输出目录
    CHECKPOINT_DIR = "./checkpoints_smap"
    OUTPUT_DIR     = "./outputs_smap"

    @property
    def top_k(self):
        n = self.NUM_CHANNELS * self.NUM_PATCHES
        return max(1, int(self.GAMMA * n))  # ceil(10% × 550) = 55


class ConfigMSL(Config):
    """MSL 数据集（27 通道，N=27，异常率 ~10.7%）"""

    DATASET_NAME  = "msl"
    DATA_DIR      = "/root/autodl-tmp/data/AT/MSL"  # 可通过 --data_dir 覆盖
    NUM_CHANNELS  = 27
    CHANNELS      = list(range(27))

    D_MODEL      = 512    # MSL 通道数适中，保持与 PSTG 一致
    BATCH_SIZE   = 64
    B_S          = 70
    TRAIN_STRIDE = 50

    P_TFI = 0.21

    CHECKPOINT_DIR = "./checkpoints_msl"
    OUTPUT_DIR     = "./outputs_msl"

    @property
    def top_k(self):
        n = self.NUM_CHANNELS * self.NUM_PATCHES
        return max(1, int(self.GAMMA * n))  # ceil(10% × 270) = 27
