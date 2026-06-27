"""PSTG-Mamba 超参数配置"""
from config import Config


class ConfigMamba(Config):
    # Mamba SSM 参数
    MAMBA_D_STATE = 16   # SSM 状态维度（d_state）
    MAMBA_D_CONV  = 4    # 局部卷积宽度（d_conv）

    # 输出目录
    CHECKPOINT_DIR = "./checkpoints_mamba"
    OUTPUT_DIR     = "./outputs_mamba"
