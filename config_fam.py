"""
PSTG-FAM 超参数配置（继承 PSTG Config）
"""

from config import Config


class ConfigFAM(Config):
    """PSTG-FAM 配置，只新增 FAM 相关参数。"""

    # ── FAM 参数 ────────────────────────────────────────────────────────
    FAM_TOP_K_RATE = 0.5    # 保留前 50% 能量最大的频率分量
                             # L=250 → n_freq=126 → keep 63 components
                             # 可调范围：[0.1, 0.7]，越小去噪越激进

    # ── 输出路径（独立于 PSTG）────────────────────────────────────────────
    CHECKPOINT_DIR = "./checkpoints_fam"
    OUTPUT_DIR     = "./outputs_fam"
