"""
PSTG 超参数配置
对应论文 Table 2（训练超参）和 Table 3（模型超参）
"""

import os


class Config:
    # ── 数据路径 ──────────────────────────────────────────
    DATA_DIR = "/root/autodl-tmp/data/ESA-Mission1"
    CHANNELS = list(range(41, 47))          # channels 41-46，共 6 个
    NUM_CHANNELS = 6                        # C

    # ── 输出路径 ──────────────────────────────────────────
    OUTPUT_DIR = "./outputs"
    CHECKPOINT_DIR = "./checkpoints"

    # ── 时序窗口 ─────────────────────────────────────────
    CONTEXT_LEN = 250                       # L：上下文长度
    FORECAST_LEN = 10                       # F：预测步长
    TAU = 1                                 # τ：推理时每次保留的步数

    # ── 多尺度 Patch 参数（Table 3）─────────────────────
    PATCH_SIZES = [25, 50, 125]             # P：三种 patch 尺寸
    PATCH_MAIN = 25                         # p_main：计算 N 的基准尺寸
    NUM_PATCHES = CONTEXT_LEN // PATCH_MAIN # N = L // p_main = 10

    # ── 模型超参（Table 3）───────────────────────────────
    D_MODEL = 512                           # D：嵌入维度
    NUM_HEADS = 4                           # H：图注意力头数
    NUM_LAYERS = 2                          # n_L：Progressive 层数
    GAMMA = 0.1                             # γ：稀疏化比例（top-k）
    P_DROPOUT = 0.1                         # Dropout 率
    P_TFI = 0.21                            # 假阳性剪枝阈值 p_δ

    # ── 训练超参（Table 2）───────────────────────────────
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 4e-4
    T_MAX = 70                              # CosineAnnealing 周期
    ETA_MIN = 0                             # 最小学习率
    BATCH_SIZE = 64
    NUM_EPOCHS = 70
    TRAIN_STRIDE = 50                       # 训练集滑窗步长

    # ── 损失函数权重 ──────────────────────────────────────
    LAMBDA1 = 0.1                           # 频域损失权重
    LAMBDA2 = 0.1                           # 形态损失权重

    # ── 异常检测（Table 3）───────────────────────────────
    P_S = 0.05                              # 平滑调节百分比
    N_S = 30                               # 基准因子
    B_S = 70                               # 测试批大小（用于计算平滑窗口）
    # 平滑窗口：W_s = p_s * n_s * B_s = 0.05 * 30 * 70 = 105

    # ── 计算设备 ──────────────────────────────────────────
    DEVICE = "cuda"                         # 或 "cpu"

    # ── 随机种子 ──────────────────────────────────────────
    SEED = 42

    # ── 日志 ─────────────────────────────────────────────
    LOG_INTERVAL = 50                       # 每隔多少 batch 打印一次

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

    @property
    def num_nodes(self):
        """时空节点数：n = C × N"""
        return self.NUM_CHANNELS * self.NUM_PATCHES   # 6 × 10 = 60

    @property
    def head_dim(self):
        """每个注意力头的维度"""
        return self.D_MODEL // self.NUM_HEADS           # 512 // 4 = 128

    @property
    def top_k(self):
        """Top-k 稀疏化的 k 值"""
        return max(1, int(self.GAMMA * self.num_nodes)) # ceil(0.1 × 60) = 6

    @property
    def smooth_window(self):
        """异常分数平滑窗口大小"""
        return int(self.P_S * self.N_S * self.B_S)     # 105
