# PSTG 论文复现

**Progressive Spatiotemporal Graph Modeling for Spacecraft Anomaly Detection**  
*Entropy 2026, 28, 426*

---

## 项目结构

```
论文复现/
├── config.py              # 所有超参数（对应论文 Table 2, 3）
├── requirements.txt
├── data/
│   └── dataset.py         # ESA-AD 数据加载 + 滑窗采样
├── models/
│   ├── patch_embedding.py # 多尺度 Patch 嵌入 + Gated Attention Fusion
│   ├── graph_module.py    # 动态图构建 + 改进 GATv2
│   └── pstg.py            # 完整 PSTG 模型
├── utils/
│   ├── loss.py            # 复合损失（MSE + Freq + Shape）
│   └── metrics.py         # Event-wise F0.5 + Affiliation-based F0.5
├── anomaly/
│   └── detector.py        # 动态阈值检测（Φ 算子）
├── train.py               # 训练脚本
├── evaluate.py            # 推理 + 评估 + 可视化
└── visualize.py           # 邻接矩阵 + 注意力矩阵热力图
```

---

## 环境安装

```bash
pip install -r requirements.txt
```

---

## 数据集

使用 **ESA-AD Mission 1 Lightweight Subset**（subsystem 5，channels 41-46）。

数据来源：  
https://zenodo.org/records/12528696

**预期目录结构**（至少需要以下任一格式）：
```
/root/autodl-tmp/data/ESA-Mission1/
├── train.csv   # 列: timestamp, channel_41, ..., channel_46, is_anomaly
└── test.csv
```

若列名不同，程序会自动识别所有数值列。

---

## 使用方法

### 1. 修改数据路径（如有需要）

编辑 `config.py` 中的 `DATA_DIR`，或通过命令行参数传入。

### 2. 训练

```bash
python train.py
# 自定义参数：
python train.py --epochs 70 --batch_size 64 --data_dir /root/autodl-tmp/data/ESA-Mission1
# 续训：
python train.py --resume ./checkpoints/last.pt
```

训练期间每轮打印 train/val loss，最优 checkpoint 保存在 `checkpoints/best.pt`。

### 3. 评估

```bash
python evaluate.py
# 指定 checkpoint：
python evaluate.py --ckpt ./checkpoints/best.pt
```

输出：
- `outputs/evaluation_results.json`：Event-wise 和 Affiliation-based F0.5 分数
- `outputs/anomaly_scores.npy`：连续异常分数序列
- `outputs/anomaly_scores.png`：异常分数时序图
- `outputs/channel_predictions.png`：各通道预测对比图

### 4. 可视化（论文 Figure 3）

```bash
python visualize.py
```

输出：
- `outputs/graph_matrices.png`：邻接矩阵 + 注意力矩阵热力图（2×2）
- 终端打印 Shannon Entropy 分析（公式 30）

---

## 超参数（对应论文 Table 2-3）

| 参数 | 值 | 说明 |
|------|----|------|
| `CONTEXT_LEN` (L) | 250 | 输入窗口长度 |
| `FORECAST_LEN` (F) | 10 | 预测步长 |
| `PATCH_SIZES` (P) | [25,50,125] | 三种 patch 尺寸 |
| `D_MODEL` (D) | 512 | 嵌入维度 |
| `NUM_HEADS` (H) | 4 | 图注意力头数 |
| `NUM_LAYERS` (n_L) | 2 | Progressive 层数 |
| `GAMMA` (γ) | 0.1 | Top-k 稀疏化比例 |
| `LEARNING_RATE` | 5e-4 | AdamW 学习率 |
| `WEIGHT_DECAY` | 4e-4 | AdamW 权重衰减 |
| `T_MAX` | 70 | CosineAnnealing 周期 |

---

## 目标结果（论文 Table 5）

| 指标 | 目标值 |
|------|--------|
| Event-wise Precision | 0.932 |
| Event-wise Recall | 0.862 |
| **Event-wise F0.5** | **0.917** |
| Affiliation Precision | 0.905 |
| Affiliation Recall | 0.844 |
| **Affiliation F0.5** | **0.892** |

---

## 模型架构摘要

```
Input: X ∈ R^(B × C × L)   [B=batch, C=6 channels, L=250 time steps]
         ↓
P: Multi-Scale Patch Embedding
   ├─ Scale 1 (p=25):  stride=25, N=10 patches → Linear(25→512) + PosEnc
   ├─ Scale 2 (p=50):  stride=22, N=10 patches → Linear(50→512) + PosEnc
   └─ Scale 3 (p=125): stride=13, N=10 patches → Linear(125→512) + PosEnc
   Gated Attention Fusion → Z_fused ∈ R^(B × 60 × 512)
         ↓
G^(1): Layer 1 Spatiotemporal Graph Reasoning
   ├─ DynamicGraphLearner: 4-head → A_final ∈ R^(B × 4 × 60 × 60)
   └─ StructureGuidedGATv2 → H^[1] ∈ R^(B × 60 × 512)
         ↓
G^(2): Layer 2 Spatiotemporal Graph Reasoning
   ├─ DynamicGraphLearner: 4-head → A_final
   └─ StructureGuidedGATv2 → H^[2] ∈ R^(B × 60 × 512)
         ↓
T: Forecast Head
   reshape → (B, 6, 10, 512) → flatten → (B, 6, 5120) → Linear(5120→10)
Output: X̂ ∈ R^(B × 6 × 10)
```
