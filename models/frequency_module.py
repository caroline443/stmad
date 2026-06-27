"""
频率感知滤波模块（Frequency-Aware Filter，FAM）

来源：ContrastAD (Pei et al., CIKM 2026)，Section 3.4
核心思想：在 patch embedding 之前对原始时序做频率域去噪，
  保留能量最大的 K 个频率分量（周期性主成分），
  抑制高频噪声（测量噪声、传感器抖动），
  使模型学到更干净的正常模式 → 降低正常数据的预测残差 → 更好的异常分离。

原论文公式：
  H_l = IFFT(TopK(|FFT(Z̄)|))

在 PSTG 中的位置：
  Input X [B, C, L]
    ↓ FrequencyAwareFilter
  X_filtered [B, C, L]   ← 频率过滤后，高频噪声被去除
    ↓ MultiScalePatchEmbedding
  Z_fused [B, n, D]
    ↓ ... (同 PSTG)

特点：
  - 零额外参数（纯 FFT 运算）
  - 可直接从 PSTG checkpoint 热启动
  - 适合航天器遥测：数据有强周期性，高频成分主要是噪声
"""

import torch
import torch.nn as nn


class FrequencyAwareFilter(nn.Module):
    """
    频率感知滤波器：对时序信号做 FFT，保留能量最大的 Top-K 频率分量，再 IFFT 还原。

    与低通滤波的区别：
      低通滤波：保留频率索引最小的 K 个分量（固定低频）
      FAM：保留幅度最大的 K 个分量（自适应地保留主要模式，包括非低频谐波）

    Args:
        top_k_rate: 保留的频率分量比例，0 < rate ≤ 1
                    rate=0.5：保留前 50% 能量最大的分量（默认）
                    rate=0.3：更激进的去噪
    """

    def __init__(self, top_k_rate: float = 0.5):
        super().__init__()
        assert 0 < top_k_rate <= 1.0
        self.top_k_rate = top_k_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] 原始时序
        Returns:
            x_filtered: [B, C, L] 频率滤波后的时序（与输入形状完全相同）
        """
        B, C, L = x.shape

        # 1. 实数 FFT（沿 L 维，即时间轴）
        #    输出：[B, C, L//2+1] 复数张量
        x_fft = torch.fft.rfft(x, dim=-1)

        # 2. 计算每个频率分量的幅度（能量的平方根）
        amp = x_fft.abs()   # [B, C, L//2+1]

        # 3. Top-K 选择：保留幅度最大的 K 个频率
        n_freq = amp.shape[-1]           # L//2 + 1
        k = max(1, int(n_freq * self.top_k_rate))

        _, top_idx = torch.topk(amp, k, dim=-1)          # [B, C, K]
        mask = torch.zeros_like(amp, dtype=torch.bool)
        mask.scatter_(-1, top_idx, True)                  # [B, C, L//2+1]

        # 4. 应用掩码（零化低能量频率分量）
        x_fft_filtered = x_fft * mask.to(x_fft.dtype)    # 复数乘以 0/1

        # 5. IFFT 还原到时域
        x_filtered = torch.fft.irfft(x_fft_filtered, n=L, dim=-1)  # [B, C, L]

        return x_filtered

    def extra_repr(self) -> str:
        return f"top_k_rate={self.top_k_rate}"
