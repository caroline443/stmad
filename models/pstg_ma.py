"""
PSTG-MA：Memory-Augmented Progressive Spatiotemporal Graph

在 PSTG 基础上增加记忆库模块：
  H^[nL] → MemoryBank → (z_hat, mem_error, entropy)
            ↓（并行）
  H^[nL] → ForecastHead → x̂

训练：L_pred + λ_mem·L_mem_recon + λ_ent·L_entropy_reg
推理：dual_signal = α·R_pred + (1-α)·R_mem

与 PSTG 的完全兼容：
  - 可从 PSTG checkpoint 热启动（仅新增记忆库参数需要重新学习）
  - 推理接口与 PSTG 一致，额外返回 mem_outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embedding import MultiScalePatchEmbedding
from .graph_module import SpatioTemporalGraphLayer
from .pstg import ForecastHead
from .memory_module import MemoryBank


class PSTG_MA(nn.Module):
    """
    Memory-Augmented PSTG

    新增超参（相比 PSTG）：
        num_memory_slots : K，记忆槽数量（默认 200）
        memory_temperature: 软寻址温度（默认 0.1）
        memory_shrink_thresh: hard shrinkage 阈值（默认 1/K）
    """

    def __init__(
        self,
        # ── PSTG 原有参数 ──────────────────────────────────────────────────
        patch_sizes:       list  = None,
        patch_main:        int   = 25,
        d_model:           int   = 512,
        num_heads:         int   = 4,
        num_layers:        int   = 2,
        top_k:             int   = 6,
        n_channels:        int   = 6,
        context_len:       int   = 250,
        forecast_len:      int   = 10,
        dropout:           float = 0.1,
        # ── MA 新增参数 ────────────────────────────────────────────────────
        num_memory_slots:      int   = 200,
        memory_temperature:    float = 0.1,
        memory_shrink_thresh:  float = None,   # None → 1/K
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        self.n_channels = n_channels
        self.n_patches  = context_len // patch_main
        self.n_nodes    = n_channels * self.n_patches
        self.num_layers = num_layers

        # ── 算子 P：多尺度 Patch 嵌入（与 PSTG 完全相同）─────────────────
        self.patch_embed = MultiScalePatchEmbedding(
            patch_sizes=patch_sizes,
            patch_main=patch_main,
            d_model=d_model,
            context_len=context_len,
            n_channels=n_channels,
        )

        # ── 算子 G：Progressive 时空图推理（与 PSTG 完全相同）─────────────
        self.graph_layers = nn.ModuleList([
            SpatioTemporalGraphLayer(
                d_model=d_model,
                num_heads=num_heads,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── 算子 T：预测头（与 PSTG 完全相同）────────────────────────────
        self.forecast_head = ForecastHead(
            n_channels=n_channels,
            n_patches=self.n_patches,
            d_model=d_model,
            forecast_len=forecast_len,
        )

        # ── 新增：记忆库（Memory Bank）────────────────────────────────────
        self.memory_bank = MemoryBank(
            num_slots=num_memory_slots,
            slot_dim=d_model,
            temperature=memory_temperature,
            shrink_thresh=memory_shrink_thresh,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── 前向传播 ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x:           torch.Tensor,   # [B, C, L]
        return_adj:  bool = False,
    ):
        """
        Returns:
            x_hat      : [B, C, F]    预测值
            mem_outputs: dict         MemoryBank 的输出（训练/推理均使用）
            adj_list   : list(可选)   各层邻接矩阵，用于可视化
        """
        # 1. 多尺度 Patch 嵌入
        h = self.patch_embed(x)          # [B, n, D]

        # 2. Progressive 时空图推理
        adj_list = []
        for layer in self.graph_layers:
            h, A_final = layer(h)
            if return_adj:
                adj_list.append(A_final.detach().cpu())

        # 3. 记忆库（作用在 H^[nL] 上）
        self._last_h = h                    # 保存供训练 loss 使用
        mem_outputs  = self.memory_bank(h)  # z_hat, w, mem_error, entropy

        # 4. 预测头（直接用 H^[nL]，不经过记忆库）
        x_hat = self.forecast_head(h)       # [B, C, F]

        if return_adj:
            return x_hat, mem_outputs, adj_list
        return x_hat, mem_outputs

    # ── 工厂方法 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg):
        return cls(
            patch_sizes=         cfg.PATCH_SIZES,
            patch_main=          cfg.PATCH_MAIN,
            d_model=             cfg.D_MODEL,
            num_heads=           cfg.NUM_HEADS,
            num_layers=          cfg.NUM_LAYERS,
            top_k=               cfg.top_k,
            n_channels=          cfg.NUM_CHANNELS,
            context_len=         cfg.CONTEXT_LEN,
            forecast_len=        cfg.FORECAST_LEN,
            dropout=             cfg.P_DROPOUT,
            num_memory_slots=    cfg.NUM_MEMORY_SLOTS,
            memory_temperature=  cfg.MEMORY_TEMPERATURE,
            memory_shrink_thresh=cfg.MEMORY_SHRINK_THRESH,
        )

    @classmethod
    def from_pstg_checkpoint(cls, ckpt_path: str, cfg, device: str = "cpu"):
        """
        从 PSTG checkpoint 热启动：加载共享参数，记忆库随机初始化。
        这样可以跳过 PSTG 的预热阶段，直接对记忆库进行微调。
        """
        import torch
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_cfg = ckpt.get("config", {})

        model = cls(
            patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
            d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
            num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
            num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
            n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
            context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
            forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
            top_k=cfg.top_k,
            dropout=cfg.P_DROPOUT,
            num_memory_slots=   cfg.NUM_MEMORY_SLOTS,
            memory_temperature= cfg.MEMORY_TEMPERATURE,
        )

        # 只加载 PSTG 共有的参数（patch_embed, graph_layers, forecast_head）
        pstg_state  = ckpt["model"]
        model_state = model.state_dict()
        matched = {
            k: v for k, v in pstg_state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        model_state.update(matched)
        model.load_state_dict(model_state)

        n_total   = len(model_state)
        n_matched = len(matched)
        print(f"[热启动] 加载 {n_matched}/{n_total} 个参数层")
        print(f"  来自 epoch={ckpt.get('epoch','?')}，val_loss={ckpt.get('val_loss','?')}")

        return model.to(device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_new_parameters(self) -> int:
        """相比 PSTG 新增的参数量（仅记忆库）"""
        return sum(p.numel() for p in self.memory_bank.parameters() if p.requires_grad)
