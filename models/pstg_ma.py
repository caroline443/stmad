"""
PSTG-MA v2：Memory-Augmented Progressive Spatiotemporal Graph

v2 关键修复：记忆库加独立预测头，在数据空间计算误差（而非 latent space）

架构：
  H^[nL] ──→ ForecastHead(主)  ──→ x̂        （直接预测，同 PSTG）
  H^[nL] ──→ MemoryBank ──→ z_hat
                  ↓
             ForecastHead(记忆) ──→ x̂_mem    （记忆引导预测，v2 新增）

推理双信号：
  R_pred = |x_true - x̂|         [0.003, 0.48]  ← 主预测残差
  R_mem  = |x_true - x̂_mem|     [同量级]        ← 记忆引导残差（v2 大幅增强）
  score  = α·R_pred + (1-α)·R_mem

训练损失：
  L_pred（主预测）+ λ_mem·L_mem（记忆引导预测）+ λ_ent·L_ent（熵正则）
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
    Memory-Augmented PSTG（v2）

    相比 v1 的改动：
    - 新增 forecast_head_mem：以 z_hat（记忆重构）为输入，输出 x̂_mem
    - 记忆误差改为数据空间：R_mem = |x_true - x̂_mem|（与主残差同量级）
    - 训练 loss 新增 L_mem_forecast：||x_future - x̂_mem_future||
    """

    def __init__(
        self,
        # ── PSTG 原有参数（不变）──────────────────────────────────────────
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
        # ── MA 新增参数 ───────────────────────────────────────────────────
        num_memory_slots:      int   = 200,
        memory_temperature:    float = 0.1,
        memory_shrink_thresh:  float = None,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [25, 50, 125]

        self.n_channels  = n_channels
        self.n_patches   = context_len // patch_main
        self.n_nodes     = n_channels * self.n_patches
        self.num_layers  = num_layers
        self.forecast_len = forecast_len

        # ── 与 PSTG 完全相同的三个模块 ───────────────────────────────────
        self.patch_embed = MultiScalePatchEmbedding(
            patch_sizes=patch_sizes, patch_main=patch_main,
            d_model=d_model, context_len=context_len, n_channels=n_channels,
        )
        self.graph_layers = nn.ModuleList([
            SpatioTemporalGraphLayer(d_model=d_model, num_heads=num_heads,
                                     top_k=top_k, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.forecast_head = ForecastHead(         # 主预测头（同 PSTG）
            n_channels=n_channels, n_patches=self.n_patches,
            d_model=d_model, forecast_len=forecast_len,
        )

        # ── MA 新增两个模块 ───────────────────────────────────────────────
        self.memory_bank = MemoryBank(
            num_slots=num_memory_slots,
            slot_dim=d_model,
            temperature=memory_temperature,
            shrink_thresh=memory_shrink_thresh,
        )
        # 记忆引导预测头：输入 z_hat（记忆重构），输出 x̂_mem
        # 与主预测头结构完全相同，但参数独立
        self.forecast_head_mem = ForecastHead(
            n_channels=n_channels, n_patches=self.n_patches,
            d_model=d_model, forecast_len=forecast_len,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── 前向传播 ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, return_adj: bool = False):
        """
        Returns:
            x_hat      : [B, C, F]  主预测（同 PSTG）
            x_hat_mem  : [B, C, F]  记忆引导预测（v2 新增，用于数据空间误差）
            mem_outputs: dict       MemoryBank 输出（z_hat/w/mem_error/entropy）
            adj_list   : list(可选)
        """
        # 1. Patch 嵌入
        h = self.patch_embed(x)           # [B, n, D]

        # 2. 图推理
        adj_list = []
        for layer in self.graph_layers:
            h, A = layer(h)
            if return_adj:
                adj_list.append(A.detach().cpu())

        # 3. 记忆库：以 H^[nL] 寻址，得到记忆重构 z_hat
        self._last_h = h
        mem_outputs  = self.memory_bank(h)     # z_hat: [B, n, D]

        # 4. 主预测头（H^[nL] → x̂）
        x_hat = self.forecast_head(h)          # [B, C, F]

        # 5. 记忆引导预测头（z_hat → x̂_mem）  ← v2 核心修改
        x_hat_mem = self.forecast_head_mem(mem_outputs["z_hat"])  # [B, C, F]

        if return_adj:
            return x_hat, x_hat_mem, mem_outputs, adj_list
        return x_hat, x_hat_mem, mem_outputs

    # ── 工厂方法 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg):
        return cls(
            patch_sizes=cfg.PATCH_SIZES, patch_main=cfg.PATCH_MAIN,
            d_model=cfg.D_MODEL, num_heads=cfg.NUM_HEADS,
            num_layers=cfg.NUM_LAYERS, top_k=cfg.top_k,
            n_channels=cfg.NUM_CHANNELS, context_len=cfg.CONTEXT_LEN,
            forecast_len=cfg.FORECAST_LEN, dropout=cfg.P_DROPOUT,
            num_memory_slots=cfg.NUM_MEMORY_SLOTS,
            memory_temperature=cfg.MEMORY_TEMPERATURE,
            memory_shrink_thresh=cfg.MEMORY_SHRINK_THRESH,
        )

    @classmethod
    def from_pstg_checkpoint(cls, ckpt_path: str, cfg, device: str = "cpu"):
        """从 PSTG checkpoint 热启动，仅记忆库和记忆预测头随机初始化。"""
        ckpt     = torch.load(ckpt_path, map_location=device)
        ckpt_cfg = ckpt.get("config", {})

        model = cls(
            patch_sizes=  ckpt_cfg.get("patch_sizes",  cfg.PATCH_SIZES),
            d_model=      ckpt_cfg.get("d_model",       cfg.D_MODEL),
            num_heads=    ckpt_cfg.get("num_heads",     cfg.NUM_HEADS),
            num_layers=   ckpt_cfg.get("num_layers",    cfg.NUM_LAYERS),
            n_channels=   ckpt_cfg.get("n_channels",    cfg.NUM_CHANNELS),
            context_len=  ckpt_cfg.get("context_len",   cfg.CONTEXT_LEN),
            forecast_len= ckpt_cfg.get("forecast_len",  cfg.FORECAST_LEN),
            top_k=cfg.top_k, dropout=cfg.P_DROPOUT,
            num_memory_slots=cfg.NUM_MEMORY_SLOTS,
            memory_temperature=cfg.MEMORY_TEMPERATURE,
        )

        pstg_state  = ckpt["model"]
        model_state = model.state_dict()
        matched = {k: v for k, v in pstg_state.items()
                   if k in model_state and model_state[k].shape == v.shape}
        model_state.update(matched)
        model.load_state_dict(model_state)

        print(f"[热启动] 加载 {len(matched)}/{len(model_state)} 个参数层")
        print(f"  来自 epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}")
        return model.to(device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_new_parameters(self) -> int:
        """相比 PSTG 新增的参数：记忆库 + 记忆预测头"""
        new = sum(p.numel() for p in self.memory_bank.parameters())
        new += sum(p.numel() for p in self.forecast_head_mem.parameters())
        return new
