"""
PSTG-Mamba 热启动脚本

把 PSTG 已收敛的图层参数（graph_layers.* + forecast_head.*）
迁移进 Mamba checkpoint，生成 best_hotstart.pt。

用法：
  python hotstart_mamba.py \
    --pstg_ckpt  checkpoints/best.pt \
    --mamba_ckpt checkpoints_mamba/best.pt \
    --out        checkpoints_mamba/best_hotstart.pt
"""

import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pstg_ckpt",  default="checkpoints/best.pt")
    p.add_argument("--mamba_ckpt", default="checkpoints_mamba/best.pt")
    p.add_argument("--out",        default="checkpoints_mamba/best_hotstart.pt")
    args = p.parse_args()

    print(f"PSTG  ckpt : {args.pstg_ckpt}")
    print(f"Mamba ckpt : {args.mamba_ckpt}")

    pstg_sd  = torch.load(args.pstg_ckpt,  map_location="cpu")["model"]
    mamba_ck = torch.load(args.mamba_ckpt, map_location="cpu")
    mamba_sd = mamba_ck["model"]

    # 迁移图层 + 预测头（两个模型这部分结构完全相同）
    PREFIXES = ("graph_layers.", "forecast_head.")
    transferred, skipped = [], []

    for key in list(mamba_sd.keys()):
        if not any(key.startswith(pfx) for pfx in PREFIXES):
            continue
        if key not in pstg_sd:
            skipped.append(key + "  [not in PSTG]")
            continue
        if pstg_sd[key].shape != mamba_sd[key].shape:
            skipped.append(f"{key}  [shape mismatch: {pstg_sd[key].shape} vs {mamba_sd[key].shape}]")
            continue
        mamba_sd[key] = pstg_sd[key].clone()
        transferred.append(key)

    print(f"\n迁移成功：{len(transferred)} 个参数张量")
    if skipped:
        print(f"跳过：{len(skipped)} 个")
        for s in skipped:
            print(f"  {s}")

    # 打印被迁移的模块名
    modules = {k.split(".")[0] + "." + k.split(".")[1] for k in transferred}
    print("\n迁移模块：")
    for m in sorted(modules):
        print(f"  {m}")

    # 重置 optimizer / scheduler，epoch 置 0（相当于重新训练图层以外的部分）
    mamba_ck["model"]     = mamba_sd
    mamba_ck["optimizer"] = None   # 优化器状态不兼容，重置
    mamba_ck["scheduler"] = None
    mamba_ck["epoch"]     = 0
    mamba_ck["val_loss"]  = float("inf")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mamba_ck, out_path)
    print(f"\n已保存：{out_path}")
    print("\n继续训练：")
    print(f"  python train_mamba.py --resume {out_path} --epochs 30 --save_every 5")


if __name__ == "__main__":
    main()
