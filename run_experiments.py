"""
SpCA 论文修订：批量实验脚本
============================
顺序运行：
  1. 推理时间（SpCA vs PSTG）
  2. 消融实验（频段数、分割点、编码器变体）
  3. 多随机种子（seed 42/123/456）

结果写入 experiment_results.json

用法：
  # 全部跑（约 10-12 小时）
  python run_experiments.py --data_dir /root/autodl-tmp/data/ESA-Mission1

  # 只测推理时间
  python run_experiments.py --data_dir /root/autodl-tmp/data/ESA-Mission1 --timing_only

  # 只跑消融（跳过多种子）
  python run_experiments.py --data_dir /root/autodl-tmp/data/ESA-Mission1 --no_multiseed
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
#  消融实验配置
# ─────────────────────────────────────────────────────────────────────────────

ABLATIONS = [
    # (实验名, ckpt_dir 后缀, train_spca.py 额外参数)
    ("N_BANDS=2",               "ab_bands2",   ["--n_bands", "2", "--band_splits", "0.4"]),
    ("N_BANDS=3 (default)",     "ab_bands3",   []),
    ("N_BANDS=4",               "ab_bands4",   ["--n_bands", "4", "--band_splits", "0.1", "0.4", "0.6"]),
    ("splits=(0.05,0.2)",       "ab_split_lo", ["--band_splits", "0.05", "0.2"]),
    ("splits=(0.2,0.5)",        "ab_split_hi", ["--band_splits", "0.2",  "0.5"]),
    ("v2 temporal attention",   "ab_v2",       ["--temporal"]),
]

SEEDS = [42, 123, 456]


# ─────────────────────────────────────────────────────────────────────────────
#  推理时间
# ─────────────────────────────────────────────────────────────────────────────

def measure_timing():
    print("\n" + "="*55 + "\n  推理时间测量\n" + "="*55)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B = 70
    results = {}

    for name, get_model in [
        ("SpCA", lambda: _get_spca()),
        ("PSTG", lambda: _get_pstg()),
    ]:
        try:
            model = get_model()
            model = model.to(device).eval()
            cfg   = _get_cfg(name)
            x     = torch.rand(B, cfg.NUM_CHANNELS, cfg.CONTEXT_LEN).to(device)

            with torch.no_grad():
                for _ in range(20): model(x)   # 预热

            if device == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(300): model(x)
            if device == "cuda": torch.cuda.synchronize()

            ms = (time.perf_counter() - t0) / 300 * 1000
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            results[name] = {
                "ms_per_batch": round(ms, 3),
                "batch_size":   B,
                "params":       n_params,
            }
            print(f"  {name:<6}: {ms:.3f} ms/batch  |  {n_params:,} 参数")
        except Exception as e:
            print(f"  {name} 计时失败: {e}")

    return results


def _get_spca():
    from config_spca import ConfigSpCA
    from models.spca import SpCA
    return SpCA.from_config(ConfigSpCA())

def _get_pstg():
    from config import Config
    from models.pstg import PSTG
    return PSTG.from_config(Config())

def _get_cfg(name):
    if name == "SpCA":
        from config_spca import ConfigSpCA; return ConfigSpCA()
    from config import Config; return Config()


# ─────────────────────────────────────────────────────────────────────────────
#  训练 + 评估一个配置
# ─────────────────────────────────────────────────────────────────────────────

def run_one(name, ckpt_suffix, extra_train_args, data_dir, epochs, output_dir):
    """
    训练 → 评估 → 从 JSON 读结果。
    返回指标字典，失败时返回 None。
    """
    ckpt_dir    = f"checkpoints_{ckpt_suffix}"
    eval_output = f"outputs_{ckpt_suffix}"

    print(f"\n{'─'*55}")
    print(f"  实验: {name}")
    print(f"  ckpt → {ckpt_dir}")
    print(f"{'─'*55}")

    # ── 训练 ──────────────────────────────────────────────────────────────
    train_cmd = [
        sys.executable, "train_spca.py",
        "--data_dir",  data_dir,
        "--epochs",    str(epochs),
        "--ckpt_dir",  ckpt_dir,
        "--save_every", "70",   # 只保存最后一轮，节省磁盘
    ] + extra_train_args

    t0 = time.time()
    proc = subprocess.run(train_cmd, capture_output=False)   # 实时输出
    train_min = (time.time() - t0) / 60

    if proc.returncode != 0:
        print(f"  ❌ 训练失败 (returncode={proc.returncode})")
        return None

    print(f"  ✓ 训练完成 ({train_min:.1f} 分钟)")

    # ── 评估 ──────────────────────────────────────────────────────────────
    eval_cmd = [
        sys.executable, "evaluate_spca.py",
        "--ckpt",     f"{ckpt_dir}/best.pt",
        "--data_dir", data_dir,
        "--output",   eval_output,
        "--no_plot",
    ]
    proc = subprocess.run(eval_cmd, capture_output=False)

    if proc.returncode != 0:
        print(f"  ❌ 评估失败")
        return None

    # ── 读取 JSON 结果 ─────────────────────────────────────────────────────
    # evaluate_spca.py 保存到 outputs_xxx/eval_xxx/evaluation_results.json
    eval_dir = Path(eval_output)
    results_files = sorted(eval_dir.glob("eval_*/evaluation_results.json"))
    if not results_files:
        print(f"  ❌ 找不到评估结果 JSON")
        return None

    metrics = json.loads(results_files[-1].read_text())  # 最新一次

    # 提取标准1和标准2指标
    ev1  = metrics.get("event_wise",       {})
    af1  = metrics.get("affiliation",      {})
    ev2  = metrics.get("event_wise_filt",  {})
    af2  = metrics.get("affiliation_filt", {})

    summary = {
        "name":          name,
        "ckpt_dir":      ckpt_dir,
        "train_min":     round(train_min, 1),
        # 标准1（33事件）
        "std1_ev_p":     ev1.get("precision"),
        "std1_ev_r":     ev1.get("recall"),
        "std1_ev_f05":   ev1.get("f0.5"),
        "std1_af_f05":   af1.get("f0.5"),
        # 标准2（24事件，duration≥2）
        "std2_ev_p":     ev2.get("precision"),
        "std2_ev_r":     ev2.get("recall"),
        "std2_ev_f05":   ev2.get("f0.5"),
        "std2_af_f05":   af2.get("f0.5"),
    }

    print(f"  标准1 Event F0.5={summary['std1_ev_f05']}  Affil F0.5={summary['std1_af_f05']}")
    print(f"  标准2 Event F0.5={summary['std2_ev_f05']}  Affil F0.5={summary['std2_af_f05']}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  汇总打印
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results):
    print("\n\n" + "="*70)
    print("  实验结果汇总")
    print("="*70)

    if results.get("timing"):
        print("\n▶ 推理时间")
        for m, t in results["timing"].items():
            print(f"  {m:<6}: {t['ms_per_batch']} ms/batch  |  {t['params']:,} 参数")

    if results.get("ablation"):
        print(f"\n▶ 消融实验（{'标准1 Event F0.5':>20}  {'Affil F0.5':>10}）")
        print(f"  {'实验名':<35} {'Std1 Ev.F05':>12} {'Std1 Af.F05':>12}")
        print(f"  {'─'*60}")
        for m in results["ablation"]:
            if m:
                print(f"  {m['name']:<35} "
                      f"{str(m.get('std1_ev_f05','?')):>12} "
                      f"{str(m.get('std1_af_f05','?')):>12}")

    if results.get("multiseed"):
        f05s = [m["std1_ev_f05"] for m in results["multiseed"] if m and m.get("std1_ev_f05")]
        if f05s:
            print(f"\n▶ 多随机种子 Event F0.5 (标准1)：{f05s}")
            print(f"  均值 ± 标准差：{np.mean(f05s):.4f} ± {np.std(f05s):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--epochs",       type=int, default=70)
    p.add_argument("--timing_only",  action="store_true")
    p.add_argument("--no_ablation",  action="store_true")
    p.add_argument("--no_multiseed", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    all_results = {
        "timestamp":  datetime.now().isoformat(),
        "data_dir":   args.data_dir,
        "epochs":     args.epochs,
        "timing":     {},
        "ablation":   [],
        "multiseed":  [],
    }
    out_path = Path("experiment_results.json")

    def save():
        out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))

    # ── 1. 推理时间 ────────────────────────────────────────────────────────
    all_results["timing"] = measure_timing()
    save()

    if args.timing_only:
        print_summary(all_results)
        return

    # ── 2. 消融实验 ────────────────────────────────────────────────────────
    if not args.no_ablation:
        print("\n\n" + "="*55)
        print(f"  消融实验（{len(ABLATIONS)} 个变体，各 {args.epochs} 轮）")
        print(f"  预计时间：{len(ABLATIONS)*args.epochs*38//3600} 小时")
        print("="*55)

        for name, suffix, extra in ABLATIONS:
            m = run_one(name, suffix, extra, args.data_dir, args.epochs,
                        output_dir=f"outputs_{suffix}")
            all_results["ablation"].append(m)
            save()

    # ── 3. 多随机种子 ──────────────────────────────────────────────────────
    if not args.no_multiseed:
        print("\n\n" + "="*55)
        print(f"  多随机种子（seeds={SEEDS}，各 {args.epochs} 轮）")
        print("="*55)

        for seed in SEEDS:
            m = run_one(
                name=f"default seed={seed}",
                ckpt_suffix=f"seed{seed}",
                extra_train_args=["--seed", str(seed)],
                data_dir=args.data_dir,
                epochs=args.epochs,
                output_dir=f"outputs_seed{seed}",
            )
            all_results["multiseed"].append(m)
            save()

    # ── 汇总 ──────────────────────────────────────────────────────────────
    print_summary(all_results)
    print(f"\n完整结果：{out_path.absolute()}")


if __name__ == "__main__":
    main()
