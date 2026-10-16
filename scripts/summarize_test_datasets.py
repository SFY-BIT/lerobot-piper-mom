#!/usr/bin/env python3
"""test 目录数据集统一 summary 统计。

通用 (所有数据集):
  - 任务成功率 = success / (success + fail)  来自 meta/results.jsonl
    (success/fail/none 计数单独列出; 无标注的轨迹算 none)
  - 每个数据集的结果同时写入自身 meta/result_summary.json

带 force 数据集 (有 observation.force_pred 列 且 存在非零预测帧) 额外:
  只统计推理帧 (force_pred 非零; 遥控/返回段 pred=0 是补零, 不算预测):
  说明: force_pred = 模型预测的力绝对值 (与 observation.force 真实绝对值同量级,
        实测 -12.7~+14N vs 真实 -11~+13N; 非 delta 变化量)
  指标 (逐帧逐轴, 分母=真值, 均用绝对值):
    1. delta 形式均值:   |Δpred - Δtrue| / |Δtrue| * 100   (Δx[t]=x[t]-x[t-1] 相邻帧变化量)
    2. 绝对值形式均值:  ||pred| - |true|| / |true| * 100
    3. max_delta_N:      max(|pred - true|)  每帧预测与真实的最大差值 (N/Nm)
  仅当分母 |·| > eps (默认 0.1 N/Nm) 的样本计入 (防除零); 内部按 P99 分位过滤
  单点爆炸后取均值 (过滤参数不输出)

结果写入 test/SUMMARY.json + 每个数据集 meta/result_summary.json, 并打印表格。

用法:
  python scripts/summarize_test_datasets.py [--root <dir>] [--eps 0.1]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(r"C:\Users\User\.cache\huggingface\lerobot\test")
FORCE_DIM = 6


def load_results(repo_dir: Path) -> dict:
    """读 results.jsonl → {ep: result}"""
    p = repo_dir / "meta" / "results.jsonl"
    out = {}
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["episode_index"]] = r.get("result")
    return out


def force_metrics(repo_dir: Path, eps: float = 0.1, clip_pct: float = 99.0):
    """统计 force_pred vs 真实 force (只统计推理帧 = pred 非零)。

    force_pred 语义 = 绝对力预测 (与 observation.force 同量级, 非 delta)。
    指标:
      - max_delta_N       = max(|pred - true|) 每帧最大差值 (N/Nm)
      - delta_pct_mean%   = mean(|Δpred-Δtrue|/|Δtrue|*100)  delta 形式 (Δx[t]=x[t]-x[t-1])
      - abs_pct_mean%     = mean(||pred|-|true||/|true|*100)  绝对值形式
    均只统计推理帧; 分母 |·|>eps 才计入; 按 clip_pct 分位过滤单点爆炸后取均值。
    """
    max_delta_N = 0.0
    delta_pcts, abs_pcts = [], []
    for p in sorted((repo_dir / "data").glob("chunk-*/episode_*.parquet")):
        df = pd.read_parquet(p)
        if "observation.force_pred" not in df.columns:
            continue
        pred = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:FORCE_DIM]
                         for x in df["observation.force_pred"]])
        true = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:FORCE_DIM]
                         for x in df["observation.force"]])
        mask = np.abs(pred).sum(axis=1) > 1e-6      # 推理帧 (pred 非零)
        if not mask.any():
            continue
        pm, tm = pred[mask], true[mask]
        max_delta_N = max(max_delta_N, float(np.abs(pm - tm).max()))
        # delta 形式: 相邻帧变化量 (第2帧起)
        if len(pm) > 1:
            dp = pm[1:] - pm[:-1]
            dt = tm[1:] - tm[:-1]
            v = np.abs(dt) > eps
            if v.any():
                delta_pcts.append(np.abs(dp - dt)[v] / np.abs(dt)[v] * 100.0)
        # 绝对值形式: 先各自取绝对值再比
        v = np.abs(tm) > eps
        if v.any():
            abs_pcts.append(np.abs(np.abs(pm) - np.abs(tm))[v] / np.abs(tm)[v] * 100.0)
    if not delta_pcts and not abs_pcts:
        return None

    def _clip_mean(arrs):
        arr = np.concatenate(arrs)
        if clip_pct < 100.0:
            arr = arr[arr <= np.percentile(arr, clip_pct)]   # 过滤单点爆炸
        return float(arr.mean())

    return {
        "max_delta_N": round(max_delta_N, 3),   # 每帧预测与真实最大差值 (N/Nm)
        "delta_pct_mean%": round(_clip_mean(delta_pcts), 2) if delta_pcts else None,
        "abs_pct_mean%": round(_clip_mean(abs_pcts), 2) if abs_pcts else None,
    }


FORCE_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def plot_force_curves(repo_dir: Path, out_dir: Path, smooth_win: int = 15,
                      show_pred: bool = True) -> int:
    """每个 episode 画一张 force(真实) vs force_pred(预测) 6轴平滑曲线图, 保存 PNG。

    曲线先做滑动平均平滑 (窗口 smooth_win 帧, 0=不平滑), 不保留原始线。
    show_pred=False 时只画真实力 (PI05 不带力模型 force_pred 全 0, 预测线无意义)。
    推理段 (pred 非零) 背景浅绿高亮; 输出 <out_dir>/ep{index:06d}.png。
    返回画出的图数量。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 优先用系统中文字体 (Windows: 微软雅黑/黑体), 找不到则退回默认 (中文会变方块)
    for _f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
        if any(_f.lower() in (x.name or "").lower() for x in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    def _smooth(arr):
        """滑动平均 (沿时间维逐轴卷积, mode=same), 窗口<=1 返回原样。
        ★ axis 必须=0 (对每列=每个力轴的时间序列卷积), 若用 axis=1 会把
        每一帧的 6 维向量当时间序列卷积, 输出变 (n,15) 且量级被稀释~10倍。"""
        if smooth_win <= 1:
            return arr
        k = np.ones(smooth_win) / smooth_win
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"),
                                   axis=0, arr=arr)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_plots = 0
    for p in sorted((repo_dir / "data").glob("chunk-*/episode_*.parquet")):
        df = pd.read_parquet(p)
        if "observation.force" not in df.columns and "observation.force_pred" not in df.columns:
            continue
        ep = int(df["episode_index"].iloc[0]) if "episode_index" in df.columns else 0
        n = len(df)
        true = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:FORCE_DIM]
                         for x in df["observation.force"]])
        # force_pred 列可选 (control_robot 普通录制无预测): 有则画预测线+推理段高亮
        pred = None
        infer = None
        if "observation.force_pred" in df.columns:
            pred = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:FORCE_DIM]
                             for x in df["observation.force_pred"]])
            infer = np.abs(pred).sum(axis=1) > 1e-6      # 推理帧 (pred 非零)
            pred = _smooth(pred)
        # 平滑 (滑动平均), 不保留原始线
        true = _smooth(true)
        x = np.arange(n)
        units = ["N"] * 3 + ["N·m"] * 3        # 前3轴力(N), 后3轴扭矩(N·m)
        fig, axes = plt.subplots(3, 2, figsize=(13, 9))
        for i, ax in enumerate(axes.flat):
            ax.plot(x, true[:, i], color="#1f77b4", lw=1.2, label="force (真实)")
            if show_pred and pred is not None:
                ax.plot(x, pred[:, i], color="#ff7f0e", lw=1.0, ls="--", label="force_pred (预测)")
            if infer is not None and infer.any():
                idx = np.where(infer)[0]
                ax.axvspan(idx[0], idx[-1], color="#2ca02c", alpha=0.08)
            ax.set_title(f"{FORCE_NAMES[i]} ({units[i]})")
            ax.set_ylabel(units[i], fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"{repo_dir.parent.name}/{repo_dir.name} — episode {ep} ({n} 帧)",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = out_dir / f"ep{ep:06d}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        n_plots += 1
    return n_plots


def main():
    p = argparse.ArgumentParser(description="test 目录数据集统一 summary (成功率 + force 差异)")
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="数据集根目录 (默认 test)")
    p.add_argument("--exclude", default="",
                   help="要跳过的数据集名 (逗号分隔, 如: Pump_bottle_eval_dataset,xxx)")
    p.add_argument("--only", default="",
                   help="只处理指定的数据集名 (逗号分隔); 空=全部")
    p.add_argument("--eps", type=float, default=0.1,
                   help="百分比统计的分母真值阈值 (|true|>eps 才计入, 默认 0.1 N/Nm)")
    p.add_argument("--clip-pct", type=float, default=99.0,
                   help="过滤单点爆炸的分位数 (默认 99.0: 去掉最大的1%%后统计 mean/max; "
                        "100=不过滤)")
    p.add_argument("--out", default="", help="输出 summary json 路径 (默认 <root>/SUMMARY.json)")
    p.add_argument("--plot-dir", default="",
                   help="force 曲线输出目录 (默认每个数据集自身 force_curves/)")
    p.add_argument("--no-plot", action="store_true",
                   help="不画 force_pred vs force 曲线图")
    p.add_argument("--smooth", type=int, default=15,
                   help="曲线平滑窗口 (滑动平均帧数, 默认 15; 0=不平滑)")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"目录不存在: {root}")
        return
    out_p = Path(args.out) if args.out else root / "SUMMARY.json"

    summary = {"root": str(root), "eps": args.eps, "exclude": args.exclude, "only": args.only,
               "datasets": {}}
    exclude = set(x.strip() for x in args.exclude.split(",") if x.strip())
    only = set(x.strip() for x in args.only.split(",") if x.strip())
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in exclude:
            continue
        if only and d.name not in only:
            continue                            # 跳过被排除的数据集
        info_p = d / "meta" / "info.json"
        if not info_p.exists():
            continue
        info = json.load(open(info_p, encoding="utf-8"))
        feats = list(info.get("features", {}).keys())
        results = load_results(d)
        label_counts = Counter(r for r in results.values()) if results else Counter()
        n_succ = label_counts.get("success", 0)
        n_fail = label_counts.get("fail", 0)
        n_none = label_counts.get("none", 0)
        success_rate = (n_succ / (n_succ + n_fail) * 100) if (n_succ + n_fail) > 0 else None

        entry = {
            "repo": f"{root.name}/{d.name}",
            "episodes": info.get("total_episodes", 0),
            "frames": info.get("total_frames", 0),
            "labels": {"success": n_succ, "fail": n_fail, "none": n_none},
            "success_rate_pct": round(success_rate, 1) if success_rate is not None else None,
            "has_force_pred_col": "observation.force_pred" in feats,
            "has_force_col": "observation.force" in feats,
        }
        # 带 force 数据集: 有列且有非零预测帧 → 额外统计 + 画每条轨迹曲线
        if entry["has_force_pred_col"]:
            fm = force_metrics(d, args.eps, args.clip_pct)
            if fm is not None:
                entry["force_metrics"] = fm
                if not args.no_plot:
                    plot_dir = Path(args.plot_dir) if args.plot_dir else d / "force_curves"
                    entry["force_curve_plots"] = plot_force_curves(d, plot_dir, args.smooth)
            else:
                entry["force_metrics"] = None      # 列在但预测全零 (PI05 不带力模型)
                if not args.no_plot:               # PI05 也出图, 只画真实力 (无预测线)
                    plot_dir = Path(args.plot_dir) if args.plot_dir else d / "force_curves"
                    entry["force_curve_plots"] = plot_force_curves(d, plot_dir, args.smooth,
                                                                   show_pred=False)
        elif entry["has_force_col"]:
            # 有力但无 force_pred (control_robot 普通录制): 只画真实力曲线
            entry["force_metrics"] = None
            if not args.no_plot:
                plot_dir = Path(args.plot_dir) if args.plot_dir else d / "force_curves"
                entry["force_curve_plots"] = plot_force_curves(d, plot_dir, args.smooth,
                                                               show_pred=False)
        summary["datasets"][d.name] = entry
        rows.append((d.name, entry))
        # 每个数据集自身 meta 也写入 (result_summary.json)
        meta_out = d / "meta" / "result_summary.json"
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"已写入: {out_p}\n")

    # ── 表格 ──
    hdr = (f"{'数据集':<32}{'ep':>4}{'succ':>5}{'fail':>5}{'none':>5}{'成功率%':>8}"
           f"  | {'maxΔ(N)':>9}{'delta%均值':>11}{'abs%均值':>10}{'曲线':>5}")
    print(hdr)
    print("-" * len(hdr))
    for name, e in rows:
        fm = e.get("force_metrics")
        if fm:
            fcol = (f"{str(fm['max_delta_N']):>9}"
                    f"{str(fm['delta_pct_mean%']):>11}"
                    f"{str(fm['abs_pct_mean%']):>10}"
                    f"{str(e.get('force_curve_plots', '-')):>5}")
        else:
            fcol = (f"{'-':>9}{'-':>11}{'-':>10}"
                    f"{str(e.get('force_curve_plots', '-')):>5}")
        print(f"{name:<32}{e['episodes']:>4}{e['labels']['success']:>5}"
              f"{e['labels']['fail']:>5}{e['labels']['none']:>5}"
              f"{str(e['success_rate_pct']):>8}  | {fcol}")
    print()
    print("maxΔ(N): 每帧|pred-true|最大差值 | delta%均值: |Δpred-Δtrue|/|Δtrue|*100 "
          "(Δ=相邻帧力变化量) | abs%均值: ||pred|-|true||/|true|*100 | 曲线: 轨迹曲线图数量")
    print(f"force_pred 为绝对力预测; 只统计推理帧 (pred 非零); 分母 |·|>{args.eps:.3g} 才计入")
    print("曲线图: <数据集>/force_curves/ep{index}.png (蓝=真实, 橙虚线=预测, 绿底=推理段, "
          f"滑动平均平滑窗口={args.smooth}帧, 不保留原始线; PI05 无预测只画真实力)")
    print("每个数据集结果已写入各自 meta/result_summary.json")


if __name__ == "__main__":
    main()
