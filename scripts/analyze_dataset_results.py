#!/usr/bin/env python3
"""整理各数据集成功率 (来自 results.jsonl) + 力预测偏差指标。

- 成功率 = success / (success + fail)  (null/无标注 单列)
- 带 force_pred 的数据集额外计算:
    force_pred 最大偏差   = max(|pred - true|)  (6轴逐帧)
    force_pred 平均偏差   = mean(|pred - true|)
    force_pred 平均偏差%  = mean(|pred - true|) / mean(|true|) * 100
  (pred = observation.force_pred, true = observation.force, 只在推理帧有 pred 非零; 遥控/返回段 pred=0 会拉低, 因此只统计 pred 非零帧)
- 结果写入各自数据集 meta/result_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = [
    Path(r"C:\Users\User\.cache\huggingface\lerobot\my_robot"),
    Path(r"C:\Users\User\.cache\huggingface\lerobot\fvt"),
]


def load_results(repo_dir):
    """读 results.jsonl → {ep: result} (result 可能为 None/\"success\"/\"fail\")"""
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


def force_pred_metrics(repo_dir):
    """遍历 parquet, 统计 force_pred vs force 偏差 (只看 force_pred 非零帧 = 推理帧)"""
    max_dev = 0.0
    devs = []            # 逐帧平均偏差 |pred-true| (6轴平均)
    true_mags = []       # 逐帧 |true| (6轴平均), 用于百分比
    n_pred_frames = 0
    for p in sorted((repo_dir / "data").glob("chunk-*/episode_*.parquet")):
        df = pd.read_parquet(p)
        if "observation.force_pred" not in df.columns:
            continue
        pred = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:6] for x in df["observation.force_pred"]])
        true = np.stack([np.asarray(x, dtype=np.float32).reshape(-1)[:6] for x in df["observation.force"]])
        mask = (np.abs(pred).sum(axis=1) > 1e-6)   # 推理帧 (pred 非零)
        if not mask.any():
            continue
        pred_m, true_m = pred[mask], true[mask]
        d = np.abs(pred_m - true_m).mean(axis=1)    # 每帧 6轴平均偏差
        t = np.abs(true_m).mean(axis=1)
        devs.append(d)
        true_mags.append(t)
        n_pred_frames += mask.sum()
        if d.max() > max_dev:
            max_dev = d.max()
    if not devs:
        return None
    devs = np.concatenate(devs)
    true_mags = np.concatenate(true_mags)
    return {
        "n_pred_frames": int(n_pred_frames),
        "pred_max_dev_N": round(float(max_dev), 3),
        "pred_mean_dev_N": round(float(devs.mean()), 3),
        "pred_mean_dev_pct": round(float(devs.mean() / (true_mags.mean() + 1e-6) * 100), 2),
    }


def main():
    summary_all = {}
    for root in ROOTS:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            info_p = d / "meta" / "info.json"
            if not info_p.exists():
                continue
            info = json.load(open(info_p, encoding="utf-8"))
            feats = list(info.get("features", {}).keys())
            results = load_results(d)
            if not results:
                # 无标注 → 跳过成功率, 但可标记
                label_counts = {}
            else:
                label_counts = {"success": 0, "fail": 0, "none": 0}
                for r in results.values():
                    label_counts[r if r in label_counts else "none"] += 1

            n_ep = info.get("total_episodes", 0)
            n_succ = label_counts.get("success", 0)
            n_fail = label_counts.get("fail", 0)
            n_none = label_counts.get("none", 0)
            success_rate = (n_succ / (n_succ + n_fail) * 100) if (n_succ + n_fail) > 0 else None

            entry = {
                "repo": f"{d.parent.name}/{d.name}",
                "episodes": n_ep,
                "frames": info.get("total_frames", 0),
                "labels": label_counts,
                "success_rate_pct": round(success_rate, 1) if success_rate is not None else None,
                "has_force_pred": "observation.force_pred" in feats,
            }
            if "observation.force_pred" in feats:
                fp = force_pred_metrics(d)
                if fp:
                    entry["force_pred_metrics"] = fp
            # 写回各自数据集
            out_p = d / "meta" / "result_summary.json"
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            summary_all[entry["repo"]] = entry

    # 打印汇总
    print(f"{'数据集':<42}{'ep':>4}{'succ':>5}{'fail':>5}{'none':>5}{'成功率%':>8}{'力预测最大偏差':>12}{'平均偏差':>10}{'偏差%':>8}")
    for repo, e in summary_all.items():
        fp = e.get("force_pred_metrics", {})
        print(f"{repo:<42}{e['episodes']:>4}{e['labels'].get('success',0):>5}"
              f"{e['labels'].get('fail',0):>5}{e['labels'].get('none',0):>5}"
              f"{str(e['success_rate_pct']):>8}"
              f"{str(fp.get('pred_max_dev_N','-')):>12}"
              f"{str(fp.get('pred_mean_dev_N','-')):>10}"
              f"{str(fp.get('pred_mean_dev_pct','-')):>8}")


if __name__ == "__main__":
    main()
