#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot_ls_wr.py
#
# 用法示例：
#   python plot_ls_wr.py
#   python plot_ls_wr.py --file results-ls-wr.txt --out accuracy_ls_wr_heatmap.png
#   python plot_ls_wr.py --metric accuracy
#
# 支持的输入格式：
# - JSON 数组（或单个对象）
# - JSON Lines（每行一个 JSON）
# - CSV/TSV（自动分隔符检测）
#
# 期望列名：
# - lr_scheduler_type（别名：ls）
# - warmup_ratio（别名：wr）
# - accuracy（别名：acc）  ← 可通过 --metric 指定其它度量

import json
import math
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_file(path: Path) -> pd.DataFrame:
    """尝试按 JSON 数组/对象 → JSON Lines → CSV/TSV 的顺序解析。"""
    if not path.exists():
        raise SystemExit(f"未找到文件：{path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"文件为空：{path}")

    # 1) JSON 数组 / 单对象
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = [obj]
        if isinstance(obj, list):
            return pd.DataFrame(obj)
    except Exception:
        pass

    # 2) JSON Lines
    rows = []
    jl_ok = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            jl_ok = False
            break
    if jl_ok and rows:
        return pd.DataFrame(rows)

    # 3) CSV/TSV（自动分隔符检测）
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise SystemExit(f"无法解析文件：{path}\n错误：{e}")


def coerce_and_alias(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """列名别名与类型清洗。"""
    # 别名容错
    rename = {}
    if "ls" in df.columns and "lr_scheduler_type" not in df.columns:
        rename["ls"] = "lr_scheduler_type"
    if "wr" in df.columns and "warmup_ratio" not in df.columns:
        rename["wr"] = "warmup_ratio"
    if "acc" in df.columns and metric not in df.columns:
        rename["acc"] = metric
    if rename:
        df = df.rename(columns=rename)

    required = ["lr_scheduler_type", "warmup_ratio", metric]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"缺少必要列：{missing}\n"
            f"需要包含：lr_scheduler_type（或 ls）、warmup_ratio（或 wr）、{metric}（或 acc）"
        )

    # 数值列转为数值
    df["warmup_ratio"] = pd.to_numeric(df["warmup_ratio"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # 清洗空值
    df = df.dropna(subset=["lr_scheduler_type", "warmup_ratio", metric]).copy()

    # 标准化调度器字符串（去空白）
    df["lr_scheduler_type"] = df["lr_scheduler_type"].astype(str).str.strip()

    return df


def make_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(ls, wr) → metric 的透视表，重复取均值。"""
    # warmup_ratio 排序，lr_scheduler_type 按字母序
    df = df.copy()
    # groupby 平均
    g = (
        df.groupby(["lr_scheduler_type", "warmup_ratio"], as_index=False)[metric]
          .mean()
    )
    # 透视：行=warmup_ratio，列=lr_scheduler_type
    pivot = g.pivot(index="warmup_ratio",
                    columns="lr_scheduler_type",
                    values=metric)

    # 排序：行（数值升序），列（字母顺序）
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    return pivot


def plot_heatmap(pivot: pd.DataFrame, metric: str, out_path: Path):
    """仅用 matplotlib 画热力图并保存。"""
    if pivot.empty:
        raise SystemExit("透视表为空，可能数据里没有有效的 (lr_scheduler_type, warmup_ratio) 组合。")

    fig, ax = plt.subplots(figsize=(9, 6))
    data = pivot.values
    im = ax.imshow(data, aspect="auto")

    # 坐标轴刻度与标签
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index.to_numpy()])

    ax.set_xlabel("lr_scheduler_type")
    ax.set_ylabel("warmup_ratio")
    ax.set_title(f"{metric} vs lr_scheduler_type & warmup_ratio")

    # 数值标注
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"✅ 已保存热力图：{out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot heatmap: lr_scheduler_type × warmup_ratio → metric")
    parser.add_argument("--file", default="results-ls-wr.txt", help="输入文件（默认：results-ls-wr.txt）")
    parser.add_argument("--out", default="accuracy_ls_wr_heatmap.png", help="输出图片路径（默认：accuracy_ls_wr_heatmap.png）")
    parser.add_argument("--metric", default="accuracy", help="度量列名（默认：accuracy；也可指定 f1 等）")
    args = parser.parse_args()

    in_path = Path(args.file)
    out_path = Path(args.out)
    metric = args.metric

    df = parse_file(in_path)
    df = coerce_and_alias(df, metric)
    pivot = make_pivot(df, metric)
    plot_heatmap(pivot, metric, out_path)


if __name__ == "__main__":
    main()
