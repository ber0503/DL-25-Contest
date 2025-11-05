# plot_from_txt.py
# 运行：python plot_from_txt.py
# 文件需放在同目录：results-lr-wd.txt
import json, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_txt(p: Path) -> pd.DataFrame:
    txt = p.read_text(encoding="utf-8").strip()
    df = pd.DataFrame()
    # JSON array / object
    try:
        x = json.loads(txt)
        if isinstance(x, dict): x = [x]
        if isinstance(x, list): df = pd.DataFrame(x)
    except Exception:
        pass
    # JSON Lines
    if df.empty:
        rows = []
        ok = True
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: ok = False; break
        if ok and rows: df = pd.DataFrame(rows)
    # CSV/TSV
    if df.empty:
        df = pd.read_csv(p, sep=None, engine="python")
    return df

def main():
    p = Path("results-lr-wd.txt")
    if not p.exists():
        raise SystemExit("请把数据保存为 results-lr-wd.txt 并与脚本同目录。")

    df = parse_txt(p)

    # 别名容错
    rename = {}
    if "lr" in df.columns and "learning_rate" not in df.columns: rename["lr"] = "learning_rate"
    if "wd" in df.columns and "weight_decay" not in df.columns: rename["wd"] = "weight_decay"
    if "acc" in df.columns and "accuracy" not in df.columns: rename["acc"] = "accuracy"
    if rename: df = df.rename(columns=rename)

    need = ["weight_decay", "learning_rate", "accuracy"]
    for c in need:
        if c not in df.columns:
            raise SystemExit(f"缺少列：{c}。需要包含 learning_rate, weight_decay, accuracy（或别名 lr/wd/acc）")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need).sort_values(["weight_decay", "learning_rate"])

    # 透视表（重复组合取均值）
    pivot = (
        df.groupby(["weight_decay", "learning_rate"], as_index=False)["accuracy"]
          .mean()
          .pivot(index="weight_decay", columns="learning_rate", values="accuracy")
          .sort_index(axis=0).sort_index(axis=1)
    )

    # 画热力图（matplotlib-only）
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns.to_numpy()])
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index.to_numpy()])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Weight Decay")
    ax.set_title("Accuracy vs Weight Decay & Learning Rate")

    # 数值标注
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    fig.tight_layout()
    fig.savefig("accuracy_lr_wd_heatmap.png", dpi=180, bbox_inches="tight")
    print("✅ 已保存：accuracy_lr_wd_heatmap.png")

if __name__ == "__main__":
    main()
