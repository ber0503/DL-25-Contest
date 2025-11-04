import pandas as pd
from functools import reduce

# 读取 5 个 CSV 文件
dfs = [pd.read_csv(f"submission{i}.csv") for i in range(1, 6)]

# 合并所有表（按 ID 对齐）
merged = reduce(lambda left, right: pd.merge(left, right, on="ID", suffixes=("", "_dup")), dfs)

# 提取所有 is_correct 列
is_correct_cols = [col for col in merged.columns if "is_correct" in col]

# 将 True/False 转为 1/0
merged["votes"] = merged[is_correct_cols].sum(axis=1)

# 多数投票（超过一半为 True）
merged["is_correct_final"] = merged["votes"] >= (len(is_correct_cols) / 2)

# 生成新的表格
result = merged[["ID", "is_correct_final"]].rename(columns={"is_correct_final": "is_correct"})

# 保存为 CSV
result.to_csv("submission_majority.csv", index=False)

print("✅ 已生成文件：submission_majority.csv")
