# voting the result from the 5 submissions

import pandas as pd
from functools import reduce

# Read 5 CSV files
dfs = [pd.read_csv(f"submission{i}.csv") for i in range(1, 6)]

# Merge all files on "ID"
merged = reduce(lambda left, right: pd.merge(left, right, on="ID", suffixes=("", "_dup")), dfs)
# Extract all "is_correct" columns
is_correct_cols = [col for col in merged.columns if "is_correct" in col]
# Convert True/False to 1/0 and count votes
merged["votes"] = merged[is_correct_cols].sum(axis=1)
# Majority vote (more than half are True)
merged["is_correct_final"] = merged["votes"] >= (len(is_correct_cols) / 2)
# Create the final result
result = merged[["ID", "is_correct_final"]].rename(columns={"is_correct_final": "is_correct"})

# Save as CSV
result.to_csv("submission_majority.csv", index=False)

print("✅ File generated: submission_majority.csv")