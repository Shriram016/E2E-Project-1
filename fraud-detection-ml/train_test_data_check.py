import pandas as pd
import numpy as np


train_data = pd.read_parquet(r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\parquet\train_merged.parquet")
test_data = pd.read_parquet(r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\parquet\test_merged.parquet")


print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

# Check for overlapping columns
overlap_columns = set(train_data.columns).intersection(set(test_data.columns))
print("Overlapping Columns:", len(overlap_columns))

# check if any column is present only in train but not in test
train_only_columns = set(train_data.columns) - set(test_data.columns)
test_only_columns = set(test_data.columns) - set(train_data.columns)

print("Columns present only in train but not in test:", len(train_only_columns))
print("Columns present only in test but not in train:", len(test_only_columns))