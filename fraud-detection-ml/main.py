import os
import pandas as pd
from src.pipelines.main_pipeline import tree_pipeline , normal_pipeline
from src.pipelines.load_merge import load_and_merge

    
train_sources = {
    "transaction": r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\raw\train_transaction.csv",
    "identity": r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\raw\train_identity.csv"
}

test_sources = {
    "transaction": r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\raw\test_transaction.csv",
    "identity": r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\raw\test_identity.csv"
}
    
train_parquet_path = r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\parquet\train_merged.parquet"
test_parquet_path =  r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\parquet\test_merged.parquet"

if not os.path.exists(train_parquet_path):
    print('Train parquet file not found. Merging and creating parquet file...')
    train_df = load_and_merge(train_sources, join_key="TransactionID")
    if train_df is None:
        raise ValueError("Merged train DataFrame is None.")
    print(train_df.shape)
    train_df.to_parquet(r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\parquet\train_merged.parquet")

else:
    print('Train parquet file found. Skipping merge.')
    train_df = pd.read_parquet(train_parquet_path)

if not os.path.exists(test_parquet_path):
    print('Test parquet file not found. Merging and creating parquet file...')
    test_df = load_and_merge(test_sources, join_key="TransactionID")
    if test_df is None:
        raise ValueError("Merged test DataFrame is None.")
    print(test_df.shape)
    test_df.to_parquet(r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\parquet\test_merged.parquet")
                        
else:
    print('Test parquet file found. Skipping merge.')
    test_df = pd.read_parquet(test_parquet_path)



print('Calling main_pipeline.fit_transform()...')

train_ready = tree_pipeline.fit_transform(train_df)

print('Pipeline processing complete.')

print('Calling normal_pipeline.fit_transform()...')
train_ready_normal = normal_pipeline.fit_transform(train_df)
print('Normal Pipeline processing complete.')