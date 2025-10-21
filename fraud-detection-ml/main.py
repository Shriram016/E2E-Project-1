import os
import pandas as pd
from src.pipelines.main_pipeline import tree_pipeline , normal_pipeline
from src.pipelines.load_merge import load_and_merge
import joblib
    
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
    print('Train dataframe shape:', train_df.shape)

if not os.path.exists(test_parquet_path):
    print('Test parquet file not found. Merging and creating parquet file...')
    test_df = load_and_merge(test_sources, join_key="TransactionID")
    if test_df is None:
        raise ValueError("Merged test DataFrame is None.")
    print(test_df.shape)
    test_df.columns = test_df.columns.str.replace('-', '_')

    test_df.to_parquet(r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\data\parquet\test_merged.parquet")
                        
else:
    print('Test parquet file found. Skipping merge.')
    test_df = pd.read_parquet(test_parquet_path)
    print('Test dataframe shape:', test_df.shape)

X = train_df.drop(columns=["isFraud"], errors="ignore")
y = train_df["isFraud"] if "isFraud" in train_df.columns else None

if not os.path.exists(r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\artifacts\tree_pipeline.joblib"):
    print('Calling main_pipeline.fit_transform()...')
   
    print('Fitting tree_pipeline..., data X and Y shapes:', X.shape, y.shape if y is not None else 'N/A')
    train_ready = tree_pipeline.fit_transform(X,y)

    joblib.dump(tree_pipeline, r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\artifacts\tree_pipeline.joblib")

else:
    print('Tree Pipeline already exists. Skipping processing.')


if not os.path.exists(r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\artifacts\normal_pipeline.joblib"):

    print('Pipeline processing complete.')

    print('Calling normal_pipeline.fit_transform()...X and y shapes:', X.shape, y.shape if y is not None else 'N/A')
    train_ready_normal = normal_pipeline.fit_transform(X,y)

    joblib.dump(normal_pipeline, r"E:\E2E Project 1\E2E-Project-1\fraud-detection-ml\artifacts\normal_pipeline.joblib")
    print('Normal Pipeline processing complete.')

else:
    print('Normal Pipeline already exists. Skipping processing.')